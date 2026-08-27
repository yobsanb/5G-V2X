"""D-FINE real-time detection transformer wrapped to this repo's detector contract.

NMS-free. Converts D-FINE's 0-indexed labels, normalized cxcywh boxes and
un-normalized [0, 1] input to the repo's conventions. See README, Models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DFineConfig, DFineForObjectDetection

from v2x_edge.registry import warn_if_restricted_weights
from v2x_edge.types import Detection
from v2x_edge.utils import resolve_device

DEFAULT_DFINE_MODEL = "ustc-community/dfine-small-coco"

# COCO-80 names used by D-FINE, mapped onto the COCO-91 names torchvision reports,
# so one `allowed_labels` list in the config works against either backend.
_COCO_ALIASES = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}


def _validate_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Detector input must use uint8 BGR pixels")
    if image_bgr.size == 0:
        raise ValueError("Image cannot be empty")


def xyxy_to_normalized_cxcywh(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if not len(boxes):
        return boxes.new_zeros((0, 4))
    relative = boxes / boxes.new_tensor([width, height, width, height])
    centre_x = (relative[:, 0] + relative[:, 2]) * 0.5
    centre_y = (relative[:, 1] + relative[:, 3]) * 0.5
    return torch.stack(
        [centre_x, centre_y, relative[:, 2] - relative[:, 0], relative[:, 3] - relative[:, 1]], dim=1
    )


def normalized_cxcywh_to_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if not len(boxes):
        return boxes.new_zeros((0, 4))
    centre_x, centre_y, box_w, box_h = boxes.unbind(-1)
    corners = torch.stack(
        [
            centre_x - box_w * 0.5,
            centre_y - box_h * 0.5,
            centre_x + box_w * 0.5,
            centre_y + box_h * 0.5,
        ],
        dim=-1,
    )
    return corners * corners.new_tensor([width, height, width, height])


def coco_label_names(config) -> list[str]:
    id2label = config.id2label or {}
    names = []
    for index in range(int(config.num_labels)):
        raw = str(id2label.get(index, f"class_{index}")).lower()
        names.append(_COCO_ALIASES.get(raw, raw))
    return names


class TrainableDFine(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        pretrained_model: str = DEFAULT_DFINE_MODEL,
        image_size: int = 640,
        max_detections: int = 300,
        freeze_backbone: bool = False,
        config_overrides: dict[str, Any] | None = None,
        hf_model: DFineForObjectDetection | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        if max_detections < 1:
            raise ValueError("max_detections must be >= 1")
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)

        if hf_model is not None:
            self.model = hf_model
        elif pretrained:
            warn_if_restricted_weights(pretrained_model)
            # The COCO head has 80 outputs; ignore_mismatched_sizes re-initialises it for
            # our class count while keeping backbone, encoder and decoder weights.
            self.model = DFineForObjectDetection.from_pretrained(
                pretrained_model,
                num_labels=self.num_classes - 1,
                ignore_mismatched_sizes=True,
                **(config_overrides or {}),
            )
        else:
            self.model = DFineForObjectDetection(
                DFineConfig(num_labels=self.num_classes - 1, **(config_overrides or {}))
            )
        if self.model.config.num_labels != self.num_classes - 1:
            raise ValueError(
                f"Model predicts {self.model.config.num_labels} classes but num_classes="
                f"{self.num_classes} implies {self.num_classes - 1}"
            )
        if freeze_backbone:
            for parameter in self.model.model.backbone.parameters():
                parameter.requires_grad_(False)

    @property
    def num_labels(self) -> int:
        return int(self.model.config.num_labels)

    def checkpoint_meta(self) -> dict[str, Any]:
        """Architecture saved with the weights so a checkpoint rebuilds itself."""
        return {
            "model_type": "dfine",
            "model_config_json": self.model.config.to_json_string(),
            "image_size": self.image_size,
            "max_detections": self.max_detections,
        }

    def _batch(self, images) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        if torch.is_tensor(images):
            images = list(images)
        if not images:
            raise ValueError("At least one image is required")
        sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
        resized = [
            F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            for image in images
        ]
        return torch.cat(resized, dim=0), sizes

    def _hf_labels(self, targets, sizes, device) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "class_labels": target["labels"].to(device=device, dtype=torch.int64) - 1,
                "boxes": xyxy_to_normalized_cxcywh(
                    target["boxes"].to(device=device, dtype=torch.float32), height, width
                ),
            }
            for target, (height, width) in zip(targets, sizes, strict=True)
        ]

    def decode(self, logits: torch.Tensor, boxes: torch.Tensor, sizes) -> list[dict[str, torch.Tensor]]:
        """Focal-loss decoding: per-class sigmoid, then top-k over (query x class)."""
        scores = logits.sigmoid()
        num_labels = scores.shape[-1]
        results = []
        for index, (height, width) in enumerate(sizes):
            flat = scores[index].flatten()
            top_scores, top_index = flat.topk(min(self.max_detections, flat.numel()))
            query = torch.div(top_index, num_labels, rounding_mode="floor")
            results.append(
                {
                    "boxes": normalized_cxcywh_to_xyxy(boxes[index][query], height, width),
                    "scores": top_scores,
                    "labels": top_index % num_labels + 1,
                }
            )
        return results

    def forward(self, images, targets=None):
        pixel_values, sizes = self._batch(images)
        if self.training:
            if targets is None:
                raise ValueError("Targets are required in training mode")
            outputs = self.model(
                pixel_values=pixel_values,
                labels=self._hf_labels(targets, sizes, pixel_values.device),
            )
            # outputs.loss is already the weighted sum over the matcher, auxiliary and
            # denoising terms in outputs.loss_dict; summing that dict would double count.
            return {"loss": outputs.loss}
        outputs = self.model(pixel_values=pixel_values)
        return self.decode(outputs.logits, outputs.pred_boxes, sizes)


def build_dfine_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[TrainableDFine, list[str]]:
    required = {"model_state", "num_classes", "model_config_json"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Invalid D-FINE checkpoint; expected keys {sorted(required)}")
    num_classes = int(checkpoint["num_classes"])
    class_names = checkpoint.get("class_names") or [f"class_{index}" for index in range(1, num_classes)]
    if len(class_names) != num_classes - 1:
        raise ValueError(
            f"Checkpoint num_classes={num_classes} implies {num_classes - 1} foreground "
            f"names, but {len(class_names)} were stored"
        )
    config = DFineConfig.from_dict(json.loads(checkpoint["model_config_json"]))
    model = TrainableDFine(
        num_classes=num_classes,
        image_size=int(checkpoint.get("image_size", 640)),
        max_detections=int(checkpoint.get("max_detections", 300)),
        hf_model=DFineForObjectDetection(config),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), list(class_names)


class DFineDetector:
    """Runtime detector: BGR frame in, `list[Detection]` out."""

    def __init__(
        self,
        pretrained_model: str = DEFAULT_DFINE_MODEL,
        confidence_threshold: float = 0.45,
        allowed_labels=None,
        device: str = "auto",
        image_size: int = 640,
        max_detections: int = 300,
        class_names: list[str] | None = None,
        model: TrainableDFine | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.device = resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.allowed_labels = set(allowed_labels) if allowed_labels else None

        if model is None:
            warn_if_restricted_weights(pretrained_model)
            loaded = DFineForObjectDetection.from_pretrained(pretrained_model)
            class_names = coco_label_names(loaded.config)
            model = TrainableDFine(
                num_classes=len(class_names) + 1,
                image_size=image_size,
                max_detections=max_detections,
                hf_model=loaded,
            )
        elif class_names is None:
            raise ValueError("class_names is required when supplying a model")

        self.class_names = list(class_names)
        self.model = model.to(self.device).eval()
        if len(self.class_names) != self.model.num_labels:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but the model predicts "
                f"{self.model.num_labels} classes"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        confidence_threshold: float = 0.45,
        allowed_labels=None,
        device: str = "auto",
    ) -> "DFineDetector":
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        model, class_names = build_dfine_from_checkpoint(checkpoint)
        return cls(
            confidence_threshold=confidence_threshold,
            allowed_labels=allowed_labels,
            device=device,
            class_names=class_names,
            model=model,
        )

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        _validate_image(image_bgr)
        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, torch.float32) / 255.0
        prediction = self.model([tensor])[0]

        boxes = prediction["boxes"].clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, float(width))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, float(height))
        keep = (
            (prediction["scores"] >= self.confidence_threshold)
            & ((boxes[:, 2] - boxes[:, 0]) > 1e-3)
            & ((boxes[:, 3] - boxes[:, 1]) > 1e-3)
        )

        detections: list[Detection] = []
        for box, score, class_id in zip(
            boxes[keep].cpu().numpy(),
            prediction["scores"][keep].cpu().numpy(),
            prediction["labels"][keep].cpu().numpy(),
            strict=True,
        ):
            name = self.class_names[int(class_id) - 1]
            if self.allowed_labels is not None and name not in self.allowed_labels:
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in box),
                    label=name,
                    score=float(score),
                    class_id=int(class_id),
                )
            )
        return detections
