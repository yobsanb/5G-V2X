"""SegFormer semantic segmentation wrapped to this repo's segmenter contract.

Normalizes inside forward() and upsamples logits to the input resolution.
The nvidia/* weights are non-commercial; see v2x_edge/registry.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from v2x_edge.registry import warn_if_restricted_weights
from v2x_edge.utils import resolve_device

DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainableSegFormer(nn.Module):
    """Takes RGB in [0, 1] as NxCxHxW; returns logits at the same HxW."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        pretrained_model: str = DEFAULT_SEGFORMER_MODEL,
        freeze_encoder: bool = False,
        config_overrides: dict[str, Any] | None = None,
        hf_model: SegformerForSemanticSegmentation | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)

        if hf_model is not None:
            self.model = hf_model
        elif pretrained:
            warn_if_restricted_weights(pretrained_model)
            # Cityscapes heads have 19 outputs; ignore_mismatched_sizes re-initialises
            # the classifier for our class count and keeps the encoder weights.
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                **(config_overrides or {}),
            )
        else:
            self.model = SegformerForSemanticSegmentation(
                SegformerConfig(num_labels=self.num_classes, **(config_overrides or {}))
            )
        if self.model.config.num_labels != self.num_classes:
            raise ValueError(
                f"Model predicts {self.model.config.num_labels} classes but num_classes="
                f"{self.num_classes}"
            )
        if freeze_encoder:
            for parameter in self.model.segformer.parameters():
                parameter.requires_grad_(False)

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

    def checkpoint_meta(self) -> dict[str, Any]:
        return {
            "model_type": "segformer",
            "model_config_json": self.model.config.to_json_string(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected segmentation input with shape Nx3xHxW")
        height, width = x.shape[-2:]
        logits = self.model(pixel_values=(x - self._mean) / self._std).logits
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


def build_segformer_from_checkpoint(checkpoint: dict[str, Any]) -> TrainableSegFormer:
    required = {"model_state", "num_classes", "model_config_json"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Invalid SegFormer checkpoint; expected keys {sorted(required)}")
    config = SegformerConfig.from_dict(json.loads(checkpoint["model_config_json"]))
    model = TrainableSegFormer(
        num_classes=int(checkpoint["num_classes"]),
        hf_model=SegformerForSemanticSegmentation(config),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


class SegFormerSegmenter:
    """Runtime segmenter: BGR frame in, HxW int32 class-ID map out."""

    def __init__(
        self,
        pretrained_model: str = DEFAULT_SEGFORMER_MODEL,
        device: str = "auto",
        inference_size: tuple[int, int] | None = (1024, 512),
        model: TrainableSegFormer | None = None,
    ) -> None:
        self.device = resolve_device(device)
        if model is None:
            warn_if_restricted_weights(pretrained_model)
            loaded = SegformerForSemanticSegmentation.from_pretrained(pretrained_model)
            model = TrainableSegFormer(num_classes=int(loaded.config.num_labels), hf_model=loaded)
        self.model = model.to(self.device).eval()
        self.num_classes = model.num_classes
        if inference_size is not None:
            width, height = inference_size
            if width < 1 or height < 1:
                raise ValueError("inference_size must be positive")
        self.inference_size = inference_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        inference_size: tuple[int, int] | None = None,
    ) -> "SegFormerSegmenter":
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        return cls(
            device=device,
            inference_size=inference_size,
            model=build_segformer_from_checkpoint(checkpoint),
        )

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape HxWx3")
        if image_bgr.dtype != np.uint8:
            raise ValueError("Segmenter input must use uint8 BGR pixels")
        if image_bgr.size == 0:
            raise ValueError("Image cannot be empty")

        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device, torch.float32) / 255.0
        tensor = tensor.unsqueeze(0)
        if self.inference_size is not None:
            target_w, target_h = self.inference_size
            tensor = F.interpolate(
                tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        logits = self.model(tensor)
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits[0].argmax(dim=0).detach().cpu().numpy().astype(np.int32)
