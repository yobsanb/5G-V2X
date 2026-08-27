from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import to_tensor

from v2x_edge.types import Detection
from v2x_edge.utils import resolve_device

COCO_FALLBACK = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


def _validate_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Detector input must use uint8 BGR pixels")
    if image_bgr.size == 0:
        raise ValueError("Image cannot be empty")


class TorchvisionCocoDetector:
    def __init__(
        self,
        backend: str = "ssdlite",
        pretrained: bool = True,
        confidence_threshold: float = 0.45,
        allowed_labels: Iterable[str] | None = None,
        device: str = "auto",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.device = resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.allowed_labels = set(allowed_labels) if allowed_labels else None

        backend = backend.lower()
        if backend in {"ssdlite", "torchvision_ssdlite", "ssdlite320_mobilenet_v3_large"}:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = ssdlite320_mobilenet_v3_large(
                weights=weights,
                weights_backbone=None if not pretrained else MobileNet_V3_Large_Weights.DEFAULT,
            )
        elif backend in {"fasterrcnn", "fasterrcnn_mobilenet_v3_large_320_fpn"}:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=weights,
                weights_backbone=None if not pretrained else MobileNet_V3_Large_Weights.DEFAULT,
            )
        else:
            raise ValueError(f"Unsupported detector backend: {backend}")

        self.categories = list(weights.meta["categories"]) if weights is not None else None
        self.model.to(self.device).eval()

    def _label_name(self, class_id: int) -> str:
        if self.categories is not None and 0 <= class_id < len(self.categories):
            return str(self.categories[class_id])
        return COCO_FALLBACK.get(class_id, f"class_{class_id}")

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        _validate_image(image_bgr)
        tensor = to_tensor(image_bgr[..., ::-1].copy()).to(self.device)
        output = self.model([tensor])[0]
        detections: list[Detection] = []

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        for box, score, class_id in zip(boxes, scores, labels, strict=True):
            score = float(score)
            if score < self.confidence_threshold:
                continue
            class_id = int(class_id)
            name = self._label_name(class_id)
            if self.allowed_labels is not None and name not in self.allowed_labels:
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(v) for v in box),
                    label=name,
                    score=score,
                    class_id=class_id,
                )
            )
        return detections


class TrainableFasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 6,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        if not 0 <= trainable_backbone_layers <= 6:
            raise ValueError("trainable_backbone_layers must be in [0, 6]")
        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers if pretrained_backbone else None,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)


class CheckpointDetector:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.45,
        device: str = "auto",
        checkpoint: dict | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if (checkpoint_path is None) == (checkpoint is None):
            raise ValueError("Provide exactly one of checkpoint_path or checkpoint")
        self.device = resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)
        if checkpoint is None:
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "num_classes" not in checkpoint:
            raise ValueError("Invalid detector checkpoint format")

        saved_names = checkpoint.get("class_names")
        if class_names is None:
            if not isinstance(saved_names, list) or not all(isinstance(name, str) and name for name in saved_names):
                raise ValueError("class_names must be provided when the checkpoint does not contain them")
            class_names = saved_names
        self.class_names = list(class_names)
        num_classes = int(checkpoint["num_classes"])
        if num_classes != len(self.class_names) + 1:
            raise ValueError(
                f"Checkpoint num_classes={num_classes}, but {len(self.class_names)} "
                "foreground class names were provided"
            )

        self.model = TrainableFasterRCNN(num_classes=num_classes, pretrained_backbone=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        _validate_image(image_bgr)
        tensor = to_tensor(image_bgr[..., ::-1].copy()).to(self.device)
        output = self.model([tensor])[0]
        detections: list[Detection] = []

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        for box, score, class_id in zip(boxes, scores, labels, strict=True):
            score = float(score)
            class_id = int(class_id)
            if score < self.confidence_threshold or not 1 <= class_id <= len(self.class_names):
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(v) for v in box),
                    label=self.class_names[class_id - 1],
                    score=score,
                    class_id=class_id,
                )
            )
        return detections
