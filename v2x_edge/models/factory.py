from __future__ import annotations

from typing import Any

import torch
from torch import nn

from v2x_edge.registry import (
    DETECTION_BACKENDS,
    DETECTION_MODELS,
    SEGMENTATION_BACKENDS,
    SEGMENTATION_MODELS,
)

from .detection import CheckpointDetector, TorchvisionCocoDetector, TrainableFasterRCNN
from .dfine import (
    DEFAULT_DFINE_MODEL,
    DFineDetector,
    TrainableDFine,
    build_dfine_from_checkpoint,
)
from .segformer import (
    DEFAULT_SEGFORMER_MODEL,
    SegFormerSegmenter,
    TrainableSegFormer,
    build_segformer_from_checkpoint,
)
from .segmentation import CheckpointSegmenter, LRASPPSegmenter, TrainableLRASPP

# --------------------------------------------------------------------------- #
# Training architectures
# --------------------------------------------------------------------------- #


def build_detection_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    name = str(cfg.get("name", "dfine")).lower()
    if name not in DETECTION_MODELS:
        raise ValueError(f"Unsupported detection model: {name}")
    if name == "dfine":
        return TrainableDFine(
            num_classes=num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            pretrained_model=str(cfg.get("pretrained_model", DEFAULT_DFINE_MODEL)),
            image_size=int(cfg.get("image_size", 640)),
            max_detections=int(cfg.get("max_detections", 300)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            config_overrides=cfg.get("config_overrides"),
        )
    return TrainableFasterRCNN(
        num_classes=num_classes,
        pretrained_backbone=bool(cfg.get("pretrained", cfg.get("pretrained_backbone", True))),
        trainable_backbone_layers=int(cfg.get("trainable_backbone_layers", 6)),
    )


def build_segmentation_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    name = str(cfg.get("name", "segformer")).lower()
    if name not in SEGMENTATION_MODELS:
        raise ValueError(f"Unsupported segmentation model: {name}")
    if name == "segformer":
        return TrainableSegFormer(
            num_classes=num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            pretrained_model=str(cfg.get("pretrained_model", DEFAULT_SEGFORMER_MODEL)),
            freeze_encoder=bool(cfg.get("freeze_encoder", False)),
            config_overrides=cfg.get("config_overrides"),
        )
    return TrainableLRASPP(
        num_classes,
        pretrained_backbone=bool(cfg.get("pretrained", cfg.get("pretrained_backbone", True))),
    )


def checkpoint_meta(model: nn.Module) -> dict[str, Any]:
    """Architecture description to store alongside the weights."""
    if hasattr(model, "checkpoint_meta"):
        return model.checkpoint_meta()
    if isinstance(model, TrainableFasterRCNN):
        return {"model_type": "fasterrcnn_mobilenet_v3_large_320_fpn"}
    if isinstance(model, TrainableLRASPP):
        return {"model_type": "lraspp_mobilenet_v3_large"}
    raise ValueError(f"Unknown model type: {type(model).__name__}")


# --------------------------------------------------------------------------- #
# Runtime perception backends
# --------------------------------------------------------------------------- #


def build_detector_from_config(cfg: dict[str, Any], device: str = "auto"):
    backend = str(cfg.get("backend", "dfine"))
    if backend not in DETECTION_BACKENDS:
        raise ValueError(f"Unsupported detector backend: {backend}")
    confidence = float(cfg.get("confidence_threshold", 0.45))
    allowed = cfg.get("allowed_labels")

    if backend == "checkpoint":
        # Dispatch on what the checkpoint says it is, so one backend name covers every
        # trained architecture. Loaded once and handed on, not re-read per branch.
        checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid detector checkpoint format")
        if str(checkpoint.get("model_type", "")) == "dfine":
            model, class_names = build_dfine_from_checkpoint(checkpoint)
            # image_size is an inference-time choice, not architecture, so a config
            # override must take effect rather than be silently dropped.
            if cfg.get("image_size"):
                model.image_size = int(cfg["image_size"])
            return DFineDetector(
                confidence_threshold=confidence,
                allowed_labels=allowed,
                device=device,
                class_names=class_names,
                model=model,
            )
        class_names = cfg.get("class_names")
        return CheckpointDetector(
            checkpoint=checkpoint,
            class_names=list(class_names) if class_names is not None else None,
            confidence_threshold=confidence,
            device=device,
        )
    if backend == "dfine":
        return DFineDetector(
            pretrained_model=str(cfg.get("pretrained_model", DEFAULT_DFINE_MODEL)),
            confidence_threshold=confidence,
            allowed_labels=allowed,
            device=device,
            image_size=int(cfg.get("image_size", 640)),
        )
    return TorchvisionCocoDetector(
        backend=backend,
        pretrained=bool(cfg.get("pretrained", True)),
        confidence_threshold=confidence,
        allowed_labels=allowed,
        device=device,
    )


def build_segmenter_from_config(cfg: dict[str, Any], device: str = "auto"):
    if not bool(cfg.get("enabled", False)):
        return None
    backend = str(cfg.get("backend", "segformer"))
    if backend not in SEGMENTATION_BACKENDS:
        raise ValueError(f"Unsupported segmentation backend: {backend}")
    size = cfg.get("inference_size")
    # Same default on both branches: a trained checkpoint run at a 1080p frame's native
    # resolution would otherwise be far slower, and far from its training resolution.
    inference_size = (int(size[0]), int(size[1])) if size else (1024, 512)

    if backend == "checkpoint":
        checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid segmenter checkpoint format")
        if str(checkpoint.get("model_type", "")) == "segformer":
            return SegFormerSegmenter(
                device=device,
                inference_size=inference_size,
                model=build_segformer_from_checkpoint(checkpoint),
            )
        return CheckpointSegmenter(checkpoint=checkpoint, device=device)
    if backend == "segformer":
        return SegFormerSegmenter(
            pretrained_model=str(cfg.get("pretrained_model", DEFAULT_SEGFORMER_MODEL)),
            device=device,
            inference_size=inference_size,
        )
    return LRASPPSegmenter(pretrained=bool(cfg.get("pretrained", True)), device=device)
