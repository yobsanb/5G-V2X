from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
from torchvision.transforms.functional import normalize, to_tensor

from v2x_edge.utils import resolve_device

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _image_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Segmenter input must use uint8 BGR pixels")
    if image_bgr.size == 0:
        raise ValueError("Image cannot be empty")
    return to_tensor(image_bgr[..., ::-1].copy())


class LRASPPSegmenter:
    def __init__(self, pretrained: bool = True, device: str = "auto") -> None:
        self.device = resolve_device(device)
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.model = lraspp_mobilenet_v3_large(
            weights=weights,
            weights_backbone=None if not pretrained else MobileNet_V3_Large_Weights.DEFAULT,
        )
        self.model.to(self.device).eval()
        self.categories = list(weights.meta["categories"]) if weights is not None else None
        self.preprocess = weights.transforms() if weights is not None else None

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        x = _image_tensor(image_bgr)
        height, width = image_bgr.shape[:2]
        if self.preprocess is not None:
            x = self.preprocess(x)
        else:
            x = normalize(x, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        logits = self.model(x.unsqueeze(0).to(self.device))["out"]
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return logits[0].argmax(dim=0).detach().cpu().numpy().astype(np.int32)


class TrainableLRASPP(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = False) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected segmentation input with shape Nx3xHxW")
        x = (x - self._mean) / self._std
        return self.model(x)["out"]


class CheckpointSegmenter:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        checkpoint: dict | None = None,
    ) -> None:
        if (checkpoint_path is None) == (checkpoint is None):
            raise ValueError("Provide exactly one of checkpoint_path or checkpoint")
        self.device = resolve_device(device)
        if checkpoint is None:
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "num_classes" not in checkpoint:
            raise ValueError("Invalid segmenter checkpoint format")
        self.num_classes = int(checkpoint["num_classes"])
        self.model = TrainableLRASPP(self.num_classes, pretrained_backbone=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        x = _image_tensor(image_bgr).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]
        return logits.argmax(dim=0).detach().cpu().numpy().astype(np.int32)
