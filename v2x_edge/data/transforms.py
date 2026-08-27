"""Augmentation for detection (image + boxes) and segmentation (image + mask)."""

from __future__ import annotations

import random
from typing import Any, Protocol

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class DetectionTransform(Protocol):
    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


class SegmentationTransform(Protocol):
    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _photometric(image: torch.Tensor, brightness: float, contrast: float, saturation: float, hue: float) -> torch.Tensor:
    for name in random.sample(["brightness", "contrast", "saturation", "hue"], 4):
        if name == "brightness" and brightness > 0.0:
            image = F.adjust_brightness(image, 1.0 + random.uniform(-brightness, brightness))
        elif name == "contrast" and contrast > 0.0:
            image = F.adjust_contrast(image, 1.0 + random.uniform(-contrast, contrast))
        elif name == "saturation" and saturation > 0.0:
            image = F.adjust_saturation(image, 1.0 + random.uniform(-saturation, saturation))
        elif name == "hue" and hue > 0.0:
            image = F.adjust_hue(image, random.uniform(-hue, hue))
    return image.clamp(0.0, 1.0)


def _scaled_size(height: int, width: int, scale: float) -> list[int]:
    return [max(1, int(round(height * scale))), max(1, int(round(width * scale)))]


class ComposeDetection:
    def __init__(self, transforms: list[DetectionTransform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class DetectionHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = float(probability)

    def __call__(self, image, target):
        if random.random() >= self.probability:
            return image, target
        width = image.shape[-1]
        image = F.hflip(image)
        boxes = target["boxes"]
        if len(boxes):
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            target = {**target, "boxes": flipped}
        return image, target


class DetectionPhotometric:
    def __init__(
        self,
        probability: float = 0.5,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.25,
        hue: float = 0.03,
    ) -> None:
        if not 0.0 <= hue <= 0.5:
            raise ValueError("hue must be in [0, 0.5]")
        self.probability = float(probability)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = _photometric(image, self.brightness, self.contrast, self.saturation, self.hue)
        return image, target


class DetectionScaleJitter:
    """Uniformly rescales the image and its boxes, dropping boxes that collapse."""

    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.25, min_box_size: float = 2.0) -> None:
        if not 0.0 < min_scale <= max_scale:
            raise ValueError("scale range must satisfy 0 < min_scale <= max_scale")
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.min_box_size = float(min_box_size)

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        height, width = image.shape[-2:]
        image = F.resize(image, _scaled_size(height, width, scale), antialias=True)
        boxes = target["boxes"] * scale
        if len(boxes):
            keep = ((boxes[:, 2] - boxes[:, 0]) >= self.min_box_size) & (
                (boxes[:, 3] - boxes[:, 1]) >= self.min_box_size
            )
            target = {
                **target,
                "boxes": boxes[keep],
                "labels": target["labels"][keep],
                "iscrowd": target["iscrowd"][keep],
                "area": (boxes[:, 2] - boxes[:, 0])[keep] * (boxes[:, 3] - boxes[:, 1])[keep],
            }
        else:
            target = {**target, "boxes": boxes}
        return image, target


class ComposeSegmentation:
    def __init__(self, transforms: list[SegmentationTransform]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class SegmentationHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(probability)

    def __call__(self, image, mask):
        if random.random() < self.probability:
            image, mask = F.hflip(image), F.hflip(mask.unsqueeze(0)).squeeze(0)
        return image, mask


class SegmentationPhotometric:
    def __init__(
        self,
        probability: float = 0.5,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.25,
        hue: float = 0.03,
    ) -> None:
        self.probability = float(probability)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    def __call__(self, image, mask):
        if random.random() < self.probability:
            image = _photometric(image, self.brightness, self.contrast, self.saturation, self.hue)
        return image, mask


class SegmentationScaleCrop:
    """Random rescale then crop or pad back to a fixed size, padded with ignore_index."""

    def __init__(
        self,
        size: tuple[int, int],
        min_scale: float = 0.75,
        max_scale: float = 1.5,
        ignore_index: int = 255,
    ) -> None:
        width, height = size
        if width < 1 or height < 1:
            raise ValueError("size must be positive")
        if not 0.0 < min_scale <= max_scale:
            raise ValueError("scale range must satisfy 0 < min_scale <= max_scale")
        self.width = int(width)
        self.height = int(height)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.ignore_index = int(ignore_index)

    def __call__(self, image, mask):
        scale = random.uniform(self.min_scale, self.max_scale)
        source_h, source_w = image.shape[-2:]
        size = _scaled_size(source_h, source_w, scale)
        image = F.resize(image, size, antialias=True)
        mask = F.resize(mask.unsqueeze(0), size, interpolation=InterpolationMode.NEAREST).squeeze(0)

        pad_h = max(0, self.height - image.shape[-2])
        pad_w = max(0, self.width - image.shape[-1])
        if pad_h or pad_w:
            image = F.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = F.pad(mask.unsqueeze(0), [0, 0, pad_w, pad_h], fill=self.ignore_index).squeeze(0)

        top = random.randint(0, image.shape[-2] - self.height)
        left = random.randint(0, image.shape[-1] - self.width)
        image = F.crop(image, top, left, self.height, self.width)
        mask = F.crop(mask.unsqueeze(0), top, left, self.height, self.width).squeeze(0)
        return image, mask


class SegmentationResize:
    def __init__(self, size: tuple[int, int]) -> None:
        width, height = size
        if width < 1 or height < 1:
            raise ValueError("size must be positive")
        self.size = [int(height), int(width)]

    def __call__(self, image, mask):
        image = F.resize(image, self.size, antialias=True)
        mask = F.resize(mask.unsqueeze(0), self.size, interpolation=InterpolationMode.NEAREST).squeeze(0)
        return image, mask


def build_detection_transform(cfg: dict[str, Any] | None) -> ComposeDetection | None:
    """Builds the training-time detection augmentation; returns None when disabled."""
    if not cfg:
        return None
    transforms: list[DetectionTransform] = []
    if cfg.get("horizontal_flip"):
        transforms.append(DetectionHorizontalFlip(float(cfg["horizontal_flip"])))
    if cfg.get("photometric"):
        transforms.append(DetectionPhotometric(float(cfg["photometric"])))
    scale = cfg.get("scale_jitter")
    if scale:
        if not isinstance(scale, (list, tuple)) or len(scale) != 2:
            raise ValueError("'scale_jitter' must be [min_scale, max_scale]")
        transforms.append(DetectionScaleJitter(float(scale[0]), float(scale[1])))
    return ComposeDetection(transforms) if transforms else None


def build_segmentation_transform(
    cfg: dict[str, Any] | None,
    size: tuple[int, int],
    ignore_index: int,
    training: bool,
) -> ComposeSegmentation:
    """Training transforms scale, crop, and jitter; evaluation only resizes."""
    if not training or not cfg:
        return ComposeSegmentation([SegmentationResize(size)])
    transforms: list[SegmentationTransform] = []
    scale = cfg.get("scale_crop")
    if scale:
        if not isinstance(scale, (list, tuple)) or len(scale) != 2:
            raise ValueError("'scale_crop' must be [min_scale, max_scale]")
        transforms.append(
            SegmentationScaleCrop(size, float(scale[0]), float(scale[1]), ignore_index)
        )
    else:
        transforms.append(SegmentationResize(size))
    if cfg.get("horizontal_flip"):
        transforms.append(SegmentationHorizontalFlip(float(cfg["horizontal_flip"])))
    if cfg.get("photometric"):
        transforms.append(SegmentationPhotometric(float(cfg["photometric"])))
    return ComposeSegmentation(transforms)
