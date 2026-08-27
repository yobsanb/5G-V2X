from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .transforms import SegmentationTransform


class SegmentationFolderDataset(Dataset):
    """Image/mask folders paired by filename stem; each mask pixel is a class ID."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: SegmentationTransform | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(self.masks_dir)
        self.transform = transform

        images = [p for p in sorted(self.images_dir.iterdir()) if p.suffix.lower() in self.IMAGE_EXTS]
        masks = [p for p in sorted(self.masks_dir.iterdir()) if p.suffix.lower() in self.IMAGE_EXTS]
        mask_by_stem: dict[str, Path] = {}
        for mask in masks:
            if mask.stem in mask_by_stem:
                raise ValueError(f"Duplicate segmentation mask stem: {mask.stem}")
            mask_by_stem[mask.stem] = mask

        missing = [image.name for image in images if image.stem not in mask_by_stem]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing masks for {len(missing)} image(s): {preview}")
        self.pairs = [(image, mask_by_stem[image.stem]) for image in images]
        if not self.pairs:
            raise ValueError("No image/mask pairs found")

    def class_histogram(self, num_classes: int, ignore_index: int = 255) -> list[int]:
        """Pixel count per class over the whole set; useful for spotting class imbalance."""
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, mask_path in self.pairs:
            mask = self._read_mask(mask_path)
            valid = mask[mask != ignore_index]
            counts += np.bincount(valid.reshape(-1), minlength=num_classes)[:num_classes]
        return counts.tolist()

    @staticmethod
    def _read_mask(mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim == 3:
            if not (mask[..., 0] == mask[..., 1]).all() or not (mask[..., 0] == mask[..., 2]).all():
                raise ValueError(f"Mask must contain class IDs in one channel: {mask_path}")
            mask = mask[..., 0]
        return mask

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        mask = self._read_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {image_path} vs {mask_path}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor(rgb)
        mask_tensor = torch.from_numpy(mask.astype("int64", copy=False))
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return image_tensor, mask_tensor
