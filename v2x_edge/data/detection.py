from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import to_tensor

from .transforms import DetectionTransform


class DetectionManifestDataset(Dataset):
    """JSONL manifest of {"image", "boxes", "labels"}; label 0 is background."""

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path | None = None,
        transform: DetectionTransform | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        self.transform = transform
        self.records: list[dict[str, Any]] = []

        with self.manifest.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at manifest line {line_number}") from exc
                self._validate_record(record, line_number)
                self.records.append(record)

        if not self.records:
            raise ValueError("Detection manifest contains no records")

    @staticmethod
    def _validate_record(record: Any, line_number: int) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"Manifest line {line_number} must be a JSON object")
        required = {"image", "boxes", "labels"}
        if not required.issubset(record):
            raise ValueError(f"Manifest line {line_number} is missing {sorted(required - record.keys())}")
        if not isinstance(record["image"], str) or not record["image"]:
            raise ValueError(f"Invalid image path at manifest line {line_number}")
        if not isinstance(record["boxes"], list) or not isinstance(record["labels"], list):
            raise ValueError(f"boxes and labels must be lists at manifest line {line_number}")
        if len(record["boxes"]) != len(record["labels"]):
            raise ValueError(f"boxes/labels length mismatch at manifest line {line_number}")
        for box in record["boxes"]:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"Each box must contain four coordinates at line {line_number}")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
                raise ValueError(f"Box coordinates must be numeric at line {line_number}")
        for label in record["labels"]:
            if isinstance(label, bool) or not isinstance(label, int) or label < 1:
                raise ValueError(f"Detection labels must be positive integers at line {line_number}")

    def validate_label_range(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        for index, record in enumerate(self.records, start=1):
            if record["labels"] and max(record["labels"]) >= num_classes:
                raise ValueError(
                    f"Manifest record {index} contains label {max(record['labels'])}, "
                    f"but num_classes={num_classes}"
                )

    def label_histogram(self, num_classes: int) -> list[int]:
        counts = [0] * num_classes
        for record in self.records:
            for label in record["labels"]:
                counts[label] += 1
        return counts

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.root / record["image"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        height, width = bgr.shape[:2]
        boxes = torch.as_tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(record["labels"], dtype=torch.int64)

        if len(boxes):
            if not torch.isfinite(boxes).all():
                raise ValueError(f"Non-finite box coordinates in {image_path}")
            if torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 1] < 0):
                raise ValueError(f"Negative box coordinates in {image_path}")
            if torch.any(boxes[:, 2] > width) or torch.any(boxes[:, 3] > height):
                raise ValueError(f"Box coordinates exceed image bounds in {image_path}")
            if torch.any(boxes[:, 2] <= boxes[:, 0]) or torch.any(boxes[:, 3] <= boxes[:, 1]):
                raise ValueError(f"Invalid box dimensions in {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "area": area,
        }
        image = to_tensor(rgb)
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


def detection_collate_fn(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int = 42) -> tuple[Subset, Subset]:
    """Deterministic random split; prefer separate manifests per recording."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    total = len(dataset)
    val_size = max(1, int(round(total * val_fraction)))
    if val_size >= total:
        raise ValueError(f"val_fraction={val_fraction} leaves no training samples")
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    return Subset(dataset, permutation[val_size:]), Subset(dataset, permutation[:val_size])
