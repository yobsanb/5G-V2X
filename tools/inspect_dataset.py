#!/usr/bin/env python3
"""Report dataset statistics before training: class balance and box geometry."""

from __future__ import annotations

import argparse
import json

import numpy as np

from v2x_edge.config import load_config
from v2x_edge.data import DetectionManifestDataset, SegmentationFolderDataset


def _detector_report(cfg: dict, split: str) -> dict:
    manifest = cfg.get(f"{split}_manifest") or cfg["train_manifest"]
    dataset = DetectionManifestDataset(manifest, cfg.get("root"))
    num_classes = int(cfg["num_classes"])
    dataset.validate_label_range(num_classes)
    names = cfg.get("class_names") or [f"class_{index}" for index in range(1, num_classes)]

    widths, heights = [], []
    empty_images = 0
    for record in dataset.records:
        if not record["boxes"]:
            empty_images += 1
        for x1, y1, x2, y2 in record["boxes"]:
            widths.append(x2 - x1)
            heights.append(y2 - y1)

    counts = dataset.label_histogram(num_classes)[1:]
    total = sum(counts)
    report = {
        "manifest": str(manifest),
        "images": len(dataset),
        "images_without_boxes": empty_images,
        "boxes": total,
        "per_class": {
            name: {"count": count, "share": round(count / total, 4) if total else 0.0}
            for name, count in zip(names, counts, strict=True)
        },
    }
    if widths:
        areas = np.sqrt(np.asarray(widths) * np.asarray(heights))
        report["box_size_px"] = {
            "min": round(float(areas.min()), 1),
            "p50": round(float(np.percentile(areas, 50)), 1),
            "p95": round(float(np.percentile(areas, 95)), 1),
            "max": round(float(areas.max()), 1),
        }
    return report


def _segmenter_report(cfg: dict, split: str) -> dict:
    images = cfg.get(f"{split}_images") or cfg["train_images"]
    masks = cfg.get(f"{split}_masks") or cfg["train_masks"]
    num_classes = int(cfg["num_classes"])
    ignore_index = int(cfg.get("ignore_index", 255))
    dataset = SegmentationFolderDataset(images, masks)
    counts = dataset.class_histogram(num_classes, ignore_index)
    total = sum(counts)
    return {
        "images": str(images),
        "pairs": len(dataset),
        "labelled_pixels": total,
        "per_class": {
            f"class_{index}": {"pixels": count, "share": round(count / total, 4) if total else 0.0}
            for index, count in enumerate(counts)
        },
        "absent_classes": [index for index, count in enumerate(counts) if count == 0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["detector", "segmenter"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    segmentation = args.task == "segmenter"
    schema = "segmentation_train" if segmentation else "detection_train"
    cfg = load_config(args.config, schema=schema)["data"]
    report = _segmenter_report(cfg, args.split) if segmentation else _detector_report(cfg, args.split)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
