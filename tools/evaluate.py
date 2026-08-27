#!/usr/bin/env python3
"""Evaluate a trained checkpoint: detection mAP or segmentation mIoU."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from v2x_edge.config import load_config
from v2x_edge.data import (
    DetectionManifestDataset,
    SegmentationFolderDataset,
    build_segmentation_transform,
    detection_collate_fn,
)
from v2x_edge.models import (
    TrainableFasterRCNN,
    TrainableLRASPP,
    build_dfine_from_checkpoint,
    build_segformer_from_checkpoint,
)
from v2x_edge.train import evaluate_detector, evaluate_segmenter
from v2x_edge.utils import resolve_device

DETECTION_TYPES = {"dfine", "fasterrcnn_mobilenet_v3_large_320_fpn"}
SEGMENTATION_TYPES = {"segformer", "lraspp_mobilenet_v3_large"}


def _load(checkpoint_path: str, device: torch.device, segmentation: bool):
    """Rebuild whichever architecture the checkpoint records having been trained."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "num_classes" not in checkpoint:
        raise ValueError("Invalid checkpoint format")
    num_classes = int(checkpoint["num_classes"])
    model_type = str(checkpoint.get("model_type", ""))

    # Fail clearly on a task/checkpoint mismatch rather than deep inside the eval loop.
    expected = SEGMENTATION_TYPES if segmentation else DETECTION_TYPES
    if model_type and model_type not in expected:
        task = "segmenter" if segmentation else "detector"
        raise ValueError(f"Checkpoint is a '{model_type}' model; it cannot be evaluated as a {task}")

    if model_type == "dfine":
        model, _ = build_dfine_from_checkpoint(checkpoint)
    elif model_type == "segformer":
        model = build_segformer_from_checkpoint(checkpoint)
    else:
        model = (
            TrainableLRASPP(num_classes, pretrained_backbone=False)
            if segmentation
            else TrainableFasterRCNN(num_classes, pretrained_backbone=False)
        )
        model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), num_classes, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["detector", "segmenter"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    segmentation = args.task == "segmenter"
    schema = "segmentation_train" if segmentation else "detection_train"
    cfg = load_config(args.config, schema=schema)["data"]
    model, num_classes, checkpoint = _load(args.checkpoint, device, segmentation)
    workers = int(cfg.get("num_workers", 0))
    batch_size = int(cfg.get("batch_size", 2))

    if segmentation:
        images = cfg.get(f"{args.split}_images") or cfg["train_images"]
        masks = cfg.get(f"{args.split}_masks") or cfg["train_masks"]
        ignore_index = int(cfg.get("ignore_index", 255))
        size = (int(cfg.get("width", 640)), int(cfg.get("height", 384)))
        dataset = SegmentationFolderDataset(
            images, masks, transform=build_segmentation_transform(None, size, ignore_index, training=False)
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
        metrics = evaluate_segmenter(model, loader, device, num_classes, ignore_index)
    else:
        manifest = cfg.get(f"{args.split}_manifest") or cfg["train_manifest"]
        dataset = DetectionManifestDataset(manifest, cfg.get("root"))
        dataset.validate_label_range(num_classes)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, collate_fn=detection_collate_fn
        )
        metrics = evaluate_detector(model, loader, device, num_classes)

    names = checkpoint.get("class_names")
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "samples": len(dataset),
                "trained_epoch": checkpoint.get("epoch"),
                "class_names": names,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
