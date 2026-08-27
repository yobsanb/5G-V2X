"""Full training loops on tiny generated data: config -> train -> checkpoint -> inference."""

import json

import cv2
import numpy as np
import pytest
import torch

from v2x_edge.config import validate_detection_train_config, validate_segmentation_train_config
from v2x_edge.models import (
    CheckpointDetector,
    CheckpointSegmenter,
    build_dfine_from_checkpoint,
    build_segformer_from_checkpoint,
)
from v2x_edge.train import train_detector, train_segmenter

# Small offline architectures so the loops stay fast; decoder_layers >= 2 is a
# D-FINE weight-tying requirement, not a tuning choice.
DFINE_TINY = {
    "name": "dfine",
    "pretrained": False,
    "image_size": 128,
    "max_detections": 12,
    "config_overrides": {"encoder_layers": 1, "decoder_layers": 2, "num_queries": 12},
}
SEGFORMER_TINY = {
    "name": "segformer",
    "pretrained": False,
    "config_overrides": {
        "depths": [1, 1, 1, 1],
        "hidden_sizes": [16, 32, 64, 128],
        "decoder_hidden_size": 128,
    },
}


def _detection_data(tmp_path, count=4):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    records = []
    for index in range(count):
        name = f"frame_{index}.png"
        cv2.imwrite(str(images / name), rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
        records.append({"image": f"images/{name}", "boxes": [[8, 8, 48, 40]], "labels": [1]})
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return manifest


def _detection_config(tmp_path, manifest):
    return {
        "run": {"name": "det", "output_dir": str(tmp_path / "outputs"), "device": "cpu", "seed": 1},
        "data": {
            "train_manifest": str(manifest),
            "root": str(tmp_path),
            "num_classes": 2,
            "class_names": ["car"],
            "batch_size": 2,
            "num_workers": 0,
            "val_fraction": 0.5,
            "augment": {"horizontal_flip": 0.5, "scale_jitter": [0.9, 1.1]},
        },
        "model": {"name": "fasterrcnn", "pretrained": False, "trainable_backbone_layers": 6},
        "optim": {"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4, "clip_grad_norm": 10.0},
        "schedule": {"name": "cosine", "epochs": 1, "warmup_epochs": 0, "min_lr_factor": 0.01},
        "train": {"amp": False, "val_interval": 1, "early_stopping_patience": 0},
    }


def test_detector_training_produces_usable_checkpoint(tmp_path):
    cfg = _detection_config(tmp_path, _detection_data(tmp_path))
    validate_detection_train_config(cfg)
    summary = train_detector(cfg)

    assert summary["epochs_completed"] == 1
    assert summary["best_val_map"] is not None
    run = tmp_path / "outputs" / "det"
    assert (run / "config.yaml").is_file()
    assert (run / "summary.json").is_file()
    logged = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "val_map" in logged[0] and "train_loss" in logged[0]

    detector = CheckpointDetector(run / "checkpoints" / "best.pt", device="cpu")
    assert detector.class_names == ["car"]
    assert isinstance(detector.predict(np.zeros((64, 96, 3), dtype=np.uint8)), list)


def _segmentation_config(tmp_path, count=4):
    rng = np.random.default_rng(1)
    for split in ("train", "val"):
        images = tmp_path / split / "images"
        masks = tmp_path / split / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(count):
            cv2.imwrite(str(images / f"s{index}.png"), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[:, 32:] = 1
            cv2.imwrite(str(masks / f"s{index}.png"), mask)
    return {
        "run": {"name": "seg", "output_dir": str(tmp_path / "outputs"), "device": "cpu", "seed": 1},
        "data": {
            "train_images": str(tmp_path / "train" / "images"),
            "train_masks": str(tmp_path / "train" / "masks"),
            "val_images": str(tmp_path / "val" / "images"),
            "val_masks": str(tmp_path / "val" / "masks"),
            "num_classes": 2,
            "width": 64,
            "height": 48,
            "batch_size": 2,
            "num_workers": 0,
            "augment": {"horizontal_flip": 0.5, "scale_crop": [0.9, 1.1]},
        },
        "model": {"name": "lraspp", "pretrained": False},
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
        "schedule": {"name": "cosine", "epochs": 1, "warmup_epochs": 0, "min_lr_factor": 0.01},
        "train": {"amp": False, "val_interval": 1, "early_stopping_patience": 0},
    }


def test_segmenter_training_produces_usable_checkpoint(tmp_path):
    cfg = _segmentation_config(tmp_path)
    validate_segmentation_train_config(cfg)
    summary = train_segmenter(cfg)

    assert summary["epochs_completed"] == 1
    assert 0.0 <= summary["best_val_miou"] <= 1.0
    run = tmp_path / "outputs" / "seg"
    logged = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "val_miou" in logged[0]

    segmenter = CheckpointSegmenter(run / "checkpoints" / "best.pt", device="cpu")
    mask = segmenter.predict(np.zeros((48, 64, 3), dtype=np.uint8))
    assert mask.shape == (48, 64)


def test_training_rejects_mask_ids_outside_num_classes(tmp_path):
    cfg = _segmentation_config(tmp_path)
    bad = np.full((48, 64), 5, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "train" / "masks" / "s0.png"), bad)
    with pytest.raises(ValueError):
        train_segmenter(cfg)


def test_early_stopping_halts_detector_training(tmp_path):
    cfg = _detection_config(tmp_path, _detection_data(tmp_path))
    cfg["schedule"]["epochs"] = 6
    cfg["train"]["early_stopping_patience"] = 1
    summary = train_detector(cfg)
    assert summary["epochs_completed"] < 6


def test_checkpoint_payload_loads_under_weights_only(tmp_path):
    cfg = _detection_config(tmp_path, _detection_data(tmp_path))
    train_detector(cfg)
    path = tmp_path / "outputs" / "det" / "checkpoints" / "last.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["num_classes"] == 2
    assert payload["format_version"] == 3
    assert payload["epoch"] == 1


def test_dfine_training_produces_usable_checkpoint(tmp_path):
    cfg = _detection_config(tmp_path, _detection_data(tmp_path))
    cfg["model"] = dict(DFINE_TINY)
    validate_detection_train_config(cfg)
    summary = train_detector(cfg)

    assert summary["epochs_completed"] == 1
    assert summary["best_val_map"] is not None
    checkpoint = torch.load(
        tmp_path / "outputs" / "det" / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["model_type"] == "dfine"
    assert checkpoint["image_size"] == 128

    model, names = build_dfine_from_checkpoint(checkpoint)
    assert names == ["car"]
    with torch.inference_mode():
        prediction = model([torch.rand(3, 64, 96)])[0]
    assert prediction["boxes"].shape[1] == 4


def test_segformer_training_produces_usable_checkpoint(tmp_path):
    cfg = _segmentation_config(tmp_path)
    cfg["model"] = dict(SEGFORMER_TINY)
    validate_segmentation_train_config(cfg)
    summary = train_segmenter(cfg)

    assert summary["epochs_completed"] == 1
    assert 0.0 <= summary["best_val_miou"] <= 1.0
    checkpoint = torch.load(
        tmp_path / "outputs" / "seg" / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["model_type"] == "segformer"

    model = build_segformer_from_checkpoint(checkpoint)
    with torch.inference_mode():
        assert model(torch.rand(1, 3, 48, 64)).shape == (1, 2, 48, 64)


def test_edge_config_loads_a_trained_dfine_checkpoint(tmp_path):
    # The runtime factory must rebuild the architecture from the checkpoint alone.
    from v2x_edge.models import build_detector_from_config

    cfg = _detection_config(tmp_path, _detection_data(tmp_path))
    cfg["model"] = dict(DFINE_TINY)
    train_detector(cfg)
    detector = build_detector_from_config(
        {
            "backend": "checkpoint",
            "checkpoint": str(tmp_path / "outputs" / "det" / "checkpoints" / "best.pt"),
            "confidence_threshold": 0.0,
        },
        device="cpu",
    )
    assert detector.class_names == ["car"]
    assert isinstance(detector.predict(np.zeros((64, 96, 3), dtype=np.uint8)), list)


def test_edge_config_loads_a_trained_segformer_checkpoint(tmp_path):
    from v2x_edge.models import build_segmenter_from_config

    cfg = _segmentation_config(tmp_path)
    cfg["model"] = dict(SEGFORMER_TINY)
    train_segmenter(cfg)
    segmenter = build_segmenter_from_config(
        {
            "enabled": True,
            "backend": "checkpoint",
            "checkpoint": str(tmp_path / "outputs" / "seg" / "checkpoints" / "best.pt"),
        },
        device="cpu",
    )
    assert segmenter.predict(np.zeros((48, 64, 3), dtype=np.uint8)).shape == (48, 64)
