import json

import cv2
import numpy as np
import pytest

from v2x_edge.data import (
    DetectionManifestDataset,
    SegmentationFolderDataset,
    build_segmentation_transform,
    split_dataset,
)


def _write_manifest(tmp_path, records):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return manifest


def test_detection_dataset(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "frame.png"), np.zeros((32, 48, 3), dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path, [{"image": "images/frame.png", "boxes": [[2, 3, 20, 25]], "labels": [1]}]
    )
    dataset = DetectionManifestDataset(manifest)
    dataset.validate_label_range(2)
    tensor, target = dataset[0]
    assert tensor.shape == (3, 32, 48)
    assert target["boxes"].shape == (1, 4)
    assert dataset.label_histogram(2) == [0, 1]


def test_detection_dataset_rejects_out_of_bounds_box(tmp_path):
    cv2.imwrite(str(tmp_path / "frame.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    manifest = _write_manifest(tmp_path, [{"image": "frame.png", "boxes": [[0, 0, 30, 10]], "labels": [1]}])
    with pytest.raises(ValueError):
        DetectionManifestDataset(manifest)[0]


def test_split_dataset_is_deterministic_and_disjoint(tmp_path):
    cv2.imwrite(str(tmp_path / "frame.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    manifest = _write_manifest(
        tmp_path, [{"image": "frame.png", "boxes": [[0, 0, 10, 10]], "labels": [1]}] * 10
    )
    dataset = DetectionManifestDataset(manifest)
    train_a, val_a = split_dataset(dataset, 0.3, seed=7)
    train_b, val_b = split_dataset(dataset, 0.3, seed=7)
    assert val_a.indices == val_b.indices and train_a.indices == train_b.indices
    assert not set(train_a.indices) & set(val_a.indices)
    assert len(train_a) + len(val_a) == 10


def _segmentation_pair(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    cv2.imwrite(str(images / "frame.png"), np.zeros((24, 32, 3), dtype=np.uint8))
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[:, 16:] = 1
    cv2.imwrite(str(masks / "frame.png"), mask)
    return images, masks


def test_segmentation_dataset_resizes_through_transform(tmp_path):
    images, masks = _segmentation_pair(tmp_path)
    transform = build_segmentation_transform(None, (16, 12), 255, training=False)
    dataset = SegmentationFolderDataset(images, masks, transform=transform)
    image_tensor, mask_tensor = dataset[0]
    assert image_tensor.shape == (3, 12, 16)
    assert mask_tensor.shape == (12, 16)
    assert set(mask_tensor.unique().tolist()) == {0, 1}


def test_segmentation_class_histogram(tmp_path):
    images, masks = _segmentation_pair(tmp_path)
    counts = SegmentationFolderDataset(images, masks).class_histogram(2)
    assert counts == [24 * 16, 24 * 16]
