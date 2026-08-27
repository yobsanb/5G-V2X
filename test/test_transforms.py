import random

import torch

from v2x_edge.data.transforms import (
    DetectionHorizontalFlip,
    DetectionScaleJitter,
    SegmentationScaleCrop,
    build_detection_transform,
    build_segmentation_transform,
)


def _target(boxes):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {
        "boxes": boxes,
        "labels": torch.ones(len(boxes), dtype=torch.int64),
        "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
    }


def test_horizontal_flip_mirrors_boxes():
    image = torch.rand(3, 20, 100)
    image, target = DetectionHorizontalFlip(probability=1.0)(image, _target([[10, 5, 30, 15]]))
    assert target["boxes"].tolist() == [[70.0, 5.0, 90.0, 15.0]]
    assert torch.all(target["boxes"][:, 2] > target["boxes"][:, 0])


def test_flip_is_an_involution():
    flip = DetectionHorizontalFlip(probability=1.0)
    original = _target([[10, 5, 30, 15]])
    image, once = flip(torch.rand(3, 20, 100), original)
    _, twice = flip(image, once)
    assert torch.allclose(twice["boxes"], original["boxes"])


def test_scale_jitter_scales_image_and_boxes_together():
    random.seed(0)
    jitter = DetectionScaleJitter(min_scale=2.0, max_scale=2.0)
    image, target = jitter(torch.rand(3, 20, 40), _target([[4, 6, 12, 16]]))
    assert image.shape == (3, 40, 80)
    assert target["boxes"].tolist() == [[8.0, 12.0, 24.0, 32.0]]


def test_scale_jitter_drops_collapsed_boxes():
    random.seed(0)
    jitter = DetectionScaleJitter(min_scale=0.05, max_scale=0.05, min_box_size=2.0)
    _, target = jitter(torch.rand(3, 200, 200), _target([[0, 0, 10, 10], [0, 0, 180, 180]]))
    assert len(target["boxes"]) == 1
    assert len(target["labels"]) == 1


def test_segmentation_scale_crop_pads_with_ignore_index():
    random.seed(0)
    crop = SegmentationScaleCrop(size=(64, 48), min_scale=0.25, max_scale=0.25, ignore_index=255)
    image, mask = crop(torch.rand(3, 40, 40), torch.zeros(40, 40, dtype=torch.int64))
    assert image.shape == (3, 48, 64)
    assert mask.shape == (48, 64)
    assert 255 in set(mask.unique().tolist())


def test_builders_respect_disabled_config():
    assert build_detection_transform(None) is None
    assert build_detection_transform({}) is None
    evaluation = build_segmentation_transform({"horizontal_flip": 1.0}, (32, 16), 255, training=False)
    image, mask = evaluation(torch.rand(3, 40, 40), torch.zeros(40, 40, dtype=torch.int64))
    assert image.shape == (3, 16, 32)


def test_detection_builder_composes_requested_stages():
    transform = build_detection_transform(
        {"horizontal_flip": 0.5, "photometric": 0.5, "scale_jitter": [0.9, 1.1]}
    )
    assert len(transform.transforms) == 3
