import torch

from v2x_edge.train import MeanAveragePrecision, SegmentationMetrics


def _prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_perfect_detection_scores_one():
    boxes = [[0, 0, 10, 10], [20, 20, 40, 40]]
    metric = MeanAveragePrecision(num_classes=2)
    metric.update([_prediction(boxes, [0.9, 0.8], [1, 1])], [_target(boxes, [1, 1])])
    result = metric.compute()
    assert abs(result["map"] - 1.0) < 1e-6
    assert abs(result["map_50"] - 1.0) < 1e-6


def test_missed_detection_halves_recall():
    metric = MeanAveragePrecision(num_classes=2, iou_thresholds=[0.5])
    metric.update(
        [_prediction([[0, 0, 10, 10]], [0.9], [1])],
        [_target([[0, 0, 10, 10], [20, 20, 40, 40]], [1, 1])],
    )
    # One of two objects found at full precision: 101-point AP is about 0.5.
    assert 0.45 < metric.compute()["map_50"] < 0.55


def test_duplicate_detection_counts_as_false_positive():
    box = [[0, 0, 10, 10]]
    metric = MeanAveragePrecision(num_classes=2, iou_thresholds=[0.5])
    metric.update([_prediction(box * 2, [0.9, 0.8], [1, 1])], [_target(box, [1])])
    # Recall is complete but the second box is unmatched, so AP stays at 1.0
    # only because precision at full recall is still 1.0 for the first detection.
    assert abs(metric.compute()["map_50"] - 1.0) < 1e-6


def test_wrong_class_scores_zero():
    metric = MeanAveragePrecision(num_classes=3, iou_thresholds=[0.5])
    metric.update([_prediction([[0, 0, 10, 10]], [0.9], [2])], [_target([[0, 0, 10, 10]], [1])])
    assert metric.compute()["map_50"] == 0.0


def test_absent_class_does_not_drag_down_map():
    boxes = [[0, 0, 10, 10]]
    metric = MeanAveragePrecision(num_classes=4, iou_thresholds=[0.5])
    metric.update([_prediction(boxes, [0.9], [1])], [_target(boxes, [1])])
    # Classes 2 and 3 never appear in the ground truth and must be skipped.
    assert abs(metric.compute()["map_50"] - 1.0) < 1e-6


def test_segmentation_metrics_perfect_and_ignore_index():
    metric = SegmentationMetrics(num_classes=2, ignore_index=255)
    target = torch.tensor([[[0, 1], [255, 1]]])
    metric.update(torch.tensor([[[0, 1], [1, 1]]]), target)
    result = metric.compute()
    assert abs(result["miou"] - 1.0) < 1e-6
    assert abs(result["pixel_accuracy"] - 1.0) < 1e-6


def test_segmentation_metrics_half_wrong():
    metric = SegmentationMetrics(num_classes=2)
    metric.update(torch.tensor([[[0, 0]]]), torch.tensor([[[0, 1]]]))
    result = metric.compute()
    assert abs(result["pixel_accuracy"] - 0.5) < 1e-6
    assert result["per_class_iou"][1] == 0.0
