"""Detection mAP (COCO protocol) and segmentation mIoU."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou

_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    # Make precision monotonically non-increasing, then sample at the recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, _RECALL_POINTS, side="left")
    sampled = np.where(indices < len(precision), precision[np.clip(indices, 0, len(precision) - 1)], 0.0)
    return float(sampled.mean())


class MeanAveragePrecision:
    def __init__(self, num_classes: int, iou_thresholds: list[float] | None = None) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        self.num_classes = int(num_classes)
        self.iou_thresholds = (
            np.asarray(iou_thresholds, dtype=np.float64)
            if iou_thresholds is not None
            else np.arange(0.5, 0.96, 0.05)
        )
        self.reset()

    def reset(self) -> None:
        self._predictions: list[dict[str, torch.Tensor]] = []
        self._targets: list[dict[str, torch.Tensor]] = []

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have the same length")
        for prediction, target in zip(predictions, targets, strict=True):
            self._predictions.append(
                {key: prediction[key].detach().cpu() for key in ("boxes", "scores", "labels")}
            )
            self._targets.append({key: target[key].detach().cpu() for key in ("boxes", "labels")})

    def compute(self) -> dict[str, float | list[float]]:
        per_class = np.full((len(self.iou_thresholds), self.num_classes - 1), np.nan)
        for class_id in range(1, self.num_classes):
            per_class[:, class_id - 1] = self._class_ap(class_id)

        # A class absent from the ground truth stays NaN and is excluded from the mean.
        present = ~np.all(np.isnan(per_class), axis=0)
        if not present.any():
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0, "per_class_ap": [0.0] * (self.num_classes - 1)}
        class_means = np.full(per_class.shape[1], np.nan)
        class_means[present] = np.nanmean(per_class[:, present], axis=0)

        def at(threshold: float) -> float:
            index = int(np.argmin(np.abs(self.iou_thresholds - threshold)))
            row = per_class[index]
            valid = ~np.isnan(row)
            return float(row[valid].mean()) if valid.any() else 0.0

        return {
            "map": float(class_means[present].mean()),
            "map_50": at(0.5),
            "map_75": at(0.75),
            "per_class_ap": [0.0 if np.isnan(value) else float(value) for value in class_means],
        }

    def _class_ap(self, class_id: int) -> np.ndarray:
        gt_boxes: dict[int, torch.Tensor] = {}
        total_gt = 0
        for image_index, target in enumerate(self._targets):
            keep = target["labels"] == class_id
            if keep.any():
                boxes = target["boxes"][keep]
                gt_boxes[image_index] = boxes
                total_gt += len(boxes)
        if total_gt == 0:
            return np.full(len(self.iou_thresholds), np.nan)

        image_ids: list[int] = []
        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        for image_index, prediction in enumerate(self._predictions):
            keep = prediction["labels"] == class_id
            if keep.any():
                boxes_list.append(prediction["boxes"][keep])
                scores_list.append(prediction["scores"][keep])
                image_ids.extend([image_index] * int(keep.sum()))
        if not boxes_list:
            return np.zeros(len(self.iou_thresholds))

        scores = torch.cat(scores_list)
        order = torch.argsort(scores, descending=True)
        boxes = torch.cat(boxes_list)[order]
        detection_images = np.asarray(image_ids)[order.numpy()]

        # IoU row of each detection against the ground truth of its own image.
        # Detections in images with no ground truth of this class keep an empty row.
        iou_rows: list[np.ndarray] = [np.zeros(0)] * len(boxes)
        for image_index in np.unique(detection_images):
            if int(image_index) not in gt_boxes:
                continue
            selection = np.flatnonzero(detection_images == image_index)
            matrix = box_iou(boxes[selection], gt_boxes[int(image_index)]).numpy()
            for row, detection_index in enumerate(selection):
                iou_rows[detection_index] = matrix[row]

        results = np.zeros(len(self.iou_thresholds))
        for threshold_index, threshold in enumerate(self.iou_thresholds):
            available = {index: np.ones(len(gt), dtype=bool) for index, gt in gt_boxes.items()}
            true_positive = np.zeros(len(boxes))
            for detection_index in range(len(boxes)):
                row = iou_rows[detection_index]
                if row.size == 0:
                    continue
                unclaimed = available[int(detection_images[detection_index])]
                candidates = np.where(unclaimed, row, -1.0)
                best = int(candidates.argmax())
                if candidates[best] >= threshold:
                    unclaimed[best] = False
                    true_positive[detection_index] = 1.0

            cumulative_tp = np.cumsum(true_positive)
            cumulative_fp = np.cumsum(1.0 - true_positive)
            recall = cumulative_tp / total_gt
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            results[threshold_index] = _interpolated_ap(recall, precision)
        return results


class SegmentationMetrics:
    """Streaming confusion matrix over class IDs, ignoring ignore_index pixels."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        indices = target[valid] * self.num_classes + prediction[valid].clamp(0, self.num_classes - 1)
        self.confusion += torch.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, float | list[float]]:
        confusion = self.confusion.double()
        true_positive = confusion.diag()
        predicted = confusion.sum(dim=0)
        actual = confusion.sum(dim=1)
        union = predicted + actual - true_positive

        present = actual > 0
        iou = torch.where(union > 0, true_positive / union.clamp(min=1e-12), torch.zeros_like(union))
        accuracy = torch.where(actual > 0, true_positive / actual.clamp(min=1e-12), torch.zeros_like(actual))
        total = actual.sum()
        return {
            "miou": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_accuracy": float(true_positive.sum() / total) if total > 0 else 0.0,
            "mean_accuracy": float(accuracy[present].mean()) if present.any() else 0.0,
            "per_class_iou": [float(value) for value in iou],
        }
