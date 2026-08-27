from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from v2x_edge.types import Detection, Track

from .kalman import BBoxKalmanFilter


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(slots=True)
class _TrackState:
    track_id: int
    kf: BBoxKalmanFilter
    label: str
    score: float
    age: int
    hits: int
    time_since_update: int
    last_timestamp: float


class MultiObjectTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 12, min_hits: int = 2) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._next_id = itertools.count(1)
        self._tracks: list[_TrackState] = []
        self._last_timestamp: float | None = None

    def _predict(self, timestamp: float) -> None:
        for track in self._tracks:
            dt = timestamp - track.last_timestamp
            if dt <= 0.0:
                dt = 1.0 / 30.0
            track.kf.predict(dt)
            track.last_timestamp = timestamp
            track.age += 1
            track.time_since_update += 1

    def update(self, detections: Iterable[Detection], timestamp: float) -> list[Track]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("Tracker timestamps must increase strictly")
        self._last_timestamp = timestamp
        detections = list(detections)
        self._predict(timestamp)
        matched_detections: set[int] = set()

        if self._tracks and detections:
            cost = np.full((len(self._tracks), len(detections)), 1e6, dtype=np.float64)
            for track_index, track in enumerate(self._tracks):
                for detection_index, detection in enumerate(detections):
                    if track.label == detection.label:
                        cost[track_index, detection_index] = 1.0 - bbox_iou(track.kf.bbox, detection.bbox)

            rows, cols = linear_sum_assignment(cost)
            for track_index, detection_index in zip(rows, cols, strict=True):
                if cost[track_index, detection_index] >= 1e5:
                    continue
                iou = 1.0 - float(cost[track_index, detection_index])
                if iou < self.iou_threshold:
                    continue
                track = self._tracks[track_index]
                detection = detections[detection_index]
                track.kf.update(detection.bbox)
                track.score = detection.score
                track.hits += 1
                track.time_since_update = 0
                matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            self._tracks.append(
                _TrackState(
                    track_id=next(self._next_id),
                    kf=BBoxKalmanFilter(detection.bbox),
                    label=detection.label,
                    score=detection.score,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    last_timestamp=timestamp,
                )
            )

        self._tracks = [track for track in self._tracks if track.time_since_update <= self.max_age]
        return [
            Track(
                track_id=track.track_id,
                bbox=track.kf.bbox,
                label=track.label,
                score=track.score,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
                timestamp=timestamp,
            )
            for track in self._tracks
            if track.hits >= self.min_hits and track.time_since_update == 0
        ]
