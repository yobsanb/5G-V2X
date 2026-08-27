from __future__ import annotations

import cv2
import numpy as np

from v2x_edge.types import Detection, Track

_COLOR = (255, 255, 255)


def _draw(frame: np.ndarray, boxes: list[tuple[tuple[float, ...], str]]) -> np.ndarray:
    out = frame.copy()
    for bbox, text in boxes:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR, 2)
        cv2.putText(
            out, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR, 1, cv2.LINE_AA
        )
    return out


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    return _draw(frame, [(d.bbox, f"{d.label} {d.score:.2f}") for d in detections])


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    return _draw(frame, [(t.bbox, f"#{t.track_id} {t.label}") for t in tracks])
