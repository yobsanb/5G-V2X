from __future__ import annotations

import math

import numpy as np


class BBoxKalmanFilter:
    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4, 0] = self._bbox_to_measurement(bbox)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([4.0, 4.0, 10.0, 10.0]).astype(np.float64)
        self.Q_base = np.diag([0.1, 0.1, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0]).astype(np.float64)

    @staticmethod
    def _bbox_to_measurement(bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {bbox}")
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dtype=np.float64)

    @staticmethod
    def _measurement_to_bbox(measurement: np.ndarray) -> tuple[float, float, float, float]:
        cx, cy, w, h = (float(v) for v in measurement[:4])
        w = max(1e-3, w)
        h = max(1e-3, h)
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def _clamp_size(self) -> None:
        self.x[2, 0] = max(1e-3, self.x[2, 0])
        self.x[3, 0] = max(1e-3, self.x[3, 0])

    def predict(self, dt: float = 1.0) -> tuple[float, float, float, float]:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Kalman dt must be finite and positive")
        F = np.eye(8, dtype=np.float64)
        for index in range(4):
            F[index, index + 4] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base * max(dt, 0.1)
        self._clamp_size()
        return self.bbox

    def update(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        z = self._bbox_to_measurement(bbox).reshape(4, 1)
        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        ph_t = self.P @ self.H.T
        K = np.linalg.solve(S.T, ph_t.T).T
        self.x = self.x + K @ residual
        identity = np.eye(8, dtype=np.float64)
        correction = identity - K @ self.H
        self.P = correction @ self.P @ correction.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp_size()
        return self.bbox

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._measurement_to_bbox(self.x[:, 0])
