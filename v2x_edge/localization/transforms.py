from __future__ import annotations

import math

import numpy as np


class RigidTransform2D:
    def __init__(self, x_m: float = 0.0, y_m: float = 0.0, yaw_deg: float = 0.0) -> None:
        if not all(math.isfinite(float(v)) for v in (x_m, y_m, yaw_deg)):
            raise ValueError("Transform values must be finite")
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.yaw_deg = float(yaw_deg)
        yaw = math.radians(self.yaw_deg)
        self.R = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        self.t = np.array([self.x_m, self.y_m], dtype=np.float64)

    @staticmethod
    def _point(value: tuple[float, float]) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Expected a finite 2-D vector")
        return point

    def apply(self, point_xy: tuple[float, float]) -> tuple[float, float]:
        point = self.R @ self._point(point_xy) + self.t
        return float(point[0]), float(point[1])

    def rotate_vector(self, vector_xy: tuple[float, float]) -> tuple[float, float]:
        vector = self.R @ self._point(vector_xy)
        return float(vector[0]), float(vector[1])
