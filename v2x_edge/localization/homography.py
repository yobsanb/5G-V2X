from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from v2x_edge.types import Track, WorldObject


class HomographyProjector:
    def __init__(self, image_to_world: np.ndarray) -> None:
        matrix = np.asarray(image_to_world, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Homography must be a finite 3x3 matrix")
        if np.linalg.matrix_rank(matrix) < 3:
            raise ValueError("Homography matrix is singular")
        scale = matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else np.linalg.norm(matrix)
        if not math.isfinite(float(scale)) or abs(float(scale)) < 1e-12:
            raise ValueError("Invalid homography scale")
        self.H = matrix / scale

    @classmethod
    def from_file(cls, path: str | Path) -> "HomographyProjector":
        return cls(np.load(Path(path), allow_pickle=False))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.H)

    def project_point(self, point_xy: tuple[float, float]) -> tuple[float, float]:
        x, y = (float(value) for value in point_xy)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Image point must be finite")
        projected = self.H @ np.array([x, y, 1.0], dtype=np.float64)
        denominator = float(projected[2])
        if abs(denominator) < 1e-12:
            raise ValueError("Point projects to infinity under the homography")
        world = projected[:2] / denominator
        if not np.isfinite(world).all():
            raise ValueError("Homography projection produced a non-finite point")
        return float(world[0]), float(world[1])

    @staticmethod
    def estimate(
        image_points: np.ndarray,
        world_points: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 0.5,
    ) -> tuple["HomographyProjector", np.ndarray | None]:
        image_points = np.asarray(image_points, dtype=np.float64)
        world_points = np.asarray(world_points, dtype=np.float64)
        if image_points.shape != world_points.shape or image_points.ndim != 2 or image_points.shape[1] != 2:
            raise ValueError("image_points and world_points must both have shape Nx2")
        if len(image_points) < 4:
            raise ValueError("At least four point correspondences are required")
        if not np.isfinite(image_points).all() or not np.isfinite(world_points).all():
            raise ValueError("Calibration points must be finite")
        if ransac_reproj_threshold <= 0.0:
            raise ValueError("ransac_reproj_threshold must be positive")
        matrix, mask = cv2.findHomography(
            image_points,
            world_points,
            method,
            float(ransac_reproj_threshold),
        )
        if matrix is None:
            raise RuntimeError("Homography estimation failed")
        return HomographyProjector(matrix), mask


class TrackToWorldProjector:
    def __init__(
        self,
        homography: HomographyProjector,
        velocity_alpha: float = 0.55,
        history_retention_s: float = 5.0,
    ) -> None:
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0, 1]")
        if history_retention_s <= 0.0:
            raise ValueError("history_retention_s must be positive")
        self.homography = homography
        self.velocity_alpha = float(velocity_alpha)
        self.history_retention_s = float(history_retention_s)
        self._history: dict[int, tuple[float, float, float, float, float, float]] = {}

    def _prune(self, timestamp: float) -> None:
        for track_id, state in list(self._history.items()):
            if timestamp - state[2] > self.history_retention_s:
                del self._history[track_id]

    def project(self, tracks: list[Track], timestamp: float | None = None) -> list[WorldObject]:
        if timestamp is None:
            timestamp = max((track.timestamp for track in tracks), default=None)
        if timestamp is not None:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("timestamp must be finite")
            self._prune(timestamp)

        objects: list[WorldObject] = []
        for track in tracks:
            x, y = self.homography.project_point(track.bottom_center)
            vx = vy = 0.0
            heading = 0.0
            previous = self._history.get(track.track_id)
            if previous is not None:
                px, py, pt, pvx, pvy, previous_heading = previous
                heading = previous_heading
                dt = track.timestamp - pt
                if dt > 1e-4:
                    raw_vx = (x - px) / dt
                    raw_vy = (y - py) / dt
                    alpha = self.velocity_alpha
                    vx = alpha * raw_vx + (1.0 - alpha) * pvx
                    vy = alpha * raw_vy + (1.0 - alpha) * pvy
                    if math.hypot(vx, vy) > 1e-4:
                        heading = math.degrees(math.atan2(vy, vx))
            self._history[track.track_id] = (x, y, track.timestamp, vx, vy, heading)
            objects.append(
                WorldObject(
                    track_id=track.track_id,
                    object_type=track.label,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    heading_deg=heading,
                    confidence=track.score,
                    timestamp=track.timestamp,
                    source="camera",
                    bbox=track.bbox,
                )
            )
        return objects
