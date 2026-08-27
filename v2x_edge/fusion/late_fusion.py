from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from v2x_edge.localization import RigidTransform2D
from v2x_edge.types import RadarDetection, WorldObject


class CameraRadarLateFusion:
    def __init__(
        self,
        max_distance_m: float = 4.0,
        radar_velocity_weight: float = 0.7,
        max_time_delta_s: float = 0.15,
        radar_to_world: RigidTransform2D | None = None,
    ) -> None:
        if max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be positive")
        if not 0.0 <= radar_velocity_weight <= 1.0:
            raise ValueError("radar_velocity_weight must be in [0, 1]")
        if max_time_delta_s <= 0.0:
            raise ValueError("max_time_delta_s must be positive")
        self.max_distance_m = float(max_distance_m)
        self.radar_velocity_weight = float(radar_velocity_weight)
        self.max_time_delta_s = float(max_time_delta_s)
        self.radar_to_world = radar_to_world or RigidTransform2D()

    def _radar_state(self, detection: RadarDetection) -> tuple[float, float, float, float, float]:
        sensor_x, sensor_y = detection.xy()
        world_x, world_y = self.radar_to_world.apply((sensor_x, sensor_y))
        angle = math.radians(detection.azimuth_deg)
        unit_x, unit_y = self.radar_to_world.rotate_vector((math.cos(angle), math.sin(angle)))
        norm = math.hypot(unit_x, unit_y)
        if norm <= 1e-12:
            raise ValueError("Invalid radar line-of-sight vector")
        return world_x, world_y, unit_x / norm, unit_y / norm, detection.radial_velocity_mps

    def fuse(self, objects: list[WorldObject], radar: list[RadarDetection]) -> list[WorldObject]:
        if not objects or not radar:
            return objects

        states = [self._radar_state(detection) for detection in radar]
        radar_xy = np.asarray([[state[0], state[1]] for state in states], dtype=np.float64)
        object_xy = np.asarray([[obj.x, obj.y] for obj in objects], dtype=np.float64)
        distances = np.linalg.norm(object_xy[:, None, :] - radar_xy[None, :, :], axis=2)
        cost = distances.copy()

        for object_index, obj in enumerate(objects):
            for radar_index, detection in enumerate(radar):
                if abs(obj.timestamp - detection.timestamp) > self.max_time_delta_s:
                    cost[object_index, radar_index] = 1e6

        rows, cols = linear_sum_assignment(cost)
        for object_index, radar_index in zip(rows, cols, strict=True):
            if cost[object_index, radar_index] >= 1e5:
                continue
            if distances[object_index, radar_index] > self.max_distance_m:
                continue

            obj = objects[object_index]
            detection = radar[radar_index]
            _, _, unit_x, unit_y, radar_radial = states[radar_index]
            camera_radial = obj.vx * unit_x + obj.vy * unit_y
            tangential_x = obj.vx - camera_radial * unit_x
            tangential_y = obj.vy - camera_radial * unit_y
            weight = self.radar_velocity_weight
            fused_radial = (1.0 - weight) * camera_radial + weight * radar_radial
            obj.vx = tangential_x + fused_radial * unit_x
            obj.vy = tangential_y + fused_radial * unit_y
            if obj.speed_mps > 1e-4:
                obj.heading_deg = math.degrees(math.atan2(obj.vy, obj.vx))
            obj.source = "camera+radar"
            obj.confidence = 0.5 * (obj.confidence + detection.confidence)
        return objects
