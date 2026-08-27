from __future__ import annotations

import itertools
import math

from v2x_edge.types import RiskEvent, WorldObject

from .collision import closest_approach

ROAD_USERS = {"person", "pedestrian", "bicycle", "car", "motorcycle", "bus", "truck"}
VRU = {"person", "pedestrian", "bicycle", "motorcycle"}


class RiskEngine:
    def __init__(
        self,
        horizon_seconds: float = 5.0,
        collision_distance_m: float = 2.5,
        minimum_relative_speed_mps: float = 0.25,
        max_object_age_seconds: float = 1.0,
        max_pair_time_skew_seconds: float = 0.2,
        critical_ttc_seconds: float = 1.5,
        warning_ttc_seconds: float = 3.0,
    ) -> None:
        if horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be positive")
        if collision_distance_m <= 0.0:
            raise ValueError("collision_distance_m must be positive")
        if minimum_relative_speed_mps < 0.0:
            raise ValueError("minimum_relative_speed_mps must be >= 0")
        if max_object_age_seconds < 0.0 or max_pair_time_skew_seconds < 0.0:
            raise ValueError("risk timing limits must be >= 0")
        if not 0.0 < critical_ttc_seconds <= warning_ttc_seconds <= horizon_seconds:
            raise ValueError("TTC thresholds must satisfy 0 < critical <= warning <= horizon")
        self.horizon_seconds = float(horizon_seconds)
        self.collision_distance_m = float(collision_distance_m)
        self.minimum_relative_speed_mps = float(minimum_relative_speed_mps)
        self.max_object_age_seconds = float(max_object_age_seconds)
        self.max_pair_time_skew_seconds = float(max_pair_time_skew_seconds)
        self.critical_ttc_seconds = float(critical_ttc_seconds)
        self.warning_ttc_seconds = float(warning_ttc_seconds)

    def evaluate(self, objects: list[WorldObject], timestamp: float) -> list[RiskEvent]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        candidates = [
            obj
            for obj in objects
            if obj.object_type in ROAD_USERS and 0.0 <= timestamp - obj.timestamp <= self.max_object_age_seconds
        ]
        events: list[RiskEvent] = []

        for first, second in itertools.combinations(candidates, 2):
            if abs(first.timestamp - second.timestamp) > self.max_pair_time_skew_seconds:
                continue
            relative_speed = math.hypot(second.vx - first.vx, second.vy - first.vy)
            if relative_speed < self.minimum_relative_speed_mps:
                continue
            approach = closest_approach(first, second, self.horizon_seconds)
            if approach.time_s <= 0.0 or approach.distance_m > self.collision_distance_m:
                continue

            if approach.time_s < self.critical_ttc_seconds:
                severity = "critical"
            elif approach.time_s < self.warning_ttc_seconds:
                severity = "warning"
            else:
                severity = "advisory"
            event_type = (
                "vru_collision_risk"
                if first.object_type in VRU or second.object_type in VRU
                else "collision_risk"
            )
            events.append(
                RiskEvent(
                    event_type=event_type,
                    severity=severity,
                    object_ids=(first.track_id, second.track_id),
                    timestamp=timestamp,
                    time_to_event_s=approach.time_s,
                    min_distance_m=approach.distance_m,
                    message=(
                        f"{event_type} between {first.object_type}#{first.track_id} and "
                        f"{second.object_type}#{second.track_id}"
                    ),
                    metadata={"relative_speed_mps": relative_speed},
                )
            )
        return sorted(
            events,
            key=lambda event: (
                math.inf if event.time_to_event_s is None else event.time_to_event_s,
                event.object_ids,
            ),
        )
