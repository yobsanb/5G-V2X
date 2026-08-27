from __future__ import annotations

import math
from dataclasses import dataclass

from v2x_edge.types import WorldObject


@dataclass(slots=True)
class ClosestApproach:
    time_s: float
    distance_m: float


def closest_approach(first: WorldObject, second: WorldObject, horizon_s: float) -> ClosestApproach:
    horizon_s = float(horizon_s)
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be finite and positive")
    px = second.x - first.x
    py = second.y - first.y
    vx = second.vx - first.vx
    vy = second.vy - first.vy
    velocity_squared = vx * vx + vy * vy
    if velocity_squared < 1e-9:
        return ClosestApproach(time_s=0.0, distance_m=math.hypot(px, py))
    time_s = -(px * vx + py * vy) / velocity_squared
    time_s = max(0.0, min(horizon_s, time_s))
    dx = px + vx * time_s
    dy = py + vy * time_s
    return ClosestApproach(time_s=time_s, distance_m=math.hypot(dx, dy))
