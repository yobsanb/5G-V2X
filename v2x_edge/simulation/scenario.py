from __future__ import annotations

import math
from dataclasses import dataclass

from v2x_edge.types import WorldObject


@dataclass(slots=True)
class SimObject:
    track_id: int
    object_type: str
    x: float
    y: float
    vx: float
    vy: float
    confidence: float = 1.0

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("Simulation dt must be positive")
        self.x += self.vx * dt
        self.y += self.vy * dt

    def as_world(self, timestamp: float) -> WorldObject:
        speed = math.hypot(self.vx, self.vy)
        heading = math.degrees(math.atan2(self.vy, self.vx)) if speed > 1e-6 else 0.0
        return WorldObject(
            track_id=self.track_id,
            object_type=self.object_type,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            heading_deg=heading,
            confidence=self.confidence,
            timestamp=timestamp,
            source="simulation",
        )


class Scenario:
    def __init__(self, objects: list[SimObject]) -> None:
        self.objects = objects
        self.time = 0.0

    def step(self, dt: float) -> list[WorldObject]:
        self.time += dt
        for obj in self.objects:
            obj.step(dt)
        return [obj.as_world(self.time) for obj in self.objects]


def crossing_scenario() -> Scenario:
    return Scenario(
        [
            SimObject(1, "car", x=-18.0, y=0.0, vx=7.0, vy=0.0),
            SimObject(2, "person", x=0.0, y=-8.0, vx=0.0, vy=2.8),
        ]
    )
