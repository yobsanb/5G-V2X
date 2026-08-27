from __future__ import annotations

import math

from v2x_edge.types import WorldObject


class WorldModel:
    def __init__(self, stale_after_seconds: float = 1.0) -> None:
        if stale_after_seconds < 0.0:
            raise ValueError("stale_after_seconds must be >= 0")
        self.stale_after_seconds = float(stale_after_seconds)
        self._objects: dict[int, WorldObject] = {}
        self._last_timestamp: float | None = None

    def update(self, observations: list[WorldObject], timestamp: float) -> None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("World model timestamps cannot move backwards")
        self._last_timestamp = timestamp
        for obj in observations:
            self._objects[obj.track_id] = obj
        self.prune(timestamp)

    def prune(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, obj in self._objects.items()
            if timestamp - obj.timestamp > self.stale_after_seconds
        ]
        for track_id in stale:
            del self._objects[track_id]

    def objects(self) -> list[WorldObject]:
        return sorted(self._objects.values(), key=lambda obj: obj.track_id)

    def get(self, track_id: int) -> WorldObject | None:
        return self._objects.get(track_id)

    def clear(self) -> None:
        self._objects.clear()
        self._last_timestamp = None
