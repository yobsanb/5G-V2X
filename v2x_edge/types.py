from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return _finite(x1, y1, x2, y2) and x2 > x1 and y2 > y1


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    class_id: int = -1

    def __post_init__(self) -> None:
        if not _valid_bbox(self.bbox):
            raise ValueError(f"Invalid detection bbox: {self.bbox}")
        if not self.label:
            raise ValueError("Detection label cannot be empty")
        if not _finite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be in [0, 1]")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise ValueError("Detection class_id must be an integer")

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    age: int
    hits: int
    time_since_update: int
    timestamp: float

    def __post_init__(self) -> None:
        if self.track_id < 1:
            raise ValueError("track_id must be positive")
        if not _valid_bbox(self.bbox):
            raise ValueError(f"Invalid track bbox: {self.bbox}")
        if not self.label:
            raise ValueError("Track label cannot be empty")
        if self.age < 1 or self.hits < 1 or self.time_since_update < 0:
            raise ValueError("Track age/hits must be positive and time_since_update non-negative")
        if not _finite(self.score, self.timestamp) or not 0.0 <= self.score <= 1.0:
            raise ValueError("Track score must be in [0, 1] and timestamp must be finite")

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)


@dataclass(slots=True)
class RadarDetection:
    range_m: float
    azimuth_deg: float
    radial_velocity_mps: float
    confidence: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(
            self.range_m,
            self.azimuth_deg,
            self.radial_velocity_mps,
            self.confidence,
            self.timestamp,
        ):
            raise ValueError("Radar detection values must be finite")
        if self.range_m < 0.0:
            raise ValueError("Radar range cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Radar confidence must be in [0, 1]")

    def xy(self) -> tuple[float, float]:
        angle = math.radians(self.azimuth_deg)
        return self.range_m * math.cos(angle), self.range_m * math.sin(angle)


@dataclass(slots=True)
class WorldObject:
    track_id: int
    object_type: str
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading_deg: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    source: str = "camera"
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.track_id < 1:
            raise ValueError("track_id must be positive")
        if not self.object_type or not self.source:
            raise ValueError("object_type and source cannot be empty")
        if not _finite(
            self.x,
            self.y,
            self.z,
            self.vx,
            self.vy,
            self.heading_deg,
            self.confidence,
            self.timestamp,
        ):
            raise ValueError("WorldObject numeric fields must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("WorldObject confidence must be in [0, 1]")
        if self.bbox is not None and not _valid_bbox(self.bbox):
            raise ValueError(f"Invalid WorldObject bbox: {self.bbox}")

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx, self.vy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EgoVehicleState:
    x: float
    y: float
    vx: float
    vy: float
    heading_deg: float
    timestamp: float
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        if not _finite(self.x, self.y, self.vx, self.vy, self.heading_deg, self.timestamp):
            raise ValueError("Ego vehicle state must contain finite values")
        if self.latitude is not None and not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if self.longitude is not None and not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")


@dataclass(slots=True)
class RiskEvent:
    event_type: str
    severity: str
    object_ids: tuple[int, ...]
    timestamp: float
    time_to_event_s: float | None = None
    min_distance_m: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_type or not self.severity:
            raise ValueError("Risk event type and severity cannot be empty")
        if not self.object_ids or any(object_id < 1 for object_id in self.object_ids):
            raise ValueError("Risk object IDs must be non-empty and positive")
        values = [self.timestamp]
        if self.time_to_event_s is not None:
            values.append(self.time_to_event_s)
        if self.min_distance_m is not None:
            values.append(self.min_distance_m)
        if not _finite(*values):
            raise ValueError("Risk event numeric fields must be finite")
        if self.time_to_event_s is not None and self.time_to_event_s < 0.0:
            raise ValueError("time_to_event_s must be >= 0")
        if self.min_distance_m is not None and self.min_distance_m < 0.0:
            raise ValueError("min_distance_m must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
