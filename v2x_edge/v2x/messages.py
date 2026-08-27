from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from v2x_edge.types import RiskEvent, WorldObject


@dataclass(slots=True)
class RoadsidePerceptionMessage:
    schema_version: str
    rsu_id: str
    session_id: str
    sequence: int
    timestamp: float
    objects: list[dict[str, Any]]
    risks: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.schema_version or not self.rsu_id or not self.session_id:
            raise ValueError("schema_version, rsu_id and session_id cannot be empty")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("message timestamp must be finite")

    @classmethod
    def from_scene(
        cls,
        rsu_id: str,
        sequence: int,
        timestamp: float,
        objects: list[WorldObject],
        risks: list[RiskEvent],
        session_id: str = "default",
    ) -> "RoadsidePerceptionMessage":
        return cls(
            schema_version="0.2",
            rsu_id=rsu_id,
            session_id=session_id,
            sequence=int(sequence),
            timestamp=float(timestamp),
            objects=[obj.to_dict() for obj in objects],
            risks=[risk.to_dict() for risk in risks],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
