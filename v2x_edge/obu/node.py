from __future__ import annotations

import time
from typing import Any

from v2x_edge.v2x import JsonV2XCodec, UdpReceiver


class VirtualOBUNode:
    def __init__(
        self,
        receiver: UdpReceiver,
        max_message_age_seconds: float = 1.0,
        future_tolerance_seconds: float = 5.0,
        max_object_age_seconds: float | None = None,
        codec: JsonV2XCodec | None = None,
    ) -> None:
        if max_message_age_seconds <= 0.0:
            raise ValueError("max_message_age_seconds must be positive")
        if future_tolerance_seconds < 0.0:
            raise ValueError("future_tolerance_seconds must be >= 0")
        self.receiver = receiver
        self.max_message_age_seconds = float(max_message_age_seconds)
        self.future_tolerance_seconds = float(future_tolerance_seconds)
        self.max_object_age_seconds = (
            float(max_object_age_seconds)
            if max_object_age_seconds is not None
            else self.max_message_age_seconds
        )
        if self.max_object_age_seconds <= 0.0:
            raise ValueError("max_object_age_seconds must be positive")
        self.codec = codec or JsonV2XCodec()
        self.last_sequence: dict[tuple[str, str], int] = {}

    def decode_and_validate(self, payload: bytes, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        msg = self.codec.decode(payload)
        age = now - float(msg["timestamp"])
        if age > self.max_message_age_seconds:
            raise ValueError(f"Stale V2X message: age={age:.3f}s")
        if age < -self.future_tolerance_seconds:
            raise ValueError(f"V2X message is {-age:.3f}s in the future")

        for obj in msg["objects"]:
            object_age = now - float(obj["timestamp"])
            if object_age > self.max_object_age_seconds:
                raise ValueError(f"Stale V2X object: age={object_age:.3f}s")
            if object_age < -self.future_tolerance_seconds:
                raise ValueError(f"V2X object is {-object_age:.3f}s in the future")

        key = str(msg["rsu_id"]), str(msg["session_id"])
        sequence = int(msg["sequence"])
        previous = self.last_sequence.get(key)
        if previous is not None and sequence <= previous:
            raise ValueError(f"Duplicate/out-of-order V2X sequence {sequence} <= {previous}")
        self.last_sequence[key] = sequence
        return msg

    def receive_once(self) -> dict[str, Any]:
        payload, _ = self.receiver.receive()
        return self.decode_and_validate(payload)

    def close(self) -> None:
        self.receiver.close()
