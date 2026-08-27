from __future__ import annotations

import json
import math
from typing import Any

from .messages import RoadsidePerceptionMessage


def _reject_constant(value: str):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class JsonV2XCodec:
    def encode(self, message: RoadsidePerceptionMessage) -> bytes:
        return json.dumps(
            message.to_dict(),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def decode(self, payload: bytes) -> dict[str, Any]:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("V2X payload must be bytes")
        try:
            data = json.loads(bytes(payload).decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid V2X JSON payload") from exc
        self.validate(data)
        return data

    @staticmethod
    def validate(data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("V2X message must be a JSON object")
        required = {"schema_version", "rsu_id", "session_id", "sequence", "timestamp", "objects", "risks"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Missing V2X message fields: {sorted(missing)}")
        if data["schema_version"] != "0.2":
            raise ValueError(f"Unsupported V2X schema version: {data['schema_version']}")
        for key in ("rsu_id", "session_id"):
            if not isinstance(data[key], str) or not data[key]:
                raise ValueError(f"{key} must be a non-empty string")
        if isinstance(data["sequence"], bool) or not isinstance(data["sequence"], int) or data["sequence"] < 1:
            raise ValueError("sequence must be a positive integer")
        _number(data["timestamp"], "timestamp")
        if not isinstance(data["objects"], list) or not isinstance(data["risks"], list):
            raise ValueError("objects and risks must be lists")

        for index, obj in enumerate(data["objects"]):
            JsonV2XCodec._validate_object(obj, index)
        for index, risk in enumerate(data["risks"]):
            JsonV2XCodec._validate_risk(risk, index)

    @staticmethod
    def _validate_object(obj: Any, index: int) -> None:
        if not isinstance(obj, dict):
            raise ValueError(f"objects[{index}] must be an object")
        required = {
            "track_id",
            "object_type",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "heading_deg",
            "confidence",
            "timestamp",
            "source",
        }
        missing = required - obj.keys()
        if missing:
            raise ValueError(f"objects[{index}] missing fields: {sorted(missing)}")
        track_id = obj["track_id"]
        if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 1:
            raise ValueError(f"objects[{index}].track_id must be a positive integer")
        for key in ("object_type", "source"):
            if not isinstance(obj[key], str) or not obj[key]:
                raise ValueError(f"objects[{index}].{key} must be a non-empty string")
        for key in ("x", "y", "z", "vx", "vy", "heading_deg", "timestamp"):
            _number(obj[key], f"objects[{index}].{key}")
        confidence = _number(obj["confidence"], f"objects[{index}].confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"objects[{index}].confidence must be in [0, 1]")
        bbox = obj.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"objects[{index}].bbox must contain four numbers or null")
            values = [_number(value, f"objects[{index}].bbox") for value in bbox]
            if values[2] <= values[0] or values[3] <= values[1]:
                raise ValueError(f"objects[{index}].bbox has invalid dimensions")

    @staticmethod
    def _validate_risk(risk: Any, index: int) -> None:
        if not isinstance(risk, dict):
            raise ValueError(f"risks[{index}] must be an object")
        required = {"event_type", "severity", "object_ids", "timestamp"}
        missing = required - risk.keys()
        if missing:
            raise ValueError(f"risks[{index}] missing fields: {sorted(missing)}")
        for key in ("event_type", "severity"):
            if not isinstance(risk[key], str) or not risk[key]:
                raise ValueError(f"risks[{index}].{key} must be a non-empty string")
        ids = risk["object_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in ids)
        ):
            raise ValueError(f"risks[{index}].object_ids must be a non-empty list of positive integers")
        _number(risk["timestamp"], f"risks[{index}].timestamp")
        for key in ("time_to_event_s", "min_distance_m"):
            if risk.get(key) is not None:
                _number(risk[key], f"risks[{index}].{key}")
