import pytest

from v2x_edge.types import RiskEvent, WorldObject
from v2x_edge.v2x import JsonV2XCodec, RoadsidePerceptionMessage


def test_codec_round_trip():
    objects = [WorldObject(1, "car", 1.0, 2.0, vx=3.0, timestamp=10.0, confidence=0.9)]
    risks = [RiskEvent("collision_risk", "warning", (1, 2), 10.0, 2.0, 1.0, "test")]
    message = RoadsidePerceptionMessage.from_scene("RSU_1", 7, 10.0, objects, risks, session_id="session")
    decoded = JsonV2XCodec().decode(JsonV2XCodec().encode(message))
    assert decoded["schema_version"] == "0.2"
    assert decoded["session_id"] == "session"
    assert decoded["rsu_id"] == "RSU_1"
    assert decoded["sequence"] == 7
    assert decoded["objects"][0]["object_type"] == "car"


def test_codec_rejects_nan():
    payload = (
        b'{"schema_version":"0.2","rsu_id":"R","session_id":"S","sequence":1,'
        b'"timestamp":NaN,"objects":[],"risks":[]}'
    )
    with pytest.raises(ValueError):
        JsonV2XCodec().decode(payload)


def test_codec_rejects_missing_object_fields():
    payload = (
        b'{"schema_version":"0.2","rsu_id":"R","session_id":"S","sequence":1,'
        b'"timestamp":1.0,"objects":[{}],"risks":[]}'
    )
    with pytest.raises(ValueError):
        JsonV2XCodec().decode(payload)
