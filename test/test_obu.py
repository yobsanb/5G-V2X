import time

import pytest

from v2x_edge.obu import VirtualOBUNode
from v2x_edge.v2x import JsonV2XCodec, RoadsidePerceptionMessage, UdpReceiver


def _payload(session: str, sequence: int, timestamp: float) -> bytes:
    message = RoadsidePerceptionMessage.from_scene(
        "RSU",
        sequence,
        timestamp,
        [],
        [],
        session_id=session,
    )
    return JsonV2XCodec().encode(message)


def test_obu_sequence_is_scoped_by_session():
    receiver = UdpReceiver("127.0.0.1", 0)
    node = VirtualOBUNode(receiver)
    now = time.time()
    node.decode_and_validate(_payload("A", 1, now), now=now)
    node.decode_and_validate(_payload("B", 1, now), now=now)
    with pytest.raises(ValueError):
        node.decode_and_validate(_payload("A", 1, now), now=now)
    node.close()


def test_obu_rejects_stale_message():
    receiver = UdpReceiver("127.0.0.1", 0)
    node = VirtualOBUNode(receiver, max_message_age_seconds=0.5)
    now = time.time()
    with pytest.raises(ValueError):
        node.decode_and_validate(_payload("A", 1, now - 1.0), now=now)
    node.close()
