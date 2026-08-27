import time

from v2x_edge.obu import VirtualOBUNode
from v2x_edge.v2x import JsonV2XCodec, RoadsidePerceptionMessage, UdpReceiver, UdpSender


def test_udp_loopback():
    receiver = UdpReceiver("127.0.0.1", 0, timeout_s=1.0)
    _, port = receiver.local_address
    sender = UdpSender("127.0.0.1", port)
    obu = VirtualOBUNode(receiver, max_message_age_seconds=1.0)
    now = time.time()
    msg = RoadsidePerceptionMessage.from_scene("RSU_TEST", 1, now, [], [])
    sender.send(JsonV2XCodec().encode(msg))
    decoded = obu.receive_once()
    assert decoded["rsu_id"] == "RSU_TEST"
    sender.close()
    obu.close()
