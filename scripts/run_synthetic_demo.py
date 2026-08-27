#!/usr/bin/env python3
from __future__ import annotations

import socket
import time

from v2x_edge.obu import VirtualOBUNode
from v2x_edge.safety import RiskEngine
from v2x_edge.simulation import crossing_scenario
from v2x_edge.v2x import JsonV2XCodec, RoadsidePerceptionMessage, UdpReceiver, UdpSender


def main() -> None:
    receiver = UdpReceiver("127.0.0.1", 0, timeout_s=0.5)
    _, port = receiver.local_address
    sender = UdpSender("127.0.0.1", port)
    obu = VirtualOBUNode(receiver, max_message_age_seconds=2.0)
    codec = JsonV2XCodec()
    risk_engine = RiskEngine(horizon_seconds=5.0, collision_distance_m=2.5)
    scenario = crossing_scenario()

    print(f"Synthetic RSU -> OBU UDP demo on localhost:{port}")
    try:
        for sequence in range(1, 13):
            objects = scenario.step(0.25)
            now = time.time()
            for obj in objects:
                obj.timestamp = now
            risks = risk_engine.evaluate(objects, now)
            message = RoadsidePerceptionMessage.from_scene(
                "RSU_SIM",
                sequence,
                now,
                objects,
                risks,
                session_id="synthetic-demo",
            )
            sender.send(codec.encode(message))
            decoded = obu.receive_once()
            print(
                f"seq={decoded['sequence']:02d} objects={len(decoded['objects'])} "
                f"risks={len(decoded['risks'])}"
            )
            for risk in decoded["risks"]:
                print(f"  {risk['severity']}: {risk['message']} t={risk['time_to_event_s']:.2f}s")
    except socket.timeout as exc:
        raise RuntimeError("Synthetic UDP loopback failed") from exc
    finally:
        sender.close()
        obu.close()


if __name__ == "__main__":
    main()
