#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket

from v2x_edge.config import load_config
from v2x_edge.obu import VirtualOBUNode
from v2x_edge.v2x import UdpReceiver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a virtual OBU UDP receiver")
    parser.add_argument("--config", default="config/edge.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-age", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    port = args.port if args.port is not None else int(config["v2x"]["port"])
    max_age = (
        args.max_age
        if args.max_age is not None
        else float(config["v2x"]["max_message_age_seconds"])
    )
    receiver = UdpReceiver(args.host, port, timeout_s=1.0)
    node = VirtualOBUNode(receiver, max_message_age_seconds=max_age)
    print(f"Virtual OBU listening on {receiver.local_address}")
    try:
        while True:
            try:
                message = node.receive_once()
            except socket.timeout:
                continue
            print(
                f"rsu={message['rsu_id']} session={message['session_id']} seq={message['sequence']} "
                f"objects={len(message['objects'])} risks={len(message['risks'])}"
            )
            for risk in message["risks"]:
                print("  WARNING:", risk["message"])
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


if __name__ == "__main__":
    main()
