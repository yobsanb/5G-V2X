from __future__ import annotations

import socket
from collections.abc import Callable


class UdpSender:
    def __init__(self, host: str, port: int, max_datagram_size: int = 65507) -> None:
        if not host:
            raise ValueError("host cannot be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not 1 <= int(max_datagram_size) <= 65507:
            raise ValueError("max_datagram_size must be in [1, 65507]")
        self.address = (host, int(port))
        self.max_datagram_size = int(max_datagram_size)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("UDP payload must be bytes")
        if len(payload) > self.max_datagram_size:
            raise ValueError(
                f"UDP payload is {len(payload)} bytes; configured maximum is {self.max_datagram_size}"
            )
        sent = self.sock.sendto(payload, self.address)
        if sent != len(payload):
            raise OSError(f"UDP send wrote {sent} of {len(payload)} bytes")

    def close(self) -> None:
        self.sock.close()


class UdpReceiver:
    def __init__(self, host: str, port: int, timeout_s: float = 1.0, max_datagram_size: int = 65507) -> None:
        if not host:
            raise ValueError("host cannot be empty")
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if not 1 <= int(max_datagram_size) <= 65507:
            raise ValueError("max_datagram_size must be in [1, 65507]")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.settimeout(float(timeout_s))
        self.max_datagram_size = int(max_datagram_size)

    @property
    def local_address(self) -> tuple[str, int]:
        host, port = self.sock.getsockname()
        return str(host), int(port)

    def receive(self) -> tuple[bytes, tuple[str, int]]:
        return self.sock.recvfrom(self.max_datagram_size)

    def close(self) -> None:
        self.sock.close()


class VendorRSUTransport:
    def __init__(
        self,
        send_fn: Callable[[bytes], int | None],
        close_fn: Callable[[], None] | None = None,
    ) -> None:
        if not callable(send_fn):
            raise TypeError("send_fn must be callable")
        self._send_fn = send_fn
        self._close_fn = close_fn

    def send(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("Vendor payload must be bytes")
        result = self._send_fn(payload)
        if isinstance(result, int) and result != len(payload):
            raise OSError(f"Vendor transport accepted {result} of {len(payload)} bytes")

    def close(self) -> None:
        if self._close_fn is not None:
            self._close_fn()
