import pytest

from v2x_edge.v2x import VendorRSUTransport


def test_vendor_transport_adapter():
    sent = []
    closed = []
    transport = VendorRSUTransport(lambda payload: sent.append(payload), lambda: closed.append(True))
    transport.send(b"abc")
    transport.close()
    assert sent == [b"abc"]
    assert closed == [True]


def test_vendor_transport_detects_partial_write():
    transport = VendorRSUTransport(lambda payload: 1)
    with pytest.raises(OSError):
        transport.send(b"abc")
