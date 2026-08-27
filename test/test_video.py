import cv2
import numpy as np

from v2x_edge.edge import OpenCVVideoSource


class _StubCapture:
    def __init__(self, positions_s: list[float]) -> None:
        self.positions_s = positions_s
        self.index = -1

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 20.0
        return self.positions_s[self.index] * 1000.0

    def read(self):
        self.index += 1
        if self.index >= len(self.positions_s):
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_file_timestamps_increase_when_media_position_stalls(monkeypatch):
    # Containers may report a frozen or rewound position; the tracker still
    # requires strictly increasing timestamps.
    positions = [0.0, 0.5, 0.5, 0.5, 0.1, 1.0, -1.0]
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: _StubCapture(positions))
    source = OpenCVVideoSource("clip.mp4")

    timestamps = []
    while True:
        ok, _, timestamp = source.read()
        if not ok:
            break
        timestamps.append(timestamp)
    source.close()

    assert len(timestamps) == len(positions)
    assert all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))
