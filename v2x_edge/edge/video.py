from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class OpenCVVideoSource:
    def __init__(
        self,
        source: int | str | Path = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        realtime: bool = False,
    ) -> None:
        self.is_file = isinstance(source, (str, Path))
        capture_source = str(source) if isinstance(source, Path) else source
        self.cap = cv2.VideoCapture(capture_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        self.nominal_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.nominal_fps <= 0.0:
            self.nominal_fps = float(fps or 30.0)
        self.realtime = bool(realtime and self.is_file)
        self._first_media_s: float | None = None
        self._last_media_s: float | None = None
        self._wall_start = time.time()

    def _file_timestamp(self) -> float:
        # Timestamps must increase strictly: containers may report a missing,
        # frozen, or rewound position, so fall back to the nominal frame period.
        media_s = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if self._last_media_s is None:
            media_s = max(0.0, media_s)
            self._first_media_s = media_s
        elif not media_s > self._last_media_s:
            media_s = self._last_media_s + 1.0 / self.nominal_fps
        self._last_media_s = media_s
        timestamp = self._wall_start + (media_s - self._first_media_s)
        if self.realtime:
            delay = timestamp - time.time()
            if delay > 0.0:
                time.sleep(delay)
        return timestamp

    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        ok, frame = self.cap.read()
        if not ok:
            return False, frame, None
        timestamp = self._file_timestamp() if self.is_file else time.time()
        return True, frame, timestamp

    def close(self) -> None:
        self.cap.release()
