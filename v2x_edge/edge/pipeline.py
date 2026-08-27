from __future__ import annotations

import itertools
import math
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from v2x_edge.fusion import CameraRadarLateFusion
from v2x_edge.localization import TrackToWorldProjector
from v2x_edge.safety import RiskEngine
from v2x_edge.tracking import MultiObjectTracker
from v2x_edge.types import Detection, RadarDetection, RiskEvent, Track, WorldObject
from v2x_edge.v2x import JsonV2XCodec, RoadsidePerceptionMessage
from v2x_edge.world import WorldModel


class _Detector(Protocol):
    def predict(self, image_bgr: np.ndarray) -> list[Detection]: ...


class _Segmenter(Protocol):
    def predict(self, image_bgr: np.ndarray) -> np.ndarray: ...


class _Transport(Protocol):
    def send(self, payload: bytes) -> None: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class EdgeStepResult:
    timestamp: float
    detections: list[Detection]
    tracks: list[Track]
    objects: list[WorldObject]
    risks: list[RiskEvent]
    payload: bytes
    segmentation_mask: np.ndarray | None = None


class EdgePipeline:
    def __init__(
        self,
        rsu_id: str,
        detector: _Detector | None,
        tracker: MultiObjectTracker,
        projector: TrackToWorldProjector,
        world_model: WorldModel,
        risk_engine: RiskEngine,
        segmenter: _Segmenter | None = None,
        codec: JsonV2XCodec | None = None,
        fusion: CameraRadarLateFusion | None = None,
        transport: _Transport | None = None,
        session_id: str | None = None,
    ) -> None:
        if not rsu_id:
            raise ValueError("rsu_id cannot be empty")
        self.rsu_id = rsu_id
        self.session_id = session_id or uuid.uuid4().hex
        self.detector = detector
        self.tracker = tracker
        self.segmenter = segmenter
        self.projector = projector
        self.world_model = world_model
        self.risk_engine = risk_engine
        self.codec = codec or JsonV2XCodec()
        self.fusion = fusion
        self.transport = transport
        self._sequence = itertools.count(1)
        self._closed = False

    def process_detections(
        self,
        detections: list[Detection],
        timestamp: float,
        radar: list[RadarDetection] | None = None,
    ) -> EdgeStepResult:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        tracks = self.tracker.update(detections, timestamp)
        objects = self.projector.project(tracks, timestamp=timestamp)
        if radar and self.fusion is not None:
            objects = self.fusion.fuse(objects, radar)
        self.world_model.update(objects, timestamp)
        scene = self.world_model.objects()
        risks = self.risk_engine.evaluate(scene, timestamp)
        message = RoadsidePerceptionMessage.from_scene(
            rsu_id=self.rsu_id,
            session_id=self.session_id,
            sequence=next(self._sequence),
            timestamp=timestamp,
            objects=scene,
            risks=risks,
        )
        payload = self.codec.encode(message)
        if self.transport is not None:
            self.transport.send(payload)
        return EdgeStepResult(timestamp, detections, tracks, scene, risks, payload)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float | None = None,
        radar: list[RadarDetection] | None = None,
    ) -> EdgeStepResult:
        if self.detector is None:
            raise RuntimeError("process_frame requires a detector")
        timestamp = time.time() if timestamp is None else float(timestamp)
        detections = self.detector.predict(frame_bgr)
        result = self.process_detections(detections, timestamp, radar=radar)
        if self.segmenter is not None:
            result.segmentation_mask = self.segmenter.predict(frame_bgr)
        return result

    def close(self) -> None:
        if not self._closed and self.transport is not None:
            self.transport.close()
        self._closed = True
