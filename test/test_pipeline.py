import numpy as np

from v2x_edge.edge import EdgePipeline
from v2x_edge.localization import HomographyProjector, TrackToWorldProjector
from v2x_edge.safety import RiskEngine
from v2x_edge.tracking import MultiObjectTracker
from v2x_edge.types import Detection
from v2x_edge.v2x import JsonV2XCodec
from v2x_edge.world import WorldModel


def test_pipeline_from_detections():
    pipeline = EdgePipeline(
        rsu_id="RSU_TEST",
        detector=None,
        tracker=MultiObjectTracker(iou_threshold=0.1, max_age=2, min_hits=1),
        projector=TrackToWorldProjector(HomographyProjector(np.eye(3))),
        world_model=WorldModel(stale_after_seconds=1.0),
        risk_engine=RiskEngine(),
    )
    result = pipeline.process_detections(
        [Detection((10, 10, 30, 30), "car", 0.9)],
        timestamp=100.0,
    )
    assert len(result.tracks) == 1
    assert len(result.objects) == 1
    decoded = JsonV2XCodec().decode(result.payload)
    assert decoded["objects"][0]["track_id"] == 1
