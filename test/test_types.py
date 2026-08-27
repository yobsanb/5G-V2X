import pytest

from v2x_edge.types import Detection, RadarDetection, WorldObject


def test_invalid_detection_rejected():
    with pytest.raises(ValueError):
        Detection((1, 1, 1, 2), "car", 0.9)


def test_invalid_world_confidence_rejected():
    with pytest.raises(ValueError):
        WorldObject(1, "car", 0.0, 0.0, confidence=1.1)


def test_invalid_radar_range_rejected():
    with pytest.raises(ValueError):
        RadarDetection(-1.0, 0.0, 0.0)
