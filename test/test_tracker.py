from v2x_edge.tracking import MultiObjectTracker, bbox_iou
from v2x_edge.types import Detection


def test_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_tracker_keeps_id():
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=3, min_hits=1)
    first = tracker.update([Detection((10, 10, 30, 30), "car", 0.9)], timestamp=0.0)
    second = tracker.update([Detection((12, 10, 32, 30), "car", 0.9)], timestamp=0.1)
    assert len(first) == 1 and len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_tracker_prediction_clock_advances_on_missed_frames():
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=3, min_hits=1)
    tracker.update([Detection((10, 10, 30, 30), "car", 0.9)], timestamp=1.0)
    tracker.update([], timestamp=1.1)
    assert abs(tracker._tracks[0].last_timestamp - 1.1) < 1e-9
    tracker.update([], timestamp=1.2)
    assert abs(tracker._tracks[0].last_timestamp - 1.2) < 1e-9
