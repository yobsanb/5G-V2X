from v2x_edge.safety import RiskEngine, closest_approach
from v2x_edge.types import WorldObject


def make(track_id, kind, x, y, vx, vy, timestamp=1.0):
    return WorldObject(track_id, kind, x, y, vx=vx, vy=vy, timestamp=timestamp, confidence=1.0)


def test_closest_approach_crossing():
    car = make(1, "car", -10, 0, 5, 0)
    person = make(2, "person", 0, -4, 0, 2)
    approach = closest_approach(car, person, 5.0)
    assert approach.time_s > 0
    assert approach.distance_m < 1e-6


def test_risk_engine_emits_vru_warning():
    car = make(1, "car", -10, 0, 5, 0)
    person = make(2, "person", 0, -4, 0, 2)
    events = RiskEngine(horizon_seconds=5, collision_distance_m=1.0).evaluate([car, person], timestamp=1.0)
    assert len(events) == 1
    assert events[0].event_type == "vru_collision_risk"


def test_no_warning_for_diverging_objects():
    first = make(1, "car", 0, 0, -5, 0)
    second = make(2, "car", 10, 0, 5, 0)
    events = RiskEngine(horizon_seconds=5, collision_distance_m=2.0).evaluate([first, second], timestamp=1.0)
    assert events == []


def test_stale_objects_are_excluded():
    car = make(1, "car", -10, 0, 5, 0, timestamp=0.0)
    person = make(2, "person", 0, -4, 0, 2, timestamp=0.0)
    events = RiskEngine(max_object_age_seconds=0.5).evaluate([car, person], timestamp=1.0)
    assert events == []


def test_time_skewed_objects_are_not_paired():
    car = make(1, "car", -10, 0, 5, 0, timestamp=1.0)
    person = make(2, "person", 0, -4, 0, 2, timestamp=1.5)
    engine = RiskEngine(max_object_age_seconds=1.0, max_pair_time_skew_seconds=0.1)
    assert engine.evaluate([car, person], timestamp=1.5) == []
