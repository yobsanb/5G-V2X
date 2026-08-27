from v2x_edge.fusion import CameraRadarLateFusion
from v2x_edge.localization import RigidTransform2D
from v2x_edge.types import RadarDetection, WorldObject


def test_radar_transform_and_velocity_fusion():
    obj = WorldObject(1, "car", 1.0, 12.0, confidence=0.8, timestamp=10.0)
    radar = RadarDetection(10.0, 0.0, 5.0, confidence=1.0, timestamp=10.0)
    fusion = CameraRadarLateFusion(
        max_distance_m=0.5,
        radar_velocity_weight=1.0,
        radar_to_world=RigidTransform2D(x_m=1.0, y_m=2.0, yaw_deg=90.0),
    )
    result = fusion.fuse([obj], [radar])
    assert abs(result[0].vx) < 1e-6
    assert abs(result[0].vy - 5.0) < 1e-6
    assert result[0].source == "camera+radar"


def test_radar_fusion_preserves_camera_tangential_velocity():
    obj = WorldObject(1, "car", 10.0, 0.0, vx=5.0, vy=2.0, confidence=0.8, timestamp=10.0)
    radar = RadarDetection(10.0, 0.0, 8.0, confidence=1.0, timestamp=10.0)
    result = CameraRadarLateFusion(max_distance_m=0.5, radar_velocity_weight=1.0).fuse([obj], [radar])
    assert abs(result[0].vx - 8.0) < 1e-6
    assert abs(result[0].vy - 2.0) < 1e-6


def test_radar_time_gate_prevents_fusion():
    obj = WorldObject(1, "car", 10.0, 0.0, confidence=0.8, timestamp=10.0)
    radar = RadarDetection(10.0, 0.0, 5.0, confidence=1.0, timestamp=11.0)
    result = CameraRadarLateFusion(max_time_delta_s=0.1).fuse([obj], [radar])
    assert result[0].source == "camera"
    assert result[0].vx == 0.0
