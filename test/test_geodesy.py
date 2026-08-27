from v2x_edge.localization import LocalENUFrame, RigidTransform2D


def test_enu_origin_is_zero():
    frame = LocalENUFrame(35.0, 33.0, 100.0)
    e, n, u = frame.geodetic_to_enu(35.0, 33.0, 100.0)
    assert abs(e) < 1e-6
    assert abs(n) < 1e-6
    assert abs(u) < 1e-6


def test_rigid_transform_2d():
    tf = RigidTransform2D(x_m=1.0, y_m=2.0, yaw_deg=90.0)
    x, y = tf.apply((1.0, 0.0))
    assert abs(x - 1.0) < 1e-6
    assert abs(y - 3.0) < 1e-6
