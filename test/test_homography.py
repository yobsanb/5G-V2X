import numpy as np
import pytest

from v2x_edge.localization import HomographyProjector


def test_identity_homography():
    projector = HomographyProjector(np.eye(3))
    x, y = projector.project_point((12.5, 44.0))
    assert abs(x - 12.5) < 1e-9
    assert abs(y - 44.0) < 1e-9


def test_scaled_homography_is_valid():
    projector = HomographyProjector(np.eye(3) * 1e-8)
    x, y = projector.project_point((3.0, 4.0))
    assert abs(x - 3.0) < 1e-9
    assert abs(y - 4.0) < 1e-9


def test_estimate_homography():
    image = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    world = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    projector, _ = HomographyProjector.estimate(image, world, method=0)
    x, y = projector.project_point((50, 50))
    assert abs(x - 5.0) < 1e-6
    assert abs(y - 5.0) < 1e-6


def test_singular_homography_rejected():
    with pytest.raises(ValueError):
        HomographyProjector(np.zeros((3, 3)))
