"""Pin the vision-direct marker geometry.

Every sign here is one a landing can hide: fly toward a marker that is 40 cm
the wrong way and it still looks like it is tracking, right up to touchdown.
"""

import math

import numpy as np
import pytest

from mpc_landing.mpc_landing_node import (
    enu_yaw_from_quaternion,
    marker_enu_from_nadir_camera,
)

EAST = 0.0
NORTH = math.pi / 2


def test_marker_on_the_optical_axis_is_directly_below():
    """5 m down the lens, nothing sideways: same x/y, 5 m lower."""
    m = marker_enu_from_nadir_camera([0.0, 0.0, 5.0], (10.0, 20.0, 30.0), EAST)
    np.testing.assert_allclose(m, [10.0, 20.0, 25.0], atol=1e-12)


def test_range_sets_height_regardless_of_heading():
    for yaw in (EAST, NORTH, -1.2, 2.9):
        m = marker_enu_from_nadir_camera([0.3, -0.4, 7.5], (0.0, 0.0, 9.0), yaw)
        assert m[2] == pytest.approx(1.5)


def test_top_of_the_image_is_the_nose():
    """Optical -y is up the image. Facing east, that must be further east."""
    m = marker_enu_from_nadir_camera([0.0, -2.0, 5.0], (0.0, 0.0, 5.0), EAST)
    np.testing.assert_allclose(m[:2], [2.0, 0.0], atol=1e-12)


def test_right_of_the_image_is_the_vehicle_right():
    """Facing east, the vehicle's right is south, i.e. -y in ENU."""
    m = marker_enu_from_nadir_camera([2.0, 0.0, 5.0], (0.0, 0.0, 5.0), EAST)
    np.testing.assert_allclose(m[:2], [0.0, -2.0], atol=1e-12)


def test_the_same_pixel_offset_rotates_with_the_airframe():
    """Nose-north, a marker up the image must be north, not east."""
    m = marker_enu_from_nadir_camera([0.0, -2.0, 5.0], (0.0, 0.0, 5.0), NORTH)
    np.testing.assert_allclose(m[:2], [0.0, 2.0], atol=1e-12)


def test_offset_magnitude_is_preserved_under_rotation():
    tvec = [0.7, -1.1, 4.0]
    base = marker_enu_from_nadir_camera(tvec, (0.0, 0.0, 4.0), EAST)
    for yaw in (0.4, 1.9, -2.7):
        m = marker_enu_from_nadir_camera(tvec, (0.0, 0.0, 4.0), yaw)
        assert np.linalg.norm(m[:2]) == pytest.approx(np.linalg.norm(base[:2]))


def test_vehicle_position_is_a_pure_offset():
    tvec = [0.5, 0.25, 3.0]
    a = marker_enu_from_nadir_camera(tvec, (0.0, 0.0, 0.0), 1.1)
    b = marker_enu_from_nadir_camera(tvec, (100.0, -50.0, 8.0), 1.1)
    np.testing.assert_allclose(b - a, [100.0, -50.0, 8.0], atol=1e-9)


@pytest.mark.parametrize('yaw', [0.0, 0.9, -2.2, 3.0])
def test_enu_yaw_round_trip(yaw):
    """The quaternion MAVROS publishes for a level vehicle at this heading."""
    q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    assert enu_yaw_from_quaternion(*q) == pytest.approx(yaw)


def test_descent_gate_sees_height_above_the_marker():
    """What _descend now measures: p_d[2] - tgt[2] must be the vision range."""
    vehicle = (3.0, -4.0, 12.0)
    m = marker_enu_from_nadir_camera([0.9, 0.2, 2.4], vehicle, 0.6)
    assert vehicle[2] - m[2] == pytest.approx(2.4)
