"""Pin the landing TF chain's rotation conventions.

Every one of these is a sign that cannot be checked by looking at it, and each
wrong sign is a landing that misses in a plausible-looking direction rather than
an error anyone sees.  The nadir case in particular is the one the aircraft
spends its whole descent in.
"""

import math

import numpy as np
import pytest

from aruco_landing.landing_tf_node import (
    _OPTICAL_FROM_LINK,
    _ry,
    _rz,
    camera_rotation_in_body,
    enu_yaw_from_rotation,
    quaternion_to_rotation,
    rotation_body_from_gimbal,
    rotation_to_quaternion,
)

# base_link is ENU body: X forward, Y left, Z up.
FORWARD = np.array([1.0, 0.0, 0.0])
LEFT = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def test_nadir_points_the_lens_at_the_ground():
    """Gimbal (0, -90, 0) must aim camera_link +X along base_link -Z."""
    rotation = rotation_body_from_gimbal(0.0, -90.0, 0.0)
    np.testing.assert_allclose(rotation @ FORWARD, -UP, atol=1e-12)
    # Camera "up" ends up pointing out the nose, which is what makes the top of
    # the image the forward direction.
    np.testing.assert_allclose(rotation @ UP, FORWARD, atol=1e-12)
    # And nothing has rolled: camera left is still vehicle left.
    np.testing.assert_allclose(rotation @ LEFT, LEFT, atol=1e-12)


def test_level_gimbal_looks_out_the_nose():
    np.testing.assert_allclose(
        rotation_body_from_gimbal(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)


def test_gimbal_yaw_is_right_positive():
    """SIYI yaw +90 must swing the lens to the vehicle's RIGHT (-Y in ENU)."""
    rotation = rotation_body_from_gimbal(0.0, 0.0, 90.0)
    np.testing.assert_allclose(rotation @ FORWARD, -LEFT, atol=1e-12)


def test_gimbal_pitch_is_up_positive():
    """SIYI pitch +25 (its upper travel limit) must raise the lens."""
    rotation = rotation_body_from_gimbal(0.0, 25.0, 0.0)
    assert (rotation @ FORWARD)[2] > 0.0


def test_optical_axes_at_nadir():
    """The full base_link <- optical rotation the transform actually carries."""
    rotation = rotation_body_from_gimbal(0.0, -90.0, 0.0) @ _OPTICAL_FROM_LINK
    expected = np.array([[0.0, -1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(rotation, expected, atol=1e-12)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)


def test_marker_straight_below_maps_straight_down():
    """A marker on the optical axis at 5 m is 5 m below base_link, nowhere else."""
    rotation = rotation_body_from_gimbal(0.0, -90.0, 0.0) @ _OPTICAL_FROM_LINK
    # Optical frame: +Z is along the lens, so "5 m in front of the camera".
    np.testing.assert_allclose(rotation @ np.array([0.0, 0.0, 5.0]),
                               np.array([0.0, 0.0, -5.0]), atol=1e-12)


def test_marker_at_the_top_of_the_image_is_forward():
    """Optical -Y is the top of the image; at nadir that must read as FORWARD.

    This is the one an operator can falsify by eye in the field: walk the marker
    toward the nose and the reported X must grow.
    """
    rotation = rotation_body_from_gimbal(0.0, -90.0, 0.0) @ _OPTICAL_FROM_LINK
    offset = rotation @ np.array([0.0, -1.0, 0.0])
    np.testing.assert_allclose(offset, FORWARD, atol=1e-12)
    # And the right of the image is the vehicle's right.
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), -LEFT,
                               atol=1e-12)


def test_earth_and_body_agree_when_the_vehicle_is_level_and_heading_east():
    """With an identity vehicle rotation there is nothing to compose."""
    for reference in ('body', 'earth', 'stabilized'):
        np.testing.assert_allclose(
            camera_rotation_in_body(reference, 0.0, -90.0, 0.0, np.eye(3)),
            rotation_body_from_gimbal(0.0, -90.0, 0.0), atol=1e-12)


def test_stabilized_gimbal_stays_world_nadir_when_the_airframe_pitches():
    """The whole reason the reference matters.

    The airframe pitches 10 deg nose-down.  A stabilized gimbal still reports
    -90 because it is still level with the world, and the transform must place
    the lens along WORLD down — which means it is 10 deg off the airframe.
    """
    # Nose-down 10 deg in ENU body is a +10 deg rotation about +Y.
    rotation_map_from_body = _ry(math.radians(10.0))
    rotation = camera_rotation_in_body('stabilized', 0.0, -90.0, 0.0,
                                       rotation_map_from_body)
    lens_in_world = rotation_map_from_body @ rotation @ FORWARD
    np.testing.assert_allclose(lens_in_world, -UP, atol=1e-12)

    # Read as body-referenced instead, the same report tilts the lens with the
    # airframe — 10 deg away from down, which at 5 m is 0.88 m of marker offset.
    body = camera_rotation_in_body('body', 0.0, -90.0, 0.0, rotation_map_from_body)
    lens_in_world_if_body = rotation_map_from_body @ body @ FORWARD
    error_rad = math.acos(float(np.clip(np.dot(lens_in_world_if_body, -UP),
                                        -1.0, 1.0)))
    assert math.degrees(error_rad) == pytest.approx(10.0, abs=1e-6)


def test_stabilized_yaw_follows_the_airframe():
    """Gimbal yaw 0 means 'aligned with the mount', not 'aligned with east'."""
    heading = math.radians(30.0)
    rotation_map_from_body = _rz(heading)
    rotation = camera_rotation_in_body('stabilized', 0.0, 0.0, 0.0,
                                       rotation_map_from_body)
    # A level, yaw-zero gimbal on a level airframe is just the airframe.
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)


def test_earth_yaw_ignores_the_airframe_heading():
    """The distinguishing behaviour of 'earth': yaw 0 is a WORLD heading."""
    rotation_map_from_body = _rz(math.radians(30.0))
    rotation = camera_rotation_in_body('earth', 0.0, 0.0, 0.0,
                                       rotation_map_from_body)
    lens_in_world = rotation_map_from_body @ rotation @ FORWARD
    np.testing.assert_allclose(lens_in_world, FORWARD, atol=1e-12)


def test_unknown_reference_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError):
        camera_rotation_in_body('gravity', 0.0, -90.0, 0.0, np.eye(3))


def test_enu_yaw_round_trip():
    for degrees in (-179.0, -90.0, 0.0, 45.0, 179.0):
        assert enu_yaw_from_rotation(_rz(math.radians(degrees))) == \
            pytest.approx(math.radians(degrees))


@pytest.mark.parametrize('angles', [
    (0.0, -90.0, 0.0), (5.0, -80.0, 30.0), (-12.0, -45.0, -120.0),
    (0.0, 25.0, 135.0),
])
def test_quaternion_round_trip(angles):
    rotation = rotation_body_from_gimbal(*angles) @ _OPTICAL_FROM_LINK
    x, y, z, w = rotation_to_quaternion(rotation)
    assert x * x + y * y + z * z + w * w == pytest.approx(1.0)
    np.testing.assert_allclose(quaternion_to_rotation(x, y, z, w), rotation,
                               atol=1e-9)
