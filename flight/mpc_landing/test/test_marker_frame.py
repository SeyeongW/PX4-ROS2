"""Pin the vision-direct marker geometry.

Every sign here is one a landing can hide: fly toward a marker that is 40 cm
the wrong way and it still looks like it is tracking, right up to touchdown.
"""

import math

import numpy as np
import pytest

# From `marker`, which is where this geometry lives and what every mission node
# imports. It used to be reached through mpc_landing_node, which meant loading
# rclpy, MAVROS and the MPC to test six lines of trigonometry.
from mpc_landing.marker import (
    enu_yaw_from_quaternion,
    gimbal_aim_for,
    marker_enu_from_gimbal_camera,
    marker_enu_from_nadir_camera,
    sweep_plan,
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


# ---------------------------------------------------------------------------
# OFF NADIR — the gimbal search. Every one of these is a sighting that used to
# be placed straight underneath the vehicle, which is the failure this geometry
# exists to remove: a marker 5 m to the east reported as 5 m below.
# ---------------------------------------------------------------------------

DOWN = math.radians(-90.0)


def test_gimbal_geometry_reduces_to_nadir():
    """The general form must BE the old one where they overlap, not merely agree."""
    tvec = [0.7, -1.1, 4.0]
    for yaw in (0.0, 1.2, -2.5):
        np.testing.assert_allclose(
            marker_enu_from_gimbal_camera(tvec, (2.0, 3.0, 9.0), yaw),
            marker_enu_from_nadir_camera(tvec, (2.0, 3.0, 9.0), yaw),
            atol=1e-12)


def test_a_marker_seen_45_deg_ahead_is_placed_ahead_not_below():
    """Vehicle at 5 m facing east, camera tilted 45 deg down, marker on the axis.

    The slant range is 5*sqrt(2), so the marker is on the ground 5 m EAST — the
    whole point of knowing the angle.
    """
    m = marker_enu_from_gimbal_camera(
        [0.0, 0.0, math.hypot(5.0, 5.0)], (0.0, 0.0, 5.0), EAST,
        gimbal_pitch_rad=math.radians(-45.0))
    np.testing.assert_allclose(m, [5.0, 0.0, 0.0], atol=1e-9)


def test_gimbal_yaw_is_relative_to_the_airframe():
    """Same look, gimbal turned 90 deg LEFT: the marker moves to the vehicle's left.

    Gimbal yaw is CCW-positive here (SIYI's right-positive value is negated by
    the caller), so +90 deg on a north-facing vehicle points west.
    """
    m = marker_enu_from_gimbal_camera(
        [0.0, 0.0, math.hypot(5.0, 5.0)], (0.0, 0.0, 5.0), NORTH,
        gimbal_pitch_rad=math.radians(-45.0),
        gimbal_yaw_rad=math.radians(90.0))
    np.testing.assert_allclose(m, [-5.0, 0.0, 0.0], atol=1e-9)


def test_aim_and_place_are_inverses():
    """Aim at a known marker, then place a fix taken at that aim: same point.

    This is the round trip the flight actually makes — SEARCH places a fix from
    an angle, DESCEND turns the fix back into an angle — so a sign error in
    either function that this test could not see would have to be present in
    both, in opposite directions.
    """
    vehicle = (12.0, -3.0, 8.0)
    heading = 0.7
    marker = np.array([15.0, 1.5, 0.5])
    yaw_deg, pitch_deg = gimbal_aim_for(vehicle, heading, marker)
    rng = float(np.linalg.norm(marker - np.array(vehicle)))
    back = marker_enu_from_gimbal_camera(
        [0.0, 0.0, rng], vehicle, heading,
        gimbal_yaw_rad=math.radians(-yaw_deg),
        gimbal_pitch_rad=math.radians(pitch_deg))
    np.testing.assert_allclose(back, marker, atol=1e-9)


def test_aim_at_a_marker_directly_below_is_nadir():
    yaw_deg, pitch_deg = gimbal_aim_for((4.0, 5.0, 6.0), 2.3, (4.0, 5.0, 0.0))
    assert pitch_deg == pytest.approx(-90.0)
    assert yaw_deg == pytest.approx(0.0, abs=1e-9)


def test_aim_yaw_is_siyi_signed_right_positive():
    """A marker off the vehicle's RIGHT gets a positive yaw, per protocol.set_angle."""
    # Facing east; the marker is to the south, which is the vehicle's right.
    yaw_deg, _pitch = gimbal_aim_for((0.0, 0.0, 5.0), EAST, (0.0, -5.0, 0.0))
    assert yaw_deg == pytest.approx(90.0)


def test_aim_yaw_is_wrapped_not_wound_up():
    """Behind the vehicle is -180/+180, never 350-odd degrees of travel."""
    for heading in (0.0, 1.0, -2.0, 3.1):
        yaw_deg, _p = gimbal_aim_for(
            (0.0, 0.0, 5.0), heading,
            (-3.0 * math.cos(heading), -3.0 * math.sin(heading), 0.0))
        assert -180.0 <= yaw_deg <= 180.0
        assert abs(abs(yaw_deg) - 180.0) < 1e-6


def test_aim_does_not_chase_noise_when_overhead():
    """Inside the deadzone the aim is nadir, not a bearing computed from 2 cm."""
    for bearing in (0.0, 1.4, -2.9):
        target = (0.05 * math.cos(bearing), 0.05 * math.sin(bearing), 0.0)
        yaw_deg, pitch_deg = gimbal_aim_for((0.0, 0.0, 5.0), 0.3, target)
        assert (yaw_deg, pitch_deg) == (0.0, -90.0)


# --------------------------------------------------------------- sweep pattern

def test_sweep_starts_at_nadir_with_a_single_look():
    plan = sweep_plan([-90.0, -60.0], 45.0, 135.0)
    assert plan[0] == (0.0, -90.0)
    assert [p for p in plan if p[1] == -90.0] == [(0.0, -90.0)]


def test_sweep_covers_the_whole_yaw_travel_in_steps():
    plan = sweep_plan([-60.0], 45.0, 135.0)
    assert [y for y, _p in plan] == [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]


def test_sweep_rings_alternate_direction():
    """The second ring starts where the first finished — no full-width slew."""
    plan = sweep_plan([-60.0, -40.0], 45.0, 90.0)
    first = [y for y, p in plan if p == -60.0]
    second = [y for y, p in plan if p == -40.0]
    assert first[-1] == second[0]
    assert second == list(reversed(first))


def test_sweep_never_leaves_the_gimbal_travel():
    plan = sweep_plan([-90.0, -70.0, -50.0, -30.0], 40.0, 135.0)
    assert all(-135.0 <= y <= 135.0 for y, _p in plan)


def test_sweep_degenerates_to_one_look_rather_than_to_nothing():
    """A zero step must not produce an empty plan; SEARCH would have nothing to do."""
    assert sweep_plan([-60.0], 0.0, 135.0) == [(0.0, -60.0)]
    assert sweep_plan([], 45.0, 135.0) == [(0.0, -90.0)]
