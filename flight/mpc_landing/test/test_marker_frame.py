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
    GimbalSweep,
    VelocityEstimate,
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


# --------------------------------------------------------- marker velocity
# The failure these exist for: a proportional descent onto a trailer creeping at
# 0.3 m/s settles at v/kp = 0.375 m and hovers there, outside the 0.30 m radius
# the sink is allowed to open in, forever. Feeding the marker's own velocity
# forward is what removes that offset — so the estimate has to be right, and
# has to be ZERO whenever it cannot be trusted.
def _drive(est, v, *, n=40, dt=0.05, start=(0.0, 0.0)):
    """Feed `n` fixes of a marker moving at constant velocity `v`."""
    p = np.array(start, dtype=float)
    t = 0.0
    for _ in range(n):
        t += dt
        p = p + np.array(v, dtype=float) * dt
        est.update(p, t)
    return est.v


def test_a_still_marker_estimates_zero():
    """The fixed-pad case must be untouched by any of this."""
    est = VelocityEstimate()
    for i in range(20):
        est.update((3.0, -1.0), i * 0.05)
    assert np.linalg.norm(est.v) < 1e-9


def test_it_converges_on_the_real_velocity():
    est = VelocityEstimate(tau_s=0.3)
    v = _drive(est, (0.0, 0.3), n=120)
    assert abs(v[0]) < 0.02 and abs(v[1] - 0.3) < 0.02


def test_the_first_fix_is_never_a_velocity():
    """One position is a position; it takes two to be a speed."""
    est = VelocityEstimate()
    assert np.allclose(est.update((5.0, 5.0), 1.0), 0.0)


def test_a_dropout_is_discarded_not_differenced():
    """Re-acquiring 4 m away after 3 s must not command 1.3 m/s at the vehicle."""
    est = VelocityEstimate(gap_s=1.0)
    _drive(est, (0.0, 0.3))
    est.update((0.0, 4.0), 100.0)                 # long gap, big jump
    assert np.allclose(est.v, 0.0)


def test_the_clamp_is_a_clamp():
    """A noise spike differenced over 50 ms is a huge, entirely fake velocity."""
    est = VelocityEstimate(tau_s=0.0, max_speed=1.0)
    est.update((0.0, 0.0), 0.0)
    est.update((0.0, 0.5), 0.05)                  # 10 m/s of nonsense
    assert np.linalg.norm(est.v) <= 1.0 + 1e-9


def test_zero_max_speed_turns_it_off():
    """The documented off switch — the plain proportional descent, unchanged."""
    est = VelocityEstimate(max_speed=0.0)
    assert np.allclose(_drive(est, (0.5, 0.5)), 0.0)


def test_reset_forgets_the_marker_as_well_as_the_velocity():
    """Otherwise the next fix differences against a stale position."""
    est = VelocityEstimate()
    _drive(est, (0.0, 0.3))
    est.reset()
    assert np.allclose(est.v, 0.0)
    assert np.allclose(est.update((99.0, 99.0), 999.0), 0.0)


# ------------------------------------------------------------- gimbal sweep
# The timing rules are the part that was got wrong first, so they are what is
# pinned here: a fix taken mid-slew is placed at the wrong angle, and off nadir
# that error is multiplied by the SLANT RANGE rather than the height.
def _arrived(s, now):
    """Report the gimbal sitting exactly where it was told to go."""
    yaw, pitch = s.aim_cmd
    s.on_attitude(now, yaw_deg=yaw, pitch_deg=pitch)


def test_the_first_look_is_always_straight_down():
    """The marker is usually under the vehicle; the cheapest look is that one."""
    s = GimbalSweep()
    s.restart(0.0)
    assert s.look(0.0) == (0.0, -90.0)


def test_disabled_is_a_nadir_hold_and_nothing_else():
    s = GimbalSweep(enabled=False)
    s.restart(0.0)
    assert s.plan == [(0.0, -90.0)]
    for t in range(20):
        assert s.look(t * 1.0) == (0.0, -90.0)


def test_nothing_is_trusted_until_the_camera_has_stopped_moving():
    s = GimbalSweep(settle_s=0.5)
    s.restart(0.0)
    s.look(0.0)
    assert not s.settled(0.1)
    _arrived(s, 0.6)
    assert s.settled(0.6)


def test_a_stale_look_does_not_advance_until_it_has_been_seen():
    """The dwell counts SETTLED time, so a long slew still gets its full look."""
    s = GimbalSweep(view_s=1.0, settle_s=0.5, look_max_s=100.0)
    s.restart(0.0)
    first = s.look(0.0)
    # Gimbal never reports arriving: the look must not advance on wall clock.
    assert s.look(3.0) == first
    # It arrives at t=3, so its second of viewing runs from there, not from 0.
    _arrived(s, 3.0)
    assert s.look(3.6) == first
    _arrived(s, 4.1)
    assert s.look(4.1) != first


def test_settled_time_must_be_continuous():
    """Knocked off by a gust and back again serves the full view again."""
    s = GimbalSweep(view_s=1.0, settle_s=0.0, look_max_s=100.0)
    s.restart(0.0)
    first = s.look(0.0)
    _arrived(s, 0.0)
    s.look(0.5)                                  # half a view banked
    s.on_attitude(0.6, yaw_deg=99.0, pitch_deg=0.0)   # blown off target
    assert s.look(0.6) == first
    _arrived(s, 0.7)
    s.look(0.7)                                  # back on target: the spell
    assert s.look(1.2) == first                  # restarts here, not at 0.0
    assert s.look(1.75) != first                 # a full second from 0.7


def test_a_gimbal_that_never_arrives_still_sweeps():
    """Without this the sweep stops dead at one look and searches nothing."""
    s = GimbalSweep(view_s=1.0, look_max_s=2.0)
    s.restart(0.0)
    first = s.look(0.0)
    assert s.look(1.9) == first
    assert s.look(2.1) != first


def test_tracking_does_not_demand_agreement():
    """Aim and feedback are never equal while following a moving marker."""
    s = GimbalSweep(settle_s=0.5, settled_deg=6.0)
    s.restart(0.0)
    s.aim(40.0, -50.0, 0.0)
    s.on_attitude(1.0, yaw_deg=25.0, pitch_deg=-50.0)   # 15 deg behind
    assert not s.settled(1.0)                            # scanning: refused
    s.stop()
    assert s.settled(1.0)                                # tracking: accepted


def test_the_settle_timer_only_restarts_on_a_real_move():
    """Tracking nudges the aim every tick; each nudge must not reset the clock."""
    s = GimbalSweep(settle_s=0.5)
    s.stop()
    s.aim(10.0, -50.0, 0.0)
    for i in range(1, 10):                     # 1 deg nudges, well inside the band
        s.aim(10.0 + i * 0.5, -50.0, i * 0.1)
    assert s.settled(0.9)
    s.aim(90.0, -50.0, 1.0)                    # a real slew
    assert not s.settled(1.2)


def test_the_measured_angle_is_what_gets_used():
    """`angles` prefers feedback; falls back to the command; then to nadir."""
    s = GimbalSweep(attitude_timeout_s=2.0)
    assert s.angles(0.0) == (0.0, -90.0)
    s.aim(30.0, -45.0, 0.0)
    assert s.angles(0.0) == (30.0, -45.0)
    s.on_attitude(0.0, yaw_deg=28.0, pitch_deg=-44.0)
    assert s.angles(0.5) == (28.0, -44.0)
    assert s.angles(9.0) == (30.0, -45.0)       # feedback went stale
