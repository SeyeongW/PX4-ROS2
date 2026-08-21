"""The trailer -> local-ENU conversion, exercised without a vehicle or a radio.

`geodesy.py` is ROS-free precisely so this can run anywhere. What is asserted
here is not "the arithmetic is arithmetic" but the two properties the mission
depends on:

  * a target is produced ONLY when every input is fresh and the trailer really
    has a fix — silence is the error signal, so a wrong `solve()` is a wrong
    coordinate the vehicle will fly to;
  * a coordinate that is obviously garbage is REFUSED rather than clamped.

The distances below are metres-scale on purpose: that is the regime the mission
actually flies, and an error that only shows up at 1000 km would never be the
one that hurts.
"""

import math

from trailer_link.geodesy import RelativeTarget, bearing_deg, enu_offset

#: Somewhere with a non-trivial cos(lat), so a latitude term dropped from the
#: longitude scaling shows up instead of cancelling.
LAT, LON = 37.4491, 126.6510          # Incheon-ish


def _ready(t=0.0, **kw):
    """A RelativeTarget with all three inputs fresh and the trailer 0/0 offset."""
    r = RelativeTarget(**kw)
    r.on_vehicle_fix(t, lat=LAT, lon=LON, alt=30.0)
    r.on_vehicle_local(t, x=10.0, y=-5.0, z=8.0)
    r.on_trailer_fix(t, lat=LAT, lon=LON, alt=28.0, has_fix=True)
    return r


# ------------------------------------------------------------------- geodesy
def test_north_and_east_go_the_right_way():
    """A sign error here flies the vehicle away from the trailer, twice as far."""
    east, north = enu_offset(LAT, LON, LAT + 0.001, LON)
    assert north > 0 and abs(east) < 0.01

    east, north = enu_offset(LAT, LON, LAT, LON + 0.001)
    assert east > 0 and abs(north) < 0.01


def test_a_metre_is_a_metre():
    """~111 km per degree of latitude, and less per degree of longitude."""
    _, north = enu_offset(LAT, LON, LAT + 0.001, LON)
    assert 110.0 < north < 112.0

    east, _ = enu_offset(LAT, LON, LAT, LON + 0.001)
    assert 88.0 < east < 89.0             # x cos(37.45 deg)


def test_the_offset_is_antisymmetric():
    """Vehicle->trailer and trailer->vehicle must agree, or closing loops drift."""
    a = enu_offset(LAT, LON, LAT + 0.0004, LON - 0.0007)
    b = enu_offset(LAT + 0.0004, LON - 0.0007, LAT, LON)
    assert math.isclose(a[0], -b[0], abs_tol=0.02)
    assert math.isclose(a[1], -b[1], abs_tol=0.02)


def test_bearing_is_compass_not_maths():
    assert math.isclose(bearing_deg(0.0, 10.0), 0.0, abs_tol=1e-6)     # north
    assert math.isclose(bearing_deg(10.0, 0.0), 90.0, abs_tol=1e-6)    # east
    assert math.isclose(bearing_deg(0.0, -10.0), 180.0, abs_tol=1e-6)  # south


# ---------------------------------------------------------------- the target
def test_the_target_lands_in_the_vehicles_own_frame():
    """The vehicle's local position is the datum; the offset rides on top of it."""
    r = _ready()
    r.on_trailer_fix(0.0, lat=LAT + 0.0009, lon=LON, alt=28.0, has_fix=True)

    x, y, z = r.solve(0.0)
    assert math.isclose(x, 10.0, abs_tol=0.05)          # no east offset
    assert 99.0 < y - (-5.0) < 101.0                    # ~100 m north of it
    assert math.isclose(z, 8.0 + (28.0 - 30.0), abs_tol=1e-6)


def test_the_frame_offset_cancels():
    """Moving the local frame's origin must not move the trailer relative to us.

    This is the reason the vehicle's own fix is the reference: whatever the EKF
    origin is, it appears in both terms and drops out of the difference.
    """
    r = _ready()
    r.on_trailer_fix(0.0, lat=LAT + 0.0002, lon=LON + 0.0002, alt=28.0,
                     has_fix=True)
    near = r.solve(0.0)

    shifted = _ready()
    shifted.on_vehicle_local(0.0, x=10.0 + 500.0, y=-5.0 - 300.0, z=8.0)
    shifted.on_trailer_fix(0.0, lat=LAT + 0.0002, lon=LON + 0.0002, alt=28.0,
                           has_fix=True)
    far = shifted.solve(0.0)

    assert math.isclose(far[0] - 500.0, near[0], abs_tol=1e-6)
    assert math.isclose(far[1] + 300.0, near[1], abs_tol=1e-6)


def test_every_missing_input_blocks_and_says_which():
    r = RelativeTarget()
    assert 'trailer' in r.blocking_reason(0.0)

    r.on_trailer_fix(0.0, lat=LAT, lon=LON, alt=28.0, has_fix=True)
    assert 'global_position' in r.blocking_reason(0.0)

    r.on_vehicle_fix(0.0, lat=LAT, lon=LON, alt=30.0)
    assert 'local_position' in r.blocking_reason(0.0)

    r.on_vehicle_local(0.0, x=0.0, y=0.0, z=0.0)
    assert r.blocking_reason(0.0) is None


def test_a_trailer_without_a_fix_is_not_a_position():
    """lat/lon fields exist even at fix_type 0; they just do not mean anything."""
    r = _ready()
    r.on_trailer_fix(0.0, lat=0.0, lon=0.0, alt=0.0, has_fix=False)
    assert 'WITHOUT a 3D fix' in r.blocking_reason(0.0)
    assert r.solve(0.0) is None


def test_stale_inputs_stop_being_facts():
    """A trailer that drives at 5 m/s is 25 m of lie after five seconds."""
    r = _ready(stale_after=3.0)
    assert r.solve(0.0) is not None
    assert r.solve(2.9) is not None
    assert r.solve(5.0) is None
    assert 'nothing fresh on /trailer/fix' in r.blocking_reason(5.0)


def test_a_zeroed_fix_is_refused_not_clamped():
    """The 0/0 failure — a receiver that reports the Gulf of Guinea.

    Clamping would keep a target pointing in a direction chosen by garbage, so
    the answer has to be no target at all.
    """
    r = _ready(max_distance=200.0)
    r.on_trailer_fix(0.0, lat=0.0, lon=0.0, alt=0.0, has_fix=True)

    why = r.blocking_reason(0.0)
    assert why is not None and 'REFUSING' in why
    assert r.solve(0.0) is None


def test_the_limit_is_the_limit():
    r = _ready(max_distance=100.0)
    r.on_trailer_fix(0.0, lat=LAT + 0.0008, lon=LON, alt=28.0, has_fix=True)
    assert r.solve(0.0) is not None                      # ~89 m: fine

    r.on_trailer_fix(0.0, lat=LAT + 0.0011, lon=LON, alt=28.0, has_fix=True)
    assert r.solve(0.0) is None                          # ~122 m: refused


def test_the_summary_says_something_an_operator_can_check():
    r = _ready()
    r.on_trailer_fix(0.0, lat=LAT, lon=LON + 0.0005, alt=28.0, has_fix=True)
    line = r.summary(0.0)
    assert '44' in line and '90 deg' in line             # ~44 m due east
