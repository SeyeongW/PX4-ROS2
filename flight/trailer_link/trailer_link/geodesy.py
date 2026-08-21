"""Where the trailer is, in the drone's OWN local frame — the geodesy alone.

Step 2 of the trailer-coordinate pipeline (Part C). `trailer_gps_node` produced a
lat/lon for the trailer; the mission flies velocity setpoints in MAVROS' local
ENU frame. This module is the bridge, and nothing else: no ROS, no threads, so
the arithmetic and — more importantly — the REFUSALS can be exercised on a
laptop (`test_geodesy.py`).

THE REFERENCE IS THE VEHICLE ITSELF, EVERY TICK
-----------------------------------------------
The obvious implementation projects the trailer's lat/lon through the EKF's
origin. That needs the origin, and the origin is exactly the thing that is hard
to get right: PX4's local frame starts wherever the estimator initialised, home
can be re-set in flight, and MAVROS' own idea of it depends on which plugins are
loaded. Get it wrong by a metre and every target is wrong by a metre, silently.

So the origin is never used. The vehicle's own global fix and its own local pose
are read at the same moment, which pins the two frames together:

    target_local = vehicle_local + ENU(vehicle_fix -> trailer_fix)

Two properties follow, and both matter more than the arithmetic:

  * Any constant offset between the frames cancels. The datum, the projection,
    the origin — none of them appear in a DIFFERENCE of two points.
  * The error shrinks as the vehicle closes in. Whatever the flat-earth
    approximation and the two receivers' disagreement cost at 100 m, they cost
    proportionally less at 10 m, and nothing at all at zero. The last metres are
    the marker's job anyway (see `aruco_landing_node`), so this only has to be
    good enough to put the trailer in the camera's frame.

WHAT THIS CANNOT FIX
--------------------
Two GNSS receivers on different antennas disagree by a metre or two even with a
perfect projection, and that error is common to every sample — averaging does not
remove it. The trailer's own fix is also the trailer's ANTENNA, not the centre of
its marker. Both are why the mission hands over to vision instead of trying to
land on a coordinate.

DISTANCE IS A SAFETY GATE, NOT A STATISTIC
------------------------------------------
A single corrupt lat/lon — a MAVLink field parsed as the wrong type, a receiver
briefly reporting 0/0 — reads as a perfectly well-formed target thousands of
kilometres away, and a mission that trusts it will fly at it. So a target beyond
`max_distance_m` is REFUSED rather than clamped: a clamped target still points
the vehicle in a direction chosen by garbage, whereas a refusal stops it.
"""

from __future__ import annotations

import math

#: WGS84. The ellipsoid the receivers actually report on.
WGS84_A = 6378137.0                       # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563             # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)      # first eccentricity squared

#: Inputs older than this are treated as absent rather than as fact. Deliberately
#: short: this feeds a moving vehicle's guidance, and a 5 s old position of a
#: trailer that drives at 5 m/s is 25 m of lie.
DEFAULT_STALE_AFTER_S = 3.0

#: Refuse a target farther away than this [m]. Sized for "the trailer is
#: somewhere on this field", not for cross-country flight — see the module
#: docstring on why this refuses instead of clamping.
DEFAULT_MAX_DISTANCE_M = 200.0


def enu_offset(lat_ref: float, lon_ref: float,
               lat: float, lon: float) -> tuple[float, float]:
    """(east, north) in metres from the reference lat/lon to the point.

    Flat-earth, on the WGS84 radii of curvature AT THE REFERENCE LATITUDE. Over
    the hundreds of metres this is used for, the curvature it neglects is
    millimetres — and it is applied to a DIFFERENCE between two fixes, so even
    the choice of ellipsoid mostly cancels (module docstring).

    Degrees in, metres out. ENU: +east, +north.
    """
    lat_r = math.radians(lat_ref)
    sin_lat = math.sin(lat_r)
    tmp = 1.0 - WGS84_E2 * sin_lat * sin_lat
    r_normal = WGS84_A / math.sqrt(tmp)                 # prime vertical radius
    r_meridian = WGS84_A * (1.0 - WGS84_E2) / (tmp ** 1.5)

    north = math.radians(lat - lat_ref) * r_meridian
    east = math.radians(lon - lon_ref) * r_normal * math.cos(lat_r)
    return east, north


def bearing_deg(east: float, north: float) -> float:
    """Compass bearing of an ENU offset, degrees clockwise from north.

    For the human reading the log — "the trailer is 40 m away at 210 deg" is
    something an operator can check against what they can see; a pair of signed
    metres is not.
    """
    return math.degrees(math.atan2(east, north)) % 360.0


class RelativeTarget:
    """The trailer's position in the vehicle's local ENU frame, or a reason why not.

    Fed plain numbers from three streams — the trailer's fix, the vehicle's fix,
    the vehicle's local pose — and asked one question: `solve(t)`. It answers
    with a point or with an explanation, never with a guess.
    """

    def __init__(self, *, stale_after: float = DEFAULT_STALE_AFTER_S,
                 max_distance: float = DEFAULT_MAX_DISTANCE_M,
                 max_input_skew: float = 0.0) -> None:
        self.stale_after = float(stale_after)
        self.max_distance = float(max_distance)
        # Zero preserves the hardware-proven trailer pipeline. The obstacle-map
        # route opts in because its local/global anchor needs a coherent sample.
        self.max_input_skew = float(max_input_skew)

        self._t_trailer: float | None = None
        self.trailer_lat = 0.0
        self.trailer_lon = 0.0
        self.trailer_alt = 0.0
        self.trailer_has_fix = False

        self._t_vehicle_fix: float | None = None
        self.vehicle_lat = 0.0
        self.vehicle_lon = 0.0
        self.vehicle_alt = 0.0

        self._t_local: float | None = None
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0

    # ------------------------------------------------------------------ input
    def on_trailer_fix(self, t: float, *, lat: float, lon: float,
                       alt: float, has_fix: bool) -> None:
        self._t_trailer = float(t)
        self.trailer_lat, self.trailer_lon = float(lat), float(lon)
        self.trailer_alt = float(alt)
        self.trailer_has_fix = bool(has_fix)

    def on_vehicle_fix(self, t: float, *, lat: float, lon: float,
                       alt: float) -> None:
        self._t_vehicle_fix = float(t)
        self.vehicle_lat, self.vehicle_lon = float(lat), float(lon)
        self.vehicle_alt = float(alt)

    def on_vehicle_local(self, t: float, *, x: float, y: float,
                         z: float) -> None:
        self._t_local = float(t)
        self.local_x, self.local_y, self.local_z = float(x), float(y), float(z)

    # ---------------------------------------------------------------- queries
    def _fresh(self, t: float, stamp: float | None) -> bool:
        return stamp is not None and (float(t) - stamp) <= self.stale_after

    def age(self, t: float) -> float:
        """Age of the OLDEST input [s], or inf while any of them is missing."""
        stamps = (self._t_trailer, self._t_vehicle_fix, self._t_local)
        if any(s is None for s in stamps):
            return float('inf')
        return float(t) - min(stamps)           # type: ignore[type-var]

    def sample_time(self) -> float | None:
        """Newest source time represented by a synchronized solved target."""
        stamps = (self._t_trailer, self._t_vehicle_fix, self._t_local)
        if any(s is None or not math.isfinite(s) for s in stamps):
            return None
        return max(stamps)                      # type: ignore[type-var]

    def blocking_reason(self, t: float) -> str | None:
        """Why no target can be produced right now, or None.

        Each reason names the topic to go and look at, because "no target" on
        the pad is otherwise indistinguishable from "radio unplugged", "trailer
        indoors" and "MAVROS not running" — the three things it usually is.
        """
        if not self._fresh(t, self._t_trailer):
            return ('no trailer position — nothing fresh on /trailer/fix '
                    '(is trailer_gps_node up, and does the trailer have a 3D fix?)')
        if not self.trailer_has_fix:
            return ('the trailer is publishing WITHOUT a 3D fix — its lat/lon is '
                    'not trustworthy yet')
        if not self._fresh(t, self._t_vehicle_fix):
            return ('no vehicle global position — nothing fresh on '
                    '/mavros/global_position/global')
        if not self._fresh(t, self._t_local):
            return ('no vehicle local position — nothing fresh on '
                    '/mavros/local_position/pose')
        stamps = (self._t_trailer, self._t_vehicle_fix, self._t_local)
        if (self.max_input_skew > 0.0
                and max(stamps) - min(stamps) > self.max_input_skew):
            return (f'trailer/vehicle fix and local pose are not time-aligned '
                    f'(>{self.max_input_skew:.2f} s)')
        east, north = self.offset()
        distance = math.hypot(east, north)
        if not math.isfinite(distance):
            return 'trailer position is not a finite coordinate'
        if distance > self.max_distance:
            return (f'trailer is {distance / 1000.0:.1f} km away — REFUSING it as '
                    f'a bad fix (limit {self.max_distance:.0f} m). Check that '
                    f'/trailer/fix is the trailer and not a stale or zeroed '
                    f'coordinate')
        return None

    def offset(self) -> tuple[float, float]:
        """(east, north) from the vehicle to the trailer [m], unchecked."""
        return enu_offset(self.vehicle_lat, self.vehicle_lon,
                          self.trailer_lat, self.trailer_lon)

    def solve(self, t: float) -> tuple[float, float, float] | None:
        """The trailer as a point in the vehicle's local ENU frame, or None.

        The z is the GNSS altitude difference carried into local z, and it is
        DIAGNOSTIC ONLY — two receivers' altitudes disagree by several metres,
        which is a fine thing to read in a log and a terrible thing to fly to.
        The mission cruises at its own takeoff altitude and lets the marker, then
        the autopilot's land detector, decide the height (see aruco_landing_node).
        """
        if self.blocking_reason(t) is not None:
            return None
        east, north = self.offset()
        return (self.local_x + east,
                self.local_y + north,
                self.local_z + (self.trailer_alt - self.vehicle_alt))

    def summary(self, t: float) -> str:
        """One line an operator can check against the field in front of them."""
        blocked = self.blocking_reason(t)
        east, north = self.offset()
        distance = math.hypot(east, north)
        if blocked is not None:
            return f'no target — {blocked}'
        return (f'trailer {distance:.1f} m away at {bearing_deg(east, north):.0f} '
                f'deg (E {east:+.1f} m, N {north:+.1f} m), '
                f'inputs {self.age(t):.1f} s old')
