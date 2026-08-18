"""WGS84 trailer fix -> the drone's own local ENU frame.

The proven ``wang`` flight code avoids guessing PX4's local origin:

    target_local = vehicle_local + ENU(vehicle_fix -> trailer_fix)

The trailer timestamp is its measurement epoch.  Vehicle global/local data
form one coherent geodetic-to-local anchor and may come from an older epoch;
callers must still expire both independently and never replay rejected data.
"""

from __future__ import annotations

import math


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

DEFAULT_STALE_AFTER_S = 1.0
DEFAULT_MAX_INPUT_SKEW_S = 0.25
DEFAULT_MAX_DISTANCE_M = 200.0


def enu_offset(lat_ref: float, lon_ref: float,
               lat: float, lon: float) -> tuple[float, float]:
    """Return metres (east, north) from ``ref`` to ``point`` on WGS84."""
    lat_r = math.radians(float(lat_ref))
    sin_lat = math.sin(lat_r)
    q = 1.0 - WGS84_E2 * sin_lat * sin_lat
    r_normal = WGS84_A / math.sqrt(q)
    r_meridian = WGS84_A * (1.0 - WGS84_E2) / (q ** 1.5)
    north = math.radians(float(lat) - float(lat_ref)) * r_meridian
    east = (math.radians(float(lon) - float(lon_ref)) * r_normal
            * math.cos(lat_r))
    return east, north


class RelativeTarget:
    """Latest-only, fail-closed GPS/local-position fusion."""

    def __init__(self, *, stale_after: float = DEFAULT_STALE_AFTER_S,
                 max_input_skew: float = DEFAULT_MAX_INPUT_SKEW_S,
                 max_distance: float = DEFAULT_MAX_DISTANCE_M) -> None:
        self.stale_after = float(stale_after)
        self.max_input_skew = float(max_input_skew)
        self.max_distance = float(max_distance)
        limits = (self.stale_after, self.max_input_skew, self.max_distance)
        if not all(math.isfinite(v) and v > 0.0 for v in limits):
            raise ValueError(
                'freshness, skew and distance limits must be positive')

        self._t_trailer: float | None = None
        self.trailer_lat = self.trailer_lon = self.trailer_alt = 0.0
        self.trailer_has_fix = False

        self._t_vehicle_fix: float | None = None
        self.vehicle_lat = self.vehicle_lon = self.vehicle_alt = 0.0
        self.vehicle_has_fix = False

        self._t_local: float | None = None
        self.local_x = self.local_y = self.local_z = 0.0

    def on_trailer_fix(self, t: float, *, lat: float, lon: float,
                       alt: float, has_fix: bool) -> None:
        self._t_trailer = float(t)
        self.trailer_lat, self.trailer_lon = float(lat), float(lon)
        self.trailer_alt = float(alt)
        self.trailer_has_fix = bool(has_fix)

    def on_vehicle_fix(self, t: float, *, lat: float, lon: float,
                       alt: float, has_fix: bool) -> None:
        self._t_vehicle_fix = float(t)
        self.vehicle_lat, self.vehicle_lon = float(lat), float(lon)
        self.vehicle_alt = float(alt)
        self.vehicle_has_fix = bool(has_fix)

    def on_vehicle_local(self, t: float, *, x: float, y: float,
                         z: float) -> None:
        self._t_local = float(t)
        self.local_x, self.local_y, self.local_z = map(float, (x, y, z))

    def source_stamp(self) -> float | None:
        return self._t_trailer

    def _fresh(self, now: float, stamp: float | None) -> bool:
        if stamp is None or not math.isfinite(stamp):
            return False
        age = float(now) - stamp
        return 0.0 <= age <= self.stale_after

    def blocking_reason(self, now: float) -> str | None:
        if not self._fresh(now, self._t_trailer):
            return 'trailer GPS is missing or stale'
        if not self.trailer_has_fix:
            return 'trailer GPS has no accepted 3D fix'
        if not self._fresh(now, self._t_vehicle_fix):
            return 'vehicle global position is missing or stale'
        if not self.vehicle_has_fix:
            return 'vehicle GPS has no accepted 3D fix'
        if not self._fresh(now, self._t_local):
            return 'vehicle local ENU position is missing or stale'

        values = (
            self.trailer_lat, self.trailer_lon, self.trailer_alt,
            self.vehicle_lat, self.vehicle_lon, self.vehicle_alt,
            self.local_x, self.local_y, self.local_z,
        )
        if not all(math.isfinite(value) for value in values):
            return 'position contains NaN or infinity'
        if not (-90.0 <= self.trailer_lat <= 90.0
                and -90.0 <= self.vehicle_lat <= 90.0
                and -180.0 <= self.trailer_lon <= 180.0
                and -180.0 <= self.vehicle_lon <= 180.0):
            return 'latitude or longitude is out of range'

        stamps = (self._t_vehicle_fix, self._t_local)
        skew = max(stamps) - min(stamps)  # type: ignore[type-var,operator]
        if skew > self.max_input_skew:
            return (f'vehicle anchor times differ by {skew:.3f}s '
                    f'(limit {self.max_input_skew:.3f}s)')

        east, north = self.offset()
        distance = math.hypot(east, north)
        if not math.isfinite(distance):
            return 'relative target is not finite'
        if distance > self.max_distance:
            return (f'trailer is {distance:.1f}m away '
                    f'(limit {self.max_distance:.1f}m)')
        return None

    def offset(self) -> tuple[float, float]:
        return enu_offset(self.vehicle_lat, self.vehicle_lon,
                          self.trailer_lat, self.trailer_lon)

    def solve(self, now: float) -> tuple[float, float, float] | None:
        """Return target local ENU.

        The z value is diagnostic and is not used by the cue node.
        """
        if self.blocking_reason(now) is not None:
            return None
        east, north = self.offset()
        return (
            self.local_x + east,
            self.local_y + north,
            self.local_z + self.trailer_alt - self.vehicle_alt,
        )
