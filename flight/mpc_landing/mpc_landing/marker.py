"""The vision-direct marker input, shared by every mission node in this package.

This is the part of `mpc_landing_node` that actually landed on the marker on the
aircraft, pulled out so a second mission node can fly the SAME marker input
rather than a lookalike of it. Nothing here knows about MPC, MAVROS or rclpy —
it takes numbers in and gives numbers back, so the geometry and the acceptance
rules can be tested without a vehicle (`test_marker_frame.py`,
`test_naive_landing.py`).

WHY THERE IS NO tf2 IN HERE
---------------------------
The frames exist and are published correctly (landing_tf_node, 50 Hz, 2 ms old
to an outside observer), but a detector process saturated by solvePnP cannot
drain its own /tf subscription and every lookup at capture time fails as an
extrapolation into the future. So the detector publishes in the camera's own
optical frame and the mission node converts, using the one thing the transform
was really being asked for — where the camera is pointing — from the gimbal's
nadir hold plus the heading MAVROS already puts on the pose we are differencing
against anyway.

The cost is the assumption: roll and pitch off the gimbal are ignored. A 3-axis
gimbal holds nadir to well under a degree and one degree at 5 m is 9 cm, but if
it is ever knocked off, this reads the error as marker offset and flies toward
it.

The upside is that range comes straight from solvePnP, so height is measured
above the MARKER instead of above whatever datum the EKF started at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def enu_yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Heading of an ENU body frame, radians CCW from East."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def marker_enu_from_nadir_camera(tvec, vehicle_enu, yaw_rad: float) -> np.ndarray:
    """Marker in the camera's optical frame -> marker in map ENU.

    Assumes the gimbal is holding nadir, so the only unknown is where the
    vehicle is pointing. Optical is X right, Y down the image, Z along the
    lens; at nadir the top of the image is the nose, which makes forward -y,
    left -x, and the marker -z below. Kept a plain function so the geometry can
    be tested without a running node — a sign error here is a landing that
    misses in a believable direction.
    """
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    fwd, left, up = -y, -x, -z
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([
        float(vehicle_enu[0]) + c * fwd - s * left,
        float(vehicle_enu[1]) + s * fwd + c * left,
        float(vehicle_enu[2]) + up,
    ])


@dataclass
class MarkerTracker:
    """Latest accepted marker fix, and whether it may be trusted yet.

    The acceptance rules are the ones the aircraft flew, in one place:

    * WHICH FRAME a fix is in comes off the message, never off a parameter, so
      the detector and the mission node can never be configured to disagree and
      flipping the detector back to `map` needs no change on this side.
    * A fix older than `timeout_s` is not a fix. Everything downstream asks
      `fresh()`, never `pos is not None`.
    * Committing to a descent takes `acquire_frames` CONSECUTIVE ticks with
      both a live fix and the detector's own `detected` flag asserted. One
      frame is enough for a false positive to start an irreversible descent;
      5 ticks (~0.25 s at 20 Hz) is still immediate to a human.
    """

    map_frame: str = 'map'
    timeout_s: float = 1.5
    acquire_frames: int = 5

    pos: np.ndarray | None = field(default=None)   # map ENU
    t_fix: float = 0.0
    detected: bool = False
    #: Has the detector been heard from AT ALL? Preflight asks this rather than
    #: whether a marker is currently visible — the marker is not normally in
    #: view from the pad, which is the whole reason SEARCH happens at altitude.
    seen: bool = False
    streak: int = 0

    def on_detected(self, flag: bool) -> None:
        self.detected = bool(flag)
        self.seen = True

    def on_pose(self, xyz, frame_id: str, now: float,
                vehicle_enu=None, yaw_rad: float = 0.0) -> bool:
        """Take one marker pose. Returns whether it was usable.

        `frame_id` empty or equal to `map_frame` is taken to be already in map
        ENU; anything else is the camera optical frame and is converted against
        the vehicle pose, which is why an optical fix arriving before any
        MAVROS pose is dropped rather than guessed at.
        """
        p = np.asarray(xyz, dtype=float)
        if frame_id and frame_id != self.map_frame:
            if vehicle_enu is None:
                return False
            p = marker_enu_from_nadir_camera(p, vehicle_enu, yaw_rad)
        self.pos = p
        self.t_fix = now
        return True

    def age(self, now: float) -> float:
        return float('inf') if self.pos is None else now - self.t_fix

    def fresh(self, now: float) -> bool:
        return self.pos is not None and (now - self.t_fix) < self.timeout_s

    def acquired(self, now: float) -> bool:
        """Advance the acquire streak by one tick and report if it is complete.

        Call this exactly once per control tick while searching: a live fix AND
        a currently-asserted `detected` both have to hold, so a lone spurious
        hit trips one tick and the streak resets.
        """
        if self.fresh(now) and self.detected:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= self.acquire_frames
