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


#: Gimbal pitch that means "straight down", in degrees. The A8 mini's travel
#: limit, and the angle siyi_gimbal_node holds by default.
NADIR_PITCH_DEG = -90.0


def marker_enu_from_gimbal_camera(tvec, vehicle_enu, yaw_rad: float, *,
                                  gimbal_yaw_rad: float = 0.0,
                                  gimbal_pitch_rad: float = -math.pi / 2
                                  ) -> np.ndarray:
    """Marker in the camera's optical frame -> marker in map ENU, at ANY aim.

    This is the nadir conversion generalised to a camera that is looking
    somewhere else, which is what makes a gimbal SEARCH worth doing: a marker
    spotted 40 deg off to one side is a position fix, not just a sighting,
    because solvePnP gives the RANGE along that line and the gimbal gives the
    line. Fix = vehicle position + range * direction, and both halves are
    measured rather than assumed.

    The frames, in order:

    * OPTICAL, what solvePnP returns: X right, Y down the image, Z along the
      lens. Rewritten as a camera-forward triple (`fwd`, `left`, `up`) so the
      rotations below read as aircraft angles rather than as image axes.
    * CAMERA -> BODY, by the gimbal's own aim: elevate by `gimbal_pitch_rad`
      (negative is down, -90 deg is nadir) and rotate by `gimbal_yaw_rad`
      (CCW/left positive, i.e. the NEGATED SIYI yaw — SIYI counts yaw
      positive to the right).
    * BODY -> ENU, by the vehicle heading, exactly as before. The two yaw
      rotations compose into one, so the total azimuth is simply
      `yaw_rad + gimbal_yaw_rad` — the heading the camera is looking along.

    At `gimbal_pitch_rad = -pi/2, gimbal_yaw_rad = 0` this reduces term for
    term to the nadir conversion that flew, which `test_marker_frame.py` pins.

    THE ASSUMPTION IS THE SAME ONE, WIDENED. The vehicle's own roll and pitch
    are still ignored: a stabilized gimbal holds its pitch against gravity, so
    `gimbal_pitch_rad` is already an earth-referenced elevation (this is
    `landing_tf_node`'s 'stabilized' convention, and it is the one the aircraft
    is set to), while its yaw is a joint angle off the mount and therefore
    composes with the airframe heading. What the nadir version could treat as
    negligible and this one cannot is the LEVER: off nadir, an angle error is
    multiplied by the slant range, not by the height. At 5 m and 45 deg the
    range is 7 m, so one degree of gimbal error is 12 cm of position error.
    That is fine for deciding where to fly next, which is all a SEARCH fix is
    used for — the descent re-measures from directly overhead.
    """
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    # Optical -> camera-forward. At nadir this is the old (-y, -x, -z) triple
    # in a frame that has not yet been rotated: see the reduction above.
    fwd, left, up = z, -x, -y
    # Elevate. Rotating about the camera's LEFT axis by -pitch takes the lens
    # axis down for a negative pitch, which is why nadir is -90 and not +90.
    cp, sp = math.cos(-gimbal_pitch_rad), math.sin(-gimbal_pitch_rad)
    b_fwd = cp * fwd + sp * up
    b_left = left
    b_up = -sp * fwd + cp * up
    # One yaw for both rotations: vehicle heading plus where the gimbal is
    # pointing relative to the airframe.
    az = yaw_rad + gimbal_yaw_rad
    c, s = math.cos(az), math.sin(az)
    return np.array([
        float(vehicle_enu[0]) + c * b_fwd - s * b_left,
        float(vehicle_enu[1]) + s * b_fwd + c * b_left,
        float(vehicle_enu[2]) + b_up,
    ])


def marker_enu_from_nadir_camera(tvec, vehicle_enu, yaw_rad: float) -> np.ndarray:
    """Marker in the camera's optical frame -> map ENU, gimbal at nadir.

    The case the aircraft flies once it is over the marker, kept as its own
    name because that is how the descent reads. It is
    `marker_enu_from_gimbal_camera` with the gimbal where it normally is.
    """
    return marker_enu_from_gimbal_camera(tvec, vehicle_enu, yaw_rad)


def gimbal_aim_for(vehicle_enu, yaw_rad: float, target_enu, *,
                   nadir_deadzone_deg: float = 5.0) -> tuple[float, float]:
    """Where to point the gimbal to look at `target_enu`. Returns (yaw, pitch) DEG.

    The inverse of the conversion above, and it exists for the half of the job
    that comes after a sighting: having found the marker off to one side, the
    vehicle has to fly to it WITHOUT losing sight of it, or the descent aborts
    on a lost marker before it has begun. Re-aiming at the last fix each tick
    keeps it in frame, and the aim walks itself back to nadir as the vehicle
    arrives overhead — no special case for "we are there now".

    Angles come back in SIYI's convention, ready for `protocol.set_angle`: yaw
    positive to the RIGHT of the airframe, pitch negative DOWN. Yaw is wrapped
    to (-180, 180]; clamping to the gimbal's travel is left to the protocol
    layer, which owns those limits.

    NEAR STRAIGHT DOWN THERE IS NO AZIMUTH TO POINT AT. Overhead the target,
    the horizontal offset is a couple of centimetres of estimator noise and its
    bearing is whatever that noise happens to be — a gimbal asked to follow it
    yaws through large angles while the camera looks at the same patch of
    ground, which spins the image at the worst possible moment. So inside
    `nadir_deadzone_deg` of the vertical the answer is plain nadir, yaw
    airframe-aligned: the same thing the mission holds when nobody is aiming.
    """
    d_e = float(target_enu[0]) - float(vehicle_enu[0])
    d_n = float(target_enu[1]) - float(vehicle_enu[1])
    d_u = float(target_enu[2]) - float(vehicle_enu[2])
    horiz = math.hypot(d_e, d_n)
    pitch_deg = math.degrees(math.atan2(d_u, horiz))
    if pitch_deg <= -90.0 + abs(nadir_deadzone_deg):
        return 0.0, -90.0
    # Azimuth of the target CCW from East, minus where the nose points, gives
    # the aim relative to the airframe; negate for SIYI's right-positive yaw.
    rel = math.atan2(d_n, d_e) - yaw_rad
    yaw_deg = -math.degrees(math.atan2(math.sin(rel), math.cos(rel)))
    return yaw_deg, pitch_deg


def sweep_plan(pitch_deg, yaw_step_deg: float, yaw_limit_deg: float
               ) -> list[tuple[float, float]]:
    """The gimbal search pattern, as a list of (yaw, pitch) looks in degrees.

    One ring per entry in `pitch_deg`, each swept end to end within the yaw
    travel, and rings ALTERNATE DIRECTION so the camera starts the next ring
    from where it finished the last instead of slewing the full width to begin
    again. A ring at (or below) -89 deg is a single look: there is no azimuth
    to sweep at the pole, and sweeping one would spin the image for nothing.

    Angles are SIYI's, ready for `protocol.set_angle`. Kept a plain function so
    the pattern can be printed, argued about and tested on the ground rather
    than emerging from a loop at altitude.
    """
    step = abs(float(yaw_step_deg))
    limit = abs(float(yaw_limit_deg))
    plan: list[tuple[float, float]] = []
    forward = True
    for pitch in (float(p) for p in pitch_deg):
        if pitch <= -89.0:
            plan.append((0.0, pitch))
            continue
        n = int(limit // step) if step > 0 else 0
        yaws = [i * step for i in range(-n, n + 1)]
        plan.extend((y, pitch) for y in (yaws if forward
                                         else list(reversed(yaws))))
        forward = not forward
    return plan or [(0.0, NADIR_PITCH_DEG)]


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
