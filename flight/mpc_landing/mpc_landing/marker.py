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
class GimbalSweep:
    """Where to point the camera while searching, and when to trust what it sees.

    The scan half of `mpc_landing_node`'s SEARCH, lifted out so a second mission
    node can run the SAME sweep rather than a lookalike of it, and so the timing
    rules — which are the part that was got wrong first — can be tested without a
    gimbal or a vehicle.

    WHY SWEEP AT ALL. A camera held at nadir sees one patch of ground: at 5 m
    with the 0.18 m marker this airframe uses, a circle about 2.5 m across. If
    the marker is not in it, the mission stares at the wrong grass for a minute
    and lands. Sweeping the gimbal searches a circle roughly 2*tan(50 deg) ~ 2.4
    vehicle-heights across WITHOUT MOVING THE VEHICLE — which, low over a trailer
    with people near it, is the difference between a search and a hazard. And
    because the gimbal angle is known at the moment of the sighting, what comes
    back is a position fix rather than "it is somewhere over there"
    (`marker_enu_from_gimbal_camera` does that part).

    THE TWO TIMING RULES, BOTH LEARNED ON THE BENCH:

    * A look's dwell is counted from when the camera ARRIVES, not from when it
      was commanded. A fixed dwell pays a 45 deg step and a 135 deg swing the
      same, and the bench measured what that bought: 36% of the sweep settled,
      64% slewing, with three of fifteen looks getting a single settled sample.
    * Settled time must be CONTINUOUS. A gimbal that arrives, is knocked off by
      a gust and comes back serves the full view again rather than banking the
      two halves — the same rule, for the same reason, as the consecutive frames
      an acquisition needs.

    And the acceptance rule: a fix taken mid-slew is placed at the wrong angle,
    and off nadir that error is multiplied by the SLANT RANGE rather than the
    height. So `settled()` gates which fixes may be used at all.
    """

    #: Rings to sweep, in gimbal pitch (negative is down). Nadir first: the
    #: marker is usually under the vehicle and the cheapest look is the one the
    #: camera already points at. Nothing shallower than about -25 deg — slant
    #: range grows as 1/sin(elevation), so a shallow look sees a long way and
    #: places what it finds very badly.
    pitch_deg: tuple = (-90.0, -60.0, -40.0)
    #: Azimuth step, and how far round each ring goes. 135 deg is the A8 mini's
    #: YAW TRAVEL LIMIT, not a choice: the sector behind the tail stays blind.
    yaw_step_deg: float = 45.0
    yaw_limit_deg: float = 135.0
    #: Settled seconds per look. ~12 detector frames against the 5 consecutive
    #: that an acquisition needs.
    view_s: float = 1.0
    #: Backstop, in wall-clock: with no attitude feedback "settled" may never
    #: arrive and the sweep would stop dead at one look.
    look_max_s: float = 4.0
    #: Minimum time since the aim last MOVED before any fix is trusted.
    settle_s: float = 0.5
    #: How closely feedback must agree with the command to count as arrived.
    #: 6 deg, not 4: settled yaw error benched at 1.5 deg mean but 3.9 deg peak,
    #: and this only decides whether a fix is ACCEPTED — placement always uses
    #: the MEASURED angle, so a wide band costs nothing but a little latency.
    settled_deg: float = 6.0
    #: False holds nadir and searches nothing, i.e. the behaviour before this.
    enabled: bool = True

    plan: list = field(default_factory=list)
    i: int = 0
    scanning: bool = False
    aim_cmd: tuple | None = field(default=None)      # (yaw, pitch) deg, SIYI
    attitude: tuple | None = field(default=None)     # (yaw, pitch) deg measured
    _t_att: float = 0.0
    _t_look: float = 0.0                             # when the aim last MOVED
    _t_settled: float | None = field(default=None)   # when it ARRIVED
    attitude_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        self.plan = ([(0.0, NADIR_PITCH_DEG)] if not self.enabled
                     else sweep_plan(self.pitch_deg, self.yaw_step_deg,
                                     self.yaw_limit_deg))

    # ------------------------------------------------------------------ input
    def on_attitude(self, now: float, *, yaw_deg: float,
                    pitch_deg: float) -> None:
        self.attitude = (float(yaw_deg), float(pitch_deg))
        self._t_att = float(now)

    def attitude_fresh(self, now: float) -> bool:
        return (self.attitude is not None
                and (float(now) - self._t_att) <= self.attitude_timeout_s)

    def angles(self, now: float) -> tuple[float, float]:
        """Where the camera IS pointing, (yaw, pitch) deg. Never None.

        Feedback first, because that is the measurement; the last commanded
        angle second, because a gimbal with no telemetry is still obeying; and
        nadir last, because that is what the gimbal node holds when nobody has
        asked for anything else.
        """
        if self.attitude_fresh(now):
            return self.attitude                     # type: ignore[return-value]
        if self.aim_cmd is not None:
            return self.aim_cmd
        return 0.0, NADIR_PITCH_DEG

    # ----------------------------------------------------------------- output
    def aim(self, yaw_deg: float, pitch_deg: float, now: float) -> tuple:
        """Record a commanded angle; restart the settle timer if it MOVED.

        Used by the sweep and by marker tracking alike, so "how long has the
        camera been pointing here" means one thing everywhere.
        """
        if (self.aim_cmd is None
                or abs(yaw_deg - self.aim_cmd[0]) > self.settled_deg
                or abs(pitch_deg - self.aim_cmd[1]) > self.settled_deg):
            self._t_look = float(now)
            self._t_settled = None
        self.aim_cmd = (float(yaw_deg), float(pitch_deg))
        return self.aim_cmd

    def restart(self, now: float) -> None:
        """Begin a sweep from the first look. Called on every entry to SEARCH.

        The sweep RESTARTS rather than resuming: after a climb, or an abort and
        a second attempt, under the vehicle is again the first place worth
        looking — and without the restart the dwell timer would still be running
        from whenever the gimbal last moved, so the first look would be skipped
        before the camera ever saw it.
        """
        self.scanning = True
        self.i = 0
        self._t_look = float(now)
        self._t_settled = None

    def stop(self) -> None:
        """No longer sweeping — tracking, or done. Loosens `settled` (see there)."""
        self.scanning = False

    def look(self, now: float) -> tuple:
        """Hold the current look, advance when its dwell is up; return the aim."""
        if len(self.plan) > 1:
            if self.settled(now):
                self._t_settled = self._t_settled or float(now)
                done = (float(now) - self._t_settled) >= self.view_s
            else:
                self._t_settled = None
                done = False
            if done or (float(now) - self._t_look) >= self.look_max_s:
                self.i = (self.i + 1) % len(self.plan)
                self._t_settled = None
        yaw, pitch = self.plan[self.i]
        return self.aim(yaw, pitch, now)

    # ---------------------------------------------------------------- queries
    def settled(self, now: float) -> bool:
        """May a fix taken right now be trusted?

        Time since the aim moved, always. Plus, WHILE SCANNING, feedback that
        agrees with the command: a sweep steps 45 deg at a time and the attitude
        poll is a few Hz, so the reported angle can be a whole sector out of date
        and a fix placed with it lands where the marker never was.

        That second test is deliberately not applied while TRACKING. There the
        aim moves a few degrees a second, so command and feedback are never far
        apart but never equal either, and demanding agreement would throw away
        every fix during the approach and abort a descent on a marker the camera
        can see perfectly well.
        """
        if float(now) - self._t_look < self.settle_s:
            return False
        if not self.scanning or self.aim_cmd is None \
                or not self.attitude_fresh(now):
            return True
        yaw, pitch = self.attitude                   # type: ignore[misc]
        return (abs(yaw - self.aim_cmd[0]) <= self.settled_deg
                and abs(pitch - self.aim_cmd[1]) <= self.settled_deg)

    def duration_s(self) -> float:
        """Roughly how long one complete sweep takes, for the startup log."""
        return len(self.plan) * (self.view_s + 1.0)


@dataclass
class VelocityEstimate:
    """How fast a tracked point is moving, from its own position fixes.

    WHY A LANDING NEEDS THIS AT ALL
    -------------------------------
    A proportional controller cannot hold zero error against a target that keeps
    moving: it settles at exactly `v_target / kp` and stays there. On a trailer
    creeping at 0.3 m/s with kp = 0.8 that is 0.375 m — just outside the 0.30 m
    radius the descent is allowed to open in, so the vehicle centres, hovers,
    and never comes down. It does not look like a failure in the log; it looks
    like a descent that is about to start, forever.

    Feeding the target's own velocity forward removes that offset instead of
    fighting it, and it does so WITHOUT raising the gain — which is the whole
    point, because a higher gain buys the same steady-state accuracy by making
    every noise spike a lurch.

    A stationary marker estimates ~0 and the feed-forward vanishes, so this
    changes nothing about the fixed-pad case it is added alongside.

    THE DROPOUT RULE
    ----------------
    A difference taken across a gap is not a velocity. If the marker was lost
    and re-acquired — possibly a different marker, possibly the same one after
    the vehicle moved — the position jump divided by the gap is a large, entirely
    fictitious speed pointed in an arbitrary direction, and it would be fed
    straight into the vehicle. So a gap longer than `gap_s` resets the estimate
    to zero and starts again.
    """

    #: Low-pass time constant [s]. Marker fixes are noisy and differencing
    #: amplifies noise, so the raw difference is never used directly.
    tau_s: float = 0.5
    #: Clamp [m/s]. Also the OFF switch: 0.0 disables the feed-forward entirely.
    max_speed: float = 1.0
    #: Longer than this between fixes and the estimate is discarded, not updated.
    gap_s: float = 1.0

    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _p: np.ndarray | None = field(default=None)
    _t: float = 0.0

    def reset(self) -> None:
        self.v = np.zeros(2)
        self._p = None

    def update(self, xy, now: float) -> np.ndarray:
        """Take one position fix; return the current velocity estimate [m/s]."""
        p = np.asarray(xy, dtype=float)[:2]
        dt = float(now) - self._t
        prev, self._p, self._t = self._p, p, float(now)
        if prev is None or dt <= 0.0 or dt > self.gap_s:
            self.v = np.zeros(2)
            return self.v
        raw = (p - prev) / dt
        alpha = dt / (self.tau_s + dt) if self.tau_s > 0.0 else 1.0
        v = (1.0 - alpha) * self.v + alpha * raw
        speed = float(np.linalg.norm(v))
        if speed > self.max_speed:
            v = v * (self.max_speed / speed) if speed > 0.0 else v
        self.v = v
        return self.v


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
