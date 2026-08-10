"""mission_manager_node — ONE job: sequence the landing mission and decide
WHICH target source the controller should believe.

The reason this node exists: a 1.5 m marker is unresolvable beyond ~30 m (at
90 m it is ~1 px per ArUco cell), so vision cannot fly the approach.  The
approach is flown to the target's REPORTED coordinates (`/marker/cue`), and
vision (`/marker/position`, from the KF) takes over only once the vehicle is
close enough for the camera to actually see the marker.  Measured detection
rate: 0.3% while chasing at 50-90 m, 60-80% hovering overhead.

This node is the single Offboard setpoint authority — never run it alongside
another setpoint publisher.

Subscribes
    /marker/cue          PointStamped    long-range target position (ENU)
    /marker/cue_velocity Vector3Stamped  long-range target velocity
    /marker/position     PointStamped    vision/KF target position (ENU)
    /marker/velocity     Vector3Stamped  vision/KF target velocity
    /marker/valid        Bool            vision usable (KF not over-coasting)
    /fmu/out/vehicle_local_position_v1
Publishes
    /fmu/in/trajectory_setpoint, /fmu/in/offboard_control_mode,
    /fmu/in/vehicle_command
    /mission/state       String          current phase (observability)

Phases

    IDLE      wait for a cue, then arm + Offboard
    TAKEOFF   climb to takeoff_alt over the launch point
    APPROACH  cruise to the CUE at approach_alt, matching its velocity.
              Position-led with velocity feed-forward — no descent yet.
    ACQUIRE   settle over the cue until the relative motion has died.
    DESCEND   hand over to the MPC, which does the relative-frame rendezvous
              and corridor-gated descent — still flying the cue, now with the
              marker correcting it.
    TOUCHDOWN contact detected -> disarm
    ABORT     cue lost -> climb and hold, then retry the approach.

THE CUE FLIES, THE MARKER CENTRES.  The whole mission tracks `/marker/cue`;
vision enters only as a slowly-filtered horizontal correction to it, and only
once low enough for the look angle to be steep (`_target`).  Vision is never a
target in its own right and its loss is never a failure — which is what killed
every earlier run, all of which ended `ABORT (vision stale)` while the cue they
were abandoning had been continuous the entire time.

Every phase closes on the target at a speed capped by what the GIMBAL can
slew (`_los_speed_cap`).  Pointing is not the limit — the gimbal tracks well —
but the line-of-sight rate is |v_rel|/range, so the same speed that is free at
70 m saturates the servo at 5 m and throws the marker out of frame.  The cap
scales with range, so the long chase is unaffected and the endgame slows itself
down.  Acceleration is bounded separately in every phase: `approach_a_max` for
the cruise ones, the MPC's own `a_max` in DESCEND.
"""

from __future__ import annotations

import math
import re

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool, String

from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleStatus)

from .frame import enu_to_ned
from .mpc import LandingMPC
from .predictor import predict_const_vel
from .reference import HorizonReference


def _px4_qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')
        p = self.declare_parameter
        self.dt = 0.02                       # 50 Hz setpoint stream
        self.mpc_dt = 0.1
        self.solve_every = 5                 # MPC re-plan at 10 Hz
        self.N = int(p('horizon', 20).value)
        self.takeoff_alt = float(p('takeoff_alt', 6.0).value)
        # Approach HIGH: the acquisition cone r(h) grows with altitude, and the
        # marker stays well resolved far above this (at 30 m it is still ~3.4 px
        # per ArUco cell).  At 8 m the cone was only 3.1 m wide, which is why the
        # handoff kept dropping out; 16 m gives 8.2 m with 6.7 px/cell to spare.
        self.approach_alt = float(p('approach_alt', 16.0).value)
        # A fixed handoff radius is wrong: a DOWN-LOOKING camera sees a CONE, not
        # a sphere.  The marker is fully in frame only while the horizontal
        # offset satisfies
        #     r(h) = h*tan(vfov/2) - s_m/2
        # (h = height above the deck).  At the 8 m approach altitude that is
        # 4.3 m, not 20 m — and r(h) -> 0 at h = 1.19 m, which is exactly the
        # blind zone.  So the same geometry explains both.  Derive the radius
        # from altitude instead of guessing it.
        self.tan_vfov_2 = float(p('tan_vfov_half', 0.6292).value)
        # The LARGEST marker of the ladder — this only feeds the startup report
        # of how big a cue error is still correctable at the approach altitude.
        self.marker_size = float(p('marker_size_m', 1.3).value)
        # Commit only once we are already moving WITH the target.  Proximity
        # alone is not enough: entering DESCEND with a velocity mismatch forces
        # the vehicle to accelerate hard, and acceleration is what swings the
        # line of sight off the marker.  Matching velocity first removes the
        # need to manoeuvre at all.
        self.handoff_v_tol = float(p('handoff_vel_tol_m_s', 1.0).value)
        # How close to the cue, horizontally, before committing to the descent.
        # This used to be derived from the camera cone because the handoff
        # needed the marker IN FRAME; the descent now starts on the cue alone,
        # so what it has to guarantee instead is that the marker WILL come into
        # frame on the way down.  At the align height (8 m) the visible radius
        # is 5.4 m, so 3 m leaves a comfortable margin.
        self.acquire_xy = float(p('acquire_xy_m', 3.0).value)
        # Tighter than handoff_v_tol: this is the last check before descending.
        self.settle_v_tol = float(p('settle_vel_tol_m_s', 0.5).value)
        # REMOVED with the cue-led rewrite: `abort_grace_s` and
        # `commit_height_m` both existed to decide when a stale VISION fix was
        # forgivable.  Nothing aborts on vision any more, so they would be
        # knobs wired to nothing — the most expensive kind to leave behind.
        self.deck_z = float(p('deck_z', 1.811).value)
        # --- vision as a CORRECTION to the cue, not a replacement -----------
        # The marker is trusted by LOOK ANGLE, not by altitude.  Altitude was
        # the obvious gate and it is the wrong one: a height threshold low
        # enough to guarantee a steep view also shrinks the visible radius
        # r(h) = h*tan(vfov/2) - s_m/2 below the very cue error it is supposed
        # to correct — at h=8 m that is 5.4 m, so a 6.4 m error never enters
        # frame at all and the vehicle lands on it.  Depression angle is what
        # the detection data is actually stratified by (measured: 80.6% at
        # -75..-60 deg, 47% at -60..-45, 30% at -45..-30, 1.2% below -15,
        # because the marker lies FLAT and foreshortens), and it is also what
        # decides the outward range bias.  Gating on it directly keeps the
        # steep-view requirement while letting a large offset be corrected from
        # altitude, where the cone is wide enough to see it.
        self.align_deg = float(p('vision_align_depression_deg', 60.0).value)
        # Filter time constant for the correction.  This is the ONLY thing
        # standing between a drifting cue frame and a lurch, so it is slow on
        # purpose: 1.5 s spreads a 1 m disagreement over ~0.7 m/s of target
        # motion, well inside the acceleration budget.
        self.bias_tau = float(p('bias_tau_s', 1.5).value)
        # ...and a HARD slew limit on top of it, because a time constant alone
        # does not bound anything: the filter's initial rate is err/tau, so the
        # disturbance it injects scales with however wrong the cue happens to
        # be — and that offset is a drifting EKF artefact measured anywhere
        # from 0.9 to 27.3 deg of heading, i.e. metres.  Correcting a 1.44 m
        # offset at tau=1.5 s starts at 0.96 m/s of target motion, which was
        # enough to push 2% of the MPC solves infeasible.  A flat 0.3 m/s cap
        # makes the correction cost the same no matter how large it is.
        self.bias_rate = float(p('bias_rate_max_m_s', 0.3).value)
        # Sanity clamp.  A correction larger than this is a bad fix or the
        # wrong marker, not a frame offset, and must not be chased.
        self.bias_max = float(p('bias_max_m', 5.0).value)
        # Residual below which the correction counts as converged and the
        # descent may continue.  See `_bias_settling`: descending through an
        # unconverged correction is a race the vehicle LOSES, because the
        # visible radius closes at vz*tan(vfov/2) = 0.46 m/s against a 0.3 m/s
        # correction.  Measured: a 4.3 m cue error landed 2.70 m off the true
        # marker, with the MPC reporting perfect alignment the whole way down
        # because it was aligned to the WRONG POINT.
        self.bias_ok_m = float(p('bias_converged_m', 0.3).value)
        # SAFETY INTERLOCK.  Making the descent cue-led removed a guarantee that
        # was never written down: the OLD design would not descend at all
        # without a marker, so "can I see the deck" was implicitly answered
        # before committing.  Cue-led descent has no such check, and the cue is
        # not trustworthy on its own — `trailer_cue_node` publishes gz world
        # coordinates as if they were PX4-local and that offset DRIFTS (0.9 to
        # 27.3 deg measured across runs).  Descending on it blind aims the
        # vehicle at a point that may be metres from the deck, and the deck is a
        # 5x5x1.9 m box moving at 3 m/s — hitting its edge flips the airframe.
        # So the cue may fly the vehicle DOWN TO this height and no lower;
        # below it the marker must have been seen and the correction converged.
        self.vision_gate_h = float(p('vision_gate_height_m', 6.0).value)
        # How long to wait at the gate before giving up and climbing away.
        self.gate_timeout = float(p('vision_gate_timeout_s', 20.0).value)
        # CUE-ONLY LANDING (default ON).  The vision gate above exists to stop a
        # blind descent on a cue that can be metres off (trailer_cue_node
        # publishes gz-world coords as PX4-local and that offset drifts), which
        # risks missing a 5x5 m deck and clipping its edge.  But when the goal is
        # only "get ONTO the platform", not centimetre precision, that gate turns
        # a continuous cue into a needless ABORT whenever the flat marker never
        # resolves.  With this ON the descent proceeds on the cue alone: follow
        # the cue, let vision centre it if the marker shows, and otherwise just
        # land on the cue.  The two descent safety limits still hold regardless
        # -- the MPC corridor (align-before-descend, so no diving lunge) and the
        # z-floor (never command below the deck) -- so this is a vertical settle
        # onto the cue, not a reckless plunge.  Turn OFF to restore the
        # precision-landing interlock that requires a converged marker fix below
        # vision_gate_h before it will commit past that height.
        self.cue_only_landing = bool(p('cue_only_landing', True).value)
        self._t_gate = None
        self._hold_z = None                  # altitude the hold froze us at
        # How long to press into the deck before concluding that PX4 is right
        # and there is nothing there.  Long enough for the land detector's
        # hysteresis, short enough not to fight a phantom.
        self.contact_timeout = float(p('contact_timeout_s', 4.0).value)
        self.touch_h = float(p('touchdown_height_m', 0.25).value)
        self.touch_xy = float(p('touchdown_xy_m', 0.6).value)
        # How long the geometric touchdown must HOLD before we disarm ourselves.
        # PX4's land detector never fires on a 3 m/s deck, so on a moving deck
        # WE own the decision — but a single-frame geometric trigger is what
        # turned a bad-cue guess into a mid-air motors-off before.  A dwell of
        # sustained low height AND low relative velocity is far harder to fake:
        # a vehicle that is genuinely sitting on the deck stays there, one that
        # is not drifts out of the box within a second.
        self.touch_dwell = float(p('touchdown_dwell_s', 2.0).value)
        # Touchdown contact-speed gate, SPLIT by axis (vz vs vxy) instead of one
        # combined norm.  The two failure modes are physically different and bind
        # differently: vz is a landing-gear / frame IMPACT limit (PX4 already
        # covers it and it ran with ~92% margin in Level 1), while vxy is a SKID
        # / TIP-OVER limit on the moving deck that PX4 has no concept of and that
        # sits right at its limit -- so vxy is the real constraint and a single
        # norm hides that (the same reason the control side already separates
        # v_max from vz_max).  Both default to the legacy `touchdown_vel_m_s`, so
        # existing launches are unchanged; set the per-axis params to tune them
        # apart.  PLACEHOLDERS: back-calculate touchdown_vxy_m_s from the
        # platform's skid/tip-over limit and touchdown_vz_m_s from the X500
        # landing-gear impact spec before a real flight (handoff Part D.1; Level
        # 1 used vz=0.5, vxy=0.2 as sim placeholders).  The 2 s dwell above is
        # the real filter -- a vehicle genuinely moving cannot hold the box that
        # long -- so the exact per-axis value is not safety-critical by itself.
        _touch_v = float(p('touchdown_vel_m_s', 0.4).value)
        self.touch_vz = float(p('touchdown_vz_m_s', _touch_v).value)
        self.touch_vxy = float(p('touchdown_vxy_m_s', _touch_v).value)
        self._t_touch_ok = None
        # Hard floor: never command the vehicle below the deck surface.  The
        # descent target is the deck, and both the MPC and a stale cue can
        # overshoot through it — which drove the vehicle underground, flipped
        # the gimbal up to stare at its own frame, and diverged.  Clamp every
        # setpoint z to this floor so that can never happen again.
        self.z_floor = self.deck_z + float(p('z_floor_margin_m', 0.05).value)
        # Ceiling on the closing speed.  The trailer runs at 3 m/s
        # (TRAILER_SPEED_M_S in run_gimbal.sh), so 3.5 m/s of CLOSING speed on
        # top of the uncapped velocity feed-forward still catches it at 6.5 m/s
        # over the ground while keeping the airframe quiet.
        self.cruise_v = float(p('cruise_speed_m_s', 3.5).value)
        # Line-of-sight rate budget: how fast the GIMBAL is allowed to have to
        # slew just to hold the target.  The gimbal cancels airframe attitude
        # open-loop, so the airframe's own body rate already eats most of its
        # 90 deg/s slew limit; what is left over for tracking the target is
        # small.  12 deg/s is an eighth of the limit and leaves the rest for
        # attitude compensation.  See `_los_speed_cap` for how it becomes a
        # speed limit.
        self.los_rate = math.radians(float(p('los_rate_budget_deg_s', 12.0).value))
        # Never cap so hard that we stop closing on the target entirely.
        self.v_rel_floor = float(p('v_rel_floor_m_s', 0.4).value)
        # Horizontal acceleration limit for APPROACH/ACQUIRE.  The MPC enforces
        # its own a_max in DESCEND, but the cruise phases had NO acceleration
        # bound of their own — they inherited PX4's MPC_ACC_HOR (3 m/s^2), which
        # is the lurch that starts the whole problem.  Slewing the speed cap at
        # this rate gives the cruise phases the same kind of bound the MPC has.
        # Applied to BOTH ramps in `_send_capped` (the closing speed and the
        # feed-forward), which can move together on entry, so the worst-case
        # horizontal acceleration is 2x this.  0.5 => 1.0 m/s^2 => ~5.8 deg of
        # tilt; the trailer's own 3 m/s / 50 m circle only demands 0.18 m/s^2
        # to hold station, so there is ample margin.
        self.approach_a = float(p('approach_a_max', 0.5).value)
        self._v_cmd = 0.0                    # rate-limited closing speed
        self._v_ff = np.zeros(2)             # rate-limited velocity feed-forward
        # MUST match PX4's MPC_XY_P (default 0.95).  The cap is enforced by
        # placing the position setpoint v_cap/MPC_XY_P ahead of the vehicle, so
        # PX4's own proportional law produces exactly v_cap and no more; get
        # this wrong and the cap silently leaks by the same factor.
        self.xy_p = float(p('px4_mpc_xy_p', 0.95).value)
        self.auto_start = bool(p('auto_start', True).value)

        # The MPC descent corridor must keep the marker WHOLLY in frame, which
        # is not the same as keeping its CENTRE inside the camera cone.
        # Corridor: p_z >= k_c*|p_xy|  <=>  |p_xy| <= p_z/k_c
        # Camera  : |p_xy| <= h*tan(vfov/2) - s/2   (the -s/2 is the marker's
        #           own half-width; without it half the code hangs outside the
        #           frame and ArUco decodes nothing)
        # so, holding at the height h_keep where the marker still has to be
        # readable,  k_c = h_keep / (h_keep*tan(vfov/2) - s/2).
        #
        # Setting k_c = 1/tan(vfov/2) — dropping the -s/2 — is what the previous
        # run flew, and it is exactly wide enough to lose the marker: at h=2.0 m
        # it permits 1.54 m of offset while the 0.30 m centre marker needs
        # <= 1.39 m to stay whole.  Measured: the vehicle rode that limit down
        # (1.43 m off at h=2.0 m), the centre marker fell out of frame, the last
        # 10 s were flown blind on a held bias, and it touched down 0.48 m off
        # the deck centre.  The size that belongs here is the SMALLEST marker's,
        # because that is the one still in use when the constraint binds.
        self.centre_marker = float(p('centre_marker_size_m', 0.30).value)
        self.keep_h = float(p('keep_visible_height_m', 1.0).value)
        _vis_r = self.keep_h * self.tan_vfov_2 - self.centre_marker / 2.0
        self.cone_k = float(p('cone_k',
                              self.keep_h / _vis_r if _vis_r > 1e-3
                              else 1.0 / self.tan_vfov_2).value)
        # shared limits (declare once; both MPCs reuse them)
        self.vz_max = float(p('vz_max', 0.6).value)
        self.j_max = float(p('j_max', 2.0).value)
        self.mpc = LandingMPC(dt_s=self.mpc_dt, horizon=self.N,
                              cone_k=self.cone_k,
                              # v_max is RETUNED every solve from the LOS budget
                              # (see DESCEND); this is only the initial value.
                              v_max=float(p('v_max', 3.5).value),
                              vz_max=self.vz_max,
                              # Acceleration still matters with a gimbal, just
                              # for a different reason.  On a body-fixed camera
                              # it was tilt: tan(theta) = a_h/g, and a_max=4 =>
                              # 22 deg => 4.1 m of line-of-sight slide at h=10 m
                              # (measured: 41 s above 10 deg, off-target 62% of
                              # the flight).  The gimbal cancels that tilt — but
                              # only up to its 90 deg/s slew rate, and it is the
                              # airframe's ANGULAR RATE, not its angle, that the
                              # gimbal has to match.  Lower accel = lower body
                              # rate = the gimbal keeps up.  1.0 m/s^2 is ~5.8
                              # deg of tilt, against 0.18 m/s^2 needed to hold
                              # station over the 3 m/s / 50 m circle.
                              a_max=float(p('a_max', 1.0).value),
                              j_max=self.j_max)
        self._ref = HorizonReference(lead_s=self.mpc_dt)
        self._t_solve = None
        # NOTE: routing APPROACH through its own jerk-limited MPC was tried and
        # REVERTED — see the APPROACH phase for the measured regression.

        self.state = 'IDLE'
        self.p_d = None
        self.v_d = np.zeros(3)
        self.cue = None
        self.cue_v = np.zeros(3)
        self.vis = None
        self.vis_v = np.zeros(3)
        self.vis_valid = False
        self._t_vis_ok = None
        self._bias = np.zeros(3)             # learned (vision - cue) offset
        self._bias_n = 0
        self._bias_res = 1e9                 # residual |vision - (cue + bias)|
        self._t_bias = 0.0                   # when the residual was last real
        self.vis_fresh = float(p('vision_fresh_s', 0.5).value)
        self.k = 0
        self.engaged = False
        self.armed = None                    # from VehicleStatus; None = unknown
        self.contact = False                 # from VehicleLandDetected
        self._t_touch = None                 # when TOUCHDOWN was entered
        self._solves = 0
        self._fails = 0

        self.sp_pub = self.create_publisher(TrajectorySetpoint,
                                            '/fmu/in/trajectory_setpoint', _px4_qos())
        self.ocm_pub = self.create_publisher(OffboardControlMode,
                                             '/fmu/in/offboard_control_mode', _px4_qos())
        self.cmd_pub = self.create_publisher(VehicleCommand,
                                             '/fmu/in/vehicle_command', _px4_qos())
        self.state_pub = self.create_publisher(String, '/mission/state', 10)

        self.create_subscription(PointStamped, '/marker/cue', self._on_cue, 10)
        self.create_subscription(Vector3Stamped, '/marker/cue_velocity',
                                 self._on_cue_v, 10)
        self.create_subscription(PointStamped, '/marker/position', self._on_vis, 10)
        self.create_subscription(Vector3Stamped, '/marker/velocity',
                                 self._on_vis_v, 10)
        self.create_subscription(Bool, '/marker/valid', self._on_valid, 10)
        self.create_subscription(VehicleLocalPosition,
                                 str(p('local_pos_topic',
                                       '/fmu/out/vehicle_local_position_v1').value),
                                 self._on_pos, _px4_qos())
        # The two feedback topics are RESOLVED, not guessed — see `_resolve_px4`.
        # Empty means "ask the graph"; set them to pin a name by hand.
        self._land_topic = str(p('land_topic', '').value)
        self._status_topic = str(p('status_topic', '').value)
        self._pending = [
            ('/fmu/out/vehicle_land_detected', self._land_topic,
             'px4_msgs/msg/VehicleLandDetected', VehicleLandDetected,
             self._on_land),
            ('/fmu/out/vehicle_status', self._status_topic,
             'px4_msgs/msg/VehicleStatus', VehicleStatus, self._on_status),
        ]
        self._bind_px4_feedback()
        self.create_timer(self.dt, self._tick)
        if not self.get_parameter('use_sim_time').value:
            self.get_logger().warn('use_sim_time=false — reference timing will '
                                   'drift from physics under RTF<1')
        else:
            # FROZEN-CLOCK WATCHDOG.  With use_sim_time the 50 Hz `_tick` runs
            # on the SIM clock, so if /clock never advances (the sensor bridge
            # isn't delivering Gazebo's clock, or Gazebo is paused/gone) the
            # whole mission silently sits in IDLE and "it won't take off" has
            # no visible cause.  This timer runs on the WALL clock, so it fires
            # regardless, and shouts if the sim clock has not moved.
            import time as _t
            from rclpy.clock import Clock, ClockType
            self._wall_t0 = _t.monotonic()
            self._sim_t0 = self._now()
            # A SYSTEM_TIME clock keeps firing while the sim clock is frozen.
            self.create_timer(2.0, self._clock_watchdog,
                              clock=Clock(clock_type=ClockType.SYSTEM_TIME))
        # The largest cue error that can ever be corrected is set by whichever
        # is tighter at the approach altitude: the depression gate (the marker
        # must be viewed steeply enough to be trusted) or the camera cone (it
        # must be in frame at all).  Report both — a cue error past this is
        # invisible by construction and the vehicle will land on it.
        h_app = self.approach_alt - self.deck_z
        r_cone = max(0.0, h_app * self.tan_vfov_2 - self.marker_size / 2.0)
        r_angle = h_app / math.tan(math.radians(self.align_deg))
        self.get_logger().info(
            f'mission_manager: the cue flies, the marker centres — the whole '
            f'mission tracks /marker/cue; vision corrects it whenever the '
            f'depression exceeds {self.align_deg:.0f} deg.  At the '
            f'{self.approach_alt:.0f} m approach that corrects a cue error up '
            f'to {min(r_cone, r_angle):.1f} m '
            f'(cone {r_cone:.1f} m, angle gate {r_angle:.1f} m)')
        self.get_logger().info(
            f'descent corridor: |p_xy| <= h/{self.cone_k:.2f}, which keeps the '
            f'{self.centre_marker:.2f} m centre marker whole in frame down to '
            f'h={self.keep_h:.1f} m')
        self.get_logger().info(
            'landing mode: CUE-ONLY — lands on the cue even if the marker is '
            'never seen; MPC corridor + z-floor still enforced'
            if self.cue_only_landing else
            f'landing mode: PRECISION — requires a converged marker fix below '
            f'{self.vision_gate_h:.0f} m or it holds and aborts')
        # The descent hold (`_bias_settling`) is what makes this safe, but say
        # so when the raw rates are on the wrong side of it anyway.
        if self.bias_rate < self.vz_max * self.tan_vfov_2:
            self.get_logger().info(
                f'note: the visible radius closes at '
                f'{self.vz_max * self.tan_vfov_2:.2f} m/s while the marker '
                f'correction closes at {self.bias_rate:.2f} m/s, so a large cue '
                f'error will hold the descent until it converges rather than '
                f'racing the camera down')

    # ------------------------------------------------------------- callbacks
    def _on_cue(self, m):
        self.cue = np.array([m.point.x, m.point.y, m.point.z])

    def _on_cue_v(self, m):
        self.cue_v = np.array([m.vector.x, m.vector.y, m.vector.z])

    def _on_vis(self, m):
        self.vis = np.array([m.point.x, m.point.y, m.point.z])

    def _on_vis_v(self, m):
        self.vis_v = np.array([m.vector.x, m.vector.y, m.vector.z])

    def _on_valid(self, m):
        self.vis_valid = bool(m.data)
        if self.vis_valid:
            self._t_vis_ok = self._now()

    def _on_pos(self, m):
        self.p_d = np.array([m.y, m.x, -m.z])
        self.v_d = np.array([m.vy, m.vx, -m.vz])

    def _clock_watchdog(self):
        """Wall-time check that the SIM clock is actually advancing."""
        import time as _t
        wall = _t.monotonic() - self._wall_t0
        sim = self._now() - self._sim_t0
        # After 3 s of wall time the sim clock should have moved noticeably.
        if wall > 3.0 and sim < 0.1:
            self.get_logger().error(
                f'/clock is NOT advancing ({sim:.2f}s sim in {wall:.1f}s wall) '
                f'— the mission timer is frozen, so it will never arm or take '
                f'off. The sensor bridge publishes /clock: check it is up and '
                f'seeing Gazebo (ros2 topic hz /clock). Nothing here is wrong; '
                f'the sim clock source is.')
        elif sim > 1.0 and self.p_d is None:
            self.get_logger().warn(
                'clock OK but no vehicle_local_position yet — waiting for '
                f'{self.get_parameter("local_pos_topic").value} (PX4 uXRCE '
                'agent up?)')
        elif sim > 1.0 and self._pending:
            self.get_logger().warn(
                'still waiting for ' + ', '.join(b for b, *_ in self._pending)
                + ' — no publisher of any message version yet, so the '
                  'touchdown cannot be confirmed and the vehicle will not '
                  'disarm itself')
        elif sim > 1.0 and self.p_d is not None and self.cue is None:
            self.get_logger().warn(
                'clock + position OK but no /marker/cue — trailer_cue_node not '
                'publishing; the mission arms only once it has a cue.')

    def _on_status(self, m):
        self.armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _on_land(self, m):
        """PX4's own verdict on contact — the ONLY thing allowed to authorise a
        forced disarm.  `ground_contact` is its first stage and is enough: it
        means thrust has dropped and the vehicle is not moving, i.e. something
        is holding it up other than the propellers."""
        self.contact = bool(m.ground_contact or m.maybe_landed or m.landed)

    # ---------------------------------------------------------------- helpers
    def _resolve_px4(self, base, type_name):
        """The advertised name of a PX4 topic, across message versions.

        PX4's uXRCE client appends each message's MESSAGE_VERSION to the DDS
        topic name, so ONE logical topic has different names on different
        firmware: on this tree it is `/fmu/out/vehicle_status_v4`, on an older
        one `_v1`, and `/fmu/out/vehicle_land_detected` carries no suffix at
        all because that message is unversioned.  A hardcoded guess is
        therefore correct on exactly one firmware, and it fails SILENTLY —
        subscribing to a name nobody publishes is not an error, it is just
        permanent silence.

        That is not hypothetical: the defaults here were `vehicle_status_v1`
        and `vehicle_land_detected_v1`, neither of which this PX4 publishes, so
        `self.armed` and `self.contact` never updated.  TOUCHDOWN then took its
        "no VehicleStatus" escape hatch, the force-disarm was never confirmed,
        and the vehicle sat ARMED on the moving deck and slid from 0.48 m to
        0.63 m off centre before the run ended.

        So ask the graph instead of asserting.  A candidate must have the right
        type AND a live publisher — the second test matters because OUR OWN
        subscription puts the wrong name in the graph with the right type, so
        type alone would happily confirm the mistake it is meant to catch.
        """
        for name, types in self.get_topic_names_and_types():
            if (type_name in types
                    and re.fullmatch(re.escape(base) + r'(_v\d+)?', name)
                    and self.count_publishers(name) > 0):
                return name
        return None

    def _bind_px4_feedback(self):
        """Subscribe to whatever PX4 is actually publishing, once it appears.

        PX4 usually starts after this node, so resolution has to be retried
        rather than done once at construction.  Runs on the WALL clock: with
        use_sim_time a frozen /clock would otherwise stop the retry too, and
        the resulting silence would look exactly like the bug above.
        """
        still = []
        for base, override, type_name, msg_type, cb in self._pending:
            name = override or self._resolve_px4(base, type_name)
            if name is None:
                still.append((base, override, type_name, msg_type, cb))
                continue
            self.create_subscription(msg_type, name, cb, _px4_qos())
            self.get_logger().info(f'{base} -> subscribed on {name}')
        self._pending = still
        if self._pending and not hasattr(self, '_bind_timer'):
            from rclpy.clock import Clock, ClockType
            self._bind_timer = self.create_timer(
                2.0, self._retry_bind, clock=Clock(clock_type=ClockType.SYSTEM_TIME))

    def _retry_bind(self):
        self._bind_px4_feedback()
        if not self._pending:
            self._bind_timer.cancel()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _cmd(self, command, p1=0.0, p2=0.0):
        c = VehicleCommand()
        c.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        c.command = command
        c.param1, c.param2 = float(p1), float(p2)
        c.target_system = c.target_component = 1
        c.source_system = c.source_component = 1
        c.from_external = True
        self.cmd_pub.publish(c)

    def _ocm(self):
        m = OffboardControlMode()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.position = True
        self.ocm_pub.publish(m)

    def _send(self, pos, vel=None, acc=None):
        # HARD Z-FLOOR (ENU up): never command the vehicle below the deck.  This
        # is the last line of defence against a descent overshoot or a stale cue
        # driving the vehicle underground; it applies to every phase because a
        # setpoint below the deck is never correct.  IDLE/TAKEOFF sit above the
        # floor anyway, so this only ever bites the endgame.
        pos = np.asarray(pos, float).copy()
        if pos[2] < self.z_floor:
            pos[2] = self.z_floor
        pp = enu_to_ned(pos)
        s = TrajectorySetpoint()
        s.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        s.position = [float(pp[0]), float(pp[1]), float(pp[2])]
        if vel is not None:
            v = enu_to_ned(vel)
            s.velocity = [float(v[0]), float(v[1]), float(v[2])]
        if acc is not None:
            a = enu_to_ned(acc)
            s.acceleration = [float(a[0]), float(a[1]), float(a[2])]
        s.yaw = float('nan')
        s.yawspeed = float('nan')
        self.sp_pub.publish(s)

    def _set_state(self, s, why=''):
        if s != self.state:
            self.get_logger().info(f'{self.state} -> {s}  {why}')
            self.state = s
            if s not in ('APPROACH', 'ACQUIRE'):
                # Leaving the carrot-driven phases: drop the ramped cap so a
                # later re-entry accelerates from rest instead of resuming a
                # stale speed and stepping the setpoint.
                self._v_cmd = 0.0
                self._v_ff = np.zeros(2)

    def _xy_to(self, target):
        d = self.p_d - target
        return float(math.hypot(d[0], d[1]))

    def _depression(self, target):
        """Angle below horizontal from the vehicle to `target`, in degrees.

        90 deg is straight down.  This is the axis the measured detection rate
        is stratified along, so it is the axis the marker is trusted along.
        """
        d = self.p_d - target
        return math.degrees(math.atan2(max(0.0, d[2]),
                                       max(float(math.hypot(d[0], d[1])), 1e-6)))

    def _target(self):
        """The point to fly at, and its velocity: the CUE, corrected by vision.

        The cue does the following and vision does the centring.  That split is
        not a compromise, it is what each source is actually good for: the cue
        is continuous, smooth and available from the first second, and the MPC
        tracks it well; the marker fix is intermittent, arrives only late, and
        is the only thing that knows where the deck really is.

        So vision enters as a slowly-filtered CORRECTION

            target = cue + bias,   bias -> (vision - cue)

        rather than as a replacement.  Three things fall out of that:

          * no step at handover.  The two sources disagree — `trailer_cue_node`
            publishes gz world coordinates as if they were PX4-local, and that
            offset DRIFTS (measured +27.3 down to +0.9 deg across runs) — so
            switching between them would jump the target by metres and undo the
            whole acceleration budget.  A filtered correction absorbs exactly
            that offset instead of being surprised by it.
          * losing the marker is no longer fatal.  The bias simply holds and
            the cue keeps flying, which deletes the failure that ended every
            previous run: `ABORT (vision stale 3.1 s)`.
          * the velocity feed-forward stays smooth, because it always comes
            from the cue and never from a differentiated intermittent fix.

        Only learned when the marker is being viewed steeply enough
        (`align_deg`) that foreshortening is not contaminating the fix.

        The correction is HORIZONTAL only.  "Align to the marker centre" is a
        horizontal statement, both sources already publish z = deck_z, and the
        range direction is the least trustworthy axis of a solvePnP fix — it
        carries the known 0.13 m lens-offset error and the oblique outward bias
        (0.43 m at 34 deg).  Letting that move the descent floor would buy
        nothing and could fly the vehicle into the deck.
        """
        cue = self.cue if self.cue is not None else self.p_d
        if (self.vis_valid and self.vis is not None and self.cue is not None
                and self._depression(self.vis) >= self.align_deg):
            err = self.vis[:2] - cue[:2]
            if float(np.linalg.norm(err)) <= self.bias_max:
                alpha = min(1.0, self.dt / max(self.bias_tau, 1e-3))
                step = alpha * (err - self._bias[:2])
                cap = self.bias_rate * self.dt
                n = float(np.linalg.norm(step))
                if n > cap:
                    step *= cap / n
                self._bias[:2] += step
                self._bias_n += 1
                self._bias_res = float(np.linalg.norm(err - self._bias[:2]))
                self._t_bias = self._now()
        return cue + self._bias, self.cue_v

    def _bias_settling(self):
        """True while the marker correction is still moving — hold altitude.

        Descending through an unconverged correction is a race against the
        camera: the visible radius r(h) = h*tan(vfov/2) - s_m/2 closes at
        vz*tan(vfov/2) (0.46 m/s at vz_max=0.6) while the correction closes at
        bias_rate (0.3 m/s).  The marker therefore leaves frame BEFORE the
        vehicle is over it, the correction freezes part-learned, and the
        vehicle lands on the error it had left.

        Nothing else catches this.  The MPC corridor is enforced against the
        CORRECTED target, so it reports perfect alignment while the true offset
        is metres — it cannot see an error it has not been told about.  So the
        gate has to be on the correction itself: stop descending until the fix
        and the cue agree, then carry on.  Holding is safe and self-terminating
        because r(h) is CONSTANT while the altitude is held and the offset only
        shrinks, so visibility can only improve.

        Only holds while vision is actually live: once the marker is gone there
        is nothing left to converge and the vehicle rides the held bias down,
        which is the intended behaviour below the blind zone.
        """
        return (self.vis_valid and self._bias_n > 0
                and self._now() - self._t_bias < self.vis_fresh
                and self._bias_res > self.bias_ok_m)

    def _descent_blocked(self):
        """Is the vehicle at the vision gate with no usable marker fix?

        Returns a reason string, or '' if the descent may continue.  Two ways
        to be blocked, and they are different failures:
          * the marker has never been seen -> we do not know where the deck is
            at all, and the cue alone is not enough to commit (see
            `vision_gate_h`);
          * it has been seen but the correction has not settled -> we know
            roughly, not precisely, and descending would lose the race with the
            shrinking view (see `_bias_settling`).

        Both are waived when `cue_only_landing` is set: the mission then commits
        to the cue and lands without ever requiring a marker, trading precision
        for always completing the descent (see that parameter's rationale).  The
        MPC corridor and z-floor still gate the descent, so waiving this removes
        only the "must have seen the marker" requirement, not the geometric
        safeguards.
        """
        if self.cue_only_landing:
            return ''
        if self.p_d[2] - self.deck_z > self.vision_gate_h:
            return ''
        if self._bias_n == 0:
            return 'no marker fix yet'
        if self._bias_settling():
            return 'correction still settling'
        return ''

    def _los_speed_cap(self, target):
        """Largest RELATIVE speed the gimbal can be asked to track right now.

        The gimbal has to slew at the line-of-sight rate just to stay pointed:

            omega_los = |v_rel_perp| / range

        so a relative speed the servo cannot match drags the aim off target no
        matter how good the pointing law is — which is the observed failure:
        the gimbal tracks beautifully until the vehicle manoeuvres.  Invert it
        into a speed limit,

            |v_rel| <= omega_budget * range,

        and the same rule does two jobs at once.  Far out (70 m) it allows
        24 m/s, i.e. it does not bind at all — so this does NOT reproduce the
        reverted slow-approach regression where the vehicle never caught the
        3 m/s trailer.  Close in it tightens automatically: 5 m -> 1.7 m/s,
        2 m -> 0.7 m/s.  Distance, not detection, is what makes speed
        expensive, so distance is what the limit is written against.

        Note this caps the speed RELATIVE TO THE TARGET.  The target's own
        velocity is carried separately as feed-forward, so a capped vehicle
        still rides along at the trailer's 3 m/s while closing gently.
        """
        rng = float(np.linalg.norm(self.p_d - target))
        return float(np.clip(self.los_rate * rng, self.v_rel_floor, self.cruise_v))

    def _send_capped(self, target, target_v, alt):
        """Fly to `target` at `alt`, closing no faster than the gimbal can track.

        Handing PX4 the far-away goal directly is what makes it sprint: its
        position law is v_cmd = v_ff + MPC_XY_P * (p_sp - p), and a 60 m error
        saturates that at MPC_XY_VEL_MAX (12 m/s here) with the tilt to match.
        So the goal is replaced by a CARROT placed

            v_cap / MPC_XY_P

        down the line of sight, which makes PX4's own proportional term come
        out at exactly v_cap.  The target's velocity goes in as feed-forward
        and is deliberately NOT capped — riding along with a 9 m/s trailer is
        free, it is the CLOSING motion that swings the line of sight.  Inside
        the carrot distance the real goal is used, so the vehicle settles on
        the target instead of orbiting a lead point.  Returns the cap.
        """
        v_cap = self._los_speed_cap(target)
        # Ramp the cap rather than stepping it: the carrot converts speed
        # DIRECTLY into position error, so a step in v_cap is a step in the
        # setpoint and PX4 answers it with MPC_ACC_HOR (3 m/s^2) of lurch.
        # Slewing at approach_a bounds the acceleration of the cruise phases,
        # which otherwise have no acceleration limit of their own at all.
        step = self.approach_a * self.dt
        self._v_cmd += max(-step, min(step, v_cap - self._v_cmd))
        # The feed-forward needs the SAME treatment, and this is easy to miss:
        # entering APPROACH from a hover hands PX4 the trailer's full 3 m/s as
        # a velocity setpoint in one tick, which is a step no cap on the
        # position setpoint can soften.  Ramp the vector too.
        dv = np.asarray(target_v, float)[:2] - self._v_ff
        n = float(np.linalg.norm(dv))
        if n > step:
            dv *= step / n
        self._v_ff += dv
        goal = np.array([target[0], target[1], alt])
        d = goal[:2] - self.p_d[:2]
        dist = float(np.linalg.norm(d))
        lead = self._v_cmd / max(self.xy_p, 1e-3)
        if dist > lead:
            goal[:2] = self.p_d[:2] + d / dist * lead
        self._send(goal, np.array([self._v_ff[0], self._v_ff[1], 0.0]))
        return self._v_cmd

    # ------------------------------------------------------------------ phases
    def _tick(self):
        self.k += 1
        self.state_pub.publish(String(data=self.state))
        if self.state == 'DONE':
            # Stop the Offboard heartbeat once the vehicle is down.  Keeping it
            # up while sending no setpoints is a contradiction PX4 resolves as
            # a failsafe; letting it lapse is the clean way to hand control
            # back.
            return
        self._ocm()
        if self.p_d is None:
            return

        if self.state == 'IDLE':
            if self.cue is None:
                return
            if self.auto_start and not self.engaged and self.k * self.dt >= 1.5:
                self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.engaged = True
                self._set_state('TAKEOFF', '(armed + offboard)')
            self._send(np.array([0.0, 0.0, self.takeoff_alt]))
            return

        if self.state == 'TAKEOFF':
            self._send(np.array([0.0, 0.0, self.takeoff_alt]))
            if self.p_d[2] >= self.takeoff_alt - 0.3:
                self._set_state('APPROACH', f'(alt {self.p_d[2]:.1f} m)')
            return

        if self.state == 'APPROACH':
            if self.cue is None:
                self._set_state('ABORT', '(cue lost)')
                return
            # Cruise to the cue at approach altitude, matching its velocity,
            # closing no faster than the gimbal can track (`_send_capped`).
            # An EARLIER attempt to smooth this by routing it through the
            # jerk-limited MPC was reverted: it capped the ABSOLUTE speed, so
            # the vehicle never caught the 3 m/s trailer at all (camera on
            # target 36% -> 2%, DESCEND never reached).  The cap here is on the
            # speed RELATIVE to the cue and scales with range, so the long
            # chase is untouched and only the endgame is slowed.
            tgt, tgt_v = self._target()
            v_cap = self._send_capped(tgt, tgt_v, self.approach_alt)
            d = self._xy_to(tgt)
            dv = float(np.hypot(self.v_d[0] - tgt_v[0], self.v_d[1] - tgt_v[1]))
            # Settle onto the cue before descending.  This gate NO LONGER waits
            # for the marker: the cue is what the descent will fly anyway, and
            # requiring a detection up here meant waiting for the one thing the
            # camera is worst at — a shallow, distant look at a marker lying
            # flat (measured 1.2% detection below 15 deg depression).  Descend
            # on the cue; the marker gets its say on the way down.
            if d < self.acquire_xy and dv < self.handoff_v_tol:
                self._set_state('ACQUIRE', f'(over the cue, d={d:.1f} m)')
            elif self.k % 100 == 0:
                self.get_logger().info(
                    f'[APPROACH] d={d:.1f}/{self.acquire_xy:.1f} m  dv={dv:.2f}/'
                    f'{self.handoff_v_tol:.2f} m/s  v_cap={v_cap:.1f} m/s')
            return

        if self.state == 'ACQUIRE':
            # Hold station over the cue at approach altitude until the relative
            # motion has actually died, then commit.  Short by design — it is a
            # settling check, not a search.
            tgt, tgt_v = self._target()
            self._send_capped(tgt, tgt_v, self.approach_alt)
            d = self._xy_to(tgt)
            dv = float(np.hypot(self.v_d[0] - tgt_v[0], self.v_d[1] - tgt_v[1]))
            if d > 2.0 * self.acquire_xy:
                self._set_state('APPROACH', f'(drifted to {d:.1f} m)')
            elif dv < self.settle_v_tol and d < self.acquire_xy:
                self._t_solve = None
                self._set_state('DESCEND',
                                f'(settled d={d:.1f} m, dv={dv:.2f} m/s)')
            elif self.k % 100 == 0:
                self.get_logger().info(
                    f'[ACQUIRE] d={d:.1f}/{self.acquire_xy:.1f} m  '
                    f'dv={dv:.2f}/{self.settle_v_tol:.2f} m/s')
            return

        if self.state == 'DESCEND':
            # The MPC flies the CUE all the way down, with vision folded in as
            # a correction (`_target`).  It used to fly the vision fix directly
            # and bail out with `ABORT (vision stale)` — which was every single
            # failed run — even though the thing it was aborting away from, the
            # cue, was continuous the whole time.  There is nothing to abort to
            # any more, so the only give-up condition left is losing the cue.
            #
            # NO radius-based bail-out either.  Horizontal alignment is already
            # enforced by the MPC corridor (cone_k = 1/tan(vfov/2)): the vehicle
            # simply cannot descend while off-centre.  Layering a radius test on
            # top was not just redundant, it was self-defeating — the threshold
            # 2*r(h) SHRINKS with altitude, so descending from an acquisition at
            # ~4.8 m offset guaranteed a trip near h=5 m and bounced the mission
            # back to APPROACH every time (identical failure at 1.5 and 3 m/s,
            # which is what proved it was logic and not target speed).
            if self.cue is None:
                self._set_state('ABORT', '(cue lost)')
                return
            tgt, tgt_v = self._target()
            # Hold at the vision gate rather than committing blind, and give up
            # rather than hovering there forever.
            blocked = self._descent_blocked()
            if blocked:
                if self._t_gate is None:
                    self._t_gate = self._now()
                    # Freeze the altitude HERE, and enforce it on the setpoint
                    # below.  `mpc.z_ref` alone does not hold anything: it is a
                    # soft reference traded off against the tracking cost (see
                    # mpc.solve stage 2, where the corridor is deliberately a
                    # reference and not a floor because a hard one is
                    # infeasible during the chase).  The previous run shows the
                    # difference plainly — it logged "holding" five times while
                    # sinking from 6.0 m to 1.3 m, i.e. it descended through the
                    # entire unconverged correction the hold exists to prevent.
                    self._hold_z = float(self.p_d[2])
                    self.get_logger().warn(
                        f'holding at h={self.p_d[2] - self.deck_z:.1f} m — '
                        f'{blocked}. The cue alone is not accurate enough to '
                        f'land on: descending on it blind risks putting the '
                        f'vehicle into the side of a moving 5x5 m deck.')
                elif self._now() - self._t_gate > self.gate_timeout:
                    self._set_state('ABORT', f'({blocked} for '
                                             f'{self.gate_timeout:.0f} s)')
                    return
            else:
                self._t_gate = None
                self._hold_z = None
            if self.k % 100 == 0:
                h = self.p_d[2] - self.deck_z
                dep = self._depression(tgt)
                src = ('cue' if not self._bias_n
                       else 'cue+marker' if dep >= self.align_deg
                       else 'cue+held bias')
                hold = f'  HOLDING ({blocked})' if blocked else ''
                self.get_logger().info(
                    f'[DESCEND] h={h:.1f} m  xy={self._xy_to(tgt):.2f} m  '
                    f'aim={src}  bias={np.linalg.norm(self._bias):.2f} m  '
                    f'res={self._bias_res:.2f}/{self.bias_ok_m:.2f} m'
                    f'  vis_valid={self.vis_valid}{hold}')
            if self.k % self.solve_every == 0 or self._t_solve is None:
                p_rel0 = self.p_d - tgt
                v_rel0 = self.v_d - tgt_v
                # The MPC already works in RELATIVE coordinates, so its v_max is
                # exactly the quantity the gimbal's slew budget constrains.
                # Retune it every solve instead of leaving it at the 7 m/s
                # worst case: at 5 m out that is a 69 deg/s line-of-sight rate
                # on top of whatever the airframe is doing, which saturates the
                # 90 deg/s servo and throws the marker out of frame right when
                # it matters most.
                self.mpc.v_max = self._los_speed_cap(tgt)
                # Align first, then descend.  `z_ref` is the MPC's own soft
                # floor on relative altitude, so parking it at the CURRENT
                # height converts "wait" into an ordinary tracking objective
                # rather than an extra mode — the horizontal solve keeps
                # closing on the marker the whole time it holds.
                self.mpc.z_ref = p_rel0[2] if blocked else 0.0
                P, V, A = predict_const_vel(tgt, tgt_v, self.mpc_dt, self.N)
                res = self.mpc.solve(p_rel0, v_rel0, P, V, A)
                self._ref.set_plan(p_rel0, v_rel0, res.pred_rel_pos,
                                   res.pred_rel_vel, res.pred_rel_acc,
                                   self.mpc_dt, tgt, tgt_v, np.zeros(3))
                self._t_solve = self.get_clock().now()
                self._solves += 1
                if not res.success:
                    self._fails += 1
            if self._ref.ready():
                tau = (self.get_clock().now() - self._t_solve).nanoseconds * 1e-9
                pos, vel, acc = self._ref.sample(tau)
                if self._hold_z is not None:
                    # Hold the altitude for real: floor the commanded z and
                    # refuse to command any downward motion.  Horizontal
                    # tracking is untouched, which is the point — the vehicle
                    # keeps closing on the marker while it waits, and the
                    # visible radius r(h) stays constant instead of shrinking
                    # out from under the correction.
                    pos = np.asarray(pos, float).copy()
                    vel = np.asarray(vel, float).copy()
                    acc = np.asarray(acc, float).copy()
                    pos[2] = max(float(pos[2]), self._hold_z)
                    vel[2] = max(float(vel[2]), 0.0)
                    acc[2] = max(float(acc[2]), 0.0)
                self._send(pos, vel, acc)
            p_rel = self.p_d - tgt
            if p_rel[2] <= self.touch_h and abs(self._xy_to(tgt)) < self.touch_xy:
                self._set_state('TOUCHDOWN',
                                f'(xy {self._xy_to(tgt):.2f} m, '
                                f'|v_rel| {np.linalg.norm(self.v_d - tgt_v):.2f} m/s)')
            return

        if self.state == 'TOUCHDOWN':
            # Sit on the deck and disarm — WITHOUT ever pushing below it.  The
            # old handler drove the setpoint to tgt_z-0.5 and, when PX4 refused
            # to disarm on the moving deck, looped back to DESCEND; together
            # those buried the vehicle under the deck and it diverged.  Two
            # rules now:
            #   * hold the setpoint at the deck surface (z_floor), tracking the
            #     target horizontally so the moving deck does not slide out;
            #     never command downward past it.
            #   * decide the disarm here rather than looping.  PX4's land
            #     detector never fires on a 3 m/s deck, so we own the call — but
            #     gated on a DWELL of sustained low height AND low relative
            #     speed, which a genuine touchdown holds and a bad-cue phantom
            #     (the mid-air-flip case) does not.
            if self._t_touch is None:
                self._t_touch = self._now()
            tgt, tgt_v = self._target()
            self._send(np.array([tgt[0], tgt[1], self.z_floor]),
                       np.array([tgt_v[0], tgt_v[1], 0.0]))

            # Is the touchdown geometry still holding this instant?  Contact
            # speed is checked PER AXIS -- vertical (landing-gear impact) and
            # horizontal (skid / tip-over on the moving deck) fail differently
            # and carry separate tolerances (see the touchdown params).
            p_rel = self.p_d - tgt
            v_rel_vec = self.v_d - tgt_v
            vz_rel = abs(float(v_rel_vec[2]))
            vxy_rel = float(np.hypot(v_rel_vec[0], v_rel_vec[1]))
            on_deck = (p_rel[2] <= self.touch_h
                       and self._xy_to(tgt) < self.touch_xy
                       and vz_rel < self.touch_vz
                       and vxy_rel < self.touch_vxy)
            if on_deck:
                if self._t_touch_ok is None:
                    self._t_touch_ok = self._now()
            else:
                self._t_touch_ok = None       # dwell must be CONTINUOUS

            dwell = (self._now() - self._t_touch_ok) if self._t_touch_ok else 0.0
            confirmed = self.contact or dwell >= self.touch_dwell

            # Always ask normally (PX4 takes it the moment it agrees); force
            # only once the touchdown is confirmed — by PX4 contact or by dwell.
            self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
            if confirmed:
                self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0,
                          21196.0)

            waited = self._now() - self._t_touch
            if self.armed is False:
                self._set_state('DONE', f'(disarmed, dwell {dwell:.1f} s)')
            elif self._xy_to(tgt) > 2.0 * self.touch_xy and waited > 2.0:
                # Genuinely drifted off the target (not a stale frame) — climb
                # back and re-run the approach rather than disarming off-deck.
                self._t_touch = self._t_touch_ok = None
                self._t_solve = None
                self._set_state('DESCEND', f'(drifted {self._xy_to(tgt):.1f} m '
                                           f'off the target)')
            elif self.armed is None and waited > 5.0:
                self._set_state('DONE', '(UNVERIFIED — no VehicleStatus)')
                self.get_logger().error(
                    'no VehicleStatus has EVER arrived — force disarm sent but '
                    'unconfirmable. Stopping Offboard. The topic is resolved '
                    'from the graph (see _resolve_px4), so this means PX4 is '
                    'not publishing vehicle_status at all: check the uXRCE '
                    'agent, then `ros2 topic list | grep vehicle_status`.')
            elif confirmed and waited > 3.0 and self.k % 50 == 0:
                self.get_logger().error(
                    f'still armed {waited:.1f} s after a confirmed touchdown '
                    f'(dwell {dwell:.1f} s) — force disarm not taking '
                    f'(armed={self.armed})')
            return

        if self.state == 'ABORT':
            # climb back to approach altitude over the last known cue, then retry
            base = self.cue if self.cue is not None else self.p_d
            self._send(np.array([base[0], base[1], self.approach_alt]))
            if self.p_d[2] >= self.approach_alt - 0.5:
                self._set_state('APPROACH', '(recovered)')
            return

    def destroy_node(self):
        if self._solves:
            self.get_logger().info(
                f'MPC solves {self._solves}, failures {self._fails}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
