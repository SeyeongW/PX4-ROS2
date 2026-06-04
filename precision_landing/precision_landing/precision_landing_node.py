#!/usr/bin/env python3
"""ArUco precision landing controller (ArduPilot / MAVROS, GUIDED mode).

Python port of offboard/src/precision_landing.cpp. The node consumes the
normalised marker offset published by camera_detection's aruco_detector_node
and flies the drone down onto the marker by streaming position setpoints.

State machine
  TAKEOFF : set GUIDED, arm, then /mavros/cmd/takeoff up to flight_alt
            (skipped if auto_takeoff=false — start in IDLE, wait for a human)
  IDLE    : wait for armed + GUIDED + (marker visible OR a fresh ENU cue)
  APPROACH: fly toward the broadcast ENU marker cue (a possibly-moving target)
            at flight_alt until the down camera acquires the marker
  ALIGN   : hover at flight_alt, slide laterally until marker is centred
  DESCEND : descend through a funnel (wide up high, tight near the surface),
            converging while descending; the marker may sit on a platform of
            height platform_height, so all altitude logic uses height ABOVE the
            marker (pos.z − platform_height)
  DONE    : force-disarm onto the platform (no LAND mode — that would descend to
            the ground and ignore the platform)

Cue → vision handoff: a coordinate publisher (moving_marker_node, or any
external source) streams the marker's local-ENU position. The drone APPROACHes
that cue blind, and the instant the down camera sees the marker it hands off to
the vision servo (ALIGN/DESCEND). If vision drops out it falls back to the cue.

Topic contract (matches aruco_detector_node):
  in  /perception/aruco_offset    geometry_msgs/Point   normalised [-1, 1]
  in  /perception/aruco_detected  std_msgs/Bool
  in  /marker/position            geometry_msgs/PointStamped  ENU cue (x=E, y=N)
  in  /mavros/state               mavros_msgs/State
  in  /mavros/local_position/pose geometry_msgs/PoseStamped (BEST_EFFORT)
  out /mavros/setpoint_raw/local  mavros_msgs/PositionTarget (velocity + yaw)
  out /precision_landing/debug    std_msgs/String
"""

import math
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from std_msgs.msg import Bool, String
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandLong, CommandTOL, SetMode


class Stage(Enum):
    TAKEOFF = 0
    IDLE = 1
    APPROACH = 2
    ALIGN = 3
    DESCEND = 4
    DONE = 5


DT = 0.05  # 50 ms control period


class PrecisionLandingNode(Node):
    def __init__(self):
        super().__init__('precision_landing_node')

        # --- Parameters -----------------------------------------------------
        self.flight_alt = self.declare_parameter('flight_alt', 5.0).value       # m, hover altitude
        # DESCENT FUNNEL (cone): the acceptable horizontal error grows with
        # altitude — wide up high, narrowing to land_align_radius near the ground.
        # The drone descends WHILE converging (it does not stall until perfectly
        # centred): inside the funnel it always creeps down (>= descend_min_scale
        # of descend_rate), faster the more centred it is; outside it, it pauses
        # and re-converges. funnel_radius(alt) = max(land_align_radius,
        # descend_cone · alt). descend_cone is the half-cone slope (m error / m alt).
        self.descend_cone = self.declare_parameter('descend_cone', 0.35).value
        self.descend_min_scale = self.declare_parameter('descend_min_scale', 0.3).value
        # Land ON TOP of an object: the marker sits at platform_height above the
        # ground (0 = flat ground decal). All altitude logic (camera projection,
        # funnel, touchdown) uses the height ABOVE THE MARKER = pos.z − platform_
        # height, not the raw AGL altitude. touchdown when that clearance drops
        # below land_clearance and we are centred → force-disarm onto the platform.
        self.platform_height = self.declare_parameter('platform_height', 0.0).value
        self.land_clearance = self.declare_parameter('land_clearance', 0.15).value
        # Below this height ABOVE the marker the ~0.8 m pattern overflows the down
        # camera, so vision drops out — finish the touchdown OPEN-LOOP on the cue
        # (exact platform centre) / KF coast instead of stalling. Roughly the
        # height at which the marker fills the frame: marker_size/2 / tan(hfov/2).
        self.final_descent_h = self.declare_parameter('final_descent_h', 0.6).value
        # Landing gate: only commit to touchdown if the marker is within this
        # horizontal error (m); otherwise re-align first so we never land grossly
        # off-target. Also the floor of the descent funnel.
        self.land_align_radius = self.declare_parameter('land_align_radius', 0.20).value
        self.descend_rate = self.declare_parameter('descend_rate', 0.3).value  # m/s downward
        # Constant-velocity Kalman filter on the marker's world (E,N) position.
        # Tracks a smooth estimate from noisy detections and COASTS (predict-only)
        # through dropped frames. kf_accel_std = process noise (m/s², how much the
        # target may manoeuvre); kf_meas_std = detection noise (m).
        self.kf_accel_std = self.declare_parameter('kf_accel_std', 0.1).value
        self.kf_meas_std = self.declare_parameter('kf_meas_std', 0.15).value
        self.coast_ticks = self.declare_parameter('coast_ticks', 30).value     # max predict-only ticks (~1.5 s)
        # VELOCITY control toward the KF estimate (visual servo): v = vel_gain · e,
        # clamped to vel_max. First-order response (ė = −vel_gain·e) ⇒ exponential
        # convergence with NO overshoot — fixes the left-right oscillation that
        # position-target control produced (nested position loops + lag).
        self.vel_gain = self.declare_parameter('vel_gain', 0.8).value          # 1/s
        self.vel_max = self.declare_parameter('vel_max', 1.0).value            # m/s, precision cap
        # APPROACH uses a higher speed limit so the drone flies quickly toward the
        # platform. Once the camera acquires the marker (ALIGN), vel_max ramps back
        # down to vel_max over approach_ramp_s seconds — smooth, no abrupt jerk.
        self.approach_vel_max = self.declare_parameter('approach_vel_max', 10.0).value  # m/s
        # approach_decel_s: how many seconds before estimated arrival to start
        # decelerating. ETA = distance / approach_vel_max; when ETA < decel_s the
        # speed cap drops linearly from approach_vel_max → vel_max.
        self.approach_decel_s = self.declare_parameter('approach_decel_s', 5.0).value   # s
        self.approach_ramp_s  = self.declare_parameter('approach_ramp_s',  2.0).value   # s
        # Down-camera intrinsics for altitude-aware pixel→metre conversion.
        # cam_hfov in rad; cam_aspect = image width / height. Update if the
        # camera model changes (current: hfov 1.20, 640×480).
        self.cam_hfov = self.declare_parameter('cam_hfov', 1.20).value
        self.cam_aspect = self.declare_parameter('cam_aspect', 4.0 / 3.0).value
        # Lateral mapping tuning (camera-mount dependent). If the drone slides the
        # WRONG way during ALIGN, flip these until it converges onto the marker:
        #   lat_swap      : swap image x/y axes
        #   lat_sign_fwd  : ±1 sign of the body-forward (+X) correction
        #   lat_sign_left : ±1 sign of the body-left (+Y) correction
        self.lat_swap = self.declare_parameter('lat_swap', False).value
        self.lat_sign_fwd = self.declare_parameter('lat_sign_fwd', 1.0).value
        self.lat_sign_left = self.declare_parameter('lat_sign_left', 1.0).value
        # auto_takeoff: node sets GUIDED, arms and climbs to flight_alt itself.
        # Set false to keep the old behaviour (a human/other node flies it up).
        self.auto_takeoff = self.declare_parameter('auto_takeoff', True).value
        # APPROACH: cue-following toward a broadcast ENU marker position before
        # the camera can see it. use_cue=false keeps the pure-vision behaviour.
        # cue_timeout: how long (s) a cue stays valid after the last message, so
        # a dead publisher doesn't strand the drone chasing a stale point.
        self.use_cue = self.declare_parameter('use_cue', True).value
        self.cue_timeout = self.declare_parameter('cue_timeout', 1.0).value

        self.stage = Stage.TAKEOFF if self.auto_takeoff else Stage.IDLE

        # --- State ----------------------------------------------------------
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]                 # local ENU position
        self.offset = Point()                      # latest marker offset
        self.fresh = 0                             # marker freshness countdown
        self.req_ticks = 0
        self.yaw = 0.0                             # current heading (ENU, rad)
        self.hold_yaw = 0.0                        # heading to keep while aligning
        # Kalman state: x=[E, N, vE, vN]; P=covariance; init lazily on 1st detect
        self.kf_x = np.zeros(4)
        self.kf_P = np.eye(4)
        self.kf_init = False
        self.kf_miss = 0                           # consecutive predict-only ticks
        self.cmd_vel = [0.0, 0.0, 0.0]             # commanded ENU velocity (E, N, Up)
        self.dbg_tick = 0                          # throttle counter for alignment debug
        self.cue = None                            # latest broadcast marker [E, N]
        self.cue_stamp = None                      # rclpy time of last cue message
        self.cue_vel = [0.0, 0.0]                  # ENU cue velocity (feed-forward)
        # Effective horizontal speed cap: starts at approach_vel_max when camera
        # first sees the marker (APPROACH→ALIGN), then ramps down to vel_max over
        # approach_ramp_s. Stays at vel_max for the rest of ALIGN + DESCEND.
        self.eff_vel_max = self.get_parameter('vel_max').value

        # --- Publishers -----------------------------------------------------
        # setpoint_raw/local lets us send a velocity setpoint (mavros converts
        # ENU→NED). Position-target control oscillated; velocity is stable.
        self.raw_pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.debug_pub = self.create_publisher(String, '/precision_landing/debug', 10)

        # --- Subscribers ----------------------------------------------------
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        # mavros publishes local_position/pose with BEST_EFFORT (sensor) QoS;
        # subscribe with sensor QoS or no messages are delivered.
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb,
            qos_profile_sensor_data)
        self.create_subscription(Point, '/perception/aruco_offset', self._offset_cb, 10)
        self.create_subscription(Bool, '/perception/aruco_detected', self._detected_cb, 10)
        # Broadcast ENU marker cue (x=East, y=North) — drives APPROACH.
        self.create_subscription(PointStamped, '/marker/position', self._cue_cb, 10)

        # --- Service clients ------------------------------------------------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        # CommandLong is used for a FORCE disarm at touchdown (ArduCopter rejects
        # a normal in-air disarm; the force magic bypasses the land check).
        self.command_cli = self.create_client(CommandLong, '/mavros/cmd/command')

        self.create_timer(DT, self.tick)

    # -----------------------------------------------------------------------
    # Callbacks
    def _state_cb(self, msg):
        self.mav_state = msg

    def _pose_cb(self, msg):
        self.pos[0] = msg.pose.position.x
        self.pos[1] = msg.pose.position.y
        self.pos[2] = msg.pose.position.z
        q = msg.pose.orientation
        # yaw about Up (ENU): 0 = facing East, CCW positive
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _offset_cb(self, msg):
        self.offset = msg

    def _detected_cb(self, msg):
        if msg.data:
            self.fresh = 10  # 10 × 50 ms = 500 ms freshness window

    def _cue_cb(self, msg):
        # ENU cue: x=East, y=North (z ignored — we hold flight_alt on approach).
        now = self.get_clock().now()
        new = [msg.point.x, msg.point.y]
        # Backward-difference the cue to a velocity so APPROACH can feed-forward
        # the target's motion (a moving marker, tracked continuously) instead of
        # forever trailing it by v/kp. Reset the estimate after a stale gap.
        if self.cue is not None and self.cue_stamp is not None:
            dt = (now - self.cue_stamp).nanoseconds * 1e-9
            if 1e-3 < dt < self.cue_timeout:
                self.cue_vel = [(new[0] - self.cue[0]) / dt,
                                (new[1] - self.cue[1]) / dt]
            else:
                self.cue_vel = [0.0, 0.0]
        self.cue = new
        self.cue_stamp = now

    def _cue_ok(self):
        if not self.use_cue or self.cue is None or self.cue_stamp is None:
            return False
        age = (self.get_clock().now() - self.cue_stamp).nanoseconds * 1e-9
        return age < self.cue_timeout

    # -----------------------------------------------------------------------
    def tick(self):
        if self.fresh > 0:
            self.fresh -= 1
        marker_ok = self.fresh > 0

        if not self.mav_state.connected:
            return

        if self.stage == Stage.TAKEOFF:
            # ArduCopter GUIDED does NOT lift from a streamed position setpoint
            # while landed — it needs an explicit takeoff command. So step
            # through GUIDED -> arm -> /mavros/cmd/takeoff (each re-sent every
            # ~2 s until it takes). No position setpoints during the climb;
            # ArduCopter holds position on its own after a guided takeoff.
            self.req_ticks += 1
            send = (self.req_ticks % 40 == 1)      # ~ every 2 s (40 × 50 ms)
            if self.mav_state.mode != 'GUIDED':
                if send:
                    self._log('TAKEOFF: set GUIDED')
                    self._set_mode('GUIDED')
            elif not self.mav_state.armed:
                if send:
                    self._log('TAKEOFF: arming')
                    self._arm(True)
            elif self.pos[2] < self.flight_alt - 0.5:
                if send:
                    self._log(f'TAKEOFF: cmd takeoff to {self.flight_alt:.1f} m')
                    self._takeoff(self.flight_alt)
            else:
                self._log('-> IDLE (reached flight_alt)')
                self.stage = Stage.IDLE
            return  # no setpoints until airborne (CommandTOL handles takeoff)

        elif self.stage == Stage.IDLE:
            self.cmd_vel = [0.0, 0.0, 0.0]           # hold position (zero velocity)
            self.hold_yaw = self.yaw     # track heading; frozen once we start aligning
            if self.mav_state.armed and self.mav_state.mode == 'GUIDED':
                if marker_ok:                        # camera already on the marker
                    self._log('-> ALIGN')
                    self._kf_reset()
                    self.eff_vel_max = self.get_parameter('vel_max').value
                    self.stage = Stage.ALIGN
                elif self._cue_ok():                 # fly to the broadcast cue first
                    self._log('-> APPROACH (following cue)')
                    self.stage = Stage.APPROACH

        elif self.stage == Stage.APPROACH:
            # Blind cue-following: servo toward the broadcast ENU marker position
            # (continuously tracking a moving target) at flight_alt. Hand off to
            # vision the instant the camera acquires the marker.
            if not self.mav_state.armed:
                self._log('disarmed -> IDLE')
                self.stage = Stage.IDLE
            else:
                approach_vmax = self.get_parameter('approach_vel_max').value
                decel_s = self.get_parameter('approach_decel_s').value
                vmax = self.get_parameter('vel_max').value

                # ETA-based deceleration: full speed until decel_s seconds out,
                # then linearly ramp the speed cap down to vel_max at arrival.
                dist = math.hypot(self.cue[0] - self.pos[0],
                                  self.cue[1] - self.pos[1])
                eta = dist / max(approach_vmax, 0.1)
                if eta < decel_s:
                    blend = eta / decel_s          # 1.0 far out, 0.0 at arrival
                    eff_vmax = vmax + (approach_vmax - vmax) * blend
                else:
                    eff_vmax = approach_vmax

                if marker_ok:
                    self._log('marker acquired -> ALIGN')
                    self._kf_reset()
                    # Start ALIGN ramp from the speed we had at acquisition
                    self.eff_vel_max = eff_vmax
                    self.stage = Stage.ALIGN
                elif not self._cue_ok():
                    self._log('cue stale -> IDLE')
                    self.stage = Stage.IDLE
                else:
                    err = self._servo_to(self.cue[0], self.cue[1],
                                         self.cue_vel[0], self.cue_vel[1],
                                         vmax_override=eff_vmax)
                    self.cmd_vel[2] = self._clamp(
                        0.5 * (self.flight_alt - self.pos[2]), -0.5, 0.5)
                    self.dbg_tick += 1
                    if self.dbg_tick % 20 == 0:
                        self._log(f'APPROACH dist={dist:.1f} eta={eta:.1f}s '
                                  f'vmax={eff_vmax:.1f}')

        elif self.stage == Stage.ALIGN:
            if not self.mav_state.armed:
                self._log('disarmed -> IDLE')
                self.stage = Stage.IDLE
            else:
                # Ramp eff_vel_max down from approach speed to precision vel_max.
                vmax = self.get_parameter('vel_max').value
                ramp_s = self.get_parameter('approach_ramp_s').value
                approach_vmax = self.get_parameter('approach_vel_max').value
                if ramp_s > 0 and self.eff_vel_max > vmax:
                    ramp_per_tick = (approach_vmax - vmax) / ramp_s * DT
                    self.eff_vel_max = max(vmax, self.eff_vel_max - ramp_per_tick)
                err = self._track(marker_ok, vmax_override=self.eff_vel_max)
                self.cmd_vel[2] = 0.0                   # hold altitude
                if self.kf_miss > self.coast_ticks:
                    # Vision lost: keep chasing the cue if it's live, else give up.
                    if self._cue_ok():
                        self._log('marker lost -> APPROACH (cue)')
                        self.stage = Stage.APPROACH
                    else:
                        self._log('marker lost -> IDLE')
                        self.stage = Stage.IDLE
                elif err is not None and marker_ok and err < self._funnel_radius():
                    self._log('-> DESCEND')
                    self.stage = Stage.DESCEND

        elif self.stage == Stage.DESCEND:
            if not self.mav_state.armed:
                self._log('disarmed -> IDLE')
                self.stage = Stage.IDLE
            else:
                err = self._track(marker_ok)            # keep KF alive + KF servo
                h = self._height_above_marker()
                if h < self.final_descent_h:
                    # FINAL (near-touchdown) descent. The marker is ~0.8 m wide, so
                    # below ~final_descent_h above it the camera no longer sees the
                    # whole pattern — insisting on vision here just strands the drone
                    # hovering low. Finish OPEN-LOOP on the most reliable horizontal
                    # reference (the cue is the platform centre = marker centre),
                    # falling back to the KF coast. Never bail out: descend straight
                    # down and disarm. Only hold the descent if not yet centred, so
                    # we never touch down off the platform.
                    if self._cue_ok():
                        lateral_err = self._servo_to(self.cue[0], self.cue[1],
                                                     self.cue_vel[0], self.cue_vel[1])
                    else:
                        lateral_err = err if err is not None else 999.0
                    if lateral_err < self.land_align_radius:
                        if h < self.land_clearance:
                            self._log(f'centred ({lateral_err:.2f} m) over platform '
                                      f'-> DONE (disarm)')
                            self.stage = Stage.DONE
                        else:
                            self.cmd_vel[2] = -self.descend_rate * self.descend_min_scale
                    else:
                        self.cmd_vel[2] = 0.0        # converge first, then drop
                elif self.kf_miss > self.coast_ticks:
                    self._log('marker lost -> ALIGN (hold alt)')
                    self.cmd_vel[2] = 0.0
                    self.stage = Stage.ALIGN            # stop descending until reacquired
                else:
                    # Descent funnel: inside funnel_radius(alt) ALWAYS creep down
                    # (>= descend_min_scale of the rate), faster the more centred;
                    # outside the funnel pause and re-converge. The funnel narrows
                    # as altitude drops, so the drone converges WHILE descending.
                    # Gated on kf_init (marker world position known), not marker_ok
                    # (pattern currently visible) — it coasts through brief dropouts.
                    lateral_err = err if err is not None else 999.0
                    if self.kf_init:
                        funnel = self._funnel_radius()
                        if lateral_err <= funnel:
                            scale = self._clamp(1.0 - lateral_err / funnel,
                                                self.descend_min_scale, 1.0)
                            self.cmd_vel[2] = -self.descend_rate * scale
                        else:
                            self.cmd_vel[2] = 0.0    # too far off for this altitude
                    else:
                        self.cmd_vel[2] = 0.0

        elif self.stage == Stage.DONE:
            # Controlled touchdown: we are within land_clearance of the platform
            # top and centred. Cut the motors with a FORCE disarm (a normal in-air
            # disarm is rejected by ArduCopter) so the drone settles onto the
            # platform — we do NOT use LAND mode, which would descend to the ground
            # and ignore the platform. Re-send until /mavros/state confirms disarm.
            if self.mav_state.armed:
                self._force_disarm()
            else:
                self._log('disarmed onto platform -> shutting down node')
                rclpy.shutdown()
            return  # stop sending setpoints; motors are (being) cut

        self._publish_velocity()

    # -----------------------------------------------------------------------
    # Track the marker with a constant-velocity Kalman filter and steer the
    # setpoint toward the smoothed estimate. Returns the horizontal distance (m)
    # from the drone to the estimated marker, or None if not yet initialised.
    def _track(self, marker_ok, vmax_override=None):
        self._kf_predict()
        if marker_ok:
            zE, zN = self._measure_marker_world()
            self._kf_update(zE, zN)
            self.kf_miss = 0
        else:
            self.kf_miss += 1

        if not self.kf_init:
            self.cmd_vel[0] = self.cmd_vel[1] = 0.0
            return None

        # Velocity visual servo toward the KF position estimate, feeding forward
        # the KF velocity estimate (kf_x[2:4]) so a MOVING marker is tracked
        # without steady-state lag. v = v_marker + vel_gain·error, clamped.
        # Stationary marker ⇒ v_marker≈0 ⇒ plain first-order exponential approach.
        eE = self.kf_x[0] - self.pos[0]
        eN = self.kf_x[1] - self.pos[1]
        self._servo_to(self.kf_x[0], self.kf_x[1], self.kf_x[2], self.kf_x[3],
                       vmax_override=vmax_override)

        # Mapping diagnostic (throttled ~1 Hz). Compare where the marker is in
        # the IMAGE (off=) against the world error we steer toward (err=) and the
        # commanded velocity (cmd=). For a CORRECT mapping the velocity must
        # reduce the image offset; if it grows, the swap/sign is wrong.
        self.dbg_tick += 1
        if self.dbg_tick % 20 == 0:
            self._log(
                f'off=({self.offset.x:+.2f},{self.offset.y:+.2f}) '
                f'yaw={math.degrees(self.hold_yaw):+.0f} '
                f'err=({eE:+.2f},{eN:+.2f}) '
                f'cmd=({self.cmd_vel[0]:+.2f},{self.cmd_vel[1]:+.2f})')
        return math.hypot(eE, eN)

    # Height (m) of the drone ABOVE THE MARKER surface. The marker rides on top
    # of an object at platform_height above the ground, so this — not the raw AGL
    # pos.z — is the distance that drives the camera projection, the funnel and
    # the touchdown gate. Floored so the pixel→metre scale never blows up.
    def _height_above_marker(self):
        ph = self.get_parameter('platform_height').value
        return max(self.pos[2] - ph, 0.05)

    # Horizontal error (m) tolerated at the current height above the marker: a
    # cone, wide up high and narrowing to land_align_radius at the surface. Read
    # live so it can be tuned mid-flight (ros2 param set descend_cone / ...).
    def _funnel_radius(self):
        r_land = self.get_parameter('land_align_radius').value
        slope = self.get_parameter('descend_cone').value
        return max(r_land, slope * self._height_above_marker())

    # First-order velocity servo toward an absolute ENU point (E, N) with an
    # optional target-velocity feed-forward: v = v_ff + kp·e, clamped to vel_max.
    # The feed-forward cancels the steady-state lag (v_target/kp) when chasing a
    # MOVING target, so the drone can actually settle over it; for a stationary
    # target v_ff≈0 and it reduces to the plain proportional servo. Sets the
    # horizontal cmd_vel and returns the remaining horizontal distance (m).
    def _servo_to(self, targetE, targetN, vffE=0.0, vffN=0.0, vmax_override=None):
        kp = self.get_parameter('vel_gain').value
        vmax = vmax_override if vmax_override is not None \
            else self.get_parameter('vel_max').value
        eE = targetE - self.pos[0]
        eN = targetN - self.pos[1]
        self.cmd_vel[0] = self._clamp(vffE + kp * eE, -vmax, vmax)
        self.cmd_vel[1] = self._clamp(vffN + kp * eN, -vmax, vmax)
        return math.hypot(eE, eN)

    # Convert the current normalised image offset to the marker's world (E,N)
    # position. Downward camera is body-fixed, so the offset is a BODY error:
    # scale by altitude (pixel→metre), map image→body (tunable mount signs),
    # rotate body→world ENU by held yaw, add the drone position.
    def _measure_marker_world(self):
        swap = self.get_parameter('lat_swap').value
        sign_fwd = self.get_parameter('lat_sign_fwd').value
        sign_left = self.get_parameter('lat_sign_left').value
        hfov = self.get_parameter('cam_hfov').value
        aspect = self.get_parameter('cam_aspect').value

        ix, iy = self.offset.x, self.offset.y   # normalised: +x right, +y down
        if swap:
            ix, iy = iy, ix
        # Distance camera→marker is the height above the MARKER (on the platform
        # top), not AGL — otherwise the pixel→metre scale is wrong near touchdown.
        alt = self._height_above_marker()
        half_w = alt * math.tan(hfov * 0.5)
        half_h = half_w / aspect
        gx = ix * half_w          # metres, image-right
        gy = iy * half_h          # metres, image-down
        fwd = sign_fwd * (-gy)    # body forward (+X)
        left = sign_left * (-gx)  # body left (+Y)
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        de = c * fwd - s * left
        dn = s * fwd + c * left
        return self.pos[0] + de, self.pos[1] + dn

    # --- Kalman filter (constant-velocity, state [E, N, vE, vN]) -------------
    def _kf_reset(self):
        self.kf_init = False
        self.kf_miss = 0

    def _kf_predict(self):
        if not self.kf_init:
            return
        dt = DT
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        sa2 = self.get_parameter('kf_accel_std').value ** 2
        Q = sa2 * np.array([[dt**4 / 4, 0, dt**3 / 2, 0],
                            [0, dt**4 / 4, 0, dt**3 / 2],
                            [dt**3 / 2, 0, dt**2, 0],
                            [0, dt**3 / 2, 0, dt**2]], dtype=float)
        self.kf_x = F @ self.kf_x
        self.kf_P = F @ self.kf_P @ F.T + Q

    def _kf_update(self, zE, zN):
        r2 = self.get_parameter('kf_meas_std').value ** 2
        if not self.kf_init:
            self.kf_x = np.array([zE, zN, 0.0, 0.0])
            self.kf_P = np.diag([r2, r2, 1.0, 1.0])
            self.kf_init = True
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = r2 * np.eye(2)
        z = np.array([zE, zN])
        y = z - H @ self.kf_x
        S = H @ self.kf_P @ H.T + R
        K = self.kf_P @ H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(4) - K @ H) @ self.kf_P

    def _publish_velocity(self):
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # mavros converts ENU→NED
        # Use velocity (E,N,Up) + yaw; ignore position, acceleration, yaw_rate.
        pt.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                        PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                        PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                        PositionTarget.IGNORE_YAW_RATE)
        pt.velocity.x = self.cmd_vel[0]   # East
        pt.velocity.y = self.cmd_vel[1]   # North
        pt.velocity.z = self.cmd_vel[2]   # Up
        # Hold heading so the body-fixed camera mapping stays valid (no spin).
        pt.yaw = self.hold_yaw
        self.raw_pub.publish(pt)

    # -----------------------------------------------------------------------
    def _set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_cli.call_async(req)

    def _arm(self, value):
        req = CommandBool.Request()
        req.value = value
        self.arming_cli.call_async(req)

    def _takeoff(self, alt):
        # ArduCopter relative guided takeoff: altitude only (lat/lon 0 = current).
        req = CommandTOL.Request()
        req.altitude = float(alt)
        self.takeoff_cli.call_async(req)

    def _force_disarm(self):
        # MAV_CMD_COMPONENT_ARM_DISARM (400), param1=0 disarm, param2=21196 = the
        # force magic that bypasses ArduCopter's "still flying" disarm check.
        req = CommandLong.Request()
        req.command = 400
        req.param1 = 0.0
        req.param2 = 21196.0
        self.command_cli.call_async(req)

    def _log(self, text):
        self.get_logger().info(text)
        self.debug_pub.publish(String(data=text))

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # DONE stage may already have called rclpy.shutdown() to self-terminate
        # once landing + disarm were confirmed; only shut down if still running.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
