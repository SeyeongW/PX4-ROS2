#!/usr/bin/env python3
"""Follow-target → vision precision-landing controller (PX4 / MAVROS).

The requested mission flow, end to end:

  1. OFFBOARD  : prime setpoints, switch to OFFBOARD (via mavros), arm.
  2. climb     : rise to flight_alt (default 5 m) under our own velocity cmd.
  3. FOLLOW    : the node COMMANDS PX4's AUTO.FOLLOW_TARGET mode through mavros
                 (/mavros/set_mode) -- that mode change is the "ROS command"
                 that drives the follow. PX4 then flies the follow itself,
                 tracking the MAVLink FOLLOW_TARGET (#144) the phone/laptop GCS
                 streams (its own GPS). We stream NOTHING once following: PX4
                 owns the vehicle in AUTO.FOLLOW_TARGET.
  4. ALIGN/DESCEND : the instant the down camera acquires the ArUco marker we
                 re-take the vehicle (OFFBOARD again, via mavros) and run the
                 proven Kalman-filter visual servo down onto the (possibly
                 MOVING) marker -- reused from precision_landing_node.
  5. DONE      : force-disarm onto the target, then shut the node down.

Division of labour: the node issues the *mode changes* over mavros (OFFBOARD for
takeoff/vision, AUTO.FOLLOW_TARGET for the follow) and runs the vision landing;
PX4 flies the follow cruise; the phone/laptop GCS supplies the follow target
position as MAVLink FOLLOW_TARGET (#144, NOT GLOBAL_POSITION_INT).

State machine
  TAKEOFF : OFFBOARD prime -> OFFBOARD -> arm -> climb to flight_alt.
  FOLLOW  : command AUTO.FOLLOW_TARGET (mavros), then hold/stream-nothing while
            PX4 follows; watch for the marker.
  ALIGN   : re-take OFFBOARD (streaming primes it), KF visual servo to centre.
  DESCEND : funnel descent onto the marker; open-loop (cue/KF coast) near touch.
  DONE    : force-disarm onto the target -> shut down.

Topic contract
  in  /perception/aruco_offset    geometry_msgs/Point   normalised [-1, 1]
  in  /perception/aruco_detected  std_msgs/Bool
  in  /marker/position            geometry_msgs/PointStamped  ENU cue (optional)
  in  /marker/velocity            geometry_msgs/Vector3Stamped (optional feed-fwd)
  in  /mavros/state               mavros_msgs/State
  in  /mavros/local_position/pose geometry_msgs/PoseStamped (BEST_EFFORT)
  out /mavros/setpoint_raw/local  mavros_msgs/PositionTarget (velocity + yaw)
  out /follow_precland/debug      std_msgs/String
"""

import math
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3Stamped
from std_msgs.msg import Bool, String
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandLong, ParamSetV2, SetMode
from rcl_interfaces.msg import ParameterValue, ParameterType


class Stage(Enum):
    TAKEOFF = 0
    FOLLOW = 1
    ALIGN = 2
    DESCEND = 3
    DONE = 4


DT = 0.05  # 50 ms control period
# PX4 rejects an OFFBOARD request unless setpoints are already streaming; prime
# for this many ticks (1 s @ 20 Hz) before requesting the switch.
OFFBOARD_PRIME_TICKS = 20


class FollowPreclandNode(Node):
    def __init__(self):
        super().__init__('follow_precland_node')

        # --- Parameters -----------------------------------------------------
        self.flight_alt = self.declare_parameter('flight_alt', 5.0).value
        self.descend_cone = self.declare_parameter('descend_cone', 0.35).value
        self.descend_min_scale = self.declare_parameter('descend_min_scale', 0.3).value
        self.platform_height = self.declare_parameter('platform_height', 0.0).value
        self.land_clearance = self.declare_parameter('land_clearance', 0.15).value
        self.final_descent_h = self.declare_parameter('final_descent_h', 0.6).value
        self.land_align_radius = self.declare_parameter('land_align_radius', 0.20).value
        self.descend_rate = self.declare_parameter('descend_rate', 0.3).value
        self.kf_accel_std = self.declare_parameter('kf_accel_std', 0.1).value
        self.kf_meas_std = self.declare_parameter('kf_meas_std', 0.15).value
        self.coast_ticks = self.declare_parameter('coast_ticks', 30).value
        self.vel_gain = self.declare_parameter('vel_gain', 0.4).value
        self.vel_max = self.declare_parameter('vel_max', 5.0).value
        self.cam_hfov = self.declare_parameter('cam_hfov', 1.74).value
        self.cam_aspect = self.declare_parameter('cam_aspect', 4.0 / 3.0).value
        self.lat_swap = self.declare_parameter('lat_swap', False).value
        self.lat_sign_fwd = self.declare_parameter('lat_sign_fwd', 1.0).value
        self.lat_sign_left = self.declare_parameter('lat_sign_left', 1.0).value
        # Camera lever arm: the down camera sits cam_offset_fwd m forward (+X body)
        # and cam_offset_left m left (+Y body) of the DRONE CENTRE. Added when
        # projecting the marker to world so the servo lands the drone centre -- not
        # the camera -- on the marker (else it lands cam_offset off-target).
        self.cam_offset_fwd = self.declare_parameter('cam_offset_fwd', 0.0).value
        self.cam_offset_left = self.declare_parameter('cam_offset_left', 0.0).value
        self.yaw_track_min_speed = self.declare_parameter('yaw_track_min_speed', 0.5).value
        self.marker_confirm_frames = self.declare_parameter('marker_confirm_frames', 3).value
        # AUTO.FOLLOW_TARGET follow height (m). If follow_set_height=true the node
        # pushes flight_alt into PX4's follow-height param on FOLLOW entry. Param
        # name is firmware-version-specific (v1.13+ FLW_TGT_HT, older NAV_FT_HT) --
        # set follow_height_param to match. Off by default (uses PX4's own value).
        self.follow_set_height = self.declare_parameter('follow_set_height', False).value
        self.follow_height_param = self.declare_parameter('follow_height_param', 'FLW_TGT_HT').value
        # follow_auto_mode: true = the NODE commands AUTO.FOLLOW_TARGET over mavros
        # after takeoff (the node drives the follow). false = an external GCS (e.g.
        # phone QGC's Follow Me button) sets the mode instead -- the node just holds
        # a hover until the mode flips, then watches for the marker.
        self.follow_auto_mode = self.declare_parameter('follow_auto_mode', True).value
        # PX4 refuses to arm without an RC/GCS datalink (NAV_DLL_ACT>0); set true
        # for a bare SITL with no GCS to clear the arming check.
        self.disable_gcs_failsafe = self.declare_parameter('disable_gcs_failsafe', False).value
        self.cue_timeout = self.declare_parameter('cue_timeout', 1.0).value

        self.stage = Stage.TAKEOFF

        # --- State ----------------------------------------------------------
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]
        self.offset = Point()
        self.fresh = 0                    # marker freshness countdown (coasts dropouts)
        self._detect_streak = 0           # consecutive raw detections (false-positive filter)
        self.req_ticks = 0
        self.yaw = 0.0
        self.hold_yaw = 0.0
        self.kf_x = np.zeros(4)           # [E, N, vE, vN]
        self.kf_P = np.eye(4)
        self.kf_init = False
        self.kf_miss = 0
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.dbg_tick = 0
        # Marker cue (/marker/position) -- used for the open-loop final descent.
        self.cue = None
        self.cue_stamp = None
        self.cue_vel = [0.0, 0.0]
        self.cue_vel_stamp = None
        self._nav_dll_attempts = 0
        self._follow_ht_attempts = 0

        # --- Publishers -----------------------------------------------------
        self.raw_pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.debug_pub = self.create_publisher(String, '/follow_precland/debug', 10)

        # --- Subscribers ----------------------------------------------------
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb,
            qos_profile_sensor_data)
        self.create_subscription(Point, '/perception/aruco_offset', self._offset_cb, 10)
        self.create_subscription(Bool, '/perception/aruco_detected', self._detected_cb, 10)
        self.create_subscription(PointStamped, '/marker/position', self._cue_cb, 10)
        self.create_subscription(Vector3Stamped, '/marker/velocity', self._vel_cb, 10)

        # --- Service clients ------------------------------------------------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.command_cli = self.create_client(CommandLong, '/mavros/cmd/command')
        self.param_set_cli = self.create_client(ParamSetV2, '/mavros/param/set')

        self.create_timer(DT, self.tick)
        self._log(f'follow_precland up: flight_alt={self.flight_alt:.1f}m '
                  f'follow_auto_mode={self.follow_auto_mode} vel_max={self.vel_max:.1f}')

    # -----------------------------------------------------------------------
    # Callbacks
    def _state_cb(self, msg):
        self.mav_state = msg

    def _pose_cb(self, msg):
        self.pos[0] = msg.pose.position.x
        self.pos[1] = msg.pose.position.y
        self.pos[2] = msg.pose.position.z
        q = msg.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _offset_cb(self, msg):
        self.offset = msg

    def _detected_cb(self, msg):
        if msg.data:
            self.fresh = 10
            self._detect_streak += 1
        else:
            self._detect_streak = 0

    def _vel_cb(self, msg):
        self.cue_vel = [msg.vector.x, msg.vector.y]
        self.cue_vel_stamp = self.get_clock().now()

    def _cue_vel_fresh(self):
        if self.cue_vel_stamp is None:
            return False
        return (self.get_clock().now() - self.cue_vel_stamp).nanoseconds * 1e-9 < self.cue_timeout

    def _cue_cb(self, msg):
        now = self.get_clock().now()
        new = [msg.point.x, msg.point.y]
        if not self._cue_vel_fresh():
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
        if self.cue is None or self.cue_stamp is None:
            return False
        return (self.get_clock().now() - self.cue_stamp).nanoseconds * 1e-9 < self.cue_timeout

    def _marker_confirmed(self):
        return self._detect_streak >= self.marker_confirm_frames

    # -----------------------------------------------------------------------
    def tick(self):
        if self.fresh > 0:
            self.fresh -= 1
            if self.fresh == 0:
                self._detect_streak = 0
        marker_ok = self.fresh > 0

        if not self.mav_state.connected:
            return

        if self.disable_gcs_failsafe and self._nav_dll_attempts < 5 and \
                self.req_ticks % 20 == 1:
            self._nav_dll_attempts += 1
            self._set_param_int('NAV_DLL_ACT', 0)

        if self.stage == Stage.TAKEOFF:
            # PX4 OFFBOARD needs setpoints streaming before/during the switch, so
            # publish every tick. OFFBOARD -> arm -> climb under velocity command.
            self.req_ticks += 1
            send = (self.req_ticks % 40 == 1)
            if self.mav_state.mode != 'OFFBOARD':
                if self.req_ticks >= OFFBOARD_PRIME_TICKS and send:
                    self._log('TAKEOFF: request OFFBOARD')
                    self._set_mode('OFFBOARD')
            elif not self.mav_state.armed:
                if send:
                    self._log('TAKEOFF: arming')
                    self._arm(True)
            elif self.pos[2] < self.flight_alt - 0.5:
                self.cmd_vel[2] = 1.0
            else:
                self._log('-> FOLLOW (reached flight_alt)')
                self.req_ticks = 0     # so the AUTO.FOLLOW_TARGET command fires at once
                self.stage = Stage.FOLLOW
            self._publish_velocity()
            return

        if self.stage == Stage.FOLLOW:
            # Drive the follow by COMMANDING AUTO.FOLLOW_TARGET over mavros; PX4
            # then follows the GCS's FOLLOW_TARGET stream. We stream nothing once
            # following (PX4 owns the vehicle).
            if not self.mav_state.armed:
                self._log('disarmed -> TAKEOFF')
                self.stage = Stage.TAKEOFF
                self.req_ticks = 0
                return
            self.req_ticks += 1
            if self.follow_set_height and self._follow_ht_attempts < 5 and \
                    self.req_ticks % 20 == 1:
                self._follow_ht_attempts += 1
                self._set_param_float(self.follow_height_param, self.flight_alt)
            if self.mav_state.mode != 'AUTO.FOLLOW_TARGET':
                # Not following yet. Keep streaming a HOVER setpoint so OFFBOARD
                # doesn't failsafe during the switch (and so we hold position while
                # waiting for an external GCS when follow_auto_mode=false). The
                # node's mode command (below) IS the "ROS command" that drives
                # the follow.
                if self.follow_auto_mode and self.req_ticks % 40 == 1:
                    self._log('FOLLOW: command AUTO.FOLLOW_TARGET (via mavros)')
                    self._set_mode('AUTO.FOLLOW_TARGET')
                self.cmd_vel = [0.0, 0.0, 0.0]
                self._publish_velocity()
                return
            # In AUTO.FOLLOW_TARGET: PX4 flies the follow -- stream NOTHING, just
            # watch for the marker to hand off to the vision servo.
            if self._marker_confirmed():
                self._log('marker acquired -> ALIGN (re-taking OFFBOARD)')
                self._kf_reset(seed_vel=self.cue_vel if self._cue_ok() else None)
                self.req_ticks = 0
                self.stage = Stage.ALIGN
            return

        if self.stage == Stage.ALIGN:
            if not self.mav_state.armed:
                self._log('disarmed -> TAKEOFF')
                self.stage = Stage.TAKEOFF
                self.req_ticks = 0
                return
            # Re-take the vehicle: command OFFBOARD over mavros. Streaming the
            # vision servo below primes it; PX4 needs ~1 s of stream first.
            self.req_ticks += 1
            if self.mav_state.mode != 'OFFBOARD' and \
                    self.req_ticks >= OFFBOARD_PRIME_TICKS and self.req_ticks % 20 == 1:
                self._log('ALIGN: request OFFBOARD')
                self._set_mode('OFFBOARD')
            err = self._track(marker_ok)
            self.cmd_vel[2] = 0.0
            if self.kf_miss > self.coast_ticks:
                self._log('marker lost -> FOLLOW (re-acquire)')
                self.stage = Stage.FOLLOW
                self.req_ticks = 0
            elif err is not None and marker_ok and err < self._funnel_radius() \
                    and self.mav_state.mode == 'OFFBOARD':
                self._log('-> DESCEND')
                self.stage = Stage.DESCEND
            self._publish_velocity()
            return

        if self.stage == Stage.DESCEND:
            if not self.mav_state.armed:
                self._log('disarmed -> TAKEOFF')
                self.stage = Stage.TAKEOFF
                self.req_ticks = 0
                return
            err = self._track(marker_ok)
            h = self._height_above_marker()
            if h < self.final_descent_h:
                # FINAL descent: marker overflows the frame; finish open-loop on
                # the marker cue (if present) / KF coast.
                if self._cue_ok():
                    lateral_err = self._servo_to(self.cue[0], self.cue[1],
                                                 self.cue_vel[0], self.cue_vel[1])
                else:
                    lateral_err = err if err is not None else 999.0
                if lateral_err < self.land_align_radius:
                    if h < self.land_clearance:
                        self._log(f'centred ({lateral_err:.2f} m) -> DONE (disarm)')
                        self.stage = Stage.DONE
                    else:
                        self.cmd_vel[2] = -self.descend_rate * self.descend_min_scale
                else:
                    self.cmd_vel[2] = 0.0
            elif self.kf_miss > self.coast_ticks and not self._cue_ok():
                self._log('marker lost (no cue) -> ALIGN (hold alt)')
                self.cmd_vel[2] = 0.0
                self.stage = Stage.ALIGN
            else:
                if self.kf_miss > self.coast_ticks and self._cue_ok():
                    err = self._servo_to(self.cue[0], self.cue[1],
                                         self.cue_vel[0], self.cue_vel[1])
                lateral_err = err if err is not None else 999.0
                if self.kf_init:
                    funnel = self._funnel_radius()
                    if lateral_err <= funnel:
                        scale = self._clamp(1.0 - lateral_err / funnel,
                                            self.descend_min_scale, 1.0)
                        self.cmd_vel[2] = -self.descend_rate * scale
                    else:
                        self.cmd_vel[2] = 0.0
                else:
                    self.cmd_vel[2] = 0.0
            self._publish_velocity()
            return

        if self.stage == Stage.DONE:
            if self.mav_state.armed:
                self._force_disarm()
            else:
                self._log('disarmed onto target -> shutting down node')
                rclpy.shutdown()
            return

    # -----------------------------------------------------------------------
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

        eE = self.kf_x[0] - self.pos[0]
        eN = self.kf_x[1] - self.pos[1]
        vffE, vffN = (self.cue_vel[0], self.cue_vel[1]) if self._cue_ok() \
            else (self.kf_x[2], self.kf_x[3])
        self._servo_to(self.kf_x[0], self.kf_x[1], vffE, vffN, vmax_override=vmax_override)
        self._face_velocity()

        self.dbg_tick += 1
        if self.dbg_tick % 20 == 0:
            self._log(f'off=({self.offset.x:+.2f},{self.offset.y:+.2f}) '
                      f'err=({eE:+.2f},{eN:+.2f}) '
                      f'cmd=({self.cmd_vel[0]:+.2f},{self.cmd_vel[1]:+.2f})')
        return math.hypot(eE, eN)

    def _height_above_marker(self):
        ph = self.get_parameter('platform_height').value
        return max(self.pos[2] - ph, 0.05)

    def _funnel_radius(self):
        r_land = self.get_parameter('land_align_radius').value
        slope = self.get_parameter('descend_cone').value
        return max(r_land, slope * self._height_above_marker())

    def _servo_to(self, targetE, targetN, vffE=0.0, vffN=0.0, vmax_override=None):
        kp = self.get_parameter('vel_gain').value
        vmax = vmax_override if vmax_override is not None \
            else self.get_parameter('vel_max').value
        eE = targetE - self.pos[0]
        eN = targetN - self.pos[1]
        vE = vffE + kp * eE
        vN = vffN + kp * eN
        speed = math.hypot(vE, vN)
        if speed > vmax and speed > 1e-6:
            scale = vmax / speed
            vE *= scale
            vN *= scale
        self.cmd_vel[0] = vE
        self.cmd_vel[1] = vN
        return math.hypot(eE, eN)

    def _face_velocity(self):
        vE, vN = self.cmd_vel[0], self.cmd_vel[1]
        if math.hypot(vE, vN) >= self.get_parameter('yaw_track_min_speed').value:
            self.hold_yaw = math.atan2(vN, vE)

    def _measure_marker_world(self):
        swap = self.get_parameter('lat_swap').value
        sign_fwd = self.get_parameter('lat_sign_fwd').value
        sign_left = self.get_parameter('lat_sign_left').value
        hfov = self.get_parameter('cam_hfov').value
        aspect = self.get_parameter('cam_aspect').value

        ix, iy = self.offset.x, self.offset.y
        if swap:
            ix, iy = iy, ix
        alt = self._height_above_marker()
        half_w = alt * math.tan(hfov * 0.5)
        half_h = half_w / aspect
        gx = ix * half_w
        gy = iy * half_h
        fwd = sign_fwd * (-gy)
        left = sign_left * (-gx)
        # Add the camera lever arm (camera is forward/left of the drone centre) so
        # the returned point is the marker relative to the DRONE CENTRE.
        fwd += self.get_parameter('cam_offset_fwd').value
        left += self.get_parameter('cam_offset_left').value
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        de = c * fwd - s * left
        dn = s * fwd + c * left
        return self.pos[0] + de, self.pos[1] + dn

    # --- Kalman filter (constant-velocity, state [E, N, vE, vN]) -------------
    def _kf_reset(self, seed_vel=None):
        self.kf_init = False
        self.kf_miss = 0
        self._kf_seed_vel = list(seed_vel) if seed_vel is not None else [0.0, 0.0]

    def _kf_predict(self):
        if not self.kf_init:
            return
        dt = DT
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
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
            sv = getattr(self, '_kf_seed_vel', [0.0, 0.0])
            self.kf_x = np.array([zE, zN, sv[0], sv[1]])
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
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        pt.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                        PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                        PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                        PositionTarget.IGNORE_YAW_RATE)
        pt.velocity.x = self.cmd_vel[0]
        pt.velocity.y = self.cmd_vel[1]
        pt.velocity.z = self.cmd_vel[2]
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

    def _force_disarm(self):
        req = CommandLong.Request()
        req.command = 400
        req.param1 = 0.0
        req.param2 = 21196.0
        self.command_cli.call_async(req)

    def _set_param_int(self, param_id, value):
        req = ParamSetV2.Request()
        req.param_id = param_id
        req.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
        self.param_set_cli.call_async(req)

    def _set_param_float(self, param_id, value):
        req = ParamSetV2.Request()
        req.param_id = param_id
        req.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        self.param_set_cli.call_async(req)

    def _log(self, text):
        self.get_logger().info(text)
        self.debug_pub.publish(String(data=text))

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))


def main(args=None):
    rclpy.init(args=args)
    node = FollowPreclandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
