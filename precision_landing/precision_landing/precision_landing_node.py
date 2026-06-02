#!/usr/bin/env python3
"""ArUco precision landing controller (ArduPilot / MAVROS, GUIDED mode).

Python port of offboard/src/precision_landing.cpp. The node consumes the
normalised marker offset published by camera_detection's aruco_detector_node
and flies the drone down onto the marker by streaming position setpoints.

State machine
  TAKEOFF : set GUIDED, arm, then /mavros/cmd/takeoff up to flight_alt
            (skipped if auto_takeoff=false — start in IDLE, wait for a human)
  IDLE    : wait for armed + GUIDED + marker visible
  ALIGN   : hover at flight_alt, slide laterally until marker is centred
  DESCEND : keep marker centred while descending; pause if marker lost
  DONE    : switch FCU to LAND mode

Topic contract (matches aruco_detector_node):
  in  /perception/aruco_offset    geometry_msgs/Point   normalised [-1, 1]
  in  /perception/aruco_detected  std_msgs/Bool
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

from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Bool, String
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


class Stage(Enum):
    TAKEOFF = 0
    IDLE = 1
    ALIGN = 2
    DESCEND = 3
    DONE = 4


DT = 0.05  # 50 ms control period


class PrecisionLandingNode(Node):
    def __init__(self):
        super().__init__('precision_landing_node')

        # --- Parameters -----------------------------------------------------
        self.flight_alt = self.declare_parameter('flight_alt', 5.0).value      # m, hover altitude
        self.align_radius = self.declare_parameter('align_radius', 0.30).value  # m, start descent inside this
        self.land_alt = self.declare_parameter('land_alt', 0.7).value          # m, hand off to LAND
        # Landing gate (option 1): only commit to LAND at land_alt if the marker
        # is within this horizontal error (m); otherwise re-align first so we
        # never land grossly off-target.
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
        self.vel_max = self.declare_parameter('vel_max', 1.0).value            # m/s, horizontal cap
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

        # --- Service clients ------------------------------------------------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

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
            if self.mav_state.armed and self.mav_state.mode == 'GUIDED' and marker_ok:
                self._log('-> ALIGN')
                self._kf_reset()
                self.stage = Stage.ALIGN

        elif self.stage == Stage.ALIGN:
            if not self.mav_state.armed:
                self._log('disarmed -> IDLE')
                self.stage = Stage.IDLE
            else:
                err = self._track(marker_ok)            # KF + horizontal velocity cmd
                self.cmd_vel[2] = 0.0                   # hold altitude
                if self.kf_miss > self.coast_ticks:
                    self._log('marker lost -> IDLE')
                    self.stage = Stage.IDLE
                elif err is not None and marker_ok and err < self.align_radius:
                    self._log('-> DESCEND')
                    self.stage = Stage.DESCEND

        elif self.stage == Stage.DESCEND:
            if not self.mav_state.armed:
                self._log('disarmed -> IDLE')
                self.stage = Stage.IDLE
            else:
                err = self._track(marker_ok)            # keep centred on KF estimate
                if self.kf_miss > self.coast_ticks:
                    self._log('marker lost -> ALIGN (hold alt)')
                    self.cmd_vel[2] = 0.0
                    self.stage = Stage.ALIGN            # stop descending until reacquired
                else:
                    # Descent rate scales with alignment: full speed when centred,
                    # zero when error >= align_radius.
                    # Condition is kf_init (world position of marker centre is known)
                    # not marker_ok (ArUco pattern currently visible) — the drone
                    # tracks the KF world estimate through momentary dropouts.
                    lateral_err = err if err is not None else 999.0
                    if self.kf_init:
                        descent_scale = max(0.0, 1.0 - lateral_err / self.align_radius)
                        self.cmd_vel[2] = -self.descend_rate * descent_scale
                    else:
                        self.cmd_vel[2] = 0.0
                    # Landing gate: at land_alt, only commit to LAND if well centred.
                    if self.pos[2] < self.land_alt:
                        e = lateral_err
                        if e < self.land_align_radius:
                            self._log(f'centred ({e:.2f} m) -> DONE (LAND)')
                            self.stage = Stage.DONE
                        else:
                            self._log(f'off-centre ({e:.2f} m) at land_alt -> ALIGN')
                            self.cmd_vel[2] = 0.0
                            self.stage = Stage.ALIGN

        elif self.stage == Stage.DONE:
            if self.mav_state.mode != 'LAND':
                self._set_mode('LAND')
            return  # stop sending setpoints; ArduCopter LAND finishes

        self._publish_velocity()

    # -----------------------------------------------------------------------
    # Track the marker with a constant-velocity Kalman filter and steer the
    # setpoint toward the smoothed estimate. Returns the horizontal distance (m)
    # from the drone to the estimated marker, or None if not yet initialised.
    def _track(self, marker_ok):
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

        # Velocity visual servo: v = vel_gain · error, clamped to vel_max.
        # First-order (exponential) approach, no overshoot.
        kp = self.get_parameter('vel_gain').value
        vmax = self.get_parameter('vel_max').value
        eE = self.kf_x[0] - self.pos[0]
        eN = self.kf_x[1] - self.pos[1]
        self.cmd_vel[0] = self._clamp(kp * eE, -vmax, vmax)
        self.cmd_vel[1] = self._clamp(kp * eN, -vmax, vmax)
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
        alt = max(self.pos[2], 0.3)
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
