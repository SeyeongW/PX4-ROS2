"""precland_hw_node — gated hardware precision landing over MAVROS.

The mission is deliberately small: take off to 5 m, look for the marker, and
descend onto it under MPC. What is NOT small is the permission model — every
irreversible step stops and waits for a human:

    preflight PASS ─approve─► ARM ─approve─► TAKEOFF ─approve─► SEARCH
                                                                  │
                                                        marker seen│
                                                                  ▼
                                                             DESCEND → LAND

Only the last step is automatic. See `mission.py` for why.

Operating it
------------
    ros2 launch precland_hw precland_hw.launch.py

    ros2 topic echo /precland_hw/state          # phase + what it is waiting for
    ros2 service call /precland_hw/approve std_srvs/srv/Trigger   # release a gate
    ros2 service call /precland_hw/abort   std_srvs/srv/Trigger   # stop, land, disarm

ALL PARAMETERS ARE DECLARED HERE, WITH THEIR VALUES, IN `_declare`.
The launch file passes none — by design, so there is exactly one place to look
when a number needs to change and no chance of a launch override silently
disagreeing with the source. They remain ROS parameters, so they are still
inspectable and overridable from the command line for a one-off test:

    ros2 run precland_hw precland_hw_node --ros-args -p takeoff_alt_m:=3.0

Interfaces
----------
Subscribes
    /mavros/state                          mavros_msgs/State
    /mavros/local_position/pose            geometry_msgs/PoseStamped
    /mavros/local_position/velocity_local  geometry_msgs/TwistStamped
    /mavros/extended_state                 mavros_msgs/ExtendedState
    /mavros/battery                        sensor_msgs/BatteryState
    /perception/down/marker_pose           geometry_msgs/PoseStamped
    /perception/down/aruco_detected        std_msgs/Bool
Publishes
    /mavros/setpoint_raw/local             mavros_msgs/PositionTarget
    /precland_hw/state                     std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from mavros_msgs.msg import ExtendedState, PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

from .descent_mpc import DescentMPC
from .mission import CheckResult, GateState, Phase


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes telemetry BEST_EFFORT; a RELIABLE subscriber gets nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class PreclandHwNode(Node):
    def __init__(self):
        super().__init__('precland_hw_node')
        self._declare()
        self._read_params()

        self.gate = GateState()
        self.mpc = DescentMPC(
            dt=self.mpc_dt, horizon=self.mpc_horizon, v_max=self.mpc_v_max,
            a_max=self.mpc_a_max, j_max=self.mpc_j_max, w_p=self.mpc_w_p,
            w_v=self.mpc_w_v, w_a=self.mpc_w_a, w_j=self.mpc_w_j)

        # --- telemetry state
        self.state: State | None = None
        self.pose: PoseStamped | None = None
        self.vel: TwistStamped | None = None
        self.ext: ExtendedState | None = None
        self.batt: BatteryState | None = None
        self.marker: np.ndarray | None = None      # local ENU
        self.marker_t = 0.0
        self.detected = False
        self._detector_seen = False   # have we heard from the detector AT ALL?
        self._checks: list[CheckResult] = []
        self._t_phase = self._now()
        self._announced = ''
        self._takeoff_xy: np.ndarray | None = None
        self._t_touch = None

        # --- MAVROS
        self.create_subscription(State, '/mavros/state', self._on_state,
                                 _sensor_qos())
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self._on_pose, _sensor_qos())
        self.create_subscription(TwistStamped,
                                 '/mavros/local_position/velocity_local',
                                 self._on_vel, _sensor_qos())
        self.create_subscription(ExtendedState, '/mavros/extended_state',
                                 self._on_ext, _sensor_qos())
        self.create_subscription(BatteryState, '/mavros/battery',
                                 self._on_batt, _sensor_qos())
        self.create_subscription(PoseStamped, self.marker_pose_topic,
                                 self._on_marker, 10)
        self.create_subscription(Bool, self.marker_detected_topic,
                                 self._on_detected, 10)

        self.sp_pub = self.create_publisher(PositionTarget,
                                            '/mavros/setpoint_raw/local', 10)
        self.state_pub = self.create_publisher(String, '~/state', 10)

        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.land_cli = self.create_client(CommandTOL, '/mavros/cmd/land')

        self.create_service(Trigger, '~/approve', self._on_approve)
        self.create_service(Trigger, '~/abort', self._on_abort)

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'precland_hw_node: takeoff {self.takeoff_alt:.1f} m, descend on '
            f'{self.marker_pose_topic}. Every step before the descent needs '
            f'`ros2 service call {self.get_name()}/approve std_srvs/srv/Trigger`.')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these numbers may be set. Launch files pass none."""
        p = self.declare_parameter
        # --- mission geometry
        p('takeoff_alt_m', 5.0)             # target altitude above takeoff point
        p('alt_tolerance_m', 0.3)           # counts as "reached" within this
        p('descent_speed_m_s', 0.35)        # vertical rate during the descent
        p('align_radius_m', 0.35)           # descend only while centred to this
        p('touchdown_alt_m', 0.25)          # below this, hand over to LAND
        p('touchdown_dwell_s', 1.5)         # ...held for this long
        # --- marker input
        p('marker_pose_topic', '/perception/down/marker_pose')
        p('marker_detected_topic', '/perception/down/aruco_detected')
        p('marker_timeout_s', 1.5)          # older than this is not a fix
        p('marker_lost_abort_s', 5.0)       # gone this long mid-descent -> abort
        p('search_timeout_s', 60.0)         # no marker in SEARCH -> abort
        # --- preflight thresholds
        p('min_battery_v', 14.0)            # 4S nominal; raise for 6S
        p('require_battery', True)          # false only for bench tests
        p('max_start_alt_m', 1.0)           # must start near the ground
        # --- flight controller
        p('offboard_mode', 'GUIDED')        # ArduPilot; use 'OFFBOARD' for PX4
        p('rate_hz', 20.0)
        p('service_timeout_s', 5.0)
        # --- descent MPC (horizontal plane only; see descent_mpc.py)
        p('mpc_dt_s', 0.1)
        p('mpc_horizon', 15)
        p('mpc_v_max_m_s', 0.8)             # gentle: this is a real airframe
        p('mpc_a_max_m_s2', 0.6)
        p('mpc_j_max_m_s3', 1.5)
        p('mpc_w_p', 6.0)
        p('mpc_w_v', 2.0)
        p('mpc_w_a', 0.05)
        p('mpc_w_j', 0.5)

    def _read_params(self) -> None:
        g = self.get_parameter
        self.takeoff_alt = float(g('takeoff_alt_m').value)
        self.alt_tol = float(g('alt_tolerance_m').value)
        self.descent_speed = float(g('descent_speed_m_s').value)
        self.align_radius = float(g('align_radius_m').value)
        self.touch_alt = float(g('touchdown_alt_m').value)
        self.touch_dwell = float(g('touchdown_dwell_s').value)
        self.marker_pose_topic = str(g('marker_pose_topic').value)
        self.marker_detected_topic = str(g('marker_detected_topic').value)
        self.marker_timeout = float(g('marker_timeout_s').value)
        self.marker_lost_abort = float(g('marker_lost_abort_s').value)
        self.search_timeout = float(g('search_timeout_s').value)
        self.min_batt = float(g('min_battery_v').value)
        self.require_batt = bool(g('require_battery').value)
        self.max_start_alt = float(g('max_start_alt_m').value)
        self.mode_name = str(g('offboard_mode').value)
        self.rate_hz = float(g('rate_hz').value)
        self.svc_timeout = float(g('service_timeout_s').value)
        self.mpc_dt = float(g('mpc_dt_s').value)
        self.mpc_horizon = int(g('mpc_horizon').value)
        self.mpc_v_max = float(g('mpc_v_max_m_s').value)
        self.mpc_a_max = float(g('mpc_a_max_m_s2').value)
        self.mpc_j_max = float(g('mpc_j_max_m_s3').value)
        self.mpc_w_p = float(g('mpc_w_p').value)
        self.mpc_w_v = float(g('mpc_w_v').value)
        self.mpc_w_a = float(g('mpc_w_a').value)
        self.mpc_w_j = float(g('mpc_w_j').value)

    # -------------------------------------------------------------- callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, m): self.state = m

    def _on_pose(self, m): self.pose = m

    def _on_vel(self, m): self.vel = m

    def _on_ext(self, m): self.ext = m

    def _on_batt(self, m): self.batt = m

    def _on_detected(self, m):
        self.detected = bool(m.data)
        self._detector_seen = True

    def _on_marker(self, m: PoseStamped):
        self.marker = np.array([m.pose.position.x, m.pose.position.y,
                                m.pose.position.z])
        self.marker_t = self._now()

    # ---------------------------------------------------------------- services
    def _on_approve(self, _req, res):
        ok, msg = self.gate.approve()
        res.success, res.message = ok, msg
        (self.get_logger().info if ok else self.get_logger().warn)(msg)
        return res

    def _on_abort(self, _req, res):
        self.gate.abort('operator aborted')
        res.success, res.message = True, 'aborting: landing and disarming'
        self.get_logger().warn(res.message)
        return res

    # ---------------------------------------------------------------- helpers
    def _call(self, client, request, name):
        """Fire-and-log a MAVROS service call.

        Async on purpose: this runs inside the control timer, and blocking on a
        future here would stall the setpoint stream that keeps the vehicle in
        offboard control — which the flight controller reads as loss of link.
        """
        if not client.service_is_ready():
            self.get_logger().error(f"service '{name}' not available")
            return False
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, n=name: self.get_logger().info(f'{n} -> {f.result()}'))
        return True

    def _fresh_marker(self) -> bool:
        return (self.marker is not None
                and (self._now() - self.marker_t) < self.marker_timeout)

    def _alt(self) -> float:
        return float(self.pose.pose.position.z) if self.pose else float('nan')

    def _rel_to_marker(self):
        """Vehicle position/velocity relative to the marker, horizontal plane."""
        p = np.array([self.pose.pose.position.x, self.pose.pose.position.y])
        v = np.array([self.vel.twist.linear.x, self.vel.twist.linear.y]) \
            if self.vel else np.zeros(2)
        return p - self.marker[:2], v

    def _send(self, vx: float, vy: float, vz: float) -> None:
        """Stream a velocity setpoint in the local ENU frame.

        Velocity rather than position: the descent is a regulation problem
        against a marker whose absolute position we only know through the same
        estimator that is moving, and a position setpoint would re-inject that
        estimate's drift as a command.
        """
        m = PositionTarget()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                       | PositionTarget.IGNORE_PZ
                       | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                       | PositionTarget.IGNORE_AFZ
                       | PositionTarget.FORCE | PositionTarget.IGNORE_YAW_RATE)
        m.velocity.x, m.velocity.y, m.velocity.z = float(vx), float(vy), float(vz)
        m.yaw = 0.0
        self.sp_pub.publish(m)

    # ------------------------------------------------------------- preflight
    def _run_checks(self) -> list[CheckResult]:
        c = []
        c.append(CheckResult(
            'FCU link', bool(self.state and self.state.connected),
            'no /mavros/state' if not self.state else
            ('connected' if self.state.connected else 'MAVROS sees no FCU')))
        c.append(CheckResult(
            'local position', self.pose is not None,
            'have EKF local position' if self.pose is not None
            else 'no /mavros/local_position/pose — EKF not ready'))
        c.append(CheckResult(
            'velocity estimate', self.vel is not None,
            'ok' if self.vel is not None else 'no local velocity'))
        on_ground = self.pose is not None and self._alt() < self.max_start_alt
        c.append(CheckResult(
            'on the ground', on_ground,
            f'alt {self._alt():.2f} m < {self.max_start_alt:.2f} m' if on_ground
            else f'alt {self._alt():.2f} m — expected to start on the ground'))
        # Say WHICH failure this is. Reporting "already ARMED" when the real
        # problem is that no telemetry has arrived sends the operator to inspect
        # the vehicle instead of the link.
        if self.state is None:
            armed_detail = 'unknown — no /mavros/state'
        elif self.state.armed:
            armed_detail = 'already ARMED — refusing to take over a live vehicle'
        else:
            armed_detail = 'disarmed'
        c.append(CheckResult(
            'disarmed at start', bool(self.state and not self.state.armed),
            armed_detail))
        # ALIVE, not SEEING. Requiring an actual detection here would ground
        # the vehicle whenever the marker is not visible from the pad — which
        # is the normal case, and the reason SEARCH happens at altitude.
        c.append(CheckResult(
            'marker pipeline', self._detector_seen,
            f'detector publishing on {self.marker_detected_topic}'
            f'{" (marker in view)" if self.detected else " (no marker yet — fine on the ground)"}'
            if self._detector_seen
            else f'silent — nothing on {self.marker_detected_topic}'))
        if self.require_batt:
            v = float(self.batt.voltage) if self.batt else 0.0
            c.append(CheckResult(
                'battery', self.batt is not None and v >= self.min_batt,
                f'{v:.1f} V >= {self.min_batt:.1f} V' if self.batt
                else 'no /mavros/battery'))
        return c

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        ph = self.gate.phase
        self._publish_state()

        if ph is Phase.PRECHECK:
            self._checks = self._run_checks()
            if all(x.passed for x in self._checks):
                for x in self._checks:
                    self.get_logger().info(str(x))
                self.gate.checks_passed()
                self._announce()
            elif self._now() - self._t_phase > 5.0:
                self._t_phase = self._now()
                for x in self._checks:
                    if not x.passed:
                        self.get_logger().warn(f'preflight blocked: {x}')
            return

        if ph in (Phase.READY_TO_ARM, Phase.READY_TO_TAKEOFF,
                  Phase.READY_TO_SEARCH):
            # Gates hold position. Once airborne that means actively holding, so
            # the offboard stream must not lapse while we wait for a human.
            if self.gate.flying:
                self._send(0.0, 0.0, 0.0)
            self._announce()
            return

        if ph is Phase.ARMING:
            if self.state and self.state.mode != self.mode_name:
                req = SetMode.Request()
                req.custom_mode = self.mode_name
                self._call(self.mode_cli, req, 'set_mode')
                return
            if self.state and self.state.armed:
                self.gate.armed_confirmed()
                self._announce()
                return
            req = CommandBool.Request()
            req.value = True
            self._call(self.arm_cli, req, 'arming')
            return

        if ph is Phase.TAKEOFF:
            if self._takeoff_xy is None and self.pose is not None:
                self._takeoff_xy = np.array([self.pose.pose.position.x,
                                             self.pose.pose.position.y])
            err = self.takeoff_alt - self._alt()
            if abs(err) <= self.alt_tol:
                self._send(0.0, 0.0, 0.0)
                self.gate.altitude_reached()
                self._announce()
                return
            # Climb at a capped rate, easing off near the target so the vehicle
            # settles instead of overshooting into the gate.
            vz = float(np.clip(err, -self.descent_speed * 2.0,
                               self.descent_speed * 2.0))
            self._send(0.0, 0.0, vz)
            return

        if ph is Phase.SEARCH:
            self._send(0.0, 0.0, 0.0)
            if self._fresh_marker():
                self.mpc.reset()
                self.gate.marker_acquired()
                self.get_logger().info(
                    'marker acquired — descending automatically from here')
                return
            if self._now() - self._t_phase > self.search_timeout:
                self.gate.abort(f'no marker within {self.search_timeout:.0f} s')
            return

        if ph is Phase.DESCEND:
            self._descend()
            return

        if ph is Phase.TOUCHDOWN:
            self._call(self.land_cli, CommandTOL.Request(), 'land')
            req = CommandBool.Request()
            req.value = False
            self._call(self.arm_cli, req, 'disarm')
            if self.state and not self.state.armed:
                self.gate.finished()
                self.get_logger().info('disarmed — mission complete')
            return

        if ph is Phase.ABORT:
            self._call(self.land_cli, CommandTOL.Request(), 'land')
            return

    def _descend(self) -> None:
        gone = self._now() - self.marker_t
        if not self._fresh_marker() and gone > self.marker_lost_abort:
            self.gate.abort(f'marker lost for {gone:.1f} s during descent')
            return
        if self.pose is None:
            return

        p_rel, v_rel = self._rel_to_marker()
        a_cmd = self.mpc.step(p_rel, v_rel)
        # The MPC returns an acceleration; MAVROS is being streamed velocity, so
        # integrate one step. Keeping the QP in acceleration is what lets the
        # jerk limit mean anything.
        v_xy = np.clip(v_rel + a_cmd * self.mpc_dt, -self.mpc_v_max,
                       self.mpc_v_max)

        # Align before descending — the one rule that decides whether the marker
        # is still in frame when it matters.
        radius = float(np.linalg.norm(p_rel))
        vz = -self.descent_speed if radius <= self.align_radius else 0.0
        self._send(-v_xy[0], -v_xy[1], vz)

        alt = self._alt()
        if alt <= self.touch_alt and radius <= self.align_radius:
            self._t_touch = self._t_touch or self._now()
            if self._now() - self._t_touch >= self.touch_dwell:
                self.gate.touched_down()
        else:
            self._t_touch = None

        if int(self._now() * self.rate_hz) % int(self.rate_hz * 2) == 0:
            self.get_logger().info(
                f'[DESCEND] alt={alt:.2f} m  xy_err={radius:.2f}/'
                f'{self.align_radius:.2f} m  vz={vz:+.2f}  '
                f'{"descending" if vz else "HOLDING to centre"}')

    # ------------------------------------------------------------------ output
    def _announce(self) -> None:
        """Say once, loudly, what the operator is being asked to authorise."""
        if not self.gate.waiting:
            self._announced = ''
            return
        key = self.gate.phase.value
        if key == self._announced:
            return
        self._announced = key
        self.get_logger().warn(
            f'>>> WAITING FOR APPROVAL — {self.gate.prompt}\n'
            f'    ros2 service call /{self.get_name()}/approve '
            f'std_srvs/srv/Trigger')

    def _publish_state(self) -> None:
        ph = self.gate.phase
        extra = f' | {self.gate.prompt}' if self.gate.waiting else ''
        if ph is Phase.ABORT:
            extra = f' | {self.gate.abort_reason}'
        self.state_pub.publish(String(data=f'{ph.value}{extra}'))
        if ph is not getattr(self, '_last_ph', None):
            self._last_ph = ph
            self._t_phase = self._now()


def main(args=None):
    rclpy.init(args=args)
    node = PreclandHwNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
