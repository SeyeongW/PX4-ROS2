"""aruco_landing_node — the plainest ArUco precision landing over MAVROS.

The next rung up from `naive_flight_node`.  Naive proved the boring half on a
real airframe (MAVROS wired, PX4 accepts OFFBOARD, the vehicle climbs, holds and
lands).  This node keeps that exact skeleton and adds ONE thing: it descends
onto an ArUco marker instead of landing where it took off.

    PRECHECK ─approve─► ARM ─► TAKEOFF ─► SEARCH ─► DESCEND ─► LAND ─► DONE
                                            │ marker acquired  │
                                            └── timeout ───────┘► LAND

Deliberately simple control — a plain proportional centre-and-descend, NOT the
MPC.  `mpc_landing_node` already flies the SITL-validated MPC; this is the
intermediate step that proves the *perception* half (the camera sees the marker,
the frames line up, the vehicle can be driven to it) with control simple enough
that nothing about the descent can be blamed on the controller.  Graduate to
`mpc_landing_node` once this centres reliably.

    horizontal:  v_xy = clip(kp * (marker_xy - vehicle_xy),  |v| <= center_v_max)
    vertical:    descend at descend_speed ONLY while centred (radius <= r),
                 otherwise hold altitude and centre first
    handover:    height above marker <= touchdown_alt AND centred -> autopilot LAND

The MARKER comes from `aruco_landing`'s `aruco_pose_node`, which already
publishes the two topics below.  It solves the marker in the camera's optical
frame and, with `landing_tf_node` running, republishes it in `map` — the frame
MAVROS local position is in — so the offset is a plain subtraction.  If the pose
still arrives in the camera frame (no tf chain), it is converted here on the
gimbal's nadir hold alone, exactly as `mpc_landing_node` does.

The MAVROS discipline is lifted verbatim from `naive_flight_node`: BEST_EFFORT
sensor QoS, stream→mode→arm order, keeping the stream alive through the gate,
ground-gated disarm, and confirming every change from telemetry.

    ros2 run mpc_landing aruco_landing_node          # ENTER approves at the gate

Under `ros2 launch` stdin is not a terminal, so approve over the service:

    ros2 run mpc_landing approve aruco_landing_node
    ros2 run mpc_landing abort   aruco_landing_node  # land now, from any phase

Interfaces
----------
Subscribes
    /mavros/state                          mavros_msgs/State
    /mavros/local_position/pose            geometry_msgs/PoseStamped
    /mavros/extended_state                 mavros_msgs/ExtendedState
    /mavros/battery                        sensor_msgs/BatteryState
    /mavros/estimator_status               mavros_msgs/EstimatorStatus
    /mavros/gpsstatus/gps1/raw             mavros_msgs/GPSRAW
    /perception/down/marker_pose           geometry_msgs/PoseStamped
    /perception/down/aruco_detected        std_msgs/Bool
Publishes
    /mavros/setpoint_raw/local             mavros_msgs/PositionTarget
    ~/state                                std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land

ALL PARAMETERS ARE DECLARED HERE, IN `_declare`, WITH THEIR VALUES.  Override for
a one-off:

    ros2 run mpc_landing aruco_landing_node --ros-args -p descend_speed_m_s:=0.25
"""

from __future__ import annotations

import math
import sys
import threading
from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from mavros_msgs.msg import (EstimatorStatus, ExtendedState, GPSRAW,
                             PositionTarget, State)
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

from .estimator import DEFAULT_SPEED_ACC_MAX, EstimatorHealth


class Phase(str, Enum):
    PRECHECK = 'PRECHECK'          # running preflight checks
    READY_TO_ARM = 'READY_TO_ARM'  # checks passed, waiting for the one approval
    ARMING = 'ARMING'              # stream -> OFFBOARD -> arm
    TAKEOFF = 'TAKEOFF'            # climbing to takeoff_alt
    SEARCH = 'SEARCH'              # holding at altitude, looking for the marker
    DESCEND = 'DESCEND'            # centring on the marker and coming down
    LAND = 'LAND'                  # handed to the autopilot's LAND, disarming
    DONE = 'DONE'


def enu_yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Heading of an ENU body frame, radians CCW from East."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def marker_enu_from_nadir_camera(tvec, vehicle_enu, yaw_rad: float) -> np.ndarray:
    """Marker in the camera's optical frame -> marker in map ENU.

    Assumes the gimbal is holding nadir, so the only unknown is where the
    vehicle is pointing.  Optical is X right, Y down the image, Z along the lens;
    at nadir the top of the image is the nose, which makes forward -y, left -x,
    and the marker -z below.  Identical to `mpc_landing_node`'s conversion.
    """
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    fwd, left, up = -y, -x, -z
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([
        float(vehicle_enu[0]) + c * fwd - s * left,
        float(vehicle_enu[1]) + s * fwd + c * left,
        float(vehicle_enu[2]) + up,
    ])


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes telemetry BEST_EFFORT; a RELIABLE subscriber gets nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class ArucoLandingNode(Node):
    def __init__(self):
        super().__init__('aruco_landing_node')
        self._declare()
        self._read_params()

        self.phase = Phase.PRECHECK
        self.state: State | None = None
        self.pose: PoseStamped | None = None
        self.ext: ExtendedState | None = None
        self.batt: BatteryState | None = None
        self.marker: np.ndarray | None = None      # marker position in map ENU
        self.marker_t = 0.0
        self.detected = False
        self._detector_seen = False
        self._acq_streak = 0
        self._t_phase = self._now()
        self._t_prestream: float | None = None
        self._t_calls: dict[str, float] = {}
        self._announced = ''
        self._prompted = ''
        self._checks_logged = False
        self._waived: set[str] = set()
        self._warned: set[str] = set()
        self.ekf = EstimatorHealth(speed_acc_max=self.speed_acc_max)
        # The ground the vehicle armed on, in EKF local z. Captured at arm
        # rather than assumed to be 0 — see `_takeoff_target`.
        self._z_ground: float | None = None

        self.create_subscription(State, '/mavros/state', self._on_state,
                                 _sensor_qos())
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self._on_pose, _sensor_qos())
        self.create_subscription(ExtendedState, '/mavros/extended_state',
                                 self._on_ext, _sensor_qos())
        self.create_subscription(BatteryState, '/mavros/battery',
                                 self._on_batt, _sensor_qos())
        # Whether the EKF is actually being aided. A pose alone does not say so
        # — see estimator.py, and the reason this node once prompted for an arm
        # PX4 was always going to refuse.
        self.create_subscription(EstimatorStatus, '/mavros/estimator_status',
                                 self._on_est, _sensor_qos())
        self.create_subscription(GPSRAW, '/mavros/gpsstatus/gps1/raw',
                                 self._on_gps, _sensor_qos())
        # BEST_EFFORT to match the detector: aruco_pose_node publishes its
        # perception topics sensor-style, and a RELIABLE subscriber is
        # INCOMPATIBLE with a BEST_EFFORT publisher — DDS delivers nothing.
        self.create_subscription(PoseStamped, self.marker_pose_topic,
                                 self._on_marker, _sensor_qos())
        self.create_subscription(Bool, self.marker_detected_topic,
                                 self._on_detected, _sensor_qos())

        self.sp_pub = self.create_publisher(PositionTarget,
                                            '/mavros/setpoint_raw/local', 10)
        self.state_pub = self.create_publisher(String, '~/state', 10)

        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.land_cli = self.create_client(CommandTOL, '/mavros/cmd/land')

        self.create_service(Trigger, '~/approve', self._on_approve)
        self.create_service(Trigger, '~/abort', self._on_abort)

        self._stdin_ok = self.interactive and sys.stdin is not None \
            and sys.stdin.isatty()
        if self._stdin_ok:
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'aruco_landing_node: PX4 mode={self.mode_name} | '
            f'takeoff {self.takeoff_alt:.1f} m | descend on '
            f'{self.marker_pose_topic}')
        if self.skip_preflight:
            self.get_logger().warn(
                'skip_preflight IS ON — link, battery and armed-state checks '
                'are waived; only local position still gates the ARM prompt')
        if self._stdin_ok:
            self.get_logger().info(
                'ARM will ask on this terminal — ENTER approves, n aborts')
        else:
            self.get_logger().info(
                f'stdin is not a terminal, so approve over the service: '
                f'ros2 run mpc_landing approve {self.get_name()}')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these numbers may be set."""
        p = self.declare_parameter
        # --- mission geometry
        p('takeoff_alt_m', 5.0)             # search altitude above takeoff point
        p('alt_tolerance_m', 0.3)           # counts as "reached" within this
        p('climb_speed_m_s', 0.7)           # TAKEOFF climb cap
        # --- marker input (aruco_pose_node's default output topics)
        p('map_frame', 'map')
        p('marker_pose_topic', '/perception/down/marker_pose')
        p('marker_detected_topic', '/perception/down/aruco_detected')
        p('marker_timeout_s', 1.5)          # older than this is not a fix
        p('marker_lost_abort_s', 5.0)       # gone this long mid-descent -> LAND
        p('search_timeout_s', 60.0)         # no marker in SEARCH -> LAND
        # SEARCH -> DESCEND is automatic and irreversible, so require the marker
        # to be seen this many CONSECUTIVE ticks before committing — one frame is
        # enough for a false positive to start a descent.
        p('marker_acquire_frames', 5)
        # --- descent control (plain proportional centre-and-descend)
        p('center_kp', 0.8)                 # horizontal P gain [1/s]
        p('center_v_max_m_s', 0.6)          # horizontal speed cap while centring
        # Only sink while the marker is within this horizontal radius; outside it
        # the vehicle holds altitude and centres first, so it never descends off
        # to the side of the pad.
        p('descend_radius_m', 0.30)
        p('descend_speed_m_s', 0.30)        # capped sink rate when centred
        # Hand to the autopilot's LAND at this height above the marker, once
        # centred — below here the camera loses the marker anyway (it leaves the
        # frame), so the autopilot's own land detector finishes the touchdown.
        p('touchdown_alt_m', 0.40)
        p('touchdown_xy_m', 0.20)
        # --- preflight thresholds
        p('min_battery_v', 14.0)            # 4S nominal; raise for 6S
        p('require_battery', True)          # false only for bench tests
        # Refuse the arm prompt while the EKF has no position aiding. Match this
        # to the vehicle's EKF2_REQ_SACC so the message quotes the real limit —
        # the default is the RAISED 1.0 m/s, not PX4's 0.5 (see estimator.py),
        # so a vehicle still at 0.5 will refuse arms this check waved through.
        p('require_gnss_aiding', True)
        p('gps_speed_acc_max_m_s', DEFAULT_SPEED_ACC_MAX)
        p('skip_preflight', False)
        # --- flight controller. PX4 assumed (OFFBOARD); ArduPilot works too.
        p('offboard_mode', 'OFFBOARD')
        p('offboard_prestream_s', 1.0)
        p('rate_hz', 20.0)
        p('interactive_approval', True)

    def _read_params(self) -> None:
        g = self.get_parameter
        self.takeoff_alt = float(g('takeoff_alt_m').value)
        self.alt_tol = float(g('alt_tolerance_m').value)
        self.climb_speed = float(g('climb_speed_m_s').value)
        self.map_frame = str(g('map_frame').value)
        self.marker_pose_topic = str(g('marker_pose_topic').value)
        self.marker_detected_topic = str(g('marker_detected_topic').value)
        self.marker_timeout = float(g('marker_timeout_s').value)
        self.marker_lost_abort = float(g('marker_lost_abort_s').value)
        self.search_timeout = float(g('search_timeout_s').value)
        self.acquire_frames = int(g('marker_acquire_frames').value)
        self.center_kp = float(g('center_kp').value)
        self.center_v_max = float(g('center_v_max_m_s').value)
        self.descend_radius = float(g('descend_radius_m').value)
        self.descend_speed = float(g('descend_speed_m_s').value)
        self.touch_alt = float(g('touchdown_alt_m').value)
        self.touch_xy = float(g('touchdown_xy_m').value)
        self.min_batt = float(g('min_battery_v').value)
        self.require_batt = bool(g('require_battery').value)
        self.require_gnss = bool(g('require_gnss_aiding').value)
        self.speed_acc_max = float(g('gps_speed_acc_max_m_s').value)
        self.skip_preflight = bool(g('skip_preflight').value)
        self.mode_name = str(g('offboard_mode').value)
        self.prestream_s = float(g('offboard_prestream_s').value)
        self.rate_hz = float(g('rate_hz').value)
        self.interactive = bool(g('interactive_approval').value)

    # -------------------------------------------------------------- callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, m): self.state = m

    def _on_pose(self, m): self.pose = m

    def _on_ext(self, m): self.ext = m

    def _on_batt(self, m): self.batt = m

    def _on_est(self, m: EstimatorStatus) -> None:
        self.ekf.on_status(
            self._now(),
            const_pos_mode=m.const_pos_mode_status_flag,
            velocity_horiz=m.velocity_horiz_status_flag,
            pos_horiz_abs=m.pos_horiz_abs_status_flag,
            gps_glitch=m.gps_glitch_status_flag)

    def _on_gps(self, m: GPSRAW) -> None:
        # GPSRAW carries accuracies in mm and mm/s; EstimatorHealth works in
        # metres, so the thresholds read like the PX4 parameters they mirror.
        self.ekf.on_gps(self._now(), fix_type=m.fix_type,
                        satellites=m.satellites_visible,
                        h_acc_m=m.h_acc * 1e-3, vel_acc_m_s=m.vel_acc * 1e-3)

    def _on_detected(self, m):
        self.detected = bool(m.data)
        self._detector_seen = True

    def _on_marker(self, m: PoseStamped):
        """Accept the marker in `map`, or convert it from the camera frame.

        Which one it is comes off the message header, not a parameter, so the
        two ends can never be configured to disagree.
        """
        p = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
        if m.header.frame_id and m.header.frame_id != self.map_frame:
            if self.pose is None:
                return
            q = self.pose.pose.orientation
            p = marker_enu_from_nadir_camera(
                p,
                (self.pose.pose.position.x, self.pose.pose.position.y,
                 self.pose.pose.position.z),
                enu_yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.marker = p
        self.marker_t = self._now()

    # ------------------------------------------------------------ terminal UI
    def _stdin_loop(self) -> None:
        """Read the one approval from the terminal. ENTER = yes, n = abort."""
        for line in sys.stdin:
            answer = line.strip().lower()
            if answer.startswith('n'):
                self._abort('operator declined at the prompt')
                print('\n  ABORTING — landing.\n', flush=True)
                continue
            if self.phase is Phase.READY_TO_ARM:
                self._approve()
                print('\n  OK: arming and taking off.\n', flush=True)
            else:
                self.get_logger().warn(
                    f'ignored "{answer or "ENTER"}" — nothing waiting for '
                    f'approval (phase {self.phase.value})')

    def _prompt(self) -> None:
        if not self._stdin_ok or self.phase is not Phase.READY_TO_ARM:
            return
        if self._prompted == Phase.READY_TO_ARM.value:
            return
        self._prompted = Phase.READY_TO_ARM.value
        print(f'\n{"=" * 72}\n  preflight PASSED — approve to ARM, take off to '
              f'{self.takeoff_alt:.1f} m and land on the marker\n{"=" * 72}\n'
              f'  proceed?  [ENTER = yes / n = abort]  ', end='', flush=True)

    # ---------------------------------------------------------------- services
    def _approve(self) -> tuple[bool, str]:
        if self.phase is not Phase.READY_TO_ARM:
            return False, (f'nothing to approve — phase is {self.phase.value}, '
                           f'which is not a gate')
        self._to(Phase.ARMING)
        return True, 'approved: arming and taking off'

    def _abort(self, reason: str) -> None:
        """Land now, from any phase."""
        if self.phase in (Phase.DONE, Phase.LAND):
            return
        self.get_logger().warn(f'abort: {reason} — landing')
        self._to(Phase.LAND)

    def _on_approve(self, _req, res):
        ok, msg = self._approve()
        res.success, res.message = ok, msg
        (self.get_logger().info if ok else self.get_logger().warn)(msg)
        return res

    def _on_abort(self, _req, res):
        self._abort('operator aborted')
        res.success, res.message = True, 'aborting: landing'
        return res

    # ---------------------------------------------------------------- helpers
    def _to(self, phase: Phase) -> None:
        if phase is self.phase:
            return
        self.get_logger().info(f'{self.phase.value} -> {phase.value}')
        self.phase = phase
        self._t_phase = self._now()

    def _call(self, client, request, name) -> bool:
        """Fire-and-log a MAVROS service call, async so the timer never blocks."""
        if not client.service_is_ready():
            self.get_logger().error(
                f"service '{name}' not available — is MAVROS running?",
                throttle_duration_sec=5.0)
            return False
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, n=name: self.get_logger().info(f'{n} -> {f.result()}'))
        return True

    def _call_throttled(self, client, request, name, period=1.0) -> None:
        """Like `_call`, but at most once per `period` s, and only if it fired."""
        if self._now() - self._t_calls.get(name, 0.0) < period:
            return
        if self._call(client, request, name):
            self._t_calls[name] = self._now()

    def _alt(self) -> float:
        return float(self.pose.pose.position.z) if self.pose else float('nan')

    def _takeoff_target(self) -> float:
        """`takeoff_alt` above the GROUND, in EKF local z.

        Not `takeoff_alt` itself. The EKF's z datum is wherever the estimator
        happened to start, and it drifts: on the pad, disarmed, this airframe
        has been seen reporting z = -5.98 m while standing still. Climbing to an
        absolute z of `takeoff_alt` from there is an eleven-metre climb when
        five were asked for. So the ground is measured at the moment of arming —
        the vehicle is provably on it then — and the target is relative to that.
        """
        base = self._z_ground if self._z_ground is not None else 0.0
        return base + self.takeoff_alt

    def _fresh_marker(self) -> bool:
        return (self.marker is not None
                and (self._now() - self.marker_t) < self.marker_timeout)

    def _on_ground(self) -> bool:
        """True once the vehicle has actually settled — never a geometric guess."""
        if self.state and not self.state.armed:
            return True
        return (self.ext is not None
                and self.ext.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

    def _send(self, vx: float, vy: float, vz: float) -> None:
        """Stream a velocity setpoint in the local ENU frame.

        Velocity, not position: centring is regulation against the marker, and a
        position setpoint would re-inject the estimator's drift. The FORCE bit is
        deliberately NOT set — PX4 does not support it on this path and may
        reject the setpoint.
        """
        m = PositionTarget()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                       | PositionTarget.IGNORE_PZ
                       | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                       | PositionTarget.IGNORE_AFZ
                       | PositionTarget.IGNORE_YAW_RATE)
        m.velocity.x, m.velocity.y, m.velocity.z = float(vx), float(vy), float(vz)
        m.yaw = 0.0
        self.sp_pub.publish(m)

    # ------------------------------------------------------------- preflight
    def _preflight_ok(self) -> bool:
        """Minimal: link up, EKF aided, disarmed, (opt) battery, detector alive.

        `skip_preflight` waives every check EXCEPT local position (TAKEOFF
        regulates on `pose.z`; without it the climb setpoint is NaN and PX4
        discards it). The marker-pipeline check is ALIVE, not SEEING: the
        detector must be publishing, but the marker is not expected from the pad.

        A POSE IS NOT AN ESTIMATE. MAVROS keeps publishing local_position even
        when the EKF has fallen back to constant-position mode with no GNSS
        fusion at all, so `pose is not None` says nothing about whether the
        vehicle can hold position — and PX4 will refuse the OFFBOARD arm anyway.
        The aiding check is what turns that into a message the operator can act
        on instead of a rejection they never see (estimator.py has the numbers
        from the day this was found).
        """
        reasons, waived = [], []
        if self.pose is None:
            reasons.append('no local position — EKF not ready')

        overridable = []
        if self.require_gnss:
            blocked = self.ekf.blocking_reason(self._now())
            if blocked:
                overridable.append(blocked)
        if not (self.state and self.state.connected):
            overridable.append('no FCU link (/mavros/state)')
        if self.state is None:
            overridable.append('no /mavros/state')
        elif self.state.armed:
            overridable.append('already ARMED — refusing to take over a live vehicle')
        if self.require_batt:
            v = float(self.batt.voltage) if self.batt else 0.0
            if self.batt is None:
                overridable.append('no /mavros/battery')
            elif v < self.min_batt:
                overridable.append(f'battery {v:.1f} V < {self.min_batt:.1f} V')
        if not self._detector_seen:
            overridable.append(
                f'marker detector silent — nothing on {self.marker_detected_topic}')
        (waived if self.skip_preflight else reasons).extend(overridable)

        for r in waived:
            if r not in self._waived:
                self._waived.add(r)
                self.get_logger().warn(f'preflight WAIVED by skip_preflight: {r}')
        if reasons:
            if self._now() - self._t_phase > 5.0:
                self._t_phase = self._now()
                for r in reasons:
                    self.get_logger().warn(f'preflight blocked: {r}')
            return False
        # Anything that will not stop the flight but will shape it. Said once,
        # before the gate, so it is a decision rather than a surprise.
        warn = self.ekf.warning(self._now())
        if warn and warn not in self._warned:
            self._warned.add(warn)
            self.get_logger().warn(f'preflight WARNING: {warn}')
        if not self._checks_logged:
            self._checks_logged = True
            self.get_logger().info(
                f'preflight PASSED — {self.ekf.summary(self._now())}')
        return True

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._publish_state()

        # Keep the offboard stream alive, unconditionally, for every armed /
        # about-to-be-armed phase — BEFORE the phase logic, so a phase that
        # returns early cannot starve it. PX4 drops offboard after ~0.5 s of
        # silence. Phases that fly a real setpoint overwrite this in the same
        # tick; publishing twice is harmless, a gap is not.
        if self.phase in (Phase.READY_TO_ARM, Phase.ARMING, Phase.TAKEOFF,
                          Phase.SEARCH, Phase.DESCEND):
            self._send(0.0, 0.0, 0.0)

        if self.phase is Phase.PRECHECK:
            if self._preflight_ok():
                self._to(Phase.READY_TO_ARM)
            return

        if self.phase is Phase.READY_TO_ARM:
            self._announce()
            return

        if self.phase is Phase.ARMING:
            # ORDER MATTERS ON PX4: stream -> mode -> arm.
            if self._t_prestream is None:
                self._t_prestream = self._now()
                self.get_logger().info(
                    f'streaming setpoints for {self.prestream_s:.1f} s before '
                    f'requesting {self.mode_name}')
                return
            if self._now() - self._t_prestream < self.prestream_s:
                return
            if self.state and self.state.mode != self.mode_name:
                req = SetMode.Request()
                req.custom_mode = self.mode_name
                self._call_throttled(self.mode_cli, req, 'set_mode')
                return
            if self.state and self.state.armed:
                # Armed but still on the ground: the one moment the vehicle's
                # true height is known, so this is where the climb datum comes
                # from (`_takeoff_target`).
                self._z_ground = self._alt()
                self.get_logger().info(
                    f'armed on the ground at local z={self._z_ground:.2f} m — '
                    f'climbing to z={self._takeoff_target():.2f} m '
                    f'({self.takeoff_alt:.1f} m above it)')
                self._to(Phase.TAKEOFF)
                return
            req = CommandBool.Request()
            req.value = True
            self._call_throttled(self.arm_cli, req, 'arming')
            return

        if self.phase is Phase.TAKEOFF:
            err = self._takeoff_target() - self._alt()
            if abs(err) <= self.alt_tol:
                self._send(0.0, 0.0, 0.0)
                self._to(Phase.SEARCH)
                return
            vz = float(np.clip(err, -self.climb_speed, self.climb_speed))
            self._send(0.0, 0.0, vz)
            return

        if self.phase is Phase.SEARCH:
            self._send(0.0, 0.0, 0.0)
            # Commit only after several CONSECUTIVE fresh detections: a live fix
            # AND a currently-asserted detected flag both have to hold, so a lone
            # spurious hit trips one tick and the streak resets.
            if self._fresh_marker() and self.detected:
                self._acq_streak += 1
            else:
                self._acq_streak = 0
            if self._acq_streak >= self.acquire_frames:
                self._to(Phase.DESCEND)
                self.get_logger().info(
                    f'marker acquired ({self._acq_streak} consecutive fixes) — '
                    f'descending')
                return
            if self._now() - self._t_phase > self.search_timeout:
                self.get_logger().warn(
                    f'no marker within {self.search_timeout:.0f} s — landing here')
                self._to(Phase.LAND)
            return

        if self.phase is Phase.DESCEND:
            self._descend()
            return

        if self.phase is Phase.LAND:
            # Hand to the autopilot's own landing, then disarm ONLY once it has
            # actually settled (extended_state), so motors are never cut in the
            # air. Both are throttled: this runs every tick.
            self._call_throttled(self.land_cli, CommandTOL.Request(), 'land')
            if self._on_ground():
                req = CommandBool.Request()
                req.value = False
                self._call_throttled(self.arm_cli, req, 'disarm')
            if self.state and not self.state.armed:
                self._to(Phase.DONE)
                self.get_logger().info('disarmed — landing complete')
            return

    def _descend(self) -> None:
        """Plain proportional centre-and-descend onto the marker.

        Horizontal velocity drives the vehicle at the marker; the sink is opened
        only once it is centred, so it never comes down off to the side of the
        pad.  When the marker has been gone longer than `marker_lost_abort`, hand
        to the autopilot's LAND where it is — by then it is centred and low, so
        straight down is the safe move, and chasing a stale fix is not.
        """
        if self.pose is None:
            return
        if not self._fresh_marker():
            if self._now() - self.marker_t > self.marker_lost_abort:
                self.get_logger().warn(
                    f'marker lost for {self._now() - self.marker_t:.1f} s during '
                    f'descent — handing to LAND')
                self._to(Phase.LAND)
            else:
                # brief dropout: hold position and wait for the fix to return
                self._send(0.0, 0.0, 0.0)
            return

        px = float(self.pose.pose.position.x)
        py = float(self.pose.pose.position.y)
        err = np.array([self.marker[0] - px, self.marker[1] - py])
        radius = float(np.linalg.norm(err))

        # Horizontal: proportional, capped.
        v = self.center_kp * err
        n = float(np.linalg.norm(v))
        if n > self.center_v_max:
            v *= self.center_v_max / n

        # Vertical: sink only while centred; otherwise hold and centre first.
        centred = radius <= self.descend_radius
        vz = -self.descend_speed if centred else 0.0
        self._send(float(v[0]), float(v[1]), vz)

        # Handover: low over the marker AND centred -> autopilot LAND finishes it
        # (the marker leaves the frame from here anyway).
        h = self._alt() - float(self.marker[2])
        if h <= self.touch_alt and radius <= self.touch_xy:
            self.get_logger().info(
                f'over the marker (h={h:.2f} m, xy={radius:.2f} m) — handing to LAND')
            self._to(Phase.LAND)
            return

        if int((self._now() - self._t_phase) * self.rate_hz) % int(self.rate_hz) == 0:
            self.get_logger().info(
                f'[DESCEND] h={h:.2f} m  xy={radius:.2f} m  '
                f'{"sinking" if centred else "centring (holding alt)"}')

    # ------------------------------------------------------------------ output
    def _announce(self) -> None:
        self._prompt()
        if self._stdin_ok:
            return
        if self._announced == Phase.READY_TO_ARM.value:
            return
        self._announced = Phase.READY_TO_ARM.value
        self.get_logger().warn(
            f'>>> WAITING FOR APPROVAL — approve to ARM and take off\n'
            f'    ros2 run mpc_landing approve {self.get_name()}')

    def _publish_state(self) -> None:
        self.state_pub.publish(String(data=self.phase.value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
