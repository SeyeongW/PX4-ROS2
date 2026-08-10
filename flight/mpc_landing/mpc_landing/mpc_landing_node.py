"""mpc_landing_node — gated hardware precision landing over MAVROS.

The mission is deliberately small: take off to 5 m, look for the marker, and
descend onto it under MPC. Two things are NOT small.

First, the MPC is `simulation/landing_mpc`'s, imported rather than reimplemented.
The point of this node is to fly the controller that was validated in SITL; a
lookalike written here would only ever validate the lookalike. The weights are
the SITL values unchanged and the limits are lowered for a real airframe over a
stationary marker (see `_declare`).

Second, the permission model — every irreversible step stops and waits for a
human:

    preflight PASS ─approve─► ARM ─approve─► TAKEOFF ─approve─► SEARCH
                                                                  │
                                                        marker seen│
                                                                  ▼
                                                             DESCEND → LAND

Only the last step is automatic. See `mission.py` for why.

Operating it
------------
    ros2 launch mpc_landing mpc_landing.launch.py

    ros2 topic echo /mpc_landing/state          # phase + what it is waiting for
    ros2 service call /mpc_landing/approve std_srvs/srv/Trigger   # release a gate
    ros2 service call /mpc_landing/abort   std_srvs/srv/Trigger   # stop, land, disarm

ALL PARAMETERS ARE DECLARED HERE, WITH THEIR VALUES, IN `_declare`.
The launch file passes none — by design, so there is exactly one place to look
when a number needs to change and no chance of a launch override silently
disagreeing with the source. They remain ROS parameters, so they are still
inspectable and overridable from the command line for a one-off test:

    ros2 run mpc_landing mpc_landing_node --ros-args -p takeoff_alt_m:=3.0

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
    /mpc_landing/state                     std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land
"""

from __future__ import annotations

import math
import sys
import threading

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

from landing_mpc.mpc import LandingMPC
from landing_mpc.predictor import predict_const_vel
from landing_mpc.reference import HorizonReference

from .mission import CheckResult, GateState, Phase


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


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes telemetry BEST_EFFORT; a RELIABLE subscriber gets nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class MpcLandingNode(Node):
    def __init__(self):
        super().__init__('mpc_landing_node')
        self._declare()
        self._read_params()

        self.gate = GateState()
        # THE controller under test — imported, not reimplemented.  This node
        # exists to fly the MPC that simulation/landing_mpc validated in SITL;
        # a lookalike written here would only ever validate the lookalike.
        # Limits are re-tuned for a real airframe (see _declare), but the law,
        # the cone and the QP are byte-for-byte the SITL ones.
        self.mpc = LandingMPC(
            dt_s=self.mpc_dt, horizon=self.mpc_horizon,
            w_xy=self.mpc_w_xy, w_z=self.mpc_w_z,
            w_vxy=self.mpc_w_vxy, w_vz=self.mpc_w_vz,
            w_a=self.mpc_w_a, w_terminal=self.mpc_w_terminal,
            v_max=self.mpc_v_max, a_max=self.mpc_a_max, vz_max=self.mpc_vz_max,
            cone_k=self.mpc_cone_k, w_jerk=self.mpc_w_jerk, j_max=self.mpc_j_max)
        self._ref = HorizonReference(lead_s=self.mpc_dt)
        self._t_solve = None
        self._solve_k = 0

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
        self._t_prestream = None
        # SEARCH commits to the descent only after this many CONSECUTIVE fresh
        # detections, so a single spurious ArUco hit cannot trip an irreversible
        # descent (see the SEARCH phase).
        self._acq_streak = 0
        # Last time each throttled service call fired, keyed by name, so the
        # TOUCHDOWN/ABORT handlers stop re-sending land/disarm every tick.
        self._t_calls: dict[str, float] = {}

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
        # BEST_EFFORT to match the detector. aruco_pose_node publishes its
        # perception topics BEST_EFFORT (sensor-style), and a RELIABLE
        # subscriber is INCOMPATIBLE with a BEST_EFFORT publisher — DDS
        # silently delivers nothing. Measured: `ros2 topic hz` showed 29 Hz
        # while this node saw zero, and preflight reported the pipeline
        # "silent" with the camera and detector both plainly running.
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

        # Terminal approval. A daemon thread blocks on stdin so the control
        # loop never does — a mission node that stops publishing setpoints
        # because it is waiting for a keystroke would drop out of offboard.
        self._stdin_ok = self.interactive and sys.stdin is not None \
            and sys.stdin.isatty()
        self._prompted = ''
        if self._stdin_ok:
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self.create_timer(1.0 / self.rate_hz, self._tick)
        # Say the mode FIRST — it is the one thing you want confirmed before a
        # real vehicle is armed in front of you.
        self.get_logger().info(
            f'mpc_landing_node: PX4 mode={self.mode_name} | '
            f'takeoff {self.takeoff_alt:.1f} m | '
            f'descend on {self.marker_pose_topic}')
        if self._stdin_ok:
            self.get_logger().info(
                'each step will ask on this terminal — ENTER approves, n aborts')
        else:
            self.get_logger().info(
                f'stdin is not a terminal, so approvals must come from the '
                f'service: ros2 service call /{self.get_name()}/approve '
                f'std_srvs/srv/Trigger')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these numbers may be set. Launch files pass none."""
        p = self.declare_parameter
        # --- mission geometry
        p('takeoff_alt_m', 5.0)             # target altitude above takeoff point
        p('alt_tolerance_m', 0.3)           # counts as "reached" within this
        p('climb_speed_m_s', 0.7)           # TAKEOFF climb cap; the descent
                                            # rate is the MPC's mpc_vz_max_m_s
        # Touchdown gate ONLY. Descent is gated by the MPC's own corridor
        # (mpc_cone_k); this is how close counts as 'on the marker'.
        p('touchdown_xy_m', 0.35)
        p('touchdown_alt_m', 0.25)          # below this, hand over to LAND
        p('touchdown_dwell_s', 1.5)         # ...held for this long
        # --- marker input
        # The frame a marker pose is ALREADY in. Anything else on that topic is
        # taken to be the camera optical frame and converted — see _on_marker.
        p('map_frame', 'map')
        p('marker_pose_topic', '/perception/down/marker_pose')
        p('marker_detected_topic', '/perception/down/aruco_detected')
        p('marker_timeout_s', 1.5)          # older than this is not a fix
        p('marker_lost_abort_s', 5.0)       # gone this long mid-descent -> abort
        p('search_timeout_s', 60.0)         # no marker in SEARCH -> abort
        # SEARCH -> DESCEND is automatic and irreversible, so require the marker
        # to be seen this many CONSECUTIVE ticks before committing.  One frame is
        # enough for a false positive to start a descent; 5 ticks (~0.25 s at
        # rate_hz) is still immediate to a human but rejects a lone bad fix.
        p('marker_acquire_frames', 5)
        # --- preflight thresholds
        p('min_battery_v', 14.0)            # 4S nominal; raise for 6S
        p('require_battery', True)          # false only for bench tests
        # --- flight controller. PX4 ONLY — see the module docstring.
        p('offboard_mode', 'OFFBOARD')
        # How long to stream setpoints before asking PX4 for offboard. PX4 wants
        # a steady stream, not a single message; 1 s at rate_hz is ~20 of them.
        p('offboard_prestream_s', 1.0)
        p('rate_hz', 20.0)
        # Prompt on the terminal at each gate and take Enter as approval. Falls
        # back to the service alone when stdin is not a terminal (e.g. under
        # `ros2 launch`), because a prompt nobody can answer would hang the
        # mission at its first gate.
        p('interactive_approval', True)
        p('service_timeout_s', 5.0)
        # --- descent MPC: landing_mpc.LandingMPC, the SITL-validated one.
        # WEIGHTS are the SITL values unchanged — they are what was validated,
        # and retuning them here would mean this flight tests something else.
        # LIMITS are lowered, because a real airframe is the one thing SITL did
        # not have: the SITL descent ran v_max up to 3.5 m/s chasing a 3 m/s
        # deck, and this first flight is over a STATIONARY marker where nothing
        # needs to be chased at all.
        p('mpc_dt_s', 0.1)
        p('mpc_horizon', 20)                # SITL value
        p('mpc_w_xy', 6.0)                  # SITL value
        p('mpc_w_z', 3.0)                   # SITL value
        p('mpc_w_vxy', 1.5)                 # SITL value
        p('mpc_w_vz', 1.5)                  # SITL value
        p('mpc_w_a', 0.05)                  # SITL value
        p('mpc_w_terminal', 40.0)           # SITL value
        p('mpc_w_jerk', 0.5)                # SITL value
        p('mpc_v_max_m_s', 0.8)             # LOWERED: stationary target
        p('mpc_a_max_m_s2', 0.6)            # LOWERED: ~3.5 deg of tilt
        p('mpc_vz_max_m_s', 0.35)           # LOWERED: the actual descent rate
        p('mpc_j_max_m_s3', 1.5)            # LOWERED from 2.0
        # Descent corridor: |p_xy| <= h/cone_k, i.e. the MPC will not come down
        # while off-centre.  1/tan(vfov/2) for the down camera; raise it to
        # descend more conservatively, lower it only with a wider lens.
        p('mpc_cone_k', 1.6)
        p('mpc_solve_every', 5)             # re-plan at rate_hz/this

    def _read_params(self) -> None:
        g = self.get_parameter
        self.takeoff_alt = float(g('takeoff_alt_m').value)
        self.alt_tol = float(g('alt_tolerance_m').value)
        self.climb_speed = float(g('climb_speed_m_s').value)
        self.touch_xy = float(g('touchdown_xy_m').value)
        self.touch_alt = float(g('touchdown_alt_m').value)
        self.touch_dwell = float(g('touchdown_dwell_s').value)
        self.map_frame = str(g('map_frame').value)
        self.marker_pose_topic = str(g('marker_pose_topic').value)
        self.marker_detected_topic = str(g('marker_detected_topic').value)
        self.marker_timeout = float(g('marker_timeout_s').value)
        self.marker_lost_abort = float(g('marker_lost_abort_s').value)
        self.search_timeout = float(g('search_timeout_s').value)
        self.acquire_frames = int(g('marker_acquire_frames').value)
        self.min_batt = float(g('min_battery_v').value)
        self.require_batt = bool(g('require_battery').value)
        self.mode_name = str(g('offboard_mode').value)
        self.prestream_s = float(g('offboard_prestream_s').value)
        self.interactive = bool(g('interactive_approval').value)
        self.rate_hz = float(g('rate_hz').value)
        self.svc_timeout = float(g('service_timeout_s').value)
        self.mpc_dt = float(g('mpc_dt_s').value)
        self.mpc_horizon = int(g('mpc_horizon').value)
        self.mpc_w_xy = float(g('mpc_w_xy').value)
        self.mpc_w_z = float(g('mpc_w_z').value)
        self.mpc_w_vxy = float(g('mpc_w_vxy').value)
        self.mpc_w_vz = float(g('mpc_w_vz').value)
        self.mpc_w_a = float(g('mpc_w_a').value)
        self.mpc_w_terminal = float(g('mpc_w_terminal').value)
        self.mpc_w_jerk = float(g('mpc_w_jerk').value)
        self.mpc_v_max = float(g('mpc_v_max_m_s').value)
        self.mpc_a_max = float(g('mpc_a_max_m_s2').value)
        self.mpc_vz_max = float(g('mpc_vz_max_m_s').value)
        self.mpc_j_max = float(g('mpc_j_max_m_s3').value)
        self.mpc_cone_k = float(g('mpc_cone_k').value)
        self.mpc_solve_every = int(g('mpc_solve_every').value)

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
        """Accept the marker in `map`, or in the camera's optical frame.

        Which one it is comes off the message, not off a parameter — the two
        ends can then never be configured to disagree, and switching the
        detector between them needs no change here.
        """
        p = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
        if m.header.frame_id and m.header.frame_id != self.map_frame:
            p = self._marker_enu_from_camera(p)
            if p is None:
                return
        self.marker = p
        self.marker_t = self._now()

    def _marker_enu_from_camera(self, tvec):
        """Camera-optical marker -> ENU, on the gimbal's nadir hold alone.

        No tf2, and deliberately so: the transforms are published correctly but
        this process cannot drain them fast enough to look one up at capture
        time, so the whole chain resolves to nothing. What is actually needed is
        one angle — where the camera is pointing in the world — and a gimbal
        commanded to nadir supplies all of it but the heading, which MAVROS
        already gives us on the pose we are differencing against anyway.

        The cost is the assumption: roll and pitch off the gimbal are ignored.
        A 3-axis gimbal holds nadir to well under a degree, and at a 5 m search
        height one degree is 9 cm — but if it is ever knocked off, this reads
        the error as marker offset and flies toward it. Watch the debug view.

        Optical is X right, Y down the image, Z along the lens. At nadir the
        top of the image is the nose, so forward is -y and left is -x, and the
        marker is -z below. Range comes straight from solvePnP, which makes the
        descent gate measure height above the MARKER instead of above whatever
        datum the EKF started at.
        """
        if self.pose is None:
            return None
        q = self.pose.pose.orientation
        return marker_enu_from_nadir_camera(
            tvec,
            (self.pose.pose.position.x, self.pose.pose.position.y,
             self.pose.pose.position.z),
            enu_yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )

    # ------------------------------------------------------------ terminal UI
    def _stdin_loop(self) -> None:
        """Read approvals from the terminal. ENTER = yes.

        Anything typed is judged on its first character, so `y`, `yes` and a
        bare ENTER all approve and `n` aborts. Deliberately forgiving in the
        approve direction and strict in the abort one: the operator is standing
        next to a vehicle, not filling in a form.
        """
        for line in sys.stdin:
            answer = line.strip().lower()
            if not self.gate.waiting:
                self.get_logger().warn(
                    f'ignored "{answer or "ENTER"}" — nothing is waiting for '
                    f'approval right now (phase {self.gate.phase.value})')
                continue
            if answer.startswith('n'):
                self.gate.abort('operator declined at the prompt')
                print('\n  ABORTING — landing and disarming.\n', flush=True)
                continue
            ok, msg = self.gate.approve()
            print(f'\n  {"OK" if ok else "REFUSED"}: {msg}\n', flush=True)

    def _prompt(self) -> None:
        """Ask, once per gate, on the terminal."""
        if not self._stdin_ok or not self.gate.waiting:
            return
        key = self.gate.phase.value
        if key == self._prompted:
            return
        self._prompted = key
        # Printed rather than logged: the logger prefixes and timestamps every
        # line, which is exactly wrong for something the operator has to read
        # and answer under time pressure.
        print(f'\n{"=" * 72}\n  {self.gate.prompt}\n'
              f'{"=" * 72}\n  proceed?  [ENTER = yes / n = abort]  ',
              end='', flush=True)

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
            # Throttled hard. This is checked at rate_hz, so an unthrottled
            # message floods 20 lines a second and buries the approval prompt
            # the operator is supposed to be reading — the one thing on screen
            # that must stay visible.
            self.get_logger().error(
                f"service '{name}' not available — is MAVROS running?",
                throttle_duration_sec=5.0)
            return False
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, n=name: self.get_logger().info(f'{n} -> {f.result()}'))
        return True

    def _call_throttled(self, client, request, name, period=1.0):
        """Like `_call`, but at most once per `period` seconds.

        The TOUCHDOWN/ABORT handlers run every tick; without this they would
        re-send land/disarm at rate_hz, flooding the operator's log and
        hammering the FCU with identical commands.  The first call after a gap
        fires immediately.
        """
        if self._now() - self._t_calls.get(name, 0.0) < period:
            return
        self._t_calls[name] = self._now()
        self._call(client, request, name)

    def _fresh_marker(self) -> bool:
        return (self.marker is not None
                and (self._now() - self.marker_t) < self.marker_timeout)

    def _on_ground(self) -> bool:
        """True once the vehicle has actually settled on the ground.

        Uses extended_state, which the FCU derives from its own land detector,
        so the disarm in TOUCHDOWN/ABORT waits for real ground contact instead
        of firing from a geometric gate that a bad fix could satisfy in mid-air.
        An already-disarmed FCU counts too, so a LAND that auto-disarmed on the
        ground still resolves even if extended_state never arrived.
        """
        if self.state and not self.state.armed:
            return True
        return (self.ext is not None
                and self.ext.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

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
        # Velocity + yaw only: ignore position and acceleration.  The FORCE bit
        # is deliberately NOT set -- it reinterprets the (ignored) accel fields
        # as a force, which PX4 does not support on this path and may reject.
        m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                       | PositionTarget.IGNORE_PZ
                       | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                       | PositionTarget.IGNORE_AFZ
                       | PositionTarget.IGNORE_YAW_RATE)
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
        # There is deliberately NO altitude check here. It used to refuse to
        # start above max_start_alt_m, but a disarmed vehicle is not flying and
        # 'disarmed at start' below already refuses a live one — so the only
        # thing the altitude gate actually caught was an EKF whose z datum had
        # not settled on the pad, which grounds the mission for a reason that
        # has nothing to do with whether it is safe to fly.
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

        # KEEP THE OFFBOARD STREAM ALIVE, unconditionally, for every phase that
        # needs it. This runs BEFORE the phase logic so that a phase which
        # returns early — a gate waiting on a human, ARMING waiting out the
        # pre-stream — cannot accidentally starve it. PX4 drops offboard after
        # ~0.5 s of silence, and the phases that would go quiet are exactly the
        # ones where the vehicle is armed and airborne.
        # Phases that fly a real setpoint (TAKEOFF, SEARCH, DESCEND) overwrite
        # this one later in the same tick; publishing twice is harmless, a gap
        # is not.
        if self.gate.needs_setpoint_stream:
            self._send(0.0, 0.0, 0.0)

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
            # Gates hold position, and on PX4 they must keep the offboard stream
            # alive too — READY_TO_TAKEOFF happens AFTER arming, and a lapse
            # there drops the vehicle out of offboard while it waits for a human.
            self._announce()
            return

        if ph is Phase.ARMING:
            # ORDER MATTERS ON PX4: stream -> mode -> arm.
            # PX4 rejects a request for OFFBOARD unless setpoints are already
            # arriving, so the stream (kept up by `needs_setpoint_stream` at the
            # top of this tick) has to lead. ArduPilot does not care, and the
            # same order works there, so there is one path rather than two.
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
                self._call(self.mode_cli, req, 'set_mode')
                return
            if self.state and self.state.armed:
                self.gate.armed_confirmed()
                self._announce()
                return
            # Only ask to arm once the FCU confirms it is IN the mode. Arming
            # first and switching after is the ArduPilot habit; on PX4 it can
            # arm in whatever mode it was in, which is not what anyone wants.
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
            vz = float(np.clip(err, -self.climb_speed, self.climb_speed))
            self._send(0.0, 0.0, vz)
            return

        if ph is Phase.SEARCH:
            self._send(0.0, 0.0, 0.0)
            # Commit only after several CONSECUTIVE fresh detections.  A live
            # fix AND a currently-asserted `detected` flag both have to hold;
            # a single spurious hit trips one tick and the streak resets, so it
            # cannot start an irreversible descent on its own.
            if self._fresh_marker() and self.detected:
                self._acq_streak += 1
            else:
                self._acq_streak = 0
            if self._acq_streak >= self.acquire_frames:
                self._t_solve = None
                self._ref = HorizonReference(lead_s=self.mpc_dt)
                self.gate.marker_acquired()
                self.get_logger().info(
                    f'marker acquired ({self._acq_streak} consecutive fixes) — '
                    f'descending automatically from here')
                return
            if self._now() - self._t_phase > self.search_timeout:
                self.gate.abort(f'no marker within {self.search_timeout:.0f} s')
            return

        if ph is Phase.DESCEND:
            self._descend()
            return

        if ph is Phase.TOUCHDOWN:
            # Hand to the autopilot's own landing, then disarm ONLY once it has
            # actually settled -- confirmed by extended_state, never by the
            # geometric gate -- so motors are never cut in the air.  AUTO.LAND
            # normally auto-disarms on the ground; the explicit disarm is a
            # gated backstop.  Both are throttled: this runs every tick.
            self._call_throttled(self.land_cli, CommandTOL.Request(), 'land')
            if self._on_ground():
                req = CommandBool.Request()
                req.value = False
                self._call_throttled(self.arm_cli, req, 'disarm')
            if self.state and not self.state.armed:
                self.gate.finished()
                self.get_logger().info('disarmed — mission complete')
            return

        if ph is Phase.ABORT:
            # Same discipline as TOUCHDOWN: land, disarm once on the ground, and
            # then finish, so an abort ends in a known DONE state instead of
            # looping on the land command forever.
            self._call_throttled(self.land_cli, CommandTOL.Request(), 'land')
            if self._on_ground():
                req = CommandBool.Request()
                req.value = False
                self._call_throttled(self.arm_cli, req, 'disarm')
            if self.state and not self.state.armed:
                self.gate.finished()
                self.get_logger().info('disarmed after abort — safe on the ground')
            return

    def _descend(self) -> None:
        """Fly the SITL-validated MPC, in the frame it was written for.

        `LandingMPC` works in RELATIVE coordinates (vehicle minus target) and
        solves all three axes: the horizontal pair, then the vertical one
        against a corridor `z_ref = cone_k*|p_xy|`, which is what stops it
        descending while off-centre.  That corridor is the reason not to
        hand-roll a vertical rule here — it is the part of the controller this
        flight is meant to check.

        The target is treated as STATIONARY: this first flight is over a fixed
        marker, so the const-velocity prediction is fed zero velocity.  The same
        call takes a real velocity the day the marker moves, which is the point
        of reusing this MPC rather than a simplified stand-in.
        """
        gone = self._now() - self.marker_t
        if not self._fresh_marker() and gone > self.marker_lost_abort:
            self.gate.abort(f'marker lost for {gone:.1f} s during descent')
            return
        if self.pose is None or self.marker is None:
            return

        # Absolute ENU state, and the target we are closing on.  MAVROS local
        # position is already ENU, so no NED conversion belongs here.
        p_d = np.array([self.pose.pose.position.x, self.pose.pose.position.y,
                        self._alt()])
        v_d = (np.array([self.vel.twist.linear.x, self.vel.twist.linear.y,
                         self.vel.twist.linear.z]) if self.vel else np.zeros(3))
        tgt = np.array([self.marker[0], self.marker[1], self.marker[2]])
        tgt_v = np.zeros(3)                     # stationary marker, for now

        self._solve_k += 1
        if self._t_solve is None or self._solve_k % self.mpc_solve_every == 0:
            p_rel0 = p_d - tgt
            v_rel0 = v_d - tgt_v
            P, V, A = predict_const_vel(tgt, tgt_v, self.mpc_dt, self.mpc_horizon)
            res = self.mpc.solve(p_rel0, v_rel0, P, V, A)
            self._ref.set_plan(p_rel0, v_rel0, res.pred_rel_pos,
                               res.pred_rel_vel, res.pred_rel_acc,
                               self.mpc_dt, tgt, tgt_v, np.zeros(3))
            self._t_solve = self._now()
            if not res.success:
                self.get_logger().warn('MPC solve failed — flying the fallback',
                                       throttle_duration_sec=2.0)

        if not self._ref.ready():
            self._send(0.0, 0.0, 0.0)
            return

        # The MPC plans at 10 Hz; `HorizonReference` interpolates it up to the
        # setpoint rate so the vehicle is never handed a stale knot.
        _pos, vel, _acc = self._ref.sample(self._now() - self._t_solve)
        vel = np.clip(vel, [-self.mpc_v_max, -self.mpc_v_max, -self.mpc_vz_max],
                      [self.mpc_v_max, self.mpc_v_max, self.mpc_vz_max])
        self._send(vel[0], vel[1], vel[2])

        radius = float(np.linalg.norm(p_d[:2] - tgt[:2]))
        # Height above the MARKER, not above the EKF's origin. With the marker
        # coming from vision this is solvePnP's own range, so the handover to
        # LAND happens at a real distance from the deck rather than at whatever
        # the estimator's datum drifted to since takeoff. When the marker
        # arrives in `map` instead, the same subtraction is still the right
        # quantity — the touchdown gate is about the deck either way.
        alt = float(p_d[2] - tgt[2])
        if alt <= self.touch_alt and radius <= self.touch_xy:
            self._t_touch = self._t_touch or self._now()
            if self._now() - self._t_touch >= self.touch_dwell:
                self.gate.touched_down()
        else:
            self._t_touch = None

        if self._solve_k % int(self.rate_hz * 2) == 0:
            corridor = self.mpc_cone_k * radius
            self.get_logger().info(
                f'[DESCEND] alt={alt:.2f} m  xy_err={radius:.2f} m  '
                f'corridor needs h>={corridor:.2f} m  vz={vel[2]:+.2f} m/s  '
                f'{"descending" if vel[2] < -0.02 else "HOLDING to centre"}')

    # ------------------------------------------------------------------ output
    def _announce(self) -> None:
        """Say once, loudly, what the operator is being asked to authorise."""
        if not self.gate.waiting:
            self._announced = ''
            self._prompted = ''
            return
        self._prompt()
        key = self.gate.phase.value
        if key == self._announced:
            return
        self._announced = key
        if self._stdin_ok:
            # The prompt already asked, on its own clean lines. Logging the
            # same thing again here would print a timestamped duplicate right
            # across the line the operator is typing into.
            return
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
    node = MpcLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
