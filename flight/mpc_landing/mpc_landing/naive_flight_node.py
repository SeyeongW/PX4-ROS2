"""naive_flight_node — takeoff, find the ArUco marker, and land on it.

This is the plain-arithmetic sibling of `mpc_landing_node`: the same marker
input, the same MAVROS discipline, the same gates — but the descent is a
proportional law you can read in one screen instead of a QP.

    PRECHECK ──approve──► ARM ──► TAKEOFF ──► HOVER ──► SEARCH ──► DESCEND ──► LAND ──► DONE
                                                          │ no marker      │ marker lost
                                                          └────────────────┴──► LAND (in place)

Only the ARM step waits for a human; everything after it runs on its own. Once
the operator has authorised the flight there is nothing more to decide — the
profile is fixed — so pausing at altitude would only leave a vehicle hovering
while it waits for a keystroke. `abort` still lands it from any phase.

WHAT IS TAKEN FROM THE FLYING CODE, AND WHAT IS NOT
---------------------------------------------------
TAKEN, not rewritten: the marker input. `marker.MarkerTracker` is the exact
acceptance path that landed on the aircraft — frame decided off the message,
optical-frame fixes converted against the vehicle heading and the gimbal's
nadir hold (no tf2; see `marker.py` for why), a fix older than
`marker_timeout_s` not counted as a fix, and a descent committed only after
several CONSECUTIVE detections. Also taken: BEST_EFFORT on the perception
topics — aruco_pose_node publishes sensor-style, and a RELIABLE subscriber to a
BEST_EFFORT publisher is INCOMPATIBLE, so DDS silently delivers nothing (this
was measured once already: `ros2 topic hz` showed 29 Hz while the mission node
saw zero).

Likewise the MAVROS discipline, lifted verbatim because it is the part that is
easy to get subtly wrong on PX4: BEST_EFFORT sensor QoS, the stream→mode→arm
ORDER, keeping the setpoint stream alive through the gate, confirming every
state change from telemetry rather than the service reply, no FORCE bit in the
type_mask, throttled land/disarm, and disarming only once the FCU's own land
detector reports ground contact.

NOT taken: the MPC. `mpc_landing_node` exists to fly the SITL-validated
controller and this node exists to be the boring baseline underneath it — if
this one flew the MPC too there would be no baseline. `descent_velocity` below
is the whole controller: centre with a P law, and only come down while inside
the same descent corridor the MPC enforces (`descend_cone_k`), so the vehicle
does not descend while off to one side.

    ros2 run mpc_landing naive_flight_node          # ENTER approves at the gate
    ros2 launch mpc_landing naive_marker_landing.launch.py   # camera + detector + this

Under `ros2 launch` stdin is not a terminal, so approve over the service:

    ros2 run mpc_landing approve naive_flight_node
    ros2 run mpc_landing abort   naive_flight_node  # land now, from any phase

`use_marker:=false` is the flight this node started as — climb, hold, land in
place, no perception involved. It is kept because it is the configuration that
has already flown, and it is the right thing to fall back to at the field when
the camera is the problem, without a rebuild.

Interfaces
----------
Subscribes
    /mavros/state                          mavros_msgs/State
    /mavros/local_position/pose            geometry_msgs/PoseStamped
    /mavros/extended_state                 mavros_msgs/ExtendedState
    /mavros/battery                        sensor_msgs/BatteryState
    /perception/down/marker_pose           geometry_msgs/PoseStamped
    /perception/down/aruco_detected        std_msgs/Bool
Publishes
    /mavros/setpoint_raw/local             mavros_msgs/PositionTarget
    ~/state                                std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land

ALL PARAMETERS ARE DECLARED HERE, IN `_declare`, WITH THEIR VALUES — the same
rule as the rest of flight/. Override one for a one-off:

    ros2 run mpc_landing naive_flight_node --ros-args -p takeoff_alt_m:=3.0

When a preflight check is in the way and the operator has eyes on the airframe,
`skip_preflight` waives all of them except local position (see `_preflight_ok`
for why that one cannot be waived):

    ros2 run mpc_landing naive_flight_node --ros-args -p skip_preflight:=true
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
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

from .marker import MarkerTracker, enu_yaw_from_quaternion


class Phase(str, Enum):
    PRECHECK = 'PRECHECK'          # running preflight checks
    READY_TO_ARM = 'READY_TO_ARM'  # checks passed, waiting for the one approval
    ARMING = 'ARMING'              # stream -> OFFBOARD -> arm
    TAKEOFF = 'TAKEOFF'            # climbing to takeoff_alt
    HOVER = 'HOVER'                # settling at altitude for hover_s
    SEARCH = 'SEARCH'              # holding, waiting for the marker
    DESCEND = 'DESCEND'            # centring on the marker and coming down
    LAND = 'LAND'                  # handed to the autopilot's LAND, disarming
    DONE = 'DONE'


@dataclass(frozen=True)
class DescentCommand:
    """One tick of the descent: a velocity, and whether it is coming down."""

    vx: float
    vy: float
    vz: float
    centred: bool           # inside the corridor, i.e. allowed to descend


def descent_velocity(err_e: float, err_n: float, alt_above_marker: float, *,
                     kp_xy: float, v_max_xy: float, cone_k: float,
                     kp_z: float, vz_max: float, vz_min: float
                     ) -> DescentCommand:
    """The entire naive descent controller, as a plain function.

    `err_*` is marker minus vehicle in ENU metres — where to go, not where it
    is — and `alt_above_marker` is height above the DECK, which with a vision
    fix is solvePnP's own range rather than anything the EKF's datum has drifted
    to since takeoff.

    THE CORRIDOR IS THE SAFETY PROPERTY, and it is the MPC's rule reused rather
    than invented: come down only while `alt >= cone_k * radius`, so a vehicle
    that is off to one side centres FIRST and descends after. Without it a P
    law happily flies a diagonal into the ground next to the marker, which
    looks like tracking the whole way down.

    Vertically it eases off near the deck (`kp_z * alt`) but never below
    `vz_min`, because a rate proportional to a height that is approaching zero
    approaches zero too and would hover just above the marker forever. The
    touchdown gate hands over to the autopilot's LAND well before that matters.

    Kept free of ROS so every sign and every gate is testable without a vehicle
    (`test_naive_landing.py`) — a wrong sign here is a landing that misses in a
    believable direction.
    """
    vx = float(np.clip(kp_xy * err_e, -v_max_xy, v_max_xy))
    vy = float(np.clip(kp_xy * err_n, -v_max_xy, v_max_xy))
    radius = float(np.hypot(err_e, err_n))
    alt = max(float(alt_above_marker), 0.0)
    centred = alt >= cone_k * radius
    vz = -float(np.clip(kp_z * alt, vz_min, vz_max)) if centred else 0.0
    return DescentCommand(vx, vy, vz, centred)


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes telemetry BEST_EFFORT; a RELIABLE subscriber gets nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class NaiveFlightNode(Node):
    def __init__(self):
        super().__init__('naive_flight_node')
        self._declare()
        self._read_params()

        self.phase = Phase.PRECHECK
        self.state: State | None = None
        self.pose: PoseStamped | None = None
        self.ext: ExtendedState | None = None
        self.batt: BatteryState | None = None
        # The marker input that flew, imported rather than reimplemented.
        self.mk = MarkerTracker(map_frame=self.map_frame,
                                timeout_s=self.marker_timeout,
                                acquire_frames=self.acquire_frames)
        self._t_phase = self._now()
        self._t_prestream: float | None = None
        self._t_hover: float | None = None
        self._t_touch: float | None = None
        self._t_calls: dict[str, float] = {}
        self._ticks = 0
        self._announced = ''
        self._prompted = ''
        self._checks_logged = False
        self._waived: set[str] = set()
        self._warned: set[str] = set()
        self.ekf = EstimatorHealth(speed_acc_max=self.speed_acc_max)
        # The ground the vehicle armed on, in EKF local z — see `_takeoff_target`.
        self._z_ground: float | None = None
        # The heading to hold for the whole flight, captured at the same moment
        # and for the same reason — see `_send`.
        self._yaw_hold: float | None = None

        self.create_subscription(State, '/mavros/state', self._on_state,
                                 _sensor_qos())
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self._on_pose, _sensor_qos())
        self.create_subscription(ExtendedState, '/mavros/extended_state',
                                 self._on_ext, _sensor_qos())
        self.create_subscription(BatteryState, '/mavros/battery',
                                 self._on_batt, _sensor_qos())
        # Whether the EKF is actually being aided. A pose alone does not say so
        # — see estimator.py.
        self.create_subscription(EstimatorStatus, '/mavros/estimator_status',
                                 self._on_est, _sensor_qos())
        self.create_subscription(GPSRAW, '/mavros/gpsstatus/gps1/raw',
                                 self._on_gps, _sensor_qos())
        # BEST_EFFORT to match the detector — see the module docstring. A
        # RELIABLE subscriber here is silently INCOMPATIBLE and receives nothing
        # while `ros2 topic hz` shows the detector publishing perfectly.
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

        # A daemon thread blocks on stdin so the control loop never does — a node
        # that stopped publishing setpoints to wait for a keystroke would drop
        # out of offboard.
        self._stdin_ok = self.interactive and sys.stdin is not None \
            and sys.stdin.isatty()
        if self._stdin_ok:
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'naive_flight_node: PX4 mode={self.mode_name} | '
            f'takeoff {self.takeoff_alt:.1f} m | hover {self.hover_s:.0f} s | '
            + (f'then descend on {self.marker_pose_topic}' if self.use_marker
               else 'then LAND in place (use_marker is OFF)'))
        if self.skip_preflight:
            self.get_logger().warn(
                'skip_preflight IS ON — link, battery, armed-state and marker '
                'checks are waived; only local position still gates the ARM '
                'prompt')
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
        p('takeoff_alt_m', 5.0)             # target altitude above takeoff point
        p('alt_tolerance_m', 0.3)           # counts as "reached" within this
        p('climb_speed_m_s', 0.7)           # TAKEOFF climb cap
        p('hover_s', 5.0)                   # settle at altitude this long first
        # --- marker. Off = the plain climb/hold/land flight this node began as.
        p('use_marker', True)
        # The frame a marker pose is ALREADY in. Anything else on that topic is
        # taken to be the camera optical frame and converted — see marker.py.
        p('map_frame', 'map')
        p('marker_pose_topic', '/perception/down/marker_pose')
        p('marker_detected_topic', '/perception/down/aruco_detected')
        p('marker_timeout_s', 1.5)          # older than this is not a fix
        p('marker_acquire_frames', 5)       # consecutive fixes before committing
        p('marker_lost_abort_s', 5.0)       # gone this long in DESCEND -> land
        p('search_timeout_s', 60.0)         # no marker in SEARCH -> land in place
        # --- descent. The naive controller, `descent_velocity` above.
        p('descend_kp_xy', 0.6)             # 1 m off centre -> 0.6 m/s toward it
        p('descend_v_max_xy_m_s', 0.8)      # matches the MPC node's lowered v_max
        p('descend_kp_z', 0.35)             # ease off near the deck
        p('descend_vz_max_m_s', 0.35)       # the MPC node's descent rate
        p('descend_vz_min_m_s', 0.1)        # ...but never stall above the marker
        # Descent corridor: come down only while alt >= cone_k * |xy error|.
        # 1/tan(vfov/2) for the down camera, and the same 1.6 the MPC enforces —
        # raise it to descend more conservatively, lower it only with a wider lens.
        p('descend_cone_k', 1.6)
        # --- touchdown gate: how close counts as 'on the marker'.
        p('touchdown_xy_m', 0.35)
        p('touchdown_alt_m', 0.25)          # below this, hand over to LAND
        p('touchdown_dwell_s', 1.5)         # ...held for this long
        # --- preflight thresholds
        p('min_battery_v', 14.0)            # 4S nominal; raise for 6S
        p('require_battery', True)          # false only for bench tests
        # Refuse the arm prompt while the EKF has no position aiding. Match this
        # to the vehicle's EKF2_REQ_SACC so the message quotes the real limit —
        # the default is the RAISED 1.0 m/s, not PX4's 0.5 (see estimator.py),
        # so a vehicle still at 0.5 will refuse arms this check waved through.
        p('require_gnss_aiding', True)
        p('gps_speed_acc_max_m_s', DEFAULT_SPEED_ACC_MAX)
        # Waive the preflight checks the operator is allowed to overrule (link,
        # battery, already-armed, marker pipeline). NOT local position — see
        # `_preflight_ok`.
        p('skip_preflight', False)
        # --- flight controller. PX4 assumed (OFFBOARD); ArduPilot works too.
        p('offboard_mode', 'OFFBOARD')
        # PX4 wants a steady setpoint stream before it will grant OFFBOARD, not a
        # single message; 1 s at rate_hz is ~20 of them.
        p('offboard_prestream_s', 1.0)
        p('rate_hz', 20.0)
        # Prompt on the terminal at the ARM gate. Falls back to the service when
        # stdin is not a terminal (e.g. under `ros2 launch`).
        p('interactive_approval', True)

    def _read_params(self) -> None:
        g = self.get_parameter
        self.takeoff_alt = float(g('takeoff_alt_m').value)
        self.alt_tol = float(g('alt_tolerance_m').value)
        self.climb_speed = float(g('climb_speed_m_s').value)
        self.hover_s = float(g('hover_s').value)
        self.use_marker = bool(g('use_marker').value)
        self.map_frame = str(g('map_frame').value)
        self.marker_pose_topic = str(g('marker_pose_topic').value)
        self.marker_detected_topic = str(g('marker_detected_topic').value)
        self.marker_timeout = float(g('marker_timeout_s').value)
        self.acquire_frames = int(g('marker_acquire_frames').value)
        self.marker_lost_abort = float(g('marker_lost_abort_s').value)
        self.search_timeout = float(g('search_timeout_s').value)
        self.kp_xy = float(g('descend_kp_xy').value)
        self.v_max_xy = float(g('descend_v_max_xy_m_s').value)
        self.kp_z = float(g('descend_kp_z').value)
        self.vz_max = float(g('descend_vz_max_m_s').value)
        self.vz_min = float(g('descend_vz_min_m_s').value)
        self.cone_k = float(g('descend_cone_k').value)
        self.touch_xy = float(g('touchdown_xy_m').value)
        self.touch_alt = float(g('touchdown_alt_m').value)
        self.touch_dwell = float(g('touchdown_dwell_s').value)
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

    def _on_detected(self, m): self.mk.on_detected(m.data)

    def _on_marker(self, m: PoseStamped):
        """Hand the fix to the tracker, with the heading it may need to convert.

        Which frame the marker is in comes off the message, not off a parameter,
        so the detector and this node can never be configured to disagree.
        """
        vehicle = yaw = None
        if self.pose is not None:
            q = self.pose.pose.orientation
            vehicle = (self.pose.pose.position.x, self.pose.pose.position.y,
                       self.pose.pose.position.z)
            yaw = enu_yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.mk.on_pose(
            (m.pose.position.x, m.pose.position.y, m.pose.position.z),
            m.header.frame_id, self._now(), vehicle, yaw or 0.0)

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
        print(f'\n{"=" * 72}\n  preflight PASSED — approve to ARM and take off '
              f'to {self.takeoff_alt:.1f} m\n{"=" * 72}\n'
              f'  proceed?  [ENTER = yes / n = abort]  ', end='', flush=True)

    # ---------------------------------------------------------------- services
    def _approve(self) -> tuple[bool, str]:
        if self.phase is not Phase.READY_TO_ARM:
            return False, (f'nothing to approve — phase is {self.phase.value}, '
                           f'which is not a gate')
        self._to(Phase.ARMING)
        return True, 'approved: arming and taking off'

    def _abort(self, reason: str) -> None:
        """Land now, from any phase. The naive stack's only emergency action."""
        if self.phase in (Phase.DONE, Phase.LAND):
            return
        self.get_logger().warn(f'abort: {reason} — landing')
        self._to(Phase.LAND)

    def _on_approve(self, _req, res):
        ok, msg = self._approve()
        res.success, res.message = ok, msg
        # SEPARATE CALL SITES, deliberately. `(info if ok else warn)(msg)`
        # reads better and crashes the node: rclpy identifies a logger call by
        # its file and LINE, and raises "Logger severity cannot be changed
        # between calls" the first time one line logs at two severities. An
        # operator who approves once too early (warn) and again at the gate
        # (info) therefore kills the mission node — in the air, which stops the
        # setpoint stream and drops PX4 out of OFFBOARD.
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(msg)
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
        """Like `_call`, but at most once per `period` s, and only if it fired.

        The LAND handler runs every tick; without this it would re-send the land
        command at rate_hz. The throttle timestamp is recorded ONLY on a call
        that actually went out, so a land command refused because the service is
        not up yet is retried on the next tick rather than blocked for a second.
        """
        if self._now() - self._t_calls.get(name, 0.0) < period:
            return
        if self._call(client, request, name):
            self._t_calls[name] = self._now()

    def _alt(self) -> float:
        return float(self.pose.pose.position.z) if self.pose else float('nan')

    def _yaw_now(self) -> float | None:
        """The vehicle's current heading in ENU, or None with no pose yet."""
        if self.pose is None:
            return None
        q = self.pose.pose.orientation
        return enu_yaw_from_quaternion(q.x, q.y, q.z, q.w)

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

    def _on_ground(self) -> bool:
        """True once the vehicle has actually settled — never a geometric guess.

        Uses extended_state (the FCU's own land detector), or an already-disarmed
        FCU, so the disarm/finish waits for real ground contact. AUTO.LAND
        normally auto-disarms on the ground; this just confirms it.
        """
        if self.state and not self.state.armed:
            return True
        return (self.ext is not None
                and self.ext.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

    def _send(self, vx: float, vy: float, vz: float) -> None:
        """Stream a velocity setpoint in the local frame.

        Velocity, not position: takeoff, hover and the descent are all
        regulation against a height or a marker, and a position setpoint would
        re-inject the estimator's drift as a command. The FORCE bit is
        deliberately NOT set — it would reinterpret the (ignored) acceleration
        fields as a force, which PX4 does not support on this path and may
        reject.

        YAW IS AN ABSOLUTE HEADING, NOT AN OFFSET. This field used to be a
        hard-coded 0.0, which does not mean "keep the current heading" — it
        commands ENU yaw 0 (due East), and the vehicle obeyed it by spinning on
        the spot before it would climb. So the heading is captured on the pad at
        arming and re-sent unchanged; before that, each setpoint carries the
        vehicle's own live heading, which is a no-op.
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
        yaw = self._yaw_hold if self._yaw_hold is not None else self._yaw_now()
        if yaw is None:
            # No pose yet: "do not rotate" is the only honest instruction when
            # we cannot say where the vehicle is pointing.
            m.type_mask = ((m.type_mask & ~PositionTarget.IGNORE_YAW_RATE)
                           | PositionTarget.IGNORE_YAW)
            m.yaw_rate = 0.0
        else:
            m.yaw = float(yaw)
        self.sp_pub.publish(m)

    # ------------------------------------------------------------- preflight
    def _preflight_ok(self) -> bool:
        """Minimal, naive: link up, EKF ready, disarmed, battery, detector alive.

        `skip_preflight` waives every check here EXCEPT local position, which is
        not a policy judgement but a physical prerequisite: TAKEOFF regulates on
        `pose.z`, so with no pose `_alt()` is NaN, the climb setpoint is NaN, and
        PX4 discards it — waiving that one would arm the vehicle and then sit
        there at zero velocity, which is worse than refusing. Everything else is
        a call the operator standing next to the airframe is allowed to make, so
        it is waived loudly (each reason is logged once) rather than silently.

        The marker check asks whether the detector is ALIVE, not whether it can
        see anything. Requiring a detection here would ground the vehicle
        whenever the marker is not visible from the pad — which is the normal
        case, and the reason SEARCH happens at altitude.
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
        if self.use_marker and not self.mk.seen:
            overridable.append(
                f'marker pipeline silent — nothing on {self.marker_detected_topic}')
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
        # Will not stop the flight, but will shape it. Said once, before the
        # gate, so it is a decision rather than a surprise.
        warn = self.ekf.warning(self._now())
        if warn and warn not in self._warned:
            self._warned.add(warn)
            self.get_logger().warn(f'preflight WARNING: {warn}')
        if not self._checks_logged:
            self._checks_logged = True
            self.get_logger().info(
                f'preflight PASSED — {self.ekf.summary(self._now())}'
                + (f' (detector alive on {self.marker_detected_topic})'
                   if self.use_marker else ''))
        return True

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._ticks += 1
        self._publish_state()

        # Keep the offboard stream alive, unconditionally, for every phase that
        # is armed or about to be — BEFORE the phase logic, so a phase that
        # returns early (the gate waiting on a human, ARMING waiting out the
        # pre-stream) cannot starve it. PX4 drops offboard after ~0.5 s of
        # silence. Phases that fly a real setpoint overwrite this later in the
        # same tick; publishing twice is harmless, a gap is not.
        if self.phase in (Phase.READY_TO_ARM, Phase.ARMING, Phase.TAKEOFF,
                          Phase.HOVER, Phase.SEARCH, Phase.DESCEND):
            self._send(0.0, 0.0, 0.0)

        if self.phase is Phase.PRECHECK:
            if self._preflight_ok():
                self._to(Phase.READY_TO_ARM)
            return

        if self.phase is Phase.READY_TO_ARM:
            self._announce()
            return

        if self.phase is Phase.ARMING:
            # ORDER MATTERS ON PX4: stream -> mode -> arm. The stream is kept up
            # above; lead with it, then request the mode, then arm only once the
            # FCU confirms it is IN the mode (arming first and switching after is
            # the ArduPilot habit; PX4 can arm in whatever mode it was in).
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
                # ...and the heading it is sitting at, for the same reason: it
                # is the one moment the vehicle is provably where the operator
                # placed it. Every setpoint from here on re-sends this, so the
                # climb goes straight up instead of yawing first (see `_send`).
                self._yaw_hold = self._yaw_now()
                heading = ('' if self._yaw_hold is None else
                           f', holding heading {np.degrees(self._yaw_hold):.0f}'
                           f' deg ENU')
                self.get_logger().info(
                    f'armed on the ground at local z={self._z_ground:.2f} m — '
                    f'climbing to z={self._takeoff_target():.2f} m '
                    f'({self.takeoff_alt:.1f} m above it){heading}')
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
                self._to(Phase.HOVER)
                return
            # Climb at a capped rate, easing off near the target so the vehicle
            # settles instead of overshooting.
            vz = float(np.clip(err, -self.climb_speed, self.climb_speed))
            self._send(0.0, 0.0, vz)
            return

        if self.phase is Phase.HOVER:
            # Settling matters more now than it did: the detector is solving a
            # marker off a moving camera, and a vehicle still swinging from the
            # climb costs detections exactly when SEARCH starts asking for them.
            self._send(0.0, 0.0, 0.0)
            if self._t_hover is None:
                self._t_hover = self._now()
            if self._now() - self._t_hover >= self.hover_s:
                self._to(Phase.SEARCH if self.use_marker else Phase.LAND)
            return

        if self.phase is Phase.SEARCH:
            # Hold and look. No search pattern on purpose — the naive flight is
            # launched over the marker, and flying a pattern is a second thing
            # to get wrong before the first one is proven.
            self._send(0.0, 0.0, 0.0)
            if self.mk.acquired(self._now()):
                self._t_touch = None
                self.get_logger().info(
                    f'marker acquired ({self.mk.streak} consecutive fixes) — '
                    f'descending automatically from here')
                self._to(Phase.DESCEND)
                return
            if self._now() - self._t_phase > self.search_timeout:
                self.get_logger().warn(
                    f'no marker within {self.search_timeout:.0f} s — landing in '
                    f'place')
                self._to(Phase.LAND)
            return

        if self.phase is Phase.DESCEND:
            self._descend()
            return

        if self.phase is Phase.LAND:
            # Hand to the autopilot's own landing, then disarm ONLY once it has
            # actually settled (extended_state), so motors are never cut in the
            # air. AUTO.LAND normally auto-disarms; the explicit disarm is a
            # gated backstop. Both are throttled: this runs every tick.
            self._call_throttled(self.land_cli, CommandTOL.Request(), 'land')
            if self._on_ground():
                req = CommandBool.Request()
                req.value = False
                self._call_throttled(self.arm_cli, req, 'disarm')
            if self.state and not self.state.armed:
                self._to(Phase.DONE)
                self.get_logger().info('disarmed — naive flight complete')
            return

    def _descend(self) -> None:
        """Centre on the marker and come down inside the corridor."""
        now = self._now()
        if not self.mk.fresh(now):
            # HOLD, do not guess. Flying the last fix while the marker is gone
            # is how a descent ends up over where the marker used to look like
            # it was; a vehicle holding still is recoverable.
            gone = self.mk.age(now)
            if gone > self.marker_lost_abort:
                self.get_logger().warn(
                    f'marker lost for {gone:.1f} s during descent — landing')
                self._to(Phase.LAND)
                return
            self._send(0.0, 0.0, 0.0)
            self.get_logger().warn(f'marker stale for {gone:.1f} s — holding',
                                   throttle_duration_sec=1.0)
            return
        if self.pose is None:
            self._send(0.0, 0.0, 0.0)
            return

        tgt = self.mk.pos
        err_e = float(tgt[0] - self.pose.pose.position.x)
        err_n = float(tgt[1] - self.pose.pose.position.y)
        # Height above the MARKER, not above the EKF's origin: with a vision fix
        # this is solvePnP's own range, so the handover to LAND happens at a real
        # distance from the deck rather than at whatever the estimator's datum
        # drifted to since takeoff. A marker already in `map` gives the same
        # quantity from the same subtraction.
        alt = float(self._alt() - tgt[2])
        radius = float(np.hypot(err_e, err_n))

        cmd = descent_velocity(err_e, err_n, alt,
                               kp_xy=self.kp_xy, v_max_xy=self.v_max_xy,
                               cone_k=self.cone_k, kp_z=self.kp_z,
                               vz_max=self.vz_max, vz_min=self.vz_min)
        self._send(cmd.vx, cmd.vy, cmd.vz)

        if alt <= self.touch_alt and radius <= self.touch_xy:
            self._t_touch = self._t_touch or now
            if now - self._t_touch >= self.touch_dwell:
                self.get_logger().info(
                    f'touchdown gate held {self.touch_dwell:.1f} s at '
                    f'{alt:.2f} m / {radius:.2f} m — handing over to LAND')
                self._to(Phase.LAND)
        else:
            self._t_touch = None

        if self._ticks % int(self.rate_hz * 2) == 0:
            self.get_logger().info(
                f'[DESCEND] alt={alt:.2f} m  xy_err={radius:.2f} m  '
                f'corridor needs h>={self.cone_k * radius:.2f} m  '
                f'vz={cmd.vz:+.2f} m/s  '
                f'{"descending" if cmd.centred else "HOLDING to centre"}')

    # ------------------------------------------------------------------ output
    def _announce(self) -> None:
        self._prompt()
        if self._stdin_ok:
            # The prompt already asked on its own clean lines; logging the same
            # thing here would print a timestamped duplicate across the line the
            # operator is typing into.
            return
        if self._announced == Phase.READY_TO_ARM.value:
            return
        self._announced = Phase.READY_TO_ARM.value
        self.get_logger().warn(
            f'>>> WAITING FOR APPROVAL — approve to ARM and take off\n'
            f'    ros2 run mpc_landing approve {self.get_name()}')

    def _publish_state(self) -> None:
        """Phase, plus what it is waiting on — this topic is how the ground
        station knows whether SEARCH is looking at anything."""
        extra = ''
        if self.phase is Phase.SEARCH:
            extra = f' | streak {self.mk.streak}/{self.acquire_frames}'
        elif self.phase is Phase.DESCEND and self.mk.pos is not None:
            extra = f' | marker age {self.mk.age(self._now()):.1f} s'
        self.state_pub.publish(String(data=f'{self.phase.value}{extra}'))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NaiveFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
