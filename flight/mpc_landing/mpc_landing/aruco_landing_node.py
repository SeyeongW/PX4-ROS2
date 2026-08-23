"""aruco_landing_node — the plainest ArUco precision landing over MAVROS.

The next rung up from `naive_flight_node`.  Naive proved the boring half on a
real airframe (MAVROS wired, PX4 accepts OFFBOARD, the vehicle climbs, holds and
lands).  This node keeps that exact skeleton and adds ONE thing: it descends
onto an ArUco marker instead of landing where it took off.

    PRECHECK ─approve─► ARM ─► TAKEOFF ─► READY
                                            │ automatic
                                            ▼
                MISSION_PLAN ─► MISSION ─► map (50,50) HOVER
                                                   │ automatic
                                                   ▼
        DONE ◄─ LAND ◄─ P DESCEND ◄─ SEARCH ◄─ RETURN ◄─ RETURN_PLAN

ONE APPROVAL PER FLIGHT
-----------------------
Only the ARM is a gate.  Everything after it — the climb, the fixed-goal leg,
the hold at the goal, the return to the trailer, the search and the descent —
runs on its own, because once the operator has authorised the flight the profile
is already decided and there is nothing further for them to choose.  Pausing
mid-mission does not add a decision; it adds an armed airframe with live props
waiting for a keystroke, which is the state you least want to extend.

`abort` still lands the vehicle from any phase, and every leg fails FORWARD: a
route that cannot be planned or reached hands on to the trailer return, and a
return that cannot be planned lands where it is.  No phase waits for a human to
notice it is stuck.

FIXED GOAL FIRST, THEN THE TRAILER
----------------------------------
With `cruise_to_trailer` the staged mission first flies to the configured CJU
map goal and settles there. Once it has, the return begins on its own:
`trailer_target_node` turns the trailer's radioed lat/lon into a point in this
vehicle's own local ENU frame (`/trailer/target_local`) and RETURN drives to it.
GNSS gets the trailer INTO THE CAMERA'S FRAME
and nothing more — two receivers disagree by a metre or two, and the trailer's
antenna is not its marker — so the moment the marker is acquired, vision takes
over completely and the coordinate is never used again.  That handover is the
whole design: coarse where coarse is enough, vision where it is not.

`planned_cruise` is the hardware integration of Wang's route mission. It runs
the exact-checked A* -> SFC -> B-spline geometry with Wang's TrackingMPC in both
MISSION and RETURN, then hands marker centring and descent to the existing
proportional ArUco controller. All PX4 I/O remains on this node's MAVROS
publisher.

SEARCH keeps station over the coordinate rather than over a fixed patch of
ground, so a trailer that rolls a few metres while the detector is settling stays
under the camera — and it descends to `search_alt_m` while doing it, because the
altitude a marker is DETECTABLE at is lower than the altitude that sees the most
ground (the table on that parameter has the numbers).

SEARCHING WITH THE GIMBAL, NOT WITH THE VEHICLE
-----------------------------------------------
A camera locked at nadir sees one circle of ground, and if the marker is not in
it the mission stares at the wrong grass until it times out.  The obvious answer
is to fly the vehicle in a widening spiral; the better one, on an airframe that
already carries a 3-axis gimbal, is to point the CAMERA around and leave the
vehicle where it is — low over a trailer with people near it, vehicle motion is
the expensive part.  So SEARCH sweeps: nadir first, then rings at progressively
shallower pitch, dwelling at each look for `scan_view_s` of SETTLED time
(`marker.GimbalSweep` owns the pattern and the timing rules).

That makes the gimbal angle load-bearing.  Because the angle is KNOWN at the
moment of the sighting, a marker spotted 40 deg off to one side is a position
fix and not merely a sighting — solvePnP gives the range along the line, the
gimbal gives the line (`marker_enu_from_gimbal_camera`).  Two consequences that
are not optional:

    * fixes taken while the gimbal is still slewing are DROPPED, because off
      nadir an angle error is multiplied by the slant range, not the height;
    * once acquired the camera is LEFT WHERE IT IS, and re-aimed
      (`gimbal_aim_for`) only if the marker is actually lost.  Pointing the
      camera at a marker costs pixels — off nadir the range becomes the slant
      range and the marker is foreshortened — and with a marker near the
      detector's floor that is enough to lose one the nadir view could read
      perfectly well.  `_track_marker` has the measured numbers.

Both the marker-only mission and final `run_px4 trailer` mission use the same
proven proportional centre-and-descend below.

    horizontal:  v_xy = clip(kp * (marker_xy - vehicle_xy),  |v| <= center_v_max)
    vertical:    descend at descend_speed ONLY while centred (radius <= r),
                 otherwise hold altitude and centre first
    handover:    height above marker <= touchdown_alt AND centred -> autopilot LAND

The MARKER comes from `aruco_landing`'s `aruco_pose_node`, which already
publishes the two topics below.  It solves the marker in the camera's optical
frame and, with `landing_tf_node` running, republishes it in `map` — the frame
MAVROS local position is in — so the offset is a plain subtraction.  If the pose
still arrives in the camera frame (no tf chain), it is converted here against the
gimbal's MEASURED angle, exactly as `mpc_landing_node` does — which is what makes
a sighting from a swept look a usable position fix.

The MAVROS discipline is lifted verbatim from `naive_flight_node`: BEST_EFFORT
sensor QoS, stream→mode→arm order, keeping the stream alive through the gate,
ground-gated disarm, and confirming every change from telemetry.

    ros2 run mpc_landing aruco_landing_node          # ENTER approves the arm

Under `ros2 launch` stdin is not a terminal, so approve over the service — or
publish ABORT on ~/command, which lands from any phase:

    ros2 run mpc_landing approve aruco_landing_node
    ros2 run mpc_landing abort   aruco_landing_node  # land now, from any phase
    ros2 topic pub --once /aruco_landing_node/command std_msgs/msg/String "{data: ABORT}"

Interfaces
----------
Subscribes
    /mavros/state                          mavros_msgs/State
    /mavros/local_position/pose            geometry_msgs/PoseStamped
    /mavros/local_position/velocity_local  geometry_msgs/TwistStamped (route)
    /mavros/extended_state                 mavros_msgs/ExtendedState
    /mavros/battery                        sensor_msgs/BatteryState
    /mavros/estimator_status               mavros_msgs/EstimatorStatus
    /mavros/gpsstatus/gps1/raw             mavros_msgs/GPSRAW
    /mavros/global_position/global          sensor_msgs/NavSatFix
                                           (only with planned_cruise)
    /perception/down/marker_pose           geometry_msgs/PoseStamped
    /perception/down/aruco_detected        std_msgs/Bool
    /trailer/target_local                  geometry_msgs/PointStamped
                                           (only with cruise_to_trailer)
    /siyi_gimbal_node/attitude             geometry_msgs/Vector3Stamped
    ~/command                              std_msgs/String
Publishes
    /mavros/setpoint_raw/local             mavros_msgs/PositionTarget
    /siyi_gimbal_node/aim                  geometry_msgs/Vector3Stamped
    ~/state                                std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land
    /mavros/param/get_parameters           the airframe's own speed and
                                           acceleration limits (see `_declare`)

ALL PARAMETERS ARE DECLARED HERE, IN `_declare`, WITH THEIR VALUES.  Override for
a one-off:

    ros2 run mpc_landing aruco_landing_node --ros-args -p descend_speed_m_s:=0.25
"""

from __future__ import annotations

import math
import multiprocessing
import queue
import signal
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import (PointStamped, PoseStamped, TwistStamped,
                               Vector3Stamped)
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import BatteryState, NavSatFix, NavSatStatus
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

from mavros_msgs.msg import (EstimatorStatus, ExtendedState, GPSRAW,
                             PositionTarget, State)
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

from .estimator import DEFAULT_SPEED_ACC_MAX, EstimatorHealth
# The geometry and the sweep both live in `marker`, which is what every mission
# node in this package imports — a second copy of either is a second thing to be
# wrong in only one place.
from .marker import (GimbalSweep, VelocityEstimate, enu_yaw_from_quaternion,
                     gimbal_aim_for, marker_enu_from_gimbal_camera)


def _planner_worker_init() -> None:
    """Keep Ctrl-C in the ROS parent; it owns orderly worker shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _header_stamp_seconds(message) -> float:
    """ROS source timestamp as seconds, or NaN when it cannot be trusted."""
    try:
        sec = int(message.header.stamp.sec)
        nanosec = int(message.header.stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return float('nan')
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        return float('nan')
    stamp = sec + nanosec * 1.0e-9
    return stamp if stamp > 0.0 and math.isfinite(stamp) else float('nan')


def _plan_route_worker(map_yaml, start_xy, goal_xy, site_origin_local_xy):
    """Spawn-safe import boundary: the child never inherits ROS/DDS objects."""
    from path_plan.cju_route import plan_route
    return plan_route(map_yaml, start_xy, goal_xy, site_origin_local_xy)


class Phase(str, Enum):
    PRECHECK = 'PRECHECK'          # running preflight checks
    READY_TO_ARM = 'READY_TO_ARM'  # checks passed, waiting for approval
    ARMING = 'ARMING'              # stream -> OFFBOARD -> arm
    TAKEOFF = 'TAKEOFF'            # climbing to takeoff_alt
    READY = 'READY'                # hover at altitude, before the fixed leg
    MISSION_PLAN = 'MISSION_PLAN'  # hold while the fixed-goal route is built
    MISSION = 'MISSION'            # TrackingMPC to the fixed CJU map goal
    HOVER = 'HOVER'                # settled at the map goal, before RETURN
    RETURN_PLAN = 'RETURN_PLAN'    # hold while the first trailer route is built
    RETURN = 'RETURN'              # TrackingMPC to the live trailer coordinate
    CRUISE = 'CRUISE'              # legacy direct-to-trailer launch mode
    SEARCH = 'SEARCH'              # holding over it, looking for the marker
    DESCEND = 'DESCEND'            # centring on the marker and coming down
    LAND = 'LAND'                  # handed to the autopilot's LAND, disarming
    DONE = 'DONE'


def _capped(v: np.ndarray, limit: float) -> np.ndarray:
    """Scale a 2-vector down to `limit` if it is longer, keeping its direction.

    Capping the VECTOR rather than each axis: clipping x and y separately turns a
    diagonal command into a different heading, which is how a vehicle ends up
    flying past the corner of a pad it was aimed at.
    """
    n = float(np.linalg.norm(v))
    if n <= limit or n == 0.0:
        return v
    return v * (limit / n)


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
        self._marker_seq = 0
        self._acq_last_marker_seq = 0
        # How fast the marker itself is moving — the trailer's deck does not
        # hold still, and a P controller alone cannot centre on something that
        # keeps going (see VelocityEstimate).
        self.marker_vel = VelocityEstimate(tau_s=self.marker_vel_tau,
                                           max_speed=self.marker_vel_max,
                                           gap_s=self.marker_timeout)
        # The trailer's coordinate, already converted into this vehicle's local
        # ENU frame by trailer_target_node — never a lat/lon here.
        self.target: np.ndarray | None = None
        self.target_t = 0.0
        self.target_sample_t = float('nan')
        self._target_seen = False
        self.pose_t = float('nan')
        self.pose_rx_t = 0.0
        self.velocity: TwistStamped | None = None
        self.velocity_t = float('nan')
        self.velocity_rx_t = 0.0
        self.vehicle_fix: NavSatFix | None = None
        self.vehicle_fix_t = float('nan')
        self.vehicle_fix_rx_t = 0.0

        # The planner is opt-in and owns geometry only. One spawned worker keeps
        # CPU-bound A*/SciPy work away from this node's MAVROS heartbeat.
        # Route state is committed as one tuple in `_route_update`, never from a
        # Future callback thread.
        self._planner_pool: ProcessPoolExecutor | None = None
        self._plan_future = None
        self._route_pending = None
        self._route_active = None
        self._route_progress = 0.0
        self._route_request_seq = 0
        self._route_last_request_t = float('-inf')
        self._route_last_error = ''
        self._route_map_info = None
        self._route_map_identity = None
        self._route_lib = None
        self._enu_offset = None
        self._route_observed_origin = None
        self._route_observed_origin_t = float('nan')
        self._route_observed_origin_rx_t = 0.0
        self._path_mpc = None
        self._path_reference = None
        self._path_solve_t = None
        self._path_last_solve_t = None
        self._last_mpc_acceleration = np.zeros(3)
        self._last_mpc_acceleration_t = None
        self._target_velocity = VelocityEstimate(
            tau_s=0.3, max_speed=self.marker_vel_max,
            gap_s=self.target_timeout)
        self._target_velocity_sample_t = float('-inf')
        if self.planned_cruise:
            from landing_mpc.reference import HorizonReference
            from path_plan.mpc import TrackingMPC
            from path_plan.mpc_reference import (
                limit_acceleration_slew, path_reference_horizon)
            import path_plan.cju_route as route_lib
            from trailer_link.geodesy import enu_offset

            self._route_lib = route_lib
            self._enu_offset = enu_offset
            self._path_reference_horizon = path_reference_horizon
            self._limit_acceleration_slew = limit_acceleration_slew
            self._tracking_mpc_cls = TrackingMPC
            # BUILT LATER, on the limits PX4 reports. Until then it stays None,
            # `_route_mpc_command` reports "not commanded" and the mission holds
            # — and the arm gate refuses, so nothing flies that far.
            self._build_path_mpc()
            self._path_reference = HorizonReference(lead_s=0.1)
            route_path = Path(self.route_map_yaml).expanduser().resolve(
                strict=True)
            self.route_map_yaml = str(route_path)
            route_stat = route_path.stat()
            self._route_map_identity = (
                route_stat.st_dev, route_stat.st_ino,
                route_stat.st_size, route_stat.st_mtime_ns)
            self._route_map_info = route_lib.route_map_info(
                self.route_map_yaml)
            self._route_rotation = route_lib.rotation_for_heading(
                self._route_map_info.heading_deg_enu)
            self._planner_pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_planner_worker_init)
        # The horizontal velocity actually being commanded, so every change to it
        # can be rate-limited (`_slew`). Guidance asks for a velocity; this is
        # what the vehicle is given.
        self._v_cmd = np.zeros(2)
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
        # Subscribed even when cruise is off, so `ros2 topic echo` is not the
        # only way to see whether the trailer link is alive before a flight.
        self.create_subscription(PointStamped, self.target_topic,
                                 self._on_target, _sensor_qos())
        if self.planned_cruise:
            self.create_subscription(
                TwistStamped, '/mavros/local_position/velocity_local',
                self._on_velocity, _sensor_qos())
            self.create_subscription(
                NavSatFix, self.route_vehicle_fix_topic,
                self._on_vehicle_fix, _sensor_qos())
        # Where the camera actually IS. Fixes are dropped while this disagrees
        # with what was commanded (`GimbalSweep.settled`).
        self.create_subscription(Vector3Stamped, self.gimbal_attitude_topic,
                                 self._on_gimbal, 10)

        self.sp_pub = self.create_publisher(PositionTarget,
                                            '/mavros/setpoint_raw/local', 10)
        self.state_pub = self.create_publisher(String, '~/state', 10)
        self.aim_pub = self.create_publisher(Vector3Stamped,
                                             self.gimbal_aim_topic, 10)

        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.land_cli = self.create_client(CommandTOL, '/mavros/cmd/land')
        # MAVROS republishes the FCU's parameter table as ROS parameters, so
        # reading PX4's limits is a plain get_parameters call.
        self.fcu_param_cli = self.create_client(
            GetParameters, '/mavros/param/get_parameters')
        self._fcu_limits_req = False

        self.create_service(Trigger, '~/approve', self._on_approve)
        self.create_service(Trigger, '~/abort', self._on_abort)
        self.create_subscription(String, '~/command', self._on_command, 10)

        self._stdin_commands = queue.SimpleQueue()
        self._stdin_ok = self.interactive and sys.stdin is not None \
            and sys.stdin.isatty()
        if self._stdin_ok:
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self.create_timer(1.0 / self.rate_hz, self._tick)
        route = (f'A*/SFC/B-spline + TrackingMPC: map '
                 f'{self._route_map_info.mission_goal_xy} then '
                 f'{self.target_topic} | ' if self.planned_cruise else '')
        self.get_logger().info(
            f'aruco_landing_node: PX4 mode={self.mode_name} | '
            f'takeoff/transit {self.takeoff_alt:.1f} m, search '
            f'{self.search_alt:.1f} m | {route}descend on '
            f'{self.marker_pose_topic}')
        if self.planned_cruise:
            info = self._route_map_info
            self.get_logger().info(
                f'route map: {info.name} | origin '
                f'{info.origin_lat:.9f},{info.origin_lon:.9f} | heading '
                f'{info.heading_deg_enu:.3f} deg ENU | vehicle clearance '
                f'{info.vehicle_clearance_m:.2f} m | '
                'worker=spawn')
            if not info.hardware_flight_approved:
                self.get_logger().warn(
                    'route map is NOT hardware-flight-approved: '
                    f'{info.horizontal_accuracy}')
        if self.skip_preflight:
            self.get_logger().warn(
                'skip_preflight IS ON — link, battery and armed-state checks '
                'are waived; only local position still gates the ARM prompt')
        if self._stdin_ok:
            self.get_logger().info(
                'ARM will ask on this terminal — ENTER approves, n aborts; '
                'everything after the arm runs on its own')
        else:
            self.get_logger().info(
                f'stdin is not a terminal, so approve over the service: '
                f'ros2 run mpc_landing approve {self.get_name()}')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these numbers may be set."""
        p = self.declare_parameter
        # --- mission geometry
        p('takeoff_alt_m', 5.0)             # takeoff AND transit altitude
        # SEARCH settles here instead, and the difference is not cosmetic. The
        # camera trades footprint against resolution, and BOTH ends bite:
        #
        #   h     marker (0.18 m) on a 719 px focal   footprint radius
        #   3 m   43 px  — detects reliably           1.5 m
        #   5 m   26 px  — marginal                   2.5 m
        #   8 m   16 px  — 2.7 px per 4x4 cell: no    4.0 m
        #
        # So climbing to see more ground is self-defeating with this marker: at
        # 8 m the marker is in frame and undetectable. 3 m is where a 0.18 m
        # marker is solid. The rule for this camera is `radius ~= 9 x marker
        # edge`, so a bigger marker is what buys a wider search, not altitude.
        #
        # Transit stays at `takeoff_alt_m`: 3 m is a search height, not a height
        # to cross a field at. SEARCH descends to this on arrival, and the marker
        # usually resolves on the way down rather than at the bottom.
        p('search_alt_m', 3.0)
        p('alt_tolerance_m', 0.3)           # counts as "reached" within this
        p('climb_speed_m_s', 0.7)           # TAKEOFF climb cap
        # Descending into SEARCH is done blind, over a trailer, so it is capped
        # well below the climb rate rather than sharing it.
        p('search_descend_speed_m_s', 0.4)
        # --- marker input (aruco_pose_node's default output topics)
        p('map_frame', 'map')
        p('marker_pose_topic', '/perception/down/marker_pose')
        p('marker_detected_topic', '/perception/down/aruco_detected')
        p('marker_timeout_s', 1.5)          # older than this is not a fix
        p('marker_lost_abort_s', 5.0)       # gone this long mid-descent -> LAND
        # No marker in SEARCH -> LAND. RAISED from 60 s for the sweep: what
        # matters is how many COMPLETE sweeps fit before giving up, and one is
        # not enough — the marker can sit on the boundary between two looks and
        # be missed by a lap that was otherwise fine. An upper bound on
        # hovering, not a plan: the mission commits the moment it sees the marker.
        p('search_timeout_s', 90.0)
        # SEARCH -> DESCEND is automatic and irreversible, so require the marker
        # to be seen this many CONSECUTIVE ticks before committing — one frame is
        # enough for a false positive to start a descent.
        p('marker_acquire_frames', 5)
        # --- cruise to the trailer's coordinate (off by default: without a
        # trailer link this node is exactly the marker-landing mission it was)
        p('cruise_to_trailer', False)
        p('trailer_target_topic', '/trailer/target_local')
        p('trailer_target_timeout_s', 3.0)   # older than this is not a target
        # Gentle by design: this is a real airframe flying at a coordinate it
        # cannot see. kp * v_max puts full speed at v_max/kp of error, and the
        # acceleration limit means no setpoint step the vehicle has to snatch at.
        p('cruise_kp', 0.35)                 # horizontal P gain [1/s]
        # --- how fast and how hard: PX4's numbers, not ours -------------------
        # THE AIRFRAME'S LIMITS ARE NOT WRITTEN IN THIS FILE. They are already
        # configured on the flight controller, where they bound the RC pilot and
        # every autopilot mode; a second copy here is a limit that can disagree
        # with the one actually enforcing, and the disagreement shows up in the
        # air. So the node READS them from PX4 over MAVROS and refuses to offer
        # the arm until it has (`_sync_limits_from_fcu`).
        #
        # THIS MOVES A DECISION ONTO THE VEHICLE. PX4 ships MPC_XY_VEL_MAX at
        # 12 m/s, which is not a speed to cross a field at toward a trailer with
        # people near it. Set it on the FCU to the speed you actually want —
        # that is now the only place it is set, for offboard and for the pilot
        # alike.
        #
        # A positive value here overrides the fetch for one run. It is an escape
        # hatch for a bench, not a default: 0 means "ask PX4".
        p('cruise_v_max_m_s', 0.0)
        p('cruise_accel_m_s2', 0.0)          # also slews the commanded velocity
        p('cruise_jerk_m_s3', 0.0)           # the tracking MPC's jerk bound
        p('fcu_speed_param', 'MPC_XY_VEL_MAX')
        p('fcu_accel_param', 'MPC_ACC_HOR')
        p('fcu_jerk_param', 'MPC_JERK_AUTO')
        # Within this -> start searching. Tight on purpose: the search footprint
        # at `search_alt_m` is only ~1.5 m in its short axis with the current
        # marker, so arriving loosely is arriving with the marker out of frame.
        p('cruise_arrive_m', 1.0)
        # State-transition tolerance only: it does not reduce path speed. READY
        # and fixed-goal HOVER open after the vehicle has actually settled.
        p('arrival_speed_tolerance_m_s', 0.2)
        # The leash. This node REFUSES to fly farther than this, at the gate and
        # in the air — a bad fix must not become a long flight. It is the
        # mission's own limit, independent of trailer_target_node's sanity check.
        p('cruise_max_distance_m', 150.0)
        p('cruise_timeout_s', 180.0)         # never arrived -> land where we are
        # How often the transit prints drone / trailer / range. A long cruise is
        # slow and uneventful, so 3 s is enough to watch it close without
        # burying the phase transitions in a scrolling wall.
        p('cruise_log_period_s', 3.0)
        # Target gone this long mid-cruise: stop chasing a stale coordinate and
        # start looking with the camera from wherever we got to.
        p('trailer_lost_search_s', 10.0)
        # --- obstacle-aware trailer cruise (run_px4 trailer) ----------------
        # Wang geometry and TrackingMPC run here; ArUco P descent, PX4 I/O and
        # final LAND stay on the hardware-proven MAVROS path.
        p('planned_cruise', False)
        p('route_map_yaml', '')
        p('route_vehicle_fix_topic', '/mavros/global_position/global')
        p('route_gps_timeout_s', 3.0)
        # Local pose/velocity drive route health and MPC collision checks, so they
        # cannot inherit the much slower GPS freshness allowance.
        p('route_state_timeout_s', 0.2)
        # Reject an unusable absolute fix without reinterpreting the geometric
        # vehicle clearance as a second localization or speed budget.
        p('route_max_horizontal_accuracy_m', 0.4)
        # Route anchoring uses one local-pose/global-fix pair. Refuse a pair
        # whose callback times are too far apart while the vehicle is moving.
        p('route_pose_fix_sync_s', 0.10)
        # Exceeding this invalidates the old anchor and forces a fresh plan.
        p('route_anchor_drift_m', 0.20)
        # A valid synchronized pose/fix pair may be reused between 5 Hz global
        # fix updates; the pair itself must still meet the 0.10 s source skew.
        p('route_anchor_timeout_s', 0.3)
        p('route_timeout_s', 300.0)
        p('route_replan_period_s', 2.0)
        p('route_lookahead_m', 6.0)
        p('route_cross_track_m', 0.25)
        # The checked-in map is an OSM/simulation snapshot. run_px4 exposes an
        # explicit override for props-off/SITL work; a props-on map should set
        # hardware_flight_approved in the YAML after field measurement.
        p('allow_unapproved_route_map', False)
        # --- gimbal search
        # The camera is on a 3-axis gimbal, so SEARCH points the CAMERA around
        # instead of flying the vehicle around: a nadir-only look sees one ~2.5 m
        # circle, and if the marker is not in it the mission stares at the wrong
        # grass until it times out. Sweeping costs no vehicle motion at all,
        # which low over a trailer is the whole argument. Angles, dwell rules and
        # the reasons for each number live in `marker.GimbalSweep`.
        p('gimbal_aim_topic', '/siyi_gimbal_node/aim')
        p('gimbal_attitude_topic', '/siyi_gimbal_node/attitude')
        p('gimbal_scan', True)
        # ONLY THE RINGS THIS MARKER CAN ACTUALLY BE READ FROM. A look off nadir
        # pays twice — the range becomes the slant range (h/sin) and the marker
        # is foreshortened (x sin) — so the marker's apparent size falls as
        # sin^2 of the elevation: `fx * edge * sin^2(elev) / h` pixels.
        #
        #   h      nadir   -60 ring   -40 ring        (0.18 m marker, fx 719,
        #   5 m    25.9*    19.4       10.7            floor ~25 px to decode)
        #   3 m    43.1*    32.3*      17.8
        #
        # mpc_landing_node's third ring at -40 deg is therefore SEVEN of its
        # fifteen looks spent staring at ground this marker cannot be read from,
        # at any altitude the mission flies. Dropping it halves the sweep, which
        # buys complete sweeps inside `search_timeout_s` instead of coverage
        # that was never real. Put it back when the marker gets bigger: at
        # 0.45 m the -40 ring clears the floor at 5 m and reaches 6 m out.
        p('scan_pitch_deg', [-90.0, -60.0])
        p('scan_yaw_step_deg', 45.0)
        p('scan_yaw_limit_deg', 135.0)
        p('scan_view_s', 1.0)
        p('scan_look_max_s', 4.0)
        p('scan_settle_s', 0.5)
        p('gimbal_settled_deg', 6.0)
        p('gimbal_attitude_timeout_s', 2.0)
        # Keep the camera on the marker while flying to it. Without this a
        # marker found 40 deg off to one side leaves the frame the moment the
        # gimbal returns to nadir, and the descent aborts on a lost marker
        # before the vehicle has covered any ground.
        p('gimbal_track', True)
        # --- descent control (plain proportional centre-and-descend)
        p('center_kp', 0.8)                 # horizontal P gain [1/s]
        p('center_v_max_m_s', 0.6)          # horizontal speed cap while centring
        # Only sink while the marker is within this horizontal radius; outside it
        # the vehicle holds altitude and centres first, so it never descends off
        # to the side of the pad.
        p('descend_radius_m', 0.30)
        p('descend_speed_m_s', 0.30)        # capped sink rate when centred
        # Feed the MARKER's own velocity forward while centring, so a deck that
        # is still rolling does not sit permanently at the P controller's
        # steady-state lag (v/kp) and block the descent. Set the max to 0.0 to
        # turn it off and fly the plain proportional descent.
        p('marker_vel_tau_s', 0.5)          # low-pass on a differenced position
        p('marker_vel_max_m_s', 1.0)
        # Kept at 1 s, unlike the cruise: the descent is the part where the
        # numbers change fast and where a log has to be dense enough to explain
        # a landing afterwards.
        p('descend_log_period_s', 1.0)
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
        self.search_alt = float(g('search_alt_m').value)
        self.alt_tol = float(g('alt_tolerance_m').value)
        self.climb_speed = float(g('climb_speed_m_s').value)
        self.search_descend = float(g('search_descend_speed_m_s').value)
        self.map_frame = str(g('map_frame').value)
        self.marker_pose_topic = str(g('marker_pose_topic').value)
        self.marker_detected_topic = str(g('marker_detected_topic').value)
        self.marker_timeout = float(g('marker_timeout_s').value)
        self.marker_lost_abort = float(g('marker_lost_abort_s').value)
        self.search_timeout = float(g('search_timeout_s').value)
        self.acquire_frames = int(g('marker_acquire_frames').value)
        self.cruise = bool(g('cruise_to_trailer').value)
        self.target_topic = str(g('trailer_target_topic').value)
        self.target_timeout = float(g('trailer_target_timeout_s').value)
        self.cruise_kp = float(g('cruise_kp').value)
        self.cruise_v_max = float(g('cruise_v_max_m_s').value)
        self.cruise_accel = float(g('cruise_accel_m_s2').value)
        self.cruise_jerk = float(g('cruise_jerk_m_s3').value)
        self.fcu_speed_param = str(g('fcu_speed_param').value)
        self.fcu_accel_param = str(g('fcu_accel_param').value)
        self.fcu_jerk_param = str(g('fcu_jerk_param').value)
        self.cruise_arrive = float(g('cruise_arrive_m').value)
        self.arrival_speed = float(
            g('arrival_speed_tolerance_m_s').value)
        self.cruise_max_dist = float(g('cruise_max_distance_m').value)
        self.cruise_timeout = float(g('cruise_timeout_s').value)
        self.cruise_log_period = float(g('cruise_log_period_s').value)
        self.trailer_lost_search = float(g('trailer_lost_search_s').value)
        self.planned_cruise = bool(g('planned_cruise').value)
        self.route_map_yaml = str(g('route_map_yaml').value)
        self.route_vehicle_fix_topic = str(
            g('route_vehicle_fix_topic').value)
        self.route_gps_timeout = float(g('route_gps_timeout_s').value)
        self.route_state_timeout = float(
            g('route_state_timeout_s').value)
        self.route_max_hacc = float(
            g('route_max_horizontal_accuracy_m').value)
        self.route_sync_tolerance = float(
            g('route_pose_fix_sync_s').value)
        self.route_anchor_drift = float(g('route_anchor_drift_m').value)
        self.route_anchor_timeout = float(
            g('route_anchor_timeout_s').value)
        self.route_timeout = float(g('route_timeout_s').value)
        self.route_replan_period = float(
            g('route_replan_period_s').value)
        self.route_lookahead = float(g('route_lookahead_m').value)
        self.route_cross_track = float(g('route_cross_track_m').value)
        self.allow_unapproved_route_map = bool(
            g('allow_unapproved_route_map').value)
        if self.planned_cruise and not self.cruise:
            raise ValueError('planned_cruise requires cruise_to_trailer')
        if self.planned_cruise and not self.route_map_yaml:
            raise ValueError('planned_cruise requires route_map_yaml')
        route_values = (self.route_gps_timeout, self.route_state_timeout,
                        self.route_max_hacc, self.route_sync_tolerance,
                        self.route_anchor_drift, self.route_anchor_timeout,
                        self.route_timeout, self.route_replan_period,
                        self.route_lookahead, self.route_cross_track,
                        self.arrival_speed)
        if (self.planned_cruise
                and not all(math.isfinite(v) and v > 0.0
                            for v in route_values)):
            raise ValueError('route timing and follower values must be positive')
        # 0 does NOT mean "no limit" here, it means "PX4 has it" — the fetch
        # fills these in and the arm gate blocks until it has. A negative or
        # non-finite override is still a mistake worth refusing to start on.
        for _name, _value in (('cruise_v_max_m_s', self.cruise_v_max),
                              ('cruise_accel_m_s2', self.cruise_accel),
                              ('cruise_jerk_m_s3', self.cruise_jerk)):
            if not math.isfinite(_value) or _value < 0.0:
                raise ValueError(
                    f'{_name} must be positive, or 0 to take PX4\'s value')
        self.gimbal_aim_topic = str(g('gimbal_aim_topic').value)
        self.gimbal_attitude_topic = str(g('gimbal_attitude_topic').value)
        self.gimbal_track = bool(g('gimbal_track').value)
        self.sweep = GimbalSweep(
            pitch_deg=tuple(float(v) for v in g('scan_pitch_deg').value),
            yaw_step_deg=float(g('scan_yaw_step_deg').value),
            yaw_limit_deg=float(g('scan_yaw_limit_deg').value),
            view_s=float(g('scan_view_s').value),
            look_max_s=float(g('scan_look_max_s').value),
            settle_s=float(g('scan_settle_s').value),
            settled_deg=float(g('gimbal_settled_deg').value),
            attitude_timeout_s=float(g('gimbal_attitude_timeout_s').value),
            enabled=bool(g('gimbal_scan').value))
        self.center_kp = float(g('center_kp').value)
        self.center_v_max = float(g('center_v_max_m_s').value)
        self.descend_radius = float(g('descend_radius_m').value)
        self.descend_speed = float(g('descend_speed_m_s').value)
        self.marker_vel_tau = float(g('marker_vel_tau_s').value)
        self.marker_vel_max = float(g('marker_vel_max_m_s').value)
        self.descend_log_period = float(g('descend_log_period_s').value)
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

    def _on_pose(self, m):
        self.pose = m
        now = self._now()
        self.pose_rx_t = now
        self.pose_t = (_header_stamp_seconds(m)
                       if self.planned_cruise else now)
        if self.planned_cruise:
            self._update_route_site_origin()

    def _on_velocity(self, m: TwistStamped) -> None:
        self.velocity = m
        self.velocity_rx_t = self._now()
        self.velocity_t = _header_stamp_seconds(m)

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

    def _on_vehicle_fix(self, m: NavSatFix) -> None:
        self.vehicle_fix = m
        self.vehicle_fix_rx_t = self._now()
        self.vehicle_fix_t = _header_stamp_seconds(m)
        self._update_route_site_origin()

    def _update_route_site_origin(self) -> None:
        """Remember the newest genuinely synchronized local/global anchor."""
        if (not self.planned_cruise or self.pose is None
                or self.vehicle_fix is None
                or not math.isfinite(self.pose_t)
                or not math.isfinite(self.vehicle_fix_t)
                or abs(self.pose_t - self.vehicle_fix_t)
                > self.route_sync_tolerance
                or self.vehicle_fix.status.status < NavSatStatus.STATUS_FIX):
            return
        values = (
            self.pose.pose.position.x, self.pose.pose.position.y,
            self.vehicle_fix.latitude, self.vehicle_fix.longitude)
        if not all(math.isfinite(float(value)) for value in values):
            return
        info = self._route_map_info
        east, north = self._enu_offset(
            info.origin_lat, info.origin_lon,
            float(self.vehicle_fix.latitude),
            float(self.vehicle_fix.longitude))
        vehicle_local = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        self._route_observed_origin = (
            vehicle_local - np.array([east, north], float))
        self._route_observed_origin_t = max(
            self.pose_t, self.vehicle_fix_t)
        self._route_observed_origin_rx_t = self._now()

    def _on_gimbal(self, m: Vector3Stamped) -> None:
        """siyi_gimbal_node publishes (roll, pitch, yaw) in degrees as x, y, z."""
        self.sweep.on_attitude(self._now(), yaw_deg=float(m.vector.z),
                               pitch_deg=float(m.vector.y))

    def _on_detected(self, m):
        self.detected = bool(m.data)
        self._detector_seen = True

    def _on_marker(self, m: PoseStamped):
        """Accept the marker in `map`, or convert it from the camera frame.

        Which one it is comes off the message header, not a parameter, so the
        two ends can never be configured to disagree.

        THE CONVERSION USES THE GIMBAL'S MEASURED ANGLE, not nadir. Once the
        camera sweeps, "assume it is pointing straight down" stops being a small
        approximation and becomes a wrong answer: a marker seen 45 deg off to
        one side at 5 m would be reported 5 m BELOW the vehicle instead of 5 m
        beside it, and the mission would fly a descent onto empty ground.

        A fix taken while the gimbal is still slewing is dropped rather than
        placed, because off nadir an angle error is multiplied by the slant
        range instead of the height.
        """
        now = self._now()
        if not self.sweep.settled(now):
            return
        p = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
        if m.header.frame_id and m.header.frame_id != self.map_frame:
            if self.pose is None:
                return
            q = self.pose.pose.orientation
            aim_yaw, aim_pitch = self.sweep.angles(now)
            p = marker_enu_from_gimbal_camera(
                p,
                (self.pose.pose.position.x, self.pose.pose.position.y,
                 self.pose.pose.position.z),
                enu_yaw_from_quaternion(q.x, q.y, q.z, q.w),
                # SIYI counts yaw positive to the RIGHT; the geometry wants
                # CCW/left positive, so it is negated exactly here and nowhere
                # else.
                gimbal_yaw_rad=math.radians(-aim_yaw),
                gimbal_pitch_rad=math.radians(aim_pitch))
        # Differenced BEFORE the new fix is stored, and on the same clock the
        # freshness checks use, so a dropout shows up as a gap here too.
        self.marker_vel.update(p, now)
        self.marker = p
        self.marker_t = now
        self._marker_seq += 1

    def _on_target(self, m: PointStamped):
        """The trailer, already in local ENU — trailer_target_node did the geodesy.

        It publishes ONLY while the target is fully valid, so arrival IS the
        validity signal and there is no flag to interpret here.
        """
        target = np.array([m.point.x, m.point.y, m.point.z])
        self.target_t = self._now()
        self.target_sample_t = _header_stamp_seconds(m)
        if (self.planned_cruise
                and math.isfinite(self.target_sample_t)
                and self.target_sample_t > self._target_velocity_sample_t):
            self._target_velocity.update(target[:2], self.target_sample_t)
            self._target_velocity_sample_t = self.target_sample_t
        self.target = target
        self._target_seen = True

    # ------------------------------------------------------------ terminal UI
    def _stdin_loop(self) -> None:
        """Queue terminal input; the ROS timer owns every phase transition."""
        for line in sys.stdin:
            self._stdin_commands.put(line.strip())

    def _drain_stdin_commands(self) -> None:
        """Execute queued terminal commands on the single ROS executor."""
        while True:
            try:
                command = self._stdin_commands.get_nowait()
            except queue.Empty:
                return
            ok, message = self._command(command)
            if ok:
                print(f'\n  OK: {message}.\n', flush=True)
            else:
                print(f'\n  BLOCKED: {message}\n', flush=True)

    def _prompt(self) -> None:
        if not self._stdin_ok or self.phase is not Phase.READY_TO_ARM:
            return
        if self._prompted == Phase.READY_TO_ARM.value:
            return
        self._prompted = Phase.READY_TO_ARM.value
        # THE WHOLE FLIGHT IS IN THE PROMPT, because this is the only place the
        # operator is asked: after this the mission does not stop again. The
        # numbers are the ones they can check against the field in front of
        # them before saying yes.
        if self.planned_cruise:
            leg = (f', fly the planned route to CJU map '
                   f'{self._route_map_info.mission_goal_xy}, then return to '
                   f'the trailer')
        elif self.cruise and self._fresh_target():
            leg = f', fly {self._target_range():.0f} m to the trailer'
        else:
            leg = ''
        print(f'\n{"=" * 72}\n  preflight PASSED — approve to ARM, take off to '
              f'{self.takeoff_alt:.1f} m{leg} and land on the marker'
              f'\n{"=" * 72}\n'
              f'  proceed?  [ENTER = yes / n = abort]  ', end='', flush=True)

    # ---------------------------------------------------------------- services
    def _command(self, command: str) -> tuple[bool, str]:
        """One word from the terminal or ~/command.

        ABORT (or `n`) lands from any phase; anything else — including a bare
        ENTER — releases the single arm gate. There is deliberately no word to
        release a later phase, because there are no later gates: a command that
        can only ever mean one thing does not need to be typed out, and a
        mission that stops again mid-air is the thing this flow removed.
        """
        word = str(command).strip().upper()
        if word in ('ABORT', 'N', 'NO'):
            self._abort(f'operator entered {word or "ABORT"}')
            return True, 'aborting: landing'
        return self._approve()

    def _approve(self) -> tuple[bool, str]:
        """Release THE gate. Arming is the one decision that stays with a human."""
        if self.phase is not Phase.READY_TO_ARM:
            return False, (f'nothing to approve — phase is {self.phase.value}, '
                           f'which is not a gate')
        # Approval is not a snapshot permission: the route that was certified
        # during PRECHECK has to still be certified on the tick that arms.
        if self.planned_cruise:
            blockers = self._route_preflight()
            if blockers:
                return False, ('route changed since PRECHECK: '
                               + '; '.join(blockers))
        self._to(Phase.ARMING)
        return True, 'approved: arming and taking off'

    def _on_command(self, message: String) -> None:
        ok, msg = self._command(message.data)
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(msg)

    def _abort(self, reason: str) -> None:
        """Land now, from any phase."""
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
        self._prompted = ''
        self._announced = ''
        if phase is Phase.SEARCH:
            self._acq_streak = 0
            self._acq_last_marker_seq = self._marker_seq
        if not self.planned_cruise:
            return
        if phase in (Phase.MISSION, Phase.RETURN, Phase.CRUISE):
            self._reset_path_mpc()
        elif phase is Phase.DESCEND:
            self._reset_path_mpc()
        elif phase in (Phase.LAND, Phase.DONE):
            self._reset_mpc_output()

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

    def _due(self, key: str, period: float) -> bool:
        """True at most once per `period` s, per key — for periodic logging.

        Replaces a tick-counter modulo, which only lands on the intended period
        when the timer fires on exact boundaries and divides cleanly by
        `rate_hz`. This is on the clock, so a 3 s log is 3 s.
        """
        now = self._now()
        if now - self._t_calls.get(key, float('-inf')) < period:
            return False
        self._t_calls[key] = now
        return True

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

    def _search_target(self) -> float:
        """`search_alt` above the same ground datum — see `_takeoff_target`."""
        base = self._z_ground if self._z_ground is not None else 0.0
        return base + self.search_alt

    def _fresh_marker(self) -> bool:
        return (self.marker is not None
                and (self._now() - self.marker_t) < self.marker_timeout)

    def _marker_acquired(self) -> bool:
        """Count accepted camera frames, never timer ticks, toward acquisition."""
        if not self._fresh_marker() or not self.detected:
            self._acq_streak = 0
            return False
        if self._marker_seq != self._acq_last_marker_seq:
            self._acq_last_marker_seq = self._marker_seq
            self._acq_streak += 1
        return self._acq_streak >= self.acquire_frames

    def _fresh_target(self) -> bool:
        now = self._now()
        if (self.target is None
                or (now - self.target_t) >= self.target_timeout):
            return False
        if not self.planned_cruise:
            return True
        source_age = now - self.target_sample_t
        return (math.isfinite(self.target_sample_t)
                and -self.route_sync_tolerance <= source_age
                < self.target_timeout)

    def _range_to(self, point) -> float:
        """Horizontal distance to a point in the local frame [m], NaN if unknown."""
        if point is None or self.pose is None:
            return float('nan')
        return float(math.hypot(float(point[0]) - self.pose.pose.position.x,
                                float(point[1]) - self.pose.pose.position.y))

    def _target_range(self) -> float:
        return self._range_to(self.target)

    def _on_ground(self) -> bool:
        """True once the vehicle has actually settled — never a geometric guess."""
        if self.state and not self.state.armed:
            return True
        return (self.ext is not None
                and self.ext.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

    def _reset_mpc_output(self) -> None:
        self._last_mpc_acceleration = np.zeros(3)
        self._last_mpc_acceleration_t = None

    # ------------------------------------------------------- PX4 flight limits
    def _flight_limits_ready(self) -> bool:
        """True once every limit has a positive value to fly on."""
        return (self.cruise_v_max > 0.0 and self.cruise_accel > 0.0
                and self.cruise_jerk > 0.0)

    def _build_path_mpc(self) -> None:
        """Build the tracking MPC on the vehicle's own bounds.

        Same Wang equations and cost weights as the simulation; only the
        actuator bounds differ, and they are PX4's rather than a number written
        here. No-op until the limits have arrived.
        """
        if not self.planned_cruise or not self._flight_limits_ready():
            return
        self._path_mpc = self._tracking_mpc_cls(
            dt_s=0.1, horizon=20,
            v_max=self.cruise_v_max, a_max=self.cruise_accel,
            j_max=self.cruise_jerk, q_pos=4.0, q_vel=0.4, r_acc=0.05,
            q_terminal=20.0)

    def _sync_limits_from_fcu(self) -> None:
        """Ask PX4 how fast and how hard this airframe may be flown.

        Retried until it lands. PARAMETER_NOT_SET means MAVROS has not finished
        pulling the FCU's parameter table yet, NOT that the parameter is
        missing — `path_plan/mavros_static_path.py` hit the same thing, and
        treating it as an error there cost a flight.
        """
        if self._fcu_limits_req or self._flight_limits_ready():
            return
        if not (self.state and self.state.connected):
            return
        if not self.fcu_param_cli.service_is_ready():
            return
        self._fcu_limits_req = True
        future = self.fcu_param_cli.call_async(
            GetParameters.Request(names=list(self._fcu_limit_names())))
        future.add_done_callback(self._on_fcu_limits)

    def _fcu_limit_names(self) -> tuple[str, str, str]:
        return (self.fcu_speed_param, self.fcu_accel_param, self.fcu_jerk_param)

    def _on_fcu_limits(self, future) -> None:
        self._fcu_limits_req = False
        names = self._fcu_limit_names()
        try:
            values = future.result().values
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().warn(
                f'could not read the PX4 flight limits ({exc}) — retrying',
                throttle_duration_sec=5.0)
            return
        if len(values) != len(names):
            self.get_logger().warn(
                f'MAVROS returned {len(values)} of {len(names)} flight limits '
                f'— retrying', throttle_duration_sec=5.0)
            return
        limits = {}
        for name, pv in zip(names, values):
            if pv.type == ParameterType.PARAMETER_NOT_SET:
                self.get_logger().info(
                    f'{name} not synced from the FCU yet — retrying',
                    throttle_duration_sec=5.0)
                return
            value = (pv.double_value
                     if pv.type == ParameterType.PARAMETER_DOUBLE
                     else float(pv.integer_value))
            if not math.isfinite(value) or value <= 0.0:
                # Not something to substitute a default for: the vehicle is
                # telling us its own limit is unusable, and inventing one here
                # is exactly the second copy this reads PX4 to avoid.
                self.get_logger().error(
                    f'{name}={value} on the FCU is not a usable limit — set it '
                    f'on the vehicle; this mission will not arm without it',
                    throttle_duration_sec=10.0)
                return
            limits[name] = value

        # A positive parameter here was an explicit override; PX4 fills the rest.
        overridden = []
        if self.cruise_v_max > 0.0:
            overridden.append('speed')
        else:
            self.cruise_v_max = limits[self.fcu_speed_param]
        if self.cruise_accel > 0.0:
            overridden.append('acceleration')
        else:
            self.cruise_accel = limits[self.fcu_accel_param]
        if self.cruise_jerk > 0.0:
            overridden.append('jerk')
        else:
            self.cruise_jerk = limits[self.fcu_jerk_param]
        note = (f' (overridden here: {", ".join(overridden)})'
                if overridden else '')
        self.get_logger().info(
            f'flight limits from PX4: {self.cruise_v_max:.2f} m/s '
            f'({self.fcu_speed_param}), {self.cruise_accel:.2f} m/s^2 '
            f'({self.fcu_accel_param}), {self.cruise_jerk:.2f} m/s^3 '
            f'({self.fcu_jerk_param}){note}')
        self._build_path_mpc()

    def _reset_path_mpc(self) -> None:
        if self._path_mpc is not None:
            self._path_mpc.reset()
        if self._path_reference is not None:
            self._path_reference.reset()
        self._path_solve_t = None
        self._path_last_solve_t = None
        self._reset_mpc_output()

    def _set_yaw(self, message: PositionTarget) -> None:
        """Put the mission's held heading on either MAVROS setpoint shape."""
        yaw = self._yaw_hold if self._yaw_hold is not None else self._yaw_now()
        if yaw is None:
            message.type_mask = (
                (message.type_mask & ~PositionTarget.IGNORE_YAW_RATE)
                | PositionTarget.IGNORE_YAW)
            message.yaw_rate = 0.0
        else:
            message.yaw = float(yaw)

    def _send(self, vx: float, vy: float, vz: float) -> None:
        """Stream a velocity setpoint in the local ENU frame.

        Velocity, not position: centring is regulation against the marker, and a
        position setpoint would re-inject the estimator's drift. The FORCE bit is
        deliberately NOT set — PX4 does not support it on this path and may
        reject the setpoint.

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
        self._set_yaw(m)
        self.sp_pub.publish(m)

    def _send_pva(self, position, velocity, acceleration,
                  jerk_m_s3: float) -> bool:
        """Stream Wang's P/V/A reference through MAVROS, never px4_msgs."""
        pos = np.asarray(position, float)
        vel = np.asarray(velocity, float)
        desired_acc = np.asarray(acceleration, float)
        if (pos.shape != (3,) or vel.shape != (3,)
                or desired_acc.shape != (3,)
                or not np.all(np.isfinite(np.r_[pos, vel, desired_acc]))):
            return False
        now = self._now()
        elapsed = (1.0 / self.rate_hz
                   if self._last_mpc_acceleration_t is None else
                   max(0.0, now - self._last_mpc_acceleration_t))
        acc = self._limit_acceleration_slew(
            self._last_mpc_acceleration, desired_acc, jerk_m_s3, elapsed)

        m = PositionTarget()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        # Position, velocity and acceleration are all active. MAVROS performs
        # the ENU/NED conversion for this LOCAL_NED raw-setpoint interface.
        m.type_mask = PositionTarget.IGNORE_YAW_RATE
        m.position.x, m.position.y, m.position.z = map(float, pos)
        m.velocity.x, m.velocity.y, m.velocity.z = map(float, vel)
        m.acceleration_or_force.x, m.acceleration_or_force.y, \
            m.acceleration_or_force.z = map(float, acc)
        self._set_yaw(m)
        self.sp_pub.publish(m)
        self._last_mpc_acceleration = acc
        self._last_mpc_acceleration_t = now
        self._v_cmd = vel[:2].copy()
        return True

    def _publish_aim(self, yaw_deg: float, pitch_deg: float) -> None:
        """Point the gimbal. siyi_gimbal_node reads pitch in y, yaw in z."""
        m = Vector3Stamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.vector.y, m.vector.z = float(pitch_deg), float(yaw_deg)
        self.aim_pub.publish(m)

    def _release_aim(self) -> None:
        """Hand the gimbal back to its own nadir hold. NaN is the release."""
        if self.sweep.aim_cmd is None:
            return
        self.sweep.aim_cmd = None
        self.sweep.stop()
        m = Vector3Stamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.vector.y = m.vector.z = float('nan')
        self.aim_pub.publish(m)

    def _track_marker(self) -> None:
        """Re-aim at the marker WHEN IT HAS BEEN LOST, not continuously.

        Pointing the camera at a target on the ground is not free, and the bill
        is paid in the one currency this marker has none of — pixels. Off nadir
        the distance to the marker becomes the SLANT range instead of the height,
        and the marker is foreshortened by the elevation on top of that. Measured
        on the case that failed: marker acquired 3.2 m to the side at 4.9 m, so
        26 px straight down; aiming at it made the range 5.9 m and the view
        oblique, which is ~19 px — under the detector's floor. The camera moved
        to look at the marker and that is precisely how it lost it.

        Nadir is the best look available whenever the marker is in frame at all:
        shortest range, no foreshortening, no slew. So the rule is DO NOT MOVE A
        CAMERA THAT IS WORKING. The aim is only re-pointed once the marker has
        actually gone, which is what a stale fix is good for — and as the vehicle
        arrives overhead `gimbal_aim_for` walks the recovery aim back to nadir on
        its own, so there is no handover at the end either.
        """
        if not self.gimbal_track or self.pose is None or self.marker is None:
            return
        if self._fresh_marker() and self.detected:
            return
        q = self.pose.pose.orientation
        yaw, pitch = gimbal_aim_for(
            (self.pose.pose.position.x, self.pose.pose.position.y,
             self.pose.pose.position.z),
            enu_yaw_from_quaternion(q.x, q.y, q.z, q.w),
            self.marker)
        self.sweep.aim(yaw, pitch, self._now())
        self._publish_aim(yaw, pitch)

    def _slew_candidate(self, v_want: np.ndarray) -> np.ndarray:
        """Rate-limit one candidate without changing the remembered command."""
        dv = np.asarray(v_want, dtype=float) - self._v_cmd
        step = self.cruise_accel / self.rate_hz
        n = float(np.linalg.norm(dv))
        if n > step:
            dv *= step / n
        return self._v_cmd + dv

    def _slew(self, v_want: np.ndarray) -> np.ndarray:
        """Rate-limit the horizontal velocity command, and remember it.

        A P controller on a 40 m error asks for full cruise speed in the very
        first tick, and a velocity setpoint that steps is a lurch: the vehicle
        pitches hard to chase it, which on a real airframe with a gimbal hanging
        underneath is both unpleasant and unnecessary. Limiting the CHANGE per
        tick turns every phase transition — start of cruise, arrival, marker
        acquired, target lost — into a ramp instead of a snatch, without touching
        the gains that decide where it ends up.
        """
        self._v_cmd = self._slew_candidate(v_want)
        return self._v_cmd

    def _hold(self, vz: float = 0.0) -> None:
        """Stop horizontally — by ramping down, not by dropping the command."""
        v = self._slew(np.zeros(2))
        self._send(float(v[0]), float(v[1]), vz)

    def _fly_to(self, target_xy, *, kp: float, v_max: float, vz: float = 0.0,
                feed_forward=None) -> float:
        """Drive at a point in the local frame. Returns the distance to it [m].

        Velocity, not position, for the reason given in `_send`: this regulates
        against a target that moves, and a position setpoint would re-inject the
        estimator's drift on top of it.

        `feed_forward` is the TARGET's own velocity, added on top of the
        correction rather than inside it: the P term keeps its full `v_max` of
        authority to close the error, and the feed-forward supplies the speed
        needed merely to keep up. Capping the sum instead would let a fast target
        eat the whole budget and leave nothing to centre with.
        """
        err = np.array([float(target_xy[0]) - float(self.pose.pose.position.x),
                        float(target_xy[1]) - float(self.pose.pose.position.y)])
        distance = float(np.linalg.norm(err))
        v = _capped(kp * err, v_max)
        if feed_forward is not None:
            ff = _capped(np.asarray(feed_forward, dtype=float)[:2],
                         self.marker_vel_max)
            v = _capped(v + ff, v_max + self.marker_vel_max)
        v = self._slew(v)
        self._send(float(v[0]), float(v[1]), vz)
        return distance

    # ----------------------------------------------------- obstacle-aware route
    def _route_flight_health_reason(self) -> str | None:
        """Why an already-anchored absolute-map route must stop, or None."""
        if not self.planned_cruise:
            return None
        try:
            route_stat = Path(self.route_map_yaml).stat()
            identity = (route_stat.st_dev, route_stat.st_ino,
                        route_stat.st_size, route_stat.st_mtime_ns)
        except OSError as exc:
            return f'route map is not readable: {exc}'
        if identity != self._route_map_identity:
            return 'route map changed after startup; restart the mission node'
        now = self._now()
        if not self.ekf.status_fresh(now):
            return 'MAVROS estimator status is missing or stale'
        if self.ekf.const_pos_mode:
            return 'PX4 EKF is in constant-position mode'
        if not self.ekf.velocity_horiz or not self.ekf.pos_horiz_abs:
            return 'PX4 EKF has no absolute horizontal position aiding'
        if self.ekf.gps_glitch:
            return 'PX4 EKF reports a GPS glitch'
        if not self.ekf.gps_fresh(now):
            return 'MAVROS GPSRAW accuracy is missing or stale'
        if (not math.isfinite(self.ekf.h_acc)
                or self.ekf.h_acc > self.route_max_hacc):
            return (f'GPS horizontal accuracy {self.ekf.h_acc:.2f} m exceeds '
                    f'route limit {self.route_max_hacc:.2f} m')
        pose_age = now - self.pose_t
        if (self.pose is None or not math.isfinite(self.pose_t)
                or now - self.pose_rx_t > self.route_state_timeout
                or pose_age > self.route_state_timeout
                or pose_age < -self.route_sync_tolerance):
            return 'MAVROS local pose is missing or stale'
        velocity_age = now - self.velocity_t
        if (self.velocity is None or not math.isfinite(self.velocity_t)
                or now - self.velocity_rx_t > self.route_state_timeout
                or velocity_age > self.route_state_timeout
                or velocity_age < -self.route_sync_tolerance):
            return 'MAVROS local velocity is missing or stale'
        measured_velocity = (
            self.velocity.twist.linear.x, self.velocity.twist.linear.y)
        if not all(math.isfinite(float(value)) for value in measured_velocity):
            return 'MAVROS local velocity is non-finite'
        if abs(self.pose_t - self.velocity_t) > self.route_sync_tolerance:
            return ('MAVROS local pose and velocity are not time-aligned '
                    f'(>{self.route_sync_tolerance:.1f} s)')
        anchor_reason = self._route_anchor_drift_reason()
        if anchor_reason is not None:
            return anchor_reason
        return None

    def _route_synchronized_site_origin(self) -> np.ndarray | None:
        """Recent site origin from a coherent fix/pose source-time pair."""
        now = self._now()
        sample_age = now - self._route_observed_origin_t
        if (self._route_observed_origin is None
                or not math.isfinite(self._route_observed_origin_t)
                or now - self._route_observed_origin_rx_t
                > self.route_anchor_timeout
                or sample_age > self.route_anchor_timeout
                or sample_age < -self.route_sync_tolerance):
            return None
        return np.asarray(self._route_observed_origin, float).copy()

    def _route_anchor_drift_reason(self) -> str | None:
        snapshot = None
        if self._route_active is not None:
            snapshot = self._route_active[1]
        elif self._route_pending is not None:
            snapshot = self._route_pending[3]
        if snapshot is None:
            return None
        current = self._route_synchronized_site_origin()
        if current is None:
            return ('current local/global route anchor is unavailable or '
                    'not time-aligned')
        drift = float(np.linalg.norm(current - np.asarray(snapshot, float)))
        if drift > self.route_anchor_drift:
            return (f'local/global route anchor moved {drift:.2f} m '
                    f'(limit {self.route_anchor_drift:.2f} m)')
        return None

    def _route_uses_trailer_goal(self) -> bool:
        return self.phase in (
            Phase.RETURN_PLAN, Phase.RETURN, Phase.CRUISE,
            Phase.SEARCH, Phase.DESCEND, Phase.LAND)

    def _route_goal_local(self, *, trailer_goal: bool | None = None) \
            -> np.ndarray | None:
        """Current route goal in MAVROS Local ENU, without mixing goal sources."""
        use_trailer = (self._route_uses_trailer_goal()
                       if trailer_goal is None else bool(trailer_goal))
        if use_trailer:
            return (None if self.target is None else
                    np.asarray(self.target[:2], float).copy())
        origin = self._route_synchronized_site_origin()
        if origin is None:
            return None
        return self._route_lib.map_to_local(
            self._route_map_info.mission_goal_xy,
            origin, self._route_rotation)

    def _route_goal_range(self, *, trailer_goal: bool | None = None) -> float:
        return self._range_to(self._route_goal_local(trailer_goal=trailer_goal))

    def _route_input_reason(self, *, trailer_goal: bool | None = None) \
            -> str | None:
        """Why the selected fixed or live-target route cannot be anchored."""
        health_reason = self._route_flight_health_reason()
        if health_reason is not None:
            return health_reason
        now = self._now()
        fix_age = now - self.vehicle_fix_t
        if (self.vehicle_fix is None or not math.isfinite(self.vehicle_fix_t)
                or now - self.vehicle_fix_rx_t > self.route_gps_timeout
                or fix_age > self.route_gps_timeout
                or fix_age < -self.route_sync_tolerance):
            return f'{self.route_vehicle_fix_topic} is missing or stale'
        if self.vehicle_fix.status.status < NavSatStatus.STATUS_FIX:
            return 'vehicle NavSatFix has no valid fix'
        fix_values = (self.vehicle_fix.latitude, self.vehicle_fix.longitude)
        if not all(math.isfinite(float(v)) for v in fix_values):
            return 'vehicle NavSatFix latitude/longitude is non-finite'
        if self._route_synchronized_site_origin() is None:
            return 'vehicle local pose/global fix pair is not coherent or fresh'
        use_trailer = (self._route_uses_trailer_goal()
                       if trailer_goal is None else bool(trailer_goal))
        if use_trailer:
            if not math.isfinite(self.target_sample_t):
                return 'trailer target has no trustworthy source timestamp'
            if (abs(self._route_observed_origin_t - self.target_sample_t)
                    > self.route_sync_tolerance):
                return ('vehicle local/global anchor and trailer target are not '
                        'time-aligned '
                        f'(>{self.route_sync_tolerance:.1f} s)')
            if not self._fresh_target():
                return 'trailer target is missing or stale'
        goal = self._route_goal_local(trailer_goal=use_trailer)
        pose_xy = (self.pose.pose.position.x, self.pose.pose.position.y)
        if (goal is None
                or not all(math.isfinite(float(v)) for v in (*pose_xy, *goal))):
            return 'route start or goal is non-finite'
        return None

    def _route_site_origin_local(self, *, trailer_goal: bool | None = None) \
            -> np.ndarray:
        """The fixed site's WGS84 origin expressed in MAVROS local ENU XY."""
        reason = self._route_input_reason(trailer_goal=trailer_goal)
        if reason is not None:
            raise RuntimeError(reason)
        origin = self._route_synchronized_site_origin()
        if origin is None:
            raise RuntimeError('vehicle local pose/global fix pair is not coherent')
        # site->vehicle ENU is tangent at the configured site origin. Subtract
        # it from the measured local pose to locate that origin in local ENU.
        return origin

    def _route_endpoint_reason(self, *, trailer_goal: bool) -> str | None:
        """Check start and selected goal against the one hard clearance map."""
        origin = self._route_synchronized_site_origin()
        goal = self._route_goal_local(trailer_goal=trailer_goal)
        if self.pose is None or origin is None or goal is None:
            return 'route start or goal is unavailable'
        start = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        for label, point in (('start', start), ('goal', goal)):
            if not self._route_lib.segment_is_free(
                    self.route_map_yaml, origin, point, point):
                return (f'route {label} is blocked or outside the mapped '
                        'vehicle clearance')
        return None

    def _route_settled(self) -> bool:
        """True once the fixed-goal vehicle is genuinely ready to hover."""
        now = self._now()
        sample_age = now - self.velocity_t
        if (self.velocity is None or not math.isfinite(self.velocity_t)
                or now - self.velocity_rx_t > self.route_state_timeout
                or sample_age > self.route_state_timeout
                or sample_age < -self.route_sync_tolerance):
            return False
        linear = self.velocity.twist.linear
        velocity = np.array([linear.x, linear.y, linear.z], float)
        return (np.all(np.isfinite(velocity))
                and float(np.linalg.norm(velocity)) <= self.arrival_speed)

    def _route_preflight(self) -> list[str]:
        """Non-waivable checks for the fixed CJU map-goal route."""
        if not self.planned_cruise:
            return []
        reasons = []
        info = self._route_map_info
        if (not info.hardware_flight_approved
                and not self.allow_unapproved_route_map):
            reasons.append(
                'route map is not hardware-flight-approved '
                f'({info.horizontal_accuracy}); survey/calibrate the YAML first')
        input_reason = self._route_input_reason(trailer_goal=False)
        if input_reason is not None:
            reasons.append(input_reason)
            return reasons
        endpoint_reason = self._route_endpoint_reason(trailer_goal=False)
        if endpoint_reason is not None:
            reasons.append(endpoint_reason)
            return reasons
        goal = self._route_goal_local(trailer_goal=False)
        if self._range_to(goal) <= self.cruise_arrive:
            if not self._route_arrival_safe(trailer_goal=False):
                reasons.append(
                    'nearby mission-goal chord or endpoint is outside the '
                    'certified map clearance')
        return reasons

    def _return_preflight(self) -> list[str]:
        """Inputs required when LAND changes the route goal to the trailer."""
        if not self.planned_cruise:
            return []
        info = self._route_map_info
        reasons = []
        if (not info.hardware_flight_approved
                and not self.allow_unapproved_route_map):
            reasons.append(
                'route map is not hardware-flight-approved '
                f'({info.horizontal_accuracy}); survey/calibrate the YAML first')
        reasons.extend(self._cruise_preflight())
        input_reason = self._route_input_reason(trailer_goal=True)
        if input_reason is not None and input_reason not in reasons:
            reasons.append(input_reason)
        if input_reason is None:
            endpoint_reason = self._route_endpoint_reason(trailer_goal=True)
            if endpoint_reason is not None:
                reasons.append(endpoint_reason)
        return reasons

    def _invalidate_route(self, reason: str) -> None:
        """Discard every route anchored before an absolute-EKF health loss."""
        had_route = self._route_active is not None or self._route_pending is not None
        self._route_active = None
        self._route_pending = None
        self._route_progress = 0.0
        self._reset_path_mpc()
        if self._plan_future is not None and self._plan_future.cancel():
            self._plan_future = None
        self._route_last_error = reason
        if had_route:
            self.get_logger().warn(
                f'route invalidated — {reason}; HOLD until a new anchor is planned')

    def _begin_route(self, phase: Phase) -> None:
        """Discard the previous leg before planning the selected goal."""
        self._route_active = None
        self._route_pending = None
        self._route_progress = 0.0
        self._route_last_request_t = float('-inf')
        self._route_last_error = ''
        self._reset_path_mpc()
        if self._plan_future is not None and self._plan_future.cancel():
            self._plan_future = None
        self._to(phase)

    def _begin_return(self) -> None:
        """Drop the fixed-goal leg before selecting the live trailer goal."""
        self._begin_route(Phase.RETURN_PLAN)

    def _route_update(self) -> None:
        """Poll/commit one worker result, then submit at most one new request."""
        if not self.planned_cruise:
            return
        allowed = (Phase.MISSION_PLAN, Phase.MISSION,
                   Phase.RETURN_PLAN, Phase.RETURN, Phase.CRUISE)
        health_reason = self._route_flight_health_reason()
        if health_reason is not None:
            self._invalidate_route(health_reason)

        if self._plan_future is not None and self._plan_future.done():
            future = self._plan_future
            pending = self._route_pending
            self._plan_future = None
            self._route_pending = None
            try:
                planned = future.result()
                if pending is None:
                    raise RuntimeError('route result has no request metadata')
                seq, _start, requested_goal, origin, _requested_t = pending
                if self.phase not in allowed:
                    raise RuntimeError(
                        f'route completed after phase changed to {self.phase.value}')
                reason = self._route_input_reason()
                if reason is not None:
                    raise RuntimeError(f'route completed with stale inputs: {reason}')
                if not np.allclose(
                        planned.path_local_xy[-1], requested_goal,
                        atol=1.0e-6, rtol=0.0):
                    raise RuntimeError('route result endpoint changed in the worker')
                current = np.array([
                    self.pose.pose.position.x,
                    self.pose.pose.position.y], float)

                def checker(a, b):
                    return self._route_lib.segment_is_free(
                        self.route_map_yaml, origin, a, b)
                joined = self._route_lib.splice_route_from_current(
                    planned.arc_m, planned.path_local_xy, current,
                    self.route_lookahead, checker)
                if joined is None:
                    raise RuntimeError(
                        'current position cannot safely join the completed route')
                # Atomic snapshot: the follower never observes half-updated arc,
                # points, origin or goal.
                self._route_active = (
                    joined, np.asarray(origin, float).copy(),
                    np.asarray(requested_goal, float).copy(),
                    int(seq), self._now())
                self._route_progress = 0.0
                self._reset_path_mpc()
                self._route_last_error = ''
                self.get_logger().info(
                    f'route #{seq} certified: {joined.arc_m[-1]:.1f} m, '
                    f'{len(joined.path_local_xy)} points, '
                    f'A* expanded {planned.expanded_nodes}')
            except Exception as exc:
                self._route_last_error = str(exc)
                self.get_logger().error(
                    f'route rejected — HOLDING, never flying straight: {exc}',
                    throttle_duration_sec=2.0)

        if (self.phase not in allowed or self._plan_future is not None
                or self._planner_pool is None):
            return
        goal = self._route_goal_local()
        reason = self._route_input_reason()
        if (reason is not None or goal is None
                or self._range_to(goal) <= self.cruise_arrive):
            return
        now = self._now()
        if now - self._route_last_request_t < self.route_replan_period:
            return
        if self._route_active is not None:
            _plan, _origin, active_goal, _seq, _committed = self._route_active
            if float(np.linalg.norm(goal - active_goal)) < max(
                    0.5, self.route_cross_track):
                return

        start = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        origin = self._route_site_origin_local()
        self._route_request_seq += 1
        seq = self._route_request_seq
        self._route_pending = (
            seq, start.copy(), goal.copy(), origin.copy(), now)
        self._route_last_request_t = now
        try:
            self._plan_future = self._planner_pool.submit(
                _plan_route_worker, self.route_map_yaml, start.tolist(),
                goal.tolist(), origin.tolist())
        except Exception as exc:
            self._route_pending = None
            self._route_last_error = f'planner worker unavailable: {exc}'
            self.get_logger().error(
                self._route_last_error, throttle_duration_sec=2.0)
            return
        self.get_logger().info(
            f'route #{seq} planning: local ({start[0]:.1f},{start[1]:.1f}) '
            f'-> ({goal[0]:.1f},{goal[1]:.1f})')

    def _route_carrot(self) -> tuple[np.ndarray | None, float]:
        """Current exact-safe local-ENU target and cross-track error."""
        # The route is already anchored: require live absolute-EKF health, but
        # do not demand a new pose/fix timestamp pair merely to follow it.
        reason = self._route_flight_health_reason()
        if reason is not None:
            self._route_last_error = f'route input rejected: {reason}'
            return None, float('inf')
        if self._route_active is None or self.pose is None:
            return None, float('inf')
        plan, origin, _goal, _seq, _committed = self._route_active
        current = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        try:
            progress, target, cross_track = self._route_lib.safe_route_target(
                self.route_map_yaml, origin, plan.arc_m, plan.path_local_xy,
                current, self._route_progress, self.route_lookahead,
                self.route_cross_track)
        except Exception as exc:
            self._route_last_error = f'route follower rejected active path: {exc}'
            self.get_logger().error(
                self._route_last_error, throttle_duration_sec=2.0)
            return None, float('inf')
        self._route_progress = progress
        return target, cross_track

    def _route_prediction_is_safe(self, predicted_positions) -> bool:
        """Require every MPC horizon chord to retain the runtime clearance."""
        if (self.pose is None
                or self._route_flight_health_reason() is not None):
            return False
        predicted = np.asarray(predicted_positions, float)
        if (predicted.ndim != 2 or predicted.shape[1] != 3
                or not np.all(np.isfinite(predicted))):
            return False
        origin = (self._route_active[1]
                  if self._route_active is not None
                  else self._route_synchronized_site_origin())
        if origin is None:
            return False
        current = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        chain = np.vstack((current, predicted[:, :2]))
        return all(self._route_lib.segment_is_free(
            self.route_map_yaml, origin, a, b)
            for a, b in zip(chain[:-1], chain[1:]))

    def _route_mpc_command(self) -> tuple[bool, float]:
        """Track the certified B-spline with Wang TrackingMPC over MAVROS."""
        carrot, cross_track = self._route_carrot()
        if (carrot is None or self.pose is None or self.velocity is None
                or self._path_mpc is None or self._path_reference is None):
            return False, cross_track
        plan = self._route_active[0]
        position = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y,
            self._alt()], float)
        velocity = np.array([
            self.velocity.twist.linear.x,
            self.velocity.twist.linear.y,
            self.velocity.twist.linear.z], float)
        now = self._now()
        solve_due = (
            self._path_last_solve_t is None
            or now < self._path_last_solve_t
            or now - self._path_last_solve_t
            >= self._path_mpc.dt - 1.0e-6)
        if solve_due:
            self._path_last_solve_t = now
            path = np.column_stack((
                plan.path_local_xy,
                np.full(len(plan.path_local_xy), self._takeoff_target())))
            try:
                moving_target = self._route_uses_trailer_goal()
                reference_p, reference_v = self._path_reference_horizon(
                    plan.arc_m, path, self._route_progress,
                    self._path_mpc.dt, self._path_mpc.N,
                    self.cruise_v_max, self.cruise_accel,
                    self._path_mpc.j_max,
                    target_velocity_xy=(self._target_velocity.v
                                        if moving_target else None),
                    target_range_xy_m=self._route_goal_range(),
                    relative_brake_start_m=10.0,
                    target_relative_speed_m_s=0.3)
                output_step = min(
                    int(self._path_reference.lead / self._path_mpc.dt),
                    self._path_mpc.N - 1)
                result = self._path_mpc.solve(
                    position, velocity, reference_p, reference_v,
                    applied_acceleration=self._last_mpc_acceleration,
                    output_step=output_step)
            except Exception as exc:
                self._route_last_error = f'TrackingMPC solve failed: {exc}'
                self._reset_path_mpc()
                self._path_last_solve_t = now
                return False, cross_track
            accepted = (
                result.success
                and np.all(np.isfinite(np.column_stack((
                    result.predicted_pos, result.predicted_vel,
                    result.predicted_acc))))
                and self._route_prediction_is_safe(result.predicted_pos))
            if not accepted:
                self._route_last_error = (
                    'TrackingMPC rejected its solve or predicted horizon')
                self._reset_path_mpc()
                self._path_last_solve_t = now
                return False, cross_track
            zeros = np.zeros(3)
            self._path_reference.set_plan(
                position, velocity,
                result.predicted_pos, result.predicted_vel,
                result.predicted_acc, self._path_mpc.dt,
                zeros, zeros, zeros)
            self._path_solve_t = now

        if (not self._path_reference.ready()
                or self._path_solve_t is None):
            return False, cross_track

        pos, vel, acc = self._path_reference.sample(
            self._now() - self._path_solve_t)
        if not self._route_prediction_is_safe(np.asarray([pos])):
            self._route_last_error = 'TrackingMPC streamed reference is unsafe'
            self._reset_path_mpc()
            self._path_last_solve_t = now
            return False, cross_track
        return (self._send_pva(
            pos, vel, acc, self._path_mpc.j_max), cross_track)

    def _route_arrival_safe(self, *, trailer_goal: bool | None = None) -> bool:
        """Require arrival at the selected goal and completion of its route."""
        if not self.planned_cruise:
            return True
        if self._route_flight_health_reason() is not None:
            return False
        use_trailer = (self._route_uses_trailer_goal()
                       if trailer_goal is None else bool(trailer_goal))
        goal = self._route_goal_local(trailer_goal=use_trailer)
        if (self.pose is None or goal is None
                or (use_trailer and not self._fresh_target())):
            return False
        if self._range_to(goal) > self.cruise_arrive:
            return False
        current = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        try:
            origin = (self._route_active[1] if self._route_active is not None
                      else self._route_site_origin_local(
                          trailer_goal=use_trailer))
        except RuntimeError:
            return False
        # Once the live goal is inside the arrival radius, only this last exact
        # chord matters. Requiring a moving target to remain near an older route
        # endpoint can deadlock arrival after an otherwise successful replan.
        return self._route_lib.segment_is_free(
            self.route_map_yaml, origin, current, goal)

    def _route_station_keep_safe(self) -> bool:
        """Prevent SEARCH's GPS station-keep from bypassing the route map."""
        if not self.planned_cruise or self.pose is None or self.target is None:
            return not self.planned_cruise
        if self._route_flight_health_reason() is not None:
            return False
        current = np.array([
            self.pose.pose.position.x, self.pose.pose.position.y], float)
        try:
            origin = (self._route_active[1] if self._route_active is not None
                      else self._route_site_origin_local())
        except RuntimeError:
            return False
        return self._route_lib.segment_is_free(
            self.route_map_yaml, origin, current, self.target[:2])

    # ------------------------------------------------------------- preflight
    def _preflight_ok(self, *, allow_armed: bool = False) -> bool:
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
        elif self.state.armed and not allow_armed:
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
        # PRECHECK proves the return link before takeoff even though the first leg
        # is the fixed map goal. The trailer coordinate is not used as that goal.
        # A live target proves radio, both fixes and the WGS84 -> Local ENU chain.
        if self.cruise:
            overridable.extend(self._cruise_preflight())
        # PX4 OWNS THE LIMITS, so no limits means no flight — and it is a hard
        # blocker, not a waivable one. `skip_preflight` exists to fly without a
        # battery reading on a bench; it does not exist to fly a vehicle whose
        # maximum speed nothing in this process knows.
        if ((self.cruise or self.planned_cruise)
                and not self._flight_limits_ready()):
            reasons.append(
                'speed and acceleration limits not read from the FCU yet ('
                + ', '.join(self._fcu_limit_names()) + ')')
        # A missing target can be waived for an ordinary bench rehearsal, but a
        # planned route may never turn that waiver into direct obstacle-blind
        # flight. Its map/input/certification checks remain hard blockers.
        if self.planned_cruise:
            reasons.extend(self._route_preflight())
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
        # A silent gimbal never BLOCKS — the sweep still commands angles and
        # places fixes at the commanded ones, and grounding a vehicle over a
        # topic that may not be wired is the worse failure. But it is worth
        # saying once, because unverified angles off nadir are the one input
        # whose error gets multiplied by the slant range.
        if self.sweep.enabled and not self.sweep.attitude_fresh(self._now()):
            gimbal_warn = (f'no gimbal attitude on {self.gimbal_attitude_topic} '
                           f'— the sweep will run on COMMANDED angles, and a fix '
                           f'taken before the camera arrives cannot be caught')
            if gimbal_warn not in self._warned:
                self._warned.add(gimbal_warn)
                self.get_logger().warn(f'preflight WARNING: {gimbal_warn}')
        if not self._checks_logged:
            self._checks_logged = True
            self.get_logger().info(
                f'preflight PASSED — {self.ekf.summary(self._now())}')
            if self.cruise and self._fresh_target():
                if self.planned_cruise:
                    self.get_logger().info(
                        f'trailer return link ready '
                        f'({self._target_range():.1f} m); first mission goal is '
                        f'CJU map {self._route_map_info.mission_goal_xy}')
                else:
                    self.get_logger().info(
                        f'trailer is {self._target_range():.1f} m away — CRUISE '
                        'will fly there before searching')
        return True

    def _cruise_preflight(self) -> list[str]:
        """What must be true about the trailer link before an arm is offered."""
        if not self._target_seen:
            return [f'no trailer target — nothing on {self.target_topic} '
                    f'(is trailer_gps_node + trailer_target_node running? its own '
                    f'log says which input is missing)']
        if not self._fresh_target():
            return [f'trailer target is stale (>{self.target_timeout:.1f} s old) '
                    f'— the link or the trailer fix has dropped']
        distance = self._target_range()
        if math.isfinite(distance) and distance > self.cruise_max_dist:
            return [f'trailer is {distance:.0f} m away, past this mission\'s '
                    f'{self.cruise_max_dist:.0f} m limit — move closer, or raise '
                    f'cruise_max_distance_m deliberately']
        return []

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._drain_stdin_commands()
        self._publish_state()
        # Ask the FCU for its speed/acceleration limits until it answers. Cheap
        # and self-cancelling: `_sync_limits_from_fcu` returns immediately once
        # they are in hand.
        self._sync_limits_from_fcu()

        # Keep the offboard stream alive, unconditionally, for every armed /
        # about-to-be-armed phase — BEFORE the phase logic, so a phase that
        # returns early cannot starve it. PX4 drops offboard after ~0.5 s of
        # silence. Phases that fly a real setpoint overwrite this in the same
        # tick; publishing twice is harmless, a gap is not.
        mpc_phase = self.planned_cruise and self.phase in (
            Phase.MISSION, Phase.HOVER, Phase.RETURN, Phase.CRUISE)
        if (self.phase in (Phase.READY_TO_ARM, Phase.ARMING, Phase.TAKEOFF,
                           Phase.READY, Phase.MISSION_PLAN, Phase.MISSION,
                           Phase.HOVER,
                           Phase.RETURN_PLAN, Phase.RETURN, Phase.CRUISE,
                           Phase.SEARCH, Phase.DESCEND)
                and not mpc_phase):
            self._send(0.0, 0.0, 0.0)

        # Poll only after the heartbeat above. Route-result validation and
        # splice work may not delay the first setpoint of this tick.
        if self.planned_cruise and not mpc_phase:
            self._route_update()

        if self.phase is Phase.PRECHECK:
            if self._preflight_ok():
                self._to(Phase.READY_TO_ARM)
            return

        if self.phase is Phase.READY_TO_ARM:
            self._announce()
            return

        if self.phase is Phase.ARMING:
            # ORDER MATTERS ON PX4: stream -> mode -> arm.
            if self.state and self.state.armed:
                # The arm request is asynchronous. Recheck every other input
                # after PX4 confirms it; if anything changed, LAND on the pad
                # instead of beginning a takeoff on stale approval.
                if not self._preflight_ok(allow_armed=True):
                    self.get_logger().error(
                        'preflight changed while arming — handing to LAND')
                    self._to(Phase.LAND)
                    return
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
                           f', holding heading {math.degrees(self._yaw_hold):.0f}'
                           f' deg ENU')
                self.get_logger().info(
                    f'armed on the ground at local z={self._z_ground:.2f} m — '
                    f'climbing to z={self._takeoff_target():.2f} m '
                    f'({self.takeoff_alt:.1f} m above it){heading}')
                self._to(Phase.TAKEOFF)
                return
            # ENTER/service approval is not a snapshot permission. Sensor,
            # target and certified-route inputs must still be live on the exact
            # tick that can issue set_mode or arm.
            if not self._preflight_ok():
                self._t_prestream = None
                self._checks_logged = False
                self._prompted = ''
                self._announced = ''
                self.get_logger().error(
                    'preflight changed after approval — returning to the arm '
                    'gate; fresh operator approval is required')
                self._to(Phase.PRECHECK)
                return
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
            req = CommandBool.Request()
            req.value = True
            self._call_throttled(self.arm_cli, req, 'arming')
            return

        if self.phase is Phase.TAKEOFF:
            err = self._takeoff_target() - self._alt()
            if abs(err) <= self.alt_tol:
                self._hold()
                if self.planned_cruise and not self._route_settled():
                    return
                if self.planned_cruise:
                    self._to(Phase.READY)
                else:
                    self._to(Phase.CRUISE if self.cruise else Phase.SEARCH)
                return
            vz = float(np.clip(err, -self.climb_speed, self.climb_speed))
            self._hold(vz)
            return

        if self.phase is Phase.READY:
            self._hold()
            blockers = self._route_preflight()
            if not blockers:
                self.get_logger().info(
                    f'at {self.takeoff_alt:.1f} m — planning the A*/SFC/'
                    f'B-spline route to map '
                    f'{self._route_map_info.mission_goal_xy}')
                self._begin_route(Phase.MISSION_PLAN)
                return
            # FAIL FORWARD. The fixed-goal leg is a detour; the landing is the
            # mission. Waiting here for a route that will not certify would
            # hover an armed airframe on a gate nobody is watching, so give it
            # one planning window and then go to the trailer instead.
            if self._now() - self._t_phase > self.route_timeout:
                self.get_logger().error(
                    f'mission route not available within '
                    f'{self.route_timeout:.0f} s ({"; ".join(blockers)}) — '
                    f'skipping the mission leg, returning to the trailer')
                self._begin_return()
                return
            self.get_logger().warn(
                'holding at altitude — mission route not ready yet: '
                + '; '.join(blockers), throttle_duration_sec=3.0)
            return

        if self.phase is Phase.MISSION_PLAN:
            self._hold()
            if (self._route_goal_range(trailer_goal=False) <= self.cruise_arrive
                    and self._route_arrival_safe(trailer_goal=False)
                    and self._route_settled()):
                self._to(Phase.HOVER)
            elif self._route_active is not None:
                self._to(Phase.MISSION)
            elif self._now() - self._t_phase > self.route_timeout:
                # NOT back to READY: READY now enters this phase by itself, so
                # returning there is an unbounded replan loop in the air.
                self.get_logger().error(
                    'fixed-goal route planning timed out — skipping the '
                    'mission leg, returning to the trailer')
                self._begin_return()
            return

        if self.phase is Phase.MISSION:
            self._mission_to_goal()
            if self.phase is Phase.MISSION:
                self._route_update()
            return

        if self.phase is Phase.HOVER:
            commanded, _cross_track = self._route_mpc_command()
            if not commanded:
                self._hold()
            blockers = self._return_preflight()
            if not blockers:
                self.get_logger().info(
                    'settled at the map goal — returning to the live trailer')
                self._begin_return()
                return
            # The trailer link is the one input the return needs and it is not
            # this node's to fix, so hold at the goal while it comes back — for
            # exactly as long as a lost target is tolerated anywhere else in
            # this mission, then land.
            if self._now() - self._t_phase > self.trailer_lost_search:
                self.get_logger().error(
                    'no trailer return available at the map goal ('
                    + '; '.join(blockers) + ') — landing here')
                self._to(Phase.LAND)
                return
            self.get_logger().warn(
                'at the map goal — waiting for the trailer: '
                + '; '.join(blockers), throttle_duration_sec=3.0)
            return

        if self.phase is Phase.RETURN_PLAN:
            self._hold()
            if (self._route_goal_range(trailer_goal=True) <= self.cruise_arrive
                    and self._route_arrival_safe(trailer_goal=True)):
                self._to(Phase.SEARCH)
            elif self._route_active is not None:
                self._to(Phase.RETURN)
            elif (not self._fresh_target()
                  and self._now() - self._t_phase > self.trailer_lost_search):
                self.get_logger().warn(
                    'trailer target lost while planning return — landing here')
                self._to(Phase.LAND)
            elif self._now() - self._t_phase > self.route_timeout:
                self.get_logger().error(
                    'trailer return route planning timed out — landing here')
                self._to(Phase.LAND)
            return

        if self.phase is Phase.RETURN:
            self._cruise_to_trailer()
            if self.phase is Phase.RETURN:
                self._route_update()
            return

        if self.phase is Phase.CRUISE:
            self._cruise_to_trailer()
            if self.phase is Phase.CRUISE:
                self._route_update()
            return

        if self.phase is Phase.SEARCH:
            # Point the camera. The sweep restarts on every entry to SEARCH, so
            # the first look is always straight down — see GimbalSweep.restart.
            if not self.sweep.scanning:
                self.sweep.restart(self._now())
                self.get_logger().info(
                    f'sweeping the gimbal: {len(self.sweep.plan)} looks, '
                    f'~{self.sweep.duration_s():.0f} s per sweep '
                    f'(search_timeout_s {self.search_timeout:.0f})')
            yaw, pitch = self.sweep.look(self._now())
            self._publish_aim(yaw, pitch)
            # Come down to the altitude the marker is actually detectable at
            # (see `search_alt_m`). The descent is part of searching, not a
            # step before it: the marker grows on the way down and is normally
            # acquired mid-descent rather than at the bottom.
            vz = float(np.clip(self._search_target() - self._alt(),
                               -self.search_descend, self.search_descend))
            # Station-keep over the trailer coordinate while looking, so a
            # trailer that rolls on stays under the camera; without a target
            # (cruise off, or link lost) this is the plain hover it always was.
            if self.cruise and self._fresh_target():
                if self.planned_cruise and not self._route_station_keep_safe():
                    self._hold()
                    self.get_logger().error(
                        'SEARCH trailer station-keep would leave the certified '
                        'map corridor — holding altitude',
                        throttle_duration_sec=2.0)
                else:
                    self._fly_to(self.target, kp=self.cruise_kp,
                                 v_max=self.center_v_max, vz=vz)
            else:
                self._hold(vz)
            # Count detector frames, not this 50 Hz timer. Re-reading one fresh
            # pose five times is still one detection, not an acquisition streak.
            if self._marker_acquired():
                # Stop sweeping, and LEAVE THE CAMERA WHERE IT IS: the look that
                # found the marker is by definition a look that can see it.
                # `_track_marker` only re-points once that stops being true.
                self.sweep.stop()
                self._to(Phase.DESCEND)
                self.get_logger().info(
                    f'marker acquired ({self._acq_streak} consecutive fixes) at '
                    f'h={self._alt() - (self._z_ground or 0.0):.1f} m — descending')
                return
            if self._now() - self._t_phase > self.search_timeout:
                self.get_logger().warn(
                    f'no marker within {self.search_timeout:.0f} s at '
                    f'h={self._alt() - (self._z_ground or 0.0):.1f} m — landing '
                    f'here. A marker outside the camera footprint is the usual '
                    f'cause; see search_alt_m')
                self._to(Phase.LAND)
            return

        if self.phase is Phase.DESCEND:
            self._track_marker()
            self._descend()
            return

        if self.phase is Phase.LAND:
            # The camera has no job left; give the gimbal back to its own hold
            # rather than leaving it wherever the last aim pointed it.
            self._release_aim()
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

    def _mission_to_goal(self) -> None:
        """Track the fixed CJU map goal, then hand on to the return."""
        if self.pose is None:
            self._hold()
            return
        goal = self._route_goal_local(trailer_goal=False)
        if goal is None:
            self._hold()
            return
        distance = self._range_to(goal)
        if (distance <= self.cruise_arrive
                and self._route_arrival_safe(trailer_goal=False)
                and self._route_settled()):
            self._hold()
            self.get_logger().info(
                f'at CJU map {self._route_map_info.mission_goal_xy} '
                f'({distance:.1f} m) — settling before the return')
            self._to(Phase.HOVER)
            return
        if self._now() - self._t_phase > self.route_timeout:
            self._hold()
            self.get_logger().error(
                f'mission goal not reached within {self.route_timeout:.0f} s — '
                'abandoning the mission leg, returning to the trailer')
            self._begin_return()
            return

        vz = float(np.clip(self._takeoff_target() - self._alt(),
                           -self.climb_speed, self.climb_speed))
        commanded, cross_track = self._route_mpc_command()
        if not commanded:
            self._hold(vz)
            self.get_logger().error(
                'no exact-safe TrackingMPC reference to the mission goal — '
                'HOLDING, no straight-line fallback',
                throttle_duration_sec=2.0)
        if self._due('mission_log', self.cruise_log_period):
            self.get_logger().info(
                f'[MISSION] drone ({self.pose.pose.position.x:8.2f},'
                f'{self.pose.pose.position.y:8.2f}) | map '
                f'{self._route_map_info.mission_goal_xy} -> local '
                f'({goal[0]:8.2f},{goal[1]:8.2f}) | {distance:6.2f} m | '
                f'route xtrack {cross_track:4.2f} m')

    def _cruise_to_trailer(self) -> None:
        """Fly to the trailer's coordinate at takeoff altitude, then start looking.

        Four ways out, and only one of them is arrival — the rest exist because
        the coordinate comes over a radio from a vehicle that is driving away:

            arrived         within `cruise_arrive_m` -> SEARCH, camera takes over
            target lost     stop first, then SEARCH from wherever we got to: a
                            stale coordinate is not worth chasing, and the marker
                            may well already be in frame
            too far         a fix that jumps past the leash is a bad fix; stop
                            and say so, rather than fly at it
            timeout         never arrived (headwind, a trailer driving faster
                            than cruise speed) -> hand to the autopilot's LAND

        No marker logic here on purpose. Acquiring is SEARCH's job, and having
        one place decide it means the descent can never be started by a detection
        this phase counted differently.
        """
        if self.pose is None:
            self._hold()
            return
        elapsed = self._now() - self._t_phase

        if not self._fresh_target():
            gone = (self._now() - self.target_t) if self.target is not None \
                else elapsed
            if gone > self.trailer_lost_search:
                if self.planned_cruise:
                    self.get_logger().warn(
                        f'trailer target lost for {gone:.1f} s during planned '
                        'cruise — handing to LAND, not descending at an '
                        'uncertified route position')
                    self._to(Phase.LAND)
                else:
                    self.get_logger().warn(
                        f'trailer target lost for {gone:.1f} s during cruise — '
                        f'searching for the marker from here')
                    self._to(Phase.SEARCH)
            else:
                self._hold()      # brief dropout: stop, wait for the coordinate
            return

        timeout = self.route_timeout if self.planned_cruise else self.cruise_timeout
        if elapsed > timeout:
            self.get_logger().warn(
                f'still {self._target_range():.1f} m from the trailer after '
                f'{timeout:.0f} s — landing here')
            self._to(Phase.LAND)
            return

        distance = self._target_range()
        if distance > self.cruise_max_dist:
            self.get_logger().error(
                f'trailer target jumped to {distance:.0f} m, past the '
                f'{self.cruise_max_dist:.0f} m leash — HOLDING, not chasing it',
                throttle_duration_sec=5.0)
            self._hold()
            return

        if distance <= self.cruise_arrive and self._route_arrival_safe():
            self._hold()
            self.get_logger().info(
                f'over the trailer coordinate ({distance:.1f} m) — searching '
                f'for the marker')
            self._to(Phase.SEARCH)
            return

        # Hold the cruise altitude on the way: the climb datum is the ground the
        # vehicle armed on, so a drifting z estimate cannot slowly walk the
        # cruise up or down over a long transit.
        vz = float(np.clip(self._takeoff_target() - self._alt(),
                           -self.climb_speed, self.climb_speed))
        cross_track = None
        if self.planned_cruise:
            commanded, cross_track = self._route_mpc_command()
            if not commanded:
                self._hold(vz)
                self.get_logger().error(
                    'no exact-safe TrackingMPC reference — HOLDING, no '
                    'straight-line fallback', throttle_duration_sec=2.0)
        else:
            self._fly_to(
                self.target, kp=self.cruise_kp, v_max=self.cruise_v_max,
                vz=vz)

        # Both positions in the SAME local ENU frame, so the two coordinates can
        # be read against each other and the range is just their separation —
        # nothing here is in lat/lon, and nothing needs converting to check it.
        if self._due('cruise_log', self.cruise_log_period):
            route_status = ('' if cross_track is None else
                            f' | route xtrack {cross_track:4.2f} m')
            self.get_logger().info(
                f'[CRUISE] drone ({self.pose.pose.position.x:8.2f},'
                f'{self.pose.pose.position.y:8.2f}) | '
                f'trailer ({self.target[0]:8.2f},{self.target[1]:8.2f}) | '
                f'{distance:6.2f} m{route_status}')

    def _descend(self) -> None:
        """Use the existing real-aircraft ArUco proportional controller."""
        self._descend_p()

    def _descend_p(self) -> None:
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
                self._hold()
            return

        # Vertical: sink only while centred; otherwise hold and centre first.
        radius = self._range_to(self.marker)
        centred = radius <= self.descend_radius
        vz = -self.descend_speed if centred else 0.0
        self._fly_to(self.marker, kp=self.center_kp, v_max=self.center_v_max,
                     vz=vz, feed_forward=self.marker_vel.v)

        # Handover: low over the marker AND centred -> autopilot LAND finishes it
        # (the marker leaves the frame from here anyway).
        h = self._alt() - float(self.marker[2])
        if h <= self.touch_alt and radius <= self.touch_xy:
            self.get_logger().info(
                f'over the marker (h={h:.2f} m, xy={radius:.2f} m) — handing to LAND')
            self._to(Phase.LAND)
            return

        # dx/dy are the marker RELATIVE TO THE VEHICLE, in local ENU — the error
        # the controller is actually closing, signed, so which way it is off is
        # readable in flight instead of only the magnitude.
        if self._due('descend_log', self.descend_log_period):
            dx = float(self.marker[0]) - float(self.pose.pose.position.x)
            dy = float(self.marker[1]) - float(self.pose.pose.position.y)
            self.get_logger().info(
                f'[DESCEND] marker dx {dx:+6.2f} dy {dy:+6.2f} | '
                f'xy {radius:5.2f} m | h {h:5.2f} m | '
                f'{"sinking" if centred else "centring (holding alt)"}')

    # ------------------------------------------------------------------ output
    def _announce(self) -> None:
        self._prompt()
        if self._stdin_ok:
            return
        if (self.phase is not Phase.READY_TO_ARM
                or self._announced == Phase.READY_TO_ARM.value):
            return
        self._announced = Phase.READY_TO_ARM.value
        self.get_logger().warn(
            f'>>> WAITING FOR APPROVAL — approve to ARM and take off\n'
            f'    ros2 run mpc_landing approve {self.get_name()}')

    def _publish_state(self) -> None:
        self.state_pub.publish(String(data=self.phase.value))

    def destroy_node(self):
        """Reap the spawned planner so run_px4 never leaves a worker behind."""
        if self._plan_future is not None:
            self._plan_future.cancel()
        if self._planner_pool is not None:
            self._planner_pool.shutdown(wait=True, cancel_futures=True)
            self._planner_pool = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoLandingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
