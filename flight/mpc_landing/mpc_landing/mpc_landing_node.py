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

SEARCH LOOKS AROUND WITH THE GIMBAL, NOT WITH THE VEHICLE
---------------------------------------------------------
The marker is not always under the vehicle when it gets to altitude. SEARCH
therefore sweeps the camera — nadir first, then rings at progressively shallower
pitch, a dwell at each look — and because the gimbal ANGLE is known at the
moment of the sighting, a marker spotted off to one side is a POSITION, not just
a direction: solvePnP supplies the range along the line the gimbal defines
(`marker.marker_enu_from_gimbal_camera`). The camera then stays on the marker
for the whole approach (`gimbal_aim_for`), so the fix that started the descent
is not lost the instant the vehicle begins to move, and the aim returns to nadir
on its own as the vehicle arrives overhead.

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
    /siyi_gimbal_node/attitude             geometry_msgs/Vector3Stamped
Publishes
    /mavros/setpoint_raw/local             mavros_msgs/PositionTarget
    /siyi_gimbal_node/aim                  geometry_msgs/Vector3Stamped
    /mpc_landing/state                     std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from mavros_msgs.msg import (EstimatorStatus, ExtendedState, GPSRAW,
                             PositionTarget, State)
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

from landing_mpc.mpc import LandingMPC
from landing_mpc.predictor import predict_const_vel
from landing_mpc.reference import HorizonReference

from .estimator import DEFAULT_SPEED_ACC_MAX, EstimatorHealth
from .marker import (NADIR_PITCH_DEG, enu_yaw_from_quaternion, gimbal_aim_for,
                     marker_enu_from_gimbal_camera, sweep_plan)
from .mission import CheckResult, GateState, Phase


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

        # A bench run starts where the thing being rehearsed starts. It cannot
        # reach SEARCH the normal way — that road goes through arming — and
        # faking the phases it skipped would rehearse those too.
        self.gate = (GateState(phase=Phase.SEARCH) if self.bench
                     else GateState(auto_after_arm=self.auto_after_arm))
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
        self.ekf = EstimatorHealth(speed_acc_max=self.speed_acc_max)
        # The ground the vehicle armed on, in EKF local z — see `_takeoff_target`.
        self._z_ground: float | None = None
        # The heading to hold for the whole flight, captured at arming — see
        # `_send`. None until then, which makes every setpoint hold whatever
        # heading the vehicle currently has.
        self._yaw_hold: float | None = None
        self._t_touch = None
        self._t_prestream = None
        # SEARCH commits to the descent only after this many CONSECUTIVE fresh
        # detections, so a single spurious ArUco hit cannot trip an irreversible
        # descent (see the SEARCH phase).
        self._acq_streak = 0
        # Last time each throttled service call fired, keyed by name, so the
        # TOUCHDOWN/ABORT handlers stop re-sending land/disarm every tick.
        self._t_calls: dict[str, float] = {}

        # --- gimbal: where the camera is looking, and where we asked it to
        # look. SEARCH sweeps it (see `_scan_plan`) and DESCEND tracks the
        # marker with it, so "straight down" is no longer an assumption this
        # node may make — it has to know the angle to place a fix at all.
        self._scan = self._scan_plan()
        self._scan_i = 0
        self._t_look = self._now()      # when the current look was commanded
        self._t_settled: float | None = None    # ...and when it arrived
        self._aim_cmd: tuple[float, float] | None = None   # (yaw, pitch) deg
        self._scanning = False          # is a sweep in progress right now?
        self._gimbal: tuple[float, float, float] | None = None  # r, p, y deg
        self._t_gimbal = 0.0
        self._found_at: tuple[float, float] | None = None

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
        # Whether the EKF is actually being aided. A pose alone does not say so
        # — see estimator.py.
        self.create_subscription(EstimatorStatus, '/mavros/estimator_status',
                                 self._on_est, _sensor_qos())
        self.create_subscription(GPSRAW, '/mavros/gpsstatus/gps1/raw',
                                 self._on_gps, _sensor_qos())
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
        # Where the camera is ACTUALLY pointing. The commanded angle is what we
        # asked for; this is what we got, and off nadir the difference is
        # multiplied by the slant range — so fixes are placed with this and
        # dropped while it disagrees with the command (`_gimbal_settled`).
        self.create_subscription(Vector3Stamped, self.gimbal_attitude_topic,
                                 self._on_gimbal, 10)

        self.sp_pub = self.create_publisher(PositionTarget,
                                            '/mavros/setpoint_raw/local', 10)
        self.aim_pub = self.create_publisher(Vector3Stamped,
                                             self.gimbal_aim_topic, 10)
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
        if self.bench:
            self.get_logger().warn(
                'BENCH_SEARCH — starting in SEARCH on the ground. No setpoint '
                'and no arm/mode/land call will be made; the gimbal WILL move. '
                'Every fix is reported, none is flown. Ctrl-C to stop.')
        # Say the mode FIRST — it is the one thing you want confirmed before a
        # real vehicle is armed in front of you.
        self.get_logger().info(
            f'mpc_landing_node: PX4 mode={self.mode_name} | '
            f'takeoff {self.takeoff_alt:.1f} m at {self.climb_speed:.1f} m/s | '
            f'descend on {self.marker_pose_topic}')
        if self.gimbal_scan and len(self._scan) > 1:
            self.get_logger().info(
                f'SEARCH sweeps the gimbal: {len(self._scan)} looks, '
                f'pitch {self.scan_pitch} deg, yaw +/-{self.scan_yaw_limit:.0f} '
                f'in {self.scan_yaw_step:.0f} deg steps, {self.scan_view:.1f} s '
                f'SETTLED at each (~{len(self._scan) * (self.scan_view + 1.0):.0f} s '
                f'per sweep incl. slew, search_timeout_s {self.search_timeout:.0f})')
        else:
            self.get_logger().info(
                'SEARCH holds the gimbal at nadir (gimbal_scan=false)')
        if self.auto_after_arm:
            self.get_logger().warn(
                'auto_after_arm: the ARM is the ONLY approval — takeoff, search '
                'and descent then run without stopping. Abort with: '
                f'ros2 run mpc_landing abort {self.get_name()}')
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
        # TAKEOFF climb rate. RAISED from 0.7: at 0.7 the climb was a laboured
        # crawl that spent most of a minute in ground effect, which is neither
        # comfortable to watch nor good for the vehicle — a multirotor is more
        # stable climbing away decisively than hanging just above its own
        # downwash. PX4's own MPC_Z_VEL_MAX_UP (3.0 default) is still the
        # ceiling above this. The descent rate is NOT this number; it is the
        # MPC's mpc_vz_max_m_s and it stays slow on purpose.
        p('climb_speed_m_s', 1.5)
        # Climb at the FULL rate until this close to the target, then ease off
        # linearly. Without it the climb is a pure P law on altitude error, so
        # raising climb_speed also moved the point where it starts backing off
        # (at 1.5 m/s it would begin slowing 1.5 m below target) — the vehicle
        # would go up faster and still finish with the same long float. This
        # separates "how fast" from "how gently it arrives".
        p('climb_ease_m', 0.8)
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
        # No marker in SEARCH -> abort. RAISED from 60 s with the sweep: a
        # measured sweep is ~35 s, and the number that matters is how many
        # COMPLETE sweeps fit before giving up. One is not enough — the marker
        # can sit on the boundary between two looks and be missed by a lap that
        # was otherwise fine. This is an upper bound on hovering, not a plan:
        # the mission commits the moment it sees the marker.
        p('search_timeout_s', 90.0)
        # SEARCH -> DESCEND is automatic and irreversible, so require the marker
        # to be seen this many CONSECUTIVE ticks before committing.  One frame is
        # enough for a false positive to start a descent; 5 ticks (~0.25 s at
        # rate_hz) is still immediate to a human but rejects a lone bad fix.
        p('marker_acquire_frames', 5)
        # --- gimbal search
        # SEARCH used to hold the vehicle still with the camera at nadir and
        # hope the marker was in the ~40 deg cone underneath it. If it was not,
        # the mission timed out having looked at one patch of ground for a
        # minute. Sweeping the gimbal instead turns those 60 s into a search of
        # a circle roughly 2*tan(50 deg) ~ 2.4 vehicle-heights across, without
        # moving the vehicle at all — and because the gimbal ANGLE is known at
        # the moment of the sighting, what comes back is a position fix and not
        # merely "it is somewhere over there" (see marker.py).
        p('gimbal_aim_topic', '/siyi_gimbal_node/aim')
        p('gimbal_attitude_topic', '/siyi_gimbal_node/attitude')
        p('gimbal_scan', True)
        # The rings to sweep, in gimbal pitch (negative is down). Nadir first,
        # because the marker is usually under the vehicle and the cheapest look
        # is the one the camera is already pointing at. Then wider rings.
        # Do not add a ring shallower than about -25 deg: the slant range grows
        # as 1/sin(elevation), so a shallow look sees a long way and places the
        # marker very badly when it does.
        p('scan_pitch_deg', [-90.0, -60.0, -40.0])
        # Azimuth step within a ring, and how far round it goes. 135 deg is the
        # A8 mini's YAW TRAVEL LIMIT, not a choice — the gimbal cannot look
        # behind the tail, so a 90 deg sector back there stays blind and is the
        # one direction a search may have to be flown rather than looked at.
        p('scan_yaw_step_deg', 45.0)
        p('scan_yaw_limit_deg', 135.0)
        # How long to LOOK at each look — settled time, not wall-clock time.
        #
        # This was a fixed 1.5 s dwell, and the bench measured what that really
        # bought: 36% of the sweep settled, 64% slewing. A 45 deg step lands in
        # ~0.5 s but the 135 deg swing into each new ring takes ~1.5 s, and a
        # fixed dwell pays both the same, so the big swings arrived just as
        # their turn ended. Three of fifteen looks got a single settled sample.
        #
        # Timing the dwell from the moment the gimbal SETTLES gives every look
        # the same real viewing time whatever it cost to get there, and removes
        # a knob whose right value depended on the slew rate of the hardware.
        # 1.0 s is ~12 detector frames, against the 5 consecutive that
        # `marker_acquire_frames` needs.
        p('scan_view_s', 1.0)
        # ...but never wait forever. With no attitude feedback, or a gimbal
        # that cannot reach a commanded angle, "settled" may never arrive and
        # the sweep would stop dead at one look. This is the guarantee that it
        # keeps moving: worst observed slew (135 deg) plus settle plus view,
        # with margin.
        p('scan_look_max_s', 4.0)
        # A fix taken while the gimbal is still slewing is placed at the wrong
        # angle, and off nadir that error is multiplied by the slant range. So
        # detections are ignored until the gimbal has been settled this long
        # AND its feedback agrees with the commanded angle.
        p('scan_settle_s', 0.5)
        # RAISED from 4.0 after the bench: settled yaw error came in at 1.5 deg
        # mean but 3.9 deg peak, which is the threshold itself — a marginally
        # slower gimbal, or one fighting a breeze, would have been judged to be
        # slewing for the whole sweep and every fix thrown away. The cost of
        # the wider band is bounded: 6 deg at a 7 m slant is 0.7 m, and it only
        # applies to whether a fix is ACCEPTED, never to where it is placed —
        # placement uses the measured angle either way.
        p('gimbal_settled_deg', 6.0)
        p('gimbal_attitude_timeout_s', 2.0)
        # Keep the camera on the marker while flying to it. Without this a
        # marker found 40 deg off to one side leaves the frame the moment the
        # gimbal snaps back to nadir, and the descent aborts on a lost marker
        # before the vehicle has covered any ground. The aim walks itself back
        # to nadir as the vehicle arrives overhead, so there is no handover.
        p('gimbal_track', True)
        # GROUND REHEARSAL. Start in SEARCH, on the bench, and never touch the
        # flight controller: no setpoint reaches /mavros/setpoint_raw/local and
        # no arm, mode or land service is called (`_send` and `_call` are the
        # two chokepoints, and both refuse). The sweep runs, the detector runs,
        # and every fix is REPORTED instead of being flown — so the half of
        # SEARCH that is new can be checked with the props off, which is the
        # only sane place to find out that the gimbal is not talking or that
        # the marker lands 3 m from where it is.
        #
        # It cannot check what only altitude provides: on the ground the whole
        # sweep points at the floor within a metre or two of the airframe.
        # Prop the vehicle up and put the marker close.
        # ONE APPROVAL FOR THE WHOLE FLIGHT. The operator authorises the ARM;
        # takeoff, search and descent then run without stopping again.
        #
        # The gates being dropped are the two that pause a flight already under
        # way, and what they were really buying was a chance to look at the
        # vehicle before the next step. That is worth having on a first flight
        # and a nuisance on the twentieth, when the operator is standing in a
        # field holding a terminal while an armed airframe waits on the pad for
        # a keystroke. `abort` still lands it from any phase, which is the
        # control that actually matters once it is airborne.
        #
        # The ARM gate cannot be dropped by this or by anything else — see
        # mission.AUTO_RELEASABLE.
        p('auto_after_arm', False)
        p('bench_search', False)
        # --- preflight thresholds
        p('min_battery_v', 14.0)            # 4S nominal; raise for 6S
        p('require_battery', True)          # false only for bench tests
        # Refuse to pass preflight while the EKF has no position aiding. Match
        # this to the vehicle's EKF2_REQ_SACC so the check quotes the real limit
        # — the default is the RAISED 1.0 m/s, not PX4's 0.5 (see estimator.py),
        # so a vehicle still at 0.5 will refuse arms this check waved through.
        p('require_gnss_aiding', True)
        p('gps_speed_acc_max_m_s', DEFAULT_SPEED_ACC_MAX)
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
        self.climb_ease = max(float(g('climb_ease_m').value), 1e-3)
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
        self.gimbal_aim_topic = str(g('gimbal_aim_topic').value)
        self.gimbal_attitude_topic = str(g('gimbal_attitude_topic').value)
        self.gimbal_scan = bool(g('gimbal_scan').value)
        self.scan_pitch = [float(v) for v in g('scan_pitch_deg').value]
        self.scan_yaw_step = abs(float(g('scan_yaw_step_deg').value))
        self.scan_yaw_limit = abs(float(g('scan_yaw_limit_deg').value))
        self.scan_view = float(g('scan_view_s').value)
        self.scan_look_max = float(g('scan_look_max_s').value)
        self.scan_settle = float(g('scan_settle_s').value)
        self.gimbal_settled_deg = float(g('gimbal_settled_deg').value)
        self.gimbal_timeout = float(g('gimbal_attitude_timeout_s').value)
        self.gimbal_track = bool(g('gimbal_track').value)
        self.auto_after_arm = bool(g('auto_after_arm').value)
        self.bench = bool(g('bench_search').value)
        self.min_batt = float(g('min_battery_v').value)
        self.require_batt = bool(g('require_battery').value)
        self.require_gnss = bool(g('require_gnss_aiding').value)
        self.speed_acc_max = float(g('gps_speed_acc_max_m_s').value)
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
        """Camera-optical marker -> ENU, from where the gimbal is looking.

        No tf2, and deliberately so: the transforms are published correctly but
        this process cannot drain them fast enough to look one up at capture
        time, so the whole chain resolves to nothing. What is actually needed is
        one angle — where the camera is pointing in the world — and that is the
        gimbal's own attitude plus the heading MAVROS already puts on the pose
        we are differencing against anyway.

        This used to assume nadir, which was true because nothing ever moved the
        gimbal. Now SEARCH sweeps it, so the angle is READ (`_gimbal_angles`)
        rather than assumed, and a fix taken while the gimbal is still slewing
        is thrown away rather than placed at an angle it no longer has
        (`_gimbal_settled`) — off nadir that error is multiplied by the slant
        range, not by the height.

        The vehicle's own roll and pitch are still ignored; see
        `marker_enu_from_gimbal_camera` for exactly what that costs. Range comes
        straight from solvePnP, which makes the descent gate measure height
        above the MARKER instead of above whatever datum the EKF started at.
        """
        if self.pose is None:
            return None
        if not self._gimbal_settled():
            return None
        q = self.pose.pose.orientation
        yaw_deg, pitch_deg = self._gimbal_angles()
        return marker_enu_from_gimbal_camera(
            tvec,
            (self.pose.pose.position.x, self.pose.pose.position.y,
             self.pose.pose.position.z),
            enu_yaw_from_quaternion(q.x, q.y, q.z, q.w),
            # SIYI counts yaw positive to the RIGHT; the geometry is written in
            # the usual CCW-positive convention, so the sign flips here and in
            # exactly one other place (`_aim`, on the way out).
            gimbal_yaw_rad=np.radians(-yaw_deg),
            gimbal_pitch_rad=np.radians(pitch_deg),
        )

    # ------------------------------------------------------------------ gimbal
    def _on_gimbal(self, m: Vector3Stamped) -> None:
        self._gimbal = (float(m.vector.x), float(m.vector.y),
                        float(m.vector.z))          # roll, pitch, yaw (deg)
        self._t_gimbal = self._now()

    def _gimbal_fresh(self) -> bool:
        return (self._gimbal is not None
                and (self._now() - self._t_gimbal) <= self.gimbal_timeout)

    def _gimbal_angles(self) -> tuple[float, float]:
        """Where the camera is pointing, (yaw, pitch) in degrees, SIYI signs.

        Feedback first, because that is the measurement; the last commanded
        angle second, because a gimbal with no telemetry is still obeying; and
        nadir last, because that is what siyi_gimbal_node holds when nobody has
        asked for anything else.
        """
        if self._gimbal_fresh():
            _roll, pitch, yaw = self._gimbal
            return yaw, pitch
        if self._aim_cmd is not None:
            return self._aim_cmd
        return 0.0, NADIR_PITCH_DEG

    def _gimbal_settled(self) -> bool:
        """Is the camera pointing somewhere we can trust a fix from?

        Enough time since the aim last MOVED, always. Plus, while SCANNING,
        feedback that agrees with the command: a sweep steps 45 deg at a time
        and the attitude poll is a few Hz, so between the slew and the poll the
        reported angle can be a whole sector out of date, and a fix placed with
        it lands somewhere the marker never was.

        That second test is deliberately NOT applied while tracking. There the
        aim moves at a few degrees a second — the MPC caps the vehicle at
        `mpc_v_max_m_s` — so command and feedback are never far apart, but they
        are never exactly equal either, and demanding agreement would throw away
        every fix during the approach and abort the descent on a marker the
        camera can see perfectly well.
        """
        if self._now() - self._t_look < self.scan_settle:
            return False
        if not self._scanning or self._aim_cmd is None \
                or not self._gimbal_fresh():
            return True
        want_yaw, want_pitch = self._aim_cmd
        _roll, pitch, yaw = self._gimbal
        return (abs(yaw - want_yaw) <= self.gimbal_settled_deg
                and abs(pitch - want_pitch) <= self.gimbal_settled_deg)

    def _aim(self, yaw_deg: float, pitch_deg: float) -> None:
        """Point the gimbal. Restarts the settle timer when it actually moves."""
        if (self._aim_cmd is None
                or abs(yaw_deg - self._aim_cmd[0]) > self.gimbal_settled_deg
                or abs(pitch_deg - self._aim_cmd[1]) > self.gimbal_settled_deg):
            self._t_look = self._now()
        self._aim_cmd = (float(yaw_deg), float(pitch_deg))
        m = Vector3Stamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.vector.y, m.vector.z = float(pitch_deg), float(yaw_deg)
        self.aim_pub.publish(m)

    def _release_aim(self) -> None:
        """Hand the gimbal back to its own nadir hold. NaN is the release."""
        if self._aim_cmd is None:
            return
        self._aim_cmd = None
        self._t_look = self._now()
        m = Vector3Stamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.vector.y = m.vector.z = float('nan')
        self.aim_pub.publish(m)

    def _scan_plan(self) -> list[tuple[float, float]]:
        """The sweep this flight will fly. `gimbal_scan=false` is nadir only."""
        if not self.gimbal_scan:
            return [(0.0, NADIR_PITCH_DEG)]
        return sweep_plan(self.scan_pitch, self.scan_yaw_step,
                          self.scan_yaw_limit)

    def _scan_tick(self) -> None:
        """Hold the current look, and move on when its dwell is up.

        The sweep RESTARTS each time SEARCH is entered, rather than resuming
        where a previous search left off: the first look is nadir, and after a
        climb — or after an abort and a second attempt — under the vehicle is
        again the first place worth looking. Without the restart the dwell
        timer would still be running from whenever the gimbal was last moved
        and the first look would be skipped before the camera ever saw it.
        """
        now = self._now()
        if not self._scanning:
            self._scanning = True
            self._scan_i = 0
            self._t_look = now
            self._t_settled = None
        elif len(self._scan) > 1:
            # Count the dwell from when the camera actually ARRIVED, so a look
            # that took a long swing to reach still gets its full `scan_view_s`
            # of looking. `scan_look_max_s` is the backstop for a gimbal that
            # never reports arriving at all.
            if self._gimbal_settled():
                self._t_settled = self._t_settled or now
                seen_enough = (now - self._t_settled) >= self.scan_view
            else:
                # CONTINUOUS settled time, so a gimbal that arrives, gets
                # knocked off by a gust and comes back has to serve the full
                # view again rather than banking the two halves. Same rule as
                # `marker_acquire_frames` and for the same reason.
                self._t_settled = None
                seen_enough = False
            if seen_enough or (now - self._t_look) >= self.scan_look_max:
                self._scan_i = (self._scan_i + 1) % len(self._scan)
                self._t_settled = None
        yaw, pitch = self._scan[self._scan_i]
        self._aim(yaw, pitch)

    def _bench_report(self) -> None:
        """One line a second: what the camera is doing and what it concluded.

        Everything an operator standing over the airframe needs to tell the
        three failures apart — the gimbal is not moving, the detector is not
        seeing, or the fix is being placed in the wrong spot — without reading
        four topics at once.
        """
        if self._now() - self._t_calls.get('bench', 0.0) < 1.0:
            return
        self._t_calls['bench'] = self._now()
        want_yaw, want_pitch = self._aim_cmd or (0.0, NADIR_PITCH_DEG)
        if self._gimbal_fresh():
            _r, pitch, yaw = self._gimbal
            att = f'at y{yaw:+6.1f} p{pitch:+6.1f}'
        else:
            att = 'NO FEEDBACK   '
        if self._fresh_marker() and self.pose is not None:
            off = self.marker - np.array([self.pose.pose.position.x,
                                          self.pose.pose.position.y,
                                          self.pose.pose.position.z])
            fix = (f'MARKER {np.linalg.norm(off):.2f} m away '
                   f'(E{off[0]:+.2f} N{off[1]:+.2f} U{off[2]:+.2f})')
        elif self.detected:
            fix = 'seen, but no usable fix (gimbal still moving?)'
        else:
            fix = 'no marker'
        self.get_logger().info(
            f'BENCH look {self._scan_i + 1}/{len(self._scan)} '
            f'want y{want_yaw:+6.1f} p{want_pitch:+6.1f} | {att} | '
            f'{"settled" if self._gimbal_settled() else "SLEWING"} | {fix}')

    def _track_marker(self) -> None:
        """Keep the camera on the last marker fix while flying to it."""
        if not self.gimbal_track or self.pose is None or self.marker is None:
            return
        q = self.pose.pose.orientation
        yaw, pitch = gimbal_aim_for(
            (self.pose.pose.position.x, self.pose.pose.position.y,
             self.pose.pose.position.z),
            enu_yaw_from_quaternion(q.x, q.y, q.z, q.w),
            self.marker)
        self._aim(yaw, pitch)

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
        if self.bench:
            # The bench never commands the vehicle. This is the chokepoint for
            # arm / set_mode / land, so an abort on the bench also stops here
            # rather than telling a parked airframe to land.
            self.get_logger().warn(f'bench_search: NOT calling {name}',
                                   throttle_duration_sec=5.0)
            return False
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

    def _takeoff_target(self) -> float:
        """`takeoff_alt` above the GROUND, in EKF local z.

        Not `takeoff_alt` itself. The EKF's z datum is wherever the estimator
        happened to start, and it drifts: on the pad, disarmed, this airframe
        has been seen reporting z = -5.98 m while standing still. Climbing to an
        absolute z of `takeoff_alt` from there is an eleven-metre climb when
        five were asked for. So the ground is measured at the moment of arming —
        the vehicle is provably on it then — and the target is relative to that.
        The same reasoning already applies to `_takeoff_xy`, which is captured
        rather than assumed to be the origin.
        """
        base = self._z_ground if self._z_ground is not None else 0.0
        return base + self.takeoff_alt

    def _alt(self) -> float:
        return float(self.pose.pose.position.z) if self.pose else float('nan')

    def _yaw_now(self) -> float | None:
        """The vehicle's current heading in ENU, or None with no pose yet."""
        if self.pose is None:
            return None
        q = self.pose.pose.orientation
        return enu_yaw_from_quaternion(q.x, q.y, q.z, q.w)

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

        YAW IS AN ABSOLUTE HEADING, NOT AN OFFSET. This field used to be a
        hard-coded 0.0, which is not "keep the current heading" — it is a
        command to point at ENU yaw 0 (due East), and the vehicle obeyed it:
        on the first flight it spun on the spot to that heading before it would
        climb. So the heading is captured on the pad at arming and re-sent
        unchanged for the rest of the flight; before it is captured, each
        setpoint carries the vehicle's own live heading, which is a no-op.
        """
        if self.bench:
            return          # the bench never streams setpoints — see _declare
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
        yaw = self._yaw_hold if self._yaw_hold is not None else self._yaw_now()
        if yaw is None:
            # No pose yet, so there is no heading to hold and none to command.
            # Ask for zero yaw RATE instead of an absolute heading: "do not
            # rotate" is the only honest instruction when we cannot say where
            # the vehicle is pointing.
            m.type_mask = ((m.type_mask & ~PositionTarget.IGNORE_YAW_RATE)
                           | PositionTarget.IGNORE_YAW)
            m.yaw_rate = 0.0
        else:
            m.yaw = float(yaw)
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
        # A POSE IS NOT AN ESTIMATE. MAVROS keeps publishing local_position even
        # when the EKF has fallen back to constant-position mode with no GNSS
        # fusion at all, so the check above says nothing about whether the
        # vehicle can hold position — and PX4 refuses the OFFBOARD arm anyway,
        # over an event MAVROS cannot decode. estimator.py has the numbers from
        # the day this was found.
        now = self._now()
        blocked = self.ekf.blocking_reason(now)
        detail = blocked or self.ekf.warning(now) or self.ekf.summary(now)
        if self.require_gnss:
            c.append(CheckResult('EKF position aiding', blocked is None, detail))
        elif blocked:
            # Waived, but never silent: the operator turned this gate off (see
            # run_px4), so the EKF state becomes information they read instead
            # of a condition that holds the mission — PX4's own arm check is
            # still there, and it is the one that decides.
            self.get_logger().warn(
                f'EKF position aiding is NOT gating this flight '
                f'(require_gnss_aiding=false): {detail}',
                throttle_duration_sec=10.0)
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

        # The gimbal belongs to SEARCH (sweeping) and DESCEND (tracking) only.
        # Anywhere else it goes back to siyi_gimbal_node's own nadir hold —
        # including after an abort, so the camera is not left staring off at
        # some search sector while the vehicle comes down.
        if ph not in (Phase.SEARCH, Phase.DESCEND):
            self._release_aim()
        if ph is not Phase.SEARCH:
            self._scanning = False

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
            err = self._takeoff_target() - self._alt()
            if abs(err) <= self.alt_tol:
                self._send(0.0, 0.0, 0.0)
                self.gate.altitude_reached()
                self._announce()
                return
            # FULL climb rate until `climb_ease` from the target, then linear
            # to zero. The old rule was `clip(err, ±climb_speed)` — a pure P
            # law with a gain of 1 — which meant the vehicle was already
            # backing off `climb_speed` metres out and spent the last stretch
            # drifting up. Braking distance and climb rate are separate
            # decisions, so they are separate numbers.
            vz = self.climb_speed * float(np.clip(err / self.climb_ease,
                                                  -1.0, 1.0))
            self._send(0.0, 0.0, vz)
            return

        if ph is Phase.SEARCH:
            self._send(0.0, 0.0, 0.0)
            # THE VEHICLE HOLDS STILL AND THE CAMERA LOOKS AROUND. Sweeping the
            # gimbal searches a circle a couple of vehicle-heights across from a
            # stationary hover; flying a search pattern to cover the same ground
            # would put a moving vehicle over unsurveyed terrain to find out
            # something a 20 gram gimbal can find out from where it already is.
            self._scan_tick()
            # Commit only after several CONSECUTIVE fresh detections.  A live
            # fix AND a currently-asserted `detected` flag both have to hold;
            # a single spurious hit trips one tick and the streak resets, so it
            # cannot start an irreversible descent on its own.
            if self._fresh_marker() and self.detected:
                self._acq_streak += 1
            else:
                self._acq_streak = 0
            if self.bench:
                # Report and keep sweeping. Committing here would be a descent
                # command from a vehicle standing on the ground; and stopping
                # at the first sighting would end the rehearsal before the
                # sweep it exists to watch has finished a lap.
                self._bench_report()
                if self._acq_streak == self.acquire_frames:
                    self.get_logger().info(
                        f'BENCH: this is where the flight would COMMIT to a '
                        f'descent — {self._acq_streak} consecutive fixes, '
                        f'gimbal at yaw {self._gimbal_angles()[0]:+.0f} pitch '
                        f'{self._gimbal_angles()[1]:+.0f} deg')
                return
            if self._acq_streak >= self.acquire_frames:
                self._t_solve = None
                self._ref = HorizonReference(lead_s=self.mpc_dt)
                self._found_at = self._gimbal_angles()
                self.gate.marker_acquired()
                # Say WHERE it was found, in both the angle it was seen at and
                # the position that angle put it at: off nadir those are the two
                # halves of the fix, and if the descent then flies somewhere
                # unexpected this line is what says which half was wrong.
                offset = (self.marker[:2] - np.array(
                    [self.pose.pose.position.x, self.pose.pose.position.y])
                ) if self.pose is not None and self.marker is not None else None
                where = ('' if offset is None else
                         f' at {np.linalg.norm(offset):.1f} m '
                         f'(E{offset[0]:+.1f} N{offset[1]:+.1f})')
                self.get_logger().info(
                    f'marker acquired ({self._acq_streak} consecutive fixes) '
                    f'with the gimbal at yaw {self._found_at[0]:+.0f} '
                    f'pitch {self._found_at[1]:+.0f} deg{where} — '
                    f'descending automatically from here')
                return
            if self._now() - self._t_phase > self.search_timeout:
                self.gate.abort(f'no marker within {self.search_timeout:.0f} s')
            return

        if ph is Phase.DESCEND:
            # Keep the camera on the marker while closing on it. The aim walks
            # back to nadir on its own as the vehicle arrives overhead.
            self._track_marker()
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
