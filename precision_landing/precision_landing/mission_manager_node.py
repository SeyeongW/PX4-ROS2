#!/usr/bin/env python3
"""Truck-launch mission manager (PX4 / MAVROS, OFFBOARD mode).

The high-level mission brain that sits ABOVE the validated precision_landing
controller. It owns the whole flight life-cycle and hands the terminal
return-and-land phase off to precision_landing_node — only one node ever streams
setpoints at a time (single setpoint authority).

This node is deliberately **simulator-agnostic** (pure ROS/MAVROS): all Gazebo
glue (driving the truck, releasing the DetachableJoint, faking GPS-link loss)
lives in moving_marker_node / the world SDF, so this brain ports to real
hardware unchanged.

State machine
  MOUNTED   : disarmed, riding the truck. Wait until the truck (broadcast on
              /marker/position) drives within trigger_dist of the mission area,
              then launch. Arming releases the DetachableJoint (moving_marker_node
              watches /mavros/state), so the drone lifts off FROM the moving truck.
  TAKEOFF   : stream setpoints (required before PX4 accepts OFFBOARD) -> OFFBOARD
              -> arm -> climb via a velocity setpoint to flight_alt. Unlike
              ArduPilot's GUIDED+CommandTOL, PX4 OFFBOARD does the whole climb
              under our own velocity command, so this stage never stops streaming.
  MISSION   : fly a patrol route (patrol_route: path to a waypoint-list YAML,
              looped indefinitely, dwelling loiter_s at each stop) or, if
              patrol_route is unset, the single (mission_area_e,
              mission_area_n) waypoint (legacy: dwell loiter_s then RETURN).
              Each tick re-evaluates the RETURN decision (battery budget vs.
              distance to the moving truck, an external task-complete signal,
              or -- single-waypoint case only -- the dwell timer) and watches
              the truck link.
  LANDING   : hand off — latch /mission/land_enable=true and STOP streaming
              setpoints. precision_landing_node (dormant until now) wakes and runs
              APPROACH->ALIGN->DESCEND->DONE onto the moving truck. We keep
              watching the link; if it drops we revoke the handoff and SAFE_LAND.
  LINK_LOST : the truck cue went stale (simulated GPS-link loss). Hover in place
              and try to reconnect for reconnect_window_s; if it comes back resume,
              otherwise SAFE_LAND.
  SAFE_LAND : land where we are (AUTO.LAND, vertical descent). Terrain-flatness
              evaluation is a vision-only PASS-THROUGH STUB here — flat SITL can't
              exercise it; real discrimination is deferred to hardware LiDAR.
  DONE      : disarmed on the ground / platform. Idle (keep publishing telemetry).

Topic contract
  in  /mavros/state                 mavros_msgs/State
  in  /mavros/local_position/pose   geometry_msgs/PoseStamped (BEST_EFFORT)
  in  /marker/position              geometry_msgs/PointStamped  truck ENU cue
  in  /marker/velocity              geometry_msgs/Vector3Stamped (optional)
  in  /mission/task_complete        std_msgs/Bool               external mission-complete signal
  out /mavros/setpoint_raw/local    mavros_msgs/PositionTarget (TAKEOFF/MISSION/LINK_LOST)
  out /mission/land_enable          std_msgs/Bool (latched)  landing-authority gate
  out /mission/phase                std_msgs/String          current FSM stage
  out /mission/battery_s            std_msgs/Float32         simulated battery left
"""

import math
from enum import Enum

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from std_msgs.msg import Bool, Float32, String
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, ParamSetV2, SetMode
from rcl_interfaces.msg import ParameterValue, ParameterType

from precision_landing import apf
from precision_landing import planner
from precision_landing import mpc as mpc_solver


class Stage(Enum):
    MOUNTED = 0
    TAKEOFF = 1
    MISSION = 2
    LANDING = 3
    LINK_LOST = 4
    SAFE_LAND = 5
    DONE = 6


DT = 0.05  # 50 ms control period
# PX4 rejects an OFFBOARD mode request unless setpoints are already streaming;
# prime for this many ticks (1 s @ 20 Hz) before requesting the switch.
OFFBOARD_PRIME_TICKS = 20


def _load_patrol_route(path):
    """Waypoint list YAML -> [(e, n), ...]. Format: `waypoints: [{e:, n:}, ...]`
    (same convention as gazebo/config/obstacle_map.yaml's obstacle list)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return [(float(w['e']), float(w['n'])) for w in data['waypoints']]


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        # --- Parameters -----------------------------------------------------
        self.flight_alt = self.declare_parameter('flight_alt', 5.0).value
        # Mission area: a fixed ENU point used as (a) the MOUNTED-stage launch-
        # trigger anchor (trigger_dist below) and (b) the legacy single
        # waypoint when patrol_route (below) is not set.
        self.mission_e = self.declare_parameter('mission_area_e', 120.0).value
        self.mission_n = self.declare_parameter('mission_area_n', 40.0).value
        # Launch trigger: take off once the truck drives within this distance (m)
        # of the mission area.
        self.trigger_dist = self.declare_parameter('trigger_dist', 50.0).value
        # Dwell time (s) at each patrol waypoint (or the single legacy
        # waypoint) before moving on / returning.
        self.loiter_s = self.declare_parameter('loiter_s', 15.0).value
        # Considered "arrived" at a waypoint within this radius (m).
        self.arrive_radius = self.declare_parameter('arrive_radius', 2.0).value
        # Horizontal velocity servo toward the current (stationary) waypoint.
        self.vel_gain = self.declare_parameter('vel_gain', 0.6).value      # 1/s
        self.vel_max = self.declare_parameter('vel_max', 8.0).value        # m/s
        # --- Patrol route (Step 1: waypoint-sequencing skeleton, no new path-
        # planning algorithm yet -- still the same P+APF velocity servo) -----
        # Empty (default): legacy single-point behaviour -- the one waypoint is
        # (mission_area_e, mission_area_n) and the dwell timer itself ends the
        # mission (RETURN), same as before this feature existed.
        # Non-empty: path to a YAML `waypoints: [{e:, n:}, ...]` list (see
        # gazebo/config/patrol_route.yaml). Loops the list indefinitely,
        # dwelling loiter_s at each stop; only low_battery or the external
        # /mission/task_complete signal ends it (there is no single "done"
        # point to time out on during a patrol).
        self.patrol_route_path = self.declare_parameter('patrol_route', '').value
        self.patrol_mode = bool(self.patrol_route_path)
        if self.patrol_mode:
            try:
                self.patrol_waypoints = _load_patrol_route(self.patrol_route_path)
                if not self.patrol_waypoints:
                    raise ValueError('waypoints list is empty')
                self.get_logger().info(
                    f'patrol: loaded {len(self.patrol_waypoints)} waypoints from '
                    f'{self.patrol_route_path}')
            except Exception as e:
                self.get_logger().error(
                    f'patrol: failed to load patrol_route "{self.patrol_route_path}": '
                    f'{e}; falling back to single mission_area waypoint')
                self.patrol_mode = False
                self.patrol_waypoints = [(self.mission_e, self.mission_n)]
        else:
            self.patrol_waypoints = [(self.mission_e, self.mission_n)]
        # --- Obstacle avoidance (APF baseline) -------------------------------
        # Off by default. Same repulsion model as precision_landing_node — see
        # apf.py / gazebo/config/obstacle_map.yaml. Layered onto _servo_to's
        # attractive term for the MISSION (outbound patrol) leg.
        self.apf_enable = self.declare_parameter('apf_enable', False).value
        self.obstacle_map_path = self.declare_parameter('obstacle_map', '').value
        self.apf_influence_radius = self.declare_parameter('apf_influence_radius', 15.0).value  # m
        self.apf_gain = self.declare_parameter('apf_gain', 6.0).value                  # m/s at the surface
        self.apf_vel_cap = self.declare_parameter('apf_vel_cap', 6.0).value            # m/s
        # --- Obstacle avoidance (MPC, roadmap step 2) ------------------------
        # Off by default; mutually meaningful with apf_enable off (both patch
        # the same MISSION-leg servo call — see _mpc_servo_to vs _servo_to).
        # Front-end (A*) + back-end (safe flight corridor) live in planner.py;
        # the QP-ish solve itself lives in mpc.py. obstacle_map (same file APF
        # uses) doubles as the corridor's known-obstacle source.
        self.mpc_enable = self.declare_parameter('mpc_enable', False).value
        self.mpc_horizon = self.declare_parameter('mpc_horizon', 10).value
        self.mpc_dt = self.declare_parameter('mpc_dt', 0.3).value              # s per horizon step
        self.mpc_vmax = self.declare_parameter('mpc_vmax', 0.0).value          # 0 -> fall back to vel_max
        # How often (s) to re-run A*+corridor from the current position toward
        # the current target. Cheap A* on this grid is milliseconds, but there
        # is no need to re-search every 50 ms tick when the target (a static
        # patrol waypoint) hasn't moved -- also re-planned immediately if the
        # drone drifts outside the current corridor box (see _mpc_servo_to).
        self.mpc_replan_period_s = self.declare_parameter('mpc_replan_period_s', 2.0).value
        self.mpc_corridor_margin = self.declare_parameter('mpc_corridor_margin', 1.5).value  # m
        self.mpc_q_track = self.declare_parameter('mpc_q_track', 1.0).value
        self.mpc_r_smooth = self.declare_parameter('mpc_r_smooth', 0.08).value
        self.mpc_r_effort = self.declare_parameter('mpc_r_effort', 0.01).value
        # Soft corridor-violation penalty weight -- deliberately large relative
        # to q_track so the solver strongly prefers staying inside the box over
        # cutting a corner toward the target (see mpc.py module docstring for
        # why this is a soft penalty rather than a hard QP constraint).
        self.mpc_corridor_weight = self.declare_parameter('mpc_corridor_weight', 40.0).value
        self.mpc_iters = self.declare_parameter('mpc_iters', 60).value
        # Re-solving the QP every 50 ms tick is unnecessary and, under a hard
        # replan (target just outside the fresh corridor box), the line-search
        # solve can take tens of ms -- measured up to ~65 ms in a stress test,
        # more than one whole tick period. Decouple: actually re-solve at most
        # this often, streaming the last solved (vE,vN) at full tick rate in
        # between (same pattern real MPC/ROS integrations use -- fast setpoint
        # stream, slower re-optimization).
        self.mpc_solve_period_s = self.declare_parameter('mpc_solve_period_s', 0.15).value
        self._mpc_corridor_box = None       # planner.Box currently constraining the servo
        self._mpc_corridor_target = None    # (e, n) the current box was planned toward
        self._mpc_local_target = None       # (e, n) next A* waypoint -- what the QP actually tracks
        self._mpc_last_replan = None        # rclpy Time of the last A*+corridor solve
        self._mpc_last_solve = None         # rclpy Time of the last actual QP solve
        self._mpc_last_cmd = (0.0, 0.0)     # cached (vE, vN) between solves
        if self.mpc_enable and not self.obstacle_map_path:
            self.get_logger().warn(
                'mpc_enable=true but obstacle_map is unset -- MPC will run with '
                'no corridor (unconstrained tracking only).')
        # --- Obstacle list, shared by APF and MPC's safety-net repulsion ----
        # Loaded whenever EITHER avoidance mode needs it. MPC layers a light
        # APF repulsion on top of its own command (_mpc_servo_to) as a
        # redundant safety net: if A* ever fails to find ANY corridor to the
        # requested target (e.g. the target itself sits inside another
        # obstacle's inflated margin -- happens with the legacy single-point
        # test aimed at obstacle_5, which is also close to obstacle_14's
        # margin), _mpc_maybe_replan has nothing to constrain the QP with and
        # would otherwise fly there completely unconstrained. Found by a
        # closed-loop collision stress test, not by inspection -- an early
        # version of this file actually crashed into obstacle_5 that way.
        self._obstacles = []
        if (self.apf_enable or self.mpc_enable) and self.obstacle_map_path:
            try:
                self._obstacles = apf.load_obstacles(self.obstacle_map_path)
                self.get_logger().info(
                    f'obstacles: loaded {len(self._obstacles)} from {self.obstacle_map_path}')
            except Exception as e:
                self.get_logger().error(
                    f'obstacles: failed to load obstacle_map "{self.obstacle_map_path}": {e}')
        # --- Simulated battery budget (param-driven, predictable for the demo) --
        # battery_s counts DOWN from battery_capacity_s at consume_rate per second
        # while airborne. RETURN is triggered when the time still needed to reach
        # the (moving) truck plus a safety reserve exceeds the battery left.
        self.battery_capacity_s = self.declare_parameter('battery_capacity_s', 120.0).value
        self.consume_rate = self.declare_parameter('consume_rate', 1.0).value  # s/s
        self.reserve_margin_s = self.declare_parameter('reserve_margin_s', 10.0).value
        # Assumed return cruise speed used for the time-to-truck estimate (m/s).
        self.effective_speed = self.declare_parameter('effective_speed', 8.0).value
        # --- Truck link (cue) freshness / failsafe --------------------------
        # cue_timeout: a cue older than this counts as a (momentary) dropout.
        # reconnect_window_s: how long to keep trying to reconnect (hover) before
        # giving up and landing safely where we are.
        self.cue_timeout = self.declare_parameter('cue_timeout', 2.0).value
        self.reconnect_window_s = self.declare_parameter('reconnect_window_s', 5.0).value
        # Point the nose along travel above this speed (else hold heading).
        self.yaw_track_min_speed = self.declare_parameter('yaw_track_min_speed', 0.5).value

        self.stage = Stage.MOUNTED

        # --- State ----------------------------------------------------------
        self.mav_state = State()
        self.pos = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.hold_yaw = 0.0
        self.cmd_vel = [0.0, 0.0, 0.0]
        self.cue = None
        self.cue_stamp = None
        self.req_ticks = 0
        self.battery_s = self.battery_capacity_s
        self.arrived = False
        self.loiter_start = None
        self.patrol_idx = 0             # index into self.patrol_waypoints
        self.mission_complete = False   # latest value from /mission/task_complete
        self.link_lost_start = None
        self.resume_stage = None        # stage to return to after a recovered link
        self.land_enabled = False
        self.was_armed = False
        self.dbg_tick = 0

        # --- Publishers -----------------------------------------------------
        self.raw_pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10)
        # Latched (transient_local) so precision_landing_node gets the current
        # authority state even if it subscribes after we publish.
        latched = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.enable_pub = self.create_publisher(Bool, '/mission/land_enable', latched)
        self.phase_pub = self.create_publisher(String, '/mission/phase', 10)
        self.battery_pub = self.create_publisher(Float32, '/mission/battery_s', 10)
        # Publish the initial (dormant) authority state right away.
        self._set_land_enable(False)

        # --- Subscribers ----------------------------------------------------
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb,
            qos_profile_sensor_data)
        self.create_subscription(PointStamped, '/marker/position', self._cue_cb, 10)
        self.create_subscription(
            Bool, '/mission/task_complete', self._task_complete_cb, 10)

        # --- Service clients ------------------------------------------------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.param_set_cli = self.create_client(ParamSetV2, '/mavros/param/set')
        self._nav_dll_act_attempts = 0
        self._param_fix_tick = 0

        self.create_timer(DT, self.tick)
        patrol_desc = (f'patrol={len(self.patrol_waypoints)}wp (loops)' if self.patrol_mode
                       else 'patrol=off (single waypoint)')
        self._log(f'mission area=({self.mission_e:.0f},{self.mission_n:.0f}) '
                  f'trigger_dist={self.trigger_dist:.0f} loiter={self.loiter_s:.0f}s '
                  f'battery={self.battery_capacity_s:.0f}s {patrol_desc}')

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

    def _cue_cb(self, msg):
        self.cue = [msg.point.x, msg.point.y]
        self.cue_stamp = self.get_clock().now()

    def _cue_fresh(self):
        if self.cue is None or self.cue_stamp is None:
            return False
        age = (self.get_clock().now() - self.cue_stamp).nanoseconds * 1e-9
        return age < self.cue_timeout

    def _task_complete_cb(self, msg):
        self.mission_complete = msg.data

    # -----------------------------------------------------------------------
    def tick(self):
        if not self.mav_state.connected:
            return

        # PX4 refuses to arm without EITHER an RC link OR a GCS heartbeat
        # (NAV_DLL_ACT > 0, its default) -- MAVROS registers as a companion
        # link, not a GCS, so it never satisfies this on its own (confirmed
        # via `commander check`: "No connection to the ground control
        # station" / "Arming denied: Resolve system health failures first"
        # until this is cleared). We have neither an RC nor a real GCS in
        # this setup, so disable the requirement. Fire a few times (once
        # isn't guaranteed to land before the service/link is fully up).
        self._param_fix_tick += 1
        if self._nav_dll_act_attempts < 5 and self._param_fix_tick % 20 == 1:
            self._nav_dll_act_attempts += 1
            self._set_param_int('NAV_DLL_ACT', 0)

        # Deplete the simulated battery whenever airborne (armed).
        if self.mav_state.armed:
            self.battery_s = max(0.0, self.battery_s - self.consume_rate * DT)

        # Detect end-of-mission: we were airborne and are now disarmed (precision_
        # landing force-disarmed onto the truck, or SAFE_LAND/LAND touched down).
        if self.was_armed and not self.mav_state.armed and \
                self.stage in (Stage.LANDING, Stage.SAFE_LAND):
            self._log('disarmed -> DONE (mission complete)')
            self.stage = Stage.DONE
        self.was_armed = self.mav_state.armed

        self.cmd_vel = [0.0, 0.0, 0.0]
        publish = False        # whether to stream a setpoint this tick

        if self.stage == Stage.MOUNTED:
            # Wait for the truck to drive within trigger_dist of the mission area.
            if self._cue_fresh():
                d = self._dist(self.cue, [self.mission_e, self.mission_n])
                self.dbg_tick += 1
                if self.dbg_tick % 40 == 0:
                    self._log(f'MOUNTED: truck {d:.0f} m from mission area '
                              f'(trigger at {self.trigger_dist:.0f})')
                if d <= self.trigger_dist:
                    self._log('truck near mission area -> TAKEOFF')
                    self.battery_s = self.battery_capacity_s
                    self.mission_complete = False   # clear any stale signal from last run
                    self.patrol_idx = 0
                    self.arrived = False
                    self.loiter_start = None
                    self.stage = Stage.TAKEOFF

        elif self.stage == Stage.TAKEOFF:
            # PX4 OFFBOARD requires setpoints to already be streaming before
            # (and continuously during/after) the mode switch, so we publish
            # every TAKEOFF tick from the very first one and never stop --
            # arm happens once already in OFFBOARD, then we climb under our
            # own velocity command (no separate AUTO.TAKEOFF hop, unlike the
            # old GUIDED+CommandTOL flow this replaces).
            publish = True
            self.req_ticks += 1
            send = (self.req_ticks % 40 == 1)
            if self.mav_state.mode != 'OFFBOARD':
                if self.req_ticks >= OFFBOARD_PRIME_TICKS and send:
                    self._log('TAKEOFF: request OFFBOARD')
                    self._set_mode('OFFBOARD')
            elif not self.mav_state.armed:
                if send:
                    self._log('TAKEOFF: arming (releases truck joint)')
                    self._arm(True)
            elif self.pos[2] < self.flight_alt - 0.5:
                # Fixed climb rate (not proportional -- error stays large from
                # the ground) up to flight_alt.
                self.cmd_vel[2] = 1.0
            else:
                self._log('-> MISSION (reached flight_alt)')
                self.stage = Stage.MISSION

        elif self.stage == Stage.MISSION:
            if not self.mav_state.armed:
                self._log('disarmed in MISSION -> DONE')
                self.stage = Stage.DONE
            elif not self._cue_fresh():
                # Lost the truck link -> try to reconnect (Phase 3 failsafe).
                self._enter_link_lost(Stage.MISSION)
            else:
                # Fly to the current patrol waypoint, dwell loiter_s, then
                # advance to the next one (looping). Legacy single-waypoint
                # case: patrol_waypoints has exactly one entry, so "advance"
                # is a no-op and dwell_done doubles as the old loiter_done.
                wp = self.patrol_waypoints[self.patrol_idx]
                d_area = self._dist(self.pos, wp)
                if not self.arrived and d_area < self.arrive_radius:
                    self.arrived = True
                    self.loiter_start = self.get_clock().now()
                    self._log(f'arrived at waypoint {self.patrol_idx} '
                              f'-> dwell {self.loiter_s:.0f}s')
                if self.mpc_enable:
                    self._mpc_servo_to(wp[0], wp[1])
                else:
                    self._servo_to(wp[0], wp[1])
                self.cmd_vel[2] = self._clamp(
                    0.5 * (self.flight_alt - self.pos[2]), -0.5, 0.5)
                self._face_velocity()
                publish = True

                dwell_done = self.arrived and self.loiter_start is not None and \
                    (self.get_clock().now() - self.loiter_start).nanoseconds * 1e-9 \
                    >= self.loiter_s
                if dwell_done and len(self.patrol_waypoints) > 1:
                    self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_waypoints)
                    self.arrived = False
                    self.loiter_start = None
                    self._log(f'-> advance to waypoint {self.patrol_idx}')

                # RETURN decision: battery budget vs. distance to the moving
                # truck, an external task-complete signal, or -- legacy
                # single-waypoint case only -- the dwell timer itself (patrol
                # mode loops indefinitely; there's no single "done" point to
                # time out on, so only battery/task_complete end it).
                d_truck = self._dist(self.pos, self.cue)
                t_return = d_truck / max(self.effective_speed, 0.1)
                low_battery = self.battery_s <= t_return + self.reserve_margin_s
                loiter_done = dwell_done and not self.patrol_mode
                self.dbg_tick += 1
                if self.dbg_tick % 20 == 0:
                    self._log(f'MISSION batt={self.battery_s:.0f}s '
                              f'd_truck={d_truck:.0f} t_ret={t_return:.0f}s '
                              f'{"[LOW BATT]" if low_battery else ""}')
                if low_battery or loiter_done or self.mission_complete:
                    why = ('low battery' if low_battery else
                           'task complete' if self.mission_complete else 'loiter done')
                    self._log(f'RETURN ({why}) -> LANDING handoff')
                    self.stage = Stage.LANDING

        elif self.stage == Stage.LANDING:
            # Hand the terminal return+land to precision_landing_node: latch the
            # authority gate and STOP streaming setpoints (single authority).
            self._set_land_enable(True)
            if not self._cue_fresh():
                # Link dropped during the landing -> revoke handoff, SAFE_LAND.
                self._log('link lost during LANDING')
                self._enter_link_lost(Stage.LANDING)
            # Completion (disarm) is handled by the was_armed edge above.

        elif self.stage == Stage.LINK_LOST:
            # Hover and wait for the truck cue to come back.
            elapsed = (self.get_clock().now() - self.link_lost_start).nanoseconds * 1e-9
            if self._cue_fresh():
                self._log(f'link restored -> resume {self.resume_stage.name}')
                self.stage = self.resume_stage
            elif elapsed >= self.reconnect_window_s:
                self._log(f'no reconnect in {self.reconnect_window_s:.0f}s -> SAFE_LAND')
                self.stage = Stage.SAFE_LAND
            else:
                # Hold position (zero horizontal velocity, hold altitude).
                self.cmd_vel = [0.0, 0.0, 0.0]
                publish = True

        elif self.stage == Stage.SAFE_LAND:
            # Release the landing controller and land vertically where we are.
            # Terrain-flatness check is a stub (flat SITL -> always "safe").
            self._set_land_enable(False)
            self.req_ticks += 1
            if self.mav_state.armed:
                if self.req_ticks % 40 == 1:
                    self._log('SAFE_LAND: flat terrain assumed safe -> AUTO.LAND mode')
                if self.mav_state.mode != 'AUTO.LAND':
                    self._set_mode('AUTO.LAND')
            # Completion (disarm) handled by the was_armed edge above.

        elif self.stage == Stage.DONE:
            self._set_land_enable(False)

        if publish:
            self._publish_velocity()

        self.phase_pub.publish(String(data=self.stage.name))
        self.battery_pub.publish(Float32(data=float(self.battery_s)))

    # -----------------------------------------------------------------------
    def _enter_link_lost(self, resume_stage):
        self.resume_stage = resume_stage
        self.link_lost_start = self.get_clock().now()
        self._set_land_enable(False)     # take authority back from the landing node
        self._log(f'-> LINK_LOST (was {resume_stage.name}); reconnecting...')
        self.stage = Stage.LINK_LOST

    def _set_land_enable(self, value):
        # Idempotent latch publish; only log on change.
        if value != self.land_enabled:
            self._log(f'land_enable = {value}')
        self.land_enabled = value
        self.enable_pub.publish(Bool(data=value))

    # First-order velocity servo toward a stationary ENU point; sets cmd_vel.
    def _servo_to(self, target_e, target_n):
        eE = target_e - self.pos[0]
        eN = target_n - self.pos[1]
        vE = self.vel_gain * eE
        vN = self.vel_gain * eN
        if self.apf_enable and self._obstacles:
            rE, rN = apf.repulsive_velocity(
                self.pos[0], self.pos[1], self._obstacles,
                self.apf_influence_radius, self.apf_gain, self.apf_vel_cap)
            vE += rE
            vN += rN
        # Clamp as a 2D vector (not per-axis) so the commanded DIRECTION is
        # preserved when the combined speed exceeds vel_max.
        speed = math.hypot(vE, vN)
        if speed > self.vel_max and speed > 1e-6:
            scale = self.vel_max / speed
            vE *= scale
            vN *= scale
        self.cmd_vel[0] = vE
        self.cmd_vel[1] = vN
        return math.hypot(eE, eN)

    # A*(front-end) -> safe-flight-corridor(back-end) -> MPC(QP-ish) servo
    # toward a stationary ENU point; sets cmd_vel. Same role/signature as
    # _servo_to (drop-in swap, gated by mpc_enable), targeting the SAME
    # MISSION-leg waypoint so apf_enable/mpc_enable stay directly comparable
    # for benchmarking. target_vel is (0,0) for every current caller (patrol
    # waypoints and the legacy mission_area point are both static) -- mpc.py
    # already supports a moving target/feed-forward for when this gets wired
    # into a trailer-chasing leg later.
    def _mpc_servo_to(self, target_e, target_n, target_vel_e=0.0, target_vel_n=0.0):
        self._mpc_maybe_replan(target_e, target_n)

        now = self.get_clock().now()
        need_solve = (self._mpc_last_solve is None or
                     (now - self._mpc_last_solve).nanoseconds * 1e-9
                     >= self.mpc_solve_period_s)
        if need_solve:
            vmax = self.mpc_vmax if self.mpc_vmax > 0.0 else self.vel_max
            # Track the LOCAL target (next A* waypoint), not the far goal --
            # see the comment in _mpc_maybe_replan for why. target_vel is
            # still the FAR target's feed-forward; harmless since every
            # current caller passes (0,0) (both patrol waypoints and the
            # legacy mission_area point are static) -- revisit if this gets
            # wired to a genuinely moving target later.
            local = self._mpc_local_target or (target_e, target_n)
            vE, vN = mpc_solver.solve(
                pos=(self.pos[0], self.pos[1]), vel=(self.cmd_vel[0], self.cmd_vel[1]),
                target=local, target_vel=(target_vel_e, target_vel_n),
                corridor_box=self._mpc_corridor_box, vmax=vmax, dt=self.mpc_dt,
                horizon=self.mpc_horizon, q_track=self.mpc_q_track,
                r_smooth=self.mpc_r_smooth, r_effort=self.mpc_r_effort,
                corridor_weight=self.mpc_corridor_weight, iters=self.mpc_iters)
            self._mpc_last_cmd = (vE, vN)
            self._mpc_last_solve = now
        else:
            vE, vN = self._mpc_last_cmd

        # Safety-net APF repulsion on top of the MPC command -- but ONLY when
        # there is NO valid corridor box at all (A* has never succeeded from
        # here), not unconditionally. Found by closed-loop simulation: running
        # APF's repulsion at full strength ALONGSIDE an already-working
        # corridor reintroduces APF's own local-minimum weakness right back
        # into the box that was specifically shaped to avoid it -- the drone
        # stalled indefinitely 11+ m short of a corridor-internal local
        # target, MPC's pull and the safety-net's push exactly cancelling
        # next to obstacle_2/8. The corridor is only genuinely unconstrained
        # (needs the net) when _mpc_corridor_box is None, e.g. the requested
        # target itself sits inside another obstacle's inflated margin (the
        # legacy mission_area=(20,16) test, close to obstacle_14's margin).
        if self._mpc_corridor_box is None and self._obstacles:
            rE, rN = apf.repulsive_velocity(
                self.pos[0], self.pos[1], self._obstacles,
                self.apf_influence_radius, self.apf_gain, self.apf_vel_cap)
            vE += rE
            vN += rN

        # Belt-and-braces vector clamp to vel_max, matching _servo_to's style.
        speed = math.hypot(vE, vN)
        if speed > self.vel_max and speed > 1e-6:
            scale = self.vel_max / speed
            vE *= scale
            vN *= scale
        self.cmd_vel[0] = vE
        self.cmd_vel[1] = vN
        return math.hypot(target_e - self.pos[0], target_n - self.pos[1])

    # Re-run A*+corridor when: no box yet, the target changed (new patrol
    # waypoint), the replan timer elapsed, the drone has drifted outside the
    # current box, or it has reached the current local target (see below).
    # Keeps the LAST good box/local-target on a failed/no-path replan rather
    # than going unconstrained, unless there never was one.
    def _mpc_maybe_replan(self, target_e, target_n):
        now = self.get_clock().now()
        target_changed = (self._mpc_corridor_target is None or
                          self._dist((target_e, target_n), self._mpc_corridor_target) > 0.5)
        timer_elapsed = (self._mpc_last_replan is None or
                         (now - self._mpc_last_replan).nanoseconds * 1e-9
                         >= self.mpc_replan_period_s)
        drifted_out = (self._mpc_corridor_box is not None and not
                       self._mpc_corridor_box.inflated(-0.5).contains(self.pos[0], self.pos[1]))
        reached_local = (self._mpc_local_target is not None and
                         self._dist(self.pos[:2], self._mpc_local_target) < self.arrive_radius)
        if not (target_changed or timer_elapsed or drifted_out or reached_local):
            return
        self._mpc_last_replan = now
        self._mpc_corridor_target = (target_e, target_n)
        if not self.obstacle_map_path:
            self._mpc_local_target = (target_e, target_n)   # no map -> aim straight at it
            return
        try:
            _, simplified, corridor = planner.plan(
                (self.pos[0], self.pos[1]), (target_e, target_n),
                self.obstacle_map_path, safety_margin=self.mpc_corridor_margin)
        except Exception as e:
            self.get_logger().error(f'MPC: corridor replan failed: {e}')
            return
        if corridor is None:
            self.get_logger().warn(
                f'MPC: no A* path from {self.pos[:2]} to ({target_e:.1f},{target_n:.1f}) '
                '-- keeping previous corridor box/local target')
            return
        self._mpc_corridor_box = corridor[0]   # box seeded at the START, i.e. here-and-now
        # AIM AT THE NEXT BEND, NOT THE FAR GOAL. A single box only bounds
        # local freedom of movement -- it carries none of A*'s routing
        # information about WHICH WAY to detour. Servoing straight at a
        # distant final target inside that box walks the tracking term right
        # back into the same straight-line-at-the-obstacle local minimum APF
        # alone has (documented in research_direction/obstacle_field_world) --
        # found by closed-loop simulation: the drone stalled 12+ m short of
        # goal, MPC's pull and the safety-net APF's push exactly cancelling
        # next to an obstacle the direct line passed close to. Targeting the
        # next waypoint of the (re-searched-from-HERE, so always current)
        # simplified path keeps the servo always aimed the way A* actually
        # wants to go.
        self._mpc_local_target = simplified[1] if len(simplified) > 1 else (target_e, target_n)

    def _face_velocity(self):
        vE, vN = self.cmd_vel[0], self.cmd_vel[1]
        if math.hypot(vE, vN) >= self.yaw_track_min_speed:
            self.hold_yaw = math.atan2(vN, vE)

    def _publish_velocity(self):
        pt = PositionTarget()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # mavros ENU->NED
        pt.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                        PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                        PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                        PositionTarget.IGNORE_YAW_RATE)
        pt.velocity.x = self.cmd_vel[0]
        pt.velocity.y = self.cmd_vel[1]
        pt.velocity.z = self.cmd_vel[2]
        pt.yaw = self.hold_yaw
        self.raw_pub.publish(pt)

    def _set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        self.set_mode_cli.call_async(req)

    def _arm(self, value):
        req = CommandBool.Request()
        req.value = value
        self.arming_cli.call_async(req)

    def _set_param_int(self, param_id, value):
        req = ParamSetV2.Request()
        req.param_id = param_id
        req.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
        self.param_set_cli.call_async(req)

    def _log(self, text):
        self.get_logger().info(text)

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
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
