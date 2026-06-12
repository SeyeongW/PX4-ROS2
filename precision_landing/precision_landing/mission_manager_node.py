#!/usr/bin/env python3
"""Truck-launch mission manager (ArduPilot / MAVROS, GUIDED mode).

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
  TAKEOFF   : GUIDED -> arm -> /mavros/cmd/takeoff to flight_alt (no setpoints;
              ArduCopter holds position itself after a guided takeoff).
  MISSION   : fly to the mission waypoint and loiter. Each tick re-evaluates the
              RETURN decision (battery budget vs. distance to the moving truck)
              and watches the truck link.
  LANDING   : hand off — latch /mission/land_enable=true and STOP streaming
              setpoints. precision_landing_node (dormant until now) wakes and runs
              APPROACH->ALIGN->DESCEND->DONE onto the moving truck. We keep
              watching the link; if it drops we revoke the handoff and SAFE_LAND.
  LINK_LOST : the truck cue went stale (simulated GPS-link loss). Hover in place
              and try to reconnect for reconnect_window_s; if it comes back resume,
              otherwise SAFE_LAND.
  SAFE_LAND : land where we are (LAND mode, vertical descent). Terrain-flatness
              evaluation is a vision-only PASS-THROUGH STUB here — flat SITL can't
              exercise it; real discrimination is deferred to hardware LiDAR.
  DONE      : disarmed on the ground / platform. Idle (keep publishing telemetry).

Topic contract
  in  /mavros/state                 mavros_msgs/State
  in  /mavros/local_position/pose   geometry_msgs/PoseStamped (BEST_EFFORT)
  in  /marker/position              geometry_msgs/PointStamped  truck ENU cue
  in  /marker/velocity              geometry_msgs/Vector3Stamped (optional)
  out /mavros/setpoint_raw/local    mavros_msgs/PositionTarget (MISSION/LINK_LOST)
  out /mission/land_enable          std_msgs/Bool (latched)  landing-authority gate
  out /mission/phase                std_msgs/String          current FSM stage
  out /mission/battery_s            std_msgs/Float32         simulated battery left
"""

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from std_msgs.msg import Bool, Float32, String
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


class Stage(Enum):
    MOUNTED = 0
    TAKEOFF = 1
    MISSION = 2
    LANDING = 3
    LINK_LOST = 4
    SAFE_LAND = 5
    DONE = 6


DT = 0.05  # 50 ms control period


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        # --- Parameters -----------------------------------------------------
        self.flight_alt = self.declare_parameter('flight_alt', 5.0).value
        # Mission area: a fixed ENU waypoint the drone flies to and loiters over.
        self.mission_e = self.declare_parameter('mission_area_e', 120.0).value
        self.mission_n = self.declare_parameter('mission_area_n', 40.0).value
        # Launch trigger: take off once the truck drives within this distance (m)
        # of the mission area.
        self.trigger_dist = self.declare_parameter('trigger_dist', 50.0).value
        # Loiter time (s) over the mission waypoint before returning.
        self.loiter_s = self.declare_parameter('loiter_s', 15.0).value
        # Considered "arrived" at the mission waypoint within this radius (m).
        self.arrive_radius = self.declare_parameter('arrive_radius', 2.0).value
        # Horizontal velocity servo toward the (stationary) mission waypoint.
        self.vel_gain = self.declare_parameter('vel_gain', 0.6).value      # 1/s
        self.vel_max = self.declare_parameter('vel_max', 8.0).value        # m/s
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

        # --- Service clients ------------------------------------------------
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_cli = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.create_timer(DT, self.tick)
        self._log(f'mission area=({self.mission_e:.0f},{self.mission_n:.0f}) '
                  f'trigger_dist={self.trigger_dist:.0f} loiter={self.loiter_s:.0f}s '
                  f'battery={self.battery_capacity_s:.0f}s')

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

    # -----------------------------------------------------------------------
    def tick(self):
        if not self.mav_state.connected:
            return

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
                    self.stage = Stage.TAKEOFF

        elif self.stage == Stage.TAKEOFF:
            # GUIDED -> arm -> takeoff, each re-sent ~every 2 s. Arming releases
            # the DetachableJoint (moving_marker_node). No setpoints during climb.
            self.req_ticks += 1
            send = (self.req_ticks % 40 == 1)
            if self.mav_state.mode != 'GUIDED':
                if send:
                    self._log('TAKEOFF: set GUIDED')
                    self._set_mode('GUIDED')
            elif not self.mav_state.armed:
                if send:
                    self._log('TAKEOFF: arming (releases truck joint)')
                    self._arm(True)
            elif self.pos[2] < self.flight_alt - 0.5:
                if send:
                    self._log(f'TAKEOFF: cmd takeoff to {self.flight_alt:.1f} m')
                    self._takeoff(self.flight_alt)
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
                # Fly to the mission waypoint, then loiter over it.
                d_area = self._dist(self.pos, [self.mission_e, self.mission_n])
                if not self.arrived and d_area < self.arrive_radius:
                    self.arrived = True
                    self.loiter_start = self.get_clock().now()
                    self._log(f'arrived at mission area -> loiter {self.loiter_s:.0f}s')
                self._servo_to(self.mission_e, self.mission_n)
                self.cmd_vel[2] = self._clamp(
                    0.5 * (self.flight_alt - self.pos[2]), -0.5, 0.5)
                self._face_velocity()
                publish = True

                # RETURN decision: battery budget vs. distance to the moving truck.
                d_truck = self._dist(self.pos, self.cue)
                t_return = d_truck / max(self.effective_speed, 0.1)
                low_battery = self.battery_s <= t_return + self.reserve_margin_s
                loiter_done = self.arrived and self.loiter_start is not None and \
                    (self.get_clock().now() - self.loiter_start).nanoseconds * 1e-9 \
                    >= self.loiter_s
                self.dbg_tick += 1
                if self.dbg_tick % 20 == 0:
                    self._log(f'MISSION batt={self.battery_s:.0f}s '
                              f'd_truck={d_truck:.0f} t_ret={t_return:.0f}s '
                              f'{"[LOW BATT]" if low_battery else ""}')
                if low_battery or loiter_done:
                    why = 'low battery' if low_battery else 'loiter done'
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
                    self._log('SAFE_LAND: flat terrain assumed safe -> LAND mode')
                if self.mav_state.mode != 'LAND':
                    self._set_mode('LAND')
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
        self.cmd_vel[0] = self._clamp(self.vel_gain * eE, -self.vel_max, self.vel_max)
        self.cmd_vel[1] = self._clamp(self.vel_gain * eN, -self.vel_max, self.vel_max)
        return math.hypot(eE, eN)

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

    def _takeoff(self, alt):
        req = CommandTOL.Request()
        req.altitude = float(alt)
        self.takeoff_cli.call_async(req)

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
