"""ROS 2 <-> MAVROS OFFBOARD bridge for the path_plan pipeline (PX4 SITL).

Flies the static-target cruise: PX4 spawn -> stationary trailer, tracking the
A* -> SFC -> B-spline -> MPC pipeline over MAVROS.

Moving-trailer pursuit (``pursuit_mode:=true``): instead of a one-shot plan to a
fixed goal, the CRUISE phase periodically republishes the trailer's LIVE position
(``/trailer/position``) to the A* planner's ``~/goal`` every ``replan_period_s``, so
the pipeline continuously replans toward the moving trailer (the live-flight version
of ``tools/pursuit_sim.py``). When the vehicle closes within ``terminal_range_m`` of
the trailer it latches ``/pursuit/land_enable`` and STOPS streaming setpoints, handing
the terminal land onto the moving deck to ``moving_land_node`` (single setpoint
authority). ``AUTO.LAND`` is used ONLY in the static demo — it descends vertically and
would miss a moving target.

Subscribes
    /mavros/state                    mavros_msgs/State       FCU arm/mode
    /mavros/local_position/odom      nav_msgs/Odometry       vehicle state (local ENU)
    /path_plan/cmd_vel               geometry_msgs/TwistStamped  MPC world vel + yawrate
Publishes
    /path_plan/odometry              nav_msgs/Odometry       vehicle state (map ENU)
    /mavros/setpoint_velocity/cmd_vel  geometry_msgs/TwistStamped  OFFBOARD setpoint
Service clients
    /mavros/cmd/arming               mavros_msgs/CommandBool
    /mavros/set_mode                 mavros_msgs/SetMode

Frame handling: PX4's EKF local-position origin sits at the vehicle spawn, which
is the map point ``map_offset_enu_m`` (default = the city map spawn 587,580,0).
Position is shifted by that offset so the pipeline (which reasons in the map's
building ENU) sees the true map coordinates.  Velocity and orientation are passed
through unchanged because the two ENU frames differ only by a translation (the
spawn yaw is 0).

OFFBOARD handshake: PX4 will only enter OFFBOARD once a setpoint stream is already
flowing (>2 Hz).  This node therefore streams a velocity setpoint every tick
(hover/zero until the MPC produces one), then requests OFFBOARD + arm.  Climb to
the cruise band is handled by the MPC's decoupled altitude hold (its vz drives the
vehicle up to the trajectory's ~25 m reference right after takeoff).
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped, TransformStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from std_msgs.msg import Bool, Float32MultiArray
from tf2_ros import TransformBroadcaster

from .ros_msgs import FRAME, msg_to_trajectory


class MavrosStaticPathNode(Node):
    def __init__(self):
        super().__init__("mavros_static_path")
        p = self.declare_parameter
        # Map point that the PX4 local-position origin (vehicle spawn) sits at.
        self.offset = np.asarray(
            p("map_offset_enu_m", [587.0, 580.0, 0.0]).value, float)
        self.rate_hz = float(p("rate_hz", 20.0).value)
        self.auto_arm = bool(p("auto_arm", True).value)
        # Pure-vertical takeoff: climb straight up to takeoff_alt_m before handing
        # over to the MPC, so the vehicle never moves laterally at low altitude and
        # always clears PX4's preflight auto-disarm window.
        self.takeoff_alt_m = float(p("takeoff_alt_m", 35.0).value)
        self.takeoff_vz = float(p("takeoff_vz_m_s", 1.5).value)
        # Auto-land once within this xy distance of the goal (the last trajectory
        # point): the bridge switches PX4 to AUTO.LAND and stops OFFBOARD.
        self.land_radius_m = float(p("land_radius_m", 12.0).value)
        # Fly at the vehicle's real max: read PX4's MPC_XY_VEL_MAX from MAVROS and
        # push it into the MPC's cruise speed (no hardcoded number).  speed_scale
        # trims it (e.g. 0.9 for margin); speed_from_fcu=false keeps config speed.
        self.speed_from_fcu = bool(p("speed_from_fcu", True).value)
        self.speed_param = str(p("speed_param", "MPC_XY_VEL_MAX").value)
        self.speed_scale = float(p("speed_scale", 1.0).value)
        mpc_node = str(p("mpc_node", "tracking_mpc").value)
        # Moving-trailer pursuit. pursuit_mode=false keeps the static-goal demo.
        self.pursuit_mode = bool(p("pursuit_mode", False).value)
        self.replan_period_s = float(p("replan_period_s", 6.0).value)
        self.terminal_range_m = float(p("terminal_range_m", 15.0).value)
        # Altitude of the replanned goal (kept inside the cruise band; A* clamps it).
        self.cruise_alt_m = float(p("cruise_alt_m", 35.0).value)
        self.trailer_topic = str(p("trailer_topic", "/trailer/position").value)
        self.goal_topic = str(p("goal_topic", "/astar_planner/goal").value)

        self._state = State()
        self._cmd = None          # latest MPC TwistStamped
        self._have_odom = False
        self._alt = 0.0           # latest altitude (map-frame z, ~AGL)
        self._pos = np.zeros(3)   # latest map-frame position
        # Flight phase state machine (advances only while armed + OFFBOARD):
        #   CLIMB  -> pure vertical climb to takeoff_alt_m (no lateral motion)
        #   HOLD   -> hover in place until the A*->B-spline trajectory is ready
        #   CRUISE -> forward the MPC velocity toward the goal
        #   LAND   -> reached the goal; hand off to PX4 AUTO.LAND
        self.phase = "CLIMB"
        self._have_traj = False
        self._goal_xy = None
        self._setpoints_sent = 0
        self._last_request = self.get_clock().now()
        self._last_land_req = self.get_clock().now()
        self._last_goal_log = self.get_clock().now()
        self._speed_done = not self.speed_from_fcu
        self._speed_req = False
        # Pursuit state: latest trailer cue + replan cadence.
        self._trailer_xy = None
        self._trailer_stamp = None
        self._last_replan_time = None

        # BEST_EFFORT subscribers are compatible with both reliable and
        # best-effort publishers, so they never silently drop MAVROS topics.
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_pub = self.create_publisher(Odometry, "/path_plan/odometry", be)
        self.sp_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)
        self.start_pub = self.create_publisher(PoseStamped, "/astar_planner/start", 10)
        # Pursuit: republish the trailer's live position as the A* goal, and latch the
        # landing-authority gate for moving_land_node (transient_local so it gets the
        # value even if it subscribes after we publish).
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.land_enable_pub = self.create_publisher(
            Bool, "/pursuit/land_enable", latched)
        if self.pursuit_mode:
            self.land_enable_pub.publish(Bool(data=False))
            self.create_subscription(
                PointStamped, self.trailer_topic, self._on_trailer, 10)

        self.create_subscription(State, "/mavros/state", self._on_state, be)
        self.create_subscription(
            Odometry, "/mavros/local_position/odom", self._on_odom, be)
        self.create_subscription(
            TwistStamped, "/path_plan/cmd_vel", self._on_cmd, 10)
        # Watch the trajectory: leave HOLD only once the path is computed, and
        # take its last point as the goal for the auto-land trigger.
        self.create_subscription(
            Float32MultiArray, "/path_plan/trajectory", self._on_traj, 1)
        # Broadcast map->base_link so RViz has a valid "map" fixed frame and can
        # show the vehicle (MAVROS does not publish TF by default).
        self.tf_bc = TransformBroadcaster(self)

        self.arm_cli = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")
        # Read PX4 params (via MAVROS ROS-param API) and write the MPC's speed.
        self.get_param_cli = self.create_client(
            GetParameters, "/mavros/param/get_parameters")
        self.set_param_cli = self.create_client(
            SetParameters, f"/{mpc_node}/set_parameters")

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"MAVROS static-path bridge ready (map offset {self.offset.tolist()}, "
            f"auto_arm={self.auto_arm})")

    # ------------------------------------------------------------- callbacks
    def _on_state(self, msg: State):
        self._state = msg

    def _on_odom(self, msg: Odometry):
        """Shift local-ENU position into map ENU and republish for the pipeline."""
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = FRAME
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.pose.pose.position.x = msg.pose.pose.position.x + self.offset[0]
        out.pose.pose.position.y = msg.pose.pose.position.y + self.offset[1]
        out.pose.pose.position.z = msg.pose.pose.position.z + self.offset[2]
        out.twist = msg.twist           # velocity is translation-invariant
        self.odom_pub.publish(out)
        self._alt = out.pose.pose.position.z
        self._pos[0] = out.pose.pose.position.x
        self._pos[1] = out.pose.pose.position.y
        self._pos[2] = out.pose.pose.position.z
        self._have_odom = True
        # map->base_link TF for RViz (fixed frame "map").
        tf = TransformStamped()
        tf.header.stamp = out.header.stamp
        tf.header.frame_id = FRAME
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = out.pose.pose.position.x
        tf.transform.translation.y = out.pose.pose.position.y
        tf.transform.translation.z = out.pose.pose.position.z
        tf.transform.rotation = out.pose.pose.orientation
        self.tf_bc.sendTransform(tf)

    def _on_cmd(self, msg: TwistStamped):
        self._cmd = msg

    def _on_traj(self, msg: Float32MultiArray):
        _, pos, _ = msg_to_trajectory(msg)
        if len(pos) >= 2:
            self._have_traj = True
            self._goal_xy = pos[-1, :2]

    def _on_trailer(self, msg: PointStamped):
        self._trailer_xy = (msg.point.x, msg.point.y)
        self._trailer_stamp = self.get_clock().now()

    # ------------------------------------------------------------- main loop
    def _tick(self):
        # 1) Always stream a setpoint so PX4 keeps OFFBOARD alive.
        sp = TwistStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = FRAME
        active = self._state.armed and self._state.mode == "OFFBOARD"
        if active:
            if self.phase == "CLIMB":
                if self._alt >= self.takeoff_alt_m:
                    self._enter_hold(sp.header.stamp)
                else:
                    sp.twist.linear.z = self.takeoff_vz   # pure vertical climb
            if self.phase == "HOLD":
                # Hover in place (hold altitude, zero lateral) until the
                # A*->B-spline trajectory is ready, THEN start cruising.  This
                # removes the slanted climb where the drone drifted sideways
                # while still ascending.
                sp.twist.linear.z = self._hold_vz()
                if self._have_traj:
                    self.phase = "CRUISE"
                    self.get_logger().info("path ready -> cruising to goal")
            if self.phase == "CRUISE":
                if self.pursuit_mode:
                    # Chase the moving trailer: periodically replan A* toward its
                    # live position, and hand off to the moving-target lander once
                    # we close in.
                    self._pursuit_replan(sp.header.stamp)
                    if self._reached_trailer():
                        self._enter_handoff()
                    elif self._cmd is not None:
                        sp.twist = self._cmd.twist
                else:
                    self._log_goal_distance()
                    if self._reached_goal():
                        self._enter_land()
                    elif self._cmd is not None:
                        sp.twist = self._cmd.twist   # MPC world velocity + yawrate
                # else: hold zero (waiting for the first MPC command)
            # LAND: leave the setpoint at zero; PX4 flies AUTO.LAND now.
        # HANDOFF: go silent so moving_land_node is the sole setpoint authority.
        if self.phase == "HANDOFF":
            return
        self.sp_pub.publish(sp)
        self._setpoints_sent += 1

        # One-shot: sync the MPC cruise speed to PX4's MPC_XY_VEL_MAX.
        if not self._speed_done and self._state.connected:
            self._sync_speed_from_fcu()

        # Once the goal is reached, keep requesting AUTO.LAND (and never re-request
        # OFFBOARD, which would fight the landing) until PX4 confirms the mode.
        if self.phase == "LAND":
            self._ensure_landing()
            return
        if not self.auto_arm:
            return
        # 2) OFFBOARD + arm handshake.  Gate on odometry so we only arm once the
        #    pipeline has vehicle state and the takeoff climb can drive the
        #    vehicle up immediately (clearing PX4's preflight auto-disarm).
        #    Requests are throttled to ~1 Hz.
        if (not self._state.connected or not self._have_odom
                or self._setpoints_sent < self.rate_hz):
            return
        now = self.get_clock().now()
        if (now - self._last_request).nanoseconds < 1_000_000_000:
            return
        if self._state.mode != "OFFBOARD":
            self._last_request = now
            if self.mode_cli.service_is_ready():
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self.mode_cli.call_async(req)
                self.get_logger().info("requesting OFFBOARD mode")
        elif not self._state.armed:
            self._last_request = now
            if self.arm_cli.service_is_ready():
                req = CommandBool.Request()
                req.value = True
                self.arm_cli.call_async(req)
                self.get_logger().info("requesting arm")

    # ----------------------------------------------------------- phase helpers
    def _enter_hold(self, stamp):
        """Reached takeoff altitude: trigger the A* plan and hover until ready."""
        self.phase = "HOLD"
        self.get_logger().info(
            f"takeoff complete at {self._alt:.1f} m -> triggering path calculation "
            f"(holding position until the trajectory is ready)")
        start = PoseStamped()
        start.header.stamp = stamp
        start.header.frame_id = FRAME
        start.pose.position.x = self._pos[0]
        start.pose.position.y = self._pos[1]
        start.pose.position.z = self._pos[2]
        self.start_pub.publish(start)

    def _hold_vz(self) -> float:
        """P-controlled vertical velocity to hold the takeoff altitude in HOLD."""
        err = self.takeoff_alt_m - self._alt
        return float(np.clip(0.5 * err, -self.takeoff_vz, self.takeoff_vz))

    def _goal_dist(self) -> float:
        return float(np.hypot(self._pos[0] - self._goal_xy[0],
                              self._pos[1] - self._goal_xy[1]))

    def _log_goal_distance(self):
        """Throttled progress log so it's visible whether the goal is reached."""
        if self._goal_xy is None:
            return
        now = self.get_clock().now()
        if (now - self._last_goal_log).nanoseconds < 2_000_000_000:
            return
        self._last_goal_log = now
        self.get_logger().info(
            f"cruising: {self._goal_dist():.1f} m to goal "
            f"(auto-land within {self.land_radius_m:.0f} m)")

    def _reached_goal(self) -> bool:
        return self._goal_xy is not None and self._goal_dist() <= self.land_radius_m

    def _enter_land(self):
        self.phase = "LAND"
        self.get_logger().info(
            f"reached goal ({self._goal_dist():.1f} m, within {self.land_radius_m:.0f} m)"
            f" -> AUTO.LAND")

    # ------------------------------------------------------- pursuit helpers
    def _trailer_fresh(self) -> bool:
        if self._trailer_xy is None or self._trailer_stamp is None:
            return False
        age = (self.get_clock().now() - self._trailer_stamp).nanoseconds * 1e-9
        return age < 2.0

    def _trailer_dist(self) -> float:
        return float(np.hypot(self._pos[0] - self._trailer_xy[0],
                              self._pos[1] - self._trailer_xy[1]))

    def _pursuit_replan(self, stamp):
        """Every replan_period_s, republish the trailer's live position as the A*
        goal so the pipeline continuously replans toward the moving target."""
        if not self._trailer_fresh():
            return
        now = self.get_clock().now()
        due = (self._last_replan_time is None
               or (now - self._last_replan_time).nanoseconds * 1e-9
               >= self.replan_period_s)
        if not due:
            return
        goal = PoseStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = FRAME
        goal.pose.position.x = float(self._trailer_xy[0])
        goal.pose.position.y = float(self._trailer_xy[1])
        goal.pose.position.z = float(self.cruise_alt_m)
        self.goal_pub.publish(goal)
        self._last_replan_time = now
        self.get_logger().info(
            f"pursuit replan -> trailer ({self._trailer_xy[0]:.0f},"
            f"{self._trailer_xy[1]:.0f}); {self._trailer_dist():.0f} m to go "
            f"(handoff within {self.terminal_range_m:.0f} m)")

    def _reached_trailer(self) -> bool:
        return self._trailer_fresh() and self._trailer_dist() <= self.terminal_range_m

    def _enter_handoff(self):
        self.phase = "HANDOFF"
        self.land_enable_pub.publish(Bool(data=True))
        self.get_logger().info(
            f"trailer within {self.terminal_range_m:.0f} m "
            f"({self._trailer_dist():.1f} m) -> handoff to moving_land_node "
            f"(/pursuit/land_enable=true); OFFBOARD setpoints released")

    def _ensure_landing(self):
        """Request AUTO.LAND at ~1 Hz until PX4 confirms it (or has disarmed)."""
        if self._state.mode == "AUTO.LAND" or not self._state.armed:
            return
        now = self.get_clock().now()
        if (now - self._last_land_req).nanoseconds < 1_000_000_000:
            return
        self._last_land_req = now
        if not self.mode_cli.service_is_ready():
            self.get_logger().warn("set_mode service not ready; retrying AUTO.LAND")
            return
        req = SetMode.Request()
        req.custom_mode = "AUTO.LAND"
        self.mode_cli.call_async(req).add_done_callback(self._on_land_result)
        self.get_logger().info("requesting AUTO.LAND")

    def _on_land_result(self, fut):
        try:
            if not fut.result().mode_sent:
                self.get_logger().warn(
                    "PX4 rejected AUTO.LAND (mode_sent=False); will retry")
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().warn(f"AUTO.LAND call failed ({exc}); will retry")

    # -------------------------------------------------- FCU max-speed -> MPC
    def _sync_speed_from_fcu(self):
        if self._speed_req:
            return
        if not (self.get_param_cli.service_is_ready()
                and self.set_param_cli.service_is_ready()):
            return
        self._speed_req = True
        fut = self.get_param_cli.call_async(
            GetParameters.Request(names=[self.speed_param]))
        fut.add_done_callback(self._on_speed_read)

    def _on_speed_read(self, fut):
        try:
            values = fut.result().values
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().warn(f"speed read failed ({exc}); keeping config")
            self._speed_done = True
            return
        if not values:
            self.get_logger().warn(f"{self.speed_param} not found; keeping config")
            self._speed_done = True
            return
        pv = values[0]
        if pv.type == ParameterType.PARAMETER_NOT_SET:
            # MAVROS hasn't synced this parameter from PX4 yet; try again next tick
            self._speed_req = False
            return
        
        val = (pv.double_value if pv.type == ParameterType.PARAMETER_DOUBLE
               else float(pv.integer_value))
        if val <= 0.0:
            self.get_logger().warn(f"{self.speed_param}={val}; keeping config")
            self._speed_done = True
            return
        
        self._speed_done = True
        speed = val * self.speed_scale
        self.get_logger().info(
            f"{self.speed_param}={val:.1f} m/s -> MPC cruise {speed:.1f} m/s "
            f"(scale {self.speed_scale})")
        dbl = ParameterType.PARAMETER_DOUBLE
        self.set_param_cli.call_async(SetParameters.Request(parameters=[
            Parameter(name="v_ref_m_s",
                      value=ParameterValue(type=dbl, double_value=speed)),
            Parameter(name="v_max_m_s",
                      value=ParameterValue(type=dbl, double_value=speed)),
        ]))


def main(args=None):
    rclpy.init(args=args)
    node = MavrosStaticPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
