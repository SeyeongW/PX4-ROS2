"""ROS 2 <-> MAVROS OFFBOARD bridge for the path_plan pipeline (PX4 SITL).

Flies the static-target cruise: PX4 spawn -> stationary trailer, tracking the
A* -> SFC -> B-spline -> MPC pipeline over MAVROS.

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
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters

from .ros_msgs import FRAME


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
        self.takeoff_alt_m = float(p("takeoff_alt_m", 20.0).value)
        self.takeoff_vz = float(p("takeoff_vz_m_s", 1.5).value)
        # Fly at the vehicle's real max: read PX4's MPC_XY_VEL_MAX from MAVROS and
        # push it into the MPC's cruise speed (no hardcoded number).  speed_scale
        # trims it (e.g. 0.9 for margin); speed_from_fcu=false keeps config speed.
        self.speed_from_fcu = bool(p("speed_from_fcu", True).value)
        self.speed_param = str(p("speed_param", "MPC_XY_VEL_MAX").value)
        self.speed_scale = float(p("speed_scale", 1.0).value)
        mpc_node = str(p("mpc_node", "tracking_mpc").value)

        self._state = State()
        self._cmd = None          # latest MPC TwistStamped
        self._have_odom = False
        self._alt = 0.0           # latest altitude (map-frame z, ~AGL)
        self._took_off = False
        self._setpoints_sent = 0
        self._last_request = self.get_clock().now()
        self._speed_done = not self.speed_from_fcu
        self._speed_req = False

        # BEST_EFFORT subscribers are compatible with both reliable and
        # best-effort publishers, so they never silently drop MAVROS topics.
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_pub = self.create_publisher(Odometry, "/path_plan/odometry", be)
        self.sp_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10)

        self.create_subscription(State, "/mavros/state", self._on_state, be)
        self.create_subscription(
            Odometry, "/mavros/local_position/odom", self._on_odom, be)
        self.create_subscription(
            TwistStamped, "/path_plan/cmd_vel", self._on_cmd, 10)

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
        self._have_odom = True

    def _on_cmd(self, msg: TwistStamped):
        self._cmd = msg

    # ------------------------------------------------------------- main loop
    def _tick(self):
        # 1) Always stream a setpoint so PX4 keeps OFFBOARD alive.
        sp = TwistStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = FRAME
        active = self._state.armed and self._state.mode == "OFFBOARD"
        if active:
            if not self._took_off and self._alt >= self.takeoff_alt_m:
                self._took_off = True
                self.get_logger().info(
                    f"takeoff complete at {self._alt:.1f} m -> handing over to MPC")
            if not self._took_off:
                # Pure vertical climb until we reach the cruise floor.
                sp.twist.linear.z = self.takeoff_vz
            elif self._cmd is not None:
                sp.twist = self._cmd.twist      # MPC world velocity + yawrate
            # else: hold zero (waiting for the first MPC command)
        self.sp_pub.publish(sp)
        self._setpoints_sent += 1

        # One-shot: sync the MPC cruise speed to PX4's MPC_XY_VEL_MAX.
        if not self._speed_done and self._state.connected:
            self._sync_speed_from_fcu()

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
        self._speed_done = True
        try:
            values = fut.result().values
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().warn(f"speed read failed ({exc}); keeping config")
            return
        if not values:
            self.get_logger().warn(f"{self.speed_param} not found; keeping config")
            return
        pv = values[0]
        val = (pv.double_value if pv.type == ParameterType.PARAMETER_DOUBLE
               else float(pv.integer_value))
        if val <= 0.0:
            self.get_logger().warn(f"{self.speed_param}={val}; keeping config")
            return
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
