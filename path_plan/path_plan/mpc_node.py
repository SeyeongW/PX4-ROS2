"""ROS 2 node wrapping :class:`UnicycleMPC` (mpc_ros-style kinematic tracking).

Subscribes
    ~/trajectory  std_msgs/Float32MultiArray  (N,7) B-spline reference [t,p,v]
    ~/odometry    nav_msgs/Odometry           vehicle state (ENU pose + world vel)
    ~/depth       sensor_msgs/Range           forward nearest obstacle distance
Publishes
    ~/cmd_vel     geometry_msgs/TwistStamped  world velocity + yaw-rate setpoint
    ~/mpc_preview nav_msgs/Path               predicted horizon (RViz)

The horizontal motion is tracked with a unicycle model (inputs yaw-rate + accel),
so the vehicle noses along the path and the forward depth camera looks ahead.
Altitude is a decoupled hold.  The double-integrator :mod:`path_plan.mpc`
remains available as an alternative controller.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Range
from std_msgs.msg import Float32MultiArray

from .mpc import depth_avoidance_offset
from .mpc_ros import UnicycleMPC, Weights
from .ros_msgs import FRAME, msg_to_trajectory, positions_to_path


def _yaw_from_quaternion(q) -> float:
    """Extract ENU yaw (rotation about +z) from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class MPCNode(Node):
    def __init__(self):
        super().__init__("tracking_mpc")
        p = self.declare_parameter
        self.dt = float(p("dt_s", 0.1).value)
        self.base_v_ref = float(p("v_ref_m_s", 4.0).value)
        self.mpc = UnicycleMPC(
            dt_s=self.dt, horizon=int(p("horizon", 20).value),
            v_ref=self.base_v_ref, v_max=p("v_max_m_s", 5.0).value,
            omega_max=p("omega_max_rad_s", 1.2).value, a_max=p("a_max_m_s2", 3.0).value,
            z_kp=p("z_kp", 0.8).value, vz_max=p("vz_max_m_s", 2.0).value,
            weights=Weights(
                cte=p("w_cte", 2.0).value, epsi=p("w_epsi", 4.0).value,
                v=p("w_v", 1.0).value, omega=p("w_omega", 0.5).value,
                a=p("w_a", 0.05).value, domega=p("w_domega", 2.0).value,
                da=p("w_da", 0.1).value))
        self.trigger_m = p("depth_trigger_m", 10.0).value
        self.emergency_m = p("depth_emergency_m", 4.0).value
        self.lateral_m = p("avoid_lateral_m", 7.0).value
        # Holonomic mode: a PX4 multicopter can strafe, so command the world
        # velocity straight toward the look-ahead reference instead of the
        # unicycle's nose-coupled velocity (which spins in place when the yaw
        # can't keep up).  Yaw is decoupled: a gentle, non-blocking turn toward
        # travel so the forward depth camera roughly faces the direction of
        # motion.  Set holonomic=false to use the unicycle controller.
        self.holonomic = bool(p("holonomic", True).value)
        self.yaw_kp = float(p("yaw_kp", 1.0).value)
        self.yaw_rate_max = float(p("yaw_rate_max_rad_s", 0.6).value)
        self.goal_slow_m = float(p("goal_slow_radius_m", 6.0).value)
        # Cruise speed can be updated live (e.g. the MAVROS bridge sets it from
        # PX4's MPC_XY_VEL_MAX so we fly at the vehicle's real max, no hardcoding).
        self.add_on_set_parameters_callback(self._on_set_params)

        self._traj_t = self._traj_p = self._traj_v = None
        self._pos = np.zeros(3)
        self._yaw = 0.0
        self._speed = 0.0
        self._depth = np.inf
        self._have_state = False

        self.cmd_pub = self.create_publisher(TwistStamped, "~/cmd_vel", 10)
        self.preview_pub = self.create_publisher(Path, "~/mpc_preview", 10)
        best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Float32MultiArray, "~/trajectory", self._on_traj, 1)
        self.create_subscription(Odometry, "~/odometry", self._on_odom, best_effort)
        self.create_subscription(Range, "~/depth", self._on_depth, best_effort)
        self.create_timer(self.dt, self._control_tick)
        self.get_logger().info("mpc_ros tracking MPC ready")

    # ------------------------------------------------------------- callbacks
    def _on_set_params(self, params):
        for prm in params:
            if prm.name == "v_ref_m_s":
                self.base_v_ref = float(prm.value)
            elif prm.name == "v_max_m_s":
                self.mpc.v_max = float(prm.value)
        return SetParametersResult(successful=True)

    def _on_traj(self, msg: Float32MultiArray):
        self._traj_t, self._traj_p, self._traj_v = msg_to_trajectory(msg)

    def _on_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        lin = msg.twist.twist.linear
        self._pos = np.array([pos.x, pos.y, pos.z])
        self._yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._speed = math.hypot(lin.x, lin.y)
        self._have_state = True

    def _on_depth(self, msg: Range):
        self._depth = float(msg.range)

    # --------------------------------------------------------- control loop
    def _reference_horizon(self):
        """Resample the trajectory at dt spacing from the nearest progress point."""
        N = self.mpc.N
        nearest = int(np.argmin(np.linalg.norm(self._traj_p - self._pos, axis=1)))
        t0 = self._traj_t[nearest]
        query = t0 + self.dt * np.arange(1, N + 1)
        ref_p = np.column_stack([np.interp(query, self._traj_t, self._traj_p[:, k])
                                 for k in range(3)])
        tangent = np.zeros(2)
        if nearest + 1 < len(self._traj_p):
            d = self._traj_p[nearest + 1, :2] - self._traj_p[nearest, :2]
            n = np.linalg.norm(d)
            tangent = d / n if n > 1e-9 else tangent
        return ref_p, tangent

    def _control_tick(self):
        if not self._have_state or self._traj_p is None or len(self._traj_p) < 2:
            return
        ref_p, tangent = self._reference_horizon()
        offset, scale = depth_avoidance_offset(
            self._depth, tangent, self.trigger_m, self.emergency_m, self.lateral_m)
        if np.any(offset):
            ref_p = ref_p + np.array([offset[0], offset[1], 0.0])
        self.mpc.v_ref = self.base_v_ref * scale     # brake near obstacles

        if self.holonomic:
            vx, vy, vz, yaw_rate, preview3d = self._holonomic_cmd(ref_p, scale)
        else:
            result = self.mpc.solve(self._pos, self._yaw, self._speed,
                                    self._pos[2], ref_p)
            vx, vy, vz = map(float, result.velocity_world)
            yaw_rate = float(result.yaw_rate)
            preview3d = np.column_stack(
                [result.predicted_xy,
                 np.full(len(result.predicted_xy), self._pos[2])])

        cmd = TwistStamped()
        cmd.header.frame_id = FRAME
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z = vx, vy, vz
        cmd.twist.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)
        self.preview_pub.publish(positions_to_path(preview3d, cmd.header.stamp))

    def _holonomic_cmd(self, ref_p, scale):
        """World-frame pure-pursuit velocity toward the look-ahead reference.

        Decouples translation from heading (a multicopter strafes), so the
        vehicle drives straight along the path regardless of yaw.  Yaw turns
        gently toward the travel direction and never gates the velocity.
        """
        pos = self._pos
        target = ref_p[-1]                       # furthest look-ahead point
        d = target[:2] - pos[:2]
        dist = float(np.hypot(d[0], d[1]))
        # Cruise at v_ref (constant), but decelerate smoothly into the final goal.
        goal_dist = float(np.linalg.norm(self._traj_p[-1, :2] - pos[:2]))
        speed = min(self.base_v_ref * scale, self.mpc.v_max)
        speed = min(speed, self.base_v_ref * max(0.0, goal_dist / self.goal_slow_m))
        vx = vy = 0.0
        if dist > 1e-3:
            vx, vy = (d / dist) * speed
        z_ref = float(ref_p[min(len(ref_p) - 1, self.mpc.N // 2), 2])
        vz = float(np.clip(self.mpc.z_kp * (z_ref - pos[2]),
                           -self.mpc.vz_max, self.mpc.vz_max))
        yaw_rate = 0.0
        if speed > 0.3 and dist > 1e-3:
            err = math.atan2(d[1], d[0]) - self._yaw
            err = math.atan2(math.sin(err), math.cos(err))   # wrap to [-pi, pi]
            yaw_rate = float(np.clip(self.yaw_kp * err,
                                     -self.yaw_rate_max, self.yaw_rate_max))
        preview3d = ref_p.copy()
        return vx, vy, vz, yaw_rate, preview3d


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
