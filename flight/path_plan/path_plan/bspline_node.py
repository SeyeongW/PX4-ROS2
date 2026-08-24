"""ROS 2 node wrapping :class:`BsplineOptimizer` (ego-planner corridor B-spline).

Subscribes
    ~/global_path  nav_msgs/Path                A* waypoints (ENU)
Publishes
    ~/trajectory        std_msgs/Float32MultiArray   (N,7) [t,px,py,pz,vx,vy,vz]
    ~/trajectory_path   nav_msgs/Path                sampled positions (RViz)
    ~/corridor_markers  visualization_msgs/MarkerArray   per-control-point boxes

The optimizer owns corridor generation and the collision rebound loop, so this
node needs only the A* path plus the map (to build free boxes and collision-check).
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from .bspline_optimizer import BsplineOptimizer
from .ros_msgs import (FRAME, corridor_to_msg, path_to_positions,
                       positions_to_path, trajectory_to_msg)
from .world_model import WorldModel

_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


class BSplineNode(Node):
    def __init__(self):
        super().__init__("bspline_optimizer")
        p = self.declare_parameter
        # ground_clearance_m defaults to 0 (no hard floor): cruise altitude is held by
        # the SFC corridor built around the A* guide points (all at cruise height), so
        # the floor is redundant for cruise but WOULD block a terminal descent whose
        # guide points drop to the deck. Keeping it 0 lets the moving-trailer landing
        # feed a drone->deck global_path straight through this optimizer (no separate
        # optimizer needed). Raise it only if a cruise-only run needs a hard sag guard.
        world = WorldModel.from_city_yaml(
            p("map_yaml", "").value,
            inflation_xy_m=p("inflation_xy_m", 1.45).value,
            roof_clearance_m=p("roof_clearance_m", 10.0).value,
            ground_clearance_m=p("ground_clearance_m", 0.0).value,
            ceiling_m=p("cruise_ceiling_m", 30.0).value,
            overfly_allowed=p("overfly_allowed", True).value)
        self.optimizer = BsplineOptimizer(
            world,
            cruise_speed_m_s=p("cruise_speed_m_s", 4.0).value,
            ctrl_spacing_m=p("ctrl_spacing_m", 5.0).value,
            max_vel=p("max_vel_m_s", 5.0).value, max_acc=p("max_acc_m_s2", 3.0).value,
            lambda_smooth=p("lambda_smooth", 1.0).value,
            lambda_dist=p("lambda_dist", 0.5).value,
            lambda_feas=p("lambda_feas", 2.0).value,
            lambda_fit=p("lambda_fit", 0.2).value,
            demarcation_m=p("demarcation_m", 0.3).value,
            max_rebound=int(p("max_rebound", 10).value))
        self.samples = int(p("sample_count", 300).value)

        self.traj_pub = self.create_publisher(Float32MultiArray, "~/trajectory", _LATCHED)
        self.path_pub = self.create_publisher(Path, "~/trajectory_path", _LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, "~/corridor_markers", _LATCHED)
        self.sfc_pub = self.create_publisher(
            Float32MultiArray, "/path_plan/sfc_corridor", _LATCHED)
        self.sfc_stats_pub = self.create_publisher(
            Float32MultiArray, "/path_plan/sfc_stats", _LATCHED)
        self.create_subscription(Path, "~/global_path", self._on_path, _LATCHED)
        self.get_logger().info("ego-planner B-spline optimizer ready")

    def _on_path(self, msg: Path):
        wp = path_to_positions(msg)
        if len(wp) < 2:
            return
        result = self.optimizer.optimize(wp)
        t, pos, vel, _acc = result.spline.sample(self.samples)
        stamp = self.get_clock().now().to_msg()
        self.traj_pub.publish(trajectory_to_msg(t, pos, vel))
        self.path_pub.publish(positions_to_path(pos, stamp))
        self.marker_pub.publish(self._to_markers(result.corridor.boxes_min,
                                                 result.corridor.boxes_max))
        try:
            widths = np.min(
                result.corridor.boxes_max[:, :2]
                - result.corridor.boxes_min[:, :2], axis=1)
            self.sfc_pub.publish(corridor_to_msg(
                result.corridor.boxes_min, result.corridor.boxes_max))
            self.sfc_stats_pub.publish(Float32MultiArray(data=[
                float(1000.0 * result.sfc_generation_time_s),
                float(np.min(widths)), float(np.mean(widths)),
                float(len(widths)),
            ]))
        except Exception as exc:
            try:
                self.get_logger().warn(
                    f'SFC observation publish failed: {exc}')
            except Exception:
                pass
        self.get_logger().info(
            f"trajectory: {len(pos)} samples, T={result.spline.duration():.1f}s, "
            f"rebound={result.rebound_iters}, collision_free={result.collision_free}, "
            f"free={result.free_fraction:.3f}")

    def _to_markers(self, lo, hi) -> MarkerArray:
        arr = MarkerArray()
        for i, (a, b) in enumerate(zip(lo, hi)):
            m = Marker()
            m.header.frame_id = FRAME
            m.ns = "sfc"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            centre = 0.5 * (a + b)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, centre)
            m.pose.orientation.w = 1.0
            size = np.maximum(b - a, 0.05)
            m.scale = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
            m.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.12)
            arr.markers.append(m)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = BSplineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
