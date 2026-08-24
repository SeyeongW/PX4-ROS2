"""ROS 2 node wrapping :class:`AStarPlanner3D` (global grid search).

Subscribes
    ~/goal   geometry_msgs/PoseStamped   replan target (ENU)
    ~/start  geometry_msgs/PoseStamped   optional start override (ENU)
Publishes
    ~/global_path  nav_msgs/Path         latched A* waypoints (ENU / "map")
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from .astar import AStarPlanner3D
from .ros_msgs import positions_to_path
from .world_model import WorldModel

_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


class AStarNode(Node):
    def __init__(self):
        super().__init__("astar_planner")
        p = self.declare_parameter
        self.map_yaml = p("map_yaml", "").value
        res = p("resolution_m", 2.0).value
        world = WorldModel.from_city_yaml(
            self.map_yaml,
            inflation_xy_m=p("inflation_xy_m", 3.0).value,
            roof_clearance_m=p("roof_clearance_m", 10.0).value,
            ground_clearance_m=p("cruise_floor_m", 30.0).value,
            ceiling_m=p("cruise_ceiling_m", 40.0).value,
            overfly_allowed=p("overfly_allowed", True).value,
        )
        self.floor = float(self.get_parameter("cruise_floor_m").value)
        self.ceiling = float(self.get_parameter("cruise_ceiling_m").value)
        floor, ceiling = self.floor, self.ceiling
        self.planner = AStarPlanner3D(
            world, resolution_m=res,
            max_expanded=int(p("max_expanded", 400000).value),
            clearance_weight=p("clearance_weight", 0.4).value,
            clearance_pref_m=p("clearance_pref_m", 3.0).value,
            altitude_weight=p("altitude_weight", 0.05).value,
            altitude_pref_m=p("altitude_pref_m", 0.5 * (floor + ceiling)).value,
            climb_weight=p("climb_weight", 0.5).value,
            heuristic_weight=p("heuristic_weight", 1.0).value)
        self.start = np.asarray(p("start_enu_m", [587.0, 580.0, 25.0]).value, float)
        self.goal = np.asarray(p("goal_enu_m", [-300.0, -300.0, 25.0]).value, float)
        # Optional ordered via-points between start and goal, flat [x,y,z,...].
        # The path is forced through each, so use them to shape a hard route.
        # An empty-list param default makes rclpy return None (it can't infer the
        # element type), so coalesce to [] before iterating.
        wps = list(p("waypoints_enu_m", []).value or [])
        self.waypoints = [np.asarray(wps[i:i + 3], float) for i in range(0, len(wps) - 2, 3)]

        self.pub = self.create_publisher(Path, "~/global_path", _LATCHED)
        # Compute metrics for the flight report: [plan_time_s, expanded_nodes,
        # n_waypoints] published once per replan.
        self.stats_pub = self.create_publisher(
            Float32MultiArray, "/path_plan/astar_stats", _LATCHED)
        self.create_subscription(PoseStamped, "~/goal", self._on_goal, 10)
        self.create_subscription(PoseStamped, "~/start", self._on_start, 10)
        # Coalesced replan: start and goal arrive as two separate messages (the
        # pursuit bridge republishes BOTH the live drone position and the moving
        # trailer each cycle). Replanning immediately in each callback would plan
        # twice -- once with a stale endpoint -- so mark the request dirty and let
        # a lightweight timer fire a single plan ~_settle_s after the last update.
        self._dirty = False
        self._last_change = self.get_clock().now()
        self._settle_s = float(p("replan_settle_s", 0.15).value)
        self.create_timer(0.05, self._maybe_replan)
        self.get_logger().info(
            f"A* ready: {len(world.boxes_min)} obstacles, res={res} m. Waiting for start/goal to trigger calculation...")

    def _mark_dirty(self):
        self._dirty = True
        self._last_change = self.get_clock().now()

    def _maybe_replan(self):
        if not self._dirty:
            return
        if (self.get_clock().now() - self._last_change).nanoseconds < self._settle_s * 1e9:
            return
        self._dirty = False
        self._replan()

    def _on_start(self, msg: PoseStamped):
        self.start = np.array([msg.pose.position.x, msg.pose.position.y,
                               msg.pose.position.z])
        self.get_logger().info(f"Start received at {self.start}. Triggering path calculation...")
        self._mark_dirty()

    def _on_goal(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y,
                              msg.pose.position.z])
        self._mark_dirty()

    def _clamp_to_band(self, p):
        """Keep an endpoint inside the cruise band so its A* start/goal cell is
        never below the planning floor or above the ceiling (the takeoff altitude
        or odometry can land a metre outside the band)."""
        q = np.asarray(p, float).copy()
        q[2] = float(np.clip(q[2], self.floor, self.ceiling))
        return q

    def _replan(self):
        route = [self._clamp_to_band(p)
                 for p in (self.start, *self.waypoints, self.goal)]
        t0 = time.perf_counter()
        result = self.planner.plan_through(route)
        plan_s = time.perf_counter() - t0
        if not result.success:
            self.get_logger().warn(f"A* failed: {result.message}")
            return
        self.pub.publish(positions_to_path(result.waypoints_m, self.get_clock().now().to_msg()))
        stats = Float32MultiArray()
        stats.data = [float(plan_s), float(result.expanded),
                      float(len(result.waypoints_m))]
        self.stats_pub.publish(stats)
        self.get_logger().info(
            f"A* path published: {len(result.waypoints_m)} waypoints "
            f"(via {len(self.waypoints)}), expanded {result.expanded}, "
            f"plan {plan_s * 1e3:.0f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
