"""ROS 2 node wrapping :class:`SafeFlightCorridor`.

Subscribes
    ~/global_path  nav_msgs/Path            A* waypoints (ENU)
Publishes
    ~/corridor     std_msgs/Float32MultiArray   (M,6) [xmin,ymin,zmin,xmax,ymax,zmax]
    ~/corridor_markers  visualization_msgs/MarkerArray   RViz boxes
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

from .ros_msgs import FRAME, corridor_to_msg, path_to_positions
from .sfc import SafeFlightCorridor
from .world_model import WorldModel

_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


class SFCNode(Node):
    def __init__(self):
        super().__init__("sfc_builder")
        p = self.declare_parameter
        world = WorldModel.from_city_yaml(
            p("map_yaml", "").value,
            inflation_xy_m=p("inflation_xy_m", 1.45).value,
            roof_clearance_m=p("roof_clearance_m", 10.0).value,
            ground_clearance_m=p("cruise_floor_m", 20.0).value,
            ceiling_m=p("cruise_ceiling_m", 30.0).value,
        )
        self.sfc = SafeFlightCorridor(
            world, step_m=p("step_m", 0.5).value,
            max_extent_m=p("max_extent_m", 8.0).value,
            seed_spacing_m=p("seed_spacing_m", 3.0).value)
        self.pub = self.create_publisher(Float32MultiArray, "~/corridor", _LATCHED)
        self.markers = self.create_publisher(MarkerArray, "~/corridor_markers", _LATCHED)
        self.create_subscription(Path, "~/global_path", self._on_path, _LATCHED)
        self.get_logger().info("SFC builder ready")

    def _on_path(self, msg: Path):
        wp = path_to_positions(msg)
        if len(wp) < 2:
            return
        corridor = self.sfc.build(wp)
        self.pub.publish(corridor_to_msg(corridor.boxes_min, corridor.boxes_max))
        self.markers.publish(self._to_markers(corridor.boxes_min, corridor.boxes_max))
        self.get_logger().info(f"SFC published: {len(corridor)} boxes")

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
            m.scale = Vector3(x=float(b[0] - a[0]), y=float(b[1] - a[1]),
                              z=float(b[2] - a[2]))
            m.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.15)
            arr.markers.append(m)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = SFCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
