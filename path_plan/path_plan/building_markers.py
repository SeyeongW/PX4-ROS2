"""Publish the city buildings as real 3D prisms (MarkerArray) for RViz.

Each building in the city YAML is a ``polygon_prism``: an exact ``footprint.outer``
polygon (4-27 vertices, not just a box) extruded from ``foundation_z_m`` to
``roof_z_m``.  We render the true polygon — NOT the inflated AABB the planner uses
for collision — so the RViz buildings sit at exactly the same coordinates as the
Gazebo world and the planned path (map ENU / "map" frame).

Markers (one MarkerArray, latched TRANSIENT_LOCAL so RViz gets it on connect):
    buildings/walls  TRIANGLE_LIST  extruded side walls, shaded by height
    buildings/roofs  TRIANGLE_LIST  ear-clipped roof caps
    buildings/edges  LINE_LIST      vertical corner edges (crisp outline)
"""

from __future__ import annotations

from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .ros_msgs import FRAME
from .world_model import _find_buildings

_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


def _signed_area2(poly) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def _point_in_tri(p, a, b, c) -> bool:
    """True if p is strictly inside triangle abc (barycentric sign test)."""
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _triangulate(poly):
    """Ear-clipping triangulation of a simple polygon (convex or concave).

    Returns index triples into ``poly``.  Handles the non-rectangular footprints
    in this map correctly (a centroid/vertex fan would splay outside concave ones).
    """
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n))
    if _signed_area2(poly) < 0.0:                 # force CCW
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        m = len(idx)
        clipped = False
        for i in range(m):
            i0, i1, i2 = idx[(i - 1) % m], idx[i], idx[(i + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0.0:                       # reflex / degenerate: not an ear
                continue
            if any(j not in (i0, i1, i2) and _point_in_tri(poly[j], a, b, c)
                   for j in idx):
                continue
            tris.append((i0, i1, i2))
            idx.pop(i)
            clipped = True
            break
        if not clipped:                            # numerical fallback
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def _pt(x, y, z) -> Point:
    return Point(x=float(x), y=float(y), z=float(z))


class BuildingMarkersNode(Node):
    def __init__(self):
        super().__init__("building_markers")
        self.map_yaml = self.declare_parameter("map_yaml", "").value
        self.pub = self.create_publisher(MarkerArray, "/path_plan/buildings", _LATCHED)
        self.msg = self._build()
        n = len(_find_buildings(yaml.safe_load(Path(self.map_yaml).read_text())) or [])
        self.get_logger().info(f"Publishing {n} buildings as 3D prisms on /path_plan/buildings")
        self.create_timer(1.0, self._publish)      # (re)latch for late RViz joiners

    def _publish(self):
        self.pub.publish(self.msg)

    def _build(self) -> MarkerArray:
        doc = yaml.safe_load(Path(self.map_yaml).read_text(encoding="utf-8"))
        buildings = _find_buildings(doc) or []
        stamp = self.get_clock().now().to_msg()

        walls = self._blank(Marker.TRIANGLE_LIST, "buildings/walls", stamp)
        roofs = self._blank(Marker.TRIANGLE_LIST, "buildings/roofs", stamp)
        edges = self._blank(Marker.LINE_LIST, "buildings/edges", stamp)
        edges.scale.x = 0.6                        # line width (m)
        edges.color = ColorRGBA(r=0.10, g=0.10, b=0.12, a=1.0)

        z_lo_all, z_hi_all = 20.0, 50.0            # height range for shading
        for b in buildings:
            outer = [(float(x), float(y)) for x, y in b["footprint"]["outer"]]
            if len(outer) < 3:
                continue
            z0 = float(b.get("foundation_z_m", 0.0))
            z1 = float(b["roof_z_m"])
            # Shade concrete-ish, a touch brighter for taller towers.
            t = max(0.0, min(1.0, (z1 - z_lo_all) / (z_hi_all - z_lo_all)))
            wall_c = ColorRGBA(r=0.60 + 0.22 * t, g=0.62 + 0.20 * t,
                               b=0.66 + 0.18 * t, a=1.0)
            roof_c = ColorRGBA(r=0.34 + 0.14 * t, g=0.36 + 0.14 * t,
                               b=0.42 + 0.16 * t, a=1.0)

            m = len(outer)
            for i in range(m):                     # side walls: quad -> 2 triangles
                x1, y1 = outer[i]
                x2, y2 = outer[(i + 1) % m]
                quad = [_pt(x1, y1, z0), _pt(x2, y2, z0), _pt(x2, y2, z1),
                        _pt(x1, y1, z1)]
                for a, bb, c in ((0, 1, 2), (0, 2, 3)):
                    walls.points += [quad[a], quad[bb], quad[c]]
                    walls.colors += [wall_c, wall_c, wall_c]
                edges.points += [_pt(x1, y1, z0), _pt(x1, y1, z1)]   # vertical edge
                edges.colors += [edges.color, edges.color]

            for i0, i1, i2 in _triangulate(outer):  # roof cap
                roofs.points += [_pt(*outer[i0], z1), _pt(*outer[i1], z1),
                                 _pt(*outer[i2], z1)]
                roofs.colors += [roof_c, roof_c, roof_c]

        return MarkerArray(markers=[walls, roofs, edges])

    def _blank(self, mtype, ns, stamp) -> Marker:
        mk = Marker()
        mk.header.frame_id = FRAME
        mk.header.stamp = stamp
        mk.ns = ns
        mk.id = 0
        mk.type = mtype
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 1.0
        mk.color.a = 1.0
        return mk


def main(args=None):
    rclpy.init(args=args)
    node = BuildingMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
