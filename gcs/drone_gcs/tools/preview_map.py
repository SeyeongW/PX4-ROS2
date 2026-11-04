#!/usr/bin/env python3
"""Render the map canvas to a PNG without an interactive session.

    python3 tools/preview_map.py --demo --out preview.png
    python3 tools/preview_map.py --plan --zoom 3 --center 300 250

The GUI cannot be driven from an automated environment, so this is how the paint
code gets checked: it builds a state, grabs one frame offscreen, and exits.  It
doubles as a quick way to look at a freshly baked map pack.

`--plan` runs the real A* -> B-spline pipeline from `path_plan` over the pack's
own map, so the preview shows a route the planner would actually fly rather than
a drawn-in guess.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

# Must be set before any Qt import when there is no display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from drone_gcs.map_pack import MapPack  # noqa: E402
from drone_gcs.map_view import MapView  # noqa: E402
from drone_gcs.qt import QtCore, QtWidgets  # noqa: E402
from drone_gcs.world_state import WorldState  # noqa: E402


def demo_route(pack: MapPack) -> list[tuple[float, float, float]]:
    """A hand-placed polyline from spawn to goal, for when the planner is not run.

    Densified to roughly the spacing a real trajectory has, so the preview shows
    the layers at their true relative lengths instead of one segment each.
    """
    z = pack.cruise_z_m
    start = pack.spawn_enu_m or (0.0, 0.0)
    goal = (pack.default_goal_enu_m or (0.0, 0.0, z))[:2]
    corners = [start, (300.0, 300.0), (60.0, 120.0), (-120.0, -60.0), (-360.0, -300.0), goal]

    route = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        steps = max(int(math.dist((x0, y0), (x1, y1)) / 5.0), 1)
        for i in range(steps):
            t = i / steps
            route.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z))
    route.append((goal[0], goal[1], z))
    return route


def planned_route(pack: MapPack, resolution_m: float):
    """Run the real planner over this pack's source map.

    Returns (astar_points, bspline_points, corridor_boxes).
    """
    sys.path.insert(0, str(REPO / "flight/path_plan"))
    from path_plan.astar import AStarPlanner3D
    from path_plan.bspline_optimizer import BsplineOptimizer
    from path_plan.world_model import WorldModel

    if pack.source_world_yaml is None:
        raise SystemExit("--plan needs a pack baked from a source world yaml")
    world_yaml = REPO / pack.source_world_yaml
    model = WorldModel.from_city_yaml(
        world_yaml,
        overfly_allowed=pack.overfly_allowed,
        ceiling_m=pack.cruise_band_m[1] if pack.cruise_band_m else 30.0,
        **pack.planner_model,
    )
    z = pack.cruise_z_m
    start = (*(pack.spawn_enu_m or (0.0, 0.0)), z)
    goal = pack.default_goal_enu_m or (0.0, 0.0, z)

    print(f"planning {start} -> {goal} at {resolution_m} m resolution ...")
    planner = AStarPlanner3D(model, resolution_m=resolution_m)
    path = planner.plan(start, tuple(goal))
    if path is None:
        raise SystemExit("A* found no route; try a coarser --plan-resolution")
    path = planner.shortcut(path)
    print(f"  A*: {len(path)} points")

    optimizer = BsplineOptimizer(model)
    spline = optimizer.optimize(path)
    _t, points, _v, _a = spline.sample(400)
    boxes = optimizer.boxes if hasattr(optimizer, "boxes") else []
    corridor = [((lo[0], lo[1]), (hi[0], hi[1])) for lo, hi in boxes] if boxes else []
    return [tuple(p) for p in path], [tuple(p) for p in points], corridor


def build_state(pack: MapPack, args) -> WorldState:
    state = WorldState()
    z = pack.cruise_z_m

    if args.plan:
        astar, bspline, corridor = planned_route(pack, args.plan_resolution)
        state.global_path.set(astar)
        state.trajectory.set(bspline)
        state.corridor = corridor
        route = bspline
    else:
        route = demo_route(pack)
        # Offset the A* layer slightly so the preview shows both it and the
        # smoothed trajectory; in flight they differ by the optimiser, not by an
        # offset.
        state.global_path.set([(x + 12.0, y + 12.0, zz) for x, y, zz in route[::6]])
        state.trajectory.set(route)
        state.corridor = [
            ((x - 18, y - 18), (x + 18, y + 18)) for x, y, _ in route[::12]
        ]

    # Fly the drone a third of the way along the route, trail behind it.
    cut = max(2, len(route) // 3)
    for x, y, *_ in route[:cut]:
        state.drone.update(x, y, z, yaw_rad=0.0, speed_m_s=4.0)
    if cut < len(route):
        ahead = route[min(cut + 4, len(route) - 1)]
        here = route[cut - 1]
        yaw = math.atan2(ahead[1] - here[1], ahead[0] - here[0])
        state.drone.update(here[0], here[1], z, yaw_rad=yaw, speed_m_s=4.0)
        # The MPC horizon is 20 steps of 0.1 s at ~4 m/s: a couple of seconds ahead.
        state.mpc_preview.set(route[cut - 1:cut + 16])

    state.goal_enu_m = pack.default_goal_enu_m
    state.waypoints_enu_m = route[::40][1:4]
    state.set_depth(14.0)
    state.status.connected = True
    state.status.armed = True
    state.status.mode = "OFFBOARD"
    state.mission_phase = "SEARCH"

    # Dynamic entities: park each at its spawn marker and give it a short trail.
    markers = {m.name: m.enu_m for m in pack.markers}
    for spec in pack.entities:
        origin = markers.get(f"{spec.name}_spawn") or markers.get("trailer_spawn")
        if origin is None:
            continue
        for step in range(12):
            track = state.entity(spec.name)
            track.update(origin[0] + 6.0 * step, origin[1], 0.0, yaw_rad=0.0, speed_m_s=6.0)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=str(PACKAGE_ROOT / "maps" / "city_uav"))
    parser.add_argument("--out", type=Path, default=Path("preview.png"))
    parser.add_argument("--size", default="1400x900", help="WIDTHxHEIGHT in pixels")
    parser.add_argument("--zoom", type=float, default=1.0, help="factor over fit-to-map")
    parser.add_argument("--center", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--demo", action="store_true", help="draw a hand-placed route")
    parser.add_argument("--plan", action="store_true", help="run the real planner")
    parser.add_argument("--plan-resolution", type=float, default=10.0)
    parser.add_argument("--empty", action="store_true", help="static map only, no telemetry")
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    pack = MapPack.load(args.map)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    view = MapView(pack)
    view.resize(width, height)
    if not args.empty:
        view.state = build_state(pack, args)
    if args.center is not None:
        view._set_projection(view.projection.centered_on(*args.center))
    if args.zoom != 1.0:
        view._set_projection(view.projection.zoomed(args.zoom))

    # One layout pass so the widget knows its size before it paints.
    view.show()
    app.processEvents()
    QtCore.QCoreApplication.processEvents()

    pixmap = view.grab()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(args.out)):
        raise SystemExit(f"failed to write {args.out}")

    print(
        f"wrote {args.out} ({pixmap.width()}x{pixmap.height()}) "
        f"map={pack.name} zoom={view.projection.px_per_m:.4f} px/m "
        f"center={tuple(round(v, 1) for v in view.projection.center_enu_m)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
