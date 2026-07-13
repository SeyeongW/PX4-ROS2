"""Generate and inspect the city mission without starting PX4 or Gazebo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from .planner_core import (
    CityMap,
    FrameTransform,
    OccupancyGrid,
    plan_smoothed_path,
)


def _default_map() -> Path:
    root = Path(__file__).resolve().parents[2]
    uav_map = root / "gazebo/maps/city_coordinates_uav.yaml"
    return uav_map if uav_map.is_file() else root / "gazebo/maps/city_coordinates.yaml"


def _corridor_polygon(start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
    tangent = end - start
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
    normal = np.asarray((-tangent[1], tangent[0]))
    return np.vstack(
        (start + radius * normal, end + radius * normal,
         end - radius * normal, start - radius * normal)
    )


def _plot(city: CityMap, plan: object, output: Path) -> None:
    figure, axes = plt.subplots(figsize=(12, 12), dpi=150)
    for building in city.buildings:
        axes.add_patch(
            Polygon(building.outer, closed=True, facecolor="#5c636a", edgecolor="none")
        )
    for segment in plan.corridor.segments:
        start = np.asarray(segment.start_xy_m, dtype=float)
        end = np.asarray(segment.end_xy_m, dtype=float)
        polygon = _corridor_polygon(start, end, segment.radius_m)
        axes.add_patch(
            Polygon(polygon, closed=True, facecolor="#7bc8f6", alpha=0.16, edgecolor="none")
        )
        axes.add_patch(Circle(start, segment.radius_m, color="#7bc8f6", alpha=0.16))
        axes.add_patch(Circle(end, segment.radius_m, color="#7bc8f6", alpha=0.16))
    raw = np.asarray(plan.raw_astar)
    simplified = np.asarray(plan.simplified)
    trajectory = np.asarray(plan.trajectory)
    axes.plot(raw[:, 0], raw[:, 1], color="#4dabf7", linewidth=0.8, label="A* grid path")
    axes.plot(
        simplified[:, 0], simplified[:, 1], "o--", color="#f59f00",
        linewidth=1.0, markersize=4, label="LOS / SFC centerline",
    )
    axes.plot(
        trajectory[:, 0], trajectory[:, 1], color="#d6336c", linewidth=2.2,
        label="validated cubic B-spline",
    )
    axes.scatter([-120.0], [115.0], marker="^", s=90, color="#2b8a3e", label="spawn")
    axes.scatter([200.0], [-125.0], marker="*", s=150, color="#c92a2a", label="goal")
    axes.set_xlim(city.bounds_x_m)
    axes.set_ylim(city.bounds_y_m)
    axes.set_aspect("equal")
    axes.set_xlabel("Gazebo ENU x / east [m]")
    axes.set_ylabel("Gazebo ENU y / north [m]")
    axes.set_title("City mission: A* + safe flight corridor + B-spline")
    axes.grid(alpha=0.15)
    axes.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=_default_map())
    parser.add_argument("--goal-x", type=float, default=200.0)
    parser.add_argument("--goal-y", type=float, default=-125.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--inflation", type=float, default=5.0)
    parser.add_argument("--clearance-weight", type=float, default=4.0)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path.home() / "city_astar_sfc_bspline.png",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    city = CityMap.from_yaml(args.map)
    grid = OccupancyGrid.from_city_map(
        city, resolution_m=args.resolution, inflation_m=args.inflation
    )
    start = city.px4_origin_enu_m[:2]
    goal = (args.goal_x, args.goal_y)
    plan = plan_smoothed_path(
        grid,
        start,
        goal,
        clearance_weight=args.clearance_weight,
        corridor_radius_m=10.0,
        output_spacing_m=args.spacing,
        smoothing=2.0,
    )
    path_length = float(np.linalg.norm(np.diff(plan.trajectory, axis=0), axis=1).sum())
    minimum_grid_clearance = min(grid.clearance_at(point) for point in plan.trajectory)
    goal_ned = FrameTransform.from_city_map(city).enu_to_ned(
        (args.goal_x, args.goal_y, city.px4_origin_enu_m[2] + 10.0)
    )
    if not args.no_plot:
        _plot(city, plan, args.plot.expanduser())
    print(
        json.dumps(
            {
                "result": "PASS",
                "start_enu_m": list(start),
                "goal_enu_m": list(goal),
                "goal_px4_local_ned_m": [float(value) for value in goal_ned],
                "astar_points": len(plan.raw_astar),
                "sfc_segments": len(plan.corridor.segments),
                "trajectory_points": len(plan.trajectory),
                "trajectory_length_m": path_length,
                "validated_cubic_bspline": plan.used_bspline,
                "smoothing": plan.smoothing_message,
                "minimum_clearance_from_inflated_cells_m": minimum_grid_clearance,
                "plot": None if args.no_plot else str(args.plot.expanduser()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
