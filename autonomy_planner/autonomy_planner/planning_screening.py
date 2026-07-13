"""Offline A*/SFC and guidance-mode comparison for the derived city map.

Run from a source checkout with::

    PYTHONPATH=autonomy_planner python3 -m autonomy_planner.planning_screening

The output is machine-readable JSON.  It deliberately reports PX4-native
waypoint acceleration/jerk as null because only PX4 flight telemetry can
measure those values truthfully.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from .planner_3d import (
    HierarchicalAStarPlanner3D,
    PrismOccupancy3D,
    TrajectoryLimits3D,
    VehicleEnvelope3D,
    compare_planning_output_modes_3d,
    load_preferred_city_map,
    plan_validated_waypoints_3d,
)
from .planner_core import CityMap


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _candidate(metrics: object) -> dict[str, object]:
    return {
        "mode": metrics.mode.value,
        "success": metrics.success,
        "reason": metrics.reason,
        "collision_free": metrics.collision_free,
        "hard_clearance_violation_count": metrics.hard_clearance_violation_count,
        "minimum_residual_clearance_m": _finite(metrics.minimum_clearance_m),
        "path_length_m": _finite(metrics.path_length_m),
        "estimated_flight_time_s": _finite(metrics.estimated_flight_time_s),
        "maximum_velocity_m_s": _finite(metrics.maximum_velocity_m_s),
        "maximum_acceleration_m_s2": _finite(metrics.maximum_acceleration_m_s2),
        "maximum_jerk_m_s3": _finite(metrics.maximum_jerk_m_s3),
        "astar_elapsed_s": metrics.astar_elapsed_s,
        "sfc_elapsed_s": metrics.sfc_elapsed_s,
        "postprocess_elapsed_s": metrics.postprocess_elapsed_s,
        "astar_expanded_nodes": metrics.astar_expanded_nodes,
        "postprocess_iterations": metrics.postprocess_iterations,
    }


def screen(
    city: CityMap,
    start_enu_m: Sequence[float],
    goal_enu_m: Sequence[float],
    *,
    maximum_z_m: float,
    nominal_speed_m_s: float,
    include_bspline: bool,
    minimum_surface_clearance_m: float = 10.0,
) -> dict[str, object]:
    envelope = VehicleEnvelope3D.from_city_map_defaults(city)
    occupancy = PrismOccupancy3D.from_vehicle_envelope(
        city,
        envelope,
        minimum_z_m=0.0,
        maximum_z_m=maximum_z_m,
        minimum_surface_clearance_m=minimum_surface_clearance_m,
    )
    planner = HierarchicalAStarPlanner3D(occupancy)
    common = plan_validated_waypoints_3d(planner, start_enu_m, goal_enu_m)
    document: dict[str, object] = {
        "map_yaml": str(city.source_path),
        "building_count": len(city.buildings),
        "start_enu_m": list(map(float, start_enu_m)),
        "goal_enu_m": list(map(float, goal_enu_m)),
        "safety_budget": {
            "body_yaw_invariant_radius_m": envelope.body_yaw_invariant_radius_m,
            "hard_inflation_radius_m": envelope.hard_horizontal_radius_m,
            "preferred_clearance_m": envelope.preferred_horizontal_radius_m,
            "preferred_extra_m": envelope.preferred_extra_m,
            "vertical_half_extent_m": envelope.vertical_half_extent_m,
            "minimum_surface_clearance_m": minimum_surface_clearance_m,
        },
        "common_plan": {
            "success": common.success,
            "reason": common.metrics.reason,
            "elapsed_s": common.metrics.elapsed_s,
            "astar_elapsed_s": common.metrics.astar_elapsed_s,
            "sfc_elapsed_s": common.metrics.sfc_elapsed_s,
            "postprocess_elapsed_s": common.metrics.postprocess_elapsed_s,
            "expanded_nodes": common.metrics.expanded_nodes,
            "raw_waypoint_count": common.metrics.raw_waypoint_count,
            "output_waypoint_count": common.metrics.output_waypoint_count,
            "corridor_box_count": common.metrics.corridor_box_count,
            "path_length_m": common.metrics.path_length_m,
            "minimum_residual_clearance_m": common.metrics.minimum_clearance_m,
            "cache_hit": common.metrics.cache_hit,
            "waypoints_enu_m": common.waypoints_enu_m.tolist(),
        },
    }
    if not include_bspline:
        document["guidance_candidates"] = {
            "px4_native_waypoints": {
                "success": common.success,
                "reason": common.metrics.reason,
            }
        }
        return document

    comparison = compare_planning_output_modes_3d(
        common,
        occupancy,
        TrajectoryLimits3D(4.0, 2.0, 4.0),
        nominal_speed_m_s=nominal_speed_m_s,
        bspline_options={
            "control_point_spacing_m": 6.0,
            "maximum_optimization_iterations": 200,
            "wall_clock_deadline_s": 20.0,
            "reference_weight": 1.0,
            "acceleration_weight": 10.0,
            "jerk_weight": 20.0,
        },
    )
    document["guidance_candidates"] = {
        "px4_native_waypoints": _candidate(comparison.waypoint_metrics),
        "sfc_bspline": _candidate(comparison.bspline_metrics),
    }
    return document


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=None)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--start", nargs=3, type=float, default=(-120.0, 115.0, 11.5))
    parser.add_argument("--goal", nargs=3, type=float, default=(200.0, -128.0, 11.5))
    parser.add_argument("--maximum-z-m", type=float, default=120.0)
    parser.add_argument("--surface-clearance-m", type=float, default=10.0)
    parser.add_argument("--nominal-speed-m-s", type=float, default=3.0)
    parser.add_argument("--skip-bspline", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    city = (
        CityMap.from_yaml(args.map)
        if args.map is not None
        else load_preferred_city_map(args.repository_root)
    )
    result = screen(
        city,
        args.start,
        args.goal,
        maximum_z_m=args.maximum_z_m,
        nominal_speed_m_s=args.nominal_speed_m_s,
        include_bspline=not args.skip_bspline,
        minimum_surface_clearance_m=args.surface_clearance_m,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
