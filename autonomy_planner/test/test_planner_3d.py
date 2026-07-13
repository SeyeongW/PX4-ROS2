from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from autonomy_planner.planner_core import CityMap, PolygonPrism
from autonomy_planner.planner_3d import (
    AStar3DConfig,
    AStarPlanner3D,
    CorridorBox3D,
    HierarchicalAStar3DConfig,
    HierarchicalAStarPlanner3D,
    PrismOccupancy3D,
    SafeFlightCorridor3D,
    TrajectoryLimits3D,
    VehicleEnvelope3D,
    build_safe_flight_corridor_3d,
    compare_planning_output_modes_3d,
    load_preferred_city_map,
    optimize_bspline_trajectory_3d,
    plan_validated_waypoints_3d,
    resolve_city_map_yaml,
)


def _building(
    building_id: str,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
    roof_z_m: float,
    *,
    foundation_z_m: float = 0.0,
) -> PolygonPrism:
    return PolygonPrism(
        building_id,
        (
            (minimum_x, minimum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
            (minimum_x, maximum_y),
        ),
        (),
        foundation_z_m,
        roof_z_m,
    )


def _city(
    *,
    bounds_x: tuple[float, float] = (-10.0, 10.0),
    bounds_y: tuple[float, float] = (-10.0, 10.0),
    buildings: tuple[PolygonPrism, ...] = (),
) -> CityMap:
    return CityMap(
        bounds_x,
        bounds_y,
        buildings,
        (0.0, 0.0, 0.0),
        0.0,
        Path("/synthetic/city_coordinates.yaml"),
    )


def _open_occupancy() -> PrismOccupancy3D:
    return PrismOccupancy3D(
        _city(),
        minimum_z_m=0.0,
        maximum_z_m=20.0,
        ground_z_m=0.0,
        horizontal_inflation_m=0.5,
        vertical_inflation_up_m=0.3,
        vertical_inflation_down_m=0.3,
    )


def test_vehicle_envelope_enforces_one_metre_yaw_invariant_body() -> None:
    envelope = VehicleEnvelope3D(
        body_length_x_m=0.3,
        body_width_y_m=0.4,
        body_height_z_m=0.4,
        map_margin_m=0.10,
        localization_margin_m=0.15,
        tracking_margin_m=0.20,
        control_margin_m=0.15,
        extra_margin_m=0.1428932188134524,
        preferred_extra_m=0.70,
    )
    assert envelope.planning_length_x_m == 1.0
    assert envelope.planning_width_y_m == 1.0
    assert envelope.body_yaw_invariant_radius_m == pytest.approx(math.sqrt(0.5))


def test_vehicle_occupancy_enforces_clearance_below_complete_body() -> None:
    envelope = VehicleEnvelope3D()
    occupancy = PrismOccupancy3D.from_vehicle_envelope(
        _city(), envelope, minimum_surface_clearance_m=10.0
    )
    boundary = 10.0 + envelope.vertical_half_extent_m
    assert occupancy.is_inflated_occupied((0.0, 0.0, boundary))
    assert not occupancy.is_inflated_occupied((0.0, 0.0, boundary + 0.1))
    assert envelope.hard_horizontal_radius_m == pytest.approx(1.45)
    assert envelope.preferred_horizontal_radius_m == pytest.approx(2.15)


def test_map_resolver_prefers_uav_derivative_and_falls_back(tmp_path: Path) -> None:
    maps = tmp_path / "gazebo" / "maps"
    maps.mkdir(parents=True)
    base = maps / "city_coordinates.yaml"
    base.touch()
    assert resolve_city_map_yaml(tmp_path) == base
    derivative = maps / "city_coordinates_uav.yaml"
    derivative.touch()
    assert resolve_city_map_yaml(tmp_path) == derivative
    assert resolve_city_map_yaml(tmp_path, prefer_uav_map=False) == base


def test_derived_city_map_safety_budget_is_reproduced_exactly() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    city = load_preferred_city_map(repository_root)
    envelope = VehicleEnvelope3D.from_city_map_defaults(city)
    if city.source_path.name == "city_coordinates_uav.yaml":
        assert envelope.hard_horizontal_radius_m == pytest.approx(1.45, abs=1.0e-12)
        assert envelope.preferred_horizontal_radius_m == pytest.approx(2.15, abs=1.0e-12)


def test_prism_occupancy_uses_height_holes_and_full_geofence_envelope() -> None:
    atrium = PolygonPrism(
        "atrium",
        ((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)),
        (((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)),),
        0.0,
        4.0,
    )
    occupancy = PrismOccupancy3D(
        _city(bounds_x=(-5.0, 5.0), bounds_y=(-5.0, 5.0), buildings=(atrium,)),
        minimum_z_m=0.0,
        maximum_z_m=10.0,
        ground_z_m=0.0,
        horizontal_inflation_m=0.4,
        vertical_inflation_up_m=0.3,
        vertical_inflation_down_m=0.2,
    )
    assert occupancy.is_occupied((1.0, 0.0, 2.0))
    assert not occupancy.is_occupied((1.0, 0.0, 5.0))
    assert not occupancy.is_occupied((0.0, 0.0, 2.0))
    assert occupancy.is_inflated_occupied((0.45, 0.0, 2.0))
    assert occupancy.highest_surface_below(1.0, 0.0, 8.0) == 4.0
    assert occupancy.is_inflated_occupied((-4.8, 0.0, 2.0))
    assert occupancy.is_inflated_occupied((0.0, 0.0, 9.9))
    assert not occupancy.box_is_free((-4.9, -0.2, 1.0), (-4.0, 0.2, 2.0))


def test_from_city_map_sums_each_horizontal_budget_term_exactly_once() -> None:
    occupancy = PrismOccupancy3D.from_city_map(
        _city(),
        drone_collision_radius_m=0.70,
        map_uncertainty_m=0.10,
        localization_uncertainty_m=0.15,
        tracking_error_margin_m=0.20,
        control_error_margin_m=0.15,
        configured_safety_margin_m=0.15,
    )
    assert occupancy.horizontal_inflation_m == pytest.approx(1.45)


def test_astar_26_neighbor_climbs_over_full_width_prism() -> None:
    wall = _building("wall", -0.75, -1.95, 0.75, 1.95, 3.5)
    occupancy = PrismOccupancy3D(
        _city(bounds_x=(-6.0, 6.0), bounds_y=(-2.0, 2.0), buildings=(wall,)),
        0.0,
        8.0,
        0.0,
        0.0,
        0.2,
        0.2,
    )
    result = AStarPlanner3D(
        occupancy,
        AStar3DConfig(
            resolution_m=1.0,
            connectivity=26,
            preferred_clearance_m=0.0,
            safe_clearance_m=0.0,
            snap_radius_m=2.5,
            planning_timeout_s=2.0,
            max_expanded_nodes=20_000,
            edge_sample_step_m=0.2,
        ),
    ).plan((-4.5, 0.0, 1.5), (4.5, 0.0, 1.5))
    assert result.success, result.metrics
    assert np.max(result.path_enu_m[:, 2]) > wall.roof_z_m + 0.2
    assert result.metrics.expanded_nodes > 0
    assert result.metrics.open_set_peak > 0
    assert all(
        occupancy.segment_is_free(left, right, sample_step_m=0.1)
        for left, right in zip(result.path_enu_m, result.path_enu_m[1:])
    )


def test_astar_rejects_diagonal_corner_cutting() -> None:
    blocker = _building("blocker", 1.0, 0.0, 2.0, 1.0, 3.0)
    occupancy = PrismOccupancy3D(
        _city(bounds_x=(0.0, 4.0), bounds_y=(0.0, 4.0), buildings=(blocker,)),
        0.0,
        5.0,
        0.0,
    )
    planner = AStarPlanner3D(
        occupancy,
        AStar3DConfig(
            resolution_m=1.0,
            preferred_clearance_m=0.0,
            safe_clearance_m=0.0,
        ),
    )
    assert planner.voxel_is_free((0, 0, 1))
    assert planner.voxel_is_free((1, 1, 1))
    assert not planner._transition_is_free((0, 0, 1), (1, 1, 1))


def test_astar_snaps_occupied_start_within_bounded_radius() -> None:
    block = _building("start_block", -0.6, -0.6, 0.6, 0.6, 3.0)
    occupancy = PrismOccupancy3D(
        _city(buildings=(block,)), 0.0, 8.0, 0.0
    )
    result = AStarPlanner3D(
        occupancy,
        AStar3DConfig(
            resolution_m=1.0,
            preferred_clearance_m=0.0,
            safe_clearance_m=0.0,
            snap_radius_m=2.5,
            planning_timeout_s=2.0,
        ),
    ).plan((0.0, 0.0, 1.5), (5.0, 0.0, 1.5))
    assert result.success
    assert result.metrics.start_snapped
    assert result.metrics.start_used_enu_m != result.metrics.start_requested_enu_m


@pytest.mark.parametrize(
    ("timeout_s", "node_limit", "expected_reason"),
    ((1.0e-12, 10_000, "planning_timeout"), (2.0, 1, "expanded_node_limit")),
)
def test_astar_has_bounded_failure_reasons(
    timeout_s: float, node_limit: int, expected_reason: str
) -> None:
    result = AStarPlanner3D(
        _open_occupancy(),
        AStar3DConfig(
            resolution_m=1.0,
            preferred_clearance_m=0.0,
            safe_clearance_m=0.0,
            planning_timeout_s=timeout_s,
            max_expanded_nodes=node_limit,
        ),
    ).plan((-8.0, -8.0, 2.0), (8.0, 8.0, 2.0))
    assert not result.success
    assert result.metrics.reason == expected_reason


def test_astar_reports_no_path_for_ceiling_height_wall() -> None:
    wall = _building("sealed", -0.6, -3.0, 0.6, 3.0, 6.0)
    occupancy = PrismOccupancy3D(
        _city(bounds_x=(-4.0, 4.0), bounds_y=(-3.0, 3.0), buildings=(wall,)),
        0.0,
        6.0,
        0.0,
    )
    result = AStarPlanner3D(
        occupancy,
        AStar3DConfig(
            resolution_m=1.0,
            preferred_clearance_m=0.0,
            safe_clearance_m=0.0,
            planning_timeout_s=2.0,
            max_expanded_nodes=50_000,
        ),
    ).plan((-3.0, 0.0, 1.5), (3.0, 0.0, 1.5))
    assert not result.success
    assert result.metrics.reason == "no_path"


def test_hierarchical_astar_fine_path_and_revalidated_lru_cache() -> None:
    occupancy = _open_occupancy()
    config = HierarchicalAStar3DConfig(
        coarse=AStar3DConfig(
            resolution_m=2.0,
            preferred_clearance_m=0.5,
            safe_clearance_m=0.5,
            planning_timeout_s=2.0,
            max_expanded_nodes=20_000,
        ),
        fine=AStar3DConfig(
            resolution_m=1.0,
            preferred_clearance_m=0.5,
            safe_clearance_m=0.5,
            planning_timeout_s=2.0,
            max_expanded_nodes=20_000,
        ),
        fine_search_band_m=3.0,
        cache_entries=2,
    )
    planner = HierarchicalAStarPlanner3D(occupancy, config)
    first = planner.plan((-7.0, -2.0, 3.0), (7.0, 2.0, 3.0))
    second = planner.plan((-7.0, -2.0, 3.0), (7.0, 2.0, 3.0))
    assert first.success and second.success
    assert first.metrics.fine_attempts == 1
    assert not first.metrics.cache_hit
    assert second.metrics.cache_hit
    assert second.metrics.coarse_expanded_nodes == 0
    assert np.allclose(first.path_enu_m, second.path_enu_m)


def test_common_astar_sfc_input_cross_checks_both_output_modes() -> None:
    occupancy = _open_occupancy()
    planner = HierarchicalAStarPlanner3D(
        occupancy,
        HierarchicalAStar3DConfig(
            coarse=AStar3DConfig(
                resolution_m=2.0,
                preferred_clearance_m=0.5,
                safe_clearance_m=0.5,
                planning_timeout_s=2.0,
                max_expanded_nodes=20_000,
            ),
            fine=AStar3DConfig(
                resolution_m=1.0,
                preferred_clearance_m=0.5,
                safe_clearance_m=0.5,
                planning_timeout_s=2.0,
                max_expanded_nodes=20_000,
            ),
            fine_search_band_m=3.0,
        ),
    )
    common = plan_validated_waypoints_3d(
        planner,
        (-5.0, 0.0, 3.0),
        (5.0, 0.0, 3.0),
        sfc_minimum_overlap_m=0.5,
        sfc_initial_margin_m=0.5,
        sfc_maximum_margin_m=2.0,
        sfc_maximum_segment_length_m=4.0,
    )
    assert common.success, common.metrics
    assert len(common.waypoints_enu_m) == 2
    comparison = compare_planning_output_modes_3d(
        common,
        occupancy,
        TrajectoryLimits3D(2.0, 1.0, 2.0),
        nominal_speed_m_s=3.0,
        bspline_options={
            "control_point_spacing_m": 2.0,
            "wall_clock_deadline_s": 5.0,
        },
    )
    assert comparison.waypoint_metrics.success
    assert comparison.waypoint_metrics.collision_free
    assert comparison.bspline_metrics.success, comparison.bspline_metrics.reason
    assert comparison.bspline_metrics.collision_free
    assert comparison.bspline_metrics.hard_clearance_violation_count == 0


def test_sfc_boxes_overlap_and_record_full_hard_envelope() -> None:
    occupancy = _open_occupancy()
    path = np.asarray(((-5.0, 0.0, 3.0), (0.0, 0.0, 3.0), (5.0, 0.0, 3.0)))
    corridor = build_safe_flight_corridor_3d(
        path,
        occupancy,
        minimum_overlap_m=0.5,
        initial_margin_m=0.5,
        maximum_margin_m=2.0,
        maximum_segment_length_m=4.0,
    )
    valid, reason = corridor.validate(occupancy)
    assert valid, reason
    assert corridor.hard_horizontal_radius_m == occupancy.horizontal_inflation_m
    assert corridor.hard_vertical_up_m == occupancy.vertical_inflation_up_m
    assert corridor.metrics is not None
    assert corridor.metrics.reason == "CONVERGED"
    assert corridor.metrics.minimum_actual_overlap_m >= 0.5
    assert all(left.overlaps(right, 0.5) for left, right in zip(corridor.boxes, corridor.boxes[1:]))


def test_sfc_vectorized_path_containment_preserves_union_and_tolerance() -> None:
    corridor = SafeFlightCorridor3D(
        boxes=(
            CorridorBox3D((-2.0, -1.0, 1.0), (0.5, 1.0, 4.0)),
            CorridorBox3D((0.0, -1.0, 1.0), (2.0, 1.0, 4.0)),
        ),
        centerline_enu_m=np.asarray(
            ((-1.5, 0.0, 2.0), (0.25, 0.0, 2.0), (1.5, 0.0, 2.0))
        ),
        minimum_overlap_m=0.5,
    )
    assert corridor.contains_path(
        np.asarray(((-2.0, -1.0, 1.0), (0.25, 0.0, 2.0), (2.0, 1.0, 4.0)))
    )
    assert corridor.contains_path(np.empty((0, 3)))
    assert corridor.contains_path(((2.0 + 5.0e-8, 0.0, 2.0),))
    assert not corridor.contains_path(((2.0 + 2.0e-7, 0.0, 2.0),))
    assert not corridor.contains_path(((0.0, 3.0, 2.0),))
    assert not corridor.contains_path(((math.nan, 0.0, 2.0),))


def test_endpoint_conditioned_bspline_keeps_long_route_at_nominal_time() -> None:
    occupancy = PrismOccupancy3D(
        _city(bounds_x=(-250.0, 250.0), bounds_y=(-20.0, 20.0)),
        minimum_z_m=0.0,
        maximum_z_m=20.0,
        ground_z_m=0.0,
        horizontal_inflation_m=0.5,
        vertical_inflation_up_m=0.3,
        vertical_inflation_down_m=0.3,
    )
    path = np.asarray(((-200.0, 0.0, 3.0), (200.0, 0.0, 3.0)))
    corridor = build_safe_flight_corridor_3d(
        path,
        occupancy,
        minimum_overlap_m=0.5,
        initial_margin_m=0.5,
        maximum_margin_m=2.0,
        maximum_segment_length_m=8.0,
    )
    result = optimize_bspline_trajectory_3d(
        path,
        occupancy,
        corridor,
        TrajectoryLimits3D(4.0, 2.0, 4.0),
        nominal_speed_m_s=3.0,
        control_point_spacing_m=6.0,
        wall_clock_deadline_s=5.0,
        reference_weight=1.0,
        acceleration_weight=10.0,
        jerk_weight=20.0,
    )
    assert result.success, result.reason
    assert result.time_scaling_iterations == 0
    assert result.final_duration_s == pytest.approx(400.0 / 3.0)
    assert result.validation is not None and result.validation.valid
    assert result.validation.maximum_velocity_m_s <= 4.0 + 2.0e-4
    assert result.validation.maximum_acceleration_m_s2 <= 2.0 + 2.0e-4
    assert result.validation.maximum_jerk_m_s3 <= 4.0 + 2.0e-4


def test_nonstationary_bspline_endpoint_constraints_remain_exact() -> None:
    occupancy = _open_occupancy()
    path = np.asarray(((-5.0, 0.0, 3.0), (5.0, 0.0, 3.0)))
    corridor = build_safe_flight_corridor_3d(
        path,
        occupancy,
        minimum_overlap_m=0.5,
        initial_margin_m=0.5,
        maximum_margin_m=2.0,
        maximum_segment_length_m=4.0,
    )
    initial_velocity = np.asarray((0.5, 0.0, 0.0))
    result = optimize_bspline_trajectory_3d(
        path,
        occupancy,
        corridor,
        TrajectoryLimits3D(3.0, 2.0, 4.0),
        initial_velocity_enu_m_s=initial_velocity,
        nominal_speed_m_s=2.0,
        control_point_spacing_m=2.0,
        wall_clock_deadline_s=5.0,
    )
    assert result.success, result.reason
    assert result.trajectory is not None
    samples = result.trajectory.sample(0.02, 500)
    assert np.allclose(samples.velocity_enu_m_s[0], initial_velocity, atol=2.0e-4)
    assert result.validation is not None and result.validation.valid


def test_sfc_deadline_is_bounded_and_reported() -> None:
    occupancy = _open_occupancy()
    path = np.asarray(((-5.0, 0.0, 3.0), (5.0, 0.0, 3.0)))
    with pytest.raises(TimeoutError, match="SFC_TIMEOUT"):
        build_safe_flight_corridor_3d(
            path,
            occupancy,
            wall_clock_deadline_s=1.0e-12,
        )


def test_constrained_cubic_bspline_endpoints_derivatives_and_time_scaling() -> None:
    occupancy = _open_occupancy()
    path = np.asarray(((-5.0, 0.0, 3.0), (0.0, 0.0, 3.0), (5.0, 0.0, 3.0)))
    corridor = build_safe_flight_corridor_3d(
        path,
        occupancy,
        minimum_overlap_m=0.5,
        initial_margin_m=0.5,
        maximum_margin_m=2.0,
        maximum_segment_length_m=4.0,
    )
    limits = TrajectoryLimits3D(2.0, 1.0, 2.0)
    result = optimize_bspline_trajectory_3d(
        path,
        occupancy,
        corridor,
        limits,
        nominal_speed_m_s=6.0,
        control_point_spacing_m=2.0,
        wall_clock_deadline_s=5.0,
    )
    assert result.success, result.reason
    assert result.termination_reason == "CONVERGED"
    assert result.time_scaling_iterations > 0
    assert result.validation is not None and result.validation.valid
    assert result.trajectory is not None and result.trajectory.degree == 3
    samples = result.trajectory.sample(0.02, 500)
    assert np.allclose(samples.position_enu_m[0], path[0], atol=2.0e-4)
    assert np.allclose(samples.position_enu_m[-1], path[-1], atol=2.0e-4)
    assert np.linalg.norm(samples.velocity_enu_m_s[0]) <= 2.0e-4
    assert np.linalg.norm(samples.acceleration_enu_m_s2[0]) <= 2.0e-4
    assert result.validation.maximum_velocity_m_s <= limits.maximum_velocity_m_s + 2.0e-4
    assert result.validation.maximum_acceleration_m_s2 <= limits.maximum_acceleration_m_s2 + 2.0e-4
    assert result.validation.maximum_jerk_m_s3 <= limits.maximum_jerk_m_s3 + 2.0e-4


def test_bspline_deadline_terminates_without_unbounded_retry() -> None:
    occupancy = _open_occupancy()
    path = np.asarray(((-2.0, 0.0, 3.0), (2.0, 0.0, 3.0)))
    corridor = build_safe_flight_corridor_3d(
        path,
        occupancy,
        minimum_overlap_m=0.2,
        initial_margin_m=0.2,
        maximum_margin_m=1.0,
    )
    result = optimize_bspline_trajectory_3d(
        path,
        occupancy,
        corridor,
        TrajectoryLimits3D(),
        wall_clock_deadline_s=1.0e-12,
    )
    assert not result.success
    assert result.reason == "optimization_timeout"
    assert result.termination_reason == "TIMEOUT"
