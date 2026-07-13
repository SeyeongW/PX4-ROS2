"""Tests for the ROS-independent global planner and local MPC."""

from pathlib import Path

import numpy as np
import pytest

from autonomy_planner.planner_core import (
    AStarPlanner,
    CityMap,
    DoubleIntegratorMPC,
    FrameTransform,
    OccupancyGrid,
    arc_length_resample,
    build_safe_flight_corridor,
    plan_smoothed_path,
    simplify_line_of_sight,
    smooth_path_in_corridor,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CITY_COORDINATES = REPOSITORY_ROOT / "gazebo" / "maps" / "city_coordinates.yaml"
START_ENU = np.asarray((-120.0, 115.0))
GOAL_ENU = np.asarray((200.0, -125.0))


@pytest.fixture(scope="module")
def city_map() -> CityMap:
    return CityMap.from_yaml(CITY_COORDINATES)


@pytest.fixture(scope="module")
def grid(city_map: CityMap) -> OccupancyGrid:
    return OccupancyGrid.from_city_map(city_map, resolution_m=2.0, inflation_m=2.0)


@pytest.fixture(scope="module")
def planned(grid: OccupancyGrid):
    return plan_smoothed_path(
        grid,
        START_ENU,
        GOAL_ENU,
        clearance_weight=2.0,
        corridor_radius_m=6.0,
        output_spacing_m=2.0,
    )


def test_loads_real_city_yaml_and_builds_two_metre_inflated_raster(
    city_map: CityMap, grid: OccupancyGrid
) -> None:
    assert city_map.bounds_x_m == (-250.0, 250.0)
    assert city_map.bounds_y_m == (-250.0, 250.0)
    assert len(city_map.buildings) == 274
    assert grid.resolution_m == 2.0
    assert grid.shape == (250, 250)
    assert np.count_nonzero(grid.occupied) > np.count_nonzero(grid.base_occupied)
    assert grid.is_free(START_ENU)
    assert grid.is_free(GOAL_ENU)

    first_building = city_map.buildings[0]
    centroid = np.mean(np.asarray(first_building.outer), axis=0)
    assert first_building.contains_xy(centroid)
    assert not grid.is_free(centroid)


def test_astar_to_requested_goal_has_no_collision_or_corner_cutting(
    grid: OccupancyGrid,
) -> None:
    path = AStarPlanner(grid, clearance_weight=2.0).plan(START_ENU, GOAL_ENU)
    assert np.allclose(path[0], START_ENU)
    assert np.allclose(path[-1], GOAL_ENU)
    assert grid.path_is_free(path)

    cells = [grid.world_to_grid(point) for point in path]
    for previous, current in zip(cells, cells[1:]):
        row_delta = current[0] - previous[0]
        column_delta = current[1] - previous[1]
        assert max(abs(row_delta), abs(column_delta)) <= 1
        if row_delta and column_delta:
            assert not grid.occupied[previous[0] + row_delta, previous[1]]
            assert not grid.occupied[previous[0], previous[1] + column_delta]


def test_los_sfc_and_bspline_output_are_all_checked_free(
    grid: OccupancyGrid, planned
) -> None:
    assert len(planned.simplified) <= len(planned.raw_astar)
    assert len(planned.corridor.segments) == len(planned.simplified) - 1
    assert grid.path_is_free(planned.simplified)
    assert grid.path_is_free(planned.trajectory)
    assert planned.corridor.contains_path(planned.trajectory)
    assert all(segment.radius_m > 0.0 for segment in planned.corridor.segments)
    assert np.allclose(planned.trajectory[0], START_ENU)
    assert np.allclose(planned.trajectory[-1], GOAL_ENU)

    # The cubic fit may be rejected when it overshoots a narrow capsule; that
    # is an expected safety result, and the checked centreline is then used.
    if planned.used_bspline:
        assert "accepted cubic B-spline" in planned.smoothing_message
    else:
        assert "polyline fallback" in planned.smoothing_message


def test_individual_planning_stages_and_arc_length_resampling(
    grid: OccupancyGrid,
) -> None:
    raw = AStarPlanner(grid).plan(START_ENU, GOAL_ENU)
    simplified = simplify_line_of_sight(raw, grid)
    corridor = build_safe_flight_corridor(simplified, grid, maximum_radius_m=5.0)
    result = smooth_path_in_corridor(
        simplified, grid, corridor, output_spacing_m=3.0, smoothing=1.0
    )
    assert grid.path_is_free(result.points)
    assert corridor.contains_path(result.points)

    resampled = arc_length_resample(result.points, spacing_m=4.0)
    segment_lengths = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
    assert np.all(segment_lengths <= 4.0 + 1.0e-6)
    assert np.allclose(resampled[[0, -1]], result.points[[0, -1]])


def test_enu_px4_ned_transform_matches_map_definition(city_map: CityMap) -> None:
    transform = FrameTransform.from_city_map(city_map)
    origin = np.asarray(city_map.px4_origin_enu_m)
    assert np.allclose(transform.enu_to_ned(origin), np.zeros(3))

    # ENU goal at 10 m above the spawn datum is north=-240, east=320,
    # down=-10 in PX4 local NED.
    goal_enu_3d = np.asarray((200.0, -125.0, origin[2] + 10.0))
    goal_ned = transform.enu_to_ned(goal_enu_3d)
    assert np.allclose(goal_ned, (-240.0, 320.0, -10.0))
    assert np.allclose(transform.ned_to_enu(goal_ned), goal_enu_3d)
    assert transform.yaw_enu_to_ned(0.0) == pytest.approx(np.pi / 2.0)
    assert transform.yaw_enu_to_ned(np.pi / 2.0) == pytest.approx(0.0)


def test_double_integrator_mpc_tracks_reference_with_bounded_acceleration() -> None:
    controller = DoubleIntegratorMPC(
        dt_s=0.2,
        horizon_steps=10,
        acceleration_limit_m_s2=(2.0, 2.5, 1.5),
    )
    position = np.zeros(3)
    velocity = np.zeros(3)
    reference_position = np.asarray((5.0, -2.0, 1.0))
    result = controller.solve(
        position,
        velocity,
        reference_position,
        reference_velocities_m_s=np.zeros(3),
    )
    assert result.success
    assert not result.used_fallback
    assert result.predicted_positions_m.shape == (10, 3)
    assert result.predicted_velocities_m_s.shape == (10, 3)
    assert np.all(np.abs(result.acceleration_m_s2) <= (2.0, 2.5, 1.5))
    assert np.linalg.norm(result.predicted_positions_m[-1] - reference_position) < np.linalg.norm(
        position - reference_position
    )


def test_mpc_bounded_pd_fallback_is_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = DoubleIntegratorMPC(
        dt_s=0.1, horizon_steps=5, acceleration_limit_m_s2=1.25
    )

    def fail_minimize(*_args, **_kwargs):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr("autonomy_planner.planner_core.optimize.minimize", fail_minimize)
    result = controller.solve(
        position_m=(0.0, 0.0, 0.0),
        velocity_m_s=(0.0, 0.0, 0.0),
        reference_positions_m=(20.0, -20.0, 5.0),
    )
    assert not result.success
    assert result.used_fallback
    assert np.all(np.isfinite(result.acceleration_m_s2))
    assert np.all(np.abs(result.acceleration_m_s2) <= 1.25)
    assert "injected optimizer failure" in result.message
