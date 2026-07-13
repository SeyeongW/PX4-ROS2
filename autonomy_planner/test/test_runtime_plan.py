from __future__ import annotations

import threading
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from autonomy_planner.planner_core import CityMap, FrameTransform
from autonomy_planner.planner_3d import (
    CorridorBox3D,
    PlanningOutputMode,
    SafeFlightCorridor3D,
    TrajectoryLimits3D,
)
from autonomy_planner.runtime_plan import (
    ActivePlan3D,
    CorridorViolationError,
    LatestPlanWorker3D,
    PlanWorkerStatus,
    RuntimePlanAdapter3D,
    RuntimePlanner3DConfig,
    RuntimePlanningError,
    enu_box_to_px4_local_ned_bounds,
    map_horizon_references_to_ned,
)


def _write_open_city(path: Path) -> CityMap:
    path.write_text(
        """\
schema_version: 2
map:
  gazebo_world_name: runtime_test
  bounds_enu_m: {x: [-12.0, 12.0], y: [-12.0, 12.0]}
frames:
  px4_local:
    origin_enu_m: [-2.0, 3.0, 0.0]
planner_defaults:
  vehicle_minimum_xy_size_m: [1.0, 1.0]
  hard_inflation_radius_m: 1.45
  preferred_clearance_m: 2.15
obstacles:
  buildings: []
""",
        encoding="utf-8",
    )
    return CityMap.from_yaml(path)


def _test_config() -> RuntimePlanner3DConfig:
    return RuntimePlanner3DConfig(
        maximum_z_m=20.0,
        coarse_resolution_m=(2.0, 2.0, 2.0),
        fine_resolution_m=(1.0, 1.0, 1.0),
        coarse_timeout_s=2.0,
        fine_timeout_s=2.0,
        coarse_max_expanded_nodes=20_000,
        fine_max_expanded_nodes=20_000,
        coarse_edge_sample_step_m=0.4,
        fine_edge_sample_step_m=0.2,
        snap_radius_m=2.0,
        fine_search_band_m=3.0,
        fine_vertical_search_band_m=2.0,
        sfc_maximum_margin_m=2.0,
        sfc_maximum_segment_length_m=4.0,
        trajectory_limits=TrajectoryLimits3D(3.0, 2.0, 4.0),
        nominal_speed_m_s=2.0,
        native_sample_spacing_m=0.5,
        bspline_sample_period_s=0.05,
        bspline_control_point_spacing_m=2.0,
        bspline_deadline_s=5.0,
        prepared_cache_entries=2,
    )


def test_enu_box_to_ned_bounds_uses_origin_axis_swap_and_down_sign() -> None:
    transform = FrameTransform((-120.0, 115.0, 0.0))
    bounds = enu_box_to_px4_local_ned_bounds(
        (-10.0, 30.0, 5.0), (20.0, 50.0, 9.0), transform
    )
    assert np.allclose(bounds, (-85.0, -65.0, 110.0, 140.0, -9.0, -5.0))


def test_prepare_once_retains_common_plan_for_bspline_and_px4_modes(
    tmp_path: Path,
) -> None:
    adapter = RuntimePlanAdapter3D(
        _write_open_city(tmp_path / "city.yaml"), _test_config()
    )
    assert adapter.envelope.hard_horizontal_radius_m == pytest.approx(1.45)
    assert adapter.envelope.preferred_horizontal_radius_m == pytest.approx(2.15)
    assert adapter.planner.config.fine.preferred_clearance_m == pytest.approx(0.70)
    assert adapter.occupancy.vertical_inflation_up_m == pytest.approx(10.4)
    assert adapter.minimum_planning_center_z_m == pytest.approx(
        10.4 + 0.5 * np.sqrt(3.0) + 0.1
    )

    with pytest.raises(RuntimePlanningError, match="MINIMUM_SURFACE_CLEARANCE_REJECTED"):
        adapter.prepare((-5.0, 0.0, 10.0), (5.0, 0.0, 10.0))
    planning_z = adapter.minimum_planning_center_z_m
    prepared = adapter.prepare((-5.0, 0.0, planning_z), (5.0, 0.0, planning_z))
    assert adapter.prepare(
        (-5.0, 0.0, planning_z), (5.0, 0.0, planning_z)
    ) is prepared
    native = adapter.activate(prepared, PlanningOutputMode.PX4_NATIVE_WAYPOINTS)
    bspline = adapter.activate(prepared, PlanningOutputMode.SFC_BSPLINE)

    assert native.mode is PlanningOutputMode.PX4_NATIVE_WAYPOINTS
    assert bspline.mode is PlanningOutputMode.SFC_BSPLINE
    assert native.waypoint_plan is prepared.waypoint_plan
    assert bspline.waypoint_plan is prepared.waypoint_plan
    assert native.corridor is bspline.corridor is prepared.corridor
    assert native.sampled_velocity_enu_m_s is None
    assert native.sampled_acceleration_enu_m_s2 is None
    assert bspline.sampled_velocity_enu_m_s is not None
    assert bspline.sampled_acceleration_enu_m_s2 is not None
    assert native.path_length_m == pytest.approx(10.0, abs=0.2)
    assert bspline.path_length_m == pytest.approx(10.0, abs=0.2)


def test_horizon_corridor_assignment_is_all_or_nothing() -> None:
    corridor = SafeFlightCorridor3D(
        boxes=(
            CorridorBox3D((-2.0, -1.0, 1.0), (1.0, 1.0, 4.0)),
            CorridorBox3D((0.0, -1.0, 1.0), (3.0, 1.0, 4.0)),
        ),
        centerline_enu_m=np.asarray(
            ((-1.0, 0.0, 2.0), (0.5, 0.0, 2.0), (2.0, 0.0, 2.0))
        ),
        minimum_overlap_m=0.5,
    )
    transform = FrameTransform((0.0, 0.0, 0.0))
    mapped = map_horizon_references_to_ned(
        ((-1.0, 0.0, 2.0), (0.5, 0.0, 2.0), (2.0, 0.0, 2.0)),
        corridor,
        transform,
    )
    assert mapped.corridor_box_indices == (0, 0, 1)
    assert mapped.bounds_px4_local_ned_m.shape == (3, 6)
    with pytest.raises(CorridorViolationError, match="SFC_REFERENCE_OUTSIDE"):
        map_horizon_references_to_ned(
            ((-1.0, 0.0, 2.0), (9.0, 0.0, 2.0)), corridor, transform
        )


def test_latest_worker_coalesces_obsolete_requests_and_cancels_pending() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[float, float, float]] = []
    sentinel = cast(ActivePlan3D, object())

    def slow_plan(
        _start: tuple[float, float, float],
        goal: tuple[float, float, float],
        _mode: PlanningOutputMode,
    ) -> ActivePlan3D:
        calls.append(goal)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2.0)
        return sentinel

    worker = LatestPlanWorker3D(slow_plan, cache_entries=2)
    try:
        first = worker.submit((0, 0, 1), (1, 0, 1), "px4_native_waypoints")
        assert entered.wait(1.0)
        second = worker.submit((0, 0, 1), (2, 0, 1), "px4_native_waypoints")
        third = worker.submit((0, 0, 1), (3, 0, 1), "px4_native_waypoints")
        assert worker.wait(first, 0.2).status is PlanWorkerStatus.SUPERSEDED
        assert worker.wait(second, 0.2).status is PlanWorkerStatus.SUPERSEDED
        assert worker.cancel(third)
        assert worker.wait(third, 0.2).status is PlanWorkerStatus.CANCELLED
        release.set()

        fourth = worker.submit((0, 0, 1), (4, 0, 1), "sfc_bspline")
        assert worker.wait(fourth, 1.0).status is PlanWorkerStatus.SUCCEEDED
        assert calls == [(1.0, 0.0, 1.0), (4.0, 0.0, 1.0)]

        cached = worker.submit((0, 0, 1), (4, 0, 1), "sfc_bspline")
        assert worker.wait(cached, 0.2).status is PlanWorkerStatus.SUCCEEDED
        assert calls == [(1.0, 0.0, 1.0), (4.0, 0.0, 1.0)]
        assert worker.stats.coalesced >= 2
        assert worker.stats.cancelled == 1
        assert worker.stats.cache_hits == 1
    finally:
        release.set()
        worker.close(2.0)
