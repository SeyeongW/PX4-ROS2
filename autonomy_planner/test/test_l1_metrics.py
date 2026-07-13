from dataclasses import replace

import numpy as np
import pytest

from autonomy_planner.l1_metrics import (
    ExecutionTrace,
    ExperimentContract,
    MetricsContractError,
    assert_paired_contracts,
    comparison_delta,
    compute_guidance_metrics,
    route_sha256,
)


ROUTE = np.array(((0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (20.0, 0.0, 10.0)))


def contract(method):
    return ExperimentContract(
        guidance_method=method,
        map_revision="map-1",
        occupancy_sha256="a" * 64,
        route_sha256=route_sha256(ROUTE),
        planner_run_id="planner-1",
        px4_git_revision="d6f12ad1c4f70ad3230afd7d86e971421e02fef4",
        scenario_seed=42,
        hard_radius_m=1.35,
        minimum_surface_clearance_m=10.0,
        wind_profile_id="calm",
        payload_profile_id="x500-depth-lidar",
        initial_position_enu_m=(0.0, 0.0, 10.0),
    )


def trace(y_error=0.0):
    timestamp = np.linspace(0.0, 10.0, 21)
    position = np.column_stack(
        (np.linspace(0.0, 20.0, 21), np.full(21, y_error), np.full(21, 10.0))
    )
    position[0, 1] = 0.0
    position[-1, 1] = 0.0
    velocity = np.tile((2.0, 0.0, 0.0), (21, 1))
    clearance = np.full(21, 3.0)
    return ExecutionTrace.from_sequences(timestamp, position, velocity, clearance)


def test_same_metrics_interface_for_both_guidance_methods():
    l1 = compute_guidance_metrics(contract("px4_mc_l1_style"), trace(0.4), ROUTE)
    bspline = compute_guidance_metrics(contract("bspline_mpc"), trace(0.1), ROUTE)
    assert l1.completed and bspline.completed
    assert l1.rms_cross_track_error_m > bspline.rms_cross_track_error_m
    delta = comparison_delta(l1, bspline)
    assert delta["l1_minus_bspline_rms_cross_track_error_m"] > 0.0
    assert delta["both_hard_clearance_safe"] is True


def test_pairing_rejects_changed_seed_map_route_or_firmware():
    l1 = contract("px4_mc_l1_style")
    bspline = replace(contract("bspline_mpc"), scenario_seed=43)
    with pytest.raises(MetricsContractError, match="scenario_seed"):
        assert_paired_contracts(l1, bspline)


def test_clearance_violations_are_counted_against_same_hard_radius():
    unsafe_trace = trace()
    unsafe_trace.clearance_m[5:8] = 1.0
    result = compute_guidance_metrics(contract("px4_mc_l1_style"), unsafe_trace, ROUTE)
    assert result.hard_clearance_violations == 3
    assert result.min_clearance_m == 1.0
