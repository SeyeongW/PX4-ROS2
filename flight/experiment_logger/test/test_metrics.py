import math

import yaml

from experiment_logger.metrics import (
    ClearanceMap, ExperimentAccumulator, parse_experiment_metrics,
    parse_plan_latency_ms, parse_sfc_generation_ms,
)


def _map(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump({
        "frames": {"map": {
            "heading_deg_enu": 0.0, "origin_enu_m": [0.0, 0.0, 0.0]}},
        "spawn": {"gazebo_spawn_pose_enu": {"x": 0.0, "y": 0.0}},
        "mission": {
            "coordinate_frame": "map", "vehicle_clearance_xy_m": 1.0,
            "obstacles": [{
                "name": "box", "center_m": [5.0, 0.0, 1.0],
                "size_m": [2.0, 2.0, 2.0]}]},
    }), encoding="utf-8")
    return ClearanceMap.from_yaml(path)


def test_required_metrics_and_legacy_aliases(tmp_path):
    metrics = ExperimentAccumulator(clearance_map=_map(tmp_path))
    metrics.set_state("MISSION")
    metrics.set_setpoint(0.0, 0.0, 0.0, 1.0)
    metrics.add_position(0.0, 0.0, 0.0, 1.0)
    metrics.set_setpoint(1.0, 0.0, 0.0, 2.0)
    metrics.add_position(2.0, 0.0, 0.0, 2.0)
    box = [
        ((-1.0, -1.0, -1.0), (3.0, 1.0, 1.0)),
        ((-1.0, -2.0, -1.0), (3.0, 2.0, 1.0)),
    ]
    metrics.set_state("MISSION_PLAN")
    metrics.add_sfc_snapshot(1, box)
    metrics.set_state("RETURN_PLAN")
    metrics.add_sfc_snapshot(2, box)
    metrics.add_sfc_snapshot(3, box)
    metrics.add_position(2.0, 0.0, 0.0, 2.5)
    metrics.add_log(
        "global A*/B-spline: 20 samples, 5.0 m, 12 A* expansions, "
        "0.25 s, SFC 3.5 ms")
    metrics.set_state("LANDING_ACQUIRE")
    metrics.set_cue(2.0, 0.0, 0.0)
    metrics.add_aruco(True)
    metrics.add_aruco(False)
    metrics.add_position(2.5, 0.0, 0.0, 3.0)
    metrics.add_log(
        "EXPERIMENT_METRICS marker_hits=1 marker_frames=2 "
        "landing_error_3d_m=0.2 landing_xy_error_m=0.1 "
        "touchdown_relative_speed_3d_m_s=0.3 "
        "touchdown_relative_vertical_speed_m_s=0.05 "
        "mpc_count=2 mpc_total_ms=8 mpc_max_ms=5")

    summary = metrics.summary()
    assert math.isclose(summary["path_length_m"], 2.5)
    assert math.isclose(summary["tracking_error_mean_m"], 0.5)
    assert math.isclose(summary["tracking_error_rmse_m"], math.sqrt(0.5))
    assert math.isclose(summary["min_clearance_m"], 1.5)
    assert summary["astar_plan_time_ms"] == 250.0
    assert summary["mpc_solve_time_ms"] == 4.0
    assert summary["replan_count"] == 1
    assert summary["sfc_generation_time_ms"] == 3.5
    assert summary["sfc_min_width_m"] == 2.0
    assert summary["sfc_avg_width_m"] == 3.0
    assert summary["sfc_corridor_count"] == 2
    assert summary["sfc_violation_count"] == 0
    assert summary["aruco_detection_rate_pct"] == 50.0
    assert summary["landing_xy_error_m"] == 0.1
    assert summary["touchdown_relative_speed_m_s"] == 0.3
    assert summary["path_tracking_rmse_m"] == summary["tracking_error_rmse_m"]


def test_existing_log_parsers():
    assert parse_plan_latency_ms(
        "global A*/B-spline: 3843 samples, 508.5 m, 44 A* expansions, "
        "2.30 s, target drift 0.10 m") == 2300.0
    assert parse_sfc_generation_ms(
        "2.30 s, target drift 0.10 m, SFC 4.25 ms") == 4.25
    values = parse_experiment_metrics(
        "prefix EXPERIMENT_METRICS marker_hits=4 marker_frames=5 "
        "landing_xy_error_m=nan")
    assert values["marker_hits"] == 4.0
    assert math.isnan(values["landing_xy_error_m"])


def test_sfc_violation_is_counted_from_active_boxes(tmp_path):
    metrics = ExperimentAccumulator(clearance_map=_map(tmp_path))
    metrics.set_state("MISSION_PLAN")
    assert metrics.add_sfc_snapshot(
        1, [((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))])
    assert not metrics.add_sfc_snapshot(
        1, [((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))])
    metrics.set_state("MISSION")
    # The implementation's SFC safety contract is planar; the z thickness is
    # only a Marker/RViz slab and must not create a false violation.
    metrics.add_position(0.0, 0.0, 5.0, 1.0)
    metrics.add_position(2.0, 0.0, 0.0, 2.0)
    metrics.add_position(2.0, 0.0, 0.0, 3.0)
    metrics.add_position(0.0, 0.0, 0.0, 4.0)
    metrics.add_position(2.0, 0.0, 0.0, 5.0)
    summary = metrics.summary()
    assert summary["sfc_violation_count"] == 2
    assert summary["sfc_violation_event_count"] == 2
    assert summary["sfc_violation_sample_count"] == 3
    assert summary["sfc_violation_rate_pct"] == 60.0
    assert summary["sfc_evaluated_samples"] == 5
