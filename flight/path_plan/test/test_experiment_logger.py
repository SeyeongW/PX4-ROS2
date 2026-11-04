import csv
import math
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from std_msgs.msg import Float32MultiArray

from pathlib import Path

from path_plan.experiment_logger import (
    SUMMARY_METRICS, CsvSink, ExperimentLoggerNode, ExperimentMetrics,
    _horizontal_corridor_widths, _polyline_distance_xy, _read_map_metadata)


def test_metrics_and_two_csv_outputs(tmp_path):
    metrics = ExperimentMetrics()
    metrics.start_at([0.0, 0.0, 0.0])
    metrics.add_pose([3.0, 4.0, 0.0])
    metrics.add_pose([3.0, 4.0, 12.0])
    metrics.add_tracking_error(1.0)
    metrics.add_tracking_error(2.0)
    metrics.add_clearance(3.0)
    metrics.add_clearance(0.75)
    metrics.add_astar(10.0, initial_for_leg=True, expanded_nodes=51)
    metrics.add_astar(20.0, initial_for_leg=False, expanded_nodes=735)
    metrics.add_mpc(2.0)
    metrics.add_mpc(4.0)
    metrics.add_sfc(5.0, 2.0, 4.0, 3)
    metrics.add_sfc(7.0, 1.0, 2.0, 2)
    metrics.add_sfc_evaluation()
    metrics.add_sfc_violation()
    metrics.add_sfc_violation()
    for detected in (True, False, True):
        metrics.add_detection(detected)
    metrics.add_relative_xy(0.2)
    metrics.add_relative_xy(0.4)
    metrics.landing_xy_error_m = 0.1
    metrics.touchdown_relative_speed_m_s = 0.3
    metrics.add_velocity([3.0, 4.0, 0.0])       # speed 5
    metrics.add_velocity([0.0, 0.0, 6.0])       # speed 6 -> peak
    metrics.add_acceleration([0.0, 0.0, 0.0])   # 0
    metrics.add_acceleration([3.0, 4.0, 0.0])   # 5 -> peak; rms sqrt(12.5)
    metrics.add_trailer_range(5.0)
    metrics.add_trailer_range(1.2)              # closest approach
    metrics.add_gps(12, 0.8)
    metrics.add_gps(9, 1.5)                     # fewest sats / worst H accuracy
    metrics.add_battery(16.4)
    metrics.add_battery(15.1)                   # lowest voltage

    summary = metrics.summary()
    assert summary['max_speed_m_s'] == 6.0
    assert summary['max_accel_m_s2'] == 5.0
    assert math.isclose(summary['accel_rms_m_s2'], math.sqrt(12.5))
    assert summary['path_length_m'] == 17.0
    assert summary['tracking_error_mean_m'] == 1.5
    assert summary['tracking_error_max_m'] == 2.0
    assert math.isclose(summary['tracking_error_rmse_m'], math.sqrt(2.5))
    assert summary['min_clearance_m'] == 0.75
    assert summary['astar_plan_time_ms'] == 15.0
    assert summary['astar_expanded_nodes_mean'] == 393.0
    assert summary['astar_expanded_nodes_max'] == 735.0
    assert summary['trailer_range_min_m'] == 1.2
    assert summary['gps_satellites_min'] == 9.0
    assert summary['gps_h_acc_max_m'] == 1.5
    assert summary['battery_voltage_min_v'] == 15.1
    assert summary['mpc_solve_time_ms'] == 3.0
    assert summary['sfc_generation_time_ms'] == 6.0
    assert summary['sfc_min_width_m'] == 1.0
    assert math.isclose(summary['sfc_avg_width_m'], 3.2)
    assert summary['sfc_corridor_count'] == 5
    assert summary['sfc_violation_count'] == 2
    assert summary['replan_count'] == 1
    assert math.isclose(summary['aruco_detection_rate_pct'], 200.0 / 3.0)
    assert math.isclose(summary['relative_xy_error_m'], 0.3)

    sink = CsvSink(str(tmp_path))
    sink.write(1.0, 'pose', 'MISSION', path_length_m=17.0)
    sink.close(summary, ended_wall=datetime.now(timezone.utc), metadata={
        'end_reason': 'test',
        'successful_plan_count': metrics.successful_plans,
        'pose_sample_count': metrics.pose_samples,
        'tracking_sample_count': len(metrics.tracking_errors),
        'sfc_evaluation_count': metrics.sfc_evaluation_count,
        'aruco_sample_count': metrics.aruco_total,
        'landing_xy_error_source': 'vision_handoff',
        'touchdown_relative_speed_source': 'onboard_estimate_test',
    })

    assert sorted(path.name for path in tmp_path.glob('*.csv')) == sorted((
        sink.timeseries_path.name, sink.summary_path.name))
    with sink.summary_path.open(newline='', encoding='utf-8') as stream:
        row = next(csv.DictReader(stream))
    assert set(SUMMARY_METRICS).issubset(row)
    assert float(row['path_length_m']) == 17.0
    assert row['landing_xy_error_source'] == 'vision_handoff'


def test_kinematics_are_blank_until_a_sample_and_ignore_non_finite():
    metrics = ExperimentMetrics()
    summary = metrics.summary()
    assert summary['max_speed_m_s'] is None
    assert summary['max_accel_m_s2'] is None
    assert summary['accel_rms_m_s2'] is None
    metrics.add_velocity([float('nan'), 0.0, 0.0])       # dropped
    metrics.add_acceleration([1.0, 2.0])                 # wrong shape, dropped
    assert metrics.summary()['max_speed_m_s'] is None
    assert metrics.summary()['accel_rms_m_s2'] is None
    metrics.add_velocity([1.0, 0.0, 0.0])
    assert metrics.summary()['max_speed_m_s'] == 1.0


def test_map_metadata_records_the_flown_map_for_reproducibility():
    maze = str(Path(__file__).parents[1] / 'config' / 'drone_maze_route.yaml')
    meta = _read_map_metadata(maze)
    assert meta['map_file'] == 'drone_maze_route.yaml'
    assert meta['map_drone_relative'] is True
    assert meta['goal_x_m'] == 50.0 and meta['goal_y_m'] == 50.0
    assert meta['obstacle_count'] == 25
    assert meta['cruise_altitude_m'] == 5.0
    # Defensive: an empty/absent map yields nothing, never an exception.
    assert _read_map_metadata('') == {}


def test_tracking_error_is_to_active_polyline_not_only_samples():
    path = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    assert _polyline_distance_xy([5.0, 3.0], path) == 3.0


def test_sfc_width_is_minimum_horizontal_span():
    low = np.array([[0.0, 0.0, -1.0], [1.0, 2.0, -10.0]])
    high = np.array([[4.0, 3.0, 1.0], [6.0, 4.0, 10.0]])
    assert np.array_equal(_horizontal_corridor_widths(low, high), [3.0, 2.0])


def test_sfc_violation_is_blank_until_corridor_was_evaluated():
    metrics = ExperimentMetrics()
    metrics.add_sfc(1.0, 2.0, 3.0, 4)
    assert metrics.summary()['sfc_violation_count'] is None
    metrics.add_sfc_evaluation()
    assert metrics.summary()['sfc_violation_count'] == 0


def test_sfc_excursion_count_starts_after_first_corridor_entry():
    state = SimpleNamespace(
        _position=None, _position_t=float('nan'), recording=True,
        metrics=ExperimentMetrics(), _active_path=None, phase='MISSION',
        clearance=None, pose_is_map_frame=True,
        _active_corridor_min=np.array([[0.0, 0.0, -1.0]]),
        _active_corridor_max=np.array([[1.0, 1.0, 1.0]]),
        _sfc_seen_inside=False, _sfc_outside=False,
        _update_clearance_anchor=lambda: None,
        _write=lambda *_args, **_kwargs: None,
        _now=lambda: 0.0,
    )
    ExperimentLoggerNode._handle_pose(state, np.array([-1.0, 0.5, 0.0]), 0.0)
    assert state.metrics.sfc_evaluation_count == 0
    ExperimentLoggerNode._handle_pose(state, np.array([0.5, 0.5, 0.0]), 1.0)
    ExperimentLoggerNode._handle_pose(state, np.array([2.0, 0.5, 0.0]), 2.0)
    ExperimentLoggerNode._handle_pose(state, np.array([3.0, 0.5, 0.0]), 3.0)
    assert state.metrics.sfc_evaluation_count == 3
    assert state.metrics.sfc_violation_count == 1


def test_compact_hardware_path_contract():
    events = []
    state = type('State', (), {
        '_active_path': None,
        '_write': lambda _self, event: events.append(event),
    })()
    message = Float32MultiArray(data=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ExperimentLoggerNode._on_active_path_xy(state, message)
    assert np.array_equal(
        state._active_path, [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    assert events == ['path']
