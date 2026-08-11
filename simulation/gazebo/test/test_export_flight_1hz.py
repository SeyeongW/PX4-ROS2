import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "simulation/gazebo/tools/export_flight_1hz.py"
SPEC = importlib.util.spec_from_file_location("export_flight_1hz", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def test_one_hz_interval_keeps_native_rate_spike():
    times = np.array([0.0, 0.40, 0.49, 0.51, 0.60, 1.0])
    values = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])

    result = EXPORT.interval_stats(times, values, 0.0, 1.0)

    assert result["count"] == 5
    assert result["max"] == 4.0
    assert result["mean"] < 1.0


def test_phase_interval_marks_mixed_transition_bin():
    events = [(10.0, "PRECHECK"), (12.25, "TAKEOFF"), (14.0, "READY")]

    assert EXPORT.phase_interval(events, 11.0, 12.0) == ("PRECHECK", 1.0, 0)
    phase, fraction, transition = EXPORT.phase_interval(events, 12.0, 13.0)
    assert (phase, transition) == ("TAKEOFF", 1)
    assert fraction == pytest.approx(0.75)


def test_odometry_uses_sim_time_and_local_spawn(tmp_path):
    message = {
        "header": {"stamp": {"sec": "7", "nsec": 500_000_000}},
        "pose": {"position": {"x": 12.0, "y": 23.0, "z": 0.0}},
        "twist": {"linear": {"x": 1.0, "y": -2.0}},
    }
    path = tmp_path / "trailer_odometry.jsonl"
    path.write_text(json.dumps(message) + "\n", encoding="utf-8")

    timestamp, east, north, up, ve, vn = EXPORT._load_odometry(
        path, np.array([10.0, 20.0]), -0.24)

    assert timestamp.tolist() == [7.5]
    assert (east[0], north[0], up[0]) == (2.0, 3.0, -0.24)
    assert (ve[0], vn[0]) == (1.0, -2.0)


def test_odometry_rotates_body_twist_to_world_enu(tmp_path):
    half_sqrt = 2.0 ** -0.5
    message = {
        "header": {"stamp": {"sec": "8"}},
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"z": half_sqrt, "w": half_sqrt},
        },
        "twist": {"linear": {"x": 1.0, "y": 0.0}},
    }
    path = tmp_path / "trailer_odometry.jsonl"
    path.write_text(json.dumps(message) + "\n", encoding="utf-8")

    *_, ve, vn = EXPORT._load_odometry(path, np.zeros(2), -0.24)

    assert ve[0] == pytest.approx(0.0, abs=1.0e-12)
    assert vn[0] == pytest.approx(1.0)


def test_csv_float_precision_preserves_one_second_at_epoch_scale():
    first = float(EXPORT._clean(1_786_424_268.25))
    second = float(EXPORT._clean(1_786_424_269.25))
    assert second - first == pytest.approx(1.0)


def test_sample_gap_includes_gap_crossing_both_bin_boundaries():
    assert EXPORT._sample_gap(
        np.array([0.9, 2.1]), 1.0, 2.0) == pytest.approx(1.2)


def test_clock_mapper_preserves_pause_offset_step():
    sync = SimpleNamespace(
        name="timesync_status",
        multi_id=0,
        data={
            "timestamp": np.array([0, 1, 2, 3, 4], dtype=np.uint64) * 1_000_000,
            "remote_timestamp": np.array([
                1_786_000_000, 1_786_000_001, 1_786_000_002,
                1_786_000_008, 1_786_000_009,
            ], dtype=np.uint64) * 1_000_000,
        },
    )
    clock = EXPORT._clock_mapper(SimpleNamespace(data_list=[sync]))

    assert float(clock(2.0)) == pytest.approx(1_786_000_002.0)
    assert float(clock(3.0)) == pytest.approx(1_786_000_008.0)


def test_descent_warning_distinguishes_command_and_actual_excess():
    threshold = 0.7 + EXPORT.DESCENT_WARN_MARGIN_M_S
    assert EXPORT.classify_descent_spike(
        "PRECLAND", 1, 0.0, 1, 0.76, threshold) == (
        True, "command_excess")
    assert EXPORT.classify_descent_spike(
        "PRECLAND", 1, 0.2, 0, np.nan, threshold) == (
        True, "actual_only_excess")
    assert EXPORT.classify_descent_spike(
        "PRECLAND", 1, 0.0, 1, 0.731, threshold) == (False, "")
    assert EXPORT.classify_descent_spike(
        "TAKEOFF", 1, 1.0, 2, 1.0, threshold) == (False, "")


def test_groundtruth_height_uses_one_world_frame():
    groundtruth = {
        "timestamp": np.array([1_000_000, 2_000_000]),
        "z": np.array([-7.051, -2.051]),
    }

    times, height = EXPORT._groundtruth_height(groundtruth, 2.051)

    assert times.tolist() == [1.0, 2.0]
    assert height.tolist() == pytest.approx([5.0, 0.0])


def test_launcher_runs_best_effort_postflight_export():
    launcher = (ROOT / "simulation/gazebo/run_gimbal.sh").read_text()
    assert 'tools/export_flight_1hz.py' in launcher
    assert 'flight_csv_1hz' in launcher
    assert 'flight_summary_csv' in launcher
    assert 'map.yaml' in launcher
    assert 'coordinates_source' in launcher
    assert 'coordinates_sha256' in launcher
    assert "printf 'coordinates\\t%s\\n' 'map.yaml'" in launcher
    assert 'PX4_MAP_COORDINATES="$LANDING_COORDINATES"' in launcher
    assert 'WARNING: 1 Hz flight CSV export failed' in launcher
    assert launcher.index('flight.ulg') < launcher.index('tools/export_flight_1hz.py')
