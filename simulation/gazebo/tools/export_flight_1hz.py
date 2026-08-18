#!/usr/bin/env python3
"""Export one paper-friendly 1 Hz interval table from a run artifact.

The ULog remains the authoritative, lossless flight record.  Each CSV row is
one [second, second + 1) interval: continuous columns are time-weighted means,
while p95/max/step columns are calculated from the native-rate samples so a
short spike is not hidden by the 1 Hz presentation rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import yaml
from pyulog import ULog


SCHEMA_VERSION = "cju_flight_1hz_v3"
SPEED_STEP_LIMIT_M_S = 0.5
ACCEL_SPIKE_LIMIT_M_S2 = 5.0
BODY_RATE_WARN_DEG_S = 90.0
OBSTACLE_RESERVE_WARN_M = 0.5
DESCENT_WARN_MARGIN_M_S = 0.05
EXPERIMENT_METRIC_FIELDS = (
    "mission_success_rate_pct",
    "path_tracking_rmse_m",
    "path_tracking_error_max_m",
    "marker_detection_rate_pct",
    "landing_error_3d_m",
    "landing_xy_error_m",
    "touchdown_relative_speed_3d_m_s",
    "touchdown_relative_vertical_speed_m_s",
    "mpc_solve_mean_ms",
    "mpc_solve_max_ms",
    "flight_battery_energy_wh",
)
_TRANSITION = re.compile(
    r"\[(\d+(?:\.\d+)?)\].*?\b([A-Z_]+) -> ([A-Z_]+)\b")
_EXPERIMENT_METRICS = re.compile(r"EXPERIMENT_METRICS\s+(.+)$")


def _manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key] = value
    return values


def _topic(ulog: ULog, name: str):
    return next(
        (item.data for item in ulog.data_list
         if item.name == name and item.multi_id == 0), None)


def _finite_time_series(times, values):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    return times[valid], values[valid]


def interval_stats(times, values, start: float, end: float) -> dict[str, float]:
    """Time-weighted mean plus native extrema for one half-open interval."""
    times, values = _finite_time_series(times, values)
    if not len(times):
        return {"mean": math.nan, "min": math.nan, "max": math.nan,
                "p95": math.nan, "count": 0}
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="left"))
    native = values[left:right]
    if not len(native):
        return {"mean": math.nan, "min": math.nan, "max": math.nan,
                "p95": math.nan, "count": 0}
    interior_t = times[left:right]
    grid_t = np.r_[start, interior_t[(interior_t > start) & (interior_t < end)], end]
    grid_v = np.interp(grid_t, times, values)
    return {
        "mean": float(np.trapz(grid_v, grid_t) / (end - start)),
        "min": float(np.min(native)),
        "max": float(np.max(native)),
        "p95": float(np.percentile(native, 95)),
        "count": int(len(native)),
    }


def _sample_gap(times, start: float, end: float) -> float:
    times = np.asarray(times, dtype=float)
    times = np.sort(times[np.isfinite(times)])
    if len(times) < 2:
        return math.nan
    left = max(0, int(np.searchsorted(times, start, side="left")) - 1)
    right = min(len(times), int(np.searchsorted(times, end, side="left")) + 1)
    before = times[left:right - 1]
    after = times[left + 1:right]
    overlap = (before < end) & (after > start)
    return (float(np.max(after[overlap] - before[overlap]))
            if np.any(overlap) else math.nan)


def _hold(times, values, when: float, default=math.nan):
    if times is None or values is None or not len(times):
        return default
    index = int(np.searchsorted(times, when, side="right")) - 1
    return values[index] if index >= 0 else default


def _duration_above(times, values, start: float, end: float,
                    threshold: float) -> float:
    times, values = _finite_time_series(times, values)
    if len(times) < 2:
        return 0.0
    left = max(0, int(np.searchsorted(times, start, side="right")) - 1)
    right = min(len(times), int(np.searchsorted(times, end, side="left")) + 1)
    t = np.clip(times[left:right], start, end)
    v = values[left:right]
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    return float(np.sum(dt[(v[:-1] > threshold) & (v[1:] > threshold)]))


def classify_descent_spike(phase, armed, actual_duration, command_count,
                           command_max, threshold):
    """Return an objective warning label without guessing controller causality."""
    if not armed or phase != "PRECLAND":
        return False, ""
    if command_count and command_max > threshold:
        return True, "command_excess"
    if actual_duration >= 0.1:
        return True, "actual_only_excess"
    return False, ""


def _groundtruth_height(groundtruth, deck_world_z):
    times = np.asarray(groundtruth["timestamp"], float) * 1e-6
    return times, -np.asarray(groundtruth["z"], float) - float(deck_world_z)


def _clock_mapper(ulog: ULog):
    sync = next((item.data for item in ulog.data_list
                 if item.name == "timesync_status" and item.multi_id == 0), None)
    if sync is None:
        return None
    local = np.asarray(sync["timestamp"], float) * 1e-6
    remote = np.asarray(sync["remote_timestamp"], float) * 1e-6
    offset = remote - local
    # The initial local==0 sample is not synchronized. Preserve subsequent
    # offset steps: a Gazebo pause is timing information, not an outlier.
    valid = np.isfinite(offset) & (remote > 1.0e8) & (local > 0.0)
    if np.count_nonzero(valid) < 3:
        return None
    local, offset = local[valid], offset[valid]
    if len(local) < 2:
        return None
    return lambda value: np.asarray(value, float) + np.interp(
        value, local, offset, left=offset[0], right=offset[-1])


def _phase_events(path: Path):
    events = []
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        first_stamp = re.search(r"\[(\d+(?:\.\d+)?)\]", text)
        if first_stamp:
            events.append((float(first_stamp.group(1)), "PRECHECK"))
        for match in _TRANSITION.finditer(text):
            timestamp, old, new = match.groups()
            events.append((float(timestamp), new))
    return events


def _planner_failure_events(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(text.count(message) for message in (
        "global A*/B-spline replan failed:",
        "global active-path SFC rejected:",
    ))


def _precision_landing_retries(path: Path) -> tuple[int, int, int]:
    """Count PX4 handoff attempts and camera-loss recoveries from transitions."""
    if not path.exists():
        return 0, 0, 0
    text = path.read_text(encoding="utf-8", errors="replace")
    transitions = [match.groups()[1:] for match in _TRANSITION.finditer(text)]
    attempts = sum(new == "PRECLAND" for _, new in transitions)
    precland_recoveries = sum(
        old == "PRECLAND" and new == "LANDING_ACQUIRE"
        for old, new in transitions)
    descend_recoveries = sum(
        old == "LANDING_DESCEND" and new == "LANDING_ACQUIRE"
        for old, new in transitions)
    return attempts, precland_recoveries, descend_recoveries


def _experiment_metrics(path: Path) -> dict[str, float]:
    """Read the final machine-readable metric line from mission-manager."""
    values: dict[str, float] = {}
    if not path.exists():
        return values
    for line in path.read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = _EXPERIMENT_METRICS.search(line)
        if not match:
            continue
        candidate = {}
        for token in match.group(1).split():
            key, separator, value = token.partition("=")
            if not separator:
                continue
            try:
                candidate[key] = float(value)
            except ValueError:
                continue
        if candidate:
            values = candidate
    return values


def _battery_energy_wh(battery, start: float, end: float) -> float:
    """Integrate measured flight-battery power; invalid current stays N/A."""
    if (battery is None or not np.isfinite(start) or not np.isfinite(end)
            or end <= start):
        return math.nan
    timestamps = np.asarray(battery.get("timestamp", []), float) * 1e-6
    voltage = np.asarray(battery.get("voltage_v", []), float)
    current = np.asarray(battery.get("current_a", []), float)
    if not (len(timestamps) == len(voltage) == len(current)):
        return math.nan
    inside = (timestamps >= start) & (timestamps <= end)
    timestamps, voltage, current = (
        values[inside] for values in (timestamps, voltage, current))
    if len(timestamps) < 2:
        return math.nan
    valid = (np.isfinite(voltage) & np.isfinite(current)
             & (voltage > 0.0) & (current >= 0.0))
    delta = np.diff(timestamps)
    pairs = (valid[:-1] & valid[1:] & (delta > 0.0) & (delta <= 1.0))
    if not np.any(pairs):
        return math.nan
    coverage = float(np.sum(delta[pairs]))
    if coverage < 0.8 * (end - start):
        return math.nan
    power = voltage * current
    return float(np.sum(
        0.5 * (power[:-1][pairs] + power[1:][pairs]) * delta[pairs]
    ) / 3600.0)


def _path_tracking_metrics(track_t, track_xy, events, clock,
                           arm_time, disarm_time):
    """RMSE/max against streamed setpoints during B-spline route phases."""
    track_t, track_xy = _finite_time_series(track_t, track_xy)
    if not len(track_t):
        return math.nan, math.nan
    mask = np.zeros(track_t.shape, dtype=bool)
    if clock is not None and events:
        mask = np.asarray([
            _at_phase(events, clock, stamp) in ("MISSION", "RETURN")
            for stamp in track_t], dtype=bool)
    if not np.any(mask):
        mask = (track_t >= arm_time if np.isfinite(arm_time)
                else np.ones(track_t.shape, dtype=bool))
        if np.isfinite(disarm_time):
            mask &= track_t <= disarm_time
    values = track_xy[mask]
    if not len(values):
        return math.nan, math.nan
    return (float(np.sqrt(np.mean(values * values))),
            float(np.max(values)))


def _minimum_aabb_residual(points, obstacles, clearance_m):
    """Return physical-AABB XY distance minus the required clearance."""
    points = np.asarray(points, float)
    centers = np.asarray([item["center_m"][:2] for item in obstacles], float)
    half_sizes = 0.5 * np.asarray(
        [item["size_m"][:2] for item in obstacles], float)
    delta = (np.abs(points[:, None, :2] - centers[None, :, :])
             - half_sizes[None, :, :])
    residual = (np.linalg.norm(np.maximum(delta, 0.0), axis=2)
                - float(clearance_m))
    sample_index, obstacle_index = np.unravel_index(
        int(np.argmin(residual)), residual.shape)
    return (float(residual[sample_index, obstacle_index]),
            int(sample_index), int(obstacle_index))


def classify_quality(fail_reasons, warn_reasons):
    fail_reasons = list(fail_reasons)
    warn_reasons = list(warn_reasons)
    quality = "FAIL" if fail_reasons else "WARN" if warn_reasons else "PASS"
    return quality, "|".join(fail_reasons + warn_reasons)


def phase_interval(events, start: float, end: float):
    if not events or not np.isfinite(start) or not np.isfinite(end):
        return "UNKNOWN", 0.0, 0
    boundaries = [start]
    boundaries.extend(t for t, _ in events if start < t < end)
    boundaries.append(end)
    durations: dict[str, float] = {}
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        state = "UNKNOWN"
        for timestamp, candidate in events:
            if timestamp <= left:
                state = candidate
            else:
                break
        durations[state] = durations.get(state, 0.0) + right - left
    state, duration = max(durations.items(), key=lambda item: item[1])
    return state, duration / (end - start), int(len(boundaries) > 2)


def _load_odometry(path: Path, spawn_xy, deck_z):
    rows = []
    if not path.exists():
        return tuple(np.array([], dtype=float) for _ in range(6))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            message = json.loads(line)
            stamp = message["header"]["stamp"]
            timestamp = float(stamp.get("sec", 0)) + float(stamp.get("nsec", 0)) * 1e-9
            position = message["pose"]["position"]
            orientation = message["pose"].get("orientation", {})
            linear = message.get("twist", {}).get("linear", {})
            qx = float(orientation.get("x", 0.0))
            qy = float(orientation.get("y", 0.0))
            qz = float(orientation.get("z", 0.0))
            qw = float(orientation.get("w", 1.0))
            yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
            body_x = float(linear.get("x", 0.0))
            body_y = float(linear.get("y", 0.0))
            world_x = math.cos(yaw) * body_x - math.sin(yaw) * body_y
            world_y = math.sin(yaw) * body_x + math.cos(yaw) * body_y
            rows.append((timestamp,
                         float(position["x"]) - spawn_xy[0],
                         float(position["y"]) - spawn_xy[1],
                         float(deck_z),
                         world_x, world_y))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not rows:
        return tuple(np.array([], dtype=float) for _ in range(6))
    columns = np.asarray(rows, dtype=float).T
    order = np.argsort(columns[0])
    return tuple(column[order] for column in columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return "" if not np.isfinite(value) else f"{float(value):.15g}"
    return value


def _at_phase(events, clock, timestamp):
    if clock is None:
        return "UNKNOWN"
    phase, _, _ = phase_interval(events, float(clock(timestamp - 1e-6)),
                                 float(clock(timestamp + 1e-6)))
    return phase


def export_run(run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir = run_dir.resolve()
    ulog_path = run_dir / "flight.ulg"
    if not ulog_path.is_file():
        raise FileNotFoundError(f"missing ULog: {ulog_path}")
    manifest = _manifest(run_dir / "manifest.tsv")
    coordinate_path = Path(manifest.get("coordinates", ""))
    if not coordinate_path.is_absolute():
        coordinate_path = run_dir / coordinate_path
    if not coordinate_path.is_file():
        raise FileNotFoundError(f"missing coordinate YAML: {coordinate_path}")
    document = yaml.safe_load(coordinate_path.read_text(encoding="utf-8"))
    spawn_pose = document["spawn"]["gazebo_spawn_pose_enu"]
    spawn_xy = np.array([float(spawn_pose["x"]), float(spawn_pose["y"])])
    mission = document["mission"]
    frame_name = mission.get("coordinate_frame")
    frame = document.get("frames", {}).get(frame_name, {})
    frame_origin = np.asarray(frame.get("origin_enu_m", [0.0, 0.0])[:2], float)
    heading = math.radians(float(frame.get("heading_deg_enu", 0.0)))
    rotation = np.array([[math.cos(heading), -math.sin(heading)],
                         [math.sin(heading), math.cos(heading)]])
    trailer = document["trailer"]
    deck_world_z = (float(trailer["spawn_pose_enu"]["z"])
                    + float(trailer["marker_surface_height_m"]))
    deck_local_z = (deck_world_z
                    - float(document["frames"]["mavros_local"]["origin_enu_m"][2]))
    land_speed = float(
        document["px4_vehicle"]["sitl_parameter_overrides"]["MPC_LAND_SPEED"])
    if not np.isfinite(land_speed) or land_speed <= 0.0:
        raise ValueError("MPC_LAND_SPEED must be finite and positive")
    descent_warn = land_speed + DESCENT_WARN_MARGIN_M_S

    wanted = ["vehicle_local_position", "vehicle_local_position_groundtruth",
              "vehicle_angular_velocity",
              "trajectory_setpoint", "vehicle_status", "vehicle_land_detected",
              "timesync_status", "battery_status"]
    ulog = ULog(str(ulog_path), message_name_filter_list=wanted)
    actual = _topic(ulog, "vehicle_local_position")
    groundtruth = _topic(ulog, "vehicle_local_position_groundtruth")
    angular_velocity = _topic(ulog, "vehicle_angular_velocity")
    setpoint = _topic(ulog, "trajectory_setpoint")
    status = _topic(ulog, "vehicle_status")
    landed = _topic(ulog, "vehicle_land_detected")
    battery = _topic(ulog, "battery_status")
    if actual is None or angular_velocity is None or status is None:
        raise RuntimeError(
            "ULog lacks vehicle_local_position, vehicle_angular_velocity, "
            "or vehicle_status")

    at = np.asarray(actual["timestamp"], float) * 1e-6
    ae, an, au = (np.asarray(actual["y"], float),
                  np.asarray(actual["x"], float), -np.asarray(actual["z"], float))
    ave, avn, avu = (np.asarray(actual["vy"], float),
                     np.asarray(actual["vx"], float), -np.asarray(actual["vz"], float))
    aae, aan, aau = (np.asarray(actual["ay"], float),
                     np.asarray(actual["ax"], float), -np.asarray(actual["az"], float))
    speed_xy = np.hypot(ave, avn)
    speed_3d = np.sqrt(ave * ave + avn * avn + avu * avu)
    accel_xy = np.hypot(aae, aan)
    accel_3d = np.sqrt(aae * aae + aan * aan + aau * aau)
    descent = np.maximum(0.0, -avu)
    speed_step_t = at[1:]
    speed_step = np.abs(np.diff(speed_3d))
    speed_step[np.diff(at) > 0.1] = np.nan
    map_xy = (np.column_stack((ae, an)) + spawn_xy - frame_origin) @ rotation
    body_rate_t = np.asarray(angular_velocity["timestamp"], float) * 1e-6
    body_rate_deg_s = np.linalg.norm(np.column_stack([
        angular_velocity[f"xyz[{axis}]"] for axis in range(3)]), axis=1
    ) * 180.0 / math.pi
    if groundtruth is None:
        gt_t = height_deck_groundtruth = np.array([], dtype=float)
    else:
        gt_t, height_deck_groundtruth = _groundtruth_height(
            groundtruth, deck_world_z)
    height_deck_estimated = au - deck_local_z

    if setpoint is None:
        st = np.array([], dtype=float)
        se = sn = su = sve = svn = svu = sp_speed = sp_descent = st
    else:
        st = np.asarray(setpoint["timestamp"], float) * 1e-6
        se, sn, su = (np.asarray(setpoint["position[1]"], float),
                      np.asarray(setpoint["position[0]"], float),
                      -np.asarray(setpoint["position[2]"], float))
        sve, svn, svu = (np.asarray(setpoint["velocity[1]"], float),
                         np.asarray(setpoint["velocity[0]"], float),
                         -np.asarray(setpoint["velocity[2]"], float))
        sp_speed = np.hypot(sve, svn)
        sp_descent = np.maximum(0.0, -svu)

    status_t = np.asarray(status["timestamp"], float) * 1e-6
    arming = np.asarray(status["arming_state"])
    armed_indices = np.flatnonzero(arming == 2)
    arm_time = float(status_t[armed_indices[0]]) if len(armed_indices) else math.nan
    disarm_after = np.flatnonzero((status_t > arm_time) & (arming != 2))
    disarm_time = float(status_t[disarm_after[0]]) if len(disarm_after) else math.nan
    land_t = (np.asarray(landed["timestamp"], float) * 1e-6
              if landed is not None else None)

    odom = _load_odometry(
        run_dir / "trailer_odometry.jsonl", spawn_xy, deck_local_z)
    tt, te, tn, tu, tve, tvn = odom
    trailer_speed = np.hypot(tve, tvn) if len(tt) else np.array([])
    if len(tt):
        target_e = np.interp(at, tt, te)
        target_n = np.interp(at, tt, tn)
        target_ve = np.interp(at, tt, tve)
        target_vn = np.interp(at, tt, tvn)
        relative_xy = np.hypot(ae - target_e, an - target_n)
        relative_speed = np.hypot(ave - target_ve, avn - target_vn)
    else:
        relative_xy = relative_speed = np.full(at.shape, np.nan)

    if len(st):
        valid_sp_position = np.isfinite(se) & np.isfinite(sn) & np.isfinite(su)
        track_t = st[valid_sp_position]
        track_xy = np.hypot(
            np.interp(track_t, at, ae) - se[valid_sp_position],
            np.interp(track_t, at, an) - sn[valid_sp_position])
        track_u = np.abs(np.interp(track_t, at, au) - su[valid_sp_position])
    else:
        track_t = track_xy = track_u = np.array([], dtype=float)

    clock = _clock_mapper(ulog)
    mission_log = run_dir / "gimbal_mission.log"
    events = _phase_events(mission_log)
    logged_metrics = _experiment_metrics(mission_log)
    parsed_attempts, parsed_precland_recoveries, parsed_descend_recoveries = (
        _precision_landing_retries(mission_log))
    abort_events = sum(state == "ABORT" for _, state in events)
    planner_failure_events = _planner_failure_events(mission_log)
    # Keep only complete one-second windows; extrapolating a partial first or
    # last bin would bias its mean while pretending it had full coverage.
    start = math.ceil(float(at[0]))
    end = math.floor(float(at[-1]))
    run_id = manifest.get("run_id", run_dir.name)
    rows = []
    for index, left in enumerate(np.arange(start, end, 1.0)):
        right = left + 1.0
        midpoint = left + 0.5
        ros_left = float(clock(left)) if clock is not None else math.nan
        ros_right = float(clock(right)) if clock is not None else math.nan
        phase, phase_fraction, phase_transition = phase_interval(
            events, ros_left, ros_right)
        phase_events = "|".join(
            state for timestamp, state in events
            if ros_left <= timestamp < ros_right)
        actual_count = int(np.count_nonzero((at >= left) & (at < right)))
        setpoint_count = int(np.count_nonzero((st >= left) & (st < right)))
        odom_count = int(np.count_nonzero((tt >= left) & (tt < right)))
        armed = int(_hold(status_t, arming, midpoint, 1) == 2)
        actual_down = interval_stats(at, descent, left, right)
        command_down = interval_stats(st, sp_descent, left, right)
        actual_down_duration = _duration_above(
            at, descent, left, right, descent_warn)
        descent_flag, spike_class = classify_descent_spike(
            phase, armed, actual_down_duration, command_down["count"],
            command_down["max"], descent_warn)
        speed_step_stats = interval_stats(speed_step_t, speed_step, left, right)
        accel_stats = interval_stats(at, accel_3d, left, right)
        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "bin_index": index,
            "t_sim_start_s": left,
            "t_sim_end_s": right,
            "t_from_arm_s": midpoint - arm_time,
            "ros_time_est_s": (ros_left + ros_right) * 0.5,
            "mission_state": phase,
            "phase_fraction": phase_fraction,
            "phase_transition": phase_transition,
            "phase_events": phase_events,
            "armed": armed,
            "nav_state": _hold(status_t, status["nav_state"], midpoint, ""),
            "failsafe": int(bool(_hold(status_t, status["failsafe"], midpoint, 0))),
            "preflight_ok": int(bool(_hold(
                status_t, status.get("pre_flight_checks_pass"), midpoint, 0))),
            "landed": int(bool(_hold(
                land_t, landed["landed"] if landed is not None else None,
                midpoint, 0))),
            "actual_samples": actual_count,
            "actual_max_gap_s": _sample_gap(at, left, right),
            "x_local_enu_m": interval_stats(at, ae, left, right)["mean"],
            "y_local_enu_m": interval_stats(at, an, left, right)["mean"],
            "z_local_enu_m": interval_stats(at, au, left, right)["mean"],
            "map_x_m": interval_stats(at, map_xy[:, 0], left, right)["mean"],
            "map_y_m": interval_stats(at, map_xy[:, 1], left, right)["mean"],
            "vx_enu_m_s": interval_stats(at, ave, left, right)["mean"],
            "vy_enu_m_s": interval_stats(at, avn, left, right)["mean"],
            "vz_enu_m_s": interval_stats(at, avu, left, right)["mean"],
            "speed_xy_mean_m_s": interval_stats(at, speed_xy, left, right)["mean"],
            "speed_xy_p95_m_s": interval_stats(at, speed_xy, left, right)["p95"],
            "speed_xy_max_m_s": interval_stats(at, speed_xy, left, right)["max"],
            "speed_3d_mean_m_s": interval_stats(at, speed_3d, left, right)["mean"],
            "speed_3d_max_m_s": interval_stats(at, speed_3d, left, right)["max"],
            "descent_rate_mean_m_s": actual_down["mean"],
            "descent_rate_max_m_s": actual_down["max"],
            "accel_xy_max_m_s2": interval_stats(at, accel_xy, left, right)["max"],
            "accel_3d_max_m_s2": accel_stats["max"],
            "speed_step_max_m_s": speed_step_stats["max"],
            "setpoint_samples": setpoint_count,
            "setpoint_max_gap_s": _sample_gap(st, left, right),
            "sp_x_enu_m": interval_stats(st, se, left, right)["mean"],
            "sp_y_enu_m": interval_stats(st, sn, left, right)["mean"],
            "sp_z_enu_m": interval_stats(st, su, left, right)["mean"],
            "sp_vx_enu_m_s": interval_stats(st, sve, left, right)["mean"],
            "sp_vy_enu_m_s": interval_stats(st, svn, left, right)["mean"],
            "sp_vz_enu_m_s": interval_stats(st, svu, left, right)["mean"],
            "sp_speed_xy_max_m_s": interval_stats(st, sp_speed, left, right)["max"],
            "sp_descent_rate_max_m_s": command_down["max"],
            "xy_tracking_error_mean_m": interval_stats(
                track_t, track_xy, left, right)["mean"],
            "xy_tracking_error_max_m": interval_stats(
                track_t, track_xy, left, right)["max"],
            "z_tracking_error_mean_m": interval_stats(
                track_t, track_u, left, right)["mean"],
            "z_tracking_error_max_m": interval_stats(
                track_t, track_u, left, right)["max"],
            "trailer_samples": odom_count,
            "trailer_max_gap_s": _sample_gap(tt, left, right),
            "trailer_x_local_enu_m": interval_stats(tt, te, left, right)["mean"],
            "trailer_y_local_enu_m": interval_stats(tt, tn, left, right)["mean"],
            "trailer_z_local_enu_m": interval_stats(tt, tu, left, right)["mean"],
            "trailer_vx_enu_m_s": interval_stats(tt, tve, left, right)["mean"],
            "trailer_vy_enu_m_s": interval_stats(tt, tvn, left, right)["mean"],
            "trailer_speed_max_m_s": interval_stats(
                tt, trailer_speed, left, right)["max"],
            "relative_xy_mean_m": interval_stats(
                at, relative_xy, left, right)["mean"],
            "relative_xy_max_m": interval_stats(
                at, relative_xy, left, right)["max"],
            "relative_speed_xy_mean_m_s": interval_stats(
                at, relative_speed, left, right)["mean"],
            "relative_speed_xy_max_m_s": interval_stats(
                at, relative_speed, left, right)["max"],
            "height_above_deck_groundtruth_mean_m": interval_stats(
                gt_t, height_deck_groundtruth, left, right)["mean"],
            "height_above_deck_groundtruth_min_m": interval_stats(
                gt_t, height_deck_groundtruth, left, right)["min"],
            "height_above_deck_estimated_mean_m": interval_stats(
                at, height_deck_estimated, left, right)["mean"],
            "height_above_deck_estimated_min_m": interval_stats(
                at, height_deck_estimated, left, right)["min"],
            "speed_jump_flag": int(
                armed and speed_step_stats["max"] > SPEED_STEP_LIMIT_M_S),
            "accel_spike_flag": int(
                armed and accel_stats["max"] > ACCEL_SPIKE_LIMIT_M_S2),
            "descent_spike_flag": int(descent_flag),
            "descent_spike_class": spike_class,
            "descent_excess_duration_s": actual_down_duration,
        }
        rows.append({key: _clean(value) for key, value in row.items()})

    csv_path = run_dir / "flight_1hz.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if np.isfinite(arm_time):
        armed_mask = ((at >= arm_time) & (at <= disarm_time)
                      if np.isfinite(disarm_time) else at >= arm_time)
    else:
        armed_mask = np.ones(at.shape, dtype=bool)
    armed_indices_actual = np.flatnonzero(armed_mask)
    max_speed_i = armed_indices_actual[np.nanargmax(speed_xy[armed_mask])]
    max_accel_i = armed_indices_actual[np.nanargmax(accel_xy[armed_mask])]
    max_down_i = armed_indices_actual[np.nanargmax(descent[armed_mask])]
    valid_step = ((speed_step_t >= arm_time)
                  & (speed_step_t <= disarm_time
                     if np.isfinite(disarm_time) else True)
                  if np.isfinite(arm_time)
                  else np.ones(speed_step_t.shape, dtype=bool))
    max_step_i = np.flatnonzero(valid_step)[np.nanargmax(speed_step[valid_step])]
    setpoint_armed = ((st >= arm_time)
                      & (st <= disarm_time if np.isfinite(disarm_time) else True)
                      if np.isfinite(arm_time)
                      else np.ones(st.shape, dtype=bool))
    max_sp_down = float(np.nanmax(sp_descent[setpoint_armed])) if np.any(setpoint_armed) else math.nan
    status_analysis = (status_t >= arm_time if np.isfinite(arm_time)
                       else np.ones(status_t.shape, dtype=bool))
    failsafe_seen = int(bool(np.any(
        np.asarray(status["failsafe"])[status_analysis])))
    body_rate_armed = ((body_rate_t >= arm_time)
                       & (body_rate_t <= disarm_time
                          if np.isfinite(disarm_time) else True)
                       if np.isfinite(arm_time)
                       else np.ones(body_rate_t.shape, dtype=bool))
    body_rate_indices = np.flatnonzero(body_rate_armed)
    max_body_rate_i = body_rate_indices[
        np.nanargmax(body_rate_deg_s[body_rate_armed])]
    # New runs own clearance on the vehicle. Keep reading old map snapshots so
    # historical artifacts remain exportable.
    clearance = float(mission.get(
        "vehicle_clearance_xy_m", mission.get("obstacle_clearance_m")))
    armed_map_xy = map_xy[armed_indices_actual]
    physical_distance, _, _ = _minimum_aabb_residual(
        armed_map_xy, mission["obstacles"], 0.0)
    clearance_residual, clearance_sample_i, clearance_obstacle_i = (
        _minimum_aabb_residual(
            armed_map_xy, mission["obstacles"], clearance))
    clearance_actual_i = armed_indices_actual[clearance_sample_i]
    obstacle_name = mission["obstacles"][clearance_obstacle_i]["name"]
    actual_max_gap = float(np.nanmax(np.diff(at)))
    speed_jump_bins = sum(int(row["speed_jump_flag"]) for row in rows)
    accel_spike_bins = sum(int(row["accel_spike_flag"]) for row in rows)
    descent_spike_bins = sum(int(row["descent_spike_flag"]) for row in rows)
    ulog_dropouts = len(ulog.dropouts)
    paper_reproducible = int(manifest.get("git_dirty") == "0")
    result = manifest.get("result", "unknown")
    route_rmse, route_max = _path_tracking_metrics(
        track_t, track_xy, events, clock, arm_time, disarm_time)
    landed_seen = int(bool(
        landed is not None and np.any(
            np.asarray(landed["landed"], bool)[
                (land_t >= arm_time) if np.isfinite(arm_time)
                else np.ones(land_t.shape, dtype=bool)])))
    done_seen = result == "done" or any(
        state == "DONE" for _, state in events)
    mission_success = int(
        done_seen and landed_seen and np.isfinite(disarm_time)
        and not failsafe_seen and not abort_events)
    precland_attempts = int(logged_metrics.get(
        "precland_attempts", parsed_attempts))
    precland_recoveries = int(logged_metrics.get(
        "precland_recoveries", parsed_precland_recoveries))
    landing_descend_recoveries = int(logged_metrics.get(
        "landing_descend_recoveries", parsed_descend_recoveries))
    precision_landing_first_try = int(
        mission_success and precland_attempts == 1
        and precland_recoveries == 0
        and landing_descend_recoveries == 0)
    marker_frames = logged_metrics.get("marker_frames", 0.0)
    marker_rate = (100.0 * logged_metrics.get("marker_hits", 0.0)
                   / marker_frames if marker_frames > 0.0 else math.nan)
    mpc_count = logged_metrics.get("mpc_count", 0.0)
    mpc_mean_ms = (logged_metrics.get("mpc_total_ms", math.nan) / mpc_count
                   if mpc_count > 0.0 else math.nan)
    battery_energy_wh = _battery_energy_wh(
        battery, arm_time, disarm_time)
    fail_reasons = []
    warn_reasons = []
    if result != "done":
        fail_reasons.append("result_not_done")
    if failsafe_seen:
        fail_reasons.append("failsafe")
    if actual_max_gap > 0.1:
        fail_reasons.append("actual_sample_gap")
    if speed_jump_bins:
        fail_reasons.append("speed_jump")
    if ulog_dropouts:
        fail_reasons.append("ulog_dropout")
    if planner_failure_events:
        fail_reasons.append("planner_failure")
    if abort_events:
        fail_reasons.append("mission_abort")
    if clearance_residual <= 0.0:
        fail_reasons.append("obstacle_clearance_violation")
    if accel_spike_bins:
        warn_reasons.append("accel_spike")
    if body_rate_deg_s[max_body_rate_i] > BODY_RATE_WARN_DEG_S:
        warn_reasons.append("body_rate")
    if 0.0 < clearance_residual < OBSTACLE_RESERVE_WARN_M:
        warn_reasons.append("low_obstacle_reserve")
    if descent_spike_bins:
        warn_reasons.append("descent_spike")
    if mission_success and not precision_landing_first_try:
        warn_reasons.append("precision_landing_retry")
    if not paper_reproducible:
        warn_reasons.append("dirty_tree")
    quality, quality_reasons = classify_quality(fail_reasons, warn_reasons)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "result": result,
        "csv_rows": len(rows),
        "arm_time_sim_s": arm_time,
        "disarm_time_sim_s": disarm_time,
        "armed_duration_s": disarm_time - arm_time,
        "max_speed_xy_m_s": speed_xy[max_speed_i],
        "max_speed_xy_time_sim_s": at[max_speed_i],
        "max_speed_xy_phase": _at_phase(events, clock, at[max_speed_i]),
        "max_speed_step_m_s": speed_step[max_step_i],
        "max_speed_step_time_sim_s": speed_step_t[max_step_i],
        "max_speed_step_phase": _at_phase(events, clock, speed_step_t[max_step_i]),
        "max_accel_xy_m_s2": accel_xy[max_accel_i],
        "max_accel_xy_time_sim_s": at[max_accel_i],
        "max_accel_xy_phase": _at_phase(events, clock, at[max_accel_i]),
        "max_descent_rate_m_s": descent[max_down_i],
        "max_descent_time_sim_s": at[max_down_i],
        "max_descent_phase": _at_phase(events, clock, at[max_down_i]),
        "max_commanded_descent_rate_m_s": max_sp_down,
        "speed_jump_bins": speed_jump_bins,
        "accel_spike_bins": accel_spike_bins,
        "descent_spike_bins": descent_spike_bins,
        "speed_step_limit_m_s": SPEED_STEP_LIMIT_M_S,
        "accel_spike_limit_m_s2": ACCEL_SPIKE_LIMIT_M_S2,
        "max_body_rate_deg_s": body_rate_deg_s[max_body_rate_i],
        "max_body_rate_time_sim_s": body_rate_t[max_body_rate_i],
        "max_body_rate_phase": _at_phase(
            events, clock, body_rate_t[max_body_rate_i]),
        "body_rate_warn_deg_s": BODY_RATE_WARN_DEG_S,
        "min_physical_obstacle_distance_m": physical_distance,
        "min_obstacle_clearance_residual_m": clearance_residual,
        "min_obstacle_clearance_time_sim_s": at[clearance_actual_i],
        "min_obstacle_clearance_phase": _at_phase(
            events, clock, at[clearance_actual_i]),
        "closest_obstacle": obstacle_name,
        "vehicle_clearance_xy_m": clearance,
        # Compatibility alias for existing post-processing notebooks. The
        # source-of-truth YAML now owns this radius on the vehicle.
        "obstacle_clearance_m": clearance,
        "obstacle_reserve_warn_m": OBSTACLE_RESERVE_WARN_M,
        "planner_failure_events": planner_failure_events,
        "abort_events": abort_events,
        "precland_attempts": precland_attempts,
        "precland_recoveries": precland_recoveries,
        "landing_descend_recoveries": landing_descend_recoveries,
        "precision_landing_first_try": precision_landing_first_try,
        "mpc_land_speed_m_s": land_speed,
        "descent_warn_m_s": descent_warn,
        "failsafe_seen": failsafe_seen,
        "ulog_dropouts": ulog_dropouts,
        "actual_max_sample_gap_s": actual_max_gap,
        "paper_reproducible": paper_reproducible,
        "quality": quality,
        "quality_reasons": quality_reasons,
        "flight_ulg_sha256": _sha256(ulog_path),
        "coordinate_yaml_sha256": _sha256(coordinate_path),
        "flight_1hz_sha256": "filled_after_write",
    }
    summary["flight_1hz_sha256"] = _sha256(csv_tmp)
    summary_path = run_dir / "flight_summary.csv"
    summary_tmp = summary_path.with_suffix(".csv.tmp")
    with summary_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow({key: _clean(value) for key, value in summary.items()})
    experiment_metrics = {
        "mission_success_rate_pct": 100.0 * mission_success,
        "path_tracking_rmse_m": route_rmse,
        "path_tracking_error_max_m": route_max,
        "marker_detection_rate_pct": marker_rate,
        "landing_error_3d_m": logged_metrics.get(
            "landing_error_3d_m", math.nan),
        "landing_xy_error_m": logged_metrics.get(
            "landing_xy_error_m", math.nan),
        "touchdown_relative_speed_3d_m_s": logged_metrics.get(
            "touchdown_relative_speed_3d_m_s", math.nan),
        "touchdown_relative_vertical_speed_m_s": logged_metrics.get(
            "touchdown_relative_vertical_speed_m_s", math.nan),
        "mpc_solve_mean_ms": mpc_mean_ms,
        "mpc_solve_max_ms": logged_metrics.get("mpc_max_ms", math.nan),
        "flight_battery_energy_wh": battery_energy_wh,
    }
    metrics_path = run_dir / "experiment_metrics.csv"
    metrics_tmp = metrics_path.with_suffix(".csv.tmp")
    with metrics_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=EXPERIMENT_METRIC_FIELDS)
        writer.writeheader()
        writer.writerow({
            key: _clean(value) for key, value in experiment_metrics.items()})
    os.replace(csv_tmp, csv_path)
    os.replace(summary_tmp, summary_path)
    os.replace(metrics_tmp, metrics_path)
    return csv_path, summary_path, metrics_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run artifact directory")
    args = parser.parse_args(argv)
    try:
        csv_path, summary_path, metrics_path = export_run(args.run_dir)
    except Exception as error:  # keep the shell's original flight result intact
        for name in ("flight_1hz.csv.tmp", "flight_summary.csv.tmp",
                     "experiment_metrics.csv.tmp"):
            (args.run_dir / name).unlink(missing_ok=True)
        print(f"flight CSV export failed: {error}", file=sys.stderr)
        return 1
    print(f"1 Hz flight CSV: {csv_path}")
    print(f"flight summary : {summary_path}")
    print(f"experiment data: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
