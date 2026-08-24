#!/usr/bin/env python3
"""Build a paper-ready city-YAML dynamic-planning data package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pursuit_sim as pursuit


TITLE = "Gazebo filght planned"
OFFLINE_SCOPE = "City-map YAML offline rollout — not Gazebo/PX4 telemetry"
DEFAULT_OUTPUT = Path.home() / "Gazebo_filght_planned_paper"
DEFAULT_ACTUAL = (
    Path.home() / ".local/state/px4-ros2-jo/city"
    / "full_mission_goal_m165_0_drone12_trailer7_5x_biaslatch_r15"
)
COLORS = {
    "obstacle": "#c3c9cf",
    "astar": "#f59f00",
    "bspline": "#2f9e44",
    "mpc": "#ae3ec9",
    "final": "#1971c2",
    "trailer": "#e8590c",
    "limit": "#c92a2a",
    "loop": "#868e96",
}
SFC_DEGENERATE_TOL_M = 1.0e-9


def _array(rows, key, *, default=math.nan):
    values = []
    for row in rows:
        value = row.get(key, default)
        try:
            values.append(float(value) if value not in (None, "") else default)
        except (TypeError, ValueError):
            values.append(default)
    return np.asarray(values, float)


def _path_length(points):
    points = np.asarray(points, float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def _xy_clearances(world, points):
    """Physical XY distance to the nearest AABB intersecting each point's z."""
    clearances = []
    for point in np.asarray(points, float):
        vertical = ((point[2] >= world.boxes_min[:, 2])
                    & (point[2] <= world.boxes_max[:, 2]))
        if not np.any(vertical):
            clearances.append(math.inf)
            continue
        low = world.boxes_min[vertical, :2]
        high = world.boxes_max[vertical, :2]
        gap = (np.maximum(low - point[:2], 0.0)
               + np.maximum(point[:2] - high, 0.0))
        clearances.append(float(np.min(np.linalg.norm(gap, axis=1))))
    return np.asarray(clearances, float)


def _indices(count, wanted):
    if count <= 0:
        return []
    return np.unique(np.linspace(0, count - 1, min(count, wanted)).astype(int)).tolist()


def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer empty CSV schema for {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_one_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one result row in {path}")
    return rows[0]


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _loop(trailer):
    distance = np.linspace(0.0, 8.0 * trailer.half, 500)
    return np.asarray([pursuit.square_loop_pos(value, trailer.half)
                       for value in distance])


def _map_background(ax, world, trailer, *, route=True):
    for low, high in zip(world.boxes_min, world.boxes_max):
        ax.add_patch(Rectangle(
            low[:2], *(high[:2] - low[:2]), facecolor=COLORS["obstacle"],
            edgecolor="none", alpha=0.48, zorder=0))
    if route:
        points = _loop(trailer)
        ax.plot(points[:, 0], points[:, 1], ":", color=COLORS["loop"],
                linewidth=0.7, alpha=0.75, zorder=1)
    ax.set_xlim(float(world.bounds_min[0]) - 5.0,
                float(world.bounds_max[0]) + 5.0)
    ax.set_ylim(float(world.bounds_min[1]) - 5.0,
                float(world.bounds_max[1]) + 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("East x [m]")
    ax.set_ylabel("North y [m]")
    ax.grid(alpha=0.12, linewidth=0.5)


def _legend(ax, names):
    styles = {
        "A*": dict(color=COLORS["astar"], linestyle="-", marker="o"),
        "B-spline": dict(color=COLORS["bspline"], linestyle="-"),
        "MPC": dict(color=COLORS["mpc"], linestyle="--"),
        "Final path": dict(color=COLORS["final"], linestyle="-"),
        "Trailer": dict(color=COLORS["trailer"], linestyle="-"),
    }
    handles = [Line2D([0], [0], linewidth=1.4, markersize=3,
                      label=name, **styles[name]) for name in names]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.92, borderpad=0.5)


def _offline_arrays(log):
    return {
        key: _array(log, key) for key in (
            "t_s", "drone_x", "drone_y", "drone_z", "drone_speed_mps",
            "drone_accel_mps2", "drone_yawrate_rads", "trailer_x",
            "trailer_y", "trailer_speed_mps", "dist_xy_m", "replan_count",
            "track_err_m",
            "track_err_dense_m", "mpc_solve_ms", "mpc_success",
            "mpc_cte_m", "mpc_epsi_rad")
    }


def _draw_selected_astar(ax, splines, selected, *, alpha=0.72):
    for index in selected:
        points = splines[index][3]
        ax.plot(points[:, 0], points[:, 1], "-o", color=COLORS["astar"],
                linewidth=0.75, markersize=1.8, alpha=alpha, zorder=4)


def _draw_selected_bspline(ax, splines, selected, *, alpha=0.72):
    for index in selected:
        points = splines[index][1]
        ax.plot(points[:, 0], points[:, 1], color=COLORS["bspline"],
                linewidth=0.9, alpha=alpha, zorder=5)


def _draw_mpc(ax, horizons, wanted=70, *, alpha=0.55):
    for index in _indices(len(horizons), wanted):
        points = np.asarray(horizons[index], float)
        ax.plot(points[:, 0], points[:, 1], "--", color=COLORS["mpc"],
                linewidth=0.75, alpha=alpha, zorder=6)


def _figure_pipeline_panels(figures, arrays, world, trailer, splines, horizons):
    selected = _indices(len(splines), 9)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for ax in axes.ravel():
        _map_background(ax, world, trailer)

    _draw_selected_astar(axes[0, 0], splines, selected)
    axes[0, 0].set_title("A* dynamic replans")
    _legend(axes[0, 0], ["A*"])

    _draw_selected_bspline(axes[0, 1], splines, selected)
    axes[0, 1].set_title("Geometry-only B-spline")
    _legend(axes[0, 1], ["B-spline"])

    _draw_mpc(axes[1, 0], horizons, 90, alpha=0.72)
    axes[1, 0].set_title("MPC receding horizons (dashed)")
    _legend(axes[1, 0], ["MPC"])

    axes[1, 1].plot(arrays["drone_x"], arrays["drone_y"],
                    color=COLORS["final"], linewidth=1.1, zorder=4)
    axes[1, 1].plot(arrays["trailer_x"], arrays["trailer_y"],
                    color=COLORS["trailer"], linewidth=0.8, zorder=3)
    axes[1, 1].set_title("Final executed path")
    _legend(axes[1, 1], ["Final path", "Trailer"])

    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nFour separated path products — "
        f"9 representative replans of {len(splines)} accepted", fontsize=15)
    _save(fig, figures / "01_pipeline_four_panels.png")


def _figure_overlay(figures, arrays, world, trailer, splines, horizons):
    selected = _indices(len(splines), 7)
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    _map_background(ax, world, trailer)
    _draw_selected_astar(ax, splines, selected, alpha=0.34)
    _draw_selected_bspline(ax, splines, selected, alpha=0.48)
    _draw_mpc(ax, horizons, 70, alpha=0.5)
    ax.plot(arrays["drone_x"], arrays["drone_y"], color=COLORS["final"],
            linewidth=1.0, alpha=0.82, zorder=3)
    ax.plot(arrays["trailer_x"], arrays["trailer_y"],
            color=COLORS["trailer"], linewidth=0.8, zorder=2)
    ax.set_title(
        f"{TITLE}\n{OFFLINE_SCOPE}\nThin-line overlay of all four path products")
    _legend(ax, ["A*", "B-spline", "MPC", "Final path", "Trailer"])
    _save(fig, figures / "02_pipeline_overlay_four_paths.png")


def _figure_single_paths(figures, arrays, world, trailer, splines, horizons):
    selected = _indices(len(splines), 12)
    specs = (
        ("03_astar_dynamic_paths.png", "A* dynamic replans", "A*"),
        ("04_bspline_dynamic_paths.png", "B-spline dynamic replans", "B-spline"),
        ("05_mpc_horizon_paths.png", "MPC receding-horizon predictions", "MPC"),
        ("06_final_executed_path.png", "Final executed path", "Final path"),
    )
    for filename, title, kind in specs:
        fig, ax = plt.subplots(figsize=(10.5, 9.2), constrained_layout=True)
        _map_background(ax, world, trailer)
        if kind == "A*":
            _draw_selected_astar(ax, splines, selected, alpha=0.78)
        elif kind == "B-spline":
            _draw_selected_bspline(ax, splines, selected, alpha=0.78)
        elif kind == "MPC":
            _draw_mpc(ax, horizons, 110, alpha=0.72)
            ax.plot(arrays["drone_x"], arrays["drone_y"], color="#adb5bd",
                    linewidth=0.55, alpha=0.7)
        else:
            ax.plot(arrays["drone_x"], arrays["drone_y"],
                    color=COLORS["final"], linewidth=1.25, zorder=6)
            ax.plot(arrays["trailer_x"], arrays["trailer_y"],
                    color=COLORS["trailer"], linewidth=0.85, zorder=5)
        ax.set_title(f"{TITLE}\n{OFFLINE_SCOPE}\n{title}")
        _legend(ax, [kind] + (["Trailer"] if kind == "Final path" else []))
        _save(fig, figures / filename)


def _figure_mpc_snapshots(figures, arrays, world, trailer, horizons, mpc_times):
    selected = _indices(len(horizons), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, index in zip(axes.ravel(), selected):
        _map_background(ax, world, trailer, route=False)
        horizon = np.asarray(horizons[index], float)
        time_s = float(mpc_times[index])
        history = (arrays["t_s"] >= max(0.0, time_s - 8.0)) & (
            arrays["t_s"] <= time_s + 2.0)
        ax.plot(arrays["drone_x"][history], arrays["drone_y"][history],
                color="#adb5bd", linewidth=0.7, label="executed context")
        ax.plot(horizon[:, 0], horizon[:, 1], "--", color=COLORS["mpc"],
                linewidth=1.1, label="MPC horizon", zorder=6)
        ax.scatter(horizon[0, 0], horizon[0, 1], s=18,
                   color=COLORS["mpc"], zorder=7)
        ax.scatter(horizon[-1, 0], horizon[-1, 1], s=24, marker="x",
                   color=COLORS["limit"], zorder=7)
        center = np.mean(horizon, axis=0)
        span = max(35.0, 0.65 * float(np.ptp(horizon, axis=0).max()) + 20.0)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_title(f"MPC solve at t={time_s:.1f} s", fontsize=9)
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nSix local receding-horizon snapshots",
        fontsize=14)
    _save(fig, figures / "18_mpc_local_horizon_snapshots.png")


def _figure_replan_snapshots(figures, arrays, world, trailer, splines):
    selected = _indices(len(splines), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, plan_index in zip(axes.ravel(), selected):
        _map_background(ax, world, trailer)
        time_s, spline, _corridor, astar = splines[plan_index]
        history = arrays["t_s"] <= float(time_s)
        ax.plot(astar[:, 0], astar[:, 1], "-o", color=COLORS["astar"],
                linewidth=0.7, markersize=1.8, zorder=5)
        ax.plot(spline[:, 0], spline[:, 1], color=COLORS["bspline"],
                linewidth=0.85, alpha=0.75, zorder=4)
        ax.plot(arrays["drone_x"][history], arrays["drone_y"][history],
                color=COLORS["final"], linewidth=0.8)
        ax.set_title(f"Replan {plan_index + 1} at t={time_s:.1f} s", fontsize=9)
    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nDynamic route evolution: A* (orange), "
        "B-spline (green), executed history (blue)", fontsize=14)
    _save(fig, figures / "07_dynamic_replan_snapshots.png")


def _figure_distance_replans(figures, arrays):
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    top.plot(arrays["t_s"], arrays["dist_xy_m"], color=COLORS["final"],
             linewidth=1.2)
    top.axhline(5.0, color=COLORS["limit"], linestyle="--", linewidth=0.9)
    top.set_ylabel("Drone–trailer range [m]")
    top.grid(alpha=0.2)
    inset = top.inset_axes([0.58, 0.42, 0.39, 0.5])
    terminal = arrays["t_s"] >= arrays["t_s"][-1] - 25.0
    inset.plot(arrays["t_s"][terminal], arrays["dist_xy_m"][terminal],
               color=COLORS["final"], linewidth=0.9)
    inset.axhline(5.0, color=COLORS["limit"], linestyle="--", linewidth=0.7)
    inset.text(0.98, 0.08, "5 m capture", transform=inset.transAxes,
               ha="right", va="bottom", color=COLORS["limit"], fontsize=7)
    inset.set_title("Terminal 25 s", fontsize=8)
    inset.set_ylim(0.0, max(65.0, float(np.max(arrays["dist_xy_m"][terminal]))))
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.16)
    bottom.step(arrays["t_s"], arrays["replan_count"], where="post",
                color=COLORS["astar"], linewidth=1.1)
    bottom.set_xlabel("Simulation time [s]")
    bottom.set_ylabel("Accepted replans")
    bottom.grid(alpha=0.2)
    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nTarget closure and dynamic replan accumulation")
    _save(fig, figures / "08_range_and_replans.png")


def _figure_dynamics(figures, arrays, scenario):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    axes[0].plot(arrays["t_s"], arrays["drone_speed_mps"],
                 color=COLORS["final"], linewidth=1.0)
    drone_limit = float(scenario["drone"]["max_speed_m_s"])
    trailer_speed = float(scenario["trailer"]["speed_m_s"])
    axes[0].axhline(drone_limit, color=COLORS["limit"], linestyle="--",
                    linewidth=0.9,
                    label=f"drone speed ceiling {drone_limit:g} m/s")
    axes[0].axhline(trailer_speed, color=COLORS["trailer"], linestyle=":",
                    linewidth=0.9,
                    label=f"trailer speed {trailer_speed:g} m/s")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(arrays["t_s"], np.abs(arrays["drone_accel_mps2"]),
                 color=COLORS["astar"], linewidth=0.9)
    axes[1].axhline(3.0, color=COLORS["limit"], linestyle="--",
                    linewidth=0.9, label="acceleration limit 3 m/s²")
    axes[1].set_ylabel("|Acceleration| [m/s²]")
    axes[1].legend(fontsize=8)
    axes[2].plot(arrays["t_s"], arrays["drone_z"],
                 color=COLORS["bspline"], linewidth=1.0)
    axes[2].axhline(10.0, color="#343a40", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Altitude [m]")
    axes[2].set_xlabel("Simulation time [s]")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nClosed-loop speed, acceleration and altitude")
    _save(fig, figures / "09_speed_acceleration_altitude.png")


def _figure_error(figures, log, arrays):
    pursuit_mask = np.asarray([row["phase"] == "pursuit" for row in log])
    time_s = arrays["t_s"][pursuit_mask]
    error = arrays["track_err_dense_m"][pursuit_mask]
    finite = np.isfinite(error)
    time_s, error = time_s[finite], error[finite]
    if not len(error):
        raise RuntimeError("no finite pursuit-phase tracking errors")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.8),
                                      constrained_layout=True)
    left.plot(time_s, error, color=COLORS["bspline"], linewidth=0.9)
    left.axhline(float(np.mean(error)), color="#343a40", linestyle="--",
                 linewidth=0.8, label=f"mean {np.mean(error):.3f} m")
    left.set_xlabel("Simulation time [s]")
    left.set_ylabel("Dense B-spline tracking error [m]")
    left.legend(fontsize=8)
    left.grid(alpha=0.2)
    right.hist(error, bins=35, color=COLORS["mpc"], alpha=0.78,
               edgecolor="white")
    right.axvline(float(np.percentile(error, 95)), color=COLORS["limit"],
                  linestyle="--", linewidth=1.0,
                  label=f"p95 {np.percentile(error, 95):.3f} m")
    right.set_xlabel("Tracking error [m]")
    right.set_ylabel("Samples")
    right.legend(fontsize=8)
    right.grid(alpha=0.15)
    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nTracking error (pursuit phase only)")
    _save(fig, figures / "10_tracking_error_analysis.png")


def _replan_rows(splines):
    rows = []
    for index, (time_s, spline, corridor, astar) in enumerate(splines, 1):
        astar_length = _path_length(astar)
        spline_length = _path_length(spline)
        reduction = astar_length - spline_length
        extents = np.asarray(corridor.boxes_max - corridor.boxes_min, float)
        horizontal_widths = np.minimum(extents[:, 0], extents[:, 1])
        non_degenerate = horizontal_widths > SFC_DEGENERATE_TOL_M
        non_degenerate_min = (float(np.min(
            horizontal_widths[non_degenerate]))
            if np.any(non_degenerate) else math.nan)
        rows.append({
            "replan_index": index,
            "simulation_time_s": round(float(time_s), 3),
            "astar_points": len(astar),
            "astar_length_m": round(astar_length, 6),
            "bspline_samples": len(spline),
            "bspline_length_m": round(spline_length, 6),
            "length_reduction_m": round(reduction, 6),
            "length_reduction_pct": round(
                100.0 * reduction / astar_length if astar_length else 0.0, 6),
            "sfc_boxes": len(corridor),
            "sfc_corridor_count": len(corridor),
            "sfc_min_width_m": round(float(np.min(horizontal_widths)), 6),
            "sfc_avg_width_m": round(float(np.mean(horizontal_widths)), 6),
            "sfc_degenerate_box_count": int(np.sum(~non_degenerate)),
            "sfc_non_degenerate_min_width_m": round(
                non_degenerate_min, 6),
            "target_x_m": round(float(spline[-1, 0]), 6),
            "target_y_m": round(float(spline[-1, 1]), 6),
        })
    return rows


def _sfc_box_rows(splines):
    """Raw optimizer control-point boxes used by every accepted B-spline."""
    rows = []
    for plan_index, (time_s, _spline, corridor, _astar) in enumerate(
            splines, 1):
        for box_index, (low, high) in enumerate(
                zip(corridor.boxes_min, corridor.boxes_max), 1):
            extent = np.asarray(high, float) - np.asarray(low, float)
            horizontal_width = float(min(extent[0], extent[1]))
            rows.append({
                "dataset": "offline_yaml",
                "plan_index": plan_index,
                "simulation_time_s": round(float(time_s), 6),
                "box_index": box_index,
                "min_x_m": round(float(low[0]), 6),
                "min_y_m": round(float(low[1]), 6),
                "min_z_m": round(float(low[2]), 6),
                "max_x_m": round(float(high[0]), 6),
                "max_y_m": round(float(high[1]), 6),
                "max_z_m": round(float(high[2]), 6),
                "extent_x_m": round(float(extent[0]), 6),
                "extent_y_m": round(float(extent[1]), 6),
                "extent_z_m": round(float(extent[2]), 6),
                "horizontal_width_m": round(horizontal_width, 6),
                "is_degenerate": int(
                    horizontal_width <= SFC_DEGENERATE_TOL_M),
                "volume_m3": round(float(np.prod(extent)), 6),
                "scope": "optimizer_control_point_sfc",
            })
    return rows


def _path_point_rows(log, splines, horizons, mpc_times):
    """Long-form path data so every plotted layer can be re-rendered."""
    rows = []

    control_dt = float(log[1]["t_s"]) - float(log[0]["t_s"])

    def append(layer, plan_index, simulation_time_s, linked_timeseries_time_s,
               time_semantics, point_index, point, color, line_style):
        rows.append({
            "dataset": "offline_yaml",
            "layer": layer,
            "plan_index": plan_index,
            "simulation_time_s": round(float(simulation_time_s), 6),
            "linked_timeseries_time_s": round(
                float(linked_timeseries_time_s), 6),
            "time_semantics": time_semantics,
            "point_index": point_index,
            "x_m": round(float(point[0]), 6),
            "y_m": round(float(point[1]), 6),
            "z_m": round(float(point[2]), 6) if len(point) > 2 else math.nan,
            "color_hex": color,
            "line_style": line_style,
        })

    for plan_index, (time_s, spline, _corridor, astar) in enumerate(splines, 1):
        for point_index, point in enumerate(astar):
            append("astar", plan_index, time_s, time_s + control_dt,
                   "planning_event_start", point_index, point,
                   COLORS["astar"], "solid")
        for point_index, point in enumerate(spline):
            append("bspline", plan_index, time_s, time_s + control_dt,
                   "planning_event_start", point_index, point,
                   COLORS["bspline"], "solid")
    for plan_index, (time_s, horizon) in enumerate(zip(mpc_times, horizons), 1):
        for point_index, point in enumerate(horizon):
            append("mpc_horizon", plan_index, time_s, time_s + control_dt,
                   "control_interval_start", point_index, point,
                   COLORS["mpc"], "dashed")
    for point_index, row in enumerate(log):
        time_s = float(row["t_s"])
        append("final_executed", 0, time_s, time_s, "state_sample",
               point_index,
               (row["drone_x"], row["drone_y"], row["drone_z"]),
               COLORS["final"], "solid")
        append("trailer", 0, time_s, time_s, "state_sample", point_index,
               (row["trailer_x"], row["trailer_y"]),
               COLORS["trailer"], "solid")
    return rows


def _gazebo_phase_rows(actual_rows):
    """Phase-fraction-weighted estimates from the published 1 Hz table."""
    groups = (
        ("MISSION", {"MISSION"}),
        ("RETURN+RETURN_PLAN", {"RETURN", "RETURN_PLAN"}),
        ("PLANNED_FLIGHT_TOTAL", {"MISSION", "RETURN", "RETURN_PLAN"}),
    )
    output = []
    for label, states in groups:
        duration = drone_distance = trailer_distance = 0.0
        drone_max = trailer_max = 0.0
        bins = 0
        for row_index, row in enumerate(actual_rows):
            state = row["mission_state"]
            fraction = 1.0
            if row["phase_transition"] == "1":
                fraction = float(row["phase_fraction"])
                event = row["phase_events"].split("|")[-1]
                if event == state:
                    previous = (actual_rows[row_index - 1]["mission_state"]
                                if row_index else "")
                    group_fraction = (
                        1.0 if state in states and previous in states
                        else fraction if state in states
                        else 0.0)
                else:
                    group_fraction = ((fraction if state in states else 0.0)
                                      + (1.0 - fraction if event in states else 0.0))
            else:
                group_fraction = 1.0 if state in states else 0.0
            if group_fraction <= 0.0:
                continue
            dt = ((float(row["t_sim_end_s"]) - float(row["t_sim_start_s"]))
                  * group_fraction)
            drone_mean = float(row["speed_xy_mean_m_s"] or math.nan)
            trailer_bin_max = float(row["trailer_speed_max_m_s"] or math.nan)
            duration += dt
            bins += 1
            if math.isfinite(drone_mean):
                drone_distance += drone_mean * dt
            if math.isfinite(trailer_bin_max):
                trailer_distance += trailer_bin_max * dt
            drone_max = max(drone_max, float(row["speed_xy_max_m_s"] or 0.0))
            trailer_max = max(
                trailer_max, trailer_bin_max if math.isfinite(trailer_bin_max) else 0.0)
        output.append({
            "source": "Gazebo/PX4 r15 flight_1hz.csv phase-fraction estimate",
            "phase_group": label,
            "bins": bins,
            "duration_s": round(duration, 6),
            "drone_distance_1hz_estimate_m": round(drone_distance, 6),
            "drone_speed_time_weighted_mean_m_s": round(
                drone_distance / duration if duration else math.nan, 6),
            "drone_speed_max_m_s": round(drone_max, 6),
            "trailer_distance_using_bin_max_m": round(trailer_distance, 6),
            "trailer_bin_max_time_weighted_mean_m_s": round(
                trailer_distance / duration if duration else math.nan, 6),
            "trailer_speed_max_m_s": round(trailer_max, 6),
        })
    return output


def _figure_path_metrics(figures, replan_rows):
    index = _array(replan_rows, "replan_index")
    astar_length = _array(replan_rows, "astar_length_m")
    spline_length = _array(replan_rows, "bspline_length_m")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(index, astar_length, color=COLORS["astar"], linewidth=1.0,
                    label="A*")
    axes[0, 0].plot(index, spline_length, color=COLORS["bspline"],
                    linewidth=1.0, label="B-spline")
    axes[0, 0].set_ylabel("Path length [m]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(index, _array(replan_rows, "length_reduction_pct"),
                    color=COLORS["mpc"], linewidth=1.0)
    axes[0, 1].set_ylabel("B-spline length reduction [%]")
    axes[1, 0].plot(index, _array(replan_rows, "astar_points"),
                    color=COLORS["astar"], linewidth=1.0)
    axes[1, 0].set_ylabel("A* guide points")
    axes[1, 1].plot(index, _array(replan_rows, "sfc_boxes"),
                    color="#1c7ed6", linewidth=1.0)
    axes[1, 1].set_ylabel("SFC boxes")
    for ax in axes.ravel():
        ax.set_xlabel("Accepted replan index")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{TITLE}\n{OFFLINE_SCOPE}\nPer-replan geometry statistics")
    _save(fig, figures / "11_replan_path_statistics.png")


def _figure_solver_timing(figures, plan_stats, arrays, scenario):
    attempts = _array(plan_stats, "attempt_index")
    accepted = _array(plan_stats, "accepted") > 0.5
    total = _array(plan_stats, "total_plan_ms")
    astar = _array(plan_stats, "astar_solve_ms")
    bspline = _array(plan_stats, "bspline_solve_ms")
    mpc_valid = np.isfinite(arrays["mpc_solve_ms"])
    mpc_time = arrays["t_s"][mpc_valid]
    mpc_ms = arrays["mpc_solve_ms"][mpc_valid]
    deadline_ms = 1000.0 * float(scenario["pursuit"]["sim_dt_s"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].scatter(attempts[accepted], total[accepted], s=14,
                       color=COLORS["bspline"], label="accepted")
    axes[0, 0].scatter(attempts[~accepted], total[~accepted], s=18,
                       marker="x", color=COLORS["limit"], label="rejected")
    axes[0, 0].set_ylabel("A* + B-spline wall time [ms]")
    axes[0, 0].set_xlabel("Planning attempt")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(attempts, astar, color=COLORS["astar"], s=12,
                       label="A*")
    valid_bspline = np.isfinite(bspline)
    axes[0, 1].scatter(attempts[valid_bspline], bspline[valid_bspline],
                       color=COLORS["bspline"], s=12, label="B-spline")
    axes[0, 1].set_ylabel("Stage wall time [ms]")
    axes[0, 1].set_xlabel("Planning attempt")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(mpc_time, mpc_ms, color=COLORS["mpc"], linewidth=0.7)
    axes[1, 0].axhline(deadline_ms, color=COLORS["limit"], linestyle="--",
                       linewidth=0.8, label=f"control interval {deadline_ms:.0f} ms")
    axes[1, 0].set_ylabel("MPC solve wall time [ms]")
    axes[1, 0].set_xlabel("Simulation time [s]")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(mpc_ms, bins=40, color=COLORS["mpc"], alpha=0.78,
                    edgecolor="white")
    p95 = float(np.percentile(mpc_ms, 95))
    axes[1, 1].axvline(p95, color=COLORS["limit"], linestyle="--",
                       linewidth=0.9, label=f"p95 {p95:.2f} ms")
    axes[1, 1].set_xlabel("MPC solve wall time [ms]")
    axes[1, 1].set_ylabel("Control steps")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
    fig.suptitle(f"{TITLE}\n{OFFLINE_SCOPE}\nPlanner and controller timing")
    _save(fig, figures / "12_offline_solver_timing.png")


def _figure_correlation(figures, arrays):
    labels = ("Range", "Speed", "|Accel|", "Tracking error", "MPC solve")
    matrix = np.column_stack((
        arrays["dist_xy_m"], arrays["drone_speed_mps"],
        np.abs(arrays["drone_accel_mps2"]), arrays["track_err_dense_m"],
        arrays["mpc_solve_ms"],
    ))
    finite = np.all(np.isfinite(matrix), axis=1)
    correlation = np.corrcoef(matrix[finite], rowvar=False)
    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, f"{correlation[row, column]:.2f}",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Pearson correlation")
    ax.set_title(f"{TITLE}\n{OFFLINE_SCOPE}\nFinite pursuit-sample correlation")
    _save(fig, figures / "13_offline_correlation_matrix.png")


def _offline_summary(log, arrays, replan_rows, plan_stats, scenario,
                     sfc_box_rows=None, world=None):
    pursuit_mask = np.asarray([row["phase"] == "pursuit" for row in log])
    error = arrays["track_err_dense_m"][pursuit_mask]
    error = error[np.isfinite(error)]
    if not len(error):
        raise RuntimeError("no finite pursuit tracking errors for summary")
    mpc_mask = np.isfinite(arrays["mpc_solve_ms"])
    mpc_solve = arrays["mpc_solve_ms"][mpc_mask]
    mpc_success = arrays["mpc_success"][mpc_mask]
    mpc_cte = arrays["mpc_cte_m"][mpc_mask]
    mpc_epsi = arrays["mpc_epsi_rad"][mpc_mask]
    plan_total = _array(plan_stats, "total_plan_ms")
    astar_time = _array(plan_stats, "astar_solve_ms")
    bspline_time = _array(plan_stats, "bspline_solve_ms")
    bspline_time = bspline_time[np.isfinite(bspline_time)]
    accepted_mask = _array(plan_stats, "accepted") > 0.5
    accepted_attempts = int(np.sum(accepted_mask))
    sfc_time = _array(plan_stats, "sfc_generation_time_ms")[accepted_mask]
    sfc_time = sfc_time[np.isfinite(sfc_time)]
    sfc_box_rows = list(sfc_box_rows or [])
    sfc_width = _array(sfc_box_rows, "horizontal_width_m")
    sfc_width = sfc_width[np.isfinite(sfc_width)]
    sfc_non_degenerate = sfc_width[sfc_width > SFC_DEGENERATE_TOL_M]
    if not len(sfc_non_degenerate):
        raise RuntimeError("all optimizer SFC boxes are degenerate")
    sfc_degenerate_count = int(np.sum(
        sfc_width <= SFC_DEGENERATE_TOL_M))
    all_sfc_time = _array(plan_stats, "sfc_generation_time_ms")
    all_sfc_time = all_sfc_time[np.isfinite(all_sfc_time)]
    attempt_times = _array(plan_stats, "simulation_time_s")
    accepted_times = _array(replan_rows, "simulation_time_s")
    attempt_intervals = np.diff(attempt_times)
    accepted_intervals = np.diff(accepted_times)
    final_points = np.column_stack((arrays["drone_x"], arrays["drone_y"]))
    path_length = _path_length(final_points)
    if world is None:
        min_clearance = min_clearance_residual = math.nan
        clearance_evaluated_samples = 0
        clearance_violation_samples = math.nan
        clearance_violation_rate_pct = math.nan
    else:
        executed_xyz = np.column_stack((
            arrays["drone_x"], arrays["drone_y"], arrays["drone_z"]))
        clearance_samples = _xy_clearances(world, executed_xyz)
        min_clearance = float(np.min(clearance_samples))
        required_clearance = float(world.xy_clearance_m)
        min_clearance_residual = min_clearance - required_clearance
        clearance_violation_samples = int(np.sum(
            clearance_samples < required_clearance - 1.0e-9))
        clearance_evaluated_samples = len(clearance_samples)
        clearance_violation_rate_pct = (
            100.0 * clearance_violation_samples / clearance_evaluated_samples)
    capture_time = float(log[-1]["t_s"])
    speed_limit = float(scenario["drone"]["max_speed_m_s"])
    deadline_ms = 1000.0 * float(scenario["pursuit"]["sim_dt_s"])
    return {
        "dataset_type": "city YAML offline kinematic dynamic-pursuit rollout",
        "controller": "mpc_ros UnicycleMPC (not Gazebo Wang TrackingMPC)",
        "captured": bool(int(log[-1]["captured"])),
        "capture_time_s": capture_time,
        "final_range_m": float(log[-1]["dist_xy_m"]),
        "accepted_replans": int(max(arrays["replan_count"])),
        "planning_attempts": len(plan_stats),
        "planning_rejections": len(plan_stats) - accepted_attempts,
        "planning_acceptance_rate_pct": 100.0 * accepted_attempts / len(plan_stats),
        "accepted_replans_per_min": 60.0 * accepted_attempts / capture_time,
        "configured_replan_period_s": float(
            scenario["pursuit"]["replan_period_s"]),
        "planning_attempt_interval_mean_s": float(np.mean(attempt_intervals)),
        "planning_attempt_interval_min_s": float(np.min(attempt_intervals)),
        "planning_attempt_interval_max_s": float(np.max(attempt_intervals)),
        "accepted_replan_interval_mean_s": float(np.mean(accepted_intervals)),
        "timeseries_rows": len(log),
        "path_length_m": path_length,
        "executed_path_length_m": path_length,
        "initial_range_m": float(arrays["dist_xy_m"][0]),
        "range_reduction_m": float(arrays["dist_xy_m"][0] - arrays["dist_xy_m"][-1]),
        "drone_speed_limit_m_s": speed_limit,
        "mpc_reference_speed_m_s": float(
            scenario["pursuit"]["mpc_reference_speed_m_s"]),
        "drone_speed_max_m_s": float(np.max(arrays["drone_speed_mps"])),
        "drone_speed_mean_m_s": float(np.mean(arrays["drone_speed_mps"])),
        "speed_limit_exceedance_samples": int(np.sum(
            arrays["drone_speed_mps"] > speed_limit + 1.0e-6)),
        "trailer_speed_m_s": float(scenario["trailer"]["speed_m_s"]),
        "absolute_acceleration_max_m_s2": float(
            np.max(np.abs(arrays["drone_accel_mps2"]))),
        "acceleration_rms_m_s2": float(np.sqrt(np.mean(
            arrays["drone_accel_mps2"] ** 2))),
        "absolute_yaw_rate_max_rad_s": float(np.max(np.abs(
            arrays["drone_yawrate_rads"]))),
        "altitude_min_m": float(np.min(arrays["drone_z"])),
        "altitude_max_m": float(np.max(arrays["drone_z"])),
        "tracking_error_mean_m": float(np.mean(error)),
        "tracking_error_rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "tracking_error_median_m": float(np.median(error)),
        "tracking_error_p95_m": float(np.percentile(error, 95)),
        "tracking_error_p99_m": float(np.percentile(error, 99)),
        "tracking_error_max_m": float(np.max(error)),
        "min_clearance_m": min_clearance,
        "min_clearance_residual_m": min_clearance_residual,
        "min_clearance_scope": "10Hz state-sample physical_aabb_xy_distance",
        "clearance_evaluated_samples": clearance_evaluated_samples,
        "clearance_violation_samples": clearance_violation_samples,
        "clearance_violation_rate_pct": clearance_violation_rate_pct,
        "mpc_solve_samples": len(mpc_solve),
        "mpc_success_rate_pct": 100.0 * float(np.mean(mpc_success)),
        "mpc_solve_mean_ms": float(np.mean(mpc_solve)),
        "mpc_solve_p95_ms": float(np.percentile(mpc_solve, 95)),
        "mpc_solve_max_ms": float(np.max(mpc_solve)),
        "mpc_solve_time_ms": float(np.mean(mpc_solve)),
        "mpc_deadline_ms": deadline_ms,
        "mpc_deadline_miss_samples": int(np.sum(mpc_solve > deadline_ms)),
        "mpc_abs_cte_mean_m": float(np.mean(np.abs(mpc_cte))),
        "mpc_abs_epsi_mean_rad": float(np.mean(np.abs(mpc_epsi))),
        "global_plan_total_mean_ms": float(np.mean(plan_total)),
        "global_plan_total_p95_ms": float(np.percentile(plan_total, 95)),
        "global_plan_total_max_ms": float(np.max(plan_total)),
        "astar_solve_mean_ms": float(np.mean(astar_time)),
        "astar_solve_max_ms": float(np.max(astar_time)),
        "astar_plan_time_ms": float(np.mean(astar_time)),
        "bspline_solve_mean_ms": float(np.mean(bspline_time)),
        "bspline_solve_max_ms": float(np.max(bspline_time)),
        "bspline_solve_time_scope": (
            "optimizer.optimize wall time inclusive of SFC generation; do "
            "not add SFC time again"),
        "sfc_generation_time_ms": float(np.mean(sfc_time)),
        "sfc_generation_time_max_ms": float(np.max(sfc_time)),
        "sfc_generation_time_count": len(sfc_time),
        "sfc_generation_time_scope": (
            "accepted attempts; initial boxes_for_points plus rebound refresh "
            "wall time"),
        "sfc_generation_all_attempts_mean_ms": float(np.mean(all_sfc_time)),
        "sfc_generation_all_attempts_p95_ms": float(np.percentile(
            all_sfc_time, 95)),
        "sfc_min_width_m": float(np.min(sfc_width)),
        "sfc_avg_width_m": float(np.mean(sfc_width)),
        "sfc_width_scope": (
            "box-weighted minimum XY extent of optimizer control-point boxes; "
            "includes degenerate seed fallbacks"),
        "sfc_corridor_count": int(
            replan_rows[-1]["sfc_corridor_count"]),
        "sfc_corridor_count_scope": (
            "compatibility name: control-point box count in latest accepted "
            "optimizer SFC"),
        "sfc_plan_count": len(replan_rows),
        "sfc_corridor_count_mean": float(np.mean(
            _array(replan_rows, "sfc_corridor_count"))),
        "sfc_corridor_count_max": int(np.max(
            _array(replan_rows, "sfc_corridor_count"))),
        "optimizer_sfc_box_count_latest": int(
            replan_rows[-1]["sfc_corridor_count"]),
        "optimizer_sfc_box_count_mean": float(np.mean(
            _array(replan_rows, "sfc_corridor_count"))),
        "optimizer_sfc_box_count_max": int(np.max(
            _array(replan_rows, "sfc_corridor_count"))),
        "optimizer_sfc_raw_min_xy_extent_m": float(np.min(sfc_width)),
        "optimizer_sfc_non_degenerate_min_xy_extent_m": float(np.min(
            sfc_non_degenerate)),
        "optimizer_sfc_non_degenerate_p05_xy_extent_m": float(np.percentile(
            sfc_non_degenerate, 5)),
        "optimizer_sfc_degenerate_box_count": sfc_degenerate_count,
        "optimizer_sfc_degenerate_box_rate_pct": (
            100.0 * sfc_degenerate_count / len(sfc_width)),
        # These optimizer control-point boxes are not an active-polyline
        # containment certificate, so a vehicle violation count would be a
        # different metric.  Keep it explicitly unavailable instead of
        # silently reporting a fabricated zero.
        "sfc_violation_count": math.nan,
        "sfc_violation_available": False,
        "sfc_violation_scope": (
            "N/A: optimizer control-point SFC is not an active-path vehicle "
            "containment contract"),
        "replan_count": int(max(arrays["replan_count"])),
        "aruco_detection_rate_pct": math.nan,
        "relative_xy_error_m": math.nan,
        "landing_xy_error_m": math.nan,
        "touchdown_relative_speed_m_s": math.nan,
        "unavailable_metric_scope": (
            "ArUco/relative-landing/touchdown metrics require Gazebo or "
            "vehicle telemetry"),
        "astar_length_mean_m": float(np.mean(
            _array(replan_rows, "astar_length_m"))),
        "bspline_length_mean_m": float(np.mean(
            _array(replan_rows, "bspline_length_m"))),
        "bspline_length_reduction_mean_pct": float(np.mean(
            _array(replan_rows, "length_reduction_pct"))),
    }


def _parse_actual(actual_dir):
    summary = _read_one_csv(actual_dir / "flight_summary.csv")
    experiment = _read_one_csv(actual_dir / "experiment_metrics.csv")
    with (actual_dir / "flight_1hz.csv").open(newline="", encoding="utf-8") as stream:
        timeseries = list(csv.DictReader(stream))
    text = (actual_dir / "gimbal_mission.log").read_text(encoding="utf-8")
    pattern = re.compile(
        r"global A\*/B-spline: (?P<samples>\d+) samples, "
        r"(?P<length>[0-9.]+) m, (?P<expanded>\d+) A\* expansions, "
        r"(?P<age>[0-9.]+) s, target drift (?P<drift>[0-9.]+) m")
    plans = []
    for index, match in enumerate(pattern.finditer(text), 1):
        plans.append({
            "accepted_return_plan": index,
            "bspline_samples": int(match.group("samples")),
            "bspline_length_m": float(match.group("length")),
            "astar_expansions": int(match.group("expanded")),
            "plan_age_sim_s": float(match.group("age")),
            "target_drift_m": float(match.group("drift")),
        })
    timing = re.search(
        r"EXPERIMENT_METRICS .*?mpc_count=(?P<count>\d+) "
        r"mpc_total_ms=(?P<total>[0-9.]+) mpc_max_ms=(?P<maximum>[0-9.]+)",
        text)
    if timing is None:
        raise ValueError("missing Gazebo r15 EXPERIMENT_METRICS timing record")
    attempts = (len(plans) + text.count("keep prior route; unsafe swap")
                + text.count("keep prior route; stale result"))
    dynamic_attempts = max(0, attempts - 1)
    dynamic_commits = max(0, len(plans) - 1)
    setpoint_speed = _array(timeseries, "sp_speed_xy_max_m_s")
    setpoint_speed = setpoint_speed[np.isfinite(setpoint_speed)]
    measured_trailer_speed = _array(timeseries, "trailer_speed_max_m_s")
    measured_trailer_speed = measured_trailer_speed[
        np.isfinite(measured_trailer_speed)]
    planned_rows = [row for row in timeseries if row["mission_state"] in {
        "MISSION", "RETURN", "RETURN_PLAN"}]
    planned_altitude = _array(planned_rows, "z_local_enu_m")
    planned_altitude = planned_altitude[np.isfinite(planned_altitude)]
    actual = {
        "dataset_type": "Gazebo/PX4 r15 measured partial flight",
        "result": summary["result"],
        "drone_speed_limit_m_s": 12.0,
        "trailer_speed_m_s": 7.0,
        "simulation_speed_factor": 5.0,
        "mission_altitude_m": 10.0,
        "actual_planned_altitude_min_m": float(np.min(planned_altitude)),
        "actual_planned_altitude_max_m": float(np.max(planned_altitude)),
        "planning_clearance_m": float(summary["vehicle_clearance_xy_m"]),
        "return_plan_attempts": attempts,
        "accepted_return_plans": len(plans),
        "return_plan_acceptance_rate_pct": 100.0 * len(plans) / attempts,
        "dynamic_replacement_attempts": dynamic_attempts,
        "dynamic_replacement_commits": dynamic_commits,
        "dynamic_replacement_commit_rate_pct": (
            100.0 * dynamic_commits / dynamic_attempts if dynamic_attempts else 0.0),
        "unsafe_swap_rejections": text.count("keep prior route; unsafe swap"),
        "stale_plan_rejections": text.count("keep prior route; stale result"),
        "bspline_length_mean_m": float(np.mean(_array(plans, "bspline_length_m"))),
        "bspline_length_min_m": float(np.min(_array(plans, "bspline_length_m"))),
        "bspline_length_max_m": float(np.max(_array(plans, "bspline_length_m"))),
        "astar_expansions_mean": float(np.mean(_array(plans, "astar_expansions"))),
        "plan_age_mean_sim_s": float(np.mean(_array(plans, "plan_age_sim_s"))),
        "target_drift_mean_m": float(np.mean(_array(plans, "target_drift_m"))),
        "actual_max_horizontal_speed_m_s": float(summary["max_speed_xy_m_s"]),
        "actual_max_setpoint_speed_m_s": float(np.max(setpoint_speed)),
        "actual_trailer_max_speed_m_s": float(np.max(measured_trailer_speed)),
        "actual_max_horizontal_acceleration_m_s2": float(
            summary["max_accel_xy_m_s2"]),
        "tracking_rmse_m": float(experiment["path_tracking_rmse_m"]),
        "tracking_error_max_m": float(experiment["path_tracking_error_max_m"]),
        "mpc_solve_mean_ms": float(experiment["mpc_solve_mean_ms"]),
        "mpc_solve_max_ms": float(experiment["mpc_solve_max_ms"]),
        "mpc_solve_count": int(timing.group("count")),
        "mpc_solve_total_ms": float(timing.group("total")),
        "minimum_physical_obstacle_distance_m": float(
            summary["min_physical_obstacle_distance_m"]),
        "minimum_clearance_residual_m": float(
            summary["min_obstacle_clearance_residual_m"]),
        "planner_failure_events": int(summary["planner_failure_events"]),
        "abort_events": int(summary["abort_events"]),
        "landing_acquire_entries": text.count("RETURN -> LANDING_ACQUIRE"),
        "landing_acquire_runway_timeouts": text.count(
            "LANDING_ACQUIRE -> RETURN"),
        "precland_attempts": int(summary["precland_attempts"]),
        "failsafe_seen": int(summary["failsafe_seen"]),
        "ulog_dropouts": int(summary["ulog_dropouts"]),
        "actual_max_sample_gap_s": float(summary["actual_max_sample_gap_s"]),
        "paper_reproducible": int(summary["paper_reproducible"]),
        "quality": summary["quality"],
        "quality_reasons": summary["quality_reasons"],
    }
    return actual, timeseries, plans


def _figure_dashboard(figures, offline, actual):
    rows = [
        ("Controller", "UnicycleMPC", "Wang TrackingMPC"),
        ("Drone limit [m/s]", f"{offline['drone_speed_limit_m_s']:.1f}",
         f"{actual['drone_speed_limit_m_s']:.1f}"),
        ("Trailer speed [m/s]", f"{offline['trailer_speed_m_s']:.1f}",
         f"{actual['trailer_speed_m_s']:.1f}"),
        ("Accepted replans", str(offline["accepted_replans"]),
         str(actual["accepted_return_plans"])),
        ("Max drone speed [m/s]", f"{offline['drone_speed_max_m_s']:.3f}",
         f"{actual['actual_max_horizontal_speed_m_s']:.3f}"),
        ("Tracking RMSE [m]",
         f"{offline['tracking_error_rmse_m']:.3f}",
         f"{actual['tracking_rmse_m']:.3f}"),
        ("Tracking error max [m]", f"{offline['tracking_error_max_m']:.3f}",
         f"{actual['tracking_error_max_m']:.3f}"),
        ("Outcome", "CAPTURED", f"{actual['result'].upper()} before landing"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=("Metric", "Offline YAML rollout", "Gazebo r15"),
        loc="center", cellLoc="center", colWidths=(0.40, 0.28, 0.28))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    for column in range(3):
        table[(0, column)].set_facecolor("#dee2e6")
        table[(0, column)].set_text_props(weight="bold")
    ax.set_title(
        f"{TITLE}\nDataset summary (different scenarios; not a paired comparison)",
        fontsize=15, pad=10)
    _save(fig, figures / "14_summary_dashboard.png")


def _actual_map_coordinates(actual_rows, actual_map):
    spawn = actual_map["spawn"]["gazebo_spawn_pose_enu"]
    sx, sy = float(spawn["x"]), float(spawn["y"])
    return {
        "t": _array(actual_rows, "t_from_arm_s"),
        "state": np.asarray([row["mission_state"] for row in actual_rows]),
        "actual_x": _array(actual_rows, "map_x_m"),
        "actual_y": _array(actual_rows, "map_y_m"),
        "setpoint_x": _array(actual_rows, "sp_x_enu_m") + sx,
        "setpoint_y": _array(actual_rows, "sp_y_enu_m") + sy,
        "trailer_x": _array(actual_rows, "trailer_x_local_enu_m") + sx,
        "trailer_y": _array(actual_rows, "trailer_y_local_enu_m") + sy,
        "speed": _array(actual_rows, "speed_xy_max_m_s"),
        "setpoint_speed": _array(actual_rows, "sp_speed_xy_max_m_s"),
        "trailer_speed": _array(actual_rows, "trailer_speed_max_m_s"),
        "range": _array(actual_rows, "relative_xy_mean_m"),
        "relative_speed": _array(actual_rows, "relative_speed_xy_mean_m_s"),
        "tracking_mean": _array(actual_rows, "xy_tracking_error_mean_m"),
        "tracking_max": _array(actual_rows, "xy_tracking_error_max_m"),
    }


def _figure_actual_route(figures, data, world, trailer):
    valid = np.isfinite(data["actual_x"]) & np.isfinite(data["actual_y"])
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    _map_background(ax, world, trailer, route=False)
    ax.plot(data["actual_x"][valid], data["actual_y"][valid],
            color=COLORS["final"], linewidth=0.9, alpha=0.75,
            zorder=4, label="actual vehicle")
    ax.plot(data["setpoint_x"], data["setpoint_y"], "--",
            color=COLORS["mpc"], linewidth=1.0, alpha=0.88, zorder=5,
            label="TrackingMPC/PX4 setpoint")
    ax.plot(data["trailer_x"], data["trailer_y"], color=COLORS["trailer"],
            linewidth=0.85, label="measured trailer")
    ax.scatter([-165.0], [0.0], marker="*", s=90, color="#fcc419",
               edgecolor="black", linewidth=0.5, label="mission goal")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.set_title(f"{TITLE}\nGazebo r15 measured map trace (12/7 m/s, 5×)")
    _save(fig, figures / "15_gazebo_r15_actual_route.png")


def _figure_actual_dynamics(figures, data):
    valid = data["t"] >= 0.0
    t = data["t"][valid]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, data["speed"][valid], color=COLORS["final"], linewidth=0.9,
                 label="actual drone")
    axes[0].plot(t, data["setpoint_speed"][valid], "--", color=COLORS["mpc"],
                 linewidth=0.8, label="setpoint")
    axes[0].plot(t, data["trailer_speed"][valid], color=COLORS["trailer"],
                 linewidth=0.8, label="trailer")
    axes[0].axhline(12.0, color=COLORS["limit"], linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].legend(fontsize=8, ncol=3, loc="lower right",
                   bbox_to_anchor=(1.0, 1.01))
    axes[1].plot(t, data["range"][valid], color=COLORS["final"], linewidth=0.9)
    axes[1].set_ylabel("Drone–trailer range [m]")
    axes[2].plot(t, data["tracking_mean"][valid], color=COLORS["bspline"],
                 linewidth=0.9, label="mean")
    axes[2].plot(t, data["tracking_max"][valid], color=COLORS["astar"],
                 linewidth=0.7, alpha=0.75, label="max")
    axes[2].set_ylabel("XY tracking error [m]")
    axes[2].set_xlabel("Time from arm [s]")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"{TITLE}\nGazebo r15 measured dynamics")
    _save(fig, figures / "16_gazebo_r15_dynamics.png")


def _figure_actual_plans(figures, plans):
    index = _array(plans, "accepted_return_plan")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (
        ("bspline_length_m", "B-spline length [m]", COLORS["bspline"]),
        ("astar_expansions", "A* expansions", COLORS["astar"]),
        ("plan_age_sim_s", "Plan age [sim s]", COLORS["mpc"]),
        ("target_drift_m", "Target drift [m]", COLORS["final"]),
    )
    for ax, (key, label, color) in zip(axes.ravel(), metrics):
        ax.plot(index, _array(plans, key), "-o", color=color,
                linewidth=0.9, markersize=3)
        ax.set_xlabel("Accepted RETURN plan")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{TITLE}\nGazebo r15 accepted dynamic-return plan statistics")
    _save(fig, figures / "17_gazebo_r15_planning_statistics.png")


def _summary_rows(offline, actual):
    rows = []
    units = {
        "capture_time_s": "s", "final_range_m": "m",
        "configured_replan_period_s": "s",
        "planning_attempt_interval_mean_s": "s",
        "planning_attempt_interval_min_s": "s",
        "planning_attempt_interval_max_s": "s",
        "accepted_replan_interval_mean_s": "s",
        "initial_range_m": "m", "range_reduction_m": "m",
        "path_length_m": "m", "executed_path_length_m": "m",
        "min_clearance_m": "m", "min_clearance_residual_m": "m",
        "clearance_violation_rate_pct": "%",
        "drone_speed_limit_m_s": "m/s",
        "mpc_reference_speed_m_s": "m/s",
        "drone_speed_max_m_s": "m/s", "drone_speed_mean_m_s": "m/s",
        "trailer_speed_m_s": "m/s", "absolute_acceleration_max_m_s2": "m/s^2",
        "acceleration_rms_m_s2": "m/s^2",
        "absolute_yaw_rate_max_rad_s": "rad/s",
        "altitude_min_m": "m", "altitude_max_m": "m",
        "mission_altitude_m": "m",
        "actual_planned_altitude_min_m": "m",
        "actual_planned_altitude_max_m": "m",
        "tracking_error_mean_m": "m", "tracking_error_rmse_m": "m",
        "tracking_error_median_m": "m", "tracking_error_p95_m": "m",
        "tracking_error_p99_m": "m",
        "tracking_error_max_m": "m", "astar_length_mean_m": "m",
        "bspline_length_mean_m": "m", "actual_max_horizontal_speed_m_s": "m/s",
        "actual_max_setpoint_speed_m_s": "m/s",
        "actual_trailer_max_speed_m_s": "m/s",
        "actual_max_horizontal_acceleration_m_s2": "m/s^2",
        "relative_xy_error_m": "m", "landing_xy_error_m": "m",
        "touchdown_relative_speed_m_s": "m/s",
        "tracking_rmse_m": "m", "mpc_solve_mean_ms": "ms",
        "mpc_solve_time_ms": "ms", "astar_plan_time_ms": "ms",
        "mpc_solve_p95_ms": "ms", "mpc_solve_max_ms": "ms",
        "mpc_solve_total_ms": "ms", "mpc_deadline_ms": "ms",
        "mpc_abs_cte_mean_m": "m", "mpc_abs_epsi_mean_rad": "rad",
        "global_plan_total_mean_ms": "ms", "global_plan_total_p95_ms": "ms",
        "global_plan_total_max_ms": "ms", "astar_solve_mean_ms": "ms",
        "astar_solve_max_ms": "ms", "bspline_solve_mean_ms": "ms",
        "bspline_solve_max_ms": "ms",
        "sfc_generation_time_ms": "ms",
        "sfc_generation_time_max_ms": "ms",
        "sfc_generation_all_attempts_mean_ms": "ms",
        "sfc_generation_all_attempts_p95_ms": "ms",
        "sfc_min_width_m": "m", "sfc_avg_width_m": "m",
        "optimizer_sfc_raw_min_xy_extent_m": "m",
        "optimizer_sfc_non_degenerate_min_xy_extent_m": "m",
        "optimizer_sfc_non_degenerate_p05_xy_extent_m": "m",
        "minimum_physical_obstacle_distance_m": "m",
        "minimum_clearance_residual_m": "m", "plan_age_mean_sim_s": "sim s",
        "target_drift_mean_m": "m", "actual_max_sample_gap_s": "s",
    }
    for dataset, values in (("offline_yaml", offline), ("gazebo_r15", actual)):
        for key, value in values.items():
            rows.append({
                "dataset": dataset,
                "metric": key,
                "value": value,
                "unit": units.get(key, ""),
            })
    return rows


def _markdown_table(values, keys):
    lines = ["| Metric | Value |", "|---|---:|"]
    for key in keys:
        value = values[key]
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {text} |")
    return "\n".join(lines)


def _write_reports(output, scenario, offline, actual, figure_names):
    offline_keys = [
        "captured", "capture_time_s", "final_range_m", "accepted_replans",
        "planning_attempts", "planning_rejections",
        "planning_acceptance_rate_pct", "accepted_replans_per_min",
        "configured_replan_period_s", "planning_attempt_interval_mean_s",
        "planning_attempt_interval_min_s", "planning_attempt_interval_max_s",
        "accepted_replan_interval_mean_s",
        "path_length_m", "executed_path_length_m", "min_clearance_m",
        "min_clearance_residual_m", "clearance_violation_samples",
        "clearance_violation_rate_pct", "drone_speed_limit_m_s",
        "mpc_reference_speed_m_s",
        "initial_range_m", "range_reduction_m", "drone_speed_max_m_s",
        "drone_speed_mean_m_s", "speed_limit_exceedance_samples",
        "trailer_speed_m_s", "absolute_acceleration_max_m_s2",
        "acceleration_rms_m_s2", "absolute_yaw_rate_max_rad_s",
        "altitude_min_m", "altitude_max_m", "tracking_error_mean_m",
        "tracking_error_rmse_m", "tracking_error_median_m",
        "tracking_error_p95_m", "tracking_error_p99_m", "tracking_error_max_m",
        "mpc_solve_samples", "mpc_success_rate_pct", "mpc_solve_mean_ms",
        "mpc_solve_p95_ms", "mpc_solve_max_ms", "mpc_deadline_ms",
        "mpc_deadline_miss_samples", "global_plan_total_mean_ms",
        "global_plan_total_p95_ms", "global_plan_total_max_ms",
        "astar_solve_mean_ms", "bspline_solve_mean_ms",
        "sfc_generation_time_ms", "sfc_generation_time_max_ms",
        "sfc_generation_all_attempts_mean_ms",
        "sfc_generation_all_attempts_p95_ms", "sfc_min_width_m",
        "sfc_avg_width_m", "optimizer_sfc_box_count_latest",
        "optimizer_sfc_box_count_mean", "optimizer_sfc_box_count_max",
        "optimizer_sfc_raw_min_xy_extent_m",
        "optimizer_sfc_non_degenerate_min_xy_extent_m",
        "optimizer_sfc_non_degenerate_p05_xy_extent_m",
        "optimizer_sfc_degenerate_box_count",
        "optimizer_sfc_degenerate_box_rate_pct",
        "astar_length_mean_m", "bspline_length_mean_m",
        "bspline_length_reduction_mean_pct",
    ]
    actual_keys = [
        "result", "return_plan_attempts", "accepted_return_plans",
        "return_plan_acceptance_rate_pct",
        "dynamic_replacement_attempts", "dynamic_replacement_commits",
        "dynamic_replacement_commit_rate_pct",
        "unsafe_swap_rejections", "stale_plan_rejections",
        "mission_altitude_m", "actual_planned_altitude_min_m",
        "actual_planned_altitude_max_m",
        "bspline_length_mean_m", "astar_expansions_mean",
        "plan_age_mean_sim_s", "target_drift_mean_m",
        "actual_max_horizontal_speed_m_s",
        "actual_max_setpoint_speed_m_s", "actual_trailer_max_speed_m_s",
        "actual_max_horizontal_acceleration_m_s2", "tracking_rmse_m",
        "tracking_error_max_m", "mpc_solve_count", "mpc_solve_mean_ms",
        "mpc_solve_max_ms",
        "minimum_physical_obstacle_distance_m", "minimum_clearance_residual_m",
        "planner_failure_events", "abort_events", "failsafe_seen",
        "landing_acquire_entries", "landing_acquire_runway_timeouts",
        "precland_attempts",
        "ulog_dropouts", "actual_max_sample_gap_s", "paper_reproducible",
        "quality", "quality_reasons",
    ]
    figures = "\n".join(
        f"{index}. `figures/{name}`" for index, name in enumerate(figure_names, 1))
    text = f"""# {TITLE}

## 연구 범위

- `offline_yaml`: 도심 Gazebo YAML 장애물과 별도 이동 트레일러 시나리오를 이용한 A* → geometry-only B-spline → `mpc_ros UnicycleMPC` 오프라인 롤아웃입니다.
- `gazebo_r15`: 실제 Gazebo/PX4 5배속 실행에서 Wang `TrackingMPC`가 만든 측정 자료입니다.
- 두 데이터셋은 시나리오와 제어기가 다르므로 직접 성능 우열 비교용 paired experiment가 아닙니다.
- 실제 Gazebo 경로구간의 단계 구성은 `A* → geometry-only B-spline → Wang double-integrator TrackingMPC → PX4 setpoint`로 유지됩니다. 다만 원본 Wang 코드가 무변경이라는 뜻은 아니며 이동표적 재계획, active-SFC 검증·안전 교체, jerk-aware 속도 프로파일, recovery와 착륙 전환 로직이 추가됐습니다.
- 오프라인 결과는 같은 도심 장애물 YAML과 A*/B-spline 핵심만 공유합니다. ±600 m 해석적 표적 루프와 `UnicycleMPC` 이상 운동학을 사용하므로 실제 Gazebo/PX4 파이프라인과 동일한 실험이 아닙니다.

## 현재 설정

- 드론 속도 상한: `{scenario['drone']['max_speed_m_s']:.1f} m/s`
- 트레일러 속도: `{scenario['trailer']['speed_m_s']:.1f} m/s`
- 고도 목표: `{scenario['drone']['cruise_altitude_m']:.1f} m`
- 동적 재계획 주기: `{scenario['pursuit']['replan_period_s']:.1f} s`
- A* 해상도: `{scenario['pursuit']['astar_resolution_m']:.1f} m`
- B-spline control spacing: `{scenario['pursuit']['bspline_control_spacing_m']:.1f} m`
- MPC: `{scenario['pursuit']['sim_dt_s']:.1f} s`, horizon `{scenario['pursuit']['mpc_horizon']}`

## 오프라인 YAML 롤아웃 결과

{_markdown_table(offline, offline_keys)}

## 실제 Gazebo r15 동적 RETURN 결과

{_markdown_table(actual, actual_keys)}

> r15은 착륙 전에 수동 종료되어 `interrupted`입니다. 동적 경로계획 구간 분석에는 사용할 수 있지만 착륙 성공률 자료로 사용하면 안 됩니다. `mpc_solve_*`는 TrackingMPC와 LandingMPC 합산 계측입니다.

> 오프라인 wall-time은 현재 컴퓨터에서 1회 실행한 기술 계측입니다. 일반화된 논문 통계를 주장하려면 동일 조건 반복실험과 평균±표준편차/신뢰구간을 추가해야 합니다.

## 핵심 해석

- 오프라인 롤아웃은 148.1 s에 포착했고 속도 상한 초과는 0회였습니다. 다만 MPC p95 solve time은 `{offline['mpc_solve_p95_ms']:.1f} ms`로 100 ms 제어주기를 넘었고 deadline miss는 `{offline['mpc_deadline_miss_samples']}`회입니다.
- 오프라인 전역계획 wall-time은 평균 `{offline['global_plan_total_mean_ms'] / 1000.0:.2f} s`, p95 `{offline['global_plan_total_p95_ms'] / 1000.0:.2f} s`입니다. 따라서 현재 단일 프로세스 Python 롤아웃은 2 s 재계획 주기의 실시간성 증명으로 사용하면 안 됩니다.
- 실제 Gazebo r15에서는 동적 교체 16회 중 11회(`{actual['dynamic_replacement_commit_rate_pct']:.2f}%`)가 commit됐고 planner failure·abort·failsafe는 모두 0회였습니다.
- 실제 r15 최대속도는 `{actual['actual_max_horizontal_speed_m_s']:.3f} m/s`로 12 m/s 상한 이내지만 최대가속도 `{actual['actual_max_horizontal_acceleration_m_s2']:.3f} m/s²`와 최소 여유 `{actual['minimum_clearance_residual_m']:.3f} m`는 개선 대상입니다.
- r15은 `interrupted`이고 품질판정도 `FAIL`이므로 착륙 성공 근거가 아니라 동적 RETURN 경로계획의 부분 실행 자료로만 사용해야 합니다.
- 오프라인 `CAPTURED`는 수평거리 5 m 미만 접근일 뿐 하강·deck 접촉·disarm을 포함한 착륙 판정이 아닙니다. r15도 `LANDING_ACQUIRE` 뒤 RETURN으로 복귀했으며 `PRECLAND`/`DONE` 자료가 없습니다.

## 그림 목록

{figures}

## 재현 명령

```bash
cd ~/PX4-ROS2-jo
python3 flight/path_plan/tools/paper_dynamic_planning_report.py
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    (output / "paper_tables.md").write_text(
        "# Paper tables\n\n## Offline YAML rollout\n\n"
        + _markdown_table(offline, offline_keys)
        + "\n\n## Gazebo r15 measured dynamic RETURN\n\n"
        + _markdown_table(actual, actual_keys) + "\n",
        encoding="utf-8")
    (output / "DATA_DICTIONARY.md").write_text(
        """# Data dictionary

| File | Scope | Intended paper use |
|---|---|---|
| `data/offline_timeseries_10hz.csv` | time-aligned offline state/control samples | range, speed, acceleration, tracking-error and MPC timing plots |
| `data/path_points.csv` | long-form A*, B-spline, dashed MPC horizon, final and trailer coordinates | exact path re-rendering; event-start and linked post-step timestamps are both stored |
| `tables/offline_plan_attempts.csv` | every accepted/rejected A*→B-spline attempt | planner timing and acceptance statistics |
| `tables/offline_replan_metrics.csv` | accepted path geometry | path length, point count and SFC statistics |
| `tables/offline_sfc_boxes.csv` | raw optimizer control-point SFC boxes | per-plan box bounds, horizontal width and volume |
| `data/gazebo_r15_flight_1hz.csv` | measured Gazebo/PX4 r15 bins | actual route/dynamics plots |
| `tables/gazebo_r15_return_plans.csv` | accepted rolling RETURN replacements | actual A*, B-spline, plan-age and target-drift statistics |
| `tables/gazebo_r15_phase_summary.csv` | phase-fraction-weighted 1 Hz estimate | measured phase-duration/speed/distance summary |
| `tables/summary_metrics.csv` | tidy long-form headline values | manuscript tables/statistical import |
| `tables/summary_metrics.yaml` | same headline values, structured | reproducibility/config tooling |
| `code/` | exact planner/controller/report scripts used for this package | implementation snapshot for audit/reproduction |

`offline_yaml` and `gazebo_r15` are independent datasets. Empty/`nan` tracking
errors mark initialization, terminal homing or unavailable measurements and are
excluded from pursuit tracking statistics.

For A*/B-spline/MPC records, `simulation_time_s` is the plan/control event start;
`linked_timeseries_time_s` points to the post-step state row. For final/trailer
state records the two timestamps are identical.
""", encoding="utf-8")


def _manifest(output):
    target = output / "manifest.sha256"
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != target:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(output).as_posix()}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=pursuit.DEFAULT_SCENARIO)
    parser.add_argument("--actual-run", type=Path, default=DEFAULT_ACTUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scenario = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
    output = args.output.resolve()
    figures, tables, data_dir, code_dir = (
        output / "figures", output / "tables", output / "data", output / "code")
    for directory in (figures, tables, data_dir, code_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (log, captured, world, trailer, splines, horizons,
     plan_stats, mpc_times) = pursuit.run_sim(scenario)
    if not captured or not splines or not horizons:
        raise RuntimeError("paper dataset requires a captured run with all path stages")
    pursuit.save_csv(log, data_dir / "offline_timeseries_10hz.csv")
    arrays = _offline_arrays(log)
    replans = _replan_rows(splines)
    sfc_boxes = _sfc_box_rows(splines)
    _write_csv(tables / "offline_replan_metrics.csv", replans)
    _write_csv(tables / "offline_plan_attempts.csv", plan_stats)
    _write_csv(tables / "offline_sfc_boxes.csv", sfc_boxes)
    _write_csv(data_dir / "path_points.csv",
               _path_point_rows(log, splines, horizons, mpc_times))

    actual, actual_rows, actual_plans = _parse_actual(args.actual_run)
    _write_csv(tables / "gazebo_r15_return_plans.csv", actual_plans)
    _write_csv(tables / "gazebo_r15_phase_summary.csv",
               _gazebo_phase_rows(actual_rows))
    actual_map = yaml.safe_load(
        (args.actual_run / "map.yaml").read_text(encoding="utf-8"))
    actual_data = _actual_map_coordinates(actual_rows, actual_map)
    offline = _offline_summary(
        log, arrays, replans, plan_stats, scenario, sfc_boxes, world)

    _figure_pipeline_panels(figures, arrays, world, trailer, splines, horizons)
    _figure_overlay(figures, arrays, world, trailer, splines, horizons)
    _figure_single_paths(figures, arrays, world, trailer, splines, horizons)
    _figure_mpc_snapshots(
        figures, arrays, world, trailer, horizons, mpc_times)
    _figure_replan_snapshots(figures, arrays, world, trailer, splines)
    _figure_distance_replans(figures, arrays)
    _figure_dynamics(figures, arrays, scenario)
    _figure_error(figures, log, arrays)
    _figure_path_metrics(figures, replans)
    _figure_solver_timing(figures, plan_stats, arrays, scenario)
    _figure_correlation(figures, arrays)
    _figure_dashboard(figures, offline, actual)
    _figure_actual_route(figures, actual_data, world, trailer)
    _figure_actual_dynamics(figures, actual_data)
    _figure_actual_plans(figures, actual_plans)

    summary_rows = _summary_rows(offline, actual)
    _write_csv(tables / "summary_metrics.csv", summary_rows)
    (tables / "summary_metrics.yaml").write_text(
        yaml.safe_dump({"offline_yaml": offline, "gazebo_r15": actual},
                       sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy2(args.scenario, data_dir / "offline_scenario.yaml")
    shutil.copy2(pursuit.REPO / scenario["base_map"],
                 data_dir / "offline_base_map.yaml")
    shutil.copy2(args.actual_run / "map.yaml", data_dir / "gazebo_r15_map.yaml")
    shutil.copy2(args.actual_run / "flight_1hz.csv",
                 data_dir / "gazebo_r15_flight_1hz.csv")
    shutil.copy2(args.actual_run / "flight_summary.csv",
                 data_dir / "gazebo_r15_flight_summary.csv")
    shutil.copy2(args.actual_run / "experiment_metrics.csv",
                 data_dir / "gazebo_r15_experiment_metrics.csv")
    shutil.copy2(args.actual_run / "gimbal_mission.log",
                 data_dir / "gazebo_r15_mission.log")
    shutil.copy2(args.actual_run / "git_status.txt",
                 data_dir / "gazebo_r15_git_status.txt")
    shutil.copy2(args.actual_run / "manifest.tsv",
                 data_dir / "gazebo_r15_source_manifest.tsv")
    for source in (
        pursuit.REPO / "flight/path_plan/path_plan/astar.py",
        pursuit.REPO / "flight/path_plan/path_plan/bspline_optimizer.py",
        pursuit.REPO / "flight/path_plan/path_plan/mpc_ros.py",
        pursuit.REPO / "flight/path_plan/path_plan/world_model.py",
        Path(pursuit.__file__), Path(__file__),
    ):
        shutil.copy2(source, code_dir / source.name)

    figure_names = sorted(path.name for path in figures.glob("*.png"))
    if len(figure_names) < 10:
        raise RuntimeError("paper package must contain at least ten PNG figures")
    _write_reports(output, scenario, offline, actual, figure_names)
    _manifest(output)

    print(f"paper package: {output}")
    print(f"figures: {len(figure_names)}")
    print(f"capture: t={offline['capture_time_s']:.1f}s "
          f"range={offline['final_range_m']:.2f}m "
          f"replans={offline['accepted_replans']}")


if __name__ == "__main__":
    main()
