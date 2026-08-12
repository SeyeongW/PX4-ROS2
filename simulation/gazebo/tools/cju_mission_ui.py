#!/usr/bin/env python3
"""Preview the CJU YAML mission or display its live ROS execution.

The UI uses the same fail-closed A* -> geometry-only B-spline planner as the
mission manager.  Motion along that geometry is only a visual preview: PX4,
not the B-spline, owns speed and acceleration in the real mission.

Run after sourcing ROS 2 and the matching px4_msgs workspace::

    python3 simulation/gazebo/tools/cju_mission_ui.py
    python3 simulation/gazebo/tools/cju_mission_ui.py --check
    python3 simulation/gazebo/tools/cju_mission_ui.py --save /tmp/cju_path.png
    python3 simulation/gazebo/tools/cju_mission_ui.py --live

Both modes are read-only and never publish flight commands.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[3]
DEFAULT_MAP = REPO / "simulation/gazebo/maps/drone_cju_track.yaml"
sys.path[:0] = [
    str(REPO / "flight/path_plan"),
    str(REPO / "simulation/landing_mpc"),
]

try:
    from landing_mpc.frame import LOCAL_ENU_FRAME_ID
    from landing_mpc.mission_manager_node import (
        _mission_planning_segment_is_free,
        _mission_segment_is_free,
        _plan_global_path,
        _retarget_path_tail,
    )
except ModuleNotFoundError as exc:
    if exc.name == "px4_msgs":
        raise SystemExit(
            "px4_msgs is unavailable. Run: source /opt/ros/humble/setup.bash && "
            "source ~/px4_ros2_ws/install/setup.bash"
        ) from exc
    raise


def _frame_contract(document):
    mission = document["mission"]
    frame = document["frames"][mission["coordinate_frame"]]
    heading = math.radians(float(frame["heading_deg_enu"]))
    rotation = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)],
    ])
    origin = np.asarray(frame["origin_enu_m"][:2], float)
    spawn_pose = document["spawn"]["gazebo_spawn_pose_enu"]
    spawn = np.asarray([spawn_pose["x"], spawn_pose["y"]], float)
    return rotation, origin, spawn


def _map_to_local(points, rotation, origin, spawn, altitude):
    points = np.atleast_2d(np.asarray(points, float))
    world = points @ rotation.T + origin
    return np.column_stack((world - spawn, np.full(len(points), altitude)))


def _local_to_map(points, rotation, origin, spawn):
    points = np.atleast_2d(np.asarray(points, float))
    return (points[:, :2] + spawn - origin) @ rotation


def _trailer_route(document, rotation, origin):
    endpoints = np.asarray(
        document["trailer"]["shuttle_endpoints_enu_m"], float)
    return (endpoints - origin) @ rotation


def _trailer_position(route, speed_m_s, time_s):
    delta = route[1] - route[0]
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError("trailer shuttle endpoints must differ")
    distance = (max(0.0, time_s) * speed_m_s) % (2.0 * length)
    fraction = distance / length if distance <= length else 2.0 - distance / length
    return route[0] + fraction * delta


def _plan_map_leg(map_path, start_map=None, goal_map=None):
    document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    rotation, origin, spawn = _frame_contract(document)
    altitude = float(document["mission"]["cruise_altitude_m"])
    start_local = (None if start_map is None else
                   _map_to_local(start_map, rotation, origin, spawn, altitude)[0])
    goal_local = (None if goal_map is None else
                  _map_to_local(goal_map, rotation, origin, spawn, altitude)[0])
    arc, path_local, expanded = _plan_global_path(
        str(map_path), start_local, goal_local)
    return arc, _local_to_map(path_local, rotation, origin, spawn), expanded


def _path_position(arc, path, distance_m):
    distance = float(np.clip(distance_m, arc[0], arc[-1]))
    return np.array([
        np.interp(distance, arc, path[:, axis]) for axis in range(path.shape[1])])


def _point_message_to_local(message):
    if message.header.frame_id != LOCAL_ENU_FRAME_ID:
        return None
    point = np.array([message.point.x, message.point.y, message.point.z], float)
    return point if np.all(np.isfinite(point)) else None


def _path_message_to_local(message):
    if message.header.frame_id != LOCAL_ENU_FRAME_ID:
        return None
    points = np.asarray([
        [pose.position.x, pose.position.y, pose.position.z]
        for pose in message.poses
    ], float)
    if not len(points):
        return np.empty((0, 3))
    return points if points.shape[1:] == (3,) and np.all(np.isfinite(points)) else None


def build_preview(map_path, dt_s=0.1):
    """Return deterministic display frames and every validated path plan."""
    map_path = Path(map_path).resolve()
    document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    mission = document["mission"]
    rotation, origin, spawn = _frame_contract(document)
    altitude = float(mission["cruise_altitude_m"])
    route = _trailer_route(document, rotation, origin)
    trailer_speed = float(document["trailer"]["cruise_speed_m_s"])
    # This controls preview progress only.  It is never passed to the B-spline.
    preview_speed = float(
        document["px4_vehicle"]["sitl_parameter_overrides"]["MPC_XY_CRUISE"])
    handoff_m = float(mission["precland_handoff_m"])
    replan_s = float(mission["return_replan_min_period_s"])
    lookahead_m = float(mission["mpc_path_lookahead_m"])
    cross_track_m = float(mission["mpc_path_cross_track_m"])
    sample_spacing_m = float(mission["bspline_sample_spacing_m"])
    if min(dt_s, trailer_speed, preview_speed, handoff_m, replan_s,
           lookahead_m, cross_track_m, sample_spacing_m) <= 0.0:
        raise ValueError("preview and YAML motion values must be positive")

    arc, path, expanded = _plan_map_leg(map_path)
    plans = [{
        "phase": "MISSION",
        "start_frame": 0,
        "arc": arc,
        "path": path,
        "expanded": expanded,
        "target": path[-1].copy(),
    }]
    phase = "MISSION"
    plan_index = 0
    progress = 0.0
    drone = path[0].copy()
    time_s = 0.0
    last_return_update_s = None
    frames = []
    for _ in range(10_000):
        trailer = _trailer_position(route, trailer_speed, time_s)
        distance = float(np.linalg.norm(drone - trailer))
        frames.append({
            "time_s": time_s,
            "phase": phase,
            "drone": drone.copy(),
            "trailer": trailer.copy(),
            "distance_m": distance,
            "plan_index": plan_index,
        })

        refresh_return = False
        if phase == "RETURN":
            drone_local = _map_to_local(
                drone, rotation, origin, spawn, altitude)[0]
            trailer_local = _map_to_local(
                trailer, rotation, origin, spawn, altitude)[0]
            if (distance <= handoff_m and _mission_segment_is_free(
                    str(map_path), drone_local, trailer_local)):
                frames[-1]["phase"] = "PRECLAND HANDOFF"
                break
            refresh_return = (
                last_return_update_s is None
                or time_s - last_return_update_s >= replan_s)

        if not refresh_return:
            progress = min(progress + preview_speed * dt_s, float(arc[-1]))
            drone = _path_position(arc, path, progress)[:2]
            time_s += dt_s
            if progress < arc[-1] - 1.0e-9:
                continue
            if phase == "MISSION":
                phase = "RETURN"
            else:
                refresh_return = True

        trailer = _trailer_position(route, trailer_speed, time_s)
        if refresh_return:
            def planning_segment_is_free(start, goal):
                local = _map_to_local(
                    np.vstack((start, goal)), rotation, origin, spawn,
                    altitude)
                return _mission_planning_segment_is_free(
                    str(map_path), local[0], local[1])

            replacement = _retarget_path_tail(
                planning_segment_is_free, arc, path, progress, trailer,
                lookahead_m, cross_track_m, sample_spacing_m)
            last_return_update_s = time_s
            if replacement is not None:
                arc, path = replacement
                plan_index = len(plans)
                plans.append({
                    "phase": "RETURN",
                    "start_frame": len(frames),
                    "planned_at_s": time_s,
                    "arc": arc,
                    "path": path,
                    "expanded": None,
                    "target": trailer.copy(),
                })
                progress = min(
                    progress + preview_speed * dt_s, float(arc[-1]))
                drone = _path_position(arc, path, progress)[:2]
                time_s += dt_s
                continue
        try:
            arc, path, expanded = _plan_map_leg(map_path, drone, trailer)
        except RuntimeError:
            if not refresh_return or phase != "RETURN":
                raise
            # A failed fallback keeps the prior exact-safe route.
            progress = min(
                progress + preview_speed * dt_s, float(arc[-1]))
            drone = _path_position(arc, path, progress)[:2]
            time_s += dt_s
            continue
        progress = 0.0
        plan_index = len(plans)
        plans.append({
            "phase": "RETURN",
            "start_frame": len(frames),
            "planned_at_s": time_s,
            "arc": arc,
            "path": path,
            "expanded": expanded,
            "target": trailer.copy(),
        })
        last_return_update_s = time_s
    else:
        raise RuntimeError("CJU mission preview exceeded its finite step budget")

    return document, route, frames, plans, preview_speed


def _draw_obstacles(ax, mission):
    """Draw physical AABBs and their Euclidean XY clearance zones."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    clearance = float(mission["obstacle_clearance_m"])
    for index, obstacle in enumerate(mission["obstacles"], 1):
        center = np.asarray(obstacle["center_m"][:2], float)
        size = np.asarray(obstacle["size_m"][:2], float)
        corner = center - size / 2.0
        ax.add_patch(FancyBboxPatch(
            corner, *size,
            boxstyle=(f"round,pad={clearance},"
                      f"rounding_size={clearance}"),
            facecolor="#f8d7da", edgecolor="#dc3545", alpha=0.25,
            linewidth=0.8))
        ax.add_patch(Rectangle(
            corner, *size,
            facecolor="#343a40", edgecolor="black", linewidth=0.8))
        ax.text(center[0] + 0.35, center[1] + 0.35, str(index),
                fontsize=7, color="#343a40")


def _make_figure(document, route, frames, plans, preview_speed):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.subplots_adjust(bottom=0.14)
    mission = document["mission"]
    _draw_obstacles(ax, mission)

    ax.plot(route[:, 0], route[:, 1], "--", color="#e8590c", linewidth=1.5,
            label="trailer shuttle (YAML)")
    start = plans[0]["path"][0]
    goal = np.asarray(mission["goal_m"], float)
    ax.scatter(*start, marker="s", s=65, color="#1971c2", label="start")
    ax.scatter(*goal, marker="*", s=150, color="#fcc419", edgecolor="black",
               label="goal (50, 50)", zorder=8)

    mission_line, = ax.plot([], [], color="#2f9e44", linewidth=2.2,
                            label="A* -> geometry B-spline")
    return_line, = ax.plot([], [], color="#7048e8", linewidth=2.2,
                           label="active return replan")
    path_lines = [mission_line, return_line]

    drone_trace, = ax.plot([], [], color="#1971c2", linewidth=1.5,
                           label="drone preview")
    trailer_trace, = ax.plot([], [], color="#e8590c", linewidth=1.2, alpha=0.7)
    drone_dot, = ax.plot([], [], "o", color="#1971c2", markersize=9, zorder=9)
    trailer_size = np.asarray(document["trailer"]["body_footprint_m"], float)
    trailer_box = Rectangle((0.0, 0.0), *trailer_size,
                            facecolor="#ff922b", edgecolor="#d9480f",
                            alpha=0.65, zorder=7)
    ax.add_patch(trailer_box)
    handoff_line, = ax.plot([], [], ":", color="#ffd43b", linewidth=2.0)
    status = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top",
                     family="monospace", bbox={"facecolor": "white", "alpha": 0.85})

    all_points = np.vstack([
        route,
        goal[None, :],
        np.asarray([o["center_m"][:2] for o in mission["obstacles"]], float),
    ])
    lo, hi = all_points.min(axis=0) - 5.0, all_points.max(axis=0) + 5.0
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="map x [m]", ylabel="map y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title("CJU YAML mission: A* -> geometry-only B-spline -> trailer")
    ax.legend(loc="lower right", fontsize=8)

    def update(frame_index):
        frame = frames[frame_index]
        active = frame["plan_index"]
        mission_path = plans[0]["path"]
        mission_line.set_data(mission_path[:, 0], mission_path[:, 1])
        mission_line.set_alpha(1.0 if active == 0 else 0.3)
        mission_line.set_linewidth(2.8 if active == 0 else 1.2)
        if active:
            return_path = plans[active]["path"]
            return_line.set_data(return_path[:, 0], return_path[:, 1])
        else:
            return_line.set_data([], [])
        drone_history = np.asarray([item["drone"] for item in frames[:frame_index + 1]])
        trailer_history = np.asarray([item["trailer"] for item in frames[:frame_index + 1]])
        drone_trace.set_data(drone_history[:, 0], drone_history[:, 1])
        trailer_trace.set_data(trailer_history[:, 0], trailer_history[:, 1])
        drone_dot.set_data([frame["drone"][0]], [frame["drone"][1]])
        trailer_box.set_xy(frame["trailer"] - trailer_size / 2.0)
        if frame["phase"] == "PRECLAND HANDOFF":
            handoff_line.set_data(
                [frame["drone"][0], frame["trailer"][0]],
                [frame["drone"][1], frame["trailer"][1]])
        else:
            handoff_line.set_data([], [])
        plan = plans[active]
        plan_kind = ("tail retarget" if plan["expanded"] is None else
                     f"A* expansions: {plan['expanded']}")
        status.set_text(
            f"phase: {frame['phase']}\n"
            f"t: {frame['time_s']:.1f} s   trailer distance: {frame['distance_m']:.1f} m\n"
            f"plan: {plan['arc'][-1]:.1f} m   {plan_kind}\n"
            f"PX4 speed preview: {preview_speed:.1f} m/s (not a B-spline input)")
        return [*path_lines, drone_trace, trailer_trace, drone_dot,
                trailer_box, handoff_line, status]

    button_ax = fig.add_axes([0.42, 0.035, 0.16, 0.055])
    button = Button(button_ax, "Pause")
    return fig, update, button


def _check(document, route, frames, plans):
    obstacles = document["mission"]["obstacles"]
    centers = np.asarray([item["center_m"][:2] for item in obstacles], float)
    assert len(centers) == 20 and np.all(centers == np.round(centers))
    assert np.allclose(route, [[5.0, 0.0], [5.0, 50.0]], atol=1.0e-6)
    assert np.allclose(plans[0]["path"][0], [5.0, 0.0], atol=1.0e-6)
    assert np.allclose(plans[0]["path"][-1], [50.0, 50.0], atol=1.0e-6)
    assert len(plans) >= 3 and plans[1]["phase"] == "RETURN"
    full_plans = [plan for plan in plans if plan["expanded"] is not None]
    assert len(full_plans) == 2  # outbound + initial return
    return_times = [plan["planned_at_s"] for plan in plans[1:]]
    cadence = float(document["mission"]["return_replan_min_period_s"])
    assert all(cadence <= right - left <= cadence + 0.11
               for left, right in zip(return_times, return_times[1:]))
    assert frames[-1]["phase"] == "PRECLAND HANDOFF"
    assert all(np.all(np.isfinite(plan["path"])) for plan in plans)
    print(
        f"PASS obstacles={len(obstacles)} outbound={plans[0]['arc'][-1]:.3f}m "
        f"plans={len(plans)} handoff={frames[-1]['distance_m']:.3f}m")


def _run_live(map_path):
    """Display ROS measurements and the manager's accepted geometry."""
    import rclpy
    from geometry_msgs.msg import PointStamped, PoseArray
    from matplotlib.animation import FuncAnimation
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from std_msgs.msg import String
    import matplotlib.pyplot as plt

    map_path = Path(map_path).resolve()
    document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    rotation, origin, spawn = _frame_contract(document)
    route = _trailer_route(document, rotation, origin)
    data = {"state": "WAITING", "vehicle": None, "cue": None,
            "path": np.empty((0, 3))}
    vehicle_history, cue_history = deque(maxlen=5000), deque(maxlen=5000)

    rclpy.init()
    node = rclpy.create_node("cju_mission_live_ui")
    path_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST, depth=1)

    def on_vehicle(message):
        point = _point_message_to_local(message)
        if point is not None:
            data["vehicle"] = point

    def on_cue(message):
        point = _point_message_to_local(message)
        if point is not None:
            data["cue"] = point

    def on_path(message):
        path = _path_message_to_local(message)
        if path is not None:
            data["path"] = path

    node.create_subscription(
        PointStamped, "/mission/vehicle_position", on_vehicle, 10)
    node.create_subscription(PointStamped, "/marker/cue", on_cue, 10)
    node.create_subscription(
        PoseArray, "/mission/planned_path", on_path, path_qos)
    node.create_subscription(
        String, "/mission/state",
        lambda message: data.__setitem__("state", message.data), 10)

    fig, ax = plt.subplots(figsize=(9, 8))
    mission = document["mission"]
    _draw_obstacles(ax, mission)

    ax.plot(route[:, 0], route[:, 1], "--", color="#e8590c",
            linewidth=1.5, label="trailer shuttle (YAML)")
    goal = np.asarray(mission["goal_m"], float)
    ax.scatter(*goal, marker="*", s=150, color="#fcc419",
               edgecolor="black", label="goal (50, 50)", zorder=8)
    planned_line, = ax.plot([], [], color="#2f9e44", linewidth=2.5,
                            label="active validated path")
    vehicle_trace, = ax.plot([], [], color="#1971c2", linewidth=1.4,
                             label="vehicle measured")
    cue_trace, = ax.plot([], [], color="#e8590c", linewidth=1.1,
                         alpha=0.7, label="trailer measured")
    vehicle_dot, = ax.plot([], [], "o", color="#1971c2", markersize=9)
    cue_dot, = ax.plot([], [], "s", color="#e8590c", markersize=8)
    status = ax.text(
        0.015, 0.985, "", transform=ax.transAxes, va="top",
        family="monospace", bbox={"facecolor": "white", "alpha": 0.85})

    centers = np.asarray(
        [item["center_m"][:2] for item in mission["obstacles"]], float)
    all_points = np.vstack((route, goal[None, :], centers))
    lo, hi = all_points.min(axis=0) - 5.0, all_points.max(axis=0) + 5.0
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="map x [m]", ylabel="map y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title("CJU live mission (read-only ROS view)")
    ax.legend(loc="lower right", fontsize=8)

    def update(_frame):
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.0)
        vehicle = data["vehicle"]
        cue = data["cue"]
        path = data["path"]
        if vehicle is not None:
            vehicle_history.append(vehicle.copy())
            mapped = _local_to_map(vehicle, rotation, origin, spawn)[0]
            vehicle_dot.set_data([mapped[0]], [mapped[1]])
        if cue is not None:
            cue_history.append(cue.copy())
            mapped = _local_to_map(cue, rotation, origin, spawn)[0]
            cue_dot.set_data([mapped[0]], [mapped[1]])
        if vehicle_history:
            mapped = _local_to_map(
                np.asarray(vehicle_history), rotation, origin, spawn)
            vehicle_trace.set_data(mapped[:, 0], mapped[:, 1])
        if cue_history:
            mapped = _local_to_map(
                np.asarray(cue_history), rotation, origin, spawn)
            cue_trace.set_data(mapped[:, 0], mapped[:, 1])
        if len(path):
            mapped = _local_to_map(path, rotation, origin, spawn)
            planned_line.set_data(mapped[:, 0], mapped[:, 1])
        else:
            planned_line.set_data([], [])
        distance = (float(np.linalg.norm(vehicle[:2] - cue[:2]))
                    if vehicle is not None and cue is not None else float("nan"))
        status.set_text(
            f"state: {data['state']}\n"
            f"vehicle-trailer: {distance:.1f} m\n"
            f"active path samples: {len(path)}\n"
            "read-only: no flight commands")
        return (planned_line, vehicle_trace, cue_trace,
                vehicle_dot, cue_dot, status)

    animation = FuncAnimation(
        fig, update, interval=100, cache_frame_data=False, blit=False)
    print("live view subscribed to /mission/* and /marker/cue", flush=True)
    try:
        plt.show()
    finally:
        del animation
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--check", action="store_true",
                        help="plan both legs and exit without opening a window")
    parser.add_argument("--save", type=Path,
                        help="save the final UI frame instead of opening a window")
    parser.add_argument("--live", action="store_true",
                        help="show actual ROS path/vehicle/trailer data")
    args = parser.parse_args()
    if args.live and (args.check or args.save):
        parser.error("--live cannot be combined with --check or --save")
    if args.check or args.save:
        matplotlib.use("Agg")

    if args.live:
        _run_live(args.map)
        return

    document, route, frames, plans, preview_speed = build_preview(args.map)
    _check(document, route, frames, plans)
    if args.check:
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, update, button = _make_figure(
        document, route, frames, plans, preview_speed)
    if args.save:
        update(len(frames) - 1)
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
        plt.close(fig)
        return

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=40, repeat=True, blit=False)
    paused = False

    def toggle(_event):
        nonlocal paused
        paused = not paused
        (animation.pause if paused else animation.resume)()
        button.label.set_text("Resume" if paused else "Pause")

    button.on_clicked(toggle)
    plt.show()


if __name__ == "__main__":
    main()
