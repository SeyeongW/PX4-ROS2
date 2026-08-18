#!/usr/bin/env python3
"""Preview the CJU YAML mission or display its live ROS execution.

The UI uses the same fail-closed A* -> geometry-only B-spline planner as the
mission manager. Motion along that geometry is a deterministic preview of the
TrackingMPC leg, not a vehicle-dynamics simulation.

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
import textwrap
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[3]
DEFAULT_MAP = REPO / "simulation/gazebo/maps/drone_cju_track.yaml"
GIMBAL_AIM_FULL_RANGE_M = 9.0
sys.path[:0] = [
    str(REPO / "flight/path_plan"),
    str(REPO / "simulation/landing_mpc"),
]

try:
    from landing_mpc.frame import LOCAL_ENU_FRAME_ID
    from landing_mpc.mission_manager_node import (
        _mission_planning_segment_is_free,
        _plan_global_path,
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
    arc, path_local, expanded, diagnostics = _plan_global_path(
        str(map_path), start_local, goal_local, include_diagnostics=True)
    return (arc, _local_to_map(path_local, rotation, origin, spawn),
            expanded, diagnostics)


def _sfc_vertices(boxes_min, boxes_max):
    """Return top-down polygons for finite, ordered SFC boxes."""
    boxes_min = np.asarray(boxes_min, float)
    boxes_max = np.asarray(boxes_max, float)
    if (boxes_min.ndim != 2 or boxes_min.shape[1] < 2
            or boxes_max.shape != boxes_min.shape
            or not np.all(np.isfinite(np.r_[boxes_min, boxes_max]))
            or np.any(boxes_max < boxes_min)):
        raise ValueError("SFC boxes must be finite ordered Nx2/Nx3 arrays")
    return np.asarray([
        [[low[0], low[1]], [high[0], low[1]],
         [high[0], high[1]], [low[0], high[1]]]
        for low, high in zip(boxes_min, boxes_max)
    ], float)


def _sfc_message_vertices(message):
    """Decode the manager's flattened xmin,ymin,xmax,ymax SFC message."""
    values = np.asarray(message.data, float)
    if values.size % 4:
        raise ValueError("SFC message data length must be divisible by four")
    if not np.all(np.isfinite(values)):
        raise ValueError("SFC message must contain only finite values")
    boxes = values.reshape(-1, 4)
    if not len(boxes):
        return np.empty((0, 4, 2))
    return _sfc_vertices(boxes[:, :2], boxes[:, 2:])


def _active_plan_marker_snapshot(message, expected_frame):
    """Decode one atomic map-frame active path/SFC MarkerArray sample."""
    from visualization_msgs.msg import Marker

    markers = list(message.markers)
    empty_path = np.empty((0, 3))
    empty_sfc = np.empty((0, 4, 2))
    if not markers or markers[0].action != Marker.DELETEALL:
        raise ValueError("active plan must start with DELETEALL")
    reference_stamp = (
        markers[0].header.stamp.sec, markers[0].header.stamp.nanosec)
    for marker in markers:
        stamp = (marker.header.stamp.sec, marker.header.stamp.nanosec)
        if marker.header.frame_id != expected_frame or stamp != reference_stamp:
            raise ValueError("active plan frame/stamp mismatch")
    if len(markers) == 1:
        return None, empty_path, empty_sfc

    additions = markers[1:]
    paths = [marker for marker in additions if marker.ns == "active_path"]
    boxes = [marker for marker in additions if marker.ns == "active_sfc"]
    if len(paths) != 1 or len(boxes) + 1 != len(additions):
        raise ValueError("active plan needs one path and only SFC boxes")
    path_marker = paths[0]
    if (path_marker.type != Marker.LINE_STRIP
            or path_marker.action != Marker.ADD or path_marker.id <= 0):
        raise ValueError("active path marker contract is invalid")
    path = np.asarray([
        [point.x, point.y, point.z] for point in path_marker.points], float)
    if (path.shape[0] < 2 or path.shape[1:] != (3,)
            or not np.all(np.isfinite(path))):
        raise ValueError("active path must contain finite map points")
    if not boxes or len({marker.id for marker in boxes}) != len(boxes):
        raise ValueError("active SFC must contain uniquely identified boxes")

    lows, highs = [], []
    for marker in boxes:
        quaternion = marker.pose.orientation
        values = np.array([
            marker.pose.position.x, marker.pose.position.y,
            marker.pose.position.z, marker.scale.x, marker.scale.y,
            marker.scale.z, quaternion.x, quaternion.y, quaternion.z,
            quaternion.w], float)
        if (marker.type != Marker.CUBE or marker.action != Marker.ADD
                or not np.all(np.isfinite(values))
                or np.any(values[3:6] <= 0.0)
                or not np.allclose(values[6:10], [0.0, 0.0, 0.0, 1.0],
                                   atol=1.0e-9, rtol=0.0)):
            raise ValueError("active SFC cube contract is invalid")
        centre, extent = values[:3], values[3:6]
        lows.append(centre - 0.5 * extent)
        highs.append(centre + 0.5 * extent)
    return path_marker.id, path, _sfc_vertices(lows, highs)


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
    replan_s = float(mission["return_replan_min_period_s"])
    if min(dt_s, trailer_speed, preview_speed, replan_s) <= 0.0:
        raise ValueError("preview and YAML motion values must be positive")
    ideal_vision_range_m = GIMBAL_AIM_FULL_RANGE_M

    arc, path, expanded, diagnostics = _plan_map_leg(map_path)
    sfc_vertices = _sfc_vertices(
        diagnostics["sfc_boxes_min_map"],
        diagnostics["sfc_boxes_max_map"])
    plans = [{
        "phase": "MISSION",
        "start_frame": 0,
        "arc": arc,
        "path": path,
        "expanded": expanded,
        "sfc_vertices": sfc_vertices,
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
            # Offline preview cannot synthesize camera detections. Mark the
            # earliest conservative *ideal* observation gate at full gimbal
            # aim; a real run additionally needs 3 KF-accepted fixes/0.5 s.
            if (distance <= ideal_vision_range_m
                    and _mission_planning_segment_is_free(
                        str(map_path), drone_local, trailer_local)):
                frames[-1]["phase"] = "LANDING MPC ENTRY (ideal 3-fix)"
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
        try:
            arc, path, expanded, diagnostics = _plan_map_leg(
                map_path, drone, trailer)
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
            "sfc_vertices": _sfc_vertices(
                diagnostics["sfc_boxes_min_map"],
                diagnostics["sfc_boxes_max_map"]),
            "target": trailer.copy(),
        })
        last_return_update_s = time_s
    else:
        raise RuntimeError("CJU mission preview exceeded its finite step budget")

    return document, route, frames, plans, preview_speed


def _draw_obstacles(ax, mission):
    """Draw the physical YAML AABBs without storing clearance on them."""
    from matplotlib.patches import Rectangle

    for index, obstacle in enumerate(mission["obstacles"], 1):
        center = np.asarray(obstacle["center_m"][:2], float)
        size = np.asarray(obstacle["size_m"][:2], float)
        corner = center - size / 2.0
        ax.add_patch(Rectangle(
            corner, *size,
            facecolor="#343a40", edgecolor="black", linewidth=0.8, zorder=4,
            label="physical obstacle" if index == 1 else None))
        ax.text(center[0] + 0.35, center[1] + 0.35, str(index),
                fontsize=7, color="#343a40", zorder=5)


def _figure_layout(bottom=0.14):
    """Keep map artists and diagnostics in non-overlapping axes."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 8))
    grid = fig.add_gridspec(
        1, 2, width_ratios=(3.4, 1.6), left=0.06, right=0.98,
        bottom=bottom, top=0.93, wspace=0.08)
    ax = fig.add_subplot(grid[0])
    info_ax = fig.add_subplot(grid[1])
    info_ax.set_axis_off()
    return fig, ax, info_ax


def _draw_vehicle_radius(ax, radius_m, *, visible=True):
    """Show the clearance owned by the drone, not by each obstacle."""
    from matplotlib.patches import Circle

    circle = Circle(
        (0.0, 0.0), float(radius_m),
        facecolor=(0.10, 0.45, 0.85, 0.08), edgecolor="#1971c2",
        linestyle="--", linewidth=1.2, zorder=6,
        label=f"drone safety radius {float(radius_m):g} m")
    circle.set_visible(visible)
    ax.add_patch(circle)
    return circle


def _expand_axes_to_points(ax, points, pad_m=1.0):
    """Expand, but never shrink, the map so live SFC boxes stay visible."""
    points = np.asarray(points, float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    low, high = points.min(axis=0) - pad_m, points.max(axis=0) + pad_m
    ax.set_xlim(min(xlim[0], low[0]), max(xlim[1], high[0]))
    ax.set_ylim(min(ylim[0], low[1]), max(ylim[1], high[1]))


def _make_figure(document, route, frames, plans, preview_speed):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button

    fig, ax, info_ax = _figure_layout(bottom=0.14)
    mission = document["mission"]
    _draw_obstacles(ax, mission)
    vehicle_radius = float(mission["vehicle_clearance_xy_m"])
    planning_clearance = (
        vehicle_radius
        + float(mission.get("bspline_clearance_margin_m", 0.5)))
    sfc_collection = PolyCollection(
        [], facecolor=(0.13, 0.55, 0.90, 0.05),
        edgecolor=(0.09, 0.39, 0.67, 0.40), linewidth=0.7, zorder=1,
        label=f"active-path SFC ({planning_clearance:g} m planning)")
    ax.add_collection(sfc_collection)

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
    drone_radius = _draw_vehicle_radius(ax, vehicle_radius)
    trailer_size = np.asarray(document["trailer"]["body_footprint_m"], float)
    trailer_box = Rectangle((0.0, 0.0), *trailer_size,
                            facecolor="#ff922b", edgecolor="#d9480f",
                            alpha=0.65, zorder=7)
    ax.add_patch(trailer_box)
    handoff_line, = ax.plot([], [], ":", color="#ffd43b", linewidth=2.0)
    status = info_ax.text(
        0.0, 1.0, "", transform=info_ax.transAxes, va="top",
        family="monospace", fontsize=9)

    all_points = np.vstack([
        route,
        goal[None, :],
        np.asarray([o["center_m"][:2] for o in mission["obstacles"]], float),
    ])
    lo, hi = all_points.min(axis=0) - 5.0, all_points.max(axis=0) + 5.0
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="map x [m]", ylabel="map y [m]")
    all_sfc = [plan["sfc_vertices"].reshape(-1, 2) for plan in plans
               if len(plan["sfc_vertices"])]
    if all_sfc:
        _expand_axes_to_points(ax, np.vstack(all_sfc))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title("CJU YAML mission preview")
    handles, labels = ax.get_legend_handles_labels()
    info_ax.legend(handles, labels, loc="lower left", fontsize=8,
                   frameon=True, borderaxespad=0.0)

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
        drone_radius.center = frame["drone"][:2]
        trailer_box.set_xy(frame["trailer"] - trailer_size / 2.0)
        if frame["phase"].startswith("LANDING MPC ENTRY"):
            handoff_line.set_data(
                [frame["drone"][0], frame["trailer"][0]],
                [frame["drone"][1], frame["trailer"][1]])
        else:
            handoff_line.set_data([], [])
        plan = plans[active]
        sfc_collection.set_verts(plan["sfc_vertices"])
        status.set_text(
            "phase:\n"
            f"  {frame['phase']}\n"
            f"time: {frame['time_s']:.1f} s\n"
            f"trailer distance: {frame['distance_m']:.1f} m\n"
            f"plan length: {plan['arc'][-1]:.1f} m\n"
            f"A* expansions: {plan['expanded']}\n"
            f"active SFC boxes: {len(plan['sfc_vertices'])}\n"
            "  A* -> optimizer SFC -> B-spline\n"
            "  final path free-box certified\n"
            f"PX4 speed preview: {preview_speed:.1f} m/s\n"
            "  (not a B-spline input)")
        return [*path_lines, drone_trace, trailer_trace, drone_dot,
                trailer_box, handoff_line, sfc_collection, drone_radius,
                status]

    button_ax = fig.add_axes([0.42, 0.035, 0.16, 0.055])
    button = Button(button_ax, "Pause")
    return fig, update, button


def _check(document, route, frames, plans):
    obstacles = document["mission"]["obstacles"]
    centers = np.asarray([item["center_m"][:2] for item in obstacles], float)
    assert len(centers) == 25 and np.all(centers == np.round(centers))
    assert np.allclose(route, [[5.0, 0.0], [5.0, 50.0]], atol=1.0e-6)
    assert np.allclose(plans[0]["path"][0], [5.0, 0.0], atol=1.0e-6)
    assert np.allclose(plans[0]["path"][-1], [50.0, 50.0], atol=1.0e-6)
    assert len(plans) >= 3 and plans[1]["phase"] == "RETURN"
    assert all(plan["expanded"] is not None for plan in plans)
    return_times = [plan["planned_at_s"] for plan in plans[1:]]
    cadence = float(document["mission"]["return_replan_min_period_s"])
    assert all(cadence <= right - left <= cadence + 0.11
               for left, right in zip(return_times, return_times[1:]))
    assert frames[-1]["phase"] == "LANDING MPC ENTRY (ideal 3-fix)"
    assert all(np.all(np.isfinite(plan["path"])) for plan in plans)
    print(
        f"PASS obstacles={len(obstacles)} outbound={plans[0]['arc'][-1]:.3f}m "
        f"plans={len(plans)} ideal_3fix={frames[-1]['distance_m']:.3f}m")


def _run_live(map_path):
    """Display ROS measurements and the manager's accepted geometry."""
    import rclpy
    from geometry_msgs.msg import PointStamped
    from matplotlib.animation import FuncAnimation
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt

    map_path = Path(map_path).resolve()
    document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    rotation, origin, spawn = _frame_contract(document)
    route = _trailer_route(document, rotation, origin)
    expected_frame = str(document["mission"]["coordinate_frame"])
    data = {"state": "WAITING", "vehicle": None, "cue": None,
            "active_plan": (
                None, np.empty((0, 3)), np.empty((0, 4, 2))),
            "landing": "landing data: waiting"}
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

    def on_active_plan(message):
        try:
            snapshot = _active_plan_marker_snapshot(message, expected_frame)
        except ValueError as exc:
            node.get_logger().warning(
                f"ignored invalid active-plan message: {exc}")
            return
        current_seq = data["active_plan"][0]
        if snapshot[0] is None or current_seq is None \
                or snapshot[0] > current_seq:
            data["active_plan"] = snapshot

    node.create_subscription(
        PointStamped, "/mission/vehicle_position", on_vehicle, 10)
    node.create_subscription(PointStamped, "/marker/cue", on_cue, 10)
    node.create_subscription(
        MarkerArray, "/mission/active_plan_markers",
        on_active_plan, path_qos)
    node.create_subscription(
        String, "/mission/state",
        lambda message: data.__setitem__("state", message.data), 10)
    node.create_subscription(
        String, "/mission/landing_diagnostics",
        lambda message: data.__setitem__("landing", message.data), 10)

    fig, ax, info_ax = _figure_layout(bottom=0.08)
    mission = document["mission"]
    _draw_obstacles(ax, mission)
    vehicle_radius = float(mission["vehicle_clearance_xy_m"])
    planning_clearance = (
        vehicle_radius
        + float(mission.get("bspline_clearance_margin_m", 0.5)))
    sfc_collection = PolyCollection(
        [], facecolor=(0.13, 0.55, 0.90, 0.05),
        edgecolor=(0.09, 0.39, 0.67, 0.40), linewidth=0.7, zorder=1,
        label=f"active-path SFC ({planning_clearance:g} m planning)")
    ax.add_collection(sfc_collection)

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
    drone_radius = _draw_vehicle_radius(
        ax, vehicle_radius, visible=False)
    status = info_ax.text(
        0.0, 1.0, "", transform=info_ax.transAxes, va="top",
        family="monospace", fontsize=9)

    centers = np.asarray(
        [item["center_m"][:2] for item in mission["obstacles"]], float)
    all_points = np.vstack((route, goal[None, :], centers))
    lo, hi = all_points.min(axis=0) - 5.0, all_points.max(axis=0) + 5.0
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]),
           xlabel="map x [m]", ylabel="map y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_title("CJU live mission (read-only ROS view)")
    handles, labels = ax.get_legend_handles_labels()
    info_ax.legend(handles, labels, loc="lower left", fontsize=8,
                   frameon=True, borderaxespad=0.0)

    def update(_frame):
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.0)
        vehicle = data["vehicle"]
        cue = data["cue"]
        plan_seq, path, sfc = data["active_plan"]
        sfc_collection.set_verts(sfc)
        if len(sfc):
            _expand_axes_to_points(ax, sfc)
        if vehicle is not None:
            vehicle_history.append(vehicle.copy())
            mapped = _local_to_map(vehicle, rotation, origin, spawn)[0]
            vehicle_dot.set_data([mapped[0]], [mapped[1]])
            drone_radius.center = mapped[:2]
            drone_radius.set_visible(True)
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
            planned_line.set_data(path[:, 0], path[:, 1])
            _expand_axes_to_points(ax, path[:, :2])
        else:
            planned_line.set_data([], [])
        distance = (float(np.linalg.norm(vehicle[:2] - cue[:2]))
                    if vehicle is not None and cue is not None else float("nan"))
        landing = "\n".join(
            textwrap.fill(part.strip(), width=39)
            for part in data["landing"].split("|") if part.strip())
        status.set_text(
            f"state: {data['state']}\n"
            f"horizontal range: {distance:.1f} m\n"
            f"{landing}\n"
            f"active plan: {plan_seq if plan_seq is not None else 'none'}\n"
            f"active path samples: {len(path)}\n"
            f"active SFC boxes: {len(sfc)}\n"
            "read-only: no flight commands")
        return (planned_line, vehicle_trace, cue_trace, sfc_collection,
                vehicle_dot, cue_dot, drone_radius, status)

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
