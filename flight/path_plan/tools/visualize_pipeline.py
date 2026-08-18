#!/usr/bin/env python3
"""Run the A* -> SFC -> B-spline -> MPC pipeline on the real city map and render
diagnostic figures (saved as PNG).  Dev/diagnostic tool, not installed.

    python3 tools/visualize_pipeline.py [--map PATH] [--config PATH]
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # silence SLSQP bound-clip chatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, Rectangle

from path_plan.astar import AStarPlanner3D
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.mpc_ros import UnicycleMPC, Weights
from path_plan.world_model import WorldModel, _find_buildings

PACKAGE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
DEFAULT_MAP = REPO / "simulation/gazebo/maps/city_coordinates_uav.yaml"
DEFAULT_CONFIG = PACKAGE / "config/city_uav.yaml"
OUT = PACKAGE / "figures"

C_OBST = "#9aa4ad"
C_ASTAR = "#e8590c"
C_SFC = "#1c7ed6"
C_BSPL = "#2f9e44"
C_MPC = "#ae3ec9"

_WORLD_KEYS = (
    "vehicle_clearance_xy_m", "roof_clearance_m", "cruise_floor_m",
    "cruise_ceiling_m", "overfly_allowed",
)


def load_pipeline_config(config_path):
    """Load one ROS parameter file and reject planner geometry drift."""
    doc = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    params = {
        name: doc[f"/{name}"]["ros__parameters"]
        for name in ("astar_planner", "sfc_builder", "bspline_optimizer",
                     "tracking_mpc")
    }
    astar = params["astar_planner"]
    for name in ("sfc_builder", "bspline_optimizer"):
        mismatch = [key for key in _WORLD_KEYS
                    if params[name].get(key) != astar.get(key)]
        if mismatch:
            raise ValueError(
                f"{name} geometry differs from astar_planner: {mismatch}")
    if params["tracking_mpc"].get("vehicle_clearance_xy_m") != astar.get(
            "vehicle_clearance_xy_m"):
        raise ValueError("tracking_mpc vehicle clearance differs from planner")
    return params


def raw_footprints(map_path):
    """Non-inflated building xy bounding boxes + roof height, for display."""
    doc = yaml.safe_load(Path(map_path).read_text())
    out = []
    for b in _find_buildings(doc) or []:
        pts = np.asarray(b["footprint"]["outer"], float)
        lo = pts.min(0)
        hi = pts.max(0)
        out.append((lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1], float(b["roof_z_m"])))
    return out


def mpc_rollout(traj, mpc, start, max_steps=None, goal_tol=2.0):
    """Closed-loop mpc_ros unicycle simulation tracking the B-spline.

    The vehicle perfectly follows the commanded world velocity setpoint each dt
    (kinematic sim): pos += v_world·dt, yaw += ω·dt, speed = |v_xy|.
    """
    _t, tp, _tv, _a = traj.sample(600)
    tt = np.linspace(0.0, traj.duration(), len(tp))
    if max_steps is None:                      # cover the whole trajectory (+30%)
        max_steps = int(1.3 * traj.duration() / mpc.dt)
    pos = np.asarray(start, float).copy()
    d0 = tp[1, :2] - tp[0, :2]
    yaw = float(np.arctan2(d0[1], d0[0]))
    speed = 0.0
    dt, N = mpc.dt, mpc.N
    hist = [pos.copy()]
    for _ in range(max_steps):
        k = int(np.argmin(np.linalg.norm(tp - pos, axis=1)))
        q = tt[k] + dt * np.arange(1, N + 1)
        ref_p = np.column_stack([np.interp(q, tt, tp[:, i]) for i in range(3)])
        res = mpc.solve(pos, yaw, speed, pos[2], ref_p)
        v = res.velocity_world
        pos = pos + v * dt
        yaw = yaw + res.yaw_rate * dt
        speed = float(np.hypot(v[0], v[1]))
        hist.append(pos.copy())
        if np.linalg.norm(pos - tp[-1]) < goal_tol:
            break
    return np.asarray(hist), tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--res", type=float)
    ap.add_argument("--start", nargs=3, type=float)
    ap.add_argument("--goal", nargs=3, type=float)
    ap.add_argument("--waypoints", nargs="*", type=float,
                    help="flat via-points x y z x y z ... (forced slalom route)")
    ap.add_argument("--no-overfly", action="store_true",
                    help="Disable flying over buildings (force lateral avoidance)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    cfg = load_pipeline_config(args.config)
    astar_cfg = cfg["astar_planner"]
    spline_cfg = cfg["bspline_optimizer"]
    mpc_cfg = cfg["tracking_mpc"]
    route_override = args.start is not None or args.goal is not None or args.waypoints is not None
    args.start = list(args.start or astar_cfg["start_enu_m"])
    args.goal = list(args.goal or astar_cfg["goal_enu_m"])
    args.waypoints = list(
        astar_cfg.get("waypoints_enu_m", [])
        if args.waypoints is None else args.waypoints)
    resolution = float(args.res or astar_cfg["resolution_m"])
    floor = float(astar_cfg["cruise_floor_m"])
    ceiling = float(astar_cfg["cruise_ceiling_m"])
    overfly = bool(astar_cfg["overfly_allowed"]) and not args.no_overfly
    args.no_overfly = not overfly
    world = WorldModel.from_city_yaml(
        args.map,
        xy_clearance_m=float(astar_cfg["vehicle_clearance_xy_m"]),
        roof_clearance_m=float(astar_cfg["roof_clearance_m"]),
        ground_clearance_m=floor,
        ceiling_m=ceiling,
        overfly_allowed=overfly)
    foots = raw_footprints(args.map)
    print(f"world: {len(world.boxes_min)} raw obstacles, "
          f"drone radius={world.xy_clearance_m:.2f} m")

    t0 = time.time()
    astar = AStarPlanner3D(
        world,
        resolution_m=resolution,
        max_expanded=int(astar_cfg["max_expanded"]),
        clearance_weight=float(astar_cfg["clearance_weight"]),
        clearance_pref_m=float(astar_cfg["clearance_pref_m"]),
        altitude_weight=float(astar_cfg["altitude_weight"]),
        altitude_pref_m=float(astar_cfg["altitude_pref_m"]),
        climb_weight=float(astar_cfg["climb_weight"]))
    start_10 = [args.start[0], args.start[1],
                float(np.clip(args.start[2], floor, ceiling))]
    goal_10 = [args.goal[0], args.goal[1],
               float(np.clip(args.goal[2], floor, ceiling))]
    
    wp_list = args.waypoints
    route = [start_10] + [wp_list[i:i + 3] for i in range(0, len(wp_list) - 2, 3)] + [goal_10]
    
    res = astar.plan_through(route)
    print(f"A* {time.time()-t0:.1f}s: {res.success} {res.message} "
          f"wpts={len(res.waypoints_m)} (via {len(route)-2}) expanded={res.expanded}")
    if not res.success:
        raise SystemExit("A* failed; pick a reachable goal")
    
    # Force exact continuous coordinates at Z=10 for the B-spline cruise phase
    wp_cruise = res.waypoints_m.copy()
    wp_cruise[0] = start_10
    wp_cruise[-1] = goal_10
    
    # Full visualization waypoints (includes ground level)
    wp = np.vstack(([args.start], wp_cruise, [args.goal]))

    t0 = time.time()
    opt = BsplineOptimizer(
        world,
        cruise_speed_m_s=float(spline_cfg["cruise_speed_m_s"]),
        ctrl_spacing_m=float(spline_cfg["ctrl_spacing_m"]),
        max_vel=float(spline_cfg["max_vel_m_s"]),
        max_acc=float(spline_cfg["max_acc_m_s2"]),
        lambda_smooth=float(spline_cfg["lambda_smooth"]),
        lambda_dist=float(spline_cfg["lambda_dist"]),
        lambda_feas=float(spline_cfg["lambda_feas"]),
        lambda_fit=float(spline_cfg["lambda_fit"]),
        demarcation_m=float(spline_cfg["demarcation_m"]),
        max_rebound=int(spline_cfg["max_rebound"]))
    ores = opt.optimize(wp_cruise)
    if not ores.accepted:
        raise SystemExit(
            "B-spline rejected: "
            f"solver={ores.solver_success}, finite={ores.solution_finite}, "
            f"collision_free={ores.collision_free}")
    corridor = ores.corridor
    traj = ores.spline
    tt, pos, vel, acc = traj.sample(int(spline_cfg["sample_count"]))
    if (not np.all(np.isfinite(np.column_stack((tt, pos, vel, acc))))
            or not all(world.segment_is_free(a, b)
                       for a, b in zip(pos[:-1], pos[1:]))):
        raise SystemExit("B-spline rejected by final exact path check")
    minimum_clearance = min(world.clearance(point) for point in pos)
    boxes_free = all(world.box_is_free(corridor.boxes_min[i], corridor.boxes_max[i])
                     for i in range(len(corridor)))
    if not boxes_free:
        raise SystemExit("B-spline rejected: optimizer SFC contains an occupied box")
    print(f"B-spline optimize {time.time()-t0:.1f}s: boxes={len(corridor)} "
          f"(all free={boxes_free}) dur={traj.duration():.1f}s "
          f"rebound_iters={ores.rebound_iters} "
          f"collision_free={ores.collision_free} free_frac={ores.free_fraction:.3f} "
          f"clearance_residual={minimum_clearance:.3f}m")

    mpc = UnicycleMPC(
        dt_s=float(mpc_cfg["dt_s"]), horizon=int(mpc_cfg["horizon"]),
        v_ref=float(mpc_cfg["v_ref_m_s"]),
        v_max=float(mpc_cfg["v_max_m_s"]),
        omega_max=float(mpc_cfg["omega_max_rad_s"]),
        a_max=float(mpc_cfg["a_max_m_s2"]),
        weights=Weights(
            cte=float(mpc_cfg["w_cte"]), epsi=float(mpc_cfg["w_epsi"]),
            v=float(mpc_cfg["w_v"]), omega=float(mpc_cfg["w_omega"]),
            a=float(mpc_cfg["w_a"]), domega=float(mpc_cfg["w_domega"]),
            da=float(mpc_cfg["w_da"])))
    t0 = time.time()
    actual_cruise, ref_xyz = mpc_rollout(traj, mpc, start_10)
    print(f"MPC rollout {time.time()-t0:.1f}s: {len(actual_cruise)} steps (mpc_ros unicycle)")

    # Construct the full trajectory: Vertical Takeoff -> B-Spline Cruise -> Vertical Landing
    takeoff_z = np.linspace(args.start[2], start_10[2], 50)
    takeoff_pos = np.column_stack((np.full(50, args.start[0]), np.full(50, args.start[1]), takeoff_z))
    
    landing_z = np.linspace(goal_10[2], args.goal[2], 50)
    landing_pos = np.column_stack((np.full(50, args.goal[0]), np.full(50, args.goal[1]), landing_z))
    
    pos_full = np.vstack((takeoff_pos, pos, landing_pos))
    actual_full = np.vstack((takeoff_pos, actual_cruise, landing_pos))

    _fig_3d_map(foots, wp, corridor, pos_full, actual_full, args)
    _fig_topdown(
        foots, wp_cruise, corridor, pos, world, args,
        "CLI route override" if route_override else "config route")
    _fig_altitude(tt, pos_full, args, foots, floor, ceiling)
    _fig_profiles(
        tt, vel, acc, float(spline_cfg["max_vel_m_s"]),
        float(spline_cfg["max_acc_m_s2"]))
    _fig_mpc(traj.sample(6000)[1], actual_cruise)   # true error metric for cruise only
    print(f"figures written to {OUT}")


def _draw_obstacles(ax, foots):
    # foots is [(x, y, w, h, z), ...]
    for x, y, w, h, z in foots:
        ax.bar3d(x, y, 0, w, h, z, color=C_OBST, alpha=0.3, edgecolor="none", shade=True)


def _draw_3d_content(ax, foots, wp, corridor, pos, actual, args):
    _draw_obstacles(ax, foots)
    
    # SFC Corridors (3D Wireframes)
    for lo, hi in zip(corridor.boxes_min, corridor.boxes_max):
        for z in [lo[2], hi[2]]:
            ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], 
                    [lo[1], lo[1], hi[1], hi[1], lo[1]], 
                    [z, z, z, z, z], color=C_SFC, alpha=0.2, lw=1)
        for x, y in zip([lo[0], hi[0], hi[0], lo[0]], [lo[1], lo[1], hi[1], hi[1]]):
            ax.plot([x, x], [y, y], [lo[2], hi[2]], color=C_SFC, alpha=0.2, lw=1)

    # 3D Paths
    ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], "-o", color=C_ASTAR, ms=5, lw=1.2, label="A* waypoints")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-", color=C_BSPL, lw=2.4, label="B-spline path")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "--", color=C_MPC, lw=2.5, label="MPC tracked")
    
    # Markers
    ax.scatter([args.start[0]], [args.start[1]], [args.start[2]], color="#2b8a3e", s=150, marker="^", label="spawn")
    ax.scatter([args.goal[0]], [args.goal[1]], [args.goal[2]], color="#c92a2a", s=200, marker="*", label="goal")
    
    # Fake legend for SFC
    ax.plot([], [], [], color=C_SFC, alpha=0.5, label="SFC corridor (Wireframe)")
    
    ax.set_xlabel("E x [m]"); ax.set_ylabel("N y [m]"); ax.set_zlabel("U z [m]")
    
    # Setup view box dimensions
    m = 20
    mid_x = (args.start[0] + args.goal[0]) / 2
    mid_y = (args.start[1] + args.goal[1]) / 2
    span = max(abs(args.start[0] - args.goal[0]), abs(args.start[1] - args.goal[1])) / 2 + m
    ax.set_xlim(mid_x - span, mid_x + span)
    ax.set_ylim(mid_y - span, mid_y + span)
    ax.set_zlim(0, 50)
    ax.set_box_aspect([1, 1, 0.4])  # Make Z axis slightly squashed for better visibility


def _fig_3d_map(foots, wp, corridor, pos, actual, args):
    fig = plt.figure(figsize=(20, 9))
    
    mode = "Overfly Buildings" if not args.no_overfly else "Avoid Buildings Laterally"
    fig.suptitle(f"Global pipeline (3D View): {mode}\nA* → SFC → B-spline → MPC", fontsize=16)

    # 1. Perspective View
    ax1 = fig.add_subplot(121, projection='3d')
    _draw_3d_content(ax1, foots, wp, corridor, pos, actual, args)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_title("Perspective View (측면 상단 뷰)", fontsize=14)
    ax1.legend(loc="upper right", framealpha=0.9)

    # 2. Side View
    ax2 = fig.add_subplot(122, projection='3d')
    _draw_3d_content(ax2, foots, wp, corridor, pos, actual, args)
    ax2.view_init(elev=5, azim=-90)
    ax2.set_title("Side View (완전 측면 고도 뷰)", fontsize=14)
    
    fig.tight_layout(); fig.savefig(OUT / "1_global_3d.png", dpi=130)
    plt.close(fig)


def _fig_topdown(foots, wp, corridor, pos, world, args, route_source):
    """Raw map + the optimizer-owned SFC + accepted path + drone radius."""
    closest = int(np.argmin([world.clearance(p) for p in pos]))
    centre = pos[closest, :2]
    residual = float(world.clearance(pos[closest]))

    def draw(ax):
        for index, (x, y, w, h, _z) in enumerate(foots):
            ax.add_patch(Rectangle(
                (x, y), w, h, facecolor=C_OBST, edgecolor="none", alpha=0.35,
                label="raw planner AABBs" if index == 0 else None))
        for index, (lo, hi) in enumerate(zip(corridor.boxes_min, corridor.boxes_max)):
            ax.add_patch(Rectangle(
                lo[:2], *(hi[:2] - lo[:2]), facecolor=C_SFC,
                edgecolor=C_SFC, alpha=0.10, lw=0.6,
                label="optimizer-owned SFC" if index == 0 else None))
        ax.plot(wp[:, 0], wp[:, 1], "-o", color=C_ASTAR, ms=2.5,
                lw=0.8, label="A* waypoints")
        ax.plot(pos[:, 0], pos[:, 1], color=C_BSPL, lw=2.0,
                label="accepted B-spline")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("E x [m]")
        ax.set_ylabel("N y [m]")

    fig, (full, zoom) = plt.subplots(1, 2, figsize=(18, 8))
    draw(full)
    full.set_title("Raw planner AABBs + optimizer-owned SFC")
    full.legend(loc="best")

    draw(zoom)
    radius = float(world.xy_clearance_m)
    zoom.add_patch(Circle(
        centre, radius, facecolor=C_SFC, edgecolor="#1864ab",
        alpha=0.30, lw=2.0, label=f"drone safety radius {radius:.2f} m"))
    span = max(10.0, 5.0 * radius)
    zoom.set_xlim(centre[0] - span, centre[0] + span)
    zoom.set_ylim(centre[1] - span, centre[1] + span)
    zoom.set_title(
        f"Closest-clearance detail: residual {residual:.2f} m "
        "(no obstacle-side halo)")
    zoom.legend(loc="best")
    config_name = Path(args.config).name
    start_xy = tuple(round(float(value), 1) for value in args.start[:2])
    goal_xy = tuple(round(float(value), 1) for value in args.goal[:2])
    fig.suptitle(
        "A* → drone-centre SFC → verified B-spline\n"
        f"{route_source}; config={config_name}; start={start_xy}; goal={goal_xy}",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "1_global_topdown.png", dpi=130)
    plt.close(fig)


def _fig_altitude(tt, pos, args, foots, floor, ceiling):
    dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
    fig, ax = plt.subplots(figsize=(11, 4))
    
    # Draw building cross-sections (obstacles the drone flies over)
    for x, y, w, h, z in foots:
        # Check which path points are above this building's footprint
        inside = (pos[:, 0] >= x) & (pos[:, 0] <= x + w) & (pos[:, 1] >= y) & (pos[:, 1] <= y + h)
        if np.any(inside):
            idx = np.where(inside)[0]
            # Fill a gray block from start_dist to end_dist, up to building roof height
            ax.fill_between([dist[idx[0]], dist[idx[-1]]], [0, 0], [z, z], 
                            color=C_OBST, alpha=0.6, zorder=1)

    ax.axhspan(floor, ceiling, color=C_SFC, alpha=0.12,
               label=f"{floor:g}–{ceiling:g} m cruise band")
    ax.plot(dist, pos[:, 2], color=C_BSPL, lw=2.2, label="B-spline altitude", zorder=2)
    ax.set_xlabel("path distance [m]"); ax.set_ylabel("altitude z [m]")
    ax.set_ylim(0, 40)
    mode = "(overfly allowed)" if not args.no_overfly else "(no overfly)"
    ax.set_title(f"Altitude Profile: Takeoff -> Cruise (10–20m) -> Landing {mode}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "2_altitude.png", dpi=130)
    plt.close(fig)


def _fig_profiles(tt, vel, acc, max_vel, max_acc):
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(acc, axis=1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(tt, speed, color=C_BSPL, lw=2)
    a1.axhline(max_vel, ls="--", color="#c92a2a",
               label=f"v_max {max_vel:g} m/s")
    a1.set_ylabel("speed [m/s]"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title("B-spline speed & acceleration profiles")
    a2.plot(tt, accel, color=C_ASTAR, lw=2)
    a2.axhline(max_acc, ls="--", color="#c92a2a",
               label=f"a_max {max_acc:g} m/s²")
    a2.set_ylabel("|accel| [m/s²]"); a2.set_xlabel("time [s]")
    a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "3_profiles.png", dpi=130)
    plt.close(fig)


def _fig_mpc(ref_xyz, actual):
    # true cross-track error = distance to the nearest point on the DENSE
    # reference (a sparse reference would sawtooth ~half the sample spacing).
    from scipy.spatial import cKDTree
    err, _ = cKDTree(ref_xyz[:, :2]).query(actual[:, :2])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    a1.plot(ref_xyz[:, 0], ref_xyz[:, 1], "-", color=C_BSPL, lw=3, alpha=0.6,
            label="reference (B-spline)")
    a1.plot(actual[:, 0], actual[:, 1], "--", color=C_MPC, lw=2, label="MPC tracked")
    a1.set_aspect("equal"); a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]")
    a1.set_title("mpc_ros unicycle MPC — closed-loop tracking (top-down)")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(err, color=C_MPC, lw=1.8)
    a2.set_xlabel("MPC step"); a2.set_ylabel("lateral tracking error [m]")
    a2.set_title(f"Tracking error (mean {err.mean():.2f} m, max {err.max():.2f} m)")
    a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "4_mpc_tracking.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
