#!/usr/bin/env python3
"""Run the A* -> SFC -> B-spline -> MPC pipeline on the real city map and render
diagnostic figures (saved as PNG).  Dev/diagnostic tool, not installed.

    python3 tools/visualize_pipeline.py [--map PATH] [--res 4.0]
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
from matplotlib.patches import Rectangle

from path_plan.astar import AStarPlanner3D
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.mpc_ros import UnicycleMPC, Weights
from path_plan.world_model import WorldModel, _find_buildings

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "gazebo/maps/city_coordinates_uav.yaml"
OUT = Path(__file__).resolve().parents[1] / "figures"

C_OBST = "#9aa4ad"
C_ASTAR = "#e8590c"
C_SFC = "#1c7ed6"
C_BSPL = "#2f9e44"
C_MPC = "#ae3ec9"


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
    ap.add_argument("--res", type=float, default=4.0)
    ap.add_argument("--start", nargs=3, type=float, default=[587, 580, 25])
    ap.add_argument("--goal", nargs=3, type=float, default=[-300, -300, 25])
    ap.add_argument("--waypoints", nargs="*", type=float, default=[],
                    help="flat via-points x y z x y z ... (forced slalom route)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    world = WorldModel.from_city_yaml(args.map, inflation_xy_m=1.0,
                                      ground_clearance_m=20.0, ceiling_m=30.0,
                                      overfly_allowed=False)
    foots = raw_footprints(args.map)
    print(f"world: {len(world.boxes_min)} obstacles")

    t0 = time.time()
    astar = AStarPlanner3D(world, resolution_m=args.res)
    wp_list = args.waypoints
    route = [args.start] + [wp_list[i:i + 3] for i in range(0, len(wp_list) - 2, 3)] \
        + [args.goal]
    res = astar.plan_through(route)
    print(f"A* {time.time()-t0:.1f}s: {res.success} {res.message} "
          f"wpts={len(res.waypoints_m)} (via {len(route)-2}) expanded={res.expanded}")
    if not res.success:
        raise SystemExit("A* failed; pick a reachable goal")
    wp = res.waypoints_m

    t0 = time.time()
    opt = BsplineOptimizer(world, cruise_speed_m_s=4.0, ctrl_spacing_m=5.0,
                           max_vel=5.0, max_acc=3.0, lambda_feas=2.0)
    ores = opt.optimize(wp)
    corridor = ores.corridor
    traj = ores.spline
    tt, pos, vel, acc = traj.sample(400)
    boxes_free = all(world.box_is_free(corridor.boxes_min[i], corridor.boxes_max[i])
                     for i in range(len(corridor)))
    print(f"B-spline optimize {time.time()-t0:.1f}s: boxes={len(corridor)} "
          f"(all free={boxes_free}) dur={traj.duration():.1f}s "
          f"rebound_iters={ores.rebound_iters} "
          f"collision_free={ores.collision_free} free_frac={ores.free_fraction:.3f}")

    mpc = UnicycleMPC(dt_s=0.1, horizon=20, v_ref=4.0, v_max=5.0,
                      weights=Weights(cte=6.0, epsi=4.0, v=1.0, omega=1.0,
                                      a=0.05, domega=6.0, da=0.5))
    t0 = time.time()
    actual, ref_xyz = mpc_rollout(traj, mpc, args.start)
    print(f"MPC rollout {time.time()-t0:.1f}s: {len(actual)} steps (mpc_ros unicycle)")

    _fig_topdown(foots, wp, corridor, pos, args)
    _fig_altitude(tt, pos)
    _fig_profiles(tt, vel, acc)
    _fig_mpc(traj.sample(6000)[1], actual)   # dense reference for a true error metric
    print(f"figures written to {OUT}")


def _draw_obstacles(ax, foots):
    for x, y, w, h, z in foots:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=C_OBST, edgecolor="none",
                               alpha=0.55))


def _fig_topdown(foots, wp, corridor, pos, args):
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_obstacles(ax, foots)
    for lo, hi in zip(corridor.boxes_min, corridor.boxes_max):
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                               facecolor=C_SFC, edgecolor=C_SFC, alpha=0.12, lw=1.2))
    ax.plot(wp[:, 0], wp[:, 1], "-o", color=C_ASTAR, ms=6, lw=1.6,
            label="A* waypoints", zorder=5)
    ax.plot(pos[:, 0], pos[:, 1], "-", color=C_BSPL, lw=2.4,
            label="B-spline path", zorder=6)
    ax.plot(*args.start[:2], "^", color="#2b8a3e", ms=15, label="spawn", zorder=7)
    ax.plot(*args.goal[:2], "*", color="#c92a2a", ms=20, label="goal (trailer)", zorder=7)
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=C_SFC, alpha=0.3,
                           label="SFC corridor"))
    ax.set_aspect("equal")
    ax.set_xlabel("E  x [m]"); ax.set_ylabel("N  y [m]")
    ax.set_title("Global pipeline (top-down): A* → SFC → B-spline\n"
                 "grey = building footprints — all avoided laterally (1 m wall clearance)",
                 fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    m = 20
    ax.set_xlim(min(args.start[0], args.goal[0]) - m, max(args.start[0], args.goal[0]) + m)
    ax.set_ylim(min(args.start[1], args.goal[1]) - m, max(args.start[1], args.goal[1]) + m)
    fig.tight_layout(); fig.savefig(OUT / "1_global_topdown.png", dpi=130)
    plt.close(fig)


def _fig_altitude(tt, pos):
    dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhspan(20, 30, color=C_SFC, alpha=0.12, label="20–30 m cruise band")
    ax.plot(dist, pos[:, 2], color=C_BSPL, lw=2.2, label="B-spline altitude")
    ax.set_xlabel("path distance [m]"); ax.set_ylabel("altitude z [m]")
    ax.set_ylim(0, 35)
    ax.set_title("Altitude stays inside the 20–30 m band")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "2_altitude.png", dpi=130)
    plt.close(fig)


def _fig_profiles(tt, vel, acc):
    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(acc, axis=1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(tt, speed, color=C_BSPL, lw=2)
    a1.axhline(5.0, ls="--", color="#c92a2a", label="v_max 5 m/s")
    a1.set_ylabel("speed [m/s]"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title("B-spline speed & acceleration profiles")
    a2.plot(tt, accel, color=C_ASTAR, lw=2)
    a2.axhline(3.0, ls="--", color="#c92a2a", label="a_max 3 m/s²")
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
