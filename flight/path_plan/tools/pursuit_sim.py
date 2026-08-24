#!/usr/bin/env python3
"""Moving-trailer pursuit simulation.

A trailer drives a square loop around the map perimeter; the drone starts at the
spawn and chases it with **real-time path generation** (periodic A*->SFC->B-spline
replans to the trailer's current position) plus **mpc_ros unicycle tracking**.
When the drone's xy coordinates overlap the trailer (< capture_radius) the run
ends in success.  Everything is logged to CSV for offline / paper analysis, and a
matplotlib summary figure (and optional live animation) is produced.

    python3 tools/pursuit_sim.py                       # run + CSV + summary PNG
    python3 tools/pursuit_sim.py --animate             # live matplotlib window
    python3 tools/pursuit_sim.py --gif pursuit.gif     # save an animation

Scenario (loop geometry, speeds, cadence) lives in the SEPARATE map file
gazebo/maps/city_uav_trailer_loop.yaml — the base map is never modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import yaml

# Allow running directly from tools/ (add the package root so `path_plan` imports).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path_plan.astar import AStarPlanner3D
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.mpc_ros import UnicycleMPC, Weights
from path_plan.world_model import WorldModel, _find_buildings

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_SCENARIO = REPO / "simulation/gazebo/maps/city_uav_trailer_loop.yaml"
_CORNER_S = {"SE": 0.0, "NE": 2.0, "NW": 4.0, "SW": 6.0}   # x H units along the loop

# Make every figure's text bold (titles, axis labels, tick labels, legend).
matplotlib.rcParams.update({
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
})

# Shared visual palette (kept consistent with tools/visualize_pipeline.py so the
# pursuit figures read the same as the offline pipeline figures).
C_OBST = "#9aa4ad"    # buildings
C_SFC = "#1c7ed6"     # SFC corridor (filled, faint)
C_ASTAR = "#f59f00"   # A* waypoints
C_BSPL = "#2f9e44"    # B-spline reference path
C_MPC = "#ae3ec9"     # MPC receding-horizon predictions
C_FINAL = "#1971c2"   # final executed drone path
C_TRAIL = "#e8590c"   # trailer (loop + actual path)


def raw_footprints(map_path):
    """Non-inflated building xy bounding boxes + roof height, for 3D display.

    Mirrors visualize_pipeline.raw_footprints: returns [(x, y, w, h, z), ...]
    using the true (un-inflated, non-pillared) roof height for a clean render.
    """
    doc = yaml.safe_load(Path(map_path).read_text())
    out = []
    for b in _find_buildings(doc) or []:
        pts = np.asarray(b["footprint"]["outer"], float)
        lo = pts.min(0)
        hi = pts.max(0)
        out.append((lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1], float(b["roof_z_m"])))
    return out


def square_loop_pos(s: float, half: float) -> np.ndarray:
    """Position at perimeter arc-length ``s`` on a CCW square of half-size ``half``."""
    p = s % (8.0 * half)
    if p < 2 * half:
        return np.array([half, -half + p])
    if p < 4 * half:
        return np.array([half - (p - 2 * half), half])
    if p < 6 * half:
        return np.array([-half, half - (p - 4 * half)])
    return np.array([-half + (p - 6 * half), -half])


class Trailer:
    def __init__(self, cfg: dict):
        self.half = float(cfg["half_size_m"])
        self.speed = float(cfg["speed_m_s"])
        self.sign = 1.0 if str(cfg.get("direction", "ccw")).lower() == "ccw" else -1.0
        self.s0 = _CORNER_S[str(cfg.get("start_corner", "SW")).upper()] * self.half

    def pos(self, t: float) -> np.ndarray:
        return square_loop_pos(self.s0 + self.sign * self.speed * t, self.half)


def _sample_spline(spline, n=400):
    """Dense (positions, times) — computed ONCE per replan and cached."""
    _, tp, _, _ = spline.sample(n)
    return tp, np.linspace(0.0, spline.duration(), len(tp))


def _horizon_from_samples(tp, tt, drone_pos, N, dt):
    """N reference points at dt spacing ahead of the drone (from cached samples)."""
    k = int(np.argmin(np.linalg.norm(tp - drone_pos, axis=1)))
    q = tt[k] + dt * np.arange(1, N + 1)
    return np.column_stack([np.interp(q, tt, tp[:, i]) for i in range(3)])


def run_sim(scenario: dict, res_override=None, verbose=True):
    base = REPO / scenario["base_map"]
    d = scenario["drone"]
    tr = scenario["trailer"]
    pu = scenario["pursuit"]
    floor, ceil = d["cruise_floor_m"], d["cruise_ceiling_m"]
    cruise_z = float(d.get("cruise_altitude_m", 0.5 * (floor + ceil)))
    res = float(res_override or pu["astar_resolution_m"])

    world = WorldModel.from_city_yaml(
        base, xy_clearance_m=d["vehicle_clearance_xy_m"],
        ground_clearance_m=floor,
        ceiling_m=ceil, overfly_allowed=d["overfly_allowed"])
    planner = AStarPlanner3D(world, resolution_m=res)
    speed_limit = float(d.get("max_speed_m_s", d["cruise_speed_m_s"]))
    optimizer = BsplineOptimizer(world, cruise_speed_m_s=None,
                                 ctrl_spacing_m=float(pu.get("bspline_control_spacing_m", 8.0)),
                                 max_vel=speed_limit,
                                 lambda_feas=2.0)
    mpc = UnicycleMPC(dt_s=pu["sim_dt_s"],
                      horizon=int(pu.get("mpc_horizon", 15)),
                      v_ref=float(pu.get(
                          "mpc_reference_speed_m_s", d["cruise_speed_m_s"])),
                      v_max=speed_limit,
                      max_iter=int(pu.get("mpc_max_iter", 20)),
                      weights=Weights(cte=6.0, epsi=4.0, v=1.0, omega=1.0,
                                      a=0.05, domega=6.0, da=0.5))
    trailer = Trailer(tr)

    dt = float(pu["sim_dt_s"])
    cap = float(pu["capture_radius_m"])
    homing = float(pu["terminal_homing_m"])
    replan_period = float(pu["replan_period_s"])
    t_max = float(pu["max_sim_time_s"])

    drone = np.array([*d["spawn_enu_m"][:2], cruise_z], float)
    t0 = trailer.pos(0.0)
    yaw = float(np.arctan2(t0[1] - drone[1], t0[0] - drone[0]))
    speed = 0.0
    spline = None
    sp_tp = sp_tt = None          # cached dense samples of the current spline
    last_replan = -1e9
    replan_count = 0
    log = []
    captured = False
    wall0 = time.time()

    t = 0.0
    step_idx = 0
    splines_log = []
    mpc_log = []
    mpc_times = []
    plan_stats = []
    initial_target = trailer.pos(t)
    initial_distance = float(np.linalg.norm(drone[:2] - initial_target))
    log.append(_row(
        t, drone, speed, 0.0, yaw, 0.0, initial_target, trailer.speed,
        initial_distance, replan_count, "pursuit", False, math.nan, math.nan,
        math.nan, False, math.nan, math.nan))
    while t < t_max - 1.0e-9:
        tpos = trailer.pos(t)
        dist = float(np.linalg.norm(drone[:2] - tpos))
        
        if verbose and step_idx % 20 == 0:
            real_t = time.time() - wall0
            print(f"Simulating: SimTime={t:.1f}s / {t_max:.1f}s | RealTime={real_t:.1f}s | dist={dist:.2f}m")
        step_idx += 1

        phase = "homing" if dist <= homing else "pursuit"
        # real-time global replan toward the trailer's current position
        if phase == "pursuit" and (spline is None or t - last_replan >= replan_period):
            plan_wall0 = time.perf_counter()
            astar_wall0 = plan_wall0
            leg = planner.plan(drone, np.array([tpos[0], tpos[1], cruise_z]))
            astar_ms = 1000.0 * (time.perf_counter() - astar_wall0)
            bspline_ms = math.nan
            sfc_ms = math.nan
            accepted = False
            if leg.success and len(leg.waypoints_m) >= 2:
                guide = leg.waypoints_m.copy()
                guide[:, 2] = cruise_z
                guide[0], guide[-1] = drone, [tpos[0], tpos[1], cruise_z]
                bspline_wall0 = time.perf_counter()
                opt = optimizer.optimize(guide)
                bspline_ms = 1000.0 * (time.perf_counter() - bspline_wall0)
                sfc_ms = opt.sfc_generation_time_ms
                if opt.accepted:
                    spline = opt.spline
                    sp_tp, sp_tt = _sample_spline(spline)
                    _, dense_tp, _, _ = spline.sample(4000)
                    splines_log.append((t, np.copy(sp_tp), opt.corridor,
                                        np.copy(guide)))
                    replan_count += 1
                    accepted = True
            plan_stats.append({
                "attempt_index": len(plan_stats) + 1,
                "simulation_time_s": round(t, 3),
                "accepted": int(accepted),
                "astar_success": int(bool(leg.success)),
                "astar_expanded": int(leg.expanded),
                "astar_waypoints": int(len(leg.waypoints_m)),
                "astar_solve_ms": astar_ms,
                "bspline_solve_ms": bspline_ms,
                "bspline_solve_scope": "optimizer_total_including_sfc",
                "sfc_generation_time_ms": sfc_ms,
                "total_plan_ms": 1000.0 * (time.perf_counter() - plan_wall0),
            })
            last_replan = t

        # MPC reference: home straight at the trailer when close, else track spline
        if phase == "homing" or spline is None:
            goal = np.array([tpos[0], tpos[1], cruise_z])
            ref = np.linspace(drone, goal, mpc.N + 1)[1:]
        else:
            ref = _horizon_from_samples(sp_tp, sp_tt, drone, mpc.N, dt)

        mpc_wall0 = time.perf_counter()
        out = mpc.solve(drone, yaw, speed, drone[2], ref)
        mpc_solve_ms = 1000.0 * (time.perf_counter() - mpc_wall0)
        mpc_log.append(np.copy(out.predicted_xy))
        mpc_times.append(t)
        v = out.velocity_world
        yaw_rate = out.yaw_rate
        new_speed = float(np.hypot(v[0], v[1]))
        accel = (new_speed - speed) / dt
        
        drone = drone + v * dt
        yaw += yaw_rate * dt
        speed = new_speed

        if phase == "homing" or spline is None:
            track_err = math.nan
            track_err_dense = math.nan
        else:
            track_err = float(np.min(np.linalg.norm(sp_tp[:, :2] - drone[:2], axis=1)))
            track_err_dense = float(np.min(np.linalg.norm(dense_tp[:, :2] - drone[:2], axis=1)))

        t_next = t + dt
        tpos_next = trailer.pos(t_next)
        dist_next = float(np.linalg.norm(drone[:2] - tpos_next))
        captured = dist_next < cap
        row_phase = "capture" if captured else phase
        log.append(_row(
            t_next, drone, speed, accel, yaw, yaw_rate, tpos_next,
            trailer.speed, dist_next, replan_count, row_phase, captured,
            track_err if not captured else math.nan,
            track_err_dense if not captured else math.nan,
            mpc_solve_ms, out.success, out.cte0, out.epsi0))
        t = t_next
        if captured:
            break

    if verbose:
        print(f"\nsim {'CAPTURED' if captured else 'timed out'} at t={t:.1f}s "
              f"(dist={log[-1]['dist_xy_m']:.2f} m), replans={replan_count}, "
              f"steps={len(log)}, wall={time.time()-wall0:.1f}s")
    return (log, captured, world, trailer, splines_log, mpc_log,
            plan_stats, mpc_times)


def _row(t, drone, speed, accel, yaw, yaw_rate, tpos, tspeed, dist,
         replans, phase, captured, track_err=math.nan,
         track_err_dense=math.nan, mpc_solve_ms=math.nan, mpc_success=False,
         mpc_cte=math.nan, mpc_epsi=math.nan):
    return dict(t_s=round(t, 3), drone_x=round(float(drone[0]), 3),
                drone_y=round(float(drone[1]), 3), drone_z=round(float(drone[2]), 3),
                drone_speed_mps=round(speed, 4), drone_accel_mps2=round(accel, 4),
                drone_yaw_rad=round(yaw, 5), drone_yawrate_rads=round(yaw_rate, 4),
                trailer_x=round(float(tpos[0]), 3), trailer_y=round(float(tpos[1]), 3),
                trailer_speed_mps=round(float(tspeed), 4),
                dist_xy_m=round(float(dist), 4), replan_count=replans,
                phase=phase, captured=int(captured), track_err_m=round(track_err, 4),
                track_err_dense_m=round(track_err_dense, 4),
                mpc_solve_ms=round(mpc_solve_ms, 6),
                mpc_success=int(bool(mpc_success)),
                mpc_cte_m=round(mpc_cte, 6),
                mpc_epsi_rad=round(mpc_epsi, 6))


def save_csv(log, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print(f"CSV written: {path}  ({len(log)} rows)")


def _fig_topdown(log, world, trailer, splines_log, mpc_log, out_path,
                 no_final_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    dx = np.array([r["drone_x"] for r in log]); dy = np.array([r["drone_y"] for r in log])
    tx = np.array([r["trailer_x"] for r in log]); ty = np.array([r["trailer_y"] for r in log])
    captured = log[-1]["captured"]

    fig_map = plt.figure(figsize=(11, 9))
    ax = fig_map.add_subplot(1, 1, 1)

    # 0. Buildings
    for lo, hi in zip(world.boxes_min, world.boxes_max):
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1],
                               facecolor=C_OBST, edgecolor="none", alpha=0.5, zorder=1))

    # Trailer full loop
    s = np.linspace(0, 8*trailer.half, 400)
    loop = np.array([square_loop_pos(v, trailer.half) for v in s])
    ax.plot(loop[:, 0], loop[:, 1], ":", color=C_TRAIL, lw=1.5, label="trailer loop", zorder=1)

    # 1. Every dynamic A*/B-spline replan.  SFC boxes are intentionally left
    # out of this comparison figure so they cannot be mistaken for a fifth
    # path (the dedicated 3-D/SFC figures below still retain them).
    for i, (t_gen, sp_tp, corridor, wp) in enumerate(splines_log):
        # A* waypoints
        ax.plot(wp[:, 0], wp[:, 1], "-o", color=C_ASTAR, ms=5, lw=1.2, alpha=0.7,
                label="A*" if i == 0 else "", zorder=3)

        # B-spline reference
        ax.plot(sp_tp[:, 0], sp_tp[:, 1], "-", color=C_BSPL, alpha=0.85, lw=2.4,
                label="B-spline" if i == 0 else "", zorder=4)

    # 2. Receding MPC horizons (sampled only for legibility).
    horizon_stride = max(1, len(mpc_log) // 100)
    for index in range(0, len(mpc_log), horizon_stride):
        horizon = np.asarray(mpc_log[index], float)
        ax.plot(horizon[:, 0], horizon[:, 1], "--", color=C_MPC, lw=1.4,
                alpha=0.58, label="MPC" if index == 0 else "", zorder=7)

    # 3. Applied closed-loop trajectory: the study's final path.
    final_line, = ax.plot(dx, dy, "-", color=C_FINAL, lw=2.8,
                          label="Final path", zorder=6)
    ax.plot(tx, ty, "-", color=C_TRAIL, lw=2.5, label="trailer path", zorder=5)

    # 5. Start/End Markers
    ax.plot(dx[0], dy[0], "^", color="#2b8a3e", ms=15, label="drone start", zorder=6)
    ax.plot(tx[-1], ty[-1], "*", color="#c92a2a", ms=20,
            label="capture" if captured else "trailer end", zorder=6)
            
    ax.set_aspect("equal"); ax.set_xlabel("E x [m]"); ax.set_ylabel("N y [m]")
    fig_map.suptitle("Gazebo filght planned", fontsize=18, y=0.98)
    ax.set_title(
        f"City YAML dynamic pursuit | drone {max(r['drone_speed_mps'] for r in log):.1f} m/s max | "
        f"trailer {trailer.speed:.1f} m/s | "
        f"{'CAPTURED' if captured else 'timeout'}",
        fontsize=10, pad=10)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    
    fig_map.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig_map.savefig(out_path, dpi=130)
    if no_final_path is not None:
        final_line.set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        visible = [(handle, label) for handle, label in zip(handles, labels)
                   if label != "Final path"]
        ax.legend(*zip(*visible), loc="upper left", framealpha=0.9,
                  fontsize=10)
        fig_map.savefig(no_final_path, dpi=130)
    plt.close(fig_map)
    print(f"Figure 5 (Map): {out_path}")
    if no_final_path is not None:
        print(f"Figure 5 (Map, no final path): {no_final_path}")

def _fig_profiles(log, out_path, band=None):
    import matplotlib.pyplot as plt
    t = np.array([r["t_s"] for r in log])
    spd = np.array([r["drone_speed_mps"] for r in log])
    acc = np.array([r["drone_accel_mps2"] for r in log])
    z = np.array([r["drone_z"] for r in log])

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    a1.plot(t, spd, color=C_BSPL, lw=2)
    a1.set_ylabel("speed [m/s]"); a1.grid(alpha=0.3)
    a1.set_title("Drone speed, acceleration & altitude profiles")
    a2.plot(t, np.abs(acc), color=C_ASTAR, lw=2)
    a2.set_ylabel("|accel| [m/s²]"); a2.grid(alpha=0.3)

    # Flight altitude vs time, with the cruise band shaded (same look as the pipeline)
    if band is not None:
        floor, ceil = band
        a3.axhspan(floor, ceil, color=C_SFC, alpha=0.12,
                   label=f"{floor:.0f}–{ceil:.0f} m cruise band")
    a3.plot(t, z, color=C_MPC, lw=2, label="flight altitude")
    a3.set_ylabel("altitude z [m]"); a3.set_xlabel("time [s]")
    a3.set_ylim(0, max(25.0, float(z.max()) + 5.0))
    a3.legend(loc="lower right"); a3.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Figure 6 (Profiles): {out_path}")


def _draw_pursuit_3d(ax, foots, splines_log, dx, dy, dz, tx, ty, trailer, captured):
    """Shared 3D content for the dual-view pursuit figure (mirrors
    visualize_pipeline._draw_3d_content, adapted to the moving-trailer chase)."""
    # Buildings as solid prisms (true roof heights)
    for x, y, w, h, z in foots:
        ax.bar3d(x, y, 0, w, h, z, color=C_OBST, alpha=0.3, edgecolor="none", shade=True)

    # SFC corridors (3D wireframe boxes) + B-spline references (z varies 10-20 m)
    ax.plot([], [], [], color=C_SFC, alpha=0.5, label="SFC corridor (wireframe)")
    for i, (t_gen, sp_tp, corridor, wp) in enumerate(splines_log):
        for lo, hi in zip(corridor.boxes_min, corridor.boxes_max):
            for z in [lo[2], hi[2]]:
                ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]],
                        [lo[1], lo[1], hi[1], hi[1], lo[1]],
                        [z, z, z, z, z], color=C_SFC, alpha=0.2, lw=1)
            for x, y in zip([lo[0], hi[0], hi[0], lo[0]], [lo[1], lo[1], hi[1], hi[1]]):
                ax.plot([x, x], [y, y], [lo[2], hi[2]], color=C_SFC, alpha=0.2, lw=1)
        ax.plot(sp_tp[:, 0], sp_tp[:, 1], sp_tp[:, 2], "-", color=C_BSPL, lw=2.0,
                alpha=0.85, label="B-spline path" if i == 0 else "")

    # Actual drone path (real logged altitude — climbs over buildings) and the
    # trailer path (on the ground)
    ax.plot(dx, dy, dz, "--", color=C_MPC, lw=2.5, label="MPC tracked")
    ax.plot(tx, ty, np.zeros_like(tx), "-", color=C_TRAIL, lw=2.0, label="trailer path")

    # Trailer full loop on the ground
    s = np.linspace(0, 8*trailer.half, 400)
    loop = np.array([square_loop_pos(v, trailer.half) for v in s])
    ax.plot(loop[:, 0], loop[:, 1], np.zeros(len(loop)), ":", color=C_TRAIL, lw=1.2)

    # Markers
    ax.scatter([dx[0]], [dy[0]], [dz[0]], color="#2b8a3e", s=150, marker="^", label="drone start")
    ax.scatter([tx[-1]], [ty[-1]], [0.0], color="#c92a2a", s=200, marker="*",
               label="capture" if captured else "trailer end")

    ax.set_xlabel("E x [m]"); ax.set_ylabel("N y [m]"); ax.set_zlabel("U z [m]")
    span = trailer.half + 40
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(0, 50)
    ax.set_box_aspect([1, 1, 0.4])


def _fig_3d(log, foots, splines_log, trailer, out_path):
    import matplotlib.pyplot as plt
    dx = np.array([r["drone_x"] for r in log]); dy = np.array([r["drone_y"] for r in log])
    dz = np.array([r["drone_z"] for r in log])
    tx = np.array([r["trailer_x"] for r in log]); ty = np.array([r["trailer_y"] for r in log])
    captured = log[-1]["captured"]

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("Moving-trailer pursuit (3D View)\nA* → SFC → B-spline → MPC",
                 fontsize=16)

    ax1 = fig.add_subplot(121, projection="3d")
    _draw_pursuit_3d(ax1, foots, splines_log, dx, dy, dz, tx, ty, trailer, captured)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_title("Perspective View", fontsize=14)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2 = fig.add_subplot(122, projection="3d")
    _draw_pursuit_3d(ax2, foots, splines_log, dx, dy, dz, tx, ty, trailer, captured)
    ax2.view_init(elev=5, azim=-90)
    ax2.set_title("Side View", fontsize=14)

    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Figure 8 (3D): {out_path}")

def _fig_mpc(log, splines_log, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    t = np.array([r["t_s"] for r in log])
    dx = np.array([r["drone_x"] for r in log])
    dy = np.array([r["drone_y"] for r in log])
    valid = np.array([
        r["phase"] == "pursuit" and np.isfinite(r["track_err_dense_m"])
        for r in log])
    t_error = t[valid]
    err = np.array([r["track_err_m"] for r in log])[valid]
    err_dense = np.array([r["track_err_dense_m"] for r in log])[valid]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    # SFC corridor boxes under the tracked path (so the corridor the drone flew
    # through is visible on the trajectory, not just in the map figure).
    a1.plot([], [], "-", color=C_SFC, alpha=0.6, lw=2, label="SFC corridor")
    for i, (t_gen, sp_tp, corridor, wp) in enumerate(splines_log):
        for lo, hi in zip(corridor.boxes_min, corridor.boxes_max):
            a1.add_patch(Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1],
                                   facecolor=C_SFC, edgecolor=C_SFC,
                                   alpha=0.12, lw=0.8, zorder=1))
        a1.plot(sp_tp[:, 0], sp_tp[:, 1], "-", color=C_BSPL, lw=3, alpha=0.6,
                label="reference (B-spline)" if i == 0 else "", zorder=2)
    a1.plot(dx, dy, "--", color=C_MPC, lw=2, label="MPC tracked", zorder=3)
    a1.set_aspect("equal"); a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]")
    a1.set_title("mpc_ros unicycle MPC — closed-loop tracking (top-down)")
    a1.legend(); a1.grid(alpha=0.3)

    a2.plot(t_error, err, color="#b2f252", lw=1.5, alpha=0.6,
            label="Raw (Sawtooth artifact)")
    a2.plot(t_error, err_dense, color="#2b8a3e", lw=2.0,
            label="True Error (Dense Spline)")
    a2.set_xlabel("time [s]"); a2.set_ylabel("lateral tracking error [m]")
    a2.set_title(f"Tracking error (mean {err_dense.mean():.2f} m, max {err_dense.max():.2f} m)")
    a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Figure 7 (MPC): {out_path}")


def animate(log, world, trailer, splines_log, gif=None, live=False):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle
    dx = np.array([r["drone_x"] for r in log]); dy = np.array([r["drone_y"] for r in log])
    tx = np.array([r["trailer_x"] for r in log]); ty = np.array([r["trailer_y"] for r in log])
    step = max(1, len(log) // 400)
    fig, ax = plt.subplots(figsize=(8, 8))
    for lo, hi in zip(world.boxes_min, world.boxes_max):
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1],
                               facecolor="#9aa4ad", edgecolor="none", alpha=0.5))
    s = np.linspace(0, 8*trailer.half, 400)
    loop = np.array([square_loop_pos(v, trailer.half) for v in s])
    ax.plot(loop[:, 0], loop[:, 1], ":", color="#c92a2a", lw=1)
    
    old_splines = []
    for _ in splines_log:
        line, = ax.plot([], [], "-", color=C_BSPL, alpha=0.5, lw=1)
        old_splines.append(line)

    dline, = ax.plot([], [], "-", color=C_MPC, lw=2)
    ddot, = ax.plot([], [], "o", color=C_MPC, ms=8)
    tdot, = ax.plot([], [], "*", color=C_TRAIL, ms=16)
    
    ax.set_aspect("equal"); ax.set_xlim(-650, 650); ax.set_ylim(-650, 650)
    ax.set_xlabel("E x [m]"); ax.set_ylabel("N y [m]"); ax.grid(alpha=0.3)
    idx = list(range(0, len(log), step)) + [len(log) - 1]

    def upd(i):
        t_curr = log[i]['t_s']
        for (t_gen, sp_tp, _, _), line in zip(splines_log, old_splines):
            if t_curr >= t_gen:
                line.set_data(sp_tp[:, 0], sp_tp[:, 1])
            else:
                line.set_data([], [])
                
        dline.set_data(dx[:i+1], dy[:i+1])
        ddot.set_data([dx[i]], [dy[i]]); tdot.set_data([tx[i]], [ty[i]])
        ax.set_title(f"t={log[i]['t_s']:.1f}s  dist={log[i]['dist_xy_m']:.1f} m")
        return [dline, ddot, tdot] + old_splines
    anim = animation.FuncAnimation(fig, upd, frames=idx, interval=40, blit=False)
    if gif:
        anim.save(gif, writer=animation.PillowWriter(fps=25)); print(f"gif: {gif}")
    if live:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    ap.add_argument("--res", type=float, default=None)
    ap.add_argument("--csv", default=str(OUT / "pursuit_log.csv"))
    ap.add_argument("--animate", action="store_true", help="live matplotlib window")
    ap.add_argument("--gif", default=None, help="save animation to this gif path")
    args = ap.parse_args()
    if not args.animate:
        matplotlib.use("Agg")
    OUT.mkdir(exist_ok=True)

    scenario = yaml.safe_load(Path(args.scenario).read_text())
    (log, captured, world, trailer, splines_log, mpc_log,
     _plan_stats, _mpc_times) = run_sim(scenario, res_override=args.res)
    save_csv(log, args.csv)

    foots = raw_footprints(REPO / scenario["base_map"])
    band = (scenario["drone"]["cruise_floor_m"], scenario["drone"]["cruise_ceiling_m"])
    _fig_topdown(
        log, world, trailer, splines_log, mpc_log,
        OUT / "Gazebo_filght_planned_4_paths.png",
        OUT / "Gazebo_filght_planned_3_paths.png")
    _fig_profiles(log, OUT / "6_pursuit_profiles.png", band=band)
    _fig_mpc(log, splines_log, OUT / "7_pursuit_mpc.png")
    _fig_3d(log, foots, splines_log, trailer, OUT / "8_pursuit_3d.png")
    
    if args.animate or args.gif:
        animate(log, world, trailer, splines_log, gif=args.gif, live=args.animate)


if __name__ == "__main__":
    main()
