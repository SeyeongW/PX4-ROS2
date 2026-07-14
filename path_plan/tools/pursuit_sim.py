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
import time
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from path_plan.astar import AStarPlanner3D
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.mpc_ros import UnicycleMPC, Weights
from path_plan.world_model import WorldModel, _find_buildings

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_SCENARIO = REPO / "gazebo/maps/city_uav_trailer_loop.yaml"
_CORNER_S = {"SE": 0.0, "NE": 2.0, "NW": 4.0, "SW": 6.0}   # x H units along the loop


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
    cruise_z = 0.5 * (floor + ceil)
    res = float(res_override or pu["astar_resolution_m"])

    world = WorldModel.from_city_yaml(
        base, inflation_xy_m=d["inflation_xy_m"], ground_clearance_m=floor,
        ceiling_m=ceil, overfly_allowed=d["overfly_allowed"])
    planner = AStarPlanner3D(world, resolution_m=res)
    optimizer = BsplineOptimizer(world, cruise_speed_m_s=d["cruise_speed_m_s"],
                                 ctrl_spacing_m=8.0, max_vel=d["cruise_speed_m_s"] + 1.0,
                                 lambda_feas=2.0)
    mpc = UnicycleMPC(dt_s=pu["sim_dt_s"], horizon=15, v_ref=d["cruise_speed_m_s"],
                      v_max=d["cruise_speed_m_s"] + 1.0, max_iter=20,
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
    while t <= t_max:
        tpos = trailer.pos(t)
        dist = float(np.linalg.norm(drone[:2] - tpos))
        
        if verbose and step_idx % 20 == 0:
            real_t = time.time() - wall0
            print(f"Simulating: SimTime={t:.1f}s / {t_max:.1f}s | RealTime={real_t:.1f}s | dist={dist:.2f}m")
        step_idx += 1

        if dist < cap:
            captured = True
            if spline is None:
                track_err = 0.0
                track_err_dense = 0.0
            else:
                track_err = float(np.min(np.linalg.norm(sp_tp[:, :2] - drone[:2], axis=1)))
                track_err_dense = float(np.min(np.linalg.norm(dense_tp[:, :2] - drone[:2], axis=1)))
            log.append(_row(t, drone, speed, 0.0, yaw, 0.0, tpos, trailer.speed, dist,
                            replan_count, "capture", True, track_err, track_err_dense))
            break

        phase = "homing" if dist <= homing else "pursuit"
        # real-time global replan toward the trailer's current position
        if phase == "pursuit" and (spline is None or t - last_replan >= replan_period):
            leg = planner.plan(drone, np.array([tpos[0], tpos[1], cruise_z]))
            if leg.success and len(leg.waypoints_m) >= 2:
                opt = optimizer.optimize(leg.waypoints_m)
                spline = opt.spline
                sp_tp, sp_tt = _sample_spline(spline)
                _, dense_tp, _, _ = spline.sample(4000)
                splines_log.append((t, np.copy(sp_tp), opt.corridor, np.copy(leg.waypoints_m)))
                replan_count += 1
            last_replan = t

        # MPC reference: home straight at the trailer when close, else track spline
        if phase == "homing" or spline is None:
            goal = np.array([tpos[0], tpos[1], cruise_z])
            ref = np.linspace(drone, goal, mpc.N + 1)[1:]
        else:
            ref = _horizon_from_samples(sp_tp, sp_tt, drone, mpc.N, dt)

        out = mpc.solve(drone, yaw, speed, drone[2], ref)
        mpc_log.append(np.copy(out.predicted_xy))
        v = out.velocity_world
        yaw_rate = out.yaw_rate
        new_speed = float(np.hypot(v[0], v[1]))
        accel = (new_speed - speed) / dt
        
        drone = drone + v * dt
        yaw += yaw_rate * dt
        speed = new_speed
        
        if phase == "homing" or spline is None:
            track_err = 0.0
            track_err_dense = 0.0
        else:
            track_err = float(np.min(np.linalg.norm(sp_tp[:, :2] - drone[:2], axis=1)))
            track_err_dense = float(np.min(np.linalg.norm(dense_tp[:, :2] - drone[:2], axis=1)))

        log.append(_row(t, drone, speed, accel, yaw, yaw_rate, tpos, trailer.speed, dist,
                        replan_count, phase, False, track_err, track_err_dense))
        t += dt

    if verbose:
        print(f"\nsim {'CAPTURED' if captured else 'timed out'} at t={t:.1f}s "
              f"(dist={log[-1]['dist_xy_m']:.2f} m), replans={replan_count}, "
              f"steps={len(log)}, wall={time.time()-wall0:.1f}s")
    return log, captured, world, trailer, splines_log, mpc_log


def _row(t, drone, speed, accel, yaw, yaw_rate, tpos, tspeed, dist, replans, phase, captured, track_err=0.0, track_err_dense=0.0):
    return dict(t_s=round(t, 3), drone_x=round(float(drone[0]), 3),
                drone_y=round(float(drone[1]), 3), drone_z=round(float(drone[2]), 3),
                drone_speed_mps=round(speed, 4), drone_accel_mps2=round(accel, 4),
                drone_yaw_rad=round(yaw, 5), drone_yawrate_rads=round(yaw_rate, 4),
                trailer_x=round(float(tpos[0]), 3), trailer_y=round(float(tpos[1]), 3),
                trailer_speed_mps=round(float(tspeed), 4),
                dist_xy_m=round(float(dist), 4), replan_count=replans,
                phase=phase, captured=int(captured), track_err_m=round(track_err, 4),
                track_err_dense_m=round(track_err_dense, 4))


def save_csv(log, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print(f"CSV written: {path}  ({len(log)} rows)")


def _fig_topdown(log, world, trailer, splines_log, mpc_log, out_path):
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
                               facecolor="#868e96", edgecolor="none", alpha=0.5, zorder=1))
                               
    # Trailer full loop
    s = np.linspace(0, 8*trailer.half, 400)
    loop = np.array([square_loop_pos(v, trailer.half) for v in s])
    ax.plot(loop[:, 0], loop[:, 1], ":", color="#e8590c", lw=1.5, label="trailer loop", zorder=1)
    
    # 1. SFC Corridors & Historical Paths
    ax.plot([], [], "-", color="#4c6ef5", alpha=0.5, lw=2, label="SFC corridor", zorder=2)
    for i, (t_gen, sp_tp, corridor, wp) in enumerate(splines_log):
        for lo, hi in zip(corridor.boxes_min, corridor.boxes_max):
            ax.add_patch(Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1],
                                   facecolor="none", edgecolor="#4c6ef5", 
                                   alpha=0.3, lw=1.0, zorder=2))
        
        # A* waypoints (Blue)
        ax.plot(wp[:, 0], wp[:, 1], "-o", color="#1c7ed6", ms=5, lw=1.2, alpha=0.7,
                label="A* waypoints" if i == 0 else "", zorder=3)
                
        # B-Spline (Red)
        ax.plot(sp_tp[:, 0], sp_tp[:, 1], "-", color="#fa5252", alpha=0.8, lw=2.4, 
                label="B-spline path" if i == 0 else "", zorder=4)

    # 4. Actual Paths (Green dashed for drone, Orange for trailer)
    ax.plot(dx, dy, "--", color="#8ce200", lw=2.5, label="MPC tracked", zorder=5)
    ax.plot(tx, ty, "-", color="#e8590c", lw=2.5, label="trailer path", zorder=5)
    
    # 5. Start/End Markers
    ax.plot(dx[0], dy[0], "^", color="#2b8a3e", ms=15, label="drone start", zorder=6)
    ax.plot(tx[-1], ty[-1], "*", color="#c92a2a", ms=20,
            label="capture" if captured else "trailer end", zorder=6)
            
    ax.set_aspect("equal"); ax.set_xlabel("E x [m]"); ax.set_ylabel("N y [m]")
    ax.set_title(f"Moving-trailer pursuit ({'CAPTURED' if captured else 'timeout'})")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    
    fig_map.tight_layout()
    fig_map.savefig(out_path, dpi=130)
    plt.close(fig_map)
    print(f"Figure 5 (Map): {out_path}")

def _fig_profiles(log, out_path):
    import matplotlib.pyplot as plt
    t = np.array([r["t_s"] for r in log])
    spd = np.array([r["drone_speed_mps"] for r in log])
    acc = np.array([r["drone_accel_mps2"] for r in log])
    
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(t, spd, color="#fa5252", lw=2)
    a1.set_ylabel("speed [m/s]"); a1.grid(alpha=0.3)
    a1.set_title("Drone speed & acceleration profiles")
    a2.plot(t, np.abs(acc), color="#fcc419", lw=2)
    a2.set_ylabel("|accel| [m/s²]"); a2.set_xlabel("time [s]")
    a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Figure 6 (Profiles): {out_path}")

def _fig_mpc(log, splines_log, out_path):
    import matplotlib.pyplot as plt
    t = np.array([r["t_s"] for r in log])
    dx = np.array([r["drone_x"] for r in log])
    dy = np.array([r["drone_y"] for r in log])
    err = np.array([r["track_err_m"] for r in log])
    err_dense = np.array([r.get("track_err_dense_m", 0.0) for r in log])
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, (t_gen, sp_tp, corridor, wp) in enumerate(splines_log):
        a1.plot(sp_tp[:, 0], sp_tp[:, 1], "-", color="#fa5252", lw=3, alpha=0.6,
                label="reference (B-spline)" if i == 0 else "")
    a1.plot(dx, dy, "--", color="#8ce200", lw=2, label="MPC tracked")
    a1.set_aspect("equal"); a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]")
    a1.set_title("mpc_ros unicycle MPC — closed-loop tracking (top-down)")
    a1.legend(); a1.grid(alpha=0.3)
    
    a2.plot(t, err, color="#b2f252", lw=1.5, alpha=0.6, label="Raw (Sawtooth artifact)")
    a2.plot(t, err_dense, color="#2b8a3e", lw=2.0, label="True Error (Dense Spline)")
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
        line, = ax.plot([], [], "-", color="#fa5252", alpha=0.5, lw=1)
        old_splines.append(line)
        
    dline, = ax.plot([], [], "-", color="#2f9e44", lw=2)
    ddot, = ax.plot([], [], "o", color="#2f9e44", ms=8)
    tdot, = ax.plot([], [], "*", color="#e8590c", ms=16)
    
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
    log, captured, world, trailer, splines_log, mpc_log = run_sim(scenario, res_override=args.res)
    save_csv(log, args.csv)
    
    _fig_topdown(log, world, trailer, splines_log, mpc_log, OUT / "5_pursuit_map.png")
    _fig_profiles(log, OUT / "6_pursuit_profiles.png")
    _fig_mpc(log, splines_log, OUT / "7_pursuit_mpc.png")
    
    if args.animate or args.gif:
        animate(log, world, trailer, splines_log, gif=args.gif, live=args.animate)


if __name__ == "__main__":
    main()
