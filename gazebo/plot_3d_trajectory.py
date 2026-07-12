#!/usr/bin/env python3
"""3D flight-path figure for papers -- obstacles (obstacle_map.yaml, extruded
to their real height) + the drone's TAKEOFF -> MISSION(A*+corridor+MPC,
mission_manager_node's exact algorithm) -> RETURN/DESCEND(APF,
precision_landing_node's exact algorithm) trajectory, rendered as a static
publication-quality PNG+PDF.

Two data sources, pick ONE:
  --log FILE.csv   a REAL recorded flight (columns: t,e,n,alt -- see
                    record_trajectory.py, which subscribes to
                    /mavros/local_position/pose during an actual sim run and
                    writes exactly this format). This is what should go in the
                    paper as the actual result.
  (no --log)        a RECONSTRUCTED illustrative trajectory: runs the same
                    A*+corridor+MPC closed loop mission_manager_node.py
                    actually executes (mirrored here standalone, ROS-free --
                    see mission_manager_node.py's _mpc_servo_to/_mpc_maybe_replan
                    for the real ROS version) for the MISSION leg, then a
                    simple APF-avoided descent for RETURN, so you can get a
                    representative figure before/without a live sim run. The
                    plot is titled "reconstructed" in this mode -- swap in
                    --log from a real run before using it as an actual paper
                    result, not just a methods-illustration figure.

Usage:
    python3 gazebo/plot_3d_trajectory.py --goal -15 -20 --out fig
    python3 gazebo/plot_3d_trajectory.py --log /tmp/traj.csv --out fig
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'precision_landing'))
from precision_landing import planner, mpc as mpc_solver, apf  # noqa: E402

import yaml  # noqa: E402


# --- MISSION leg: exact closed-loop replanning mission_manager_node.py runs,
# mirrored standalone (no ROS) -- see _mpc_servo_to/_mpc_maybe_replan there.
def simulate_mpc_mission(start, goal, obstacle_map_path, flight_alt, vmax=6.0,
                         dt=0.05, replan_period=2.0, solve_period=0.15,
                         margin=1.5, arrive_radius=2.0, max_t=90.0):
    pos = np.array(start, dtype=float)
    vel = np.array([0.0, 0.0])
    box = None
    local_target = None
    last_replan = -999.0
    last_solve = -999.0
    last_cmd = (0.0, 0.0)
    t = 0.0
    path = [(0.0, pos[0], pos[1], flight_alt)]
    corridors_used = []
    while t < max_t:
        reached_local = (local_target is not None and
                        np.hypot(*(pos - np.array(local_target))) < arrive_radius)
        drifted_out = box is not None and not box.inflated(-0.5).contains(pos[0], pos[1])
        if box is None or (t - last_replan) >= replan_period or drifted_out or reached_local:
            _, simplified, corridor = planner.plan(
                tuple(pos), goal, obstacle_map_path, safety_margin=margin)
            if corridor is not None:
                box = corridor[0]
                local_target = simplified[1] if len(simplified) > 1 else goal
                corridors_used.append(box)
            last_replan = t
        if (t - last_solve) >= solve_period:
            tgt = local_target if local_target is not None else goal
            ve, vn = mpc_solver.solve(tuple(pos), tuple(vel), tgt, (0.0, 0.0),
                                     box, vmax, 0.3, 10, iters=60)
            last_cmd = (ve, vn)
            last_solve = t
        else:
            ve, vn = last_cmd
        if box is None and apf_obs:
            rE, rN = apf.repulsive_velocity(pos[0], pos[1], apf_obs, 15.0, 6.0, 6.0)
            ve, vn = ve + rE, vn + rN
        sp = math.hypot(ve, vn)
        if sp > vmax:
            ve, vn = ve * vmax / sp, vn * vmax / sp
        vel = np.array([ve, vn])
        pos = pos + vel * dt
        t += dt
        path.append((t, pos[0], pos[1], flight_alt))
        if np.hypot(*(pos - np.array(goal))) < 1.0:
            break
    return path, corridors_used


# --- RETURN/DESCEND leg: APF avoidance (precision_landing_node's algorithm)
# + a simple altitude ramp using the tuned descend_rate/descend_min_scale.
def simulate_return_and_descend(start_e, start_n, start_t, platform_e, platform_n,
                                flight_alt, obstacle_map_path,
                                vel_gain=0.4, vmax=8.0, dt=0.05,
                                descend_rate=0.8, descend_min_scale=0.5,
                                platform_height=1.0, land_clearance=0.2):
    pos = np.array([start_e, start_n], dtype=float)
    alt = flight_alt
    t = start_t
    path = []
    target = np.array([platform_e, platform_n])
    touchdown_alt = platform_height + land_clearance
    while True:
        e_err, n_err = target[0] - pos[0], target[1] - pos[1]
        vE, vN = vel_gain * e_err, vel_gain * n_err
        if apf_obs:
            rE, rN = apf.repulsive_velocity(pos[0], pos[1], apf_obs, 15.0, 6.0, 6.0)
            vE, vN = vE + rE, vN + rN
        sp = math.hypot(vE, vN)
        if sp > vmax:
            vE, vN = vE * vmax / sp, vN * vmax / sp
        lateral_err = math.hypot(e_err, n_err)
        # descend once reasonably aligned, funnel-style (narrows near ground)
        funnel = max(0.2, 0.35 * max(alt - platform_height, 0.05))
        if lateral_err <= funnel and alt > touchdown_alt:
            scale = max(descend_min_scale, min(1.0, 1.0 - lateral_err / funnel))
            alt -= descend_rate * scale * dt
        pos = pos + np.array([vE, vN]) * dt
        t += dt
        path.append((t, pos[0], pos[1], alt))
        if alt <= touchdown_alt and lateral_err < 0.3:
            break
        if t - start_t > 120.0:
            break
    return path


def load_obstacle_boxes_and_height(obstacle_map_path):
    data = yaml.safe_load(open(obstacle_map_path))
    default_height = data['height']
    # Per-obstacle height override (e.g. obstacle_map_city.yaml, where every
    # building has its own real height instead of one shared default).
    heights = [o.get('height', default_height) for o in data['obstacles']]
    boxes = planner.load_obstacle_boxes(obstacle_map_path)
    return boxes, heights


def draw_prism(ax, e_min, e_max, n_min, n_max, z_min, z_max, color, alpha, edgecolor):
    x = [e_min, e_max, e_max, e_min]
    y = [n_min, n_min, n_max, n_max]
    verts_bottom = [list(zip(x, y, [z_min] * 4))]
    verts_top = [list(zip(x, y, [z_max] * 4))]
    sides = []
    for i in range(4):
        j = (i + 1) % 4
        sides.append([(x[i], y[i], z_min), (x[j], y[j], z_min),
                      (x[j], y[j], z_max), (x[i], y[i], z_max)])
    poly = Poly3DCollection(verts_bottom + verts_top + sides,
                            facecolor=color, alpha=alpha, edgecolor=edgecolor,
                            linewidths=0.4)
    ax.add_collection3d(poly)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--goal', nargs=2, type=float, default=[-15.0, -20.0],
                    help='MISSION-leg target (E N) -- ignored if --log given')
    ap.add_argument('--platform', nargs=2, type=float, default=[0.0, 0.0],
                    help='RETURN/landing target (E N) -- ignored if --log given')
    ap.add_argument('--map', default=os.path.join(SCRIPT_DIR, 'config', 'obstacle_map.yaml'))
    ap.add_argument('--flight-alt', type=float, default=5.0)
    ap.add_argument('--log', default=None, help='real recorded trajectory CSV (t,e,n,alt)')
    ap.add_argument('--no-corridor', action='store_true',
                    help='hide the MPC safe-flight-corridor overlay')
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'trajectory_3d'),
                    help='output path WITHOUT extension -- writes .png and .pdf')
    args = ap.parse_args()

    global apf_obs
    apf_obs = apf.load_obstacles(args.map)
    obstacles, obs_heights = load_obstacle_boxes_and_height(args.map)

    corridors = []
    if args.log:
        path = []
        with open(args.log) as f:
            for row in csv.DictReader(f):
                path.append((float(row['t']), float(row['e']), float(row['n']), float(row['alt'])))
        mission_end_idx = len(path)   # no phase split known from a plain log
        reconstructed = False
    else:
        mission_path, corridors = simulate_mpc_mission(
            (0.0, 0.0), tuple(args.goal), args.map, args.flight_alt)
        mission_end_idx = len(mission_path)
        return_path = simulate_return_and_descend(
            mission_path[-1][1], mission_path[-1][2], mission_path[-1][0],
            args.platform[0], args.platform[1], args.flight_alt, args.map)
        # TAKEOFF: vertical climb at the origin before MISSION starts
        n_climb = 30
        climb = [(-((n_climb - i) * 0.1), 0.0, 0.0, args.flight_alt * i / n_climb)
                for i in range(n_climb)]
        path = climb + mission_path + return_path
        mission_end_idx = len(climb) + mission_path.__len__()
        reconstructed = True

    path = np.array(path)  # columns: t, e, n, alt

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    for ob, obs_height in zip(obstacles, obs_heights):
        draw_prism(ax, ob.e_min, ob.e_max, ob.n_min, ob.n_max, 0.0, obs_height,
                  color='#6b7680', alpha=0.35, edgecolor='#3a4750')

    if corridors and not args.no_corridor:
        seen = set()
        for b in corridors:
            key = (round(b.e_min, 1), round(b.e_max, 1), round(b.n_min, 1), round(b.n_max, 1))
            if key in seen:
                continue
            seen.add(key)
            draw_prism(ax, b.e_min, b.e_max, b.n_min, b.n_max, 0.0, args.flight_alt,
                      color='#1f8aa3', alpha=0.06, edgecolor='#1f8aa3')

    mission_pts = path[:mission_end_idx]
    return_pts = path[mission_end_idx - 1:] if mission_end_idx < len(path) else path[:0]
    ax.plot(mission_pts[:, 1], mission_pts[:, 2], mission_pts[:, 3],
           color='#f2a13c', linewidth=2.2, label='MISSION (A*+corridor+MPC)' if reconstructed else 'flight path')
    if len(return_pts):
        ax.plot(return_pts[:, 1], return_pts[:, 2], return_pts[:, 3],
               color='#c23458', linewidth=2.2, label='RETURN/DESCEND (APF)')

    ax.scatter(*path[0, 1:], color='#2f9e46', s=60, label='takeoff', depthshade=False)
    ax.scatter(args.platform[0], args.platform[1], 1.0, color='#e8577c', marker='v',
              s=90, label='platform (landing)', depthshade=False)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    title = 'Reconstructed flight path (illustrative)' if reconstructed else 'Recorded flight path'
    ax.set_title(f'{title}\nobstacle_field, A*+corridor+MPC (MISSION) / APF (RETURN)')
    ax.view_init(elev=28, azim=-60)
    ax.set_box_aspect([1, 1, 0.5])

    legend_elems = ax.get_legend_handles_labels()[0]
    legend_elems.append(Patch(facecolor='#6b7680', alpha=0.35, label='obstacle'))
    if corridors and not args.no_corridor:
        legend_elems.append(Patch(facecolor='#1f8aa3', alpha=0.15, label='safe flight corridor'))
    ax.legend(handles=legend_elems, loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=300)
    fig.savefig(args.out + '.pdf')
    print(f'wrote {args.out}.png and {args.out}.pdf')
    if reconstructed:
        print('NOTE: this is a RECONSTRUCTED/illustrative trajectory (planner+MPC/APF run '
             'standalone, not a real Gazebo/ArduPilot flight). For a paper RESULT figure, '
             'record a real run with gazebo/record_trajectory.py and pass --log.')


if __name__ == '__main__':
    main()
