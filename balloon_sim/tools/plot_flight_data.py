#!/usr/bin/env python3
"""Comprehensive flight data plotter for STL balloon SITL missions.

Auto-finds the latest /tmp/px4_balloon_* directory, extracts telemetry,
and generates multi-panel analysis plots saved to balloon_sim/plots/.

Usage:
  python3 tools/plot_flight_data.py                    # latest run
  python3 tools/plot_flight_data.py --mission /path/to/mission  # specific run
"""
import argparse
import json
import math
from glob import glob
from pathlib import Path

import numpy as np


def load_mission(mission_dir):
    mission_dir = Path(mission_dir)
    plan = json.loads((mission_dir / 'plan.json').read_text())
    rows = []
    with (mission_dir / 'telemetry.jsonl').open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    events = json.loads((mission_dir / 'events.json').read_text()) if (mission_dir / 'events.json').exists() else []
    report = json.loads((mission_dir / 'report.json').read_text()) if (mission_dir / 'report.json').exists() else {}
    return plan, rows, events, report


def extract_arrays(rows):
    t, pos, vel, des_pos, des_vel, des_acc, err, phase = [], [], [], [], [], [], [], []
    heading, speed, des_speed = [], [], []
    for r in rows:
        stamp = r.get('own_stamp', r.get('stamp'))
        if stamp is None:
            continue
        t.append(float(stamp))
        p = r.get('position', [np.nan]*3)
        v = r.get('velocity', [np.nan]*3)
        dp = r.get('desired_position', [np.nan]*3)
        dv = r.get('desired_velocity', [np.nan]*3)
        da = r.get('desired_acceleration', [np.nan]*3)
        pos.append(p)
        vel.append(v)
        des_pos.append(dp if dp else [np.nan]*3)
        des_vel.append(dv if dv else [np.nan]*3)
        des_acc.append(da if da else [np.nan]*3)
        e = r.get('tracking_error_m', np.nan)
        err.append(float(e) if e is not None else np.nan)
        phase.append(r.get('phase', ''))
        v_arr = np.array(v, dtype=float)
        speed.append(float(np.linalg.norm(v_arr)))
        dv_arr = np.array(dv if dv else [0,0,0], dtype=float)
        des_speed.append(float(np.linalg.norm(dv_arr)))
        q = r.get('attitude_quaternion_world')
        if q and len(q) == 4:
            from scipy.spatial.transform import Rotation
            yaw = Rotation.from_quat(q).as_euler('xyz')[2]
            heading.append(float(np.degrees(yaw)))
        else:
            heading.append(np.nan)
    return dict(
        t=np.array(t), pos=np.array(pos), vel=np.array(vel),
        des_pos=np.array(des_pos), des_vel=np.array(des_vel), des_acc=np.array(des_acc),
        err=np.array(err), phase=np.array(phase),
        speed=np.array(speed), des_speed=np.array(des_speed), heading=np.array(heading))


def find_latest_mission():
    dirs = sorted(glob('/tmp/px4_balloon_*/mission'), key=lambda d: Path(d).stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError('No /tmp/px4_balloon_*/mission found. Run SITL first.')
    return Path(dirs[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mission', type=Path, help='mission directory (default: latest)')
    parser.add_argument('--output', type=Path, help='output directory (default: balloon_sim/plots/)')
    args = parser.parse_args(argv)

    mission_dir = args.mission or find_latest_mission()
    plan, rows, events, report = load_mission(mission_dir)
    d = extract_arrays(rows)

    output_dir = args.output or Path(__file__).resolve().parent.parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    style = {'font.family': 'DejaVu Sans', 'font.size': 9,
             'axes.unicode_minus': False, 'axes.spines.top': False,
             'axes.spines.right': False, 'savefig.facecolor': 'white'}

    t0 = d['t'][0]
    t = d['t'] - t0

    plan_samples = plan.get('samples', [])
    plan_pos = np.array([s['position'] for s in plan_samples])
    plan_vel = np.array([s['velocity'] for s in plan_samples])
    plan_t = np.array([s.get('time_s', s.get('stamp', 0)) for s in plan_samples])

    home = np.array(plan['home'])
    goal = np.array(plan['goal'])
    target = np.array(plan.get('target_world', goal))

    phase_changes = []
    for i, e in enumerate(events):
        if 'phase' in str(e):
            phase_changes.append(e)

    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

        # 1) 3D trajectory
        ax = fig.add_subplot(3, 2, 1, projection='3d')
        axes[0, 0].set_visible(False)
        ax.plot(*plan_pos.T, 'k--', lw=1, alpha=0.5, label='Planned')
        ax.plot(*d['pos'].T, 'b-', lw=1.5, label='Actual')
        ax.scatter(*home, c='green', s=60, marker='o', label='Home', depthshade=False)
        ax.scatter(*target, c='red', s=80, marker='*', label='Target', depthshade=False)
        ax.scatter(*goal, c='purple', s=60, marker='s', label='Goal', depthshade=False)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('3D Trajectory')
        ax.legend(fontsize=7, loc='upper left')

        # 2) XYZ position vs time
        ax2 = axes[0, 1]
        labels_xyz = ['X','Y','Z']
        colors_xyz = ['#e41a1c','#377eb8','#4daf4a']
        for i, (lab, col) in enumerate(zip(labels_xyz, colors_xyz)):
            ax2.plot(t, d['pos'][:,i], color=col, lw=1.2, label=f'{lab} actual')
            if d['des_pos'].shape == d['pos'].shape:
                ax2.plot(t, d['des_pos'][:,i], color=col, lw=0.8, ls='--', alpha=0.6)
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Position (m)')
        ax2.set_title('Position vs Time (solid=actual, dashed=desired)')
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        # 3) Speed profile
        ax3 = axes[1, 0]
        ax3.plot(t, d['speed'], 'b-', lw=1.3, label='Actual speed')
        ax3.plot(t, d['des_speed'], 'r--', lw=1, alpha=0.7, label='Desired speed')
        max_speed = plan.get('limits', {}).get('max_speed_m_s',
                    plan.get('command_limits', {}).get('speed_m_s', None))
        if max_speed:
            ax3.axhline(max_speed, color='gray', ls=':', lw=0.8, label=f'Limit {max_speed:.1f} m/s')
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed Profile')
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

        # 4) Tracking error
        ax4 = axes[1, 1]
        valid = np.isfinite(d['err'])
        if valid.any():
            ax4.plot(t[valid], d['err'][valid], 'b-', lw=1.2)
            ax4.fill_between(t[valid], 0, d['err'][valid], alpha=0.15)
            mean_err = np.nanmean(d['err'][valid])
            max_err = np.nanmax(d['err'][valid])
            ax4.axhline(mean_err, color='orange', ls='--', lw=0.8,
                        label=f'Mean {mean_err:.2f}m')
            ax4.set_title(f'Tracking Error (max {max_err:.2f}m, mean {mean_err:.2f}m)')
        else:
            ax4.set_title('Tracking Error (no data)')
        ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Error (m)')
        ax4.set_ylim(bottom=0); ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

        # 5) Heading vs time
        ax5 = axes[2, 0]
        valid_h = np.isfinite(d['heading'])
        if valid_h.any():
            ax5.plot(t[valid_h], d['heading'][valid_h], 'b-', lw=1.2)
            vel_heading = np.degrees(np.arctan2(d['vel'][:,1], d['vel'][:,0]))
            fast = d['speed'] > 0.5
            if fast.any():
                ax5.plot(t[fast], vel_heading[fast], 'r--', lw=0.8, alpha=0.6, label='Velocity heading')
            ax5.set_title('Heading (yaw)')
            ax5.legend(fontsize=7)
        else:
            ax5.set_title('Heading (no data)')
        ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Angle (deg)')
        ax5.grid(True, alpha=0.3)

        # 6) Velocity components
        ax6 = axes[2, 1]
        for i, (lab, col) in enumerate(zip(['Vx','Vy','Vz'], colors_xyz)):
            ax6.plot(t, d['vel'][:,i], color=col, lw=1.2, label=lab)
        ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Velocity (m/s)')
        ax6.set_title('Velocity Components')
        ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3)

        # Phase annotations on all time-axis plots
        phases_seen = set()
        for i in range(len(d['phase'])):
            if d['phase'][i] and d['phase'][i] not in phases_seen:
                phases_seen.add(d['phase'][i])
                for ax_t in [ax2, ax3, ax4, ax5, ax6]:
                    ax_t.axvline(t[i], color='gray', ls=':', lw=0.5, alpha=0.5)

        duration = plan.get('duration_s', 0)
        motion = plan.get('motion_validation', {})
        peaks = motion.get('sampled_peaks', {})
        title = (f'Flight Analysis — {plan.get("path_kind","?")} | '
                 f'Distance {plan.get("sampled_length_m",0):.1f}m | '
                 f'Duration {duration:.1f}s | '
                 f'Peak Speed {peaks.get("speed_m_s",0):.1f}m/s | '
                 f'Peak Accel {peaks.get("acceleration_m_s2",0):.1f}m/s²')
        fig.suptitle(title, fontsize=12)

        stem = output_dir / f'flight_{mission_dir.parent.name}'
        for ext in ('.png', '.svg'):
            fig.savefig(str(stem) + ext, dpi=150)
        plt.close(fig)

    print(f'Saved: {stem}.png, {stem}.svg')

    summary = dict(
        mission=str(mission_dir),
        path_kind=plan.get('path_kind'),
        duration_s=plan.get('duration_s'),
        sampled_length_m=plan.get('sampled_length_m'),
        motion_validation=motion,
        tracking_error_max_m=float(np.nanmax(d['err'])) if np.isfinite(d['err']).any() else None,
        tracking_error_mean_m=float(np.nanmean(d['err'][np.isfinite(d['err'])])) if np.isfinite(d['err']).any() else None,
        max_speed_actual_m_s=float(np.nanmax(d['speed'])),
        phases=list(dict.fromkeys(d['phase'])),
        report_outcome=report.get('outcome'),
        report_reason=report.get('reason'),
    )
    summary_path = str(stem) + '_summary.json'
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Summary: {summary_path}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
