#!/usr/bin/env python3
"""PX4 테일시터 ULog (.ulg) 비행 로그 파서 및 텔레메트리/3D 궤적 그래프 생성 도구.

외부 pyulog 패키지 없이 순수 Python(struct + numpy + matplotlib)만으로
PX4 SITL의 .ulg 바이너리 로그를 파싱하여 plots/ 디렉토리에 분석 그래프 2종과 요약 JSON을 저장합니다:
  - plots/tailsitter_telemetry_dashboard.png
  - plots/tailsitter_3d_trajectory.png
  - plots/tailsitter_flight_summary.json

사용법:
    ./tools/plot_tailsitter_log.py
    ./tools/plot_tailsitter_log.py /path/to/log.ulg --out-dir ./plots
"""
import argparse
import glob
import json
import math
import os
import re
import struct
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HOME_WORLD = np.array([-2.0, 0.0, 0.2], dtype=float)
TARGET_WORLD = np.array([
    float(os.environ.get('FLY_TARGET_X', '90.0')),
    float(os.environ.get('FLY_TARGET_Y', '0.0')),
    float(os.environ.get('FLY_TARGET_Z', '40.0')),
], dtype=float)
NORTH_YAW = math.pi / 2.0

ULOG_TYPE_MAP = {
    'int8_t': ('b', 1),
    'uint8_t': ('B', 1),
    'int16_t': ('h', 2),
    'uint16_t': ('H', 2),
    'int32_t': ('i', 4),
    'uint32_t': ('I', 4),
    'int64_t': ('q', 8),
    'uint64_t': ('Q', 8),
    'float': ('f', 4),
    'double': ('d', 8),
    'bool': ('?', 1),
    'char': ('c', 1),
}


def world_to_ned(wx, wy, wz):
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    dx, dy, dz = wx - HOME_WORLD[0], wy - HOME_WORLD[1], wz - HOME_WORLD[2]
    return np.array([c * dx + s * dy, s * dx - c * dy, -dz], dtype=float)


def ned_to_world(n, e, d):
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    wx = HOME_WORLD[0] + c * n + s * e
    wy = HOME_WORLD[1] + s * n - c * e
    wz = HOME_WORLD[2] - d
    return wx, wy, wz


def quat_to_nose_and_euler(q_wxyz):
    """쿼터니언 (N, 4) [w, x, y, z] -> 노즈콘 단위벡터, 기울기(Tilt), 카메라 앙각(Elev), 방위각(Az), 날개 롤(Wing Roll)."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    nose_n = -2.0 * (xz + wy)
    nose_e = -2.0 * (yz - wx)
    nose_d = -(1.0 - 2.0 * (xx + yy))

    tilt_deg = np.degrees(np.arccos(np.clip(-nose_d, -1.0, 1.0)))
    elev_deg = 90.0 - tilt_deg
    az_deg = np.degrees(np.arctan2(nose_e, nose_n))

    wing_d = 2.0 * (yz + wx)
    wing_roll_deg = np.degrees(np.arcsin(np.clip(wing_d, -1.0, 1.0)))

    return np.column_stack((nose_n, nose_e, nose_d)), tilt_deg, elev_deg, az_deg, wing_roll_deg


def compile_format(fmt_name, formats):
    """ULog 포맷 정의를 struct unpack 문자열과 필드 목록으로 변환 (trailing _padding 제외)."""
    fields = formats[fmt_name]
    struct_chars = ['<']
    field_info = []
    for t_raw, v_name in fields:
        if v_name.startswith('_padding'):
            continue
        m = re.match(r'^([a-zA-Z0-9_]+)(?:\[(\d+)\])?$', t_raw)
        if not m:
            continue
        base_t, arr_len = m.group(1), m.group(2)
        cnt = int(arr_len) if arr_len else 1
        if base_t in ULOG_TYPE_MAP:
            ch, _ = ULOG_TYPE_MAP[base_t]
            struct_chars.append(f'{cnt}{ch}' if cnt > 1 else ch)
            field_info.append((v_name, cnt, ch))
    fmt_str = ''.join(struct_chars)
    return fmt_str, struct.calcsize(fmt_str), field_info


def parse_ulog(ulog_path, wanted_topics):
    """지정된 토픽들만 선택적으로 고속 언패킹하여 dict of dict(numpy arrays)로 반환."""
    data = Path(ulog_path).read_bytes()
    if data[:7] != b'ULog\x01\x12\x35':
        raise ValueError(f'유효한 PX4 ULog 파일이 아닙니다: {ulog_path}')

    pos = 16
    formats = {}
    subs = {}
    compiled = {}
    records = {t: [] for t in wanted_topics}

    while pos + 3 <= len(data):
        msg_size, msg_type = struct.unpack_from('<HB', data, pos)
        payload_start = pos + 3
        payload_end = payload_start + msg_size
        if payload_end > len(data):
            break
        pos = payload_end

        if msg_type == ord('F'):
            f_str = data[payload_start:payload_end].decode('latin1')
            name, fields_str = f_str.split(':', 1)
            fields = []
            for f in fields_str.split(';'):
                if not f:
                    continue
                t_name, v_name = f.split(' ')
                fields.append((t_name, v_name))
            formats[name] = fields

        elif msg_type == ord('A'):
            multi_id = data[payload_start]
            msg_id = struct.unpack_from('<H', data, payload_start + 1)[0]
            msg_name = data[payload_start + 3:payload_end].decode('latin1')
            if msg_name in wanted_topics and multi_id == 0:
                if msg_name not in compiled and msg_name in formats:
                    compiled[msg_name] = compile_format(msg_name, formats)
                subs[msg_id] = msg_name

        elif msg_type == ord('D'):
            msg_id = struct.unpack_from('<H', data, payload_start)[0]
            msg_name = subs.get(msg_id)
            if msg_name and msg_name in compiled:
                fmt_str, expected_sz, _ = compiled[msg_name]
                d_start = payload_start + 2
                if payload_end - d_start >= expected_sz:
                    records[msg_name].append(struct.unpack_from(fmt_str, data, d_start))

    out = {}
    for msg_name, rows in records.items():
        if not rows or msg_name not in compiled:
            continue
        _, _, field_info = compiled[msg_name]
        arr_dict = {}
        col_idx = 0
        for v_name, cnt, ch in field_info:
            dtype = np.float64 if ch in ('f', 'd') else np.int64
            if cnt == 1:
                arr_dict[v_name] = np.array([r[col_idx] for r in rows], dtype=dtype)
            else:
                arr_dict[v_name] = np.array([r[col_idx:col_idx + cnt] for r in rows], dtype=dtype)
            col_idx += cnt
        out[msg_name] = arr_dict
    return out


def find_latest_ulog():
    candidates = glob.glob('/tmp/px4_balloon_*/px4_rootfs_*/log/*/*.ulg')
    if not candidates:
        raise FileNotFoundError('실행 중인 시뮬레이션(/tmp/px4_balloon_*)에서 .ulg 파일을 찾지 못했습니다.')
    return Path(max(candidates, key=os.path.getmtime))


def generate_plots(ulog_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {
        'vehicle_local_position',
        'vehicle_attitude',
        'vehicle_attitude_setpoint',
        'actuator_motors',
        'vehicle_status',
    }
    parsed = parse_ulog(ulog_path, wanted)
    if 'vehicle_local_position' not in parsed:
        raise RuntimeError('ULog에 vehicle_local_position 데이터가 없습니다.')

    lpos = parsed['vehicle_local_position']
    att = parsed.get('vehicle_attitude', {})
    att_sp = parsed.get('vehicle_attitude_setpoint', {})
    motors = parsed.get('actuator_motors', {})
    vstat = parsed.get('vehicle_status', {})

    t0_us = float(lpos['timestamp'][0])
    if 'timestamp' in vstat and 'arming_state' in vstat:
        armed_idx = np.where(vstat['arming_state'] == 2)[0]
        if len(armed_idx) > 0:
            t0_us = float(vstat['timestamp'][armed_idx[0]])

    t_pos = (lpos['timestamp'] - t0_us) * 1e-6
    mask_pos = t_pos >= -1.0
    t_pos = t_pos[mask_pos]

    n = lpos['x'][mask_pos]
    e = lpos['y'][mask_pos]
    d = lpos['z'][mask_pos]
    vn = lpos['vx'][mask_pos]
    ve = lpos['vy'][mask_pos]
    vd = lpos['vz'][mask_pos]

    wx, wy, wz = ned_to_world(n, e, d)
    spd_3d = np.sqrt(vn**2 + ve**2 + vd**2)
    spd_horiz = np.hypot(vn, ve)
    climb_rate = -vd

    target_ned = world_to_ned(*TARGET_WORLD)
    dist_3d = np.sqrt((n - target_ned[0])**2 + (e - target_ned[1])**2 + (d - target_ned[2])**2)
    dist_horiz = np.hypot(n - target_ned[0], e - target_ned[1])

    cpa_idx = int(np.argmin(dist_3d))
    cpa_time = float(t_pos[cpa_idx])
    cpa_dist = float(dist_3d[cpa_idx])
    cpa_world = (float(wx[cpa_idx]), float(wy[cpa_idx]), float(wz[cpa_idx]))
    cpa_spd = float(spd_3d[cpa_idx])

    t_end_focus = min(float(t_pos[-1]), cpa_time + 8.0)

    t_att = (att['timestamp'] - t0_us) * 1e-6
    m_att = t_att >= -1.0
    t_att = t_att[m_att]
    nose_vec, tilt_deg, elev_deg, az_deg, wing_roll_deg = quat_to_nose_and_euler(att['q'][m_att])

    t_sp = (att_sp['timestamp'] - t0_us) * 1e-6
    m_sp = t_sp >= -1.0
    t_sp = t_sp[m_sp]
    _, tilt_sp_deg, elev_sp_deg, az_sp_deg, _ = quat_to_nose_and_euler(att_sp['q_d'][m_sp])
    thrust_sp = -att_sp['thrust_body'][m_sp, 2]

    los_vec_true = target_ned[None, :] - np.column_stack((n, e, d))
    los_horiz = np.hypot(los_vec_true[:, 0], los_vec_true[:, 1])
    true_los_elev_deg = np.degrees(np.arctan2(-los_vec_true[:, 2], np.maximum(los_horiz, 0.1)))

    # ── Figure 1: 6-Panel 텔레메트리 종합 대시보드 ──
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 2, figsize=(16, 12.5), dpi=160)
    fig.suptitle(
        f'PX4 Quad-Tailsitter (SYS_AUTOSTART=1045) 30cm Balloon Vision Guidance Telemetry\n'
        f'Target World=({TARGET_WORLD[0]:.1f}, {TARGET_WORLD[1]:.1f}, {TARGET_WORLD[2]:.1f})m | '
        f'Closest Approach Distance (CPA) = {cpa_dist:.2f} m @ t = {cpa_time:.2f} s (Speed = {cpa_spd:.1f} m/s)',
        fontsize=13, fontweight='bold', y=0.985
    )

    ax = axes[0, 0]
    ax.plot(t_pos, dist_3d, label='3D Distance to Balloon [m]', color='#1f77b4', lw=2.2)
    ax.plot(t_pos, dist_horiz, label='Horizontal Distance [m]', color='#17becf', lw=1.6, ls='--')
    ax.axvline(cpa_time, color='#d62728', ls=':', lw=1.8, label=f'Pass / CPA ({cpa_dist:.2f}m @ {cpa_time:.1f}s)')
    ax.scatter([cpa_time], [cpa_dist], color='#d62728', s=55, zorder=5)
    ax.axhline(15.0, color='#ff7f0e', ls='-.', lw=1.1, alpha=0.8, label='Lock Gate (15m)')
    ax.set_title('1. Distance to 30cm Balloon & Pass Point (CPA)', fontweight='bold')
    ax.set_xlabel('Time since Arm [s]')
    ax.set_ylabel('Distance [m]')
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper right', fontsize=8.5)

    ax = axes[0, 1]
    ax.plot(t_pos, wz, label='Drone World Altitude Z [m]', color='#2ca02c', lw=2.2)
    ax.axhline(TARGET_WORLD[2], color='#d62728', ls='--', lw=1.5, label=f'Balloon Altitude ({TARGET_WORLD[2]:.1f}m)')
    ax.axhline(HOME_WORLD[2] + 5.0, color='#9467bd', ls=':', lw=1.3, label='Takeoff Hover Alt (5.2m)')
    ax.axvline(cpa_time, color='#d62728', ls=':', lw=1.5)
    ax.scatter([cpa_time], [cpa_world[2]], color='#d62728', s=55, zorder=5,
               label=f'Alt at CPA ({cpa_world[2]:.2f}m)')
    ax.set_title('2. Altitude Profile (Takeoff -> Diagonal Climb -> RTL Landing)', fontweight='bold')
    ax.set_xlabel('Time since Arm [s]')
    ax.set_ylabel('World Altitude Z [m]')
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper right', fontsize=8.5)

    ax = axes[1, 0]
    ax.plot(t_pos, spd_3d, label='3D Total Speed ||v|| [m/s]', color='#d62728', lw=2.1)
    ax.plot(t_pos, spd_horiz, label='Horizontal Speed [m/s]', color='#ff7f0e', lw=1.6, ls='--')
    ax.plot(t_pos, climb_rate, label='Climb Rate (+Up) [m/s]', color='#2ca02c', lw=1.5, alpha=0.85)
    ax.axvline(cpa_time, color='#d62728', ls=':', lw=1.5)
    ax.set_title('3. Tailsitter Velocity Profile (Distance-Proportional Speed Control)', fontweight='bold')
    ax.set_xlabel('Time since Arm [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper right', fontsize=8.5)

    ax = axes[1, 1]
    m_att_f = t_att <= t_end_focus
    m_sp_f = t_sp <= t_end_focus
    m_pos_f = (t_pos <= cpa_time) & (dist_3d > 2.0)
    ax.plot(t_sp[m_sp_f], elev_sp_deg[m_sp_f], label='Commanded Nose/Camera Elev [deg]',
            color='#ff7f0e', lw=1.8, ls='--')
    ax.plot(t_att[m_att_f], elev_deg[m_att_f], label='Actual Nose/Camera Elev [deg]',
            color='#1f77b4', lw=2.1)
    ax.plot(t_pos[m_pos_f], true_los_elev_deg[m_pos_f], label='Geometric LOS Elev to Balloon [deg]',
            color='#d62728', lw=1.5, ls=':')
    ax.plot(t_att[m_att_f], wing_roll_deg[m_att_f], label='Wing Roll Angle (Zero-Roll Lock) [deg]',
            color='#2ca02c', lw=1.4, alpha=0.85)
    ax.axvline(cpa_time, color='#d62728', ls=':', lw=1.5, label='Balloon Pass (CPA)')
    ax.set_title('4. Nose Cone (Camera Optical Axis) Elevation & Zero-Roll Tracking', fontweight='bold')
    ax.set_xlabel('Time since Arm [s] (Approach & Back-Transition Window)')
    ax.set_ylabel('Angle [deg]')
    ax.set_ylim(-25, 95)
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper right', fontsize=8.0)

    ax = axes[2, 0]
    ax.plot(t_sp[m_sp_f], thrust_sp[m_sp_f], label='Normalized Thrust Setpoint', color='#111111', lw=2.2, zorder=4)
    if 'control' in motors:
        t_mot = (motors['timestamp'] - t0_us) * 1e-6
        m_mot = (t_mot >= -1.0) & (t_mot <= t_end_focus)
        colors_m = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
        for i in range(4):
            ax.plot(t_mot[m_mot], motors['control'][m_mot, i],
                    label=f'Rotor {i} Output', color=colors_m[i], lw=1.1, alpha=0.75)
    ax.axvline(cpa_time, color='#d62728', ls=':', lw=1.5)
    ax.set_title('5. Tailsitter Thrust Setpoint & 4-Rotor Differential Allocation', fontweight='bold')
    ax.set_xlabel('Time since Arm [s] (Approach & Back-Transition Window)')
    ax.set_ylabel('Normalized Command [0..1]')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper right', fontsize=8.0, ncol=2)

    ax = axes[2, 1]
    idx_pass_end = np.searchsorted(t_pos, cpa_time + 1.5)
    ax.plot(wx[:idx_pass_end], wz[:idx_pass_end], color='#1f77b4', lw=2.5, label='Approach Trajectory (Tailsitter)')
    ax.plot(wx[idx_pass_end:], wz[idx_pass_end:], color='#7f7f7f', lw=1.6, ls='--', alpha=0.8, label='Braking & RTL Return Trajectory')
    ax.plot([HOME_WORLD[0], TARGET_WORLD[0]], [HOME_WORLD[2] + 5.0, TARGET_WORLD[2]],
            color='#2ca02c', ls=':', lw=1.4, alpha=0.7, label='Ideal 3D Diagonal Line')

    sample_times = np.arange(max(2.0, t_pos[0]), cpa_time, 0.9)
    for st in sample_times:
        pi = int(np.argmin(np.abs(t_pos - st)))
        ai = int(np.argmin(np.abs(t_att - st)))
        nn, ne, nd = nose_vec[ai]
        c_y, s_y = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
        n_wx = c_y * nn + s_y * ne
        n_wz = -nd
        ax.annotate(
            '', xy=(wx[pi] + 5.5 * n_wx, wz[pi] + 5.5 * n_wz),
            xytext=(wx[pi], wz[pi]),
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5, alpha=0.85)
        )
    ax.plot([], [], color='#ff7f0e', lw=1.5, label='Nose Cone / Camera Optical Axis')
    ax.scatter([TARGET_WORLD[0]], [TARGET_WORLD[2]], color='#d62728', s=130, marker='o', zorder=6,
               label=f'30cm Balloon ({TARGET_WORLD[0]:.0f}m, {TARGET_WORLD[2]:.0f}m)')
    ax.scatter([HOME_WORLD[0]], [HOME_WORLD[2]], color='#111111', s=70, marker='^', zorder=6, label='Home / Launch')
    ax.set_title('6. Side-View (World X-Z) Trajectory & Nose Cone Pointing Vectors', fontweight='bold')
    ax.set_xlabel('World X [m]')
    ax.set_ylabel('World Z (Altitude) [m]')
    ax.grid(True, alpha=0.35)
    ax.legend(loc='upper left', fontsize=8.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    dashboard_path = out_dir / 'tailsitter_telemetry_dashboard.png'
    fig.savefig(dashboard_path, dpi=160)
    plt.close(fig)

    # ── Figure 2: 3D 궤적 & 평면도(X-Y) 정렬 정확도 ──
    fig2 = plt.figure(figsize=(15, 6.5), dpi=160)
    fig2.suptitle(
        f'3D Flight Trajectory & Terminal Approach Accuracy (30cm Target Balloon, CPA = {cpa_dist:.2f} m)',
        fontsize=13, fontweight='bold', y=0.98
    )

    ax3d = fig2.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(wx[:idx_pass_end], wy[:idx_pass_end], wz[:idx_pass_end],
              color='#1f77b4', lw=2.5, label='Tailsitter Approach & Pass')
    ax3d.plot(wx[idx_pass_end:], wy[idx_pass_end:], wz[idx_pass_end:],
              color='#7f7f7f', lw=1.5, ls='--', label='Back-Transition & RTL')
    ax3d.scatter([HOME_WORLD[0]], [HOME_WORLD[1]], [HOME_WORLD[2]],
                 color='#111111', s=60, marker='^', label='Home (-2, 0, 0.2)')
    ax3d.scatter([TARGET_WORLD[0]], [TARGET_WORLD[1]], [TARGET_WORLD[2]],
                 color='#d62728', s=120, marker='o', label='30cm Balloon (90, 0, 40)')
    ax3d.set_title('3D Trajectory (World Frame)', fontweight='bold')
    ax3d.set_xlabel('World X [m]')
    ax3d.set_ylabel('World Y [m]')
    ax3d.set_zlabel('World Z [m]')
    ax3d.set_ylim(-15, 15)
    ax3d.legend(loc='upper left', fontsize=8.5)

    ax_xy = fig2.add_subplot(1, 2, 2)
    ax_xy.plot(wx[:idx_pass_end], wy[:idx_pass_end], color='#1f77b4', lw=2.4, label='Approach Track (X-Y)')
    ax_xy.plot(wx[idx_pass_end:], wy[idx_pass_end:], color='#7f7f7f', lw=1.4, ls='--', alpha=0.75, label='RTL Track (X-Y)')
    ax_xy.axhline(0.0, color='#2ca02c', ls=':', lw=1.2, label='Centerline (Y = 0.0 m)')
    ax_xy.scatter([TARGET_WORLD[0]], [TARGET_WORLD[1]], color='#d62728', s=120, zorder=6, label='30cm Balloon Center')
    ax_xy.scatter([cpa_world[0]], [cpa_world[1]], color='#ff7f0e', s=80, marker='x', lw=2.5, zorder=7,
                  label=f'CPA Point ({cpa_world[0]:.2f}, {cpa_world[1]:+.2f}, {cpa_world[2]:.2f})m')
    ax_xy.set_title(
        f'Top-Down (World X-Y) Lateral Alignment\n'
        f'(Lateral Error at CPA: dY = {cpa_world[1] - TARGET_WORLD[1]:+.2f}m, Vertical Error: dZ = {cpa_world[2] - TARGET_WORLD[2]:+.2f}m)',
        fontweight='bold'
    )
    ax_xy.set_xlabel('World X [m]')
    ax_xy.set_ylabel('World Y (Lateral) [m]')
    ax_xy.set_ylim(-5.0, 5.0)
    ax_xy.grid(True, alpha=0.35)
    ax_xy.legend(loc='upper left', fontsize=8.5)

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    traj3d_path = out_dir / 'tailsitter_3d_trajectory.png'
    fig2.savefig(traj3d_path, dpi=160)
    plt.close(fig2)

    summary = {
        'firmware': '1045_gazebo-classic_quadtailsitter',
        'ca_airframe': 4,
        'target_world_m': TARGET_WORLD.tolist(),
        'balloon_diameter_m': 0.30,
        'duration_s': round(float(t_pos[-1]), 2),
        'cpa_time_s': round(cpa_time, 2),
        'cpa_dist_3d_m': round(cpa_dist, 3),
        'cpa_world_m': [round(v, 3) for v in cpa_world],
        'cpa_error_xyz_m': [
            round(cpa_world[0] - float(TARGET_WORLD[0]), 3),
            round(cpa_world[1] - float(TARGET_WORLD[1]), 3),
            round(cpa_world[2] - float(TARGET_WORLD[2]), 3),
        ],
        'cpa_speed_m_s': round(cpa_spd, 2),
        'max_approach_speed_m_s': round(float(np.max(spd_3d[:idx_pass_end])), 2),
        'dashboard_png': str(dashboard_path),
        'trajectory_3d_png': str(traj3d_path),
    }
    summary_path = out_dir / 'tailsitter_flight_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description='PX4 테일시터 ULog 비행 로그 그래프 생성기')
    parser.add_argument('ulog', nargs='?', default=None, help='.ulg 파일 경로 (생략 시 최신 로그 자동 탐색)')
    parser.add_argument('--out-dir', default=str(Path(__file__).resolve().parents[1] / 'plots'),
                        help='그래프 PNG 저장 디렉토리 (기본값: ./plots)')
    args = parser.parse_args()

    ulog_path = Path(args.ulog) if args.ulog else find_latest_ulog()
    print(f'분석 대상 ULog: {ulog_path}', flush=True)
    summary = generate_plots(ulog_path, args.out_dir)

    print(f"총 비행 시간       : {summary['duration_s']:.1f} 초", flush=True)
    print(f"풍선 최근접 시각   : {summary['cpa_time_s']:.2f} 초", flush=True)
    print(f"풍선 최근접 거리   : {summary['cpa_dist_3d_m']:.3f} m (오차 dX={summary['cpa_error_xyz_m'][0]:+.2f}m, "
          f"dY={summary['cpa_error_xyz_m'][1]:+.2f}m, dZ={summary['cpa_error_xyz_m'][2]:+.2f}m)", flush=True)
    print(f"통과 시점 속도     : {summary['cpa_speed_m_s']:.2f} m/s (접근 최대 속도: {summary['max_approach_speed_m_s']:.2f} m/s)", flush=True)
    print(f"대시보드 그래프    : {summary['dashboard_png']}", flush=True)
    print(f"3D 궤적 그래프     : {summary['trajectory_3d_png']}", flush=True)


if __name__ == '__main__':
    main()
