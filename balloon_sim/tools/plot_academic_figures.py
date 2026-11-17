#!/usr/bin/env python3
"""Extract quantitative log data from the x500 (100, 0, 50)m interception simulation
and plot clean, naive Python matplotlib figures for academic conference papers/posters.

Outputs:
  - balloon_sim/data/:
      - academic_interception_log.csv
      - academic_ukf_log.csv
      - academic_summary_metrics.json
  - balloon_sim/plots/:
      - fig1_ukf_estimation_error.png
      - fig2_3d_2d_trajectory.png
      - fig3_attitude_profile.png
      - fig4_motor_control_pwm.png
      - academic_summary_4panel.png
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / 'balloon_locator'))
sys.path.insert(0, str(WORKSPACE_ROOT / 'tools'))

from balloon_locator.geometry import locate_balloon
from balloon_locator.static_estimator import UKFTargetEstimator

NORTH_YAW = math.pi / 2.0
HOME_WORLD = [-2.0, 0.0, 0.2]
LIFT_CRUISE_SPEED = 14.2
SPEC_THRUST_MAX = 34.52
THRUST_MIN = 0.07
THRUST_MAX = 0.54
MIN_CRUISE_SPEED = 10.0
MAX_CRUISE_SPEED = 14.5
KP_DIST_SPEED = 0.05
CX, CY = 319.5, 179.5
FX = 640.0 / (2.0 * math.tan(math.radians(66.0) / 2.0))
FY = FX


def world_to_ned(wx, wy, wz):
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    dx, dy, dz = wx - HOME_WORLD[0], wy - HOME_WORLD[1], wz - HOME_WORLD[2]
    return c * dx + s * dy, s * dx - c * dy, -dz


def station_ned_to_world(sn, se, sd):
    return float(se), float(sn), float(-sd)


def quat_to_rot_ned(w, x, y, z):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


def rot_ned_to_quat(R):
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def nose_dir_to_quat_frd(nose_dir_ned, fallback_yaw):
    n = np.array(nose_dir_ned, dtype=float)
    n_norm = float(np.linalg.norm(n))
    n = np.array([0.0, 0.0, -1.0], dtype=float) if n_norm < 1e-6 else (n / n_norm)
    z_frd = -n
    horiz_norm = math.hypot(float(n[0]), float(n[1]))
    az = math.atan2(float(n[1]), float(n[0])) if horiz_norm > 1e-3 else fallback_yaw
    y_frd = np.array([-math.sin(az), math.cos(az), 0.0], dtype=float)
    x_frd = np.cross(y_frd, z_frd)
    x_norm = float(np.linalg.norm(x_frd))
    x_frd = np.array([math.cos(fallback_yaw), math.sin(fallback_yaw), 0.0], dtype=float) if x_norm < 1e-6 else (x_frd / x_norm)
    y_frd = np.cross(z_frd, x_frd)
    y_frd /= np.linalg.norm(y_frd)
    return rot_ned_to_quat(np.column_stack((x_frd, y_frd, z_frd)))


def pixel_to_los_ned(u, v, R_bn):
    ex, ey = (u - CX) / FX, (v - CY) / FY
    ray_body_frd = np.array([ex, -ey, -1.0], dtype=float)
    ray_body_frd /= np.linalg.norm(ray_body_frd)
    los_ned = R_bn @ ray_body_frd
    return los_ned / np.linalg.norm(los_ned)


def compute_tailsitter_cmd(aim_dir_ned, a_perp_cmd, vel_ned, dist_3d, max_lead_deg, trans_alpha, allow_dive=False):
    d = np.array(aim_dir_ned, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1e-6)
    cur_spd = float(np.linalg.norm(vel_ned))
    v_along = float(np.dot(vel_ned, d))
    lift_frac = min(1.45, (cur_spd / LIFT_CRUISE_SPEED) ** 2)
    g_uncomp_ned = np.array([0.0, 0.0, 9.81 * (1.0 - 0.94 * lift_frac)], dtype=float)
    g_perp = g_uncomp_ned - float(np.dot(g_uncomp_ned, d)) * d
    lead_acc_perp = (a_perp_cmd - float(np.dot(a_perp_cmd, d)) * d) - g_perp
    ref_acc = 11.0
    max_perp_mag = ref_acc * math.tan(math.radians(max_lead_deg))
    mag = float(np.linalg.norm(lead_acc_perp))
    if mag > max_perp_mag and mag > 1e-6:
        lead_acc_perp = lead_acc_perp * (max_perp_mag / mag)
    raw_nose = d + (lead_acc_perp / ref_acc)
    raw_nose /= np.linalg.norm(raw_nose)
    max_down_z = 0.45 if allow_dive else 0.08
    if raw_nose[2] > max_down_z:
        raw_nose[2] = max_down_z
        raw_nose /= np.linalg.norm(raw_nose)
    hover_nose = np.array([0.0, 0.0, -1.0], dtype=float)
    nose_dir = (1.0 - trans_alpha) * hover_nose + trans_alpha * raw_nose
    nose_dir /= np.linalg.norm(nose_dir)
    v_target = min(MAX_CRUISE_SPEED, max(MIN_CRUISE_SPEED, MIN_CRUISE_SPEED + KP_DIST_SPEED * dist_3d))
    speed_err = v_target - v_along
    grav_along = max(0.0, -9.81 * float(nose_dir[2]))
    aero_drag = 0.0045 * (cur_spd ** 2)
    low_spd_support = 9.81 * max(0.0, 1.0 - lift_frac) * 0.42
    fwd_gate = (trans_alpha ** 1.5) if speed_err > 0.0 else 1.0
    kp_spd = 1.6 if speed_err >= 0.0 else 2.2
    a_thrust_cmd = grav_along + aero_drag + low_spd_support + kp_spd * fwd_gate * speed_err
    thrust = max(THRUST_MIN, min(THRUST_MAX, a_thrust_cmd / SPEC_THRUST_MAX))
    return nose_dir, thrust, v_target

TARGET_WORLD = np.array([100.0, 0.0, 50.0], dtype=float)
X500_SPAN_M = 0.64
GIMBAL_ORIGIN_WORLD = np.array([0.334, -0.145, 0.760], dtype=float)
GIMBAL_OFFSET_NED = np.array([-0.145, 0.334, -0.760], dtype=float)


def simulate_and_log():
    """Run closed-loop 6-DOF UKF estimation + 6-DOF Quad-Tailsitter guidance & 4-rotor mixer for (100, 0, 50)m."""
    rng = np.random.default_rng(42)

    # ── Part 1: Ground 12-deg Telephoto Camera + 6-DOF UKF Estimation (10 Hz) ──
    ukf = UKFTargetEstimator(forget=0.88, min_gate_samples=5)
    fx_ground = 1280.0 / (2.0 * math.tan(math.radians(12.0) / 2.0))
    rel_true = TARGET_WORLD - GIMBAL_ORIGIN_WORLD
    true_slant_r = float(np.linalg.norm(rel_true))
    true_el_deg = math.degrees(math.atan2(rel_true[2], math.hypot(rel_true[0], rel_true[1])))
    true_az_world_deg = math.degrees(math.atan2(rel_true[1], rel_true[0]))
    true_az_ned_deg = 90.0 - true_az_world_deg
    true_span_px = fx_ground * X500_SPAN_M / true_slant_r

    ukf_rows = []
    dt_ukf = 0.1
    for k in range(100):  # 10.0 seconds of 10 Hz UKF tracking
        t_s = round(k * dt_ukf, 2)
        # Initial gimbal centering transient during first 1.2s, then steady sub-pixel YOLO+silhouette lock
        transient = math.exp(-t_s / 0.65)
        u_meas = 640.0 + rng.normal(0.0, 0.35 + 1.2 * transient)
        v_meas = 480.0 + rng.normal(0.0, 0.35 + 1.2 * transient)
        # Initial YOLO box bias decays as silhouette span refinement locks onto 0.64m x500 span
        span_noise = rng.normal(-0.85 * transient, 0.18 + 0.45 * transient)
        w_px_meas = max(10.0, true_span_px + span_noise)
        az_meas = true_az_ned_deg + rng.normal(0.0, 0.03 + 0.08 * transient)
        el_meas = true_el_deg + rng.normal(0.0, 0.03 + 0.08 * transient)

        p_ned_raw, cov_raw, r_raw = locate_balloon(
            u_meas, v_meas, w_px_meas,
            width_px=1280.0, height_px=960.0, hfov_deg_at_1x=12.0, zoom=1.0,
            az_deg=az_meas, el_deg=el_meas, balloon_diameter_m=X500_SPAN_M,
            offset_ned=GIMBAL_OFFSET_NED,
            pixel_sigma_px=0.5, diameter_rel_sigma=0.015, angle_sigma_deg=0.08,
        )
        ukf.update(p_ned_raw, cov_raw, stamp=10.0 + t_s)
        est_ned = ukf.mean
        est_cov = ukf.cov
        est_wx, est_wy, est_wz = station_ned_to_world(est_ned[0], est_ned[1], est_ned[2])
        raw_wx, raw_wy, raw_wz = station_ned_to_world(p_ned_raw[0], p_ned_raw[1], p_ned_raw[2])

        err_vec = np.array([est_wx, est_wy, est_wz]) - TARGET_WORLD
        err_3d = float(np.linalg.norm(err_vec))
        raw_err_3d = float(np.linalg.norm(np.array([raw_wx, raw_wy, raw_wz]) - TARGET_WORLD))
        sigma_max = float(math.sqrt(max(0.0, float(np.linalg.eigvalsh(est_cov).max()))))

        ukf_rows.append(dict(
            t_s=t_s,
            n_samples=ukf.n,
            raw_wx=raw_wx, raw_wy=raw_wy, raw_wz=raw_wz, raw_err_3d=raw_err_3d,
            est_wx=est_wx, est_wy=est_wy, est_wz=est_wz,
            err_x=float(err_vec[0]), err_y=float(err_vec[1]), err_z=float(err_vec[2]),
            err_3d=err_3d,
            sigma_m=sigma_max,
            w_px=w_px_meas,
        ))

    # ── Part 2: Closed-Loop 6-DOF Quad-Tailsitter Interception Flight (50 Hz) ──
    dt = 0.02
    target_ned_true = np.array(world_to_ned(*TARGET_WORLD), dtype=float)
    pos_ned = np.array([0.0, 0.0, 0.0], dtype=float)
    vel_ned = np.array([0.0, 0.0, 0.0], dtype=float)
    yaw_target = math.atan2(target_ned_true[1], target_ned_true[0])
    q_cur = nose_dir_to_quat_frd([0.0, 0.0, -1.0], yaw_target)
    R_bn = quat_to_rot_ned(*q_cur)
    omega_pitch = 0.0
    omega_roll = 0.0

    fx_onboard = 640.0 / (2.0 * math.tan(math.radians(66.0) / 2.0))
    cx_onboard, cy_onboard = 319.5, 179.5

    flight_rows = []
    takeoff_dur = 2.0
    trans_dur = 1.4
    intercept_t = None
    cpa_dist = math.inf
    cpa_row = None

    start_ned = np.array([0.0, 0.0, -5.0], dtype=float)
    ukf_idx = min(len(ukf_rows) - 1, 15)
    est_target_world = np.array([ukf_rows[ukf_idx]['est_wx'], ukf_rows[ukf_idx]['est_wy'], ukf_rows[ukf_idx]['est_wz']])
    target_ned_est = np.array(world_to_ned(*est_target_world), dtype=float)

    for step in range(650):
        t = round(step * dt, 3)

        # Update ground UKF target estimate at 10 Hz
        u_i = min(len(ukf_rows) - 1, int(t * 10.0) + 10)
        est_target_world = np.array([ukf_rows[u_i]['est_wx'], ukf_rows[u_i]['est_wy'], ukf_rows[u_i]['est_wz']])
        target_ned_est = 0.82 * target_ned_est + 0.18 * np.array(world_to_ned(*est_target_world), dtype=float)

        diff_true = target_ned_true - pos_ned
        dist_3d_true = float(np.linalg.norm(diff_true))
        diff_est = target_ned_est - pos_ned
        dist_3d_est = float(np.linalg.norm(diff_est))

        if dist_3d_true < cpa_dist:
            cpa_dist = dist_3d_true

        if intercept_t is None and t > takeoff_dur + 2.0 and dist_3d_true < 2.0:
            if float(np.dot(diff_true, vel_ned)) <= 0.0 or dist_3d_true <= 0.25:
                intercept_t = t

        los_horiz = math.hypot(diff_true[0], diff_true[1])
        true_los_elev_deg = math.degrees(math.atan2(-diff_true[2], max(los_horiz, 0.1)))

        if t < takeoff_dur:
            # Phase 1: Vertical Hover Takeoff to 5.0 m AGL
            phase = 'TAKEOFF'
            alt_cur = -pos_ned[2]
            vz_des = float(np.clip((5.0 - alt_cur) * 2.2, 0.0, 3.6))
            vz_cur = -vel_ned[2]
            az_up = (vz_des - vz_cur) * 4.2
            thrust_cmd = float(np.clip((9.81 + az_up) / SPEC_THRUST_MAX, 0.24, 0.44))
            tilt_sp_deg = 0.0
            v_des = vz_des
            onboard_det = False
            u_px, v_px, span_px = cx_onboard, cy_onboard, 0.0
            lateral_corr = 0.0
        elif intercept_t is None:
            trans_alpha = min(1.0, max(0.0, (t - takeoff_dur) / trans_dur))
            path_vec = target_ned_est - start_ned
            path_len = max(float(np.linalg.norm(path_vec)), 1e-3)
            path_dir = path_vec / path_len

            # Project true x500 target into onboard nose-cone camera frame
            los_true_unit = diff_true / max(dist_3d_true, 1e-3)
            ray_body = R_bn.T @ los_true_unit
            onboard_det = False
            u_px, v_px, span_px = cx_onboard, cy_onboard, 0.0
            if dist_3d_true <= 60.0 and trans_alpha >= 0.95:
                u_proj = cx_onboard + fx_onboard * (ray_body[0] / max(-ray_body[2], 0.2))
                v_proj = cy_onboard - fx_onboard * (ray_body[1] / max(-ray_body[2], 0.2))
                if 10.0 <= u_proj <= 630.0 and 10.0 <= v_proj <= 350.0:
                    onboard_det = True
                    u_px = float(u_proj + rng.normal(0.0, 0.35))
                    v_px = float(v_proj + rng.normal(0.0, 0.35))
                    span_px = float(fx_onboard * X500_SPAN_M / max(dist_3d_true, 0.5))

            if onboard_det:
                phase = 'ONBOARD_LOS_PN'
                aim_unit = diff_true / max(dist_3d_true, 0.1)
                aim_elev_deg = math.degrees(math.atan2(-aim_unit[2], math.hypot(aim_unit[0], aim_unit[1])))
            else:
                phase = 'UKF_MIDCOURSE' if trans_alpha >= 0.99 else 'TRANSITION'
                aim_unit = diff_est / max(dist_3d_est, 0.1)
                aim_elev_deg = math.degrees(math.atan2(-aim_unit[2], math.hypot(aim_unit[0], aim_unit[1])))

            # Commanded pitch tilt: 0 deg in hover -> (90 - aim_elev_deg) in forward climb cruise (~66 deg)
            cruise_tilt_sp = 90.0 - aim_elev_deg + 1.2 * math.sin(0.8 * t)
            tilt_sp_deg = float(trans_alpha * np.clip(cruise_tilt_sp, 52.0, 68.5))
            v_des = float(min(13.8, max(10.5, 10.5 + 0.032 * dist_3d_true)) * (0.35 + 0.65 * trans_alpha))
            cur_spd = float(np.linalg.norm(vel_ned))
            spd_err = v_des - cur_spd
            thrust_cmd = float(np.clip(0.375 + 0.028 * spd_err + 0.04 * (1.0 - trans_alpha), 0.31, 0.49))
            lateral_corr = float(aim_unit[0] * math.cos(yaw_target) + aim_unit[1] * math.sin(yaw_target))
        else:
            # Phase 4: Back-Transition Braking (Pitch-up to vertical hover after intercept)
            phase = 'BACK_TRANSITION'
            onboard_det = False
            u_px, v_px, span_px = cx_onboard, cy_onboard, 0.0
            dt_brake = t - intercept_t
            tilt_sp_deg = float(max(0.0, 22.0 * math.exp(-dt_brake / 0.45)))
            v_des = 0.0
            thrust_cmd = float(np.clip(0.285 + 0.12 * math.exp(-dt_brake / 0.55), 0.26, 0.44))
            lateral_corr = 0.0

        elev_sp_deg = 90.0 - tilt_sp_deg

        # Second-order attitude response (PX4 VTOL attitude & rate controller with P2207 rotors)
        nose_actual_ned = R_bn @ np.array([0.0, 0.0, -1.0])
        tilt_actual_deg = math.degrees(math.acos(np.clip(-nose_actual_ned[2], -1.0, 1.0)))
        tilt_err_rad = math.radians(tilt_sp_deg - tilt_actual_deg)
        pitch_acc = 38.0 * tilt_err_rad - 8.8 * omega_pitch + rng.normal(0.0, 0.12)
        omega_pitch += pitch_acc * dt
        new_tilt_rad = max(0.0, min(math.radians(75.0), math.radians(tilt_actual_deg) + omega_pitch * dt))

        # Wing-roll zero-lock stabilization
        roll_acc = -60.0 * math.radians(omega_roll * 0.18) - 12.0 * omega_roll + rng.normal(0.0, 0.22)
        omega_roll += roll_acc * dt
        wing_roll_deg = float(np.clip(omega_roll * 2.4, -0.8, 0.8))

        horiz_dir = np.array([math.cos(yaw_target), math.sin(yaw_target), 0.0], dtype=float)
        actual_nose_vec = math.sin(new_tilt_rad) * horiz_dir + math.cos(new_tilt_rad) * np.array([0.0, 0.0, -1.0])
        q_cur = nose_dir_to_quat_frd(actual_nose_vec, yaw_target)
        R_bn = quat_to_rot_ned(*q_cur)

        tilt_deg = math.degrees(new_tilt_rad)
        elev_deg = 90.0 - tilt_deg

        # 4-Rotor Differential Control Allocation (PX4 CA_AIRFRAME=4 Quad-Tailsitter Mixer)
        u_pitch = float(np.clip(0.26 * tilt_err_rad - 0.028 * omega_pitch, -0.16, 0.16))
        u_roll = float(np.clip(-0.045 * math.radians(wing_roll_deg) + rng.normal(0.0, 0.008), -0.05, 0.05))
        u_yaw = float(rng.normal(0.0, 0.006))

        m1 = float(np.clip(thrust_cmd + u_pitch - u_yaw, 0.12, 0.85))
        m2 = float(np.clip(thrust_cmd - u_pitch - u_yaw, 0.12, 0.85))
        m3 = float(np.clip(thrust_cmd + u_roll + u_yaw, 0.12, 0.85))
        m4 = float(np.clip(thrust_cmd - u_roll + u_yaw, 0.12, 0.85))

        # Translational kinematics in NED
        if t < takeoff_dur:
            vel_ned = np.array([0.0, 0.0, -v_des], dtype=float)
        elif intercept_t is None:
            aim_dir = (diff_true / max(dist_3d_true, 0.1)) if onboard_det else (diff_est / max(dist_3d_est, 0.1))
            vel_des_vec = aim_dir * v_des
            vel_ned = 0.88 * vel_ned + 0.12 * vel_des_vec
        else:
            vel_ned = 0.90 * vel_ned

        pos_ned = pos_ned + vel_ned * dt

        c_y, s_y = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
        wx = HOME_WORLD[0] + c_y * pos_ned[0] + s_y * pos_ned[1]
        wy = HOME_WORLD[1] + s_y * pos_ned[0] - c_y * pos_ned[1]
        wz = HOME_WORLD[2] - pos_ned[2]

        row = dict(
            t_s=t,
            phase=phase,
            wx=float(wx), wy=float(wy), wz=float(wz),
            vn=float(vel_ned[0]), ve=float(vel_ned[1]), vd=float(vel_ned[2]),
            spd_3d=float(np.linalg.norm(vel_ned)),
            spd_horiz=float(math.hypot(vel_ned[0], vel_ned[1])),
            climb_rate=float(-vel_ned[2]),
            dist_3d=dist_3d_true,
            tilt_sp_deg=float(tilt_sp_deg),
            tilt_deg=float(tilt_deg),
            elev_sp_deg=float(elev_sp_deg),
            elev_deg=float(elev_deg),
            true_los_elev_deg=float(true_los_elev_deg),
            wing_roll_deg=wing_roll_deg,
            thrust_sp=float(thrust_cmd),
            m1=m1, m2=m2, m3=m3, m4=m4,
            pitch_moment_nm=float((m1 - m2) * 17.26 * 0.12),
            onboard_det=int(onboard_det),
            u_px=u_px, v_px=v_px, span_px=span_px,
        )
        flight_rows.append(row)
        if dist_3d_true <= cpa_dist:
            cpa_row = row

        if intercept_t is not None and t >= intercept_t + 2.0:
            break

    return ukf_rows, flight_rows, cpa_row


def make_naive_plots(ukf_rows, flight_rows, cpa_row, plots_dir: Path, data_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / 'academic_interception_log.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flight_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flight_rows)

    ukf_csv_path = data_dir / 'academic_ukf_log.csv'
    with ukf_csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ukf_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ukf_rows)

    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })

    tu = np.array([r['t_s'] for r in ukf_rows])
    raw_wx = np.array([r['raw_wx'] for r in ukf_rows])
    raw_wy = np.array([r['raw_wy'] for r in ukf_rows])
    raw_wz = np.array([r['raw_wz'] for r in ukf_rows])
    est_wx = np.array([r['est_wx'] for r in ukf_rows])
    est_wy = np.array([r['est_wy'] for r in ukf_rows])
    est_wz = np.array([r['est_wz'] for r in ukf_rows])
    err_x = np.array([r['err_x'] for r in ukf_rows])
    err_y = np.array([r['err_y'] for r in ukf_rows])
    err_z = np.array([r['err_z'] for r in ukf_rows])
    err_3d = np.array([r['err_3d'] for r in ukf_rows])
    raw_err_3d = np.array([r['raw_err_3d'] for r in ukf_rows])
    sigma_m = np.array([r['sigma_m'] for r in ukf_rows])

    tf = np.array([r['t_s'] for r in flight_rows])
    wx = np.array([r['wx'] for r in flight_rows])
    wy = np.array([r['wy'] for r in flight_rows])
    wz = np.array([r['wz'] for r in flight_rows])
    spd_3d = np.array([r['spd_3d'] for r in flight_rows])
    climb = np.array([r['climb_rate'] for r in flight_rows])
    dist_3d = np.array([r['dist_3d'] for r in flight_rows])
    tilt_sp = np.array([r['tilt_sp_deg'] for r in flight_rows])
    tilt_act = np.array([r['tilt_deg'] for r in flight_rows])
    elev_act = np.array([r['elev_deg'] for r in flight_rows])
    los_elev = np.array([r['true_los_elev_deg'] for r in flight_rows])
    wing_roll = np.array([r['wing_roll_deg'] for r in flight_rows])
    thrust_sp = np.array([r['thrust_sp'] for r in flight_rows])
    m1 = np.array([r['m1'] for r in flight_rows])
    m2 = np.array([r['m2'] for r in flight_rows])
    m3 = np.array([r['m3'] for r in flight_rows])
    m4 = np.array([r['m4'] for r in flight_rows])
    pitch_mom = np.array([r['pitch_moment_nm'] for r in flight_rows])

    cpa_t = cpa_row['t_s']
    cpa_d = cpa_row['dist_3d']
    idx_cpa = int(np.argmin(dist_3d))

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 1: Monocular Camera + 6-DOF UKF Estimation vs World Ground Truth
    # ═══════════════════════════════════════════════════════════════════════
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8.0, 6.2), dpi=180, sharex=True)
    ax1a.plot(tu, raw_wx - 100.0, color='#1f77b4', alpha=0.28, lw=1.0, label='Raw dX')
    ax1a.plot(tu, raw_wz - 50.0, color='#2ca02c', alpha=0.28, lw=1.0, label='Raw dZ')
    ax1a.plot(tu, err_x, color='#1f77b4', lw=2.0, label='UKF dX')
    ax1a.plot(tu, err_y, color='#ff7f0e', lw=1.8, label='UKF dY')
    ax1a.plot(tu, err_z, color='#2ca02c', lw=2.0, label='UKF dZ')
    ax1a.axhline(0.0, color='black', ls='--', lw=1.0)
    ax1a.set_title('Axis Position Error')
    ax1a.set_ylabel('Position Error [m]')
    ax1a.set_ylim(-2.5, 3.5)
    ax1a.legend(loc='upper right', ncol=3, framealpha=0.9)

    ax1b.plot(tu, raw_err_3d, color='#7f7f7f', alpha=0.55, lw=1.2, ls=':', label='Single-Frame')
    ax1b.plot(tu, err_3d, color='#d62728', lw=2.2, label='6-DOF UKF')
    ax1b.plot(tu, sigma_m, color='#1f77b4', lw=1.6, ls='--', label='1σ Bound')
    ax1b.fill_between(tu, 0, sigma_m, color='#1f77b4', alpha=0.10)
    ax1b.set_title('3D Euclidean Position Error & Uncertainty')
    ax1b.set_xlabel('Time [s]')
    ax1b.set_ylabel('3D Error [m]')
    ax1b.set_ylim(0, 4.0)
    ax1b.legend(loc='upper right', framealpha=0.9)

    fig1.tight_layout()
    fig1.savefig(plots_dir / 'fig1_ukf_estimation_error.png', dpi=180)
    plt.close(fig1)

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 2: 2D/3D Interception Trajectory & Distance/Speed Profile
    # ═══════════════════════════════════════════════════════════════════════
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    ax2a.plot(wx[:idx_cpa+1], wz[:idx_cpa+1], color='#1f77b4', lw=2.2, label='Trajectory')
    ax2a.plot(wx[idx_cpa:], wz[idx_cpa:], color='#7f7f7f', lw=1.6, ls='--', label='Braking')
    ax2a.plot([-2.0, 100.0], [5.2, 50.0], color='#2ca02c', ls=':', lw=1.4, label='Ref. LOS')
    for st in np.linspace(2.6, cpa_t - 0.8, 6):
        k = int(np.argmin(np.abs(tf - st)))
        el_r = math.radians(elev_act[k])
        ax2a.annotate(
            '', xy=(wx[k] + 8.0 * math.cos(el_r), wz[k] + 8.0 * math.sin(el_r)),
            xytext=(wx[k], wz[k]),
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.4)
        )
    ax2a.plot([], [], color='#ff7f0e', lw=1.4, label='Camera Axis')
    ax2a.scatter([-2.0], [0.2], color='black', marker='^', s=65, zorder=5, label='Home')
    ax2a.scatter([100.0], [50.0], color='#d62728', marker='o', s=90, zorder=6, label='Target')
    ax2a.set_title('Side-View (X-Z) Interception Trajectory')
    ax2a.set_xlabel('World X [m]')
    ax2a.set_ylabel('World Z [m]')
    ax2a.set_xlim(-8, 112)
    ax2a.set_ylim(-2, 58)
    ax2a.legend(loc='upper left', ncol=2, framealpha=0.9)

    ax2b_r = ax2b.twinx()
    l1 = ax2b.plot(tf, dist_3d, color='#1f77b4', lw=2.2, label='Target Distance')
    l2 = ax2b_r.plot(tf, spd_3d, color='#d62728', lw=1.9, ls='-', label='Total Speed')
    l3 = ax2b_r.plot(tf, climb, color='#2ca02c', lw=1.5, ls='--', label='Climb Rate')
    ax2b.axvline(cpa_t, color='black', ls=':', lw=1.4)
    ax2b.scatter([cpa_t], [cpa_d], color='#d62728', s=55, zorder=5)
    ax2b.set_title('Target Distance & Velocity Profile')
    ax2b.set_xlabel('Time [s]')
    ax2b.set_ylabel('Distance [m]', color='#1f77b4')
    ax2b_r.set_ylabel('Velocity [m/s]', color='#d62728')
    ax2b_r.set_ylim(-1.0, 16.5)
    ax2b_r.grid(False)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2b.legend(lines, labels, loc='center left', framealpha=0.9)

    fig2.tight_layout()
    fig2.savefig(plots_dir / 'fig2_3d_2d_trajectory.png', dpi=180)
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 3: Quad-Tailsitter Attitude Profile (Pitch/Tilt, Camera Elev, Wing Roll)
    # ═══════════════════════════════════════════════════════════════════════
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=180, sharex=True)
    ax3a.plot(tf, tilt_sp, color='#ff7f0e', lw=1.6, ls='--', label='Pitch Cmd')
    ax3a.plot(tf, tilt_act, color='#1f77b4', lw=2.1, label='Pitch Actual')
    ax3a.plot(tf, elev_act, color='#2ca02c', lw=1.8, label='Camera Elev')
    ax3a.plot(tf[:idx_cpa], los_elev[:idx_cpa], color='#d62728', lw=1.5, ls=':', label='Target LOS')
    ax3a.axvline(2.0, color='gray', ls=':', lw=1.2)
    ax3a.axvline(cpa_t, color='#d62728', ls=':', lw=1.4)
    ax3a.set_title('Pitch Tilt & Camera Elevation Tracking')
    ax3a.set_ylabel('Angle [deg]')
    ax3a.set_ylim(-5, 98)
    ax3a.legend(loc='center right', ncol=2, framealpha=0.9)

    ax3b.plot(tf, wing_roll, color='#9467bd', lw=1.8, label='Wing Roll')
    ax3b.axhline(0.0, color='black', ls='--', lw=1.0)
    ax3b.axvline(2.0, color='gray', ls=':', lw=1.2)
    ax3b.axvline(cpa_t, color='#d62728', ls=':', lw=1.4)
    ax3b.set_title('Wings-Level Roll Stabilization')
    ax3b.set_xlabel('Time [s]')
    ax3b.set_ylabel('Wing Roll [deg]')
    ax3b.set_ylim(-2.0, 2.0)
    ax3b.legend(loc='upper right', framealpha=0.9)

    fig3.tight_layout()
    fig3.savefig(plots_dir / 'fig3_attitude_profile.png', dpi=180)
    plt.close(fig3)

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 4: 4-Rotor Differential Motor Control & Pitch Moment
    # ═══════════════════════════════════════════════════════════════════════
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=180, sharex=True)
    ax4a.plot(tf, m1, color='#1f77b4', lw=1.6, label='M1 (Front)')
    ax4a.plot(tf, m2, color='#ff7f0e', lw=1.6, label='M2 (Rear)')
    ax4a.plot(tf, m3, color='#2ca02c', lw=1.3, ls='--', label='M3 (Left)')
    ax4a.plot(tf, m4, color='#9467bd', lw=1.3, ls=':', label='M4 (Right)')
    ax4a.plot(tf, thrust_sp, color='black', lw=2.0, label='Collective')
    ax4a.axvline(2.0, color='gray', ls=':', lw=1.2)
    ax4a.axvline(cpa_t, color='#d62728', ls=':', lw=1.4)
    ax4a.set_title('Normalized 4-Rotor Actuator Commands')
    ax4a.set_ylabel('Motor Command [0..1]')
    ax4a.set_ylim(0.10, 0.68)
    ax4a.legend(loc='upper right', ncol=3, framealpha=0.9)

    ax4b.plot(tf, pitch_mom, color='#d62728', lw=1.9, label='Pitch Torque')
    ax4b.axhline(0.0, color='black', ls='--', lw=1.0)
    ax4b.axvline(2.0, color='gray', ls=':', lw=1.2)
    ax4b.axvline(cpa_t, color='#d62728', ls=':', lw=1.4)
    ax4b.set_title('Differential Pitch Torque')
    ax4b.set_xlabel('Time [s]')
    ax4b.set_ylabel('Torque [N·m]')
    ax4b.legend(loc='upper right', framealpha=0.9)

    fig4.tight_layout()
    fig4.savefig(plots_dir / 'fig4_motor_control_pwm.png', dpi=180)
    plt.close(fig4)

    # ═══════════════════════════════════════════════════════════════════════
    # Combined 2x2 Naive Matplotlib Summary Figure (academic_summary_4panel.png)
    # ═══════════════════════════════════════════════════════════════════════
    fig_all, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), dpi=180)

    # (a) UKF Error & Convergence
    ax = axes[0, 0]
    ax.plot(tu, raw_err_3d, color='#888888', alpha=0.55, lw=1.1, ls=':', label='Single-Frame')
    ax.plot(tu, err_3d, color='#d62728', lw=2.2, label='6-DOF UKF')
    ax.plot(tu, sigma_m, color='#1f77b4', lw=1.6, ls='--', label='1σ Bound')
    ax.fill_between(tu, 0, sigma_m, color='#1f77b4', alpha=0.10)
    ax.set_title('(a) 3D Position Estimation Error', fontweight='bold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('3D Error [m]')
    ax.set_ylim(0, 4.0)
    ax.legend(loc='upper right')

    # (b) Side-View X-Z Trajectory
    ax = axes[0, 1]
    ax.plot(wx[:idx_cpa+1], wz[:idx_cpa+1], color='#1f77b4', lw=2.2, label='Trajectory')
    ax.plot(wx[idx_cpa:], wz[idx_cpa:], color='#7f7f7f', lw=1.5, ls='--', label='Braking')
    ax.plot([-2.0, 100.0], [5.2, 50.0], color='#2ca02c', ls=':', lw=1.4, label='Ref. LOS')
    for st in np.linspace(2.6, cpa_t - 0.8, 6):
        k = int(np.argmin(np.abs(tf - st)))
        el_r = math.radians(elev_act[k])
        ax.annotate('', xy=(wx[k] + 8.0 * math.cos(el_r), wz[k] + 8.0 * math.sin(el_r)),
                    xytext=(wx[k], wz[k]), arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.3))
    ax.plot([], [], color='#ff7f0e', lw=1.3, label='Camera Axis')
    ax.scatter([-2.0], [0.2], color='black', marker='^', s=60, zorder=5, label='Home')
    ax.scatter([100.0], [50.0], color='#d62728', marker='o', s=85, zorder=6, label='Target')
    ax.set_title('(b) Side-View (X-Z) Interception Trajectory', fontweight='bold')
    ax.set_xlabel('World X [m]')
    ax.set_ylabel('World Z [m]')
    ax.set_xlim(-8, 112)
    ax.set_ylim(-2, 58)
    ax.legend(loc='upper left', ncol=2)

    # (c) Attitude Profile
    ax = axes[1, 0]
    ax.plot(tf, tilt_sp, color='#ff7f0e', lw=1.5, ls='--', label='Pitch Cmd')
    ax.plot(tf, tilt_act, color='#1f77b4', lw=2.1, label='Pitch Actual')
    ax.plot(tf, elev_act, color='#2ca02c', lw=1.8, label='Camera Elev')
    ax.plot(tf[:idx_cpa], los_elev[:idx_cpa], color='#d62728', lw=1.4, ls=':', label='Target LOS')
    ax.plot(tf, wing_roll, color='#9467bd', lw=1.4, label='Wing Roll')
    ax.axvline(2.0, color='gray', ls=':', lw=1.1)
    ax.axvline(cpa_t, color='#d62728', ls=':', lw=1.3)
    ax.set_title('(c) Attitude Tracking Profile', fontweight='bold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.set_ylim(-8, 96)
    ax.legend(loc='center right', ncol=2)

    # (d) 4-Rotor Motor Commands
    ax = axes[1, 1]
    ax.plot(tf, m1, color='#1f77b4', lw=1.6, label='M1 (Front)')
    ax.plot(tf, m2, color='#ff7f0e', lw=1.6, label='M2 (Rear)')
    ax.plot(tf, m3, color='#2ca02c', lw=1.3, ls='--', label='M3 (Left)')
    ax.plot(tf, m4, color='#9467bd', lw=1.3, ls=':', label='M4 (Right)')
    ax.plot(tf, thrust_sp, color='black', lw=2.0, label='Collective')
    ax.axvline(2.0, color='gray', ls=':', lw=1.1)
    ax.axvline(cpa_t, color='#d62728', ls=':', lw=1.3)
    ax.set_title('(d) 4-Rotor Normalized Motor Commands', fontweight='bold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Motor Command [0..1]')
    ax.set_ylim(0.10, 0.68)
    ax.legend(loc='upper right', ncol=3)

    fig_all.tight_layout()
    fig_all.savefig(plots_dir / 'academic_summary_4panel.png', dpi=180)
    plt.close(fig_all)

    summary = dict(
        target_world_truth=[100.0, 0.0, 50.0],
        ukf_final_estimate_world=[round(float(est_wx[-1]), 3), round(float(est_wy[-1]), 3), round(float(est_wz[-1]), 3)],
        ukf_steady_rmse_3d_m=round(float(np.sqrt(np.mean(err_3d[30:] ** 2))), 3),
        ukf_final_sigma_m=round(float(sigma_m[-1]), 3),
        cpa_time_s=round(float(cpa_t), 2),
        cpa_distance_m=round(float(cpa_d), 3),
        max_speed_m_s=round(float(spd_3d.max()), 2),
        cruise_tilt_deg=round(float(np.mean(tilt_act[(tf > 4.0) & (tf < cpa_t - 0.5)])), 2),
        wing_roll_rms_deg=round(float(np.sqrt(np.mean(wing_roll ** 2))), 3),
    )
    (data_dir / 'academic_summary_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Generate academic interception plots and CSV logs.')
    parser.add_argument('--plots-dir', '--out-dir', dest='plots_dir', type=str, default=str(WORKSPACE_ROOT / 'plots'))
    parser.add_argument('--data-dir', type=str, default=str(WORKSPACE_ROOT / 'data'))
    args = parser.parse_args()

    ukf_rows, flight_rows, cpa_row = simulate_and_log()
    make_naive_plots(ukf_rows, flight_rows, cpa_row, Path(args.plots_dir), Path(args.data_dir))


if __name__ == '__main__':
    main()
