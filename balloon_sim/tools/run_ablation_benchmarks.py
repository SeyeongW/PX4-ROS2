#!/usr/bin/env python3
"""실측 비교 실험 스크립트 (Gazebo 없이 실제 balloon_locator 모듈을 직접 구동):

  [실험 ①] 거리 오차 vs 표적 거리 (30m ~ 130m, 현재 표적 (100,0,50)m = 111.8m 포함)
    - 비교 대상 (기존): 사각 바운딩 박스 정수 직경 max(w, h) (cv2.boundingRect)
    - 우리 방법 (제안): 타원 맞춤 / 서브픽셀 실루엣 직경 (balloon_detector.py)

  [실험 ②] 3D 위치 추정 정확도 비교 ((100, 0, 50)m 표적, 10 Hz, 100 프레임 + 외란/이상치 포함)
    - 비교 1: 영상 1장 역투영 (Single-Frame Back-Projection)
    - 비교 2: 단순 누적 평균 (Simple Running Average)
    - 우리 방법: 6-DOF 무향 칼만 필터 (UKFTargetEstimator, static_estimator.py 실제 클래스 구동)

실행:
    python3 balloon_sim/tools/run_ablation_benchmarks.py --out-dir balloon_sim/plots
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 실제 프로젝트 모듈 import
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'balloon_locator'))

from balloon_locator.balloon_detector import detect_balloons, parse_hsv_ranges
from balloon_locator.geometry import focal_px, range_from_diameter
from balloon_locator.static_estimator import UKFTargetEstimator


# 지상 망원 카메라 광학 파라미터 (gazebo.launch.py / gazebo_scene.py 동일)
IMG_W = 1280
IMG_H = 960
HFOV_DEG = 12.0
F_PX = focal_px(IMG_W, HFOV_DEG, 1.0)  # 6089.19 px
TARGET_SPAN_M = 0.64                   # x500 표적 드론 스팬 (0.64 m)
TARGET_WORLD = np.array([100.0, 0.0, 50.0], dtype=float)
STATION_WORLD = np.array([0.0, 0.0, 0.19], dtype=float)
TRUE_RANGE_100_50 = float(np.linalg.norm(TARGET_WORLD - STATION_WORLD))  # 111.72 m


def render_synthetic_target_frame(diameter_px: float, subpixel_dx: float, subpixel_dy: float,
                                  roi_size: int = 160) -> np.ndarray:
    """거리별 이론상 픽셀 크기를 갖는 표적을 슈퍼샘플링(16x) 안티앨리어싱으로 합성."""
    ss = 16
    canvas = np.zeros((roi_size * ss, roi_size * ss, 3), dtype=np.uint8)
    # 배경: 하늘색 (BGR)
    canvas[:, :] = (215, 175, 135)

    cx_ss = int(round((roi_size * 0.5 + subpixel_dx) * ss))
    cy_ss = int(round((roi_size * 0.5 + subpixel_dy) * ss))
    radius_ss = max(1, int(round((diameter_px * 0.5) * ss)))

    # 표적 실루엣 렌더링 (balloon_detector.py가 동시에 사각박스 bbox와 fitEllipse diameter_px를 반환하도록 색상 지정)
    cv2.circle(canvas, (cx_ss, cy_ss), radius_ss, (20, 95, 235), thickness=-1, lineType=cv2.LINE_AA)

    # 16x 슈퍼샘플링 -> 원본 해상도로 INTER_AREA 다운샘플링 (실제 카메라 센서 픽셀 적분 물리 모사)
    bgr = cv2.resize(canvas, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    return bgr


def run_experiment_1_range_vs_distance(plots_dir: Path, data_dir: Path):
    """실험 ①: 사각 박스 정수 직경 max(w, h) vs 타원 맞춤 서브픽셀 직경 (balloon_detector.py 직접 구동)."""
    hsv_ranges = parse_hsv_ranges([3, 100, 70, 28, 255, 255])
    distances_m = np.arange(30.0, 132.0, 2.0)

    # 6x6 = 36개의 서브픽셀 위상 오프셋 (카메라 픽셀 그리드 위에서 표적이 맺히는 임의의 서브픽셀 위치)
    sub_offsets = np.linspace(-0.45, 0.45, 6)

    rows = []
    for dist_m in distances_m:
        true_d_px = (F_PX * TARGET_SPAN_M) / dist_m

        bbox_err_m_list = []
        ours_err_m_list = []
        bbox_d_err_px_list = []
        ours_d_err_px_list = []

        for dx in sub_offsets:
            for dy in sub_offsets:
                bgr = render_synthetic_target_frame(true_d_px, dx, dy)
                dets = detect_balloons(
                    bgr, hsv_ranges,
                    min_diameter_px=4.0, max_diameter_px=800.0,
                    min_circularity=0.55, min_fill=0.50,
                    scale=1.0, morph_px=0
                )
                if not dets:
                    continue
                d = dets[0]
                # 1) 기존 방식: 정수 사각 박스 직경 max(w, h)
                _, _, bw, bh = d.bbox
                diam_bbox_px = float(max(bw, bh))
                # 2) 우리 방식: cv2.fitEllipse 기반 서브픽셀 직경 (d.diameter_px)
                diam_ours_px = float(d.diameter_px)

                r_bbox = range_from_diameter(diam_bbox_px, TARGET_SPAN_M, F_PX)
                r_ours = range_from_diameter(diam_ours_px, TARGET_SPAN_M, F_PX)

                bbox_err_m_list.append(abs(r_bbox - dist_m))
                ours_err_m_list.append(abs(r_ours - dist_m))
                bbox_d_err_px_list.append(abs(diam_bbox_px - true_d_px))
                ours_d_err_px_list.append(abs(diam_ours_px - true_d_px))

        rows.append({
            'distance_m': round(float(dist_m), 2),
            'true_diam_px': round(float(true_d_px), 3),
            'bbox_mae_m': round(float(np.mean(bbox_err_m_list)), 4),
            'bbox_std_m': round(float(np.std(bbox_err_m_list)), 4),
            'bbox_max_m': round(float(np.max(bbox_err_m_list)), 4),
            'ours_mae_m': round(float(np.mean(ours_err_m_list)), 4),
            'ours_std_m': round(float(np.std(ours_err_m_list)), 4),
            'ours_max_m': round(float(np.max(ours_err_m_list)), 4),
            'bbox_px_mae': round(float(np.mean(bbox_d_err_px_list)), 4),
            'ours_px_mae': round(float(np.mean(ours_d_err_px_list)), 4),
        })

    # CSV 저장
    csv_path = data_dir / 'exp1_range_error.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # (100, 0, 50)m -> R = 111.8m 지점 수치 추출 (가장 가까운 112.0m 행)
    idx_112 = int(np.argmin(np.abs(distances_m - TRUE_RANGE_100_50)))
    r_112 = rows[idx_112]

    # 그래프 1 생성 (Naive Matplotlib)
    dist_arr = np.array([r['distance_m'] for r in rows])
    bbox_mae = np.array([r['bbox_mae_m'] for r in rows])
    bbox_std = np.array([r['bbox_std_m'] for r in rows])
    ours_mae = np.array([r['ours_mae_m'] for r in rows])
    ours_std = np.array([r['ours_std_m'] for r in rows])
    bbox_px = np.array([r['bbox_px_mae'] for r in rows])
    ours_px = np.array([r['ours_px_mae'] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.4), dpi=200, sharex=True)

    ax1.plot(dist_arr, bbox_mae, 'r--o', markersize=3.5, linewidth=1.6,
             label='Integer Box')
    ax1.fill_between(dist_arr, np.maximum(0.0, bbox_mae - bbox_std), bbox_mae + bbox_std,
                     color='tab:red', alpha=0.15)

    ax1.plot(dist_arr, ours_mae, 'b-s', markersize=3.5, linewidth=2.0,
             label='Sub-Pixel Fit')
    ax1.fill_between(dist_arr, np.maximum(0.0, ours_mae - ours_std), ours_mae + ours_std,
                     color='tab:blue', alpha=0.18)

    ax1.axvline(TRUE_RANGE_100_50, color='tab:green', linestyle=':', linewidth=1.5,
                label='Target Range')
    ax1.set_ylabel('Range MAE [m]')
    ax1.set_title('Monocular Range Error vs. Target Distance')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9.5)

    ax2.plot(dist_arr, bbox_px, 'r--o', markersize=3.5, linewidth=1.5,
             label='Integer Box')
    ax2.plot(dist_arr, ours_px, 'b-s', markersize=3.5, linewidth=1.8,
             label='Sub-Pixel Fit')
    ax2.axvline(TRUE_RANGE_100_50, color='tab:green', linestyle=':', linewidth=1.5)
    ax2.set_xlabel('Target Distance [m]')
    ax2.set_ylabel('Diameter Error [px]')
    ax2.set_title('Pixel Span Error vs. Target Distance')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9.5)

    fig.tight_layout()
    fig.savefig(plots_dir / 'exp1_range_error_vs_distance.png')
    plt.close(fig)

    return {
        'target_range_100_50_m': round(TRUE_RANGE_100_50, 2),
        'at_112m_bbox_mae_m': r_112['bbox_mae_m'],
        'at_112m_bbox_max_m': r_112['bbox_max_m'],
        'at_112m_ours_mae_m': r_112['ours_mae_m'],
        'at_112m_ours_max_m': r_112['ours_max_m'],
        'overall_bbox_px_mae': round(float(np.mean(bbox_px)), 3),
        'overall_ours_px_mae': round(float(np.mean(ours_px)), 3),
        'error_reduction_pct_at_112m': round(100.0 * (1.0 - r_112['ours_mae_m'] / max(r_112['bbox_mae_m'], 1e-6)), 1),
    }


def simulate_single_trial_ukf_vs_baselines(seed: int, n_steps: int = 100, dt: float = 0.1):
    """단일 시행(10초, 10Hz = 100프레임):
    바람에 흔들리는 상공 표적((100, 0, 50)m 부근 호버링 진동) + 간헐적 YOLO/검출 스파이크 노이즈에 대해
    1) 1장 역투영, 2) 단순 누적 평균, 3) 실제 UKFTargetEstimator(static_estimator.py)를 동시 비교.
    """
    rng = np.random.default_rng(seed)

    # 지상 관측소 NED 원점: Station NED = [World_Y, World_X, -World_Z]
    origin_ned = np.array([0.0, 0.0, -0.19], dtype=float)
    base_ned = np.array([0.0, 100.0, -50.0], dtype=float)

    ukf = UKFTargetEstimator(
        gate_chi2=16.27,
        min_gate_samples=5,
        reset_after_rejects=15,
        forget=0.92,
        process_pos_std=0.05,
        process_vel_std=0.15,
        vel_damping=0.88,
    )

    sum_pos = np.zeros(3, dtype=float)
    count_pos = 0

    records = []
    for k in range(n_steps):
        t = k * dt
        # 바람/호버링에 의한 실제 표적의 완만한 움직임 (진폭 ~0.35m)
        drift_ned = np.array([
            0.25 * math.sin(0.7 * t),
            0.35 * math.sin(0.5 * t + 0.4),
            0.20 * math.cos(0.6 * t),
        ], dtype=float)
        true_ned = base_ned + drift_ned

        rel_true = true_ned - origin_ned
        true_rng = float(np.linalg.norm(rel_true))
        true_az = math.atan2(rel_true[1], rel_true[0])
        true_el = math.atan2(-rel_true[2], math.hypot(rel_true[0], rel_true[1]))
        true_d_px = (F_PX * TARGET_SPAN_M) / true_rng

        # 일반 가우시안 관측 노이즈
        az_noise = math.radians(rng.normal(0.0, 0.06))
        el_noise = math.radians(rng.normal(0.0, 0.06))
        d_noise = rng.normal(0.0, 0.38)

        # 간헐적 튀는 측정(Outlier: 프로펠러 블러/배경 오검출 등으로 픽셀 지름이 4~6px 튀는 현상, 약 9% 확률)
        is_outlier = False
        if k >= 6 and (k in (14, 27, 43, 58, 72, 86) or rng.random() < 0.04):
            is_outlier = True
            d_noise += rng.choice([-4.8, 5.5])
            az_noise += math.radians(rng.choice([-0.45, 0.45]))

        meas_az = true_az + az_noise
        meas_el = true_el + el_noise
        meas_d_px = max(5.0, true_d_px + d_noise)

        ray_ned = np.array([
            math.cos(meas_el) * math.cos(meas_az),
            math.cos(meas_el) * math.sin(meas_az),
            -math.sin(meas_el),
        ], dtype=float)

        # (A) 1장 역투영 (Single-Frame Back-Projection)
        raw_rng = (F_PX * TARGET_SPAN_M) / meas_d_px
        raw_ned = origin_ned + raw_rng * ray_ned

        # (B) 단순 누적 평균 (Simple Cumulative Average)
        sum_pos += raw_ned
        count_pos += 1
        avg_ned = sum_pos / float(count_pos)

        # (C) 우리 방법: 실제 UKFTargetEstimator.update_spherical 호출
        accepted = ukf.update_spherical(
            origin_ned=origin_ned,
            ray_unit_ned=ray_ned,
            diam_px=meas_d_px,
            focal_px=F_PX,
            balloon_diam_m=TARGET_SPAN_M,
            pixel_sigma_px=0.45,
            angle_sigma_deg=0.08,
            stamp=t,
        )
        ukf_ned = ukf.mean if ukf.mean is not None else raw_ned

        err_raw = float(np.linalg.norm(raw_ned - true_ned))
        err_avg = float(np.linalg.norm(avg_ned - true_ned))
        err_ukf = float(np.linalg.norm(ukf_ned - true_ned))

        records.append({
            'step': k,
            'time_s': round(t, 2),
            'is_outlier': int(is_outlier),
            'ukf_accepted': int(accepted),
            'err_single_m': round(err_raw, 4),
            'err_average_m': round(err_avg, 4),
            'err_ukf_m': round(err_ukf, 4),
            'ukf_sigma_m': round(float(ukf.sigma), 4),
        })

    return records, ukf.rejects


def run_experiment_2_ukf_vs_average_vs_single(plots_dir: Path, data_dir: Path):
    """실험 ②: 1장 역투영 vs 단순 누적 평균 vs 6-DOF UKF (단일 시계열 + 50회 몬테카를로 통계)."""
    records, rejects_rep = simulate_single_trial_ukf_vs_baselines(seed=42, n_steps=100, dt=0.1)

    csv_path = data_dir / 'exp2_position_accuracy.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # 50회 몬테카를로 반복 실험으로 정량 통계 산출 (수렴 구간 t >= 2.0s)
    mc_single_rmse, mc_avg_rmse, mc_ukf_rmse = [], [], []
    mc_single_max, mc_avg_max, mc_ukf_max = [], [], []
    mc_rejects = []
    for s in range(100, 150):
        recs, rej = simulate_single_trial_ukf_vs_baselines(seed=s, n_steps=100, dt=0.1)
        steady = [r for r in recs if r['time_s'] >= 2.0]
        e_s = np.array([r['err_single_m'] for r in steady])
        e_a = np.array([r['err_average_m'] for r in steady])
        e_u = np.array([r['err_ukf_m'] for r in steady])
        mc_single_rmse.append(float(np.sqrt(np.mean(e_s ** 2))))
        mc_avg_rmse.append(float(np.sqrt(np.mean(e_a ** 2))))
        mc_ukf_rmse.append(float(np.sqrt(np.mean(e_u ** 2))))
        mc_single_max.append(float(np.max(e_s)))
        mc_avg_max.append(float(np.max(e_a)))
        mc_ukf_max.append(float(np.max(e_u)))
        mc_rejects.append(int(rej))

    t_arr = np.array([r['time_s'] for r in records])
    err_single = np.array([r['err_single_m'] for r in records])
    err_avg = np.array([r['err_average_m'] for r in records])
    err_ukf = np.array([r['err_ukf_m'] for r in records])
    outlier_mask = np.array([r['is_outlier'] == 1 for r in records])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=200,
                                   gridspec_kw={'width_ratios': [1.6, 1.0]})

    # 좌측 패널: 시간에 따른 3D 위치 오차 비교
    ax1.plot(t_arr, err_single, color='0.65', linestyle=':', linewidth=1.2,
             label='Single-Frame')
    ax1.scatter(t_arr[outlier_mask], err_single[outlier_mask], color='tab:red', marker='x', s=45, zorder=5,
                label='Outlier')
    ax1.plot(t_arr, err_avg, color='tab:orange', linestyle='--', linewidth=1.8,
             label='Simple Average')
    ax1.plot(t_arr, err_ukf, color='tab:blue', linestyle='-', linewidth=2.2,
             label='6-DOF UKF')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('3D Position Error [m]')
    ax1.set_title('(a) Real-Time 3D Position Error')
    ax1.set_ylim(0.0, max(float(np.max(err_single)) * 1.08, 12.0))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', ncol=2, fontsize=9.0)

    # 우측 패널: 50회 몬테카를로 평균 3D RMSE 및 최대 오차 막대 그래프
    methods = ['Single-Frame', 'Simple Avg', '6-DOF UKF']
    rmse_vals = [float(np.mean(mc_single_rmse)), float(np.mean(mc_avg_rmse)), float(np.mean(mc_ukf_rmse))]
    max_vals = [float(np.mean(mc_single_max)), float(np.mean(mc_avg_max)), float(np.mean(mc_ukf_max))]

    x = np.arange(len(methods))
    width = 0.35
    b1 = ax2.bar(x - width / 2, rmse_vals, width, label='3D RMSE', color=['0.6', 'tab:orange', 'tab:blue'])
    b2 = ax2.bar(x + width / 2, max_vals, width, label='Max Error', color=['0.8', '#ffcc80', '#90caf9'])

    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.2, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.2, f'{h:.2f}', ha='center', va='bottom', fontsize=8.0)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('3D Error [m]')
    ax2.set_title('(b) Monte Carlo Error Summary')
    ax2.set_ylim(0.0, max(max_vals) * 1.18)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9.0)

    fig.tight_layout()
    fig.savefig(plots_dir / 'exp2_ukf_vs_average_vs_single.png')
    plt.close(fig)

    return {
        'single_frame_rmse_m': round(float(np.mean(mc_single_rmse)), 3),
        'single_frame_max_m': round(float(np.mean(mc_single_max)), 3),
        'simple_avg_rmse_m': round(float(np.mean(mc_avg_rmse)), 3),
        'simple_avg_max_m': round(float(np.mean(mc_avg_max)), 3),
        'ukf_rmse_m': round(float(np.mean(mc_ukf_rmse)), 3),
        'ukf_max_m': round(float(np.mean(mc_ukf_max)), 3),
        'mean_outliers_rejected_per_10s': round(float(np.mean(mc_rejects)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Run real detector & UKF comparative experiments (Exp 1 & Exp 2).')
    parser.add_argument('--plots-dir', '--out-dir', dest='plots_dir', type=str, default=str(REPO_ROOT / 'plots'))
    parser.add_argument('--data-dir', type=str, default=str(REPO_ROOT / 'data'))
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    data_dir = Path(args.data_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    exp1_summary = run_experiment_1_range_vs_distance(plots_dir, data_dir)
    exp2_summary = run_experiment_2_ukf_vs_average_vs_single(plots_dir, data_dir)

    summary = {
        'experiment_1_range_vs_distance': exp1_summary,
        'experiment_2_ukf_vs_average_vs_single': exp2_summary,
    }
    (data_dir / 'ablation_benchmark_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
