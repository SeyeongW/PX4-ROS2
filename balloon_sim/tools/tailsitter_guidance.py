#!/usr/bin/env python3
"""지상 관측 카메라 UKF 탐색·락온 + 사용자 출격 승인 트리거 + PX4 쿼드 테일시터(1045) 다중 폐루프 유도 제어기.

폐루프(Closed-Loop) 협력 시퀀스:
  1. [지상 카메라 상공 탐색(Search) -> 풍선 식별 및 6-DOF UKF 락온(Lock)]
     - 지상 짐벌 카메라(/jol/camera/image_raw, 1280x960, HFOV 12°)가 초기 오프셋 방향(좌측 -20.6°, 저앙각 +8.0°)에서
       출발하여 상공을 대각선으로 스캔합니다.
     - 풍선의 색상(HSV) + 원형도(Circularity >= 0.6) + 채움비(Fill >= 0.55) 조건을 통과한 객체를 발견하면
       즉시 짐벌 비주얼 서보잉으로 화면 중앙에 고정(Lock)하고, 6자유도 무향 칼만 필터(UKF)로 3D 좌표 (X, Y, Z)를
       10 Hz(0.1초 주기)로 추정·브로드캐스트합니다.
       * n : UKF가 누적 융합한 유효 관측 프레임(샘플) 개수
       * s (sigma) : 3x3 위치 오차 공분산 행렬의 최대 방향 표준편차(불확실도) [m]

  2. [사용자 출격 승인 트리거(Launch Approval Trigger)]
     - 지상 카메라가 풍선을 락온(n >= 3, s <= 2.8m)하면 드론은 지상에서 대기하며 사용자 승인을 요청합니다.
     - 승인 방법 (택 1):
       (A) 카메라 뷰어 창(onboard)을 클릭하고 키보드 [SPACE] 또는 [ENTER] 또는 [y] 키 입력
       (B) 터미널에서 ./tools/approve_launch.sh 실행

  3. [승인 즉시 수직 이륙 -> 중기 지상 UKF 실시간 유도(10 Hz) -> 종말 온보드 3D LOS/PN 유도(50 Hz)]
     - 승인 즉시 ARM 및 5m 수직 이륙 후 지체 없이 테일시터 전진 피치다운 전환을 수행합니다.
     - 비행 중 온보드 카메라 포착 전까지는 지상 UKF가 10 Hz(초당 10회)로 보내주는 최신 3D 좌표를 계속 반영하며,
       온보드 노즈콘 카메라(/balloon/onboard/detections)에 풍선이 포착되는 순간 온보드 3D LOS/PN 종말 비전 유도로
       전환하여 풍선 정중앙을 통과한 뒤 후면 전환 감속 및 RTL 착륙합니다.

실행:
    ./tools/run_tailsitter_sim.sh
"""
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

try:
    import rclpy
    from std_msgs.msg import Bool, String
    from vision_msgs.msg import Detection2DArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. 비행 및 다중 폐루프 유도 제어 파라미터
# ═══════════════════════════════════════════════════════════════════════════
FALLBACK_TARGET_WORLD = [
    float(os.environ.get('FLY_TARGET_X', '90.0')),
    float(os.environ.get('FLY_TARGET_Y', '0.0')),
    float(os.environ.get('FLY_TARGET_Z', '40.0')),
]
HOVER_ALT = float(os.environ.get('FLY_HOVER_ALT', '5.0'))
AUTO_APPROVE = os.environ.get('AUTO_APPROVE', '0').lower() in ('1', 'true', 'yes')
APPROVAL_FLAG_FILE = Path('/tmp/balloon_launch_approved')

UKF_LOCK_SIGMA_M = 2.8     # 지상 UKF 락온 인정 최대 표준편차 s (sigma) [m]
UKF_MIN_SAMPLES  = 3       # 지상 UKF 락온 최소 누적 샘플 수 n

NORTH_YAW  = math.pi / 2.0
HOME_WORLD = [-2.0, 0.0, 0.2]
PX4_UDP    = 'udpin:0.0.0.0:14540'

# 테일시터 속도 및 추력 제어 파라미터
TAKEOFF_ALT       = 5.0    # 수직 이륙 목표 고도 [m]
TRANSITION_DUR    = 1.5    # 수직 호버링 -> 테일시터 대각선 자세 전환 시간 [s]
MIN_CRUISE_SPEED  = 10.0   # 근접 구간 최소 목표 속도 [m/s]
MAX_CRUISE_SPEED  = 14.5   # 원거리 구간 최대 목표 속도 [m/s]
KP_DIST_SPEED     = 0.05   # 거리 비례 속도 이득: V_des = clip(10.0 + 0.05 * dist_3d, 10.0, 14.5)
LIFT_CRUISE_SPEED = 14.2   # 주익 양력이 중력 수직 성분을 완전히 지지하는 기준 속도 [m/s]
SPEC_THRUST_MAX   = 34.52  # 4로터 최대 비추력 (69.04 N / 2.0 kg) [m/s^2]
THRUST_MIN        = 0.07   # 비행 중 최소 정규화 추력 (과속 시 감속 허용 + 자세 제어권 유지)
THRUST_MAX        = 0.54   # 비행 중 최대 정규화 추력 (초기 급상승 오버슈트 방지)
MAX_LOS_LEAD_DEG  = 15.0   # 비전 추적 시 LOS 대비 노즈콘 최대 리드각 [deg] (화각 내부 유지 보장)
MAX_PATH_LEAD_DEG = 22.0   # 대각선 경로 추종 시 최대 리드각 [deg]
COAST_TIMEOUT     = 0.8    # 비전 일시 소실 시 마지막 LOS 유지 시간 [s]
BALLOON_DIAM_M    = 0.30   # 목표 풍선 실제 지름 [m]

# "풍선 지났다!" 통과 판정 파라미터 (30cm 풍선 기준)
PASS_MIN_DIAM_PX      = 16.0  # [조건 1] 최소 확대 픽셀 지름 (~9m 이내 접근 시 도달)
PASS_GROWTH_RATIO     = 1.5   # [조건 1] 최초 검출 대비 최소 확대 비율
PASS_LOST_FRAMES      = 3     # [조건 1] 확대 후 연속 유실 프레임 수
PASS_IMPACT_DIAM_PX   = 72.0  # [조건 1] 초근접(~2.0m 이내) 즉시 통과 인정 픽셀 지름

LOCK_DIST             = 15.0  # [조건 2] 통과 감시 활성화 거리 게이트 [m]
OVERSHOOT_DIST        = 0.8   # [조건 2] 최근접 거리(CPA) 이후 거리 증가량 [m]
PAST_PLANE_DIST       = 0.4   # [조건 2] 목표 수직 평면 통과 거리 [m]

FORCE_PAST_PLANE_DIST = 1.5   # [무조건 RTL] 목표 수직 평면 강제 복귀 거리 [m]
FORCE_OVERSHOOT_DIST  = 2.0   # [무조건 RTL] 최근접 거리 대비 강제 복귀 거리 증가량 [m]

# 온보드 노즈콘 카메라 광학 파라미터 (Raspberry Pi Camera Module 3 Standard, 640x360, HFOV 66°)
IMG_W    = 640
IMG_H    = 360
HFOV_DEG = 66.0
CX       = (IMG_W - 1) / 2.0
CY       = (IMG_H - 1) / 2.0
FX       = IMG_W / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
FY       = FX


# ═══════════════════════════════════════════════════════════════════════════
# 2. 좌표 변환 및 3D 테일시터 자세(Wings-Level Quaternion) 계산 함수
# ═══════════════════════════════════════════════════════════════════════════
def world_to_ned(wx, wy, wz):
    """Gazebo World (X, Y, Z) -> PX4 Local NED (N, E, D) 변환 (홈 기준)."""
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    dx, dy, dz = wx - HOME_WORLD[0], wy - HOME_WORLD[1], wz - HOME_WORLD[2]
    return c * dx + s * dy, s * dx - c * dy, -dz


def station_ned_to_world(sn, se, sd):
    """지상 관측소(Station) NED [N=World_Y, E=World_X, D=-World_Z] -> Gazebo World [X, Y, Z] 변환."""
    return float(se), float(sn), float(-sd)


def quat_to_rot_ned(w, x, y, z):
    """단위 쿼터니언 (w, x, y, z) -> 3x3 회전행렬 R_bn (Body FRD -> Local NED)."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def rot_ned_to_quat(R):
    """3x3 회전행렬 R_bn (Body FRD -> Local NED) -> 단위 쿼터니언 [w, x, y, z]."""
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
    """노즈콘 지향 단위벡터 nose_dir_ned (Body -Z_FRD = +Z_FLU = 카메라 광축)로부터
    날개 수평(Wings-Level, 카메라 롤 회전 0°)을 보장하는 SO(3) 자세 쿼터니언 [w, x, y, z] 계산.
    """
    n = np.array(nose_dir_ned, dtype=float)
    n_norm = float(np.linalg.norm(n))
    n = np.array([0.0, 0.0, -1.0], dtype=float) if n_norm < 1e-6 else (n / n_norm)

    z_frd = -n
    horiz_norm = math.hypot(float(n[0]), float(n[1]))
    az = math.atan2(float(n[1]), float(n[0])) if horiz_norm > 1e-3 else fallback_yaw

    # 우측 날개(+Y_FRD)를 수평면에 구속 -> Wing Roll = 0°
    y_frd = np.array([-math.sin(az), math.cos(az), 0.0], dtype=float)
    x_frd = np.cross(y_frd, z_frd)
    x_norm = float(np.linalg.norm(x_frd))
    if x_norm < 1e-6:
        x_frd = np.array([math.cos(fallback_yaw), math.sin(fallback_yaw), 0.0], dtype=float)
    else:
        x_frd /= x_norm
    y_frd = np.cross(z_frd, x_frd)
    y_frd /= np.linalg.norm(y_frd)

    R_bn = np.column_stack((x_frd, y_frd, z_frd))
    return rot_ned_to_quat(R_bn)


def pixel_to_los_ned(u, v, R_bn):
    """노즈콘 카메라 픽셀 (u, v) -> NED 3D 시선(LOS) 단위벡터 변환."""
    ex = (u - CX) / FX
    ey = (v - CY) / FY
    ray_body_frd = np.array([ex, -ey, -1.0], dtype=float)
    ray_body_frd /= np.linalg.norm(ray_body_frd)
    los_ned = R_bn @ ray_body_frd
    return los_ned / np.linalg.norm(los_ned)


def clamp_vec(vec, max_mag):
    mag = float(np.linalg.norm(vec))
    if mag > max_mag and mag > 1e-6:
        return vec * (max_mag / mag)
    return vec


def compute_tailsitter_cmd(aim_dir_ned, a_perp_cmd, vel_ned, dist_3d, max_lead_deg, trans_alpha, allow_dive=False):
    """목표 지향 벡터(aim_dir_ned) 및 수직 보정 가속도(a_perp_cmd)로부터
    테일시터 노즈콘 지향 벡터(nose_dir)와 정규화 추력(thrust)을 분리 산출.
    """
    d = np.array(aim_dir_ned, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1e-6)

    cur_spd = float(np.linalg.norm(vel_ned))
    v_along = float(np.dot(vel_ned, d))

    # 저속 구간에서는 부족한 날개 양력만큼 위쪽(-D) 중력 보상을 추가하고, 고속 구간에서는 잉여 주익 양력을 감쇄
    lift_frac = min(1.45, (cur_spd / LIFT_CRUISE_SPEED) ** 2)
    g_uncomp_ned = np.array([0.0, 0.0, 9.81 * (1.0 - 0.94 * lift_frac)], dtype=float)

    g_perp = g_uncomp_ned - float(np.dot(g_uncomp_ned, d)) * d
    lead_acc_perp = (a_perp_cmd - float(np.dot(a_perp_cmd, d)) * d) - g_perp

    ref_acc = 11.0
    max_perp_mag = ref_acc * math.tan(math.radians(max_lead_deg))
    lead_acc_perp = clamp_vec(lead_acc_perp, max_perp_mag)

    raw_nose = d + (lead_acc_perp / ref_acc)
    raw_nose /= np.linalg.norm(raw_nose)

    # 종말 비전 추적(allow_dive=True) 시 풍선이 기체보다 아래에 있으면 기수를 아래(-26.7°)까지 숙여 고도 오차를 즉시 제거
    max_down_z = 0.45 if allow_dive else 0.08
    if raw_nose[2] > max_down_z:
        raw_nose[2] = max_down_z
        raw_nose /= np.linalg.norm(raw_nose)

    hover_nose = np.array([0.0, 0.0, -1.0], dtype=float)
    nose_dir = (1.0 - trans_alpha) * hover_nose + trans_alpha * raw_nose
    nose_dir /= np.linalg.norm(nose_dir)

    # 남은 거리(dist_3d)에 비례하는 목표 접근 속도 추종 추력 계산
    v_target = min(MAX_CRUISE_SPEED, max(MIN_CRUISE_SPEED, MIN_CRUISE_SPEED + KP_DIST_SPEED * dist_3d))
    speed_err = v_target - v_along

    grav_along = max(0.0, -9.81 * float(nose_dir[2]))
    aero_drag = 0.0045 * (cur_spd ** 2)
    low_spd_support = 9.81 * max(0.0, 1.0 - lift_frac) * 0.42
    # 피치다운 전환 초기(기수가 위를 향할 때) 과도한 추력으로 수직 급상승하는 현상 방지
    fwd_gate = (trans_alpha ** 1.5) if speed_err > 0.0 else 1.0
    kp_spd = 1.6 if speed_err >= 0.0 else 2.2
    a_thrust_cmd = grav_along + aero_drag + low_spd_support + kp_spd * fwd_gate * speed_err

    thrust = max(THRUST_MIN, min(THRUST_MAX, a_thrust_cmd / SPEC_THRUST_MAX))
    return nose_dir, thrust, v_target


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROS 2 지상 UKF 추정(10Hz), 출격 승인 트리거 및 온보드 비전 검출 수신 헬퍼
# ═══════════════════════════════════════════════════════════════════════════
class VisionListener:
    """ROS 2 지상 카메라 UKF 추정(/balloon/status, /jol/static_target/estimate),
    출격 승인 트리거(/mission/approve_launch) 및 온보드 비전(/balloon/onboard/detections) 수신 관리.
    """

    def __init__(self):
        self.lock = threading.Lock()
        # 온보드 카메라 상태
        self.u = 0.0
        self.v = 0.0
        self.diam = 0.0
        self.initial_diam = 0.0
        self.max_diam = 0.0
        self.last_rx = 0.0
        self.lost_frames = 0

        # 지상 UKF 추정 상태 (10 Hz 수신)
        self.ukf_valid = False
        self.ukf_world = None
        self.ukf_sigma_m = math.inf
        self.ukf_n = 0
        self.ukf_reason = 'waiting_for_ground_ukf'
        self.ukf_last_rx = 0.0
        self.ukf_rx_count = 0
        self.ukf_hz = 10.0

        # 사용자 출격 승인 상태
        self.launch_approved = False

    def update(self, u, v, diam):
        with self.lock:
            self.u = u
            self.v = v
            self.diam = diam
            if self.initial_diam <= 0.0:
                self.initial_diam = diam
            if diam > self.max_diam:
                self.max_diam = diam
            self.last_rx = time.monotonic()
            self.lost_frames = 0

    def mark_lost(self):
        with self.lock:
            if self.last_rx > 0.0:
                self.lost_frames += 1

    def update_ground_ukf(self, world_xyz, sigma_m, n_samples, valid, reason):
        with self.lock:
            now_t = time.monotonic()
            if world_xyz is not None and len(world_xyz) == 3 and all(math.isfinite(c) for c in world_xyz):
                if self.ukf_last_rx > 0.0 and now_t > self.ukf_last_rx:
                    dt_rx = now_t - self.ukf_last_rx
                    if 0.01 < dt_rx < 1.0:
                        self.ukf_hz = 0.8 * self.ukf_hz + 0.2 * (1.0 / dt_rx)
                self.ukf_world = [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])]
                self.ukf_last_rx = now_t
                self.ukf_rx_count += 1
            if sigma_m is not None and math.isfinite(sigma_m):
                self.ukf_sigma_m = float(sigma_m)
            self.ukf_n = int(n_samples)
            self.ukf_valid = bool(valid)
            self.ukf_reason = str(reason)

    def set_approved(self):
        with self.lock:
            self.launch_approved = True

    def is_approved(self):
        if APPROVAL_FLAG_FILE.is_file() or AUTO_APPROVE:
            return True
        with self.lock:
            return self.launch_approved

    def get(self):
        with self.lock:
            age = time.monotonic() - self.last_rx if self.last_rx > 0 else math.inf
            return (
                self.u,
                self.v,
                self.diam,
                self.initial_diam,
                self.max_diam,
                self.lost_frames,
                age,
                self.last_rx,
            )

    def get_ground_ukf(self):
        with self.lock:
            age = time.monotonic() - self.ukf_last_rx if self.ukf_last_rx > 0 else math.inf
            return (
                list(self.ukf_world) if self.ukf_world is not None else None,
                self.ukf_sigma_m,
                self.ukf_n,
                self.ukf_valid,
                self.ukf_reason,
                age,
                self.ukf_rx_count,
                self.ukf_hz,
            )


def start_ros_vision_thread(vision_state):
    if not HAS_ROS2:
        print('WARNING: rclpy를 불러올 수 없어 비전 검출 없이 실행합니다.', flush=True)
        return

    def _spin():
        rclpy.init()
        node = rclpy.create_node('tailsitter_cooperative_guidance_listener')

        def _on_onboard_det(msg):
            if not msg.detections:
                vision_state.mark_lost()
                return
            det = msg.detections[0]
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)
            diam = float((det.bbox.size_x + det.bbox.size_y) * 0.5)
            if math.isfinite(u) and math.isfinite(v) and diam > 2.5:
                vision_state.update(u, v, diam)
            else:
                vision_state.mark_lost()

        def _on_ground_status(msg):
            try:
                data = json.loads(msg.data)
                ned = data.get('ned')
                world_xyz = station_ned_to_world(ned[0], ned[1], ned[2]) if (ned and len(ned) == 3) else None
                vision_state.update_ground_ukf(
                    world_xyz=world_xyz,
                    sigma_m=data.get('sigma_m'),
                    n_samples=data.get('n', 0),
                    valid=data.get('valid', False),
                    reason=data.get('reason', 'unknown'),
                )
            except Exception:
                pass

        def _on_ground_estimate(msg):
            try:
                data = json.loads(msg.data)
                if data.get('valid') and data.get('position'):
                    pos = data['position']
                    _, cur_sig, cur_n, _, _, _, _, _ = vision_state.get_ground_ukf()
                    vision_state.update_ground_ukf(
                        world_xyz=pos,
                        sigma_m=data.get('maximum_position_std_m', cur_sig),
                        n_samples=max(cur_n, 3),
                        valid=True,
                        reason='ready',
                    )
            except Exception:
                pass

        def _on_approve(msg):
            if msg.data:
                vision_state.set_approved()

        node.create_subscription(Detection2DArray, '/balloon/onboard/detections', _on_onboard_det, 10)
        node.create_subscription(String, '/balloon/status', _on_ground_status, 10)
        node.create_subscription(String, '/jol/static_target/estimate', _on_ground_estimate, 10)
        node.create_subscription(Bool, '/mission/approve_launch', _on_approve, 10)
        try:
            rclpy.spin(node)
        except Exception:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    threading.Thread(target=_spin, daemon=True).start()


def heartbeat_thread(conn):
    while True:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        time.sleep(1.0)


def send_pos(conn, n, e, d, yaw, vn=0.0, ve=0.0, vd=0.0):
    """수직 이륙 및 호버링 위치 고정용 MAVLink 명령 전송."""
    type_mask = 0b0000_100_111_000_000
    conn.mav.set_position_target_local_ned_send(
        0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
        n, e, d, vn, ve, vd, 0, 0, 0, yaw, 0)


def send_attitude_target(conn, q_wxyz, thrust):
    """테일시터 3D 자세 쿼터니언(q_wxyz) + 추력(thrust) 직접 전송 (SET_ATTITUDE_TARGET)."""
    type_mask = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )
    conn.mav.set_attitude_target_send(
        0,
        conn.target_system,
        conn.target_component,
        type_mask,
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,
        float(thrust),
    )


def request_stream(conn, msg_id, hz):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)


def set_param(conn, name, value):
    conn.mav.param_set_send(
        conn.target_system, conn.target_component,
        name.encode('utf-8'), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.08)
    ack = conn.recv_match(type='PARAM_VALUE', blocking=True, timeout=1.5)
    if ack:
        print(f'  PARAM {name} = {ack.param_value:.1f}', flush=True)
    else:
        print(f'  PARAM {name} = {value:.1f} (no ack)', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. 메인 협력 폐루프 비행 시퀀스
# ═══════════════════════════════════════════════════════════════════════════
def main():
    try:
        if APPROVAL_FLAG_FILE.is_file():
            APPROVAL_FLAG_FILE.unlink()
    except Exception:
        pass

    startup_delay = float(os.environ.get('FLY_STARTUP_DELAY', '0'))
    if startup_delay > 0:
        print(f'PX4 및 지상 관측 카메라 초기화 대기 {startup_delay:.0f}초 ...', flush=True)
        time.sleep(startup_delay)

    vision = VisionListener()
    start_ros_vision_thread(vision)

    _, _, takeoff_d = world_to_ned(HOME_WORLD[0], HOME_WORLD[1], HOME_WORLD[2] + TAKEOFF_ALT)
    init_tn, init_te, _ = world_to_ned(*FALLBACK_TARGET_WORLD)
    yaw = math.atan2(init_te, init_tn)

    print('========================================================================', flush=True)
    print(' [다중 폐루프 협력 시스템] 지상 짐벌 카메라 탐색/UKF + 사용자 승인 출격 + 비전 유도', flush=True)
    print('  - n : 지상 UKF에 누적 융합된 유효 풍선 관측 프레임(샘플) 개수', flush=True)
    print('  - s : 지상 UKF 3D 위치 추정의 최대 방향 표준편차(불확실도, sigma) [m]', flush=True)
    print('  - 지상->드론 좌표 수신 주기: 10 Hz (0.1초마다 실시간 3D 좌표 패킷 수신)', flush=True)
    print('========================================================================\n', flush=True)

    print(f'PX4 연결 대기  {PX4_UDP} ...', flush=True)
    conn = mavutil.mavlink_connection(PX4_UDP)
    conn.wait_heartbeat(timeout=60)
    print(f'연결 OK  sysid={conn.target_system}', flush=True)

    threading.Thread(target=heartbeat_thread, args=(conn,), daemon=True).start()

    request_stream(conn, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 60)
    request_stream(conn, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 60)

    print('테일시터 파라미터 확인 ...', flush=True)
    set_param(conn, 'SYS_HAS_NUM_ASPD', 0.0)
    set_param(conn, 'FW_USE_AIRSPD', 0.0)
    set_param(conn, 'FD_FAIL_R', 85.0)
    set_param(conn, 'FD_FAIL_P', 85.0)

    # ── 0단계: 지상 대기 상태에서 지상 카메라 상공 탐색 -> 풍선 락온 -> 사용자 출격 승인 대기 ──
    print('\n[1단계: 지상 카메라 상공 탐색 중] 짐벌이 좌측 하단(-20.6°, +8.0°)에서 상공을 스캔하며 풍선을 찾고 있습니다...', flush=True)
    last_wait_log = 0.0
    locked_banner_printed = False
    active_target_world = list(FALLBACK_TARGET_WORLD)

    while True:
        # PX4 버퍼 비우기
        while conn.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION'], blocking=False) is not None:
            pass

        ukf_world, ukf_sigma, ukf_n, ukf_valid, ukf_reason, ukf_age, ukf_rx_cnt, ukf_hz = vision.get_ground_ukf()
        now_w = time.monotonic()

        is_locked = (
            ukf_world is not None and ukf_age < 2.5 and (
                ukf_valid or (ukf_n >= UKF_MIN_SAMPLES and ukf_sigma <= UKF_LOCK_SIGMA_M)
            )
        )

        if is_locked:
            active_target_world = ukf_world
            if not locked_banner_printed:
                locked_banner_printed = True
                ex = active_target_world[0] - FALLBACK_TARGET_WORLD[0]
                ey = active_target_world[1] - FALLBACK_TARGET_WORLD[1]
                ez = active_target_world[2] - FALLBACK_TARGET_WORLD[2]
                err_3d = math.sqrt(ex * ex + ey * ey + ez * ez)
                print('\n========================================================================', flush=True)
                print(' ★ [2단계: 지상 카메라 풍선 발견 & 6-DOF UKF 락온 완료!]', flush=True)
                print('   - 풍선 검증 : 색상(HSV) + 원형도(Circularity) + 채움비 검증 통과', flush=True)
                print(f'   - 실제 좌표 : World = ({FALLBACK_TARGET_WORLD[0]:.2f}, {FALLBACK_TARGET_WORLD[1]:.2f}, {FALLBACK_TARGET_WORLD[2]:.2f}) m', flush=True)
                print(f'   - 추정 좌표 : World = ({active_target_world[0]:.2f}, {active_target_world[1]:.2f}, {active_target_world[2]:.2f}) m', flush=True)
                print(f'   - 추정 오차 : dX={ex:+.2f}m, dY={ey:+.2f}m, dZ={ez:+.2f}m (3D 오차 = {err_3d:.2f} m)', flush=True)
                print(f'   - 필터 상태 : 샘플 수 n = {ukf_n} 개 | 위치 불확실도 s (sigma) = {ukf_sigma:.2f} m', flush=True)
                print(f'   - 수신 주기 : {ukf_hz:.1f} Hz (누적 {ukf_rx_cnt}회 수신 중)', flush=True)
                print(' ------------------------------------------------------------------------', flush=True)
                print(' ▶ [출격 승인 대기 중 (Trigger Waiting)]', flush=True)
                print('   1) 카메라 뷰어 창(onboard)을 클릭하고 [SPACE] 또는 [ENTER] 키를 누르거나,', flush=True)
                print('   2) 새 터미널에서 ./tools/approve_launch.sh 를 실행하면 즉시 이륙합니다!', flush=True)
                print('========================================================================\n', flush=True)

            if vision.is_approved():
                print(
                    f'★ [출격 승인 확인!] 최종 지상 UKF 목표 좌표 World=({active_target_world[0]:.2f}, '
                    f'{active_target_world[1]:.2f}, {active_target_world[2]:.2f})m (n={ukf_n}, s={ukf_sigma:.2f}m) -> 드론 이륙 시작!',
                    flush=True)
                break

            if now_w - last_wait_log > 2.0:
                ex = active_target_world[0] - FALLBACK_TARGET_WORLD[0]
                ey = active_target_world[1] - FALLBACK_TARGET_WORLD[1]
                ez = active_target_world[2] - FALLBACK_TARGET_WORLD[2]
                err_3d = math.sqrt(ex * ex + ey * ey + ez * ez)
                print(
                    f'  [승인 대기 중 | 수신 {ukf_hz:.1f}Hz (총 {ukf_rx_cnt}회)] '
                    f'UKF=({active_target_world[0]:.2f}, {active_target_world[1]:.2f}, {active_target_world[2]:.2f})m | '
                    f'실제오차=({ex:+.2f}, {ey:+.2f}, {ez:+.2f})m (3D오차 {err_3d:.2f}m) | '
                    f'n={ukf_n} | s={ukf_sigma:.2f}m',
                    flush=True)
                last_wait_log = now_w
        else:
            if now_w - last_wait_log > 1.2:
                if ukf_world is not None:
                    print(
                        f'  [지상 UKF 수렴 중] World=({ukf_world[0]:.1f}, {ukf_world[1]:.1f}, {ukf_world[2]:.1f})m | '
                        f'n={ukf_n} | s={ukf_sigma:.2f}m',
                        flush=True)
                else:
                    print(f'  [지상 짐벌 상공 탐색 중...] 상태={ukf_reason}', flush=True)
                last_wait_log = now_w

        time.sleep(0.04)

    # ── 1단계: 승인 완료 후 OFFBOARD 전환, ARM 및 5m 수직 이륙 ──
    tn, te, t_d = world_to_ned(*active_target_world)
    yaw = math.atan2(te, tn)

    print('setpoint 스트림 시작 ...', flush=True)
    for _ in range(35):
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(0.04)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        6, 0, 0, 0, 0, 0)
    print('OFFBOARD 요청', flush=True)
    time.sleep(0.25)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0)
    print('ARM 요청', flush=True)

    armed = False
    for _ in range(80):
        msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=0.2)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            armed = True
            break
        send_pos(conn, 0, 0, takeoff_d, yaw)
    if not armed:
        print('ERROR: arm 실패', flush=True)
        sys.exit(1)
    print('ARM 완료 — 5m 수직 이륙 시작 (Tailsitter Hover 모드)', flush=True)

    pos_ned = np.array([0.0, 0.0, 0.0])
    vel_ned = np.array([0.0, 0.0, 0.0])
    R_bn = np.eye(3)

    for _ in range(300):
        send_pos(conn, 0, 0, takeoff_d, yaw)
        msg = conn.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION'], blocking=True, timeout=0.05)
        if msg:
            if msg.get_type() == 'LOCAL_POSITION_NED':
                pos_ned = np.array([msg.x, msg.y, msg.z])
                vel_ned = np.array([msg.vx, msg.vy, msg.vz])
                if abs(msg.z - takeoff_d) < 0.4 and abs(msg.vz) < 0.8:
                    print(f'5m 이륙 완료! 현재고도 {-msg.z:.2f}m (목표 {-takeoff_d:.2f}m)', flush=True)
                    break
            elif msg.get_type() == 'ATTITUDE_QUATERNION':
                R_bn = quat_to_rot_ned(msg.q1, msg.q2, msg.q3, msg.q4)
    else:
        print('WARNING: 이륙 고도 도달 시간 초과, 다음 단계로 진행', flush=True)

    # ── 2단계 & 3단계: 지상 UKF 실시간 수신(10 Hz) + 테일시터 대각선 비행 -> 온보드 풍선 검출 시 3D LOS/PN 추적 ──
    start_ned = pos_ned.copy()
    target_ned = np.array([tn, te, t_d], dtype=float)
    path_vec = target_ned - start_ned
    path_len = float(np.linalg.norm(path_vec))
    path_dir = path_vec / path_len if path_len > 1e-3 else np.array([1.0, 0.0, 0.0])

    print(f'테일시터 전진 피치다운 전환 및 폐루프 유도 비행 시작: NED({start_ned[0]:.1f}, {start_ned[1]:.1f}, {start_ned[2]:.1f}) '
          f'-> UKF 목표 NED({tn:.1f}, {te:.1f}, {t_d:.1f})', flush=True)

    trans_start_t = time.monotonic()
    last_log = 0.0
    last_los_ned = None
    smoothed_nose_ned = np.array([0.0, 0.0, -1.0], dtype=float)
    vision_locked = False
    target_locked = False
    min_dist_after_lock = math.inf
    case1_vision_passed = False
    case2_target_passed = False
    last_ukf_update_n = 0

    while True:
        while True:
            msg = conn.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION'], blocking=False)
            if msg is None:
                break
            if msg.get_type() == 'LOCAL_POSITION_NED':
                pos_ned = np.array([msg.x, msg.y, msg.z])
                vel_ned = np.array([msg.vx, msg.vy, msg.vz])
            elif msg.get_type() == 'ATTITUDE_QUATERNION':
                R_bn = quat_to_rot_ned(msg.q1, msg.q2, msg.q3, msg.q4)

        now = time.monotonic()
        trans_alpha = min(1.0, max(0.0, (now - trans_start_t) / TRANSITION_DUR))

        u, v, diam, init_diam, max_diam, lost_frames, det_age, _ = vision.get()
        det_fresh = (lost_frames == 0) and (det_age < 0.25)
        det_coast = det_age < COAST_TIMEOUT

        # 중기 유도 구간(온보드 비전 포착 전)에서 지상 카메라 UKF가 10Hz로 보내주는 최신 3D 좌표를 실시간 반영!
        ukf_world, ukf_sigma, ukf_n, ukf_valid, _, ukf_age, ukf_rx_cnt, ukf_hz = vision.get_ground_ukf()
        if (not vision_locked and not target_locked and ukf_world is not None
                and ukf_age < 1.0 and ukf_n >= UKF_MIN_SAMPLES and ukf_n != last_ukf_update_n):
            new_tn, new_te, new_td = world_to_ned(*ukf_world)
            new_target_ned = np.array([new_tn, new_te, new_td], dtype=float)
            if float(np.linalg.norm(new_target_ned - target_ned)) < 25.0:
                target_ned = 0.75 * target_ned + 0.25 * new_target_ned
                active_target_world = ukf_world
                path_vec = target_ned - start_ned
                path_len = float(np.linalg.norm(path_vec))
                if path_len > 1e-3:
                    path_dir = path_vec / path_len
                last_ukf_update_n = ukf_n

        diff_to_target = target_ned - pos_ned
        dist_3d = float(np.linalg.norm(diff_to_target))

        # ── "풍선 지났다!" 동시 판정(조건 1 AND 조건 2) 및 무조건 RTL 안전 조건 ──
        along_past = float(np.dot(pos_ned - target_ned, path_dir))
        if not target_locked and dist_3d <= LOCK_DIST:
            target_locked = True
            min_dist_after_lock = dist_3d
            print(f'▶ 표적 {LOCK_DIST:.0f}m 이내 진입 (dist={dist_3d:.2f}m): 풍선/목표점 통과 감시 활성화', flush=True)
        elif target_locked and dist_3d < min_dist_after_lock:
            min_dist_after_lock = dist_3d

        grew_large = (
            target_locked and
            max_diam >= PASS_MIN_DIAM_PX and
            (init_diam <= 0.0 or max_diam >= init_diam * PASS_GROWTH_RATIO)
        )
        suddenly_disappeared = (lost_frames >= PASS_LOST_FRAMES) or (det_age > 0.15)
        point_blank_impact = target_locked and det_fresh and (diam >= PASS_IMPACT_DIAM_PX)

        if not case1_vision_passed and ((grew_large and suddenly_disappeared) or point_blank_impact):
            case1_vision_passed = True
            print(
                f'✔ [조건 1 달성: 풍선 디텍팅 유실점 확인] '
                f'(초기 {init_diam:.1f}px -> 최대 {max_diam:.1f}px 확대 후 유실, 현재 목표점 거리={dist_3d:.2f}m)',
                flush=True)

        overshoot_passed = target_locked and ((dist_3d - min_dist_after_lock) >= OVERSHOOT_DIST)
        plane_passed = target_locked and (along_past >= PAST_PLANE_DIST)

        if not case2_target_passed and (overshoot_passed or plane_passed):
            case2_target_passed = True
            print(
                f'✔ [조건 2 달성: 목표점 통과 확인] '
                f'(최근접거리={min_dist_after_lock:.2f}m -> 현재거리={dist_3d:.2f}m, 목표평면 진행초과={along_past:+.2f}m)',
                flush=True)

        if case1_vision_passed and case2_target_passed:
            print('★ 풍선 지났다! [조건 1(풍선 확대 후 디텍팅 유실) & 조건 2(목표점 통과) 동시 만족!] -> RTL 복귀 시작!', flush=True)
            break

        force_plane_passed = target_locked and (along_past >= FORCE_PAST_PLANE_DIST)
        force_overshoot = target_locked and ((dist_3d - min_dist_after_lock) >= FORCE_OVERSHOOT_DIST)
        if force_plane_passed or force_overshoot:
            print(
                f'★ 목표점 통과/이탈 감지로 무조건 RTL! '
                f'(최근접거리={min_dist_after_lock:.2f}m -> 현재거리={dist_3d:.2f}m, 목표평면 진행초과={along_past:+.2f}m) -> RTL 복귀 시작!',
                flush=True)
            break

        # ── 테일시터 유도 명령 산출 (중기 UKF 유도 -> 종말 온보드 비전 LOS/PN 유도) ──
        if det_fresh and dist_3d > 1.5:
            los_candidate = pixel_to_los_ned(u, v, R_bn)
            if float(np.dot(los_candidate, path_dir)) >= 0.72:
                if not vision_locked:
                    print(
                        f'★ [온보드 노즈콘 카메라 풍선 포착!] (px=({u:.0f}, {v:.0f}), 지름={diam:.1f}px, 거리={dist_3d:.1f}m) '
                        f'-> 지상 UKF 중기 유도에서 온보드 3D LOS/PN 종말 비전 유도로 전환!',
                        flush=True)
                    vision_locked = True

                los_ned = los_candidate
                last_los_ned = los_ned

                # 온보드 광학 거리(R_opt = FX * D / diam)를 이용해 지상 단안 카메라의 거리 오차를 실시간 보정
                if 5.0 <= diam <= 70.0:
                    r_opt = (FX * BALLOON_DIAM_M) / diam
                    onboard_target_ned = pos_ned + r_opt * los_ned
                    if float(np.linalg.norm(onboard_target_ned - target_ned)) < 15.0:
                        target_ned = 0.82 * target_ned + 0.18 * onboard_target_ned
                        diff_to_target = target_ned - pos_ned
                        dist_3d = float(np.linalg.norm(diff_to_target))

                vel_along_los = float(np.dot(vel_ned, los_ned)) * los_ned
                vel_perp_los = vel_ned - vel_along_los
                pn_gain_eff = min(4.0, max(2.4, 2.8 * float(np.linalg.norm(vel_ned)) / max(dist_3d, 4.0)))
                a_perp_cmd = -vel_perp_los * pn_gain_eff

                lead_lim = MAX_LOS_LEAD_DEG if dist_3d > 8.0 else max(7.0, MAX_LOS_LEAD_DEG * (dist_3d / 8.0))
                nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                    los_ned, a_perp_cmd, vel_ned, dist_3d, lead_lim, trans_alpha, allow_dive=True)
                mode_str = f'TS_LOS_PURSUIT(px={u-CX:+.0f},{v-CY:+.0f} d={diam:.0f}px Vd={v_des:.1f})'
            else:
                coast_dir = last_los_ned if last_los_ned is not None else path_dir
                nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                    coast_dir, np.zeros(3), vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha, allow_dive=True)
                mode_str = 'TS_PASS_GUARD'

        elif vision_locked and det_coast and last_los_ned is not None:
            vel_along_los = float(np.dot(vel_ned, last_los_ned)) * last_los_ned
            vel_perp_los = vel_ned - vel_along_los
            a_perp_cmd = -vel_perp_los * 2.2
            nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                last_los_ned, a_perp_cmd, vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha, allow_dive=True)
            mode_str = 'TS_LOS_COAST'

        elif target_locked or case1_vision_passed:
            coast_dir = last_los_ned if last_los_ned is not None else path_dir
            nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                coast_dir, np.zeros(3), vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha, allow_dive=True)
            mode_str = 'TS_PASS_THROUGH'

        else:
            if vision_locked:
                print('온보드 비전 신호 일시 소실 -> 지상 UKF 추정 좌표 경로 비행으로 복귀', flush=True)
                vision_locked = False

            unit_to_target = diff_to_target / max(dist_3d, 0.1)
            if float(np.dot(unit_to_target, path_dir)) < 0.75:
                unit_to_target = path_dir

            prog = max(0.0, min(path_len, float(np.dot(pos_ned - start_ned, path_dir))))
            closest_pt = start_ned + prog * path_dir
            cross_pos_err = closest_pt - pos_ned

            vel_along = float(np.dot(vel_ned, unit_to_target)) * unit_to_target
            vel_perp = vel_ned - vel_along
            a_perp_cmd = cross_pos_err * 1.8 - vel_perp * 1.6

            nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                unit_to_target, a_perp_cmd, vel_ned, dist_3d, MAX_PATH_LEAD_DEG, trans_alpha, allow_dive=False)
            ukf_tag = f'UKF(n={ukf_n},s={ukf_sigma:.1f}m,{ukf_hz:.0f}Hz)' if ukf_n > 0 else 'UKF_INIT'
            mode_str = f'TS_UKF_GUIDE[{ukf_tag}](Vd={v_des:.1f} thr={thrust_cmd:.2f})'

        filt_alpha = 0.28 if trans_alpha >= 1.0 else 0.18
        smoothed_nose_ned = (1.0 - filt_alpha) * smoothed_nose_ned + filt_alpha * nose_cmd
        smoothed_nose_ned /= np.linalg.norm(smoothed_nose_ned)

        q_cmd = nose_dir_to_quat_frd(smoothed_nose_ned, yaw)
        send_attitude_target(conn, q_cmd, thrust_cmd)

        if now - last_log > 0.4:
            cur_spd = float(np.linalg.norm(vel_ned))
            nose_ned = R_bn @ np.array([0.0, 0.0, -1.0])
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -nose_ned[2]))))
            nose_elev_deg = 90.0 - tilt_deg
            print(f'  [{mode_str}] 거리 {dist_3d:5.1f}m | 고도 {-pos_ned[2]:4.1f}m | 속도 {cur_spd:4.1f}m/s | '
                  f'기울기(Tilt) {tilt_deg:4.1f}° (카메라앙각 {nose_elev_deg:4.1f}°)', flush=True)
            last_log = now

        time.sleep(0.02)

    # ── 4단계: 테일시터 후면 전환(Back-Transition: 수직 자세 복귀 및 감속) -> RTL ──
    print('풍선 통과 완료 -> 테일시터 수직 호버링 복귀 전환(Back-Transition) 및 감속 시작 ...', flush=True)
    brake_start = time.monotonic()
    while time.monotonic() - brake_start < 3.5:
        while True:
            msg = conn.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION'], blocking=False)
            if msg is None:
                break
            if msg.get_type() == 'LOCAL_POSITION_NED':
                pos_ned = np.array([msg.x, msg.y, msg.z])
                vel_ned = np.array([msg.vx, msg.vy, msg.vz])
            elif msg.get_type() == 'ATTITUDE_QUATERNION':
                R_bn = quat_to_rot_ned(msg.q1, msg.q2, msg.q3, msg.q4)

        cur_spd = float(np.linalg.norm(vel_ned))
        if cur_spd < 2.8:
            print(f'  테일시터 감속 및 수직 복귀 완료 (현재 속도 {cur_spd:.1f}m/s) -> RTL 전환!', flush=True)
            break

        horiz_vel = np.array([vel_ned[0], vel_ned[1], 0.0], dtype=float)
        brake_tilt = clamp_vec(-horiz_vel * 0.08, 0.45)
        brake_nose = np.array([brake_tilt[0], brake_tilt[1], -1.0], dtype=float)
        brake_nose /= np.linalg.norm(brake_nose)

        q_brake = nose_dir_to_quat_frd(brake_nose, yaw)
        vz_err = 0.0 - float(vel_ned[2])
        brake_thr = max(0.18, min(0.45, 0.284 - 0.04 * vz_err))
        send_attitude_target(conn, q_brake, brake_thr)
        time.sleep(0.02)

    for _ in range(25):
        send_pos(conn, pos_ned[0], pos_ned[1], pos_ned[2], yaw)
        time.sleep(0.02)

    print('RTL(Return to Launch) 명령 전송 ...', flush=True)
    for _ in range(5):
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,
            0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.1)

    print('RTL 복귀 및 자동 착륙 진행 중 (Ctrl+C로 종료 가능) ...', flush=True)
    try:
        while True:
            msg = conn.recv_match(type=['LOCAL_POSITION_NED', 'HEARTBEAT'], blocking=True, timeout=0.5)
            if msg and msg.get_type() == 'LOCAL_POSITION_NED':
                now = time.monotonic()
                if now - last_log > 1.0:
                    home_dist = math.hypot(msg.x, msg.y)
                    print(f'  [RTL] 홈까지 수평거리 {home_dist:5.1f}m | 고도 {-msg.z:4.1f}m', flush=True)
                    last_log = now
            elif msg and msg.get_type() == 'HEARTBEAT':
                if not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    print('★ RTL 착륙 및 Disarm 완료!', flush=True)
                    break
    except KeyboardInterrupt:
        pass

    conn.close()
    print('종료', flush=True)


if __name__ == '__main__':
    main()
