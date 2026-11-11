#!/usr/bin/env python3
"""PX4 쿼드 테일시터(Quad-Tailsitter, SYS_AUTOSTART=1045) 3D 비전 LOS/PN 유도 제어기.

동작 순서:
  1. [수직 이륙 (MC Hover)]
     홈 위치(-2.0, 0.0, 0.2)m에서 5m 고도까지 수직 이륙하여 자세를 안정화합니다.
  2. [테일시터 전면 피치다운 전환 & 3D 대각선 상승 (TS_DIAGONAL)]
     SET_ATTITUDE_TARGET(날개 수평 3D 쿼터니언 + 정규화 추력)을 통해 기수를 전방 대각선(~68° 기울기,
     카메라 앙각 ~22°)으로 숙이고, 주익 양력(libLiftDragPlugin.so)으로 하중을 지지하며 목표 풍선(90, 0, 40)m을 향해 비행합니다.
  3. [온보드 카메라 30cm 풍선 검출 & 3D LOS/PN 중심 정렬 (TS_LOS_PURSUIT)]
     노즈콘 카메라(/balloon/onboard/detections)에 풍선이 포착되면 즉시 3D 시선(LOS) 단위벡터와
     무잡음 3D 비례항법(PN) 속도 정렬 보정으로 날개 롤(Roll=0°)을 고정한 채 풍선 정중앙을 조준·통과합니다.
  4. [통과 판정 & 후면 전환(Back-Transition) 감속 후 RTL 착륙]
     표적 반경 15m 진입 후 [조건 1: 풍선 확대 후 디텍팅 유실] AND [조건 2: 목표점 통과] 동시 만족 시
     (또는 목표 평면/최근접 거리 초과 시 무조건) 수직 호버링 자세로 복귀 감속한 뒤 RTL로 귀환·착륙합니다.

실행:
    ./tools/run_tailsitter_sim.sh
"""
import math
import os
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

try:
    import rclpy
    from vision_msgs.msg import Detection2DArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. 비행 및 유도 제어 파라미터
# ═══════════════════════════════════════════════════════════════════════════
TARGET_WORLD = [
    float(os.environ.get('FLY_TARGET_X', '90.0')),
    float(os.environ.get('FLY_TARGET_Y', '0.0')),
    float(os.environ.get('FLY_TARGET_Z', '40.0')),
]
HOVER_ALT = float(os.environ.get('FLY_HOVER_ALT', '5.0'))

NORTH_YAW  = math.pi / 2.0
HOME_WORLD = [-2.0, 0.0, 0.2]
PX4_UDP    = 'udpin:0.0.0.0:14540'

# 테일시터 속도 및 추력 제어 파라미터
TAKEOFF_ALT       = 5.0    # 수직 이륙 목표 고도 [m]
TRANSITION_DUR    = 1.6    # 수직 호버링 -> 테일시터 대각선 자세 전환 시간 [s]
MIN_CRUISE_SPEED  = 10.5   # 근접 구간 최소 목표 속도 [m/s]
MAX_CRUISE_SPEED  = 15.0   # 원거리 구간 최대 목표 속도 [m/s]
KP_DIST_SPEED     = 0.06   # 거리 비례 속도 이득: V_des = clip(10.5 + 0.06 * dist_3d, 10.5, 15.0)
LIFT_CRUISE_SPEED = 14.5   # 주익 양력이 중력 수직 성분을 완전히 지지하는 기준 속도 [m/s]
SPEC_THRUST_MAX   = 34.52  # 4로터 최대 비추력 (69.04 N / 2.0 kg) [m/s^2]
THRUST_MIN        = 0.12   # 비행 중 최소 정규화 추력 (차동 추력 자세 제어권 보장)
THRUST_MAX        = 0.72   # 비행 중 최대 정규화 추력
MAX_LOS_LEAD_DEG  = 13.0   # 비전 추적 시 LOS 대비 노즈콘 최대 리드각 [deg] (화각 내부 유지 보장)
MAX_PATH_LEAD_DEG = 24.0   # 대각선 경로 추종 시 최대 리드각 [deg]
COAST_TIMEOUT     = 0.8    # 비전 일시 소실 시 마지막 LOS 유지 시간 [s]

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
    """Gazebo World (X, Y, Z) -> PX4 Local NED (N, E, D) 변환."""
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    dx, dy, dz = wx - HOME_WORLD[0], wy - HOME_WORLD[1], wz - HOME_WORLD[2]
    return c * dx + s * dy, s * dx - c * dy, -dz


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


def compute_tailsitter_cmd(aim_dir_ned, a_perp_cmd, vel_ned, dist_3d, max_lead_deg, trans_alpha):
    """목표 지향 벡터(aim_dir_ned) 및 수직 보정 가속도(a_perp_cmd)로부터
    테일시터 노즈콘 지향 벡터(nose_dir)와 정규화 추력(thrust)을 분리 산출.
    """
    d = np.array(aim_dir_ned, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1e-6)

    cur_spd = float(np.linalg.norm(vel_ned))
    v_along = float(np.dot(vel_ned, d))

    # 저속 구간에서는 부족한 날개 양력만큼 위쪽(-D) 중력 보상을 추가하고, 순항 속도에서는 날개 양력이 지지
    lift_frac = min(1.0, (cur_spd / LIFT_CRUISE_SPEED) ** 2)
    g_uncomp_ned = np.array([0.0, 0.0, 9.81 * (1.0 - 0.92 * lift_frac)], dtype=float)

    g_perp = g_uncomp_ned - float(np.dot(g_uncomp_ned, d)) * d
    lead_acc_perp = (a_perp_cmd - float(np.dot(a_perp_cmd, d)) * d) - g_perp

    ref_acc = 11.0
    max_perp_mag = ref_acc * math.tan(math.radians(max_lead_deg))
    lead_acc_perp = clamp_vec(lead_acc_perp, max_perp_mag)

    raw_nose = d + (lead_acc_perp / ref_acc)
    raw_nose /= np.linalg.norm(raw_nose)

    if raw_nose[2] > -0.05:
        raw_nose[2] = -0.05
        raw_nose /= np.linalg.norm(raw_nose)

    hover_nose = np.array([0.0, 0.0, -1.0], dtype=float)
    nose_dir = (1.0 - trans_alpha) * hover_nose + trans_alpha * raw_nose
    nose_dir /= np.linalg.norm(nose_dir)

    # 남은 거리(dist_3d)에 비례하는 목표 접근 속도 추종 추력 계산
    v_target = min(MAX_CRUISE_SPEED, max(MIN_CRUISE_SPEED, MIN_CRUISE_SPEED + KP_DIST_SPEED * dist_3d))
    speed_err = v_target - v_along

    grav_along = max(0.0, -9.81 * float(nose_dir[2]))
    aero_drag = 0.0045 * (cur_spd ** 2)
    low_spd_support = 9.81 * (1.0 - lift_frac) * 0.45
    a_thrust_cmd = grav_along + aero_drag + low_spd_support + 1.8 * speed_err

    thrust = max(THRUST_MIN, min(THRUST_MAX, a_thrust_cmd / SPEC_THRUST_MAX))
    return nose_dir, thrust, v_target


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROS 2 온보드 비전 검출 수신 및 MAVLink 통신 헬퍼
# ═══════════════════════════════════════════════════════════════════════════
class VisionListener:
    """ROS 2 /balloon/onboard/detections 토픽 비동기 수신 상태 관리."""

    def __init__(self):
        self.lock = threading.Lock()
        self.u = 0.0
        self.v = 0.0
        self.diam = 0.0
        self.initial_diam = 0.0
        self.max_diam = 0.0
        self.last_rx = 0.0
        self.lost_frames = 0

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


def start_ros_vision_thread(vision_state):
    if not HAS_ROS2:
        print('WARNING: rclpy를 불러올 수 없어 비전 검출 없이 실행합니다.', flush=True)
        return

    def _spin():
        rclpy.init()
        node = rclpy.create_node('tailsitter_vision_guidance_listener')

        def _cb(msg):
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

        node.create_subscription(Detection2DArray, '/balloon/onboard/detections', _cb, 10)
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
# 4. 메인 비행 시퀀스
# ═══════════════════════════════════════════════════════════════════════════
def main():
    startup_delay = float(os.environ.get('FLY_STARTUP_DELAY', '0'))
    if startup_delay > 0:
        print(f'PX4 부팅 대기 {startup_delay:.0f}초 ...', flush=True)
        time.sleep(startup_delay)

    vision = VisionListener()
    start_ros_vision_thread(vision)

    tn, te, t_d = world_to_ned(*TARGET_WORLD)
    _, _, takeoff_d = world_to_ned(HOME_WORLD[0], HOME_WORLD[1], HOME_WORLD[2] + TAKEOFF_ALT)
    yaw = math.atan2(te, tn)

    print(f'목표 world    {TARGET_WORLD}', flush=True)
    print(f'목표 NED      N={tn:.1f}  E={te:.1f}  D={t_d:.1f}', flush=True)
    print(f'이륙 고도     {TAKEOFF_ALT} m (D={takeoff_d:.1f})', flush=True)
    print('펌웨어/유도   PX4 Quad-Tailsitter(1045) + 3D 노즈콘 자세/추력 분리 제어 + 비전 LOS/PN 유도', flush=True)
    print(f'초기 헤딩     {math.degrees(yaw):.1f}°\n', flush=True)

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

    print('setpoint 스트림 시작 ...', flush=True)
    for _ in range(40):
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(0.05)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        6, 0, 0, 0, 0, 0)
    print('OFFBOARD 요청', flush=True)
    time.sleep(0.3)

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

    # ── 1단계: 5m 수직 이륙 ──
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

    # ── 2단계 & 3단계: 테일시터 피치다운 전환 + 대각선 비행 -> 풍선 검출 시 3D LOS/PN 추적 ──
    start_ned = pos_ned.copy()
    target_ned = np.array([tn, te, t_d], dtype=float)
    path_vec = target_ned - start_ned
    path_len = float(np.linalg.norm(path_vec))
    path_dir = path_vec / path_len if path_len > 1e-3 else np.array([1.0, 0.0, 0.0])

    print(f'테일시터 전진 피치다운 전환 및 대각선 비행 시작: NED({start_ned[0]:.1f}, {start_ned[1]:.1f}, {start_ned[2]:.1f}) '
          f'-> 목표 NED({tn:.1f}, {te:.1f}, {t_d:.1f})', flush=True)

    trans_start_t = time.monotonic()
    last_log = 0.0
    last_los_ned = None
    smoothed_nose_ned = np.array([0.0, 0.0, -1.0], dtype=float)
    vision_locked = False
    target_locked = False
    min_dist_after_lock = math.inf
    case1_vision_passed = False
    case2_target_passed = False

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

        # ── 테일시터 유도 명령 산출 ──
        if det_fresh and dist_3d > 1.8:
            los_candidate = pixel_to_los_ned(u, v, R_bn)
            if float(np.dot(los_candidate, path_dir)) >= 0.75:
                if not vision_locked:
                    print(f'★ 풍선 검출! (px=({u:.0f}, {v:.0f}), 지름={diam:.1f}px, 거리={dist_3d:.1f}m) -> 테일시터 3D LOS/PN 추적 전환',
                          flush=True)
                    vision_locked = True

                los_ned = los_candidate
                last_los_ned = los_ned

                vel_along_los = float(np.dot(vel_ned, los_ned)) * los_ned
                vel_perp_los = vel_ned - vel_along_los
                pn_gain_eff = min(3.2, max(1.8, 2.2 * float(np.linalg.norm(vel_ned)) / max(dist_3d, 5.0)))
                a_perp_cmd = -vel_perp_los * pn_gain_eff

                nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                    los_ned, a_perp_cmd, vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha)
                mode_str = f'TS_LOS_PURSUIT(px={u-CX:+.0f},{v-CY:+.0f} d={diam:.0f}px Vd={v_des:.1f})'
            else:
                coast_dir = last_los_ned if last_los_ned is not None else path_dir
                nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                    coast_dir, np.zeros(3), vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha)
                mode_str = 'TS_PASS_GUARD'

        elif vision_locked and det_coast and last_los_ned is not None:
            vel_along_los = float(np.dot(vel_ned, last_los_ned)) * last_los_ned
            vel_perp_los = vel_ned - vel_along_los
            a_perp_cmd = -vel_perp_los * 1.6
            nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                last_los_ned, a_perp_cmd, vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha)
            mode_str = 'TS_LOS_COAST'

        elif target_locked or case1_vision_passed:
            coast_dir = last_los_ned if last_los_ned is not None else path_dir
            nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
                coast_dir, np.zeros(3), vel_ned, dist_3d, MAX_LOS_LEAD_DEG, trans_alpha)
            mode_str = 'TS_PASS_THROUGH'

        else:
            if vision_locked:
                print('비전 신호 소실 -> 테일시터 대각선 경로 비행으로 복귀', flush=True)
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
                unit_to_target, a_perp_cmd, vel_ned, dist_3d, MAX_PATH_LEAD_DEG, trans_alpha)
            mode_str = f'TS_DIAGONAL(Vd={v_des:.1f} thr={thrust_cmd:.2f})'

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
