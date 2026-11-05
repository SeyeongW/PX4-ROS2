#!/usr/bin/env python3
"""인터셉터 드론 유도 스크립트: 가속도(ACC_ONLY) 제어를 통한 강제 기울기 유지 + 대각선 비행 + 비전 LOS/PN 유도.

핵심 원리 (인터셉터 드론이 수평으로 서지 않고 기울어져서 날아가는 원리):
- 일반 위치/속도(Position/Velocity) 제어기를 쓰면 등속 구간에서 가속도가 0이 되어
  PX4가 중력만 보상하려고 기체를 다시 꼿꼿이 수직으로 세워버림 -> 상단 카메라가 하늘만 보게 됨.
- 반면 가속도(Acceleration, ACC_ONLY) 명령을 직접 인가하면:
  PX4의 추력 벡터 T = a_cmd - [0, 0, g] 방향으로 기체 상단(+Z_body = 노즈콘/카메라)이 강제로 정렬됨!
- 따라서:
  1) 5m 수직 이륙 후,
  2) 풍선 좌표까지의 3D 대각선 경로를 따라 지속 전방 가속도(a_forward) + 경로 추종 가속도(a_cross)를 인가하면,
     기체가 앞으로 확 숙여진(Pitch-down 45°~60°) 상태를 끝까지 유지하며 대각선으로 가속 비행함 -> 노즈콘 카메라가 전방 대각선의 풍선을 정면으로 바라봄.
  3) 카메라에 풍선이 디텍팅되면 즉시 3D LOS(시선) 방향 가속도 + 비례항법(PN) 횡가속도를 인가해 풍선 정중앙을 타격함.

실행:
    ./tools/run_vision_center.sh
"""
import math
import os
import sys
import time
import threading

import numpy as np
from pymavlink import mavutil

try:
    import rclpy
    from vision_msgs.msg import Detection2DArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# ═══════════════════════════════════════════════════════════
TARGET_WORLD = [
    float(os.environ.get('FLY_TARGET_X', '90.0')),
    float(os.environ.get('FLY_TARGET_Y', '0.0')),
    float(os.environ.get('FLY_TARGET_Z', '40.0')),
]
HOVER_ALT = float(os.environ.get('FLY_HOVER_ALT', '5.0'))

NORTH_YAW     = math.pi / 2
HOME_WORLD    = [-2.0, 0.0, 0.2]
PX4_UDP       = 'udpin:0.0.0.0:14540'

TAKEOFF_ALT        = 5.0    # 이륙 고도 (m)
MIN_TILT_ACC       = 6.0    # 근접 시에도 ~35° 이상 전방 기울기를 유지하기 위한 최소 전방 가속도 [m/s^2]
MAX_FWD_ACC        = 8.5    # 초기 출발 시 과도한 폭주 없이 안정적인 전방 기울기(~40°)를 만드는 최대 전방 가속도 [m/s^2]
KP_DIST_ACC        = 0.03   # 거리 오차(dist_3d) 비례 전방 가속도 이득: a_fwd = clip(6.0 + 0.03 * d, 6.0, 8.5)
MAX_APPROACH_SPEED = 22.0   # 접근 최대 속도 제한 [m/s] (초과 시 가속도를 부드럽게 줄여 폭주 및 EKF 리셋 방지)
MAX_ACC            = 12.0   # 전체 가속도 벡터 클램프 [m/s^2]
MAX_PERP_ACC       = 3.5    # 경로/LOS 수직 보정 가속도 최대치 [m/s^2] (전방 가속을 압도해 공중 회전하는 현상 방지)
PN_GAIN            = 3.0    # 비전 LOS 비례항법(PN) 이득
COAST_TIMEOUT      = 0.8    # 비전 일시 소실 시 마지막 LOS 방향 유지 시간 (s)

# ── "풍선 지났다!" (Pass-through) 판정 파라미터 (30cm 풍선 기준) ──
# [조건 1] 풍선 디텍팅 유실점 파악 (풍선 크기가 커졌다가 갑자기 사라졌는지)
PASS_MIN_DIAM_PX    = 14.0   # 30cm 풍선 기준 약 10m 이내 근접 시 픽셀 크기 (초기 ~3.5px 대비 4배 확대)
PASS_GROWTH_RATIO   = 1.4    # 최초 검출 대비 최소 1.4배 이상 커졌을 때 (확대 확인)
PASS_LOST_FRAMES    = 3      # 풍선이 커진 상태에서 연속 N프레임(~0.1초) 갑자기 사라지면 유실점 판정
PASS_IMPACT_DIAM_PX = 80.0   # 30cm 풍선 초근접(~1.8m 이내) 시 즉시 유실/통과 판정

# [조건 2] 목표점을 통과했는지 (조건 1과 동시 판정용)
LOCK_DIST           = 15.0   # 목표점 근접 감시 시작 거리 (m)
OVERSHOOT_DIST      = 1.0    # 최근접 거리 찍고 이만큼 멀어지면 목표점 통과 판정 (m)
PAST_PLANE_DIST     = 0.5    # 목표점 수직 평면(path_dir 기준)을 이 거리(m) 이상 넘어가면 목표점 통과 판정 (m)

# [무조건 RTL 안전 조건] 목표점을 지나치거나 일정 거리 이상 멀어지면 조건 1 무관 무조건 RTL
FORCE_PAST_PLANE_DIST = 1.5  # 목표점 수직 평면을 1.5m 이상 지나치면 무조건 RTL (m)
FORCE_OVERSHOOT_DIST  = 2.0  # 최근접 거리 대비 2.0m 이상 멀어지면 무조건 RTL (m)

# 온보드 카메라 파라미터: Raspberry Pi Camera Module 3 Standard (수직 노즈콘 고정)
# sensor_rpy = [pi/2, -pi/2, 0]:
#   카메라 광축(정면) = 기체 상단 노즈콘 방향 (FLU +Z = FRD -Z)
#   이미지 우측 (+u)  = 기체 전방 Pitch-down 방향 (FLU +X = FRD +X)
#   이미지 하단 (+v)  = 기체 좌측 (FLU +Y = FRD -Y)
#   수평 화각(HFOV) = 66° (이미지 가로 방향 = 기체 Pitch 방향, 중심에서 ±33°)
#   수직 화각(VFOV) = 41° (이미지 세로 방향 = 기체 Roll/좌우 방향, 중심에서 ±20.5°)
IMG_W = 640
IMG_H = 360
HFOV_DEG = 66.0
CX = (IMG_W - 1) / 2.0
CY = (IMG_H - 1) / 2.0
FX = IMG_W / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
FY = FX
# ═══════════════════════════════════════════════════════════


def world_to_ned(wx, wy, wz):
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    dx, dy, dz = wx - HOME_WORLD[0], wy - HOME_WORLD[1], wz - HOME_WORLD[2]
    return c * dx + s * dy, s * dx - c * dy, -dz


def quat_to_rot_ned(w, x, y, z):
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix (Body FRD -> NED)."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def pixel_to_los_ned(u, v, R_bn):
    """상단 노즈콘 카메라 픽셀 (u, v) -> NED 좌표계 3D 시선(LOS) 단위벡터.

    stl_airframe.py의 sensor_rpy = [pi/2, -pi/2, 0] 기준:
    - 광축(카메라 정면 +Z_opt) = 기체 위쪽 노즈콘 (FLU +Z = Body FRD -Z)
    - 이미지 오른쪽 (+X_opt, u > CX) = 기체 Pitch-down 방향 (FLU +X = Body FRD +X)
    - 이미지 아래쪽 (+Y_opt, v > CY) = 기체 왼쪽 (FLU +Y = Body FRD -Y)
    """
    ex = (u - CX) / FX
    ey = (v - CY) / FY
    # Body FRD: [Forward(+X), Right(+Y), Down(+Z)]
    ray_body_frd = np.array([ex, -ey, -1.0], dtype=float)
    ray_body_frd /= np.linalg.norm(ray_body_frd)
    los_ned = R_bn @ ray_body_frd
    return los_ned / np.linalg.norm(los_ned)


def clamp_vec(vec, max_mag):
    mag = float(np.linalg.norm(vec))
    if mag > max_mag and mag > 1e-6:
        return vec * (max_mag / mag)
    return vec


def enforce_forward_tilt(a_cmd, yaw):
    """수직 보정 가속도(a_cross / a_align) 때문에 전방 가속이 상쇄되거나 뒤로 꺾여
    기체가 공중에서 뒤집히거나(한바퀴 회전) 수직으로 서버리는 현상을 원천 차단.

    물리식: tan(tilt) = a_horiz_fwd / (9.81 - a_D)
    - a_D를 [-4.2, +2.0] 범위로 제한하고,
    - 진행 방향(yaw) 수평 가속도가 최소 0.65 * (9.81 - a_D) 이상이 되도록 보장하여
      기체 기울기(Tilt)가 항상 전방 33°~48° 범위 안에 안정적으로 유지되게 함.
    """
    out = np.array(a_cmd, dtype=float)
    out[2] = max(-4.2, min(2.0, float(out[2])))

    fwd_h = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    lat_h = np.array([-math.sin(yaw), math.cos(yaw)], dtype=float)
    a_h = out[:2]

    a_fwd_h = float(np.dot(a_h, fwd_h))
    a_lat_h = float(np.dot(a_h, lat_h))

    min_fwd_h = max(5.2, 0.65 * (9.81 - out[2]))
    if a_fwd_h < min_fwd_h:
        a_fwd_h = min_fwd_h
    a_lat_h = max(-3.5, min(3.5, a_lat_h))

    out[:2] = fwd_h * a_fwd_h + lat_h * a_lat_h
    return clamp_vec(out, MAX_ACC)


class VisionListener:
    """ROS 2 /balloon/onboard/detections 토픽을 백그라운드 스레드에서 수신.

    매 카메라 프레임(30Hz)마다 콜백이 호출되며,
    - 검출 성공 시: 픽셀 좌표(u, v), 현재 지름(diam), 최초 검출 지름(initial_diam), 최대 지름(max_diam) 갱신, lost_frames = 0
    - 검출 실패(빈 배열) 시: 연속 유실 프레임 수(lost_frames) 증가 -> '커졌다가 갑자기 사라진' 유실점 즉시 포착!
    """
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
        node = rclpy.create_node('fly_to_target_vision_listener')

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

    t = threading.Thread(target=_spin, daemon=True)
    t.start()


def heartbeat_thread(conn):
    while True:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        time.sleep(1)


def send_pos(conn, n, e, d, yaw, vn=0.0, ve=0.0, vd=0.0):
    """Position + velocity + yaw (이륙 및 호버링용)."""
    type_mask = 0b0000_100_111_000_000
    conn.mav.set_position_target_local_ned_send(
        0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
        n, e, d, vn, ve, vd, 0, 0, 0, yaw, 0)


def send_acc(conn, an, ae, ad, yaw):
    """Acceleration + yaw (인터셉터 가속/기울기 강제 유지 제어).
    Position과 Velocity를 모두 Ignore(0b0000_100_000_111_111)하고 가속도 벡터만 전달.
    PX4는 추력 방향을 (a_cmd - g)에 맞추기 때문에 기체가 목표 방향으로 확 기울어진 상태를 유지함!
    """
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )
    conn.mav.set_position_target_local_ned_send(
        0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
        0, 0, 0, 0, 0, 0, float(an), float(ae), float(ad), float(yaw), 0)


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
    time.sleep(0.1)
    ack = conn.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
    if ack:
        print(f'  PARAM {name} = {ack.param_value:.1f}', flush=True)
    else:
        print(f'  PARAM {name} = {value:.1f} (no ack)', flush=True)


def main():
    startup_delay = float(os.environ.get('FLY_STARTUP_DELAY', '0'))
    if startup_delay > 0:
        print(f'PX4 부팅 대기 {startup_delay:.0f}초 ...', flush=True)
        time.sleep(startup_delay)

    vision = VisionListener()
    start_ros_vision_thread(vision)

    # 목표 = 송신받은 풍선 위치 (고도 포함)
    tn, te, t_d = world_to_ned(*TARGET_WORLD)
    # 5m 이륙 목표 고도 (NED D 좌표)
    _, _, takeoff_d = world_to_ned(HOME_WORLD[0], HOME_WORLD[1], HOME_WORLD[2] + TAKEOFF_ALT)
    yaw = math.atan2(te, tn)

    print(f'목표 world    {TARGET_WORLD}', flush=True)
    print(f'목표 NED      N={tn:.1f}  E={te:.1f}  D={t_d:.1f}', flush=True)
    print(f'이륙 고도     {TAKEOFF_ALT} m (D={takeoff_d:.1f})', flush=True)
    print(f'유도 방식     인터셉터 가속도(ACC_ONLY) 유도 -> 기울기 강제 유지 + LOS/PN 중심 타격', flush=True)
    print(f'초기 헤딩     {math.degrees(yaw):.1f}°', flush=True)
    print(flush=True)

    # ── PX4 연결 ──
    print(f'PX4 연결 대기  {PX4_UDP} ...', flush=True)
    conn = mavutil.mavlink_connection(PX4_UDP)
    conn.wait_heartbeat(timeout=60)
    print(f'연결 OK  sysid={conn.target_system}', flush=True)

    t = threading.Thread(target=heartbeat_thread, args=(conn,), daemon=True)
    t.start()

    # ── 텔레메트리 스트림 요청 (위치 + 자세 쿼터니언 60Hz) ──
    request_stream(conn, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 60)
    request_stream(conn, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 60)

    # ── 최대 기울기 제한 해제 (아까 성공했던 조건 복원: 65°까지 숙일 수 있게 허용) ──
    print('파라미터 설정 ...', flush=True)
    set_param(conn, 'MPC_TILTMAX_AIR', 65.0)
    set_param(conn, 'MPC_XY_VEL_MAX', 30.0)
    set_param(conn, 'MPC_Z_VEL_MAX_UP', 15.0)
    set_param(conn, 'MPC_Z_VEL_MAX_DN', 10.0)
    set_param(conn, 'MPC_ACC_HOR', 15.0)
    set_param(conn, 'MPC_ACC_HOR_MAX', 20.0)
    set_param(conn, 'MPC_ACC_UP_MAX', 15.0)
    set_param(conn, 'MPC_ACC_DOWN_MAX', 12.0)

    # ── 사전 setpoint (이륙용 position mode) ──
    print('setpoint 스트림 시작 ...', flush=True)
    for _ in range(40):
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(0.05)

    # ── OFFBOARD + ARM ──
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
    print('ARM 완료 — 5m 수직 이륙 시작', flush=True)

    # ── 1단계: 5m 이륙 (position mode) ──
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
                if abs(msg.z - takeoff_d) < 0.4:
                    print(f'5m 이륙 완료! 현재고도 {-msg.z:.2f}m (목표 {-takeoff_d:.2f}m)', flush=True)
                    break
            elif msg.get_type() == 'ATTITUDE_QUATERNION':
                R_bn = quat_to_rot_ned(msg.q1, msg.q2, msg.q3, msg.q4)
    else:
        print('WARNING: 이륙 고도 도달 시간 초과, 다음 단계로 진행', flush=True)

    # ── 2단계 & 3단계: 가속도 기반 대각선 비행 → 풍선 검출 시 LOS/PN 가속도 추적 ──
    start_ned = pos_ned.copy()
    target_ned = np.array([tn, te, t_d], dtype=float)
    path_vec = target_ned - start_ned
    path_len = float(np.linalg.norm(path_vec))
    path_dir = path_vec / path_len if path_len > 1e-3 else np.array([1.0, 0.0, 0.0])

    print(f'인터셉터 대각선 가속 비행 시작: NED({start_ned[0]:.1f}, {start_ned[1]:.1f}, {start_ned[2]:.1f}) '
          f'-> 목표 NED({tn:.1f}, {te:.1f}, {t_d:.1f})', flush=True)

    last_log = 0.0
    prev_los_ned = None
    prev_los_t = None
    last_los_ned = None
    vision_locked = False
    target_locked = False
    min_dist_after_lock = math.inf
    global_min_dist = math.inf
    case1_vision_passed = False
    case2_target_passed = False

    while True:
        while True:
            msg = conn.recv_match(
                type=['LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION'],
                blocking=False)
            if msg is None:
                break
            if msg.get_type() == 'LOCAL_POSITION_NED':
                pos_ned = np.array([msg.x, msg.y, msg.z])
                vel_ned = np.array([msg.vx, msg.vy, msg.vz])
            elif msg.get_type() == 'ATTITUDE_QUATERNION':
                R_bn = quat_to_rot_ned(msg.q1, msg.q2, msg.q3, msg.q4)

        u, v, diam, init_diam, max_diam, lost_frames, det_age, det_t = vision.get()
        det_fresh = (lost_frames == 0) and (det_age < 0.25)
        det_coast = det_age < COAST_TIMEOUT

        diff_to_target = target_ned - pos_ned
        dist_3d = float(np.linalg.norm(diff_to_target))
        dist_2d = float(math.hypot(diff_to_target[0], diff_to_target[1]))
        if dist_3d < global_min_dist:
            global_min_dist = dist_3d

        # ══════════════════════════════════════════════════════════════════
        # "풍선 지났다!" 동시 판정(조건 1 AND 조건 2) + 목표점 초과/이탈 시 무조건 RTL
        # ══════════════════════════════════════════════════════════════════
        # [조건 1] 풍선 디텍팅 유실점 파악 (풍선의 크기가 커졌다가 갑자기 사라졌는지)
        grew_large = (
            max_diam >= PASS_MIN_DIAM_PX and
            (init_diam <= 0.0 or max_diam >= init_diam * PASS_GROWTH_RATIO)
        )
        suddenly_disappeared = (lost_frames >= PASS_LOST_FRAMES) or (det_age > 0.15)
        point_blank_impact = det_fresh and (diam >= PASS_IMPACT_DIAM_PX)

        if not case1_vision_passed and ((grew_large and suddenly_disappeared) or point_blank_impact):
            case1_vision_passed = True
            print(
                f'✔ [조건 1 달성: 풍선 디텍팅 유실점 확인] '
                f'(초기 {init_diam:.1f}px -> 최대 {max_diam:.1f}px 확대 후 유실, 현재 목표점 거리={dist_3d:.2f}m)',
                flush=True)

        # [조건 2] 목표점을 통과했는지 (송신받은 목표 좌표 통과 판정)
        along_past = float(np.dot(pos_ned - target_ned, path_dir))
        if not target_locked and dist_3d <= LOCK_DIST:
            target_locked = True
            min_dist_after_lock = dist_3d
            print(f'▶ 표적 {LOCK_DIST:.0f}m 이내 진입 (dist={dist_3d:.2f}m): 목표점 통과 감시 시작', flush=True)
        elif target_locked and dist_3d < min_dist_after_lock:
            min_dist_after_lock = dist_3d

        overshoot_passed = target_locked and ((dist_3d - min_dist_after_lock) >= OVERSHOOT_DIST)
        plane_passed = along_past >= PAST_PLANE_DIST

        if not case2_target_passed and (overshoot_passed or plane_passed):
            case2_target_passed = True
            print(
                f'✔ [조건 2 달성: 목표점 통과 확인] '
                f'(최근접거리={min_dist_after_lock:.2f}m -> 현재거리={dist_3d:.2f}m, 목표평면 진행초과={along_past:+.2f}m)',
                flush=True)

        # ★ 1) 조건 1(풍선 확대 후 유실) AND 조건 2(목표점 통과) 동시 만족 시 RTL!
        if case1_vision_passed and case2_target_passed:
            print(
                f'★ 풍선 지났다! [조건 1(풍선 확대 후 디텍팅 유실) & 조건 2(목표점 통과) 동시 만족!] -> RTL 복귀 시작!',
                flush=True)
            break

        # ★ 2) 목표점을 확실히 지나치거나 최근접 거리 대비 일정 거리 이상 멀어지면 무조건 RTL!
        force_plane_passed = along_past >= FORCE_PAST_PLANE_DIST
        force_overshoot = target_locked and ((dist_3d - min_dist_after_lock) >= FORCE_OVERSHOOT_DIST)
        if force_plane_passed or force_overshoot:
            print(
                f'★ 목표점 통과/이탈 감지로 무조건 RTL! '
                f'(최근접거리={min_dist_after_lock:.2f}m -> 현재거리={dist_3d:.2f}m, 목표평면 진행초과={along_past:+.2f}m) -> RTL 복귀 시작!',
                flush=True)
            break

        # 거리 오차(dist_3d)에 비례하는 전방 가속도 크기 (출발 시 8.5 m/s^2 -> 근접 시 6.0 m/s^2)
        # 과도한 초기 가속(13.5 m/s^2)으로 인한 속도 폭주(60m/s)를 막고,
        # ~36°~43°의 안정적인 전방 기울기를 유지하며 거리에 비례해 가속도를 조절함.
        a_fwd_mag = min(MAX_FWD_ACC, max(MIN_TILT_ACC, MIN_TILT_ACC + KP_DIST_ACC * dist_3d))
        cur_spd_now = float(np.linalg.norm(vel_ned))
        if cur_spd_now > MAX_APPROACH_SPEED:
            a_fwd_mag = MIN_TILT_ACC

        if det_fresh and dist_3d > 2.5:
            # ── 3단계: 풍선 검출됨 -> 거리 오차 비례 전방 가속 + 3D LOS/PN 유도 ──
            los_candidate = pixel_to_los_ned(u, v, R_bn)

            # 풍선 바로 밑/옆을 스쳐 지나갈 때 LOS 벡터가 진행 방향(path_dir) 대비 40° 이상 꺾이면
            # 공중에서 한바퀴 회전하지 않도록 즉시 직진 통과(PASS_THROUGH)로 넘김!
            if float(np.dot(los_candidate, path_dir)) >= 0.76:
                if not vision_locked:
                    print(f'★ 풍선 검출! (px=({u:.0f}, {v:.0f}), 지름={diam:.1f}px, 거리={dist_3d:.1f}m) -> LOS 비례 가속 추적 전환',
                          flush=True)
                    vision_locked = True

                los_ned = los_candidate
                last_los_ned = los_ned

                # 1) 접근 속도(closing_speed) 및 남은 거리(dist_3d)에 비례해 LOS 중심 정렬
                closing_speed = min(float( MAX_APPROACH_SPEED ), max(float(np.dot(vel_ned, los_ned)), 5.0))
                vel_along_los = float(np.dot(vel_ned, los_ned)) * los_ned
                vel_perp_los = vel_ned - vel_along_los
                align_gain = min(2.6, max(1.5, 2.0 * closing_speed / max(dist_3d, 6.0)))
                a_align = -vel_perp_los * align_gain

                # 2) LOS 각속도 기반 PN(비례항법) 보정 가속도
                a_pn = np.zeros(3)
                if prev_los_ned is not None and prev_los_t is not None:
                    dt_vis = det_t - prev_los_t
                    if 1e-3 < dt_vis < 0.5:
                        los_rate = np.cross(prev_los_ned, los_ned) / dt_vis
                        a_pn = np.cross(los_rate, los_ned) * (PN_GAIN * closing_speed)
                prev_los_ned, prev_los_t = los_ned, det_t

                # 3) 수직 보정 가속도가 전방 가속을 압도해 기체가 뒤집히는 현상 방지
                a_perp = clamp_vec(a_align + a_pn, MAX_PERP_ACC)
                a_cmd = enforce_forward_tilt(los_ned * a_fwd_mag + a_perp, yaw)

                send_acc(conn, a_cmd[0], a_cmd[1], a_cmd[2], yaw)
                mode_str = f'LOS_PURSUIT(px={u-CX:+.0f},{v-CY:+.0f} d={diam:.0f}px a={a_fwd_mag:.1f})'
            else:
                # 근접 통과 중 시선이 급격히 꺾임 -> 회전 방지, 직진 통과!
                coast_dir = last_los_ned if last_los_ned is not None else path_dir
                a_cmd = enforce_forward_tilt(coast_dir * a_fwd_mag, yaw)
                send_acc(conn, a_cmd[0], a_cmd[1], a_cmd[2], yaw)
                mode_str = 'PASS_THROUGH_GUARD'

        elif vision_locked and det_coast and last_los_ned is not None:
            # ── 비전 일시 소실 시 Coast: 마지막 LOS 방향으로 거리 비례 가속 유지 ──
            a_cmd = enforce_forward_tilt(last_los_ned * a_fwd_mag, yaw)
            send_acc(conn, a_cmd[0], a_cmd[1], a_cmd[2], yaw)
            mode_str = 'LOS_COAST'

        elif target_locked or case1_vision_passed:
            # ── 풍선 초근접 통과 구간: 조건 1(디텍팅 유실점) 달성 후 조건 2(목표점 통과)까지 마지막 진행 방향으로 그대로 통과 ──
            coast_dir = last_los_ned if last_los_ned is not None else path_dir
            a_cmd = enforce_forward_tilt(coast_dir * a_fwd_mag, yaw)
            send_acc(conn, a_cmd[0], a_cmd[1], a_cmd[2], yaw)
            mode_str = 'PASS_THROUGH'

        else:
            # ── 2단계: 풍선 미검출 구간 -> 거리 비례 전방 가속(a_fwd_mag)으로 안정적으로 기울여 대각선 상승! ──
            if vision_locked:
                print('비전 신호 소실 -> 대각선 가속 비행으로 복귀', flush=True)
                vision_locked = False
            prev_los_ned = prev_los_t = None

            # 현재 위치에서 목표(target_ned)를 향하는 단위벡터 (근접 시 뒤로 꺾이지 않게 path_dir 보호)
            unit_to_target = diff_to_target / max(dist_3d, 0.1)
            if float(np.dot(unit_to_target, path_dir)) < 0.7:
                unit_to_target = path_dir

            # 5m 이륙 지점(start_ned)과 풍선 좌표(target_ned)를 잇는 3D 대각선 직선으로부터의 이탈 오차(cross-track error)
            prog = float(np.dot(pos_ned - start_ned, path_dir))
            prog = max(0.0, min(path_len, prog))
            closest_pt = start_ned + prog * path_dir
            cross_pos_err = closest_pt - pos_ned

            # 속도 벡터를 대각선 진행 성분과 이탈 성분으로 분해
            speed_along = float(np.dot(vel_ned, unit_to_target))
            vel_along = speed_along * unit_to_target
            vel_perp = vel_ned - vel_along

            # 대각선 경로 이탈 보정(a_cross)이 전방 가속을 상쇄해 기체가 공중에서 뒤로 젖혀지는 현상 원천 차단!
            a_forward = unit_to_target * a_fwd_mag
            a_cross = clamp_vec(cross_pos_err * 1.6 - vel_perp * 1.4, MAX_PERP_ACC)
            a_cmd = enforce_forward_tilt(a_forward + a_cross, yaw)

            send_acc(conn, a_cmd[0], a_cmd[1], a_cmd[2], yaw)
            mode_str = f'DIAGONAL_ACC(a={a_fwd_mag:.1f})'

        now = time.monotonic()
        if now - last_log > 0.4:
            cur_spd = float(np.linalg.norm(vel_ned))
            # 기체 상단(+Z_body = -Z_frd)이 수직(천장)으로부터 앞으로 얼마나 기울어졌는지(Tilt 각도)와 앙각 계산
            nose_ned = R_bn @ np.array([0.0, 0.0, -1.0])
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -nose_ned[2]))))
            nose_elev_deg = 90.0 - tilt_deg
            print(f'  [{mode_str}] 거리 {dist_3d:5.1f}m | 고도 {-pos_ned[2]:4.1f}m | 속도 {cur_spd:4.1f}m/s | '
                  f'기체기울기(Tilt) {tilt_deg:4.1f}° (카메라앙각 {nose_elev_deg:4.1f}°)', flush=True)
            last_log = now

        time.sleep(0.02)

    # ── 풍선 통과 후 감속 제동(Braking) -> 안정화 후 RTL 전환 ──
    # 고속 상태에서 곧바로 RTL을 걸면 200m 이상 밀려나며 PX4 EKF2 위치 리셋이 발생해 공중에서 헤매므로,
    # 먼저 Offboard 상태에서 부드럽게 감속해 속도를 2.5m/s 이하로 낮춘 뒤 RTL을 호출함!
    print('풍선 통과 완료 -> 공중 감속 제동(Braking) 시작 ...', flush=True)
    brake_start = time.monotonic()
    while time.monotonic() - brake_start < 4.5:
        while True:
            msg = conn.recv_match(type=['LOCAL_POSITION_NED'], blocking=False)
            if msg is None:
                break
            pos_ned = np.array([msg.x, msg.y, msg.z])
            vel_ned = np.array([msg.vx, msg.vy, msg.vz])

        cur_spd = float(np.linalg.norm(vel_ned))
        if cur_spd < 2.5:
            print(f'  감속 완료 (현재 속도 {cur_spd:.1f}m/s) -> 안전하게 RTL 전환!', flush=True)
            break

        # 현재 속도의 반대 방향으로 최대 6.0 m/s^2 부드러운 역추진 감속 (급격한 뒤집힘 방지)
        a_brake = clamp_vec(-vel_ned * 1.4, 6.0)
        send_acc(conn, a_brake[0], a_brake[1], a_brake[2], yaw)
        time.sleep(0.02)

    # 0.4초간 현재 위치 호버링 명령으로 자세 수평 안정화
    for _ in range(20):
        send_pos(conn, pos_ned[0], pos_ned[1], pos_ned[2], yaw)
        time.sleep(0.02)

    # ── 풍선 통과 후 RTL (Return to Launch) ──
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



