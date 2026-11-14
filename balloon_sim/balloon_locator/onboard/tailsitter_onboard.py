#!/usr/bin/env python3
"""실기체 쿼드 테일시터(Quad-Tailsitter) 온보드 폐루프 유도 제어기 (ROS 의존성 없음).

지상 관측 시스템(SIYI A8 mini + 6-DOF UKF + mavlink_bridge_node)으로부터 SiK 텔레메트리 모뎀을 통해
수신한 유효 풍선 좌표(COMMAND_INT, MAV_CMD_USER_1)로 접근합니다.
온보드 검출 UDP 16B에는 원영상 노출 시각·보정 식별자가 없어 최신 자세와 결합할 수 없습니다.
검출 수신은 유지하지만 픽셀 기반 종말 조향은 비활성화합니다. 원노출 시각, 해당 시각의
기체 자세 및 실측 카메라 외부·내부 보정이 갖춰지기 전에는 영상으로 좌표 만료를 우회하지 않습니다.

동작 단계 (State Machine):
  0) WAIT_TARGET     : 지상 대기 — 지상국에서 발진 승인된 유효 6-DOF UKF 표적(COMMAND_INT) 수신 대기
  1) TAKEOFF         : 수직 호버링 자세(Pitch=0°) 스트리밍 -> OFFBOARD 진입 & ARM -> 목표 고도 수직 상승
  2) TRANSITION      : 표적 방위각으로 기수 숙임(Pitch -25° -> -55°) 전진 가속 천이 비행 (2.2초)
  3) TS_UKF_APPROACH : 지상 10Hz UKF 좌표/속도 실시간 수신 + 50Hz SO(3) 쿼터니언/추력 폐루프 중기 유도
  4) TS_BRAKE_TO_HOVER: 좌표 기반 근접 및 전방 평면 통과 시 역천이 제동(2.5초) -> AUTO.RTL 인계

안전 장치:
  - 기본 실행(플래그 없음)은 오프라인 파라미터 검증만 수행하며, 명시적 --execute 인자에서만 하드웨어 I/O 시작
  - 조종사가 RC 스위치로 OFFBOARD 모드를 해제하면 즉시 제어권을 반환하고 안전 종료
  - 홈 기준 반경(max_dist) 및 고도(min_alt ~ max_alt) 지오펜스 + JumpFilter 이상치 차단
"""
import argparse
from contextlib import ExitStack
import math
import socket
import struct
import time

import numpy as np
from pymavlink import mavutil

from common import (
    JumpFilter, CommandWatchdog, add_execution_options, validate_flight_options,
    target_message_ok, home_position, record_telemetry, telemetry_fresh,
    command, dist3_m, horiz_dist_m, request_streams, target_ok, wait_fcu,
    M, PX4_AUTO, PX4_AUTO_LOITER, PX4_OFFBOARD, EARTH_R,
)
from guidance import quat_to_rot


# ---------------------------------------------------------------------------
# SO(3) 쿼드 테일시터 자세/추력 및 3D LOS/PN 수학 함수
# ---------------------------------------------------------------------------
def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 회전 행렬(R_ned_from_frd) -> 단위 쿼터니언 [w, x, y, z] (Shepperd 알고리즘)."""
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def nose_dir_to_quat_frd(nose_ned: np.ndarray, yaw_fallback: float = 0.0) -> np.ndarray:
    """쿼드 테일시터 코(-Z_b)를 목표 방향(nose_ned)으로 정렬하는 Wings-Level(Roll=0) 쿼터니언 생성."""
    norm = float(np.linalg.norm(nose_ned))
    if norm < 1e-6:
        nose_ned = np.array([0.0, 0.0, -1.0])
    else:
        nose_ned = nose_ned / norm

    z_b = -nose_ned
    horiz = math.hypot(float(nose_ned[0]), float(nose_ned[1]))
    if horiz > 1e-3:
        yaw = math.atan2(float(nose_ned[1]), float(nose_ned[0]))
    else:
        yaw = yaw_fallback

    # Same wings-level construction as tools/tailsitter_guidance.py: crossing
    # the horizontal plane must not reverse the right-wing axis by 180 degrees.
    y_b = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    x_b = np.cross(y_b, z_b)
    x_b = x_b / np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)
    y_b = y_b / np.linalg.norm(y_b)
    R = np.column_stack((x_b, y_b, z_b))
    return rotmat_to_quat(R)


def pixel_to_los_ned(dx_px: float, dy_px: float, fx: float, fy: float, R_ned_from_frd: np.ndarray) -> np.ndarray:
    """온보드 카메라 픽셀 오프셋(dx_px: 우측+, dy_px: 상방+) -> NED 3차원 시선(LOS) 단위벡터.

    쿼드 테일시터 노즈콘 카메라:
      - 광축(전방) = 기체 -Z_b ([0, 0, -1])
      - 이미지 우측(+dx) = 기체 우측 날개 +Y_b ([0, 1, 0])
      - 이미지 상단(+dy) = 기체 등쪽 -X_b ([-1, 0, 0])
    """
    rx_frd = -dy_px / fy
    ry_frd = dx_px / fx
    rz_frd = -1.0
    v_frd = np.array([rx_frd, ry_frd, rz_frd], dtype=float)
    v_frd /= np.linalg.norm(v_frd)
    v_ned = R_ned_from_frd @ v_frd
    return v_ned / np.linalg.norm(v_ned)


def compute_tailsitter_cmd(
    pos_ned: np.ndarray,
    vel_ned: np.ndarray,
    target_ned: np.ndarray,
    desired_speed: float,
    hover_thrust: float,
    los_ray_ned: np.ndarray = None,
    yaw_fallback: float = 0.0,
    max_tilt_deg: float = 65.0,
    target_vel_ned: np.ndarray = None,
    allow_dive: bool = False,
):
    """쿼드 테일시터 자세(쿼터니언) 및 추력(Thrust) 분리 제어 + 주익 공력 양력 보상."""
    dp = target_ned - pos_ned
    dist = float(np.linalg.norm(dp))
    u_to_target = dp / max(dist, 1e-3)
    t_vel = target_vel_ned if target_vel_ned is not None else np.zeros(3)

    if los_ray_ned is not None:
        u_dir = los_ray_ned / max(float(np.linalg.norm(los_ray_ned)), 1e-6)
    else:
        u_dir = u_to_target

    v_des = u_dir * desired_speed + t_vel
    ev = v_des - vel_ned

    # 3차원 비례항법 (3D Proportional Navigation)
    v_rel = t_vel - vel_ned
    r_vec = dp if los_ray_ned is None else u_dir * max(dist, 3.0)
    r_norm_sq = max(float(np.dot(r_vec, r_vec)), 4.0)
    omega_los = np.cross(r_vec, v_rel) / r_norm_sq
    v_closing = max(float(-np.dot(v_rel, u_dir)), 4.0)
    a_pn = 4.0 * v_closing * np.cross(omega_los, u_dir)

    kp_v_horiz = 2.0
    kp_v_vert = 2.8
    a_N = float(np.clip(kp_v_horiz * ev[0] + a_pn[0], -12.0, 12.0))
    a_E = float(np.clip(kp_v_horiz * ev[1] + a_pn[1], -12.0, 12.0))

    pos_z_err = float(dp[2])
    a_D_pd = 1.6 * pos_z_err + kp_v_vert * float(ev[2]) + float(a_pn[2])
    a_D = float(np.clip(a_D_pd, -8.0, 7.0))

    # 전진 비행 속도에 따른 주익 공기역학적 양력 보상
    airspeed = float(np.linalg.norm(vel_ned))
    lift_relief = float(np.clip(1.0 - 0.042 * max(0.0, airspeed - 4.0), 0.35, 1.0))
    g_eff = 9.81 * lift_relief
    f_dir = np.array([a_N, a_E, a_D - g_eff], dtype=float)
    f_dir_norm = float(np.linalg.norm(f_dir))
    raw_nose = f_dir / max(f_dir_norm, 1e-3)

    if los_ray_ned is not None:
        blend_ray = np.array([
            u_dir[0],
            u_dir[1],
            u_dir[2] - 0.16 * lift_relief,
        ], dtype=float)
        blend_ray /= max(float(np.linalg.norm(blend_ray)), 1e-6)
        raw_nose = 0.65 * blend_ray + 0.35 * raw_nose
        raw_nose /= max(float(np.linalg.norm(raw_nose)), 1e-6)

    min_up = math.cos(math.radians(max_tilt_deg))
    if not allow_dive:
        if -raw_nose[2] < min_up:
            h = math.hypot(float(raw_nose[0]), float(raw_nose[1]))
            max_h = math.sin(math.radians(max_tilt_deg))
            if h > 1e-6:
                raw_nose[0] = raw_nose[0] / h * max_h
                raw_nose[1] = raw_nose[1] / h * max_h
            raw_nose[2] = -min_up
            raw_nose /= np.linalg.norm(raw_nose)
        if raw_nose[2] > -0.05:
            raw_nose[2] = -0.05
            raw_nose /= np.linalg.norm(raw_nose)
    else:
        if raw_nose[2] > 0.45:
            raw_nose[2] = 0.45
            raw_nose /= np.linalg.norm(raw_nose)

    q_des = nose_dir_to_quat_frd(raw_nose, yaw_fallback=yaw_fallback)

    speed_curr = float(np.dot(vel_ned, u_dir))
    speed_err = desired_speed - speed_curr
    alt_climb_need = max(0.0, -pos_z_err)
    thrust_base = hover_thrust * 0.80
    thrust = thrust_base + 0.030 * speed_err + 0.015 * alt_climb_need - 0.018 * a_D
    thrust = float(np.clip(thrust, 0.38, 0.90))
    return q_des, thrust, raw_nose


def send_attitude_target(mav, q_wxyz, thrust: float):
    """MAVLink SET_ATTITUDE_TARGET (type_mask=0b00000111: body rate 무시, 자세 쿼터니언 + 추력 제어)."""
    mav.mav.set_attitude_target_send(
        0,
        mav.target_system,
        mav.target_component,
        0b00000111,
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,
        float(np.clip(thrust, 0.0, 1.0)),
    )


def confirm_auto_mode(mav, submode, log):
    """Stop setpoints and bound the wait for FCU ACK *and* observed AUTO mode.

    A confirmed handoff is not proof of a completed return or landing. If the
    link fails, no stale attitude setpoints keep PX4's Offboard stream alive.
    """
    commands = CommandWatchdog()
    commands.request(mav, time.monotonic(), M.MAV_CMD_DO_SET_MODE,
                     M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_AUTO, submode)
    while True:
        msg = mav.recv_match(blocking=False)
        if (msg is not None and msg.get_srcSystem() == mav.target_system
                and msg.get_srcComponent() == mav.target_component):
            if msg.get_type() == 'COMMAND_ACK':
                commands.receive(msg)
            elif msg.get_type() == 'HEARTBEAT':
                commands.observe(msg)
        failure = commands.failure(time.monotonic())
        if failure:
            log(f'AUTO handoff unconfirmed: {failure}')
            return False
        if commands.pending is None and commands.waiting_state is None:
            log(f'AUTO handoff confirmed: submode={submode}; return/landing not verified')
            return True
        time.sleep(.01)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_execution_options(ap)
    ap.add_argument('--url', default='udpin:0.0.0.0:14601',
                    help='FCU MAVLink 연결 경로 (예: /dev/ttyACM0, /dev/ttyTHS1, udpin:0.0.0.0:14601)')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--ground-sys', type=int, default=250)
    ap.add_argument('--udp-port', type=int, default=15760,
                    help='온보드 detector.py로부터 16B 패킷을 수신할 로컬 UDP 포트')
    ap.add_argument('--rate', type=float, default=50.0,
                    help='자세/추력 제어 루프 주파수 [Hz] (권장 50Hz)')
    ap.add_argument('--timeout', type=float, default=1.5,
                    help='지상국 표적 패킷 유효 시간 [s]')
    ap.add_argument('--lost', type=float, default=5.0,
                    help='지상국 표적 장기 두절 시 Hold 전환 시간 [s]')
    ap.add_argument('--max-dist', type=float, default=200.0)
    ap.add_argument('--min-alt', type=float, default=5.0)
    ap.add_argument('--max-alt', type=float, default=80.0)
    ap.add_argument('--max-jump', type=float, default=12.0)
    ap.add_argument('--confirm', type=int, default=5)
    ap.add_argument('--alt-offset', type=float, default=0.0)
    ap.add_argument('--takeoff-alt', type=float, default=20.0,
                    help='수직 이륙 목표 상대고도 [m]')
    ap.add_argument('--cruise-speed', type=float, default=14.0,
                    help='테일시터 전진 추적 목표 속도 [m/s]')
    ap.add_argument('--terminal-dist', type=float, default=40.0,
                    help='온보드 카메라 종말 비전 유도 전환 거리 [m]')
    ap.add_argument('--pass-radius', type=float, default=2.2,
                    help='풍선 통과 판정 Condition 1 진입 반경 [m]')
    ap.add_argument('--hover-thrust', type=float, default=0.68,
                    help='수직 호버링 기준 추력 [0.3 ~ 0.9]')
    ap.add_argument('--balloon-diameter', type=float, default=0.30,
                    help='실제 풍선 직경 [m] (근접 광학 거리 R_opt 계산용)')
    ap.add_argument('--fx', type=float, default=640.0,
                    help='온보드 노즈콘 카메라 수평 초점거리 [px]')
    ap.add_argument('--fy', type=float, default=640.0,
                    help='온보드 노즈콘 카메라 수직 초점거리 [px]')
    ap.add_argument('--log', default='', help='비행 로그 텍스트 파일 저장 경로')
    a = ap.parse_args(argv)
    try:
        validate_flight_options(a)
        positive = ('takeoff_alt', 'cruise_speed', 'terminal_dist', 'pass_radius',
                    'hover_thrust', 'balloon_diameter', 'fx', 'fy')
        if (any(not math.isfinite(getattr(a, k)) or getattr(a, k) <= 0 for k in positive)
                or not 1 <= a.udp_port <= 65535
                or not a.min_alt <= a.takeoff_alt <= a.max_alt
                or not 0.25 <= a.hover_thrust <= 0.95):
            raise ValueError('invalid_tailsitter_onboard_configuration')
    except ValueError as error:
        ap.error(str(error))
    return a


def main(argv=None):
    a = parse_args(argv)
    if not a.execute:
        print('오프라인 설정검사 통과. 쿼드 테일시터 온보드 제어기 I/O는 실행하지 않았습니다. 실제 실행은 --execute가 필요합니다.')
        return 0
    with ExitStack() as resources:
        return run(a, resources)


def run(a, resources):
    logf = resources.enter_context(open(a.log, 'w')) if a.log else None

    def log(text):
        line = f'[{time.monotonic():.3f}] {text}'
        print(line, flush=True)
        if logf:
            logf.write(line + '\n')
            logf.flush()

    mav = mavutil.mavlink_connection(a.url, baud=a.baud, source_system=a.system_id, source_component=191)
    resources.callback(mav.close)
    wait_fcu(mav, timeout=a.connect_timeout, expected_system=a.system_id)
    log(f'FCU 연결 완료: sys={mav.target_system} comp={mav.target_component}')
    log('onboard_pixel_guidance_disabled: exposure_timestamp_and_calibrated_pose_history_required')

    request_streams(mav, [
        (M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 50),
        (M.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 50),
        (M.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 60),
        (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 5),
    ])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    resources.callback(sock.close)
    sock.bind(('127.0.0.1', a.udp_port))
    sock.setblocking(False)

    jump = JumpFilter(a.max_jump, a.confirm)
    commands = CommandWatchdog()
    received, boot_times = {}, {}

    # FCU 상태 변수
    pos_global = None         # (lat_deg, lon_deg, rel_alt_m)
    pos_ned = np.zeros(3)     # [N, E, D] in local NED frame
    vel_ned = np.zeros(3)     # [vN, vE, vD] in m/s
    has_local_ned = False
    R_ned_from_frd = np.eye(3)
    home = None               # (home_lat_deg, home_lon_deg)
    home_alt_amsl = None
    landed = None
    armed = False
    was_offboard = False
    offboard_requested = False
    offboard_stream_started = None
    takeoff_origin = None

    # 지상국 UKF 표적 상태
    target_global = None      # (lat_deg, lon_deg, rel_alt_m)
    target_ned = None         # np.array([N, E, D]) in drone's local NED frame
    target_vel_ned = np.zeros(3)
    t_rx = None
    ukf_rx_count = 0

    # 온보드 노즈콘 카메라 검출 상태 (detector.py UDP)
    detected = 0.0
    dx_px = dy_px = radius_px = 0.0
    t_vis = 0.0
    vis_streak = 0
    max_diam_seen_px = 0.0

    # 테일시터 비행 단계 상태기계
    stage = 'WAIT_TARGET'
    stage_t0 = time.monotonic()
    last_yaw = 0.0
    entered_hit_zone = False
    min_dist_to_target = math.inf
    approach_unit_vec = np.array([1.0, 0.0, 0.0])

    t_start = time.monotonic()
    t_next = t_cmd = t_hb = t_log = 0.0

    while True:
        # 1. 온보드 카메라 UDP 패킷 수신 (16B: detected, dx_px, dy_px, radius_px)
        for _ in range(64):
            try:
                data, peer = sock.recvfrom(64)
                if peer[0] != '127.0.0.1' or len(data) != 16:
                    continue
                det_flag, _dx, _dy, _rad = struct.unpack('<4f', data)
                if (not all(math.isfinite(v) for v in (det_flag, _dx, _dy, _rad))
                        or det_flag not in (0., 1.) or _rad < 0 or (det_flag == 1. and _rad <= 0)):
                    continue
                next_vis = time.monotonic()
                if next_vis - t_vis > .25:
                    vis_streak = 0
                t_vis = next_vis
                detected, dx_px, dy_px, radius_px = det_flag, _dx, _dy, _rad
                if det_flag == 1.0:
                    vis_streak += 1
                    diam_px = 2.0 * _rad
                    if diam_px > max_diam_seen_px:
                        max_diam_seen_px = diam_px
                else:
                    vis_streak = 0
            except BlockingIOError:
                break

        # 2. FCU 및 지상국 MAVLink 메시지 수신
        msg = mav.recv_match(blocking=False)
        now = time.monotonic()
        if msg is not None:
            typ = msg.get_type()
            from_fcu = (msg.get_srcSystem() == mav.target_system
                        and msg.get_srcComponent() == mav.target_component)

            if typ == 'COMMAND_ACK' and from_fcu:
                commands.receive(msg)
            if from_fcu and typ != 'COMMAND_ACK':
                from_fcu = record_telemetry(msg, received, boot_times, now)
                if from_fcu:
                    commands.observe(msg)

            if typ == 'COMMAND_INT' and target_message_ok(msg, a):
                t = (msg.x * 1e-7, msg.y * 1e-7, msg.z + a.alt_offset)
                if (home is None or pos_global is None or not has_local_ned
                        or not telemetry_fresh(received, now, a.telemetry_timeout,
                                               attitude=True, local_position=True)):
                    pass
                elif not target_ok(t, home, a.max_dist, a.min_alt, a.max_alt):
                    if now - t_log > 2.0:
                        log(f'범위 밖 표적 거부: {t}')
                        t_log = now
                elif jump(t):
                    target_global, t_rx = t, now
                    ukf_rx_count += 1
                    # param3, param4에 담긴 지상 UKF 속도 추정치 (vn, ve) 추출
                    vn = float(getattr(msg, 'param3', 0.0))
                    ve = float(getattr(msg, 'param4', 0.0))
                    if math.isfinite(vn) and math.isfinite(ve):
                        target_vel_ned = np.array([vn, ve, 0.0], dtype=float)
                    # 드론 현재 위치 기준 차분 GPS -> 로컬 NED 좌표 변환 (EKF 원점 오프셋 무관)
                    dN = math.radians(t[0] - pos_global[0]) * EARTH_R
                    dE = math.radians(t[1] - pos_global[1]) * EARTH_R * math.cos(math.radians(pos_global[0]))
                    dD = -(t[2] - pos_global[2])
                    new_ned = np.array([pos_ned[0] + dN, pos_ned[1] + dE, pos_ned[2] + dD], dtype=float)
                    if target_ned is None:
                        target_ned = new_ned
                        log(f'첫 UKF 표적 수신 완료 (#{ukf_rx_count}): local_ned=({new_ned[0]:.1f}, {new_ned[1]:.1f}, {new_ned[2]:.1f})m')
                    else:
                        target_ned = 0.35 * new_ned + 0.65 * target_ned

            elif from_fcu and typ == 'GLOBAL_POSITION_INT':
                pos_global = (msg.lat * 1e-7, msg.lon * 1e-7, msg.relative_alt * 1e-3)

            elif from_fcu and typ == 'LOCAL_POSITION_NED':
                if all(math.isfinite(v) for v in (msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz)):
                    pos_ned = np.array([float(msg.x), float(msg.y), float(msg.z)], dtype=float)
                    vel_ned = np.array([float(msg.vx), float(msg.vy), float(msg.vz)], dtype=float)
                    has_local_ned = True

            elif from_fcu and typ == 'ATTITUDE_QUATERNION':
                R_ned_from_frd = quat_to_rot(msg.q1, msg.q2, msg.q3, msg.q4)

            elif from_fcu and typ == 'HOME_POSITION':
                next_home, next_alt = home_position(msg)
                if commands.started and home is not None and (next_home != home or next_alt != home_alt_amsl):
                    log('home_reference_changed_after_control_start_restart_required')
                    if was_offboard or offboard_requested or armed or landed == M.MAV_LANDED_STATE_IN_AIR:
                        confirm_auto_mode(mav, PX4_AUTO_LOITER, log)
                    return 1
                home, home_alt_amsl = next_home, next_alt

            elif from_fcu and typ == 'EXTENDED_SYS_STATE':
                landed = msg.landed_state

            elif from_fcu and typ == 'HEARTBEAT':
                armed = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
                in_offboard = (msg.custom_mode >> 16) & 0xFF == PX4_OFFBOARD
                if in_offboard and not offboard_requested:
                    log('다른 제어기의 OFFBOARD 상태: 제어권을 획득하지 않고 종료')
                    return 1
                if was_offboard and not in_offboard:
                    log('조종사 OFFBOARD 해제 감지 -> 제어권 즉시 반환 및 종료')
                    return 0
                if in_offboard and not was_offboard:
                    log('OFFBOARD 모드 진입 확인!')
                was_offboard = was_offboard or in_offboard

        # 3. 1Hz 하트비트 및 초기 텔레메트리 확인
        if now - t_hb > 1.0:
            t_hb = now
            mav.mav.heartbeat_send(M.MAV_TYPE_ONBOARD_CONTROLLER, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            if home is None or home_alt_amsl is None:
                command(mav, M.MAV_CMD_REQUEST_MESSAGE, M.MAVLINK_MSG_ID_HOME_POSITION)
            if now - t_start > 3.0 and None in (pos_global, home, landed):
                log(f'FCU 텔레메트리 대기 중: GLOBAL_POS={pos_global is not None}, '
                    f'HOME={home is not None}, EXTENDED_SYS={landed is not None}')

        if now < t_next or pos_global is None or home is None or home_alt_amsl is None:
            if now - t_start > a.connect_timeout and (pos_global is None or home is None or home_alt_amsl is None):
                log('필수 위치/홈 AMSL telemetry 수신 시간 초과')
                return 1
            time.sleep(0.002)
            continue
        t_next = now + 1.0 / a.rate

        failure = commands.failure(now)
        feedback_ok = telemetry_fresh(received, now, a.telemetry_timeout,
                                      attitude=True, local_position=True)
        if failure or not feedback_ok:
            if failure or stage != 'WAIT_TARGET' or was_offboard or commands.started or now - t_start > a.connect_timeout:
                log(failure or 'FCU telemetry stale: 제어 종료')
                if was_offboard or offboard_requested or armed or landed == M.MAV_LANDED_STATE_IN_AIR:
                    confirm_auto_mode(mav, PX4_AUTO_LOITER, log)
                return 1
            continue

        age = math.inf if t_rx is None else now - t_rx
        fresh = 0. <= age < a.timeout
        if not 0. <= now - t_vis <= .25 or detected != 1.0:
            vis_streak = 0
        guidance_stale = (stage in ('TAKEOFF', 'TRANSITION', 'TS_UKF_APPROACH') and not fresh)
        if guidance_stale:
            log(f'guidance_stale:{stage}: target_age={age:.3f}s; 제어 종료')
            if was_offboard or offboard_requested or armed or landed == M.MAV_LANDED_STATE_IN_AIR:
                confirm_auto_mode(mav, PX4_AUTO_LOITER, log)
            return 1

        # 4. 쿼드 테일시터 상태기계 실행 (50 Hz)
        if stage == 'WAIT_TARGET':
            # 지상 대기 중에도 수직 호버링 쿼터니언을 스트리밍하여 즉시 OFFBOARD 전환 가능하도록 준비
            q_up = nose_dir_to_quat_frd(np.array([0.0, 0.0, -1.0]), yaw_fallback=last_yaw)
            send_attitude_target(mav, q_up, 0.15 if not armed else a.hover_thrust)
            if offboard_stream_started is None:
                offboard_stream_started = now

            if fresh and target_ned is not None:
                takeoff_origin = pos_ned.copy()
                dp = target_ned - pos_ned
                last_yaw = math.atan2(float(dp[1]), float(dp[0]))
                stage = 'TAKEOFF'
                stage_t0 = now
                log(f'[STAGE -> TAKEOFF] 발진 승인 표적 수신 완료! 수직 이륙 개시 (목표 고도={a.takeoff_alt:.1f}m)')
            continue

        if stage == 'TAKEOFF':
            alt_curr = float(takeoff_origin[2] - pos_ned[2])
            alt_err = a.takeoff_alt - alt_curr
            # 수직 상승: 코 방향은 수직 하늘([0, 0, -1]) + 수평 드리프트 억제
            kx = float(np.clip(-0.08 * (pos_ned[0] - takeoff_origin[0]) - 0.12 * vel_ned[0], -0.15, 0.15))
            ky = float(np.clip(-0.08 * (pos_ned[1] - takeoff_origin[1]) - 0.12 * vel_ned[1], -0.15, 0.15))
            nose_up = np.array([kx, ky, -1.0], dtype=float)
            q_cmd = nose_dir_to_quat_frd(nose_up, yaw_fallback=last_yaw)
            vz_des = float(np.clip(0.8 * alt_err, -1.5, 4.0))
            vz_curr = -float(vel_ned[2])
            thrust_cmd = float(np.clip(a.hover_thrust + 0.06 * (vz_des - vz_curr), 0.45, 0.88))
            send_attitude_target(mav, q_cmd, thrust_cmd)

            if now - t_cmd >= 1.0:
                if not was_offboard:
                    if (now - offboard_stream_started >= 1.
                            and commands.request(mav, now, M.MAV_CMD_DO_SET_MODE,
                                                 M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_OFFBOARD)):
                        offboard_requested = True
                        t_cmd = now
                        log('OFFBOARD 모드 전환 요청')
                elif not armed:
                    if commands.request(mav, now, M.MAV_CMD_COMPONENT_ARM_DISARM, 1):
                        t_cmd = now
                        log('ARM 요청 (쿼드 테일시터 수직 이륙)')

            if armed and alt_curr >= a.takeoff_alt - 1.5:
                stage = 'TRANSITION'
                stage_t0 = now
                dp = target_ned - pos_ned
                last_yaw = math.atan2(float(dp[1]), float(dp[0]))
                log(f'[STAGE -> TRANSITION] 고도 {alt_curr:.1f}m 도달 -> 전진 피치다운 천이 개시!')
            continue

        if stage == 'TRANSITION':
            dt_trans = now - stage_t0
            dp = target_ned - pos_ned
            last_yaw = math.atan2(float(dp[1]), float(dp[0]))
            # 2.2초 동안 Pitch -25° -> -55°로 부드럽게 숙이며 전진 속도 확보
            frac = min(1.0, dt_trans / 2.2)
            tilt_rad = math.radians(25.0 + 30.0 * frac)
            nose_trans = np.array([
                math.sin(tilt_rad) * math.cos(last_yaw),
                math.sin(tilt_rad) * math.sin(last_yaw),
                -math.cos(tilt_rad),
            ], dtype=float)
            q_cmd = nose_dir_to_quat_frd(nose_trans, yaw_fallback=last_yaw)
            send_attitude_target(mav, q_cmd, min(0.85, a.hover_thrust + 0.08))

            if dt_trans >= 2.2:
                stage = 'TS_UKF_APPROACH'
                stage_t0 = now
                u_app = dp / max(float(np.linalg.norm(dp)), 1e-3)
                approach_unit_vec = u_app
                log(f'[STAGE -> TS_UKF_APPROACH] 중기 10Hz UKF 폐루프 유도 개시 (거리={float(np.linalg.norm(dp)):.1f}m)')
            continue

        if stage == 'TS_UKF_APPROACH':
            dp = target_ned - pos_ned
            dist = float(np.linalg.norm(dp))
            if dist < min_dist_to_target:
                min_dist_to_target = dist
            if dist <= a.pass_radius:
                entered_hit_zone = True

            # 동시 만족 통과 판정 (Condition 1 AND Condition 2)
            rel_from_target = pos_ned - target_ned
            along_track = float(np.dot(rel_from_target, approach_unit_vec))
            cond1_hit = entered_hit_zone and (min_dist_to_target <= a.pass_radius)
            cond2_passed = along_track > 0.3
            if cond1_hit and cond2_passed:
                stage = 'TS_BRAKE_TO_HOVER'
                stage_t0 = now
                log(f'[PASS-THROUGH COMPLETE] 풍선 통과 확인! (CPA={min_dist_to_target:.2f}m, max_diam={max_diam_seen_px:.1f}px) -> 역천이 제동 개시')
                continue

            if math.hypot(float(dp[0]), float(dp[1])) > 1.5:
                last_yaw = math.atan2(float(dp[1]), float(dp[0]))

            q_cmd, thrust_cmd, _ = compute_tailsitter_cmd(
                pos_ned=pos_ned,
                vel_ned=vel_ned,
                target_ned=target_ned,
                desired_speed=a.cruise_speed,
                hover_thrust=a.hover_thrust,
                los_ray_ned=None,
                yaw_fallback=last_yaw,
                target_vel_ned=target_vel_ned,
                allow_dive=False,
            )
            send_attitude_target(mav, q_cmd, thrust_cmd)
            continue

        if stage == 'TS_BRAKE_TO_HOVER':
            dt_brake = now - stage_t0
            # 역천이 제동: 기수 들기(Pitch +18°)로 수평 속도 감속 후 수직 자세 복귀
            pitch_up_rad = math.radians(18.0) if dt_brake < 1.5 else 0.0
            nose_brake = np.array([
                -math.sin(pitch_up_rad) * math.cos(last_yaw),
                -math.sin(pitch_up_rad) * math.sin(last_yaw),
                -math.cos(pitch_up_rad),
            ], dtype=float)
            q_brake = nose_dir_to_quat_frd(nose_brake, yaw_fallback=last_yaw)
            send_attitude_target(mav, q_brake, a.hover_thrust * 0.85)

            if dt_brake >= 2.5:
                log('[STAGE -> RTL] 역천이 감속 완료 -> AUTO.RTL 제어권 인계 요청')
                return 0 if confirm_auto_mode(mav, 5, log) else 1


if __name__ == '__main__':
    raise SystemExit(main())
