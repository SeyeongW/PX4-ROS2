#!/usr/bin/env python3
"""온보드 풍선 추적기: 지상 웨이포인트 수신 → ARM/이륙 → 비전 PN 추적 → RTL.

수신:
  COMMAND_INT (지상국 mavlink_bridge_node): 풍선 GPS 위치
  UDP 16B (onboard detector.py): [detected, dx_px, dy_px, radius_px]
동작:
  0) 지상 대기: 첫 유효 웨이포인트 수신 시 ARM → TAKEOFF
  1) 이륙고도 도달 후 OFFBOARD 진입, 웨이포인트로 접근 (position setpoint)
  2) 5m 이내 진입 시 웨이포인트 고정 (lock)
  3) 카메라에 풍선 보이면 PN 가이던스 (acceleration setpoint)
  4) 비전 끊겨도 coast_timeout 동안 마지막 방향 유지
  5) 최소거리에서 3m 이상 멀어지면 통과 판정 → RTL
안전:
  - OFFBOARD 해제 시 다시 뺏지 않고 종료
"""
import argparse
import math
import socket
import struct
import time

import numpy as np
from pymavlink import mavutil

from common import (
    ACC_ONLY, JumpFilter, command, dist3_m, horiz_dist_m, request_streams,
    send_acc, send_setpoint, target_ok, wait_fcu,
    M, PX4_AUTO, PX4_AUTO_LOITER, PX4_OFFBOARD,
)
from guidance import BearingPN, quat_to_rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='udpin:0.0.0.0:14601')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--ground-sys', type=int, default=250)
    ap.add_argument('--udp-port', type=int, default=15760)
    ap.add_argument('--rate', type=float, default=60.0)
    ap.add_argument('--timeout', type=float, default=1.0)
    ap.add_argument('--lost', type=float, default=5.0)
    ap.add_argument('--max-dist', type=float, default=150.0)
    ap.add_argument('--min-alt', type=float, default=5.0)
    ap.add_argument('--max-alt', type=float, default=60.0)
    ap.add_argument('--max-jump', type=float, default=10.0)
    ap.add_argument('--confirm', type=int, default=10)
    ap.add_argument('--alt-offset', type=float, default=0.0)
    ap.add_argument('--lock-dist', type=float, default=5.0)
    ap.add_argument('--overshoot-dist', type=float, default=3.0)
    ap.add_argument('--coast-timeout', type=float, default=1.0,
                    help='비전 끊김 시 마지막 방향으로 유지하는 시간 [s]')
    ap.add_argument('--pn-gain', type=float, default=4.0)
    ap.add_argument('--pn-amax', type=float, default=15.0)
    ap.add_argument('--pn-forward-acc', type=float, default=5.0)
    ap.add_argument('--fx', type=float, default=400.0)
    ap.add_argument('--fy', type=float, default=400.0)
    ap.add_argument('--takeoff-alt', type=float, default=15.0,
                    help='이륙 고도 [m]. 이 고도 도달 후 OFFBOARD 진입')
    ap.add_argument('--log', default='', help='비행 로그 파일 경로 (비어있으면 stdout만)')
    a = ap.parse_args()

    logf = open(a.log, 'w') if a.log else None

    def log(text):
        line = f'[{time.monotonic():.3f}] {text}'
        print(line, flush=True)
        if logf:
            logf.write(line + '\n')
            logf.flush()

    mav = mavutil.mavlink_connection(a.url, baud=a.baud, source_system=1, source_component=191)
    wait_fcu(mav)
    log(f'FCU sys {mav.target_system} comp {mav.target_component}')

    request_streams(mav, [
        (M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 50),
        (M.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 60),
        (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2),
    ])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', a.udp_port))
    sock.setblocking(False)

    jump = JumpFilter(a.max_jump, a.confirm)
    pn = BearingPN(a.pn_gain, a.fx, a.fy, a.pn_amax, a.pn_forward_acc)

    target, t_rx, hold = None, None, None
    locked_target = None
    min_dist_after_lock = math.inf

    pos = home = landed = None
    vel_ned = np.zeros(3)
    R_bn = np.eye(3)
    armed = False
    was_offboard = False

    detected = 0.0
    dx = dy = 0.0
    t_vis = 0.0

    t_start = time.monotonic()
    t_next = t_cmd = t_hb = t_log = 0.0

    while True:
        # --- vision UDP 수신 ---
        while True:
            try:
                data, _ = sock.recvfrom(16)
                det_flag, _dx, _dy, _ = struct.unpack('<4f', data)
                t_vis = time.monotonic()
                detected, dx, dy = det_flag, _dx, _dy
            except BlockingIOError:
                break

        # --- MAVLink 수신 ---
        msg = mav.recv_match(blocking=False)
        now = time.monotonic()
        if msg is not None:
            typ = msg.get_type()
            from_fcu = (msg.get_srcSystem() == mav.target_system
                        and msg.get_srcComponent() == mav.target_component)

            if (typ == 'COMMAND_INT' and msg.command == M.MAV_CMD_USER_1
                    and msg.get_srcSystem() == a.ground_sys
                    and msg.target_component == 191
                    and msg.frame == M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT):
                if locked_target is None:
                    t = (msg.x * 1e-7, msg.y * 1e-7, msg.z + a.alt_offset)
                    if home is None:
                        pass
                    elif not target_ok(t, home, a.max_dist, a.min_alt, a.max_alt):
                        if now - t_log > 2.0:
                            log(f'범위 밖 표적 거부: {t}')
                            t_log = now
                    elif jump(t):
                        target, t_rx = t, now
                    elif now - t_log > 2.0:
                        log(f'점프 표적 보류: {t}')
                        t_log = now

            elif from_fcu and typ == 'GLOBAL_POSITION_INT':
                pos = (msg.lat * 1e-7, msg.lon * 1e-7, msg.relative_alt * 1e-3)
                vel_ned = np.array([msg.vx * 1e-2, msg.vy * 1e-2, msg.vz * 1e-2])

            elif from_fcu and typ == 'ATTITUDE_QUATERNION':
                R_bn = quat_to_rot(msg.q1, msg.q2, msg.q3, msg.q4)

            elif from_fcu and typ == 'HOME_POSITION':
                home = (msg.latitude * 1e-7, msg.longitude * 1e-7)

            elif from_fcu and typ == 'EXTENDED_SYS_STATE':
                landed = msg.landed_state

            elif from_fcu and typ == 'HEARTBEAT':
                armed = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
                in_offboard = (msg.custom_mode >> 16) & 0xFF == PX4_OFFBOARD
                if was_offboard and not in_offboard:
                    log('OFFBOARD 해제됨, 종료')
                    return
                if in_offboard and not was_offboard:
                    log('OFFBOARD 진입')
                was_offboard = was_offboard or in_offboard

        # --- heartbeat / 초기 대기 ---
        if now - t_hb > 1.0:
            t_hb = now
            mav.mav.heartbeat_send(M.MAV_TYPE_ONBOARD_CONTROLLER, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            if home is None:
                command(mav, M.MAV_CMD_REQUEST_MESSAGE, M.MAVLINK_MSG_ID_HOME_POSITION)
            if now - t_start > 3.0 and None in (pos, home, landed):
                log(f'대기: GLOBAL_POSITION_INT={pos is not None}, '
                    f'HOME_POSITION={home is not None}, '
                    f'EXTENDED_SYS_STATE={landed is not None}')

        if now < t_next or pos is None or home is None:
            time.sleep(0.002)
            continue
        t_next = now + 1.0 / a.rate

        # --- lock / overshoot 판정 ---
        active_target = locked_target or target
        if active_target is not None:
            d_curr = dist3_m(pos, active_target)
            if locked_target is None and d_curr <= a.lock_dist:
                locked_target = target
                min_dist_after_lock = d_curr
                log(f'{a.lock_dist}m 이내 (dist={d_curr:.2f}m): 표적 고정 {locked_target}')
            elif locked_target is not None:
                if d_curr < min_dist_after_lock:
                    min_dist_after_lock = d_curr
                elif d_curr - min_dist_after_lock >= a.overshoot_dist:
                    log(f'통과 판정: 최소 {min_dist_after_lock:.2f}m → 현재 {d_curr:.2f}m. RTL.')
                    command(mav, M.MAV_CMD_NAV_RETURN_TO_LAUNCH)
                    return

        # --- 가이던스 ---
        vis_fresh = (now - t_vis) < 0.2
        vis_coast = (now - t_vis) < a.coast_timeout

        if detected > 0.5 and vis_fresh and was_offboard:
            # 비전 PN 가이던스
            a_cmd = pn.update(dx, dy, R_bn, vel_ned, t_vis)
            if a_cmd is not None:
                send_acc(mav, a_cmd[0], a_cmd[1], a_cmd[2])
                continue
        elif vis_coast and pn.has_lock and was_offboard:
            # 비전 끊김 → coast: 마지막 방향으로 전진
            a_cmd = pn.coast()
            if a_cmd is not None:
                send_acc(mav, a_cmd[0], a_cmd[1], a_cmd[2])
                continue
        else:
            pn.reset()

        # --- 웨이포인트 가이던스 (lock 후에는 timeout 무시) ---
        age = 0.0 if locked_target is not None else (math.inf if t_rx is None else now - t_rx)
        fresh = age < a.timeout
        if fresh:
            sp, hold = active_target, None
        elif active_target is None:
            sp = pos
        else:
            hold = hold or pos
            sp = hold
        send_setpoint(mav, *sp)

        if now - t_cmd < 1.0:
            continue

        if was_offboard and age > a.lost:
            command(mav, M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    PX4_AUTO, PX4_AUTO_LOITER)
            t_cmd = now
            log(f'표적 {age:.1f}s 끊김, Hold 요청')
        elif fresh and not armed and landed == M.MAV_LANDED_STATE_ON_GROUND:
            # 첫 유효 웨이포인트 수신 + 지상 → ARM
            command(mav, M.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            t_cmd = now
            log('ARM 요청')
        elif fresh and armed and landed == M.MAV_LANDED_STATE_ON_GROUND:
            # ARM 완료, 아직 지상 → TAKEOFF
            command(mav, M.MAV_CMD_NAV_TAKEOFF,
                    0, 0, 0, float('nan'), float('nan'), float('nan'), a.takeoff_alt)
            t_cmd = now
            log(f'TAKEOFF 요청 → {a.takeoff_alt:.0f}m')
        elif (fresh and not was_offboard
              and landed == M.MAV_LANDED_STATE_IN_AIR
              and pos[2] >= a.takeoff_alt - 1.0):
            # 이륙고도 도달 → OFFBOARD
            command(mav, M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_OFFBOARD)
            t_cmd = now
            log(f'OFFBOARD 요청, 표적 {sp}')


if __name__ == '__main__':
    main()
