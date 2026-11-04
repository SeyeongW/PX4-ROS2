#!/usr/bin/env python3
"""온보드 Pi: 지상국 풍선 위치를 받아 PX4 OFFBOARD로 비행. ROS 없이 pymavlink만 사용.

수신 (balloon_locator/mavlink_bridge_node.py 가 송신):
  COMMAND_INT  command=MAV_CMD_USER_1, frame=GLOBAL_RELATIVE_ALT_INT,
               x=lat*1e7, y=lon*1e7, z=드론 홈 지면 기준 상대고도[m], param1=n, param2=sigma_m
안전:
  - 공중(EXTENDED_SYS_STATE landed_state=IN_AIR)일 때만 OFFBOARD 요청. 이 스크립트는 arm/이륙하지 않는다.
  - 홈 기준 수평거리/고도 범위 밖 표적은 거부.
  - 마지막 채택 표적에서 max_jump 이상 튄 표적은 같은 자리로 confirm회 연속 들어와야 채택.
  - 표적이 timeout 동안 끊기면 그 자리에서 정지, lost 동안 끊기면 Hold(AUTO.LOITER)로 전환.
  - OFFBOARD에서 빠져나가면(조종사 전환 또는 Hold 전환) 다시 뺏지 않고 종료.
"""
import argparse
import math
import time

from pymavlink import mavutil

from common import (
    JumpFilter, command, request_streams, send_setpoint, target_ok, wait_fcu,
    M, PX4_AUTO, PX4_AUTO_LOITER, PX4_OFFBOARD,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='udpin:0.0.0.0:14601')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--ground-sys', type=int, default=250)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--timeout', type=float, default=1.0)
    ap.add_argument('--lost', type=float, default=5.0)
    ap.add_argument('--max-dist', type=float, default=150.0)
    ap.add_argument('--min-alt', type=float, default=5.0)
    ap.add_argument('--max-alt', type=float, default=60.0)
    ap.add_argument('--max-jump', type=float, default=10.0)
    ap.add_argument('--confirm', type=int, default=10)
    ap.add_argument('--alt-offset', type=float, default=0.0)
    a = ap.parse_args()

    mav = mavutil.mavlink_connection(a.url, baud=a.baud, source_system=1, source_component=191)
    wait_fcu(mav)
    print(f'FCU sys {mav.target_system} comp {mav.target_component}')

    request_streams(mav, [
        (M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20),
        (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2),
    ])

    jump = JumpFilter(a.max_jump, a.confirm)
    target, t_rx, hold = None, None, None
    pos = home = landed = None
    was_offboard = False
    t_start = time.monotonic()
    t_next = t_cmd = t_hb = t_log = 0.0

    while True:
        msg = mav.recv_match(blocking=True, timeout=0.01)
        now = time.monotonic()
        if msg is not None:
            typ = msg.get_type()
            from_fcu = (msg.get_srcSystem() == mav.target_system
                        and msg.get_srcComponent() == mav.target_component)

            if (typ == 'COMMAND_INT' and msg.command == M.MAV_CMD_USER_1
                    and msg.get_srcSystem() == a.ground_sys
                    and msg.target_component == 191
                    and msg.frame == M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT):
                t = (msg.x * 1e-7, msg.y * 1e-7, msg.z + a.alt_offset)
                if home is None:
                    pass
                elif not target_ok(t, home, a.max_dist, a.min_alt, a.max_alt):
                    if now - t_log > 2.0:
                        print(f'범위 밖 표적 거부 {t}')
                        t_log = now
                elif jump(t):
                    target, t_rx = t, now
                elif now - t_log > 2.0:
                    print(f'점프 표적 보류 {t}')
                    t_log = now
            elif from_fcu and typ == 'GLOBAL_POSITION_INT':
                pos = (msg.lat * 1e-7, msg.lon * 1e-7, msg.relative_alt * 1e-3)
            elif from_fcu and typ == 'HOME_POSITION':
                home = (msg.latitude * 1e-7, msg.longitude * 1e-7)
            elif from_fcu and typ == 'EXTENDED_SYS_STATE':
                landed = msg.landed_state
            elif from_fcu and typ == 'HEARTBEAT':
                in_offboard = (msg.custom_mode >> 16) & 0xFF == PX4_OFFBOARD
                if was_offboard and not in_offboard:
                    print('OFFBOARD 해제됨, 종료')
                    return
                if in_offboard and not was_offboard:
                    print('OFFBOARD 진입')
                was_offboard = was_offboard or in_offboard

        if now - t_hb > 1.0:
            t_hb = now
            mav.mav.heartbeat_send(M.MAV_TYPE_ONBOARD_CONTROLLER, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            if home is None:
                command(mav, M.MAV_CMD_REQUEST_MESSAGE, M.MAVLINK_MSG_ID_HOME_POSITION)
            if now - t_start > 3.0 and None in (pos, home, landed):
                print(f'대기: GLOBAL_POSITION_INT={pos is not None}, HOME_POSITION={home is not None}, '
                      f'EXTENDED_SYS_STATE={landed is not None}')

        if now < t_next or pos is None or home is None:
            continue
        t_next = now + 1.0 / a.rate

        age = math.inf if t_rx is None else now - t_rx
        fresh = age < a.timeout
        if fresh:
            sp, hold = target, None
        elif target is None:
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
            print(f'표적 {age:.1f}s 끊김, Hold 요청')
        elif fresh and not was_offboard and landed == M.MAV_LANDED_STATE_IN_AIR:
            command(mav, M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_OFFBOARD)
            t_cmd = now
            print(f'OFFBOARD 요청, 표적 {sp}')


if __name__ == '__main__':
    main()
