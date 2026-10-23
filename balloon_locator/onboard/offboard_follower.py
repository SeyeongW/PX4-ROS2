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

M = mavutil.mavlink
POS_ONLY = (M.POSITION_TARGET_TYPEMASK_VX_IGNORE | M.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            M.POSITION_TARGET_TYPEMASK_VZ_IGNORE | M.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            M.POSITION_TARGET_TYPEMASK_AY_IGNORE | M.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            M.POSITION_TARGET_TYPEMASK_YAW_IGNORE | M.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
PX4_AUTO, PX4_OFFBOARD = 4, 6                          # PX4 custom main mode
PX4_AUTO_LOITER = 3                                    # PX4 AUTO sub mode
EARTH_R = 6378137.0


def horiz_dist_m(lat1, lon1, lat2, lon2):
    dn = math.radians(lat2 - lat1) * EARTH_R
    de = math.radians(lon2 - lon1) * EARTH_R * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


def dist3_m(a, b):
    """(lat, lon, alt) 두 점 사이 거리 [m]."""
    return math.hypot(horiz_dist_m(a[0], a[1], b[0], b[1]), b[2] - a[2])


def target_ok(t, home, max_dist, min_alt, max_alt):
    """유한값, 홈 기준 고도 범위, 홈 기준 수평거리 확인. home = (lat, lon)."""
    if not all(math.isfinite(v) for v in t):
        return False
    return min_alt <= t[2] <= max_alt and horiz_dist_m(home[0], home[1], t[0], t[1]) <= max_dist


class JumpFilter:
    """마지막 채택값에서 max_jump [m] 이상 튄 표적은 tol 이내 같은 자리로 confirm회 연속일 때만 채택."""

    def __init__(self, max_jump, confirm, tol=2.0):
        self.max_jump, self.confirm, self.tol = max_jump, confirm, tol
        self.accepted, self.cand, self.count = None, None, 0

    def __call__(self, t) -> bool:
        if self.accepted is None or dist3_m(self.accepted, t) <= self.max_jump:
            self.accepted, self.cand, self.count = t, None, 0
            return True
        if self.cand is not None and dist3_m(self.cand, t) <= self.tol:
            self.count += 1
        else:
            self.cand, self.count = t, 1
        if self.count >= self.confirm:
            self.accepted, self.cand, self.count = t, None, 0
            return True
        return False


def send_setpoint(mav, lat, lon, alt):
    mav.mav.set_position_target_global_int_send(
        0, mav.target_system, mav.target_component, M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        POS_ONLY, int(round(lat * 1e7)), int(round(lon * 1e7)), alt, 0, 0, 0, 0, 0, 0, 0, 0)


def command(mav, cmd, *params):
    p = list(params) + [0] * (7 - len(params))
    mav.mav.command_long_send(mav.target_system, mav.target_component, cmd, 0, *p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='udpin:0.0.0.0:14601')    # mavlink-router endpoint
    ap.add_argument('--baud', type=int, default=921600)        # serial url에서만 사용
    ap.add_argument('--ground-sys', type=int, default=250)     # mavlink_bridge_node source_system
    ap.add_argument('--rate', type=float, default=20.0)        # setpoint 주기 [Hz]
    ap.add_argument('--timeout', type=float, default=1.0)      # 표적 끊김 -> 정지 [s]
    ap.add_argument('--lost', type=float, default=5.0)         # 표적 끊김 -> Hold 전환 [s]
    ap.add_argument('--max-dist', type=float, default=150.0)   # 홈 기준 수평거리 한계 [m]
    ap.add_argument('--min-alt', type=float, default=5.0)      # 홈 기준 고도 하한 [m]
    ap.add_argument('--max-alt', type=float, default=60.0)     # 홈 기준 고도 상한 [m]
    ap.add_argument('--max-jump', type=float, default=10.0)    # 표적 점프 한계 [m]
    ap.add_argument('--confirm', type=int, default=10)         # 점프 표적 채택에 필요한 연속 횟수
    ap.add_argument('--alt-offset', type=float, default=0.0)   # 표적 고도에 더함 [m] (+ 위, 풍선과 거리 두기)
    a = ap.parse_args()

    mav = mavutil.mavlink_connection(a.url, baud=a.baud, source_system=1, source_component=191)
    while True:                                        # 포워딩된 GCS heartbeat는 건너뜀
        hb = mav.recv_match(type='HEARTBEAT', blocking=True)
        if hb.autopilot == M.MAV_AUTOPILOT_PX4:
            mav.target_system, mav.target_component = hb.get_srcSystem(), hb.get_srcComponent()
            break
    print(f'FCU sys {mav.target_system} comp {mav.target_component}')
    for msg_id, hz in ((M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20), (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2)):
        command(mav, M.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)

    jump = JumpFilter(a.max_jump, a.confirm)
    target, t_rx, hold = None, None, None
    pos = home = landed = None                         # pos = (lat, lon, 홈 기준 고도)
    was_offboard = False
    t_start = time.monotonic()
    t_next = t_cmd = t_hb = t_log = 0.0
    while True:
        msg = mav.recv_match(blocking=True, timeout=0.01)
        now = time.monotonic()
        if msg is not None:
            typ = msg.get_type()
            from_fcu = msg.get_srcSystem() == mav.target_system and msg.get_srcComponent() == mav.target_component
            if (typ == 'COMMAND_INT' and msg.command == M.MAV_CMD_USER_1 and msg.get_srcSystem() == a.ground_sys
                    and msg.target_component == 191 and msg.frame == M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT):
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
                was_offboard |= in_offboard

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
            sp = pos                                   # 표적 없음: 현재 위치를 따라감
        else:
            hold = hold or pos                         # 끊긴 자리에서 정지
            sp = hold
        send_setpoint(mav, *sp)                        # OFFBOARD 전후 계속 스트리밍

        if now - t_cmd < 1.0:
            continue
        if was_offboard and age > a.lost:
            command(mav, M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_AUTO, PX4_AUTO_LOITER)
            t_cmd = now
            print(f'표적 {age:.1f}s 끊김, Hold 요청')
        elif fresh and not was_offboard and landed == M.MAV_LANDED_STATE_IN_AIR:
            command(mav, M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_OFFBOARD)
            t_cmd = now
            print(f'OFFBOARD 요청, 표적 {sp}')


if __name__ == '__main__':
    main()
