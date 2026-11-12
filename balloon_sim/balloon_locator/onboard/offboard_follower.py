#!/usr/bin/env python3
"""Legacy 온보드 waypoint follower. 기본은 오프라인 설정검사이며 --execute에서만 I/O한다.
UI의 3m 이륙/Bézier 경로 제어기가 아니다. ROS 없이 pymavlink만 사용한다.

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
from contextlib import ExitStack
import math
import time

from pymavlink import mavutil

from common import (
    JumpFilter, CommandWatchdog, add_execution_options, validate_flight_options,
    target_message_ok, home_position, record_telemetry, telemetry_fresh,
    command, request_streams, send_setpoint, target_ok, wait_fcu,
    M, PX4_AUTO, PX4_AUTO_LOITER, PX4_OFFBOARD,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    add_execution_options(ap)
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
    a = ap.parse_args(argv)
    try:validate_flight_options(a)
    except ValueError as error:ap.error(str(error))
    return a


def main(argv=None):
    a=parse_args(argv)
    if not a.execute:
        print('오프라인 설정검사 통과. Legacy waypoint follower I/O는 실행하지 않았습니다. 실제 연결은 --execute가 필요합니다.')
        return 0
    with ExitStack() as resources:
        return run(a,resources)


def run(a,resources):

    mav = mavutil.mavlink_connection(a.url, baud=a.baud, source_system=a.system_id, source_component=191)
    resources.callback(mav.close)
    wait_fcu(mav,timeout=a.connect_timeout,expected_system=a.system_id)
    print(f'FCU sys {mav.target_system} comp {mav.target_component}')

    request_streams(mav, [
        (M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20),
        (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2),
    ])

    jump = JumpFilter(a.max_jump, a.confirm)
    target, t_rx, hold = None, None, None
    pos = home = landed = None
    home_alt_amsl=None
    was_offboard = False
    offboard_requested=False
    received,boot_times={},{}
    commands=CommandWatchdog()
    stream_start=last_setpoint=None
    t_start = time.monotonic()
    t_next = t_cmd = t_hb = t_log = 0.0

    while True:
        msg = mav.recv_match(blocking=True, timeout=0.01)
        now = time.monotonic()
        if msg is not None:
            typ = msg.get_type()
            from_fcu = (msg.get_srcSystem() == mav.target_system
                        and msg.get_srcComponent() == mav.target_component)

            if typ=='COMMAND_ACK' and from_fcu:
                commands.receive(msg)
            if from_fcu and typ!='COMMAND_ACK':
                from_fcu=record_telemetry(msg,received,boot_times,now)
                if from_fcu:commands.observe(msg)

            if typ=='COMMAND_INT' and target_message_ok(msg,a):
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
                next_home,next_alt=home_position(msg)
                if commands.started and home is not None and (next_home!=home or next_alt!=home_alt_amsl):
                    print('home_reference_changed_after_control_start_restart_required')
                    if was_offboard:
                        command(mav,M.MAV_CMD_DO_SET_MODE,M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,PX4_AUTO,PX4_AUTO_LOITER)
                    return 1
                home,home_alt_amsl=next_home,next_alt
            elif from_fcu and typ == 'EXTENDED_SYS_STATE':
                landed = msg.landed_state
            elif from_fcu and typ == 'HEARTBEAT':
                in_offboard = (msg.custom_mode >> 16) & 0xFF == PX4_OFFBOARD
                if in_offboard and not offboard_requested:
                    print('다른 제어기의 OFFBOARD 상태: 제어권을 획득하지 않고 종료')
                    return 1
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
            if now-t_start>a.connect_timeout and (pos is None or home is None):
                print('필수 위치/홈 telemetry 수신 시간 초과')
                return 1
            continue
        t_next = now + 1.0 / a.rate

        failure=commands.failure(now)
        if failure or not telemetry_fresh(received,now,a.telemetry_timeout):
            stream_start=last_setpoint=None
            if failure or was_offboard or commands.started or now-t_start>a.connect_timeout:
                print(failure or 'FCU telemetry stale: 제어 종료')
                if was_offboard:
                    command(mav,M.MAV_CMD_DO_SET_MODE,M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,PX4_AUTO,PX4_AUTO_LOITER)
                return 1
            continue

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
        if last_setpoint is None or now-last_setpoint>.5:stream_start=now
        last_setpoint=now

        if now - t_cmd < 1.0:
            continue
        if was_offboard and age > a.lost:
            if commands.request(mav,now,M.MAV_CMD_DO_SET_MODE, M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    PX4_AUTO, PX4_AUTO_LOITER):
                t_cmd=now;print(f'표적 {age:.1f}s 끊김, Hold 요청')
        elif (fresh and not was_offboard and landed == M.MAV_LANDED_STATE_IN_AIR
                and stream_start is not None and now-stream_start>1.):
            if commands.request(mav,now,M.MAV_CMD_DO_SET_MODE,M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,PX4_OFFBOARD):
                offboard_requested=True;t_cmd=now;print(f'OFFBOARD 요청, 표적 {sp}')


if __name__ == '__main__':
    raise SystemExit(main())
