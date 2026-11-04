"""Shared MAVLink utilities for onboard scripts (no ROS dependency)."""
import math

from pymavlink import mavutil

M = mavutil.mavlink

POS_ONLY = (M.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            M.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            M.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            M.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            M.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            M.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            M.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            M.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

ACC_ONLY = (M.POSITION_TARGET_TYPEMASK_X_IGNORE |
            M.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            M.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            M.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            M.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            M.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            M.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            M.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

PX4_AUTO = 4
PX4_OFFBOARD = 6
PX4_AUTO_LOITER = 3
EARTH_R = 6378137.0


def horiz_dist_m(lat1, lon1, lat2, lon2):
    dn = math.radians(lat2 - lat1) * EARTH_R
    de = math.radians(lon2 - lon1) * EARTH_R * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


def dist3_m(a, b):
    return math.hypot(horiz_dist_m(a[0], a[1], b[0], b[1]), b[2] - a[2])


def target_ok(t, home, max_dist, min_alt, max_alt):
    if not all(math.isfinite(v) for v in t):
        return False
    return min_alt <= t[2] <= max_alt and horiz_dist_m(home[0], home[1], t[0], t[1]) <= max_dist


class JumpFilter:
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


def send_acc(mav, ax, ay, az):
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        M.MAV_FRAME_LOCAL_NED, ACC_ONLY,
        0, 0, 0, 0, 0, 0, float(ax), float(ay), float(az), 0, 0)


def command(mav, cmd, *params):
    p = list(params) + [0] * (7 - len(params))
    mav.mav.command_long_send(mav.target_system, mav.target_component, cmd, 0, *p)


def wait_fcu(mav):
    while True:
        hb = mav.recv_match(type='HEARTBEAT', blocking=True)
        if hb.autopilot == M.MAV_AUTOPILOT_PX4:
            mav.target_system = hb.get_srcSystem()
            mav.target_component = hb.get_srcComponent()
            return


def request_streams(mav, streams):
    for msg_id, hz in streams:
        command(mav, M.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
