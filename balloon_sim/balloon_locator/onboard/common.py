"""Shared MAVLink utilities for onboard scripts (no ROS dependency)."""
import math
import time

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


def add_execution_options(parser):
    parser.add_argument('--execute',action='store_true',help='명시적으로 legacy 실물 I/O/제어 실행; 기본은 설정검사만')
    parser.add_argument('--system-id',type=int,default=1,help='연결할 PX4와 온보드의 MAVLink system ID')
    parser.add_argument('--ground-comp',type=int,default=191)
    parser.add_argument('--connect-timeout',type=float,default=10.)
    parser.add_argument('--telemetry-timeout',type=float,default=.5)
    parser.add_argument('--min-samples',type=int,default=30)
    parser.add_argument('--max-sigma',type=float,default=2.)


def validate_flight_options(args):
    positive=('rate','timeout','lost','max_dist','max_jump','connect_timeout','telemetry_timeout','max_sigma')
    if (any(not math.isfinite(getattr(args,key)) or getattr(args,key)<=0 for key in positive)
            or args.rate<=2. or args.lost<=args.timeout or args.confirm<1 or args.min_samples<1
            or args.baud<=0 or not all(1<=value<=255 for value in (args.system_id,args.ground_sys,args.ground_comp))
            or not all(math.isfinite(value) for value in (args.min_alt,args.max_alt,args.alt_offset))
            or not 0<=args.min_alt<args.max_alt):
        raise ValueError('invalid_legacy_flight_configuration')


def target_message_ok(msg,args):
    """The legacy packet carries quality/addressing, but no source timestamp."""
    return bool(msg.command==M.MAV_CMD_USER_1 and msg.get_srcSystem()==args.ground_sys
        and msg.get_srcComponent()==args.ground_comp and msg.target_system==args.system_id
        and msg.target_component==191 and msg.frame==M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
        and math.isfinite(msg.param1) and msg.param1>=args.min_samples
        and math.isfinite(msg.param2) and 0<=msg.param2<=args.max_sigma)


def home_position(msg):
    """FCU HOME_POSITION: latitude/longitude degE7, altitude millimetres MSL."""
    values=[getattr(msg,name,None) for name in ('latitude','longitude','altitude')]
    if (any(type(value) not in (int,float) or not math.isfinite(value) for value in values)
            or not -900000000<values[0]<900000000 or not -1800000000<=values[1]<=1800000000
            or not -(2**31)<=values[2]<2**31):
        raise ValueError('valid_FCU_home_position_and_AMSL_altitude_required')
    return (values[0]*1e-7,values[1]*1e-7),values[2]*1e-3


def record_telemetry(msg,received,boot_times,now):
    """Do not renew feedback freshness with replayed or malformed FCU samples."""
    kind=msg.get_type()
    if kind=='GLOBAL_POSITION_INT':
        if (not -900000000<msg.lat<900000000 or not -1800000000<=msg.lon<=1800000000
                or not all(math.isfinite(v) for v in (msg.relative_alt,msg.vx,msg.vy,msg.vz))):return False
    elif kind=='ATTITUDE_QUATERNION':
        q=(msg.q1,msg.q2,msg.q3,msg.q4)
        if not all(math.isfinite(v) for v in q) or abs(sum(v*v for v in q)-1.)>.01:return False
    elif kind=='HOME_POSITION':
        try:home_position(msg)
        except ValueError:return False
        # MAVLink 2 extension: zero/absent time_usec has no ordering evidence.
        stamp=getattr(msg,'time_usec',0)
        if type(stamp) is not int or stamp<0:return False
        if stamp:
            if stamp<=boot_times.get(kind,-1):return False
            boot_times[kind]=stamp
    elif kind not in ('HEARTBEAT','EXTENDED_SYS_STATE'):return False
    if hasattr(msg,'time_boot_ms'):
        stamp=msg.time_boot_ms
        if stamp<=boot_times.get(kind,-1):return False
        boot_times[kind]=stamp
    received[kind]=now
    return True


def telemetry_fresh(received,now,timeout,*,attitude=False):
    limits={'GLOBAL_POSITION_INT':timeout,'HEARTBEAT':2.,'EXTENDED_SYS_STATE':2.}
    if attitude:limits['ATTITUDE_QUATERNION']=timeout
    return all(kind in received and 0<=now-received[kind]<=age for kind,age in limits.items())


class CommandWatchdog:
    """ACK acceptance never replaces actual armed/mode feedback from heartbeat."""
    def __init__(self,timeout=3.,state_timeout=5.,takeoff_timeout=30.):
        self.timeout=timeout;self.state_timeout=state_timeout;self.takeoff_timeout=takeoff_timeout
        self.pending=None;self.waiting_state=None;self.error=None
        self.in_air=False;self.relative_alt=None;self.takeoff_relative_alt=None;self.started=False

    def request(self,mav,now,cmd,*params,relative_altitude=None):
        if self.pending is not None or self.waiting_state is not None or self.error:return False
        if cmd==M.MAV_CMD_NAV_TAKEOFF:
            if relative_altitude is None or not math.isfinite(relative_altitude) or relative_altitude<=0:
                raise ValueError('relative_takeoff_height_required_for_state_confirmation')
            self.takeoff_relative_alt=relative_altitude
        command(mav,cmd,*params);self.pending=(cmd,now)
        self.started=True
        if cmd in (M.MAV_CMD_COMPONENT_ARM_DISARM,M.MAV_CMD_DO_SET_MODE,M.MAV_CMD_NAV_TAKEOFF):
            self.waiting_state=(cmd,params,now)
        return True

    def receive(self,msg):
        if self.pending is None or msg.command!=self.pending[0]:return
        if msg.result==M.MAV_RESULT_ACCEPTED:self.pending=None
        elif msg.result!=M.MAV_RESULT_IN_PROGRESS:self.error=f'command_rejected:{msg.command}:{msg.result}'

    def observe(self,msg):
        """Only validated, fresh FCU telemetry reaches this method from callers."""
        kind=msg.get_type()
        if kind=='EXTENDED_SYS_STATE':self.in_air=msg.landed_state==M.MAV_LANDED_STATE_IN_AIR
        elif kind=='GLOBAL_POSITION_INT':self.relative_alt=msg.relative_alt*1e-3
        if self.waiting_state is None:return
        cmd,params,_=self.waiting_state
        observed=False
        if kind=='HEARTBEAT':
            if cmd==M.MAV_CMD_COMPONENT_ARM_DISARM:
                observed=bool(msg.base_mode&M.MAV_MODE_FLAG_SAFETY_ARMED)==bool(params[0])
            elif cmd==M.MAV_CMD_DO_SET_MODE:
                main=(msg.custom_mode>>16)&0xFF;sub=(msg.custom_mode>>24)&0xFF
                observed=main==int(params[1]) and sub==int(params[2] if len(params)>2 else 0)
        elif cmd==M.MAV_CMD_NAV_TAKEOFF and kind in ('EXTENDED_SYS_STATE','GLOBAL_POSITION_INT'):
            observed=self.in_air and self.relative_alt is not None and self.relative_alt>=self.takeoff_relative_alt-1.
        if observed:self.waiting_state=None

    def failure(self,now):
        if self.pending is not None and now-self.pending[1]>self.timeout:
            self.error=f'command_ack_timeout:{self.pending[0]}'
        if self.waiting_state is not None:
            cmd,_,started=self.waiting_state
            limit=self.takeoff_timeout if cmd==M.MAV_CMD_NAV_TAKEOFF else self.state_timeout
            if now-started>limit:self.error=f'command_state_timeout:{cmd}'
        return self.error


def horiz_dist_m(lat1, lon1, lat2, lon2):
    dn = math.radians(lat2 - lat1) * EARTH_R
    de = math.radians(lon2 - lon1) * EARTH_R * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


def dist3_m(a, b):
    return math.hypot(horiz_dist_m(a[0], a[1], b[0], b[1]), b[2] - a[2])


def target_ok(t, home, max_dist, min_alt, max_alt):
    if (len(t)!=3 or len(home)!=2 or not all(math.isfinite(v) for v in (*t,*home))
            or not -90.<t[0]<90. or not -180.<=t[1]<=180.
            or not -90.<home[0]<90. or not -180.<=home[1]<=180.):
        return False
    return min_alt <= t[2] <= max_alt and horiz_dist_m(home[0], home[1], t[0], t[1]) <= max_dist


class JumpFilter:
    def __init__(self, max_jump, confirm, tol=2.0):
        if not math.isfinite(max_jump) or max_jump<=0 or not math.isfinite(tol) or tol<=0 or confirm<1:
            raise ValueError('invalid_jump_filter_configuration')
        self.max_jump, self.confirm, self.tol = max_jump, confirm, tol
        self.accepted, self.cand, self.count = None, None, 0

    def __call__(self, t) -> bool:
        if len(t)!=3 or not all(math.isfinite(v) for v in t):
            self.cand,self.count=None,0
            return False
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
    if not all(math.isfinite(v) for v in (lat,lon,alt)) or not -90.<lat<90. or not -180.<=lon<=180.:
        raise ValueError('invalid_global_setpoint')
    mav.mav.set_position_target_global_int_send(
        0, mav.target_system, mav.target_component, M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        POS_ONLY, int(round(lat * 1e7)), int(round(lon * 1e7)), alt, 0, 0, 0, 0, 0, 0, 0, 0)


def send_acc(mav, ax, ay, az):
    if not all(math.isfinite(v) for v in (ax,ay,az)):raise ValueError('invalid_acceleration_setpoint')
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        M.MAV_FRAME_LOCAL_NED, ACC_ONLY,
        0, 0, 0, 0, 0, 0, float(ax), float(ay), float(az), 0, 0)


def command(mav, cmd, *params):
    p = list(params) + [0] * (7 - len(params))
    mav.mav.command_long_send(mav.target_system, mav.target_component, cmd, 0, *p)


def wait_fcu(mav,timeout=10.,expected_system=1):
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('invalid_FCU_timeout')
    deadline=time.monotonic()+timeout
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('PX4_heartbeat_timeout')
        hb=mav.recv_match(type='HEARTBEAT',blocking=True,timeout=min(.5,remaining))
        if (hb is not None and hb.autopilot==M.MAV_AUTOPILOT_PX4
                and hb.get_srcSystem()==expected_system and hb.get_srcComponent()==M.MAV_COMP_ID_AUTOPILOT1):
            mav.target_system = hb.get_srcSystem()
            mav.target_component = hb.get_srcComponent()
            return


def request_streams(mav, streams):
    for msg_id, hz in streams:
        command(mav, M.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
