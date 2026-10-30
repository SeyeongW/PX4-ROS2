"""Legacy onboard preparation checks: fake MAVLink only, never hardware."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ONBOARD=Path(__file__).resolve().parents[1]/'onboard'
sys.path.insert(0,str(ONBOARD))
import common


def script(name):
    spec=importlib.util.spec_from_file_location('audit_'+name,ONBOARD/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name',['interceptor','offboard_follower','detector'])
def test_default_cli_never_opens_hardware_or_logs(monkeypatch,tmp_path,name):
    module=script(name)
    def forbidden(*args,**kwargs):
        pytest.fail('default CLI attempted hardware I/O')
    monkeypatch.setattr(common.mavutil,'mavlink_connection',forbidden)
    if hasattr(module,'socket'):monkeypatch.setattr(module.socket,'socket',forbidden)
    if hasattr(module,'cv2'):monkeypatch.setattr(module.cv2,'VideoCapture',forbidden)
    args=['--log',str(tmp_path/'flight.log')] if name=='interceptor' else []
    monkeypatch.setattr(sys,'argv',[name+'.py',*args])
    module.main()
    assert not list(tmp_path.iterdir())


def test_fcu_wait_has_finite_deadline_and_ignores_other_vehicle(monkeypatch):
    ticks=iter([0.,0.,.1,1.1]);monkeypatch.setattr(common.time,'monotonic',lambda:next(ticks))
    other=SimpleNamespace(autopilot=common.M.MAV_AUTOPILOT_PX4,
        get_srcSystem=lambda:2,get_srcComponent=lambda:1)
    calls=[]
    def receive(**kwargs):calls.append(kwargs);return other
    mav=SimpleNamespace(recv_match=receive)
    with pytest.raises(TimeoutError):common.wait_fcu(mav,timeout=1.,expected_system=1)
    assert calls and all(0<call['timeout']<=.5 for call in calls)


def test_jump_filter_rejects_nonfinite_first_sample():
    gate=common.JumpFilter(10.,3)
    assert not gate((float('nan'),127.,10.))
    assert gate.accepted is None


def target_packet(**overrides):
    values=dict(command=common.M.MAV_CMD_USER_1,target_system=1,target_component=191,
        frame=common.M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,param1=30.,param2=.3,
        get_srcSystem=lambda:250,get_srcComponent=lambda:191)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('change',[{'target_system':2},{'target_component':1},
    {'param1':0.},{'param2':float('nan')},{'param2':3.},{'get_srcComponent':lambda:1}])
def test_target_address_and_uncertainty_are_checked(change):
    args=script('offboard_follower').parse_args([])
    assert common.target_message_ok(target_packet(),args)
    assert not common.target_message_ok(target_packet(**change),args)


def test_telemetry_replay_cannot_refresh_feedback_deadline():
    received,boots={},{}
    msg=SimpleNamespace(get_type=lambda:'GLOBAL_POSITION_INT',time_boot_ms=1000,
        lat=375000000,lon=1270000000,relative_alt=10000,vx=0,vy=0,vz=0)
    assert common.record_telemetry(msg,received,boots,1.)
    assert not common.record_telemetry(msg,received,boots,2.)
    received.update(HEARTBEAT=2.,EXTENDED_SYS_STATE=2.)
    assert not common.telemetry_fresh(received,2.,.5)
    msg.time_boot_ms=2000
    assert common.record_telemetry(msg,received,boots,2.)
    assert common.telemetry_fresh(received,2.,.5)
    assert not common.telemetry_fresh(received,2.,.5,attitude=True)


def test_timestamped_home_replay_cannot_replace_reference():
    received,stamps={},{}
    msg=SimpleNamespace(get_type=lambda:'HOME_POSITION',latitude=375000000,
        longitude=1270000000,altitude=143000,time_usec=200)
    assert common.record_telemetry(msg,received,stamps,1.)
    msg.time_usec=100;msg.altitude=1000
    assert not common.record_telemetry(msg,received,stamps,2.)
    msg.time_usec=200
    assert not common.record_telemetry(msg,received,stamps,2.)
    assert received['HOME_POSITION']==1.


@pytest.mark.parametrize('field',['altitude','latitude','longitude'])
@pytest.mark.parametrize('timestamped',[False,True])
def test_home_change_after_arm_stops_before_takeoff(monkeypatch,capsys,field,timestamped):
    module=script('interceptor');clock=[0.];index=[0];armed=[False];acks=[];sent=[]
    def now():clock[0]+=.02;return clock[0]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=now,sleep=lambda _:None))
    def message(kind,**values):
        return SimpleNamespace(get_type=lambda:kind,get_srcSystem=lambda:1,get_srcComponent=lambda:1,**values)
    def receive(**kwargs):
        if clock[0]>4.:pytest.fail('changed HOME did not end the active command session')
        if acks:return acks.pop(0)
        index[0]+=1;slot=index[0]%6
        if slot==0:return message('HEARTBEAT',base_mode=common.M.MAV_MODE_FLAG_SAFETY_ARMED if armed[0] else 0,custom_mode=0)
        if slot==1:return message('GLOBAL_POSITION_INT',time_boot_ms=index[0]*20,lat=375000000,lon=1270000000,
            relative_alt=0,vx=0,vy=0,vz=0)
        if slot==2:
            values=dict(latitude=375000000,longitude=1270000000,altitude=143000)
            if timestamped:values['time_usec']=index[0]*20000
            if armed[0]:values[field]=1000 if field=='altitude' else values[field]+1
            return message('HOME_POSITION',**values)
        if slot==3:return message('EXTENDED_SYS_STATE',landed_state=common.M.MAV_LANDED_STATE_ON_GROUND)
        if slot==4:return message('ATTITUDE_QUATERNION',time_boot_ms=index[0]*20,q1=1.,q2=0.,q3=0.,q4=0.)
        return target_packet(get_type=lambda:'COMMAND_INT',x=375001000,y=1270000000,z=10.)
    def send(*packet):
        sent.append(packet)
        if packet[2]==common.M.MAV_CMD_NAV_TAKEOFF:pytest.fail('takeoff was sent after the HOME reference changed')
        if packet[2]==common.M.MAV_CMD_COMPONENT_ARM_DISARM:
            armed[0]=True
            acks.append(message('COMMAND_ACK',command=packet[2],result=common.M.MAV_RESULT_ACCEPTED))
    mav=SimpleNamespace(target_system=1,target_component=1,recv_match=receive,close=lambda:None,
        mav=SimpleNamespace(command_long_send=send,heartbeat_send=lambda *args:None,
            set_position_target_global_int_send=lambda *args:None))
    monkeypatch.setattr(module.mavutil,'mavlink_connection',lambda *args,**kwargs:mav)
    monkeypatch.setattr(module,'wait_fcu',lambda *args,**kwargs:None)
    def no_vision(*args):raise BlockingIOError
    sock=SimpleNamespace(bind=lambda *a:None,setblocking=lambda *a:None,recvfrom=no_vision,close=lambda:None)
    monkeypatch.setattr(module.socket,'socket',lambda *args:sock)
    assert module.main(['--execute'])==1
    assert sum(p[2]==common.M.MAV_CMD_COMPONENT_ARM_DISARM for p in sent)==1
    assert 'home_reference_changed' in capsys.readouterr().out


def test_ack_watchdog_blocks_duplicate_request_and_rejection(monkeypatch):
    sent=[];monkeypatch.setattr(common,'command',lambda *args:sent.append(args))
    gate=common.CommandWatchdog(timeout=3.)
    assert gate.request(None,1.,common.M.MAV_CMD_DO_SET_MODE,1,common.PX4_OFFBOARD)
    assert not gate.request(None,1.1,common.M.MAV_CMD_COMPONENT_ARM_DISARM,1)
    assert gate.failure(4.1)=='command_ack_timeout:176'
    other=common.CommandWatchdog();other.request(None,1.,common.M.MAV_CMD_COMPONENT_ARM_DISARM,1)
    other.receive(SimpleNamespace(command=common.M.MAV_CMD_COMPONENT_ARM_DISARM,result=common.M.MAV_RESULT_DENIED))
    assert other.failure(1.1).startswith('command_rejected:')
    assert len(sent)==2


@pytest.mark.parametrize('cmd,params',[
    (common.M.MAV_CMD_COMPONENT_ARM_DISARM,(1,)),
    (common.M.MAV_CMD_DO_SET_MODE,(1,common.PX4_OFFBOARD)),
])
def test_accepted_ack_without_observed_state_has_bounded_wait(monkeypatch,cmd,params):
    sent=[];monkeypatch.setattr(common,'command',lambda *args:sent.append(args))
    gate=common.CommandWatchdog()
    assert gate.request(None,1.,cmd,*params)
    gate.receive(SimpleNamespace(command=cmd,result=common.M.MAV_RESULT_ACCEPTED))
    assert not gate.request(None,2.,cmd,*params)
    assert gate.failure(7.)==f'command_state_timeout:{cmd}'
    assert len(sent)==1


@pytest.mark.parametrize('ack_first',[False,True])
def test_state_and_ack_both_required_in_either_arrival_order(monkeypatch,ack_first):
    monkeypatch.setattr(common,'command',lambda *args:None)
    gate=common.CommandWatchdog();gate.request(None,1.,common.M.MAV_CMD_COMPONENT_ARM_DISARM,1)
    ack=SimpleNamespace(command=common.M.MAV_CMD_COMPONENT_ARM_DISARM,result=common.M.MAV_RESULT_ACCEPTED)
    heartbeat=SimpleNamespace(get_type=lambda:'HEARTBEAT',base_mode=common.M.MAV_MODE_FLAG_SAFETY_ARMED)
    first,second=((lambda:gate.receive(ack)),(lambda:gate.observe(heartbeat)))
    if not ack_first:first,second=second,first
    first()
    assert not gate.request(None,1.1,common.M.MAV_CMD_DO_SET_MODE,1,common.PX4_OFFBOARD)
    second()
    assert gate.request(None,1.2,common.M.MAV_CMD_DO_SET_MODE,1,common.PX4_OFFBOARD)


@pytest.mark.parametrize('relative_mm',[0,5000])
def test_takeoff_checks_relative_height_separately_from_amsl(monkeypatch,relative_mm):
    monkeypatch.setattr(common,'command',lambda *args:None)
    gate=common.CommandWatchdog()
    gate.request(None,1.,common.M.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,148.,relative_altitude=5.)
    gate.receive(SimpleNamespace(command=common.M.MAV_CMD_NAV_TAKEOFF,result=common.M.MAV_RESULT_ACCEPTED))
    gate.observe(SimpleNamespace(get_type=lambda:'GLOBAL_POSITION_INT',relative_alt=relative_mm))
    assert not gate.request(None,2.,common.M.MAV_CMD_DO_SET_MODE,1,common.PX4_OFFBOARD)
    gate.observe(SimpleNamespace(get_type=lambda:'EXTENDED_SYS_STATE',landed_state=common.M.MAV_LANDED_STATE_IN_AIR))
    if relative_mm==5000:
        assert gate.failure(32.) is None
        assert gate.request(None,32.,common.M.MAV_CMD_DO_SET_MODE,1,common.PX4_OFFBOARD)
    else:assert gate.failure(32.)==f'command_state_timeout:{common.M.MAV_CMD_NAV_TAKEOFF}'


@pytest.mark.parametrize('name,transition,home_fault',[
    ('interceptor','arm',None),('interceptor','offboard',None),('offboard_follower','offboard',None),
    ('interceptor','takeoff',None),('interceptor','arm','no_home'),('interceptor','arm','missing_alt'),
    ('interceptor','arm','bad_alt'),('interceptor','arm','bad_lat'),('interceptor','arm','wrong_source')])
def test_main_requires_surveyed_home_and_observed_state(monkeypatch,name,transition,home_fault):
    module=script(name);clock=[0.];index=[0];acks=[];sent=[]
    def now():clock[0]+=.02;return clock[0]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=now,sleep=lambda _:None))
    def message(kind,**values):
        return SimpleNamespace(get_type=lambda:kind,get_srcSystem=lambda:1,get_srcComponent=lambda:1,**values)
    def receive(**kwargs):
        if clock[0]>(40. if transition=='takeoff' else 8.):pytest.fail('actual vehicle state transition had no finite deadline')
        if acks:return acks.pop(0)
        index[0]+=1;slot=index[0]%6;airborne=transition=='offboard'
        if slot==0:return message('HEARTBEAT',base_mode=common.M.MAV_MODE_FLAG_SAFETY_ARMED if transition in ('offboard','takeoff') else 0,custom_mode=0)
        if slot==1:return message('GLOBAL_POSITION_INT',time_boot_ms=index[0]*20,lat=375000000,lon=1270000000,
            relative_alt=10000 if airborne else 0,vx=0,vy=0,vz=0)
        if slot==2:
            if home_fault=='no_home':return None
            home=message('HOME_POSITION',latitude=375000000,longitude=1270000000,altitude=143000)
            if home_fault=='missing_alt':del home.altitude
            elif home_fault=='bad_alt':home.altitude=float('nan')
            elif home_fault=='bad_lat':home.latitude=950000000
            elif home_fault=='wrong_source':home.get_srcSystem=lambda:2
            return home
        if slot==3:return message('EXTENDED_SYS_STATE',landed_state=common.M.MAV_LANDED_STATE_IN_AIR if airborne else common.M.MAV_LANDED_STATE_ON_GROUND)
        if slot==4:return message('ATTITUDE_QUATERNION',time_boot_ms=index[0]*20,q1=1.,q2=0.,q3=0.,q4=0.)
        return target_packet(get_type=lambda:'COMMAND_INT',x=375001000,y=1270000000,z=10.)
    def send(*packet):
        sent.append(packet)
        if packet[2] in (common.M.MAV_CMD_COMPONENT_ARM_DISARM,common.M.MAV_CMD_DO_SET_MODE,common.M.MAV_CMD_NAV_TAKEOFF):
            acks.append(message('COMMAND_ACK',command=packet[2],result=common.M.MAV_RESULT_ACCEPTED))
    mav=SimpleNamespace(target_system=1,target_component=1,recv_match=receive,close=lambda:None,
        mav=SimpleNamespace(command_long_send=send,heartbeat_send=lambda *args:None,
            set_position_target_global_int_send=lambda *args:None))
    monkeypatch.setattr(module.mavutil,'mavlink_connection',lambda *args,**kwargs:mav)
    monkeypatch.setattr(module,'wait_fcu',lambda *args,**kwargs:None)
    if name=='interceptor':
        def no_vision(*args):raise BlockingIOError
        sock=SimpleNamespace(bind=lambda *a:None,setblocking=lambda *a:None,recvfrom=no_vision,close=lambda:None)
        monkeypatch.setattr(module.socket,'socket',lambda *args:sock)
    assert module.main(['--execute','--connect-timeout','2.0' if home_fault else '10.0'])==1
    if home_fault:
        assert not any(p[2] in (common.M.MAV_CMD_COMPONENT_ARM_DISARM,common.M.MAV_CMD_DO_SET_MODE,common.M.MAV_CMD_NAV_TAKEOFF) for p in sent)
        return
    if transition=='takeoff':
        requests=[p for p in sent if p[2]==common.M.MAV_CMD_NAV_TAKEOFF]
        assert len(requests)==1 and requests[0][10]==148.  # HOME 143m AMSL + relative climb 5m.
    elif transition=='arm':requests=[p for p in sent if p[2]==common.M.MAV_CMD_COMPONENT_ARM_DISARM]
    else:requests=[p for p in sent if p[2]==common.M.MAV_CMD_DO_SET_MODE and p[5]==common.PX4_OFFBOARD]
    assert len(requests)==1


@pytest.mark.parametrize('name',['interceptor','offboard_follower'])
def test_explicit_connection_timeout_closes_link(monkeypatch,name):
    module=script(name);closed=[]
    fake=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(module.mavutil,'mavlink_connection',lambda *args,**kwargs:fake)
    def timeout(*args,**kwargs):raise TimeoutError('test_timeout')
    monkeypatch.setattr(module,'wait_fcu',timeout)
    with pytest.raises(TimeoutError):module.main(['--execute'])
    assert closed==[True]


def test_camera_open_failure_releases_without_name_error(monkeypatch):
    module=script('detector');closed=[]
    monkeypatch.setattr(module.cv2,'VideoCapture',lambda *a:SimpleNamespace(
        isOpened=lambda:False,release=lambda:closed.append(True)))
    with pytest.raises(RuntimeError,match='카메라 열기 실패'):module.main(['--execute'])
    assert closed==[True]


@pytest.mark.parametrize('stop',['flight_exit','sigterm'])
def test_launcher_uses_private_output_and_reaps_mock_children(tmp_path,stop):
    import os
    import subprocess
    import textwrap
    (tmp_path/'run.sh').write_text((ONBOARD/'run.sh').read_text())
    (tmp_path/'interceptor.py').write_text(textwrap.dedent('''
        import argparse,os,time
        def parse_args(argv):
            parser=argparse.ArgumentParser()
            parser.add_argument('--execute',action='store_true')
            parser.add_argument('--udp-port',type=int,default=15760)
            parser.add_argument('--log',default='')
            return parser.parse_args(argv)
        if __name__=='__main__':
            print('MOCK_RUNTIME='+os.environ['TMPDIR'],flush=True)
            time.sleep(30 if os.environ.get('MOCK_STOP')=='sigterm' else .2)
    '''))
    (tmp_path/'detector.py').write_text("import os,time\nfrom pathlib import Path\n"
        +f"Path({str(tmp_path/'detector.pid')!r}).write_text(str(os.getpid()))\n"
        +"try: time.sleep(30)\nexcept KeyboardInterrupt: pass\n")
    process=subprocess.Popen(['bash',str(tmp_path/'run.sh'),'--execute'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        env=dict(os.environ,TMPDIR=str(tmp_path),PYTHONDONTWRITEBYTECODE='1',MOCK_STOP=stop))
    first=process.stdout.readline()
    if stop=='sigterm':process.terminate()
    stdout,stderr=process.communicate(timeout=10)
    assert process.returncode==(143 if stop=='sigterm' else 0),stderr
    runtime=next(line.split('=',1)[1] for line in (first+stdout).splitlines() if line.startswith('MOCK_RUNTIME='))
    assert not Path(runtime).exists()
    assert not list(tmp_path.glob('balloon_onboard_*'))
    if (tmp_path/'detector.pid').exists():
        with pytest.raises(ProcessLookupError):os.kill(int((tmp_path/'detector.pid').read_text()),0)


@pytest.mark.parametrize('name',['interceptor','offboard_follower'])
def test_replayed_feedback_cannot_skip_the_control_watchdog(monkeypatch,name):
    module=script(name);clock=[0.];sent=[];closed=[]
    def now():clock[0]+=.05;return clock[0]
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=now,sleep=lambda _:None))
    def message(kind,**values):
        return SimpleNamespace(get_type=lambda:kind,get_srcSystem=lambda:1,get_srcComponent=lambda:1,**values)
    replay=message('GLOBAL_POSITION_INT',time_boot_ms=10,lat=375000000,lon=1270000000,
        relative_alt=10000,vx=0,vy=0,vz=0)
    pending=[message('HEARTBEAT',base_mode=0,custom_mode=0),
        message('HOME_POSITION',latitude=375000000,longitude=1270000000,altitude=143000),
        message('EXTENDED_SYS_STATE',landed_state=common.M.MAV_LANDED_STATE_IN_AIR),
        message('ATTITUDE_QUATERNION',time_boot_ms=10,q1=1.,q2=0.,q3=0.,q4=0.),replay]
    def receive(**kwargs):
        if clock[0]>3.:pytest.fail('replayed messages bypassed the telemetry watchdog')
        return pending.pop(0) if pending else replay
    mav=SimpleNamespace(target_system=1,target_component=1,recv_match=receive,close=lambda:closed.append(True),
        mav=SimpleNamespace(command_long_send=lambda *args:sent.append(args),heartbeat_send=lambda *args:None,
            set_position_target_global_int_send=lambda *args:None))
    monkeypatch.setattr(module.mavutil,'mavlink_connection',lambda *args,**kwargs:mav)
    monkeypatch.setattr(module,'wait_fcu',lambda *args,**kwargs:None)
    if name=='interceptor':
        def no_vision(*args):raise BlockingIOError
        sock=SimpleNamespace(bind=lambda *a:None,setblocking=lambda *a:None,recvfrom=no_vision,close=lambda:None)
        monkeypatch.setattr(module.socket,'socket',lambda *args:sock)
    assert module.main(['--execute','--connect-timeout','0.6'])==1
    assert closed==[True]
    assert not any(packet[2]==common.M.MAV_CMD_COMPONENT_ARM_DISARM for packet in sent)
