"""센서 유래 계획·기하 제한·자기 기체 제외의 독립 검사. 비행 자체는 Gazebo에서 확인한다."""
from copy import deepcopy
import numpy as np
import pytest
from jol_perception.observer_flight import make_flight_plan, nearest_approach
from jol_perception.pointing_core import forecast
from jol_perception.tracking_core import Track, Config
from jol_perception.motion_compensation import observer_filter_points


def inputs():
    track = Track(id=1,state=np.array([30.,-20.,6.,0.,0.,0.]),covariance=np.eye(6)*.01,
                  size=np.ones(3),born=9.,last_seen=10.,hits=3,is_confirmed=True,process_acceleration_std=.1)
    item = track.as_dict(10.,config=Config(acceleration_std=.1,low_speed_acceleration_std=.1))
    goal = lambda horizon: dict(forecast(item,10.,10.,horizon),target_id=1,frame_id='world')
    return dict(status=dict(stamp=10.,frame_id='world',command_mode='follow',source_stamp=10.,target_id=1,
                            track_position=item['position'],prediction=goal(.5),turret_prediction=goal(0.),
                            arm_state=dict(control_state_source='camera_lidar')),
                tracks=dict(stamp=10.,frame_id='world',tracks=[item]),
                own=dict(stamp=10.,frame_id='world',position=[-5.,-6.,.09],velocity=[0.,0.,0.]),
                scene=dict(frame_id='world',bounds=dict(min=[-70.,-70.,0.],max=[70.,70.,25.]),obstacles=[]),now=10.)


def test_fusion_drives_bounded_curve_and_landing_corridor():
    data=inputs(); plan=make_flight_plan(**data)
    assert plan['source']=='camera_lidar' and plan['target_id']==1
    assert plan['execution_policy']=='gazebo_classic_bounded_observation'
    for segment in plan['segments']:
        assert np.max(np.linalg.norm(np.array(segment['control_points'])[:,:2]-np.array(plan['home'][:2]),axis=1)) <= 12.+1e-8
    assert plan['dynamic_collision_guaranteed'] is False
    blocked=deepcopy(data)
    goal=np.array(plan['goal'])
    blocked['scene']['obstacles']=[dict(min=(goal+[-.1,-.1,-5.8]).tolist(),max=(goal+[.1,.1,-5.5]).tolist())]
    with pytest.raises(ValueError): make_flight_plan(**blocked)


def test_raw_lidar_and_stale_inputs_cannot_start_flight():
    data=inputs(); data['status']['arm_state']['control_state_source']='lidar'
    with pytest.raises(ValueError,match='camera_lidar'):make_flight_plan(**data)
    data=inputs(); data['now']=11.
    with pytest.raises(ValueError):make_flight_plan(**data)
    data=inputs(); data['own']['velocity']=[.2,0.,0.]
    with pytest.raises(ValueError):make_flight_plan(**data)


def test_known_track_approach_uses_measurements_not_truth():
    packet=inputs()['tracks']; packet['tracks'][0].update(position=[20.,0.,0.],velocity=[-10.,0.,0.])
    assert nearest_approach(np.zeros(3),np.zeros(3),packet,10.) == 5.
    assert np.isinf(nearest_approach(np.zeros(3),np.zeros(3),packet,11.))


def test_only_fresh_cooperative_vehicle_points_are_removed():
    points=np.array([[1.,0.,1.],[1.6,0.,1.],[20.,0.,1.]])
    own=dict(stamp=10.,position=[1.,0.,1.],velocity=[2.,0.,0.])
    assert np.array_equal(observer_filter_points(points,10.,own),points[-1:])
    assert np.array_equal(observer_filter_points(points,11.,own),points)
    assert np.array_equal(observer_filter_points(points,10.,None),points)


def test_causal_odometry_defers_future_dds_samples_without_hiding_loss():
    from jol_perception.observer_flight import causal_odometry
    history=[dict(stamp=21.981),dict(stamp=22.001)]
    assert causal_odometry(history,22.)['stamp']==21.981
    assert causal_odometry(history,22.01)['stamp']==22.001
    assert causal_odometry(history,22.3) is None
    assert causal_odometry(history,21.) is None


def test_plan_no_longer_requires_a_physical_arm():
    data=inputs()
    data['status']['prediction_state']=data['status'].pop('arm_state')
    assert make_flight_plan(**data)['source']=='camera_lidar'


def test_overshoot_abort_landing_stays_inside_plugin_command_envelope():
    from jol_perception.observer_flight import bounded_landing_start
    home=np.array([-5.,-6.,.1]);measured=home+[14.,3.,10.]
    point=bounded_landing_start(measured,home)
    assert np.linalg.norm(point[:2]-home[:2])<=12.+1e-10 and point[2]<=8.5
    assert np.array_equal(measured,home+[14.,3.,10.])
    assert np.array_equal(bounded_landing_start(home,home),home)
