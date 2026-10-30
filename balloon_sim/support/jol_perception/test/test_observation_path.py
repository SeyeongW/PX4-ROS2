"""고정 관찰 경로의 연속성·운동 한계·샘플 사이 충돌 회귀 검사."""
import json

import numpy as np
import pytest

from jol_perception.observation_path import build_plan, sample_plan


def test_takeoff_then_rising_curve_reaches_exact_goal_and_survives_json():
    plan = build_plan([2., -1., .2], [10., 2., 3.2])
    restored = json.loads(json.dumps(plan, allow_nan=False))
    takeoff_end = plan['segments'][0]['duration_s']
    for t in np.linspace(0., takeoff_end, 41):
        value = sample_plan(restored, t)
        assert np.allclose(value['position'][:2], [2., -1.])
        assert .2 <= value['position'][2] <= 1.2+1e-12
    for t, position in [(0., [2., -1., .2]), (takeoff_end, [2., -1., 1.2]),
                        (plan['duration_s'], [10., 2., 3.2])]:
        value = sample_plan(restored, t)
        assert np.allclose(value['position'], position)
        assert np.allclose(value['velocity'], 0.)
        assert np.allclose(value['acceleration'], 0.)
    assert plan['frame_id'] == 'world'
    assert np.all(np.diff([row['time_s'] for row in plan['samples']]) > 0.)
    assert max(np.diff([row['time_s'] for row in plan['samples']])) <= .05+1e-10
    assert sample_plan(restored, plan['duration_s']+5.)['phase'] == 'hold'
    assert sample_plan(restored, -1.)['velocity'] == [0., 0., 0.]


def test_exact_curve_has_vertical_then_horizontal_tangent_and_continuous_join():
    plan = build_plan([0, 0, 0], [8, 0, 3], sample_period=100.)
    segment = plan['segments'][1]
    points = np.asarray(segment['control_points'])
    assert np.allclose(points[1]-points[0], [0., 0., 4./3.])
    assert np.allclose(points[3]-points[2], [8./3., 0., 0.])
    join = segment['start_time_s']
    left, right = sample_plan(plan, join-1e-5), sample_plan(plan, join+1e-5)
    for field in ('position', 'velocity', 'acceleration'):
        assert np.linalg.norm(np.array(left[field])-right[field]) < 1e-4
    halfway = sample_plan(plan, join+segment['duration_s']/2.)
    assert np.allclose(halfway['position'], [3., 0., 2.5])
    # 저장 표본이 시작/구간끝뿐이어도 선형보간하지 않는다.
    assert not np.allclose(halfway['position'], (points[0]+points[-1])/2.)
    slopes = []
    for fraction in np.linspace(.01, .99, 80):
        velocity = sample_plan(plan, join+segment['duration_s']*fraction)['velocity']
        assert velocity[0] > 0 and velocity[2] > 0
        slopes.append(velocity[2]/velocity[0])
    assert np.all(np.diff(slopes) < 0.)


def test_dense_off_grid_derivatives_respect_continuous_limits():
    limits = dict(max_speed=1.3, max_acceleration=.7, max_jerk=.9)
    plan = build_plan([0, 0, 0], [8, -4, 3], sample_period=1.7, **limits)
    values = [sample_plan(plan, t) for t in np.linspace(0., plan['duration_s'], 4003)]
    for field, key in [('velocity', 'max_speed'), ('acceleration', 'max_acceleration'), ('jerk', 'max_jerk')]:
        assert max(np.linalg.norm(row[field]) for row in values) <= limits[key]*(1+1e-10)
    other = build_plan([0, 0, 0], [8, -4, 3], sample_period=.05, **limits)
    assert other['segments'] == plan['segments']
    # 해석적 속도·가속도를 독립적인 중앙 차분으로 검산한다.
    t, dt = plan['duration_s']*.7, 1e-3
    low, mid, high = (sample_plan(plan, value) for value in (t-dt, t, t+dt))
    assert np.allclose((np.array(high['position'])-low['position'])/(2*dt), mid['velocity'], atol=1e-7)
    assert np.allclose((np.array(high['velocity'])-low['velocity'])/(2*dt), mid['acceleration'], atol=1e-7)
    assert np.allclose((np.array(high['acceleration'])-low['acceleration'])/(2*dt), mid['jerk'], atol=1e-7)


@pytest.mark.parametrize('goal,segments', [([0, 0, 1], 1), ([0, 0, 3], 2)])
def test_vertical_only_goal(goal, segments):
    plan = build_plan([0, 0, 0], goal)
    assert len(plan['segments']) == segments
    assert np.allclose(plan['samples'][-1]['position'], goal)
    assert all(np.allclose(row['position'][:2], 0.) for row in plan['samples'])


@pytest.mark.parametrize('obstacle', [
    {'min': [-.01, -.01, .431], 'max': [.01, .01, .432]},
    {'min': [2.999, -.001, 2.499], 'max': [3.001, .001, 2.501]},
    {'min': [8., 0., 3.], 'max': [8.1, .1, 3.1]},
])
def test_thin_or_endpoint_obstacle_rejected_between_arbitrarily_sparse_samples(obstacle):
    with pytest.raises(ValueError, match='obstacle|clearance'):
        build_plan([0, 0, 0], [8, 0, 3], sample_period=100., obstacles=[obstacle], clearance=0.)


def test_overlapping_control_bounding_box_can_still_prove_clear_curve():
    # 전체 제어점 상자와 겹치지만 실제 곡선과는 떨어진 장애물이다.
    obstacle = {'min': [.1, -.01, 2.9], 'max': [.2, .01, 3.]}
    assert build_plan([0, 0, 0], [8, 0, 3], obstacles=[obstacle], clearance=.01)
    near = {'min': [2.99, .2, 2.49], 'max': [3.01, .3, 2.51]}
    assert build_plan([0, 0, 0], [8, 0, 3], obstacles=[near], clearance=.1)
    with pytest.raises(ValueError):
        build_plan([0, 0, 0], [8, 0, 3], obstacles=[near], clearance=.25)


@pytest.mark.parametrize('kwargs', [
    {'home': [float('nan'), 0, 0]}, {'goal': [8, 0, float('inf')]}, {'goal': [8, 0, .9]},
    {'goal': [8, 0, 1.]}, {'max_speed': 0.}, {'max_acceleration': -1.}, {'max_jerk': float('nan')},
    {'sample_period': 0.}, {'sample_period': 1e-15}, {'takeoff_height': 0.}, {'clearance': -.1},
    {'max_speed': 1e-300}, {'max_acceleration': 1e-300},
    {'obstacles': [{'min': [1, 1, 1], 'max': [0, 0, 0]}]}, {'obstacles': [None]},
])
def test_invalid_or_unflyable_inputs_are_rejected(kwargs):
    arguments = dict(home=[0, 0, 0], goal=[8, 0, 3])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        build_plan(**arguments)


def test_sample_rejects_unknown_model_and_nonfinite_time():
    plan = build_plan([0, 0, 0], [8, 0, 3])
    for time_s in [float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            sample_plan(plan, time_s)
    plan['model'] = 'unknown'
    with pytest.raises(ValueError):
        sample_plan(plan, .1)


def test_numeric_input_does_not_coerce_boolean_or_string():
    for kwargs in ({'home': [True, 0, 0]}, {'goal': ['8', 0, 3]},
                   {'max_speed': True}, {'clearance': '0.75'}, {'sample_period': False}):
        values = dict(home=[0, 0, 0], goal=[8, 0, 3])
        values.update(kwargs)
        with pytest.raises(ValueError):
            build_plan(**values)
    plan = build_plan([0, 0, 0], [8, 0, 3])
    for stamp in (True, '0.5'):
        with pytest.raises(ValueError):
            sample_plan(plan, stamp)


def test_rounded_right_angle_geometry_derivatives_and_limits():
    from jol_perception.observation_path import build_corner_plan
    plan=build_corner_plan([2.,-1.,.2],[10.,3.,6.2],turn_radius=1.5)
    g=plan['corner_geometry'];p0,pg,c,a,b=(np.array(g[k]) for k in ('P0','Pg','C','A','B'))
    assert np.allclose(c,[p0[0],p0[1],pg[2]])
    assert abs((c-p0)@(pg-c))<1e-12
    assert g['vertical_length_m']>=1.
    assert [s['phase'] for s in plan['segments']]==['takeoff','corner','approach']
    q=np.array(plan['segments'][1]['control_points'])
    assert np.allclose(q[0],a) and np.allclose(q[3],b)
    assert np.allclose((q[1]-q[0])[:2],0.) and q[1,2]>q[0,2]
    assert abs((q[3]-q[2])[2])<1e-12
    assert np.cross(q[3]-q[2],pg-c)@np.cross(q[3]-q[2],pg-c)<1e-20
    for segment in plan['segments']:
        end=segment['start_time_s']+segment['duration_s']
        left,right=sample_plan(plan,end-1e-5),sample_plan(plan,end+1e-5)
        for field in ('position','velocity','acceleration'):
            assert np.linalg.norm(np.array(left[field])-right[field])<1e-4
    for t in np.linspace(0.,plan['segments'][0]['duration_s'],31):
        assert np.allclose(sample_plan(plan,t)['position'][:2],p0[:2])
    assert plan['motion_validation']['passed']
    for field,key in [('velocity','speed_m_s'),('acceleration','acceleration_m_s2'),('jerk','jerk_m_s3')]:
        assert max(np.linalg.norm(sample_plan(plan,t)[field]) for t in np.linspace(0.,plan['duration_s'],1501)) <= plan['limits'][key]*(1+1e-10)
    t=plan['segments'][1]['start_time_s']+.37*plan['segments'][1]['duration_s'];dt=1e-4
    lo,mi,hi=(sample_plan(plan,x) for x in (t-dt,t,t+dt))
    assert np.allclose((np.array(hi['position'])-lo['position'])/(2*dt),mi['velocity'],atol=1e-7)
    assert np.allclose((np.array(hi['velocity'])-lo['velocity'])/(2*dt),mi['acceleration'],atol=1e-7)


def test_rounded_corner_collision_and_degenerate_inputs():
    from jol_perception.observation_path import build_corner_plan
    for goal in ([8,0,.9],[8,0,1.]):
        with pytest.raises(ValueError):build_corner_plan([0,0,0],goal)
    with pytest.raises(ValueError):build_corner_plan([0,0,0],[8,0,3],turn_radius=0)
    plan=build_corner_plan([0,0,0],[8,0,3],sample_period=100.)
    segment=plan['segments'][1];point=np.array(sample_plan(plan,segment['start_time_s']+segment['duration_s']/2)['position'])
    obstacle=dict(min=(point-.001).tolist(),max=(point+.001).tolist())
    with pytest.raises(ValueError,match='obstacle|clearance'):
        build_corner_plan([0,0,0],[8,0,3],sample_period=100.,obstacles=[obstacle],clearance=0.)
    assert build_corner_plan([0,0,0],[0,0,3])['corner_geometry']['vertical_only']
