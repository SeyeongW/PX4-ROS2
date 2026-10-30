import numpy as np
import pytest
from jol_perception.static_flight_core import build_static_plan,to_ned,to_world,valid_estimate
from jol_perception.observation_path import sample_plan, _evaluate

FRAME=dict(controller_ready=True,setpoint_frame='jol/flight_ned',origin_world=[-5,-6,.12],north_yaw_world_rad=np.pi/2)

def test_ned_origin_heading_and_roundtrip():
    np.testing.assert_allclose(to_ned([-5,-6,3.12],FRAME),[0,0,-3],atol=1e-12)
    np.testing.assert_allclose(to_ned([40,0,8],FRAME),[6,45,-7.88],atol=1e-12)
    rotated=dict(FRAME,north_yaw_world_rad=.31)
    for frame in (FRAME,rotated):
        np.testing.assert_allclose(to_world(to_ned([40,0,8],frame),frame),[40,0,8],atol=1e-12)
        np.testing.assert_allclose(to_world(to_ned([1,2,3],frame,True),frame,True),[1,2,3],atol=1e-12)

def test_three_metre_vertical_and_C2_junctions():
    for kind in ('spline','right_angle'):
        plan=build_static_plan([-5,-6,.12],[39.4,0,8],path_kind=kind)
        first=plan['segments'][0]
        np.testing.assert_allclose(first['control_points'][-1],[-5,-6,3.12])
        for segment in plan['segments'][:-1]:
            t=segment['start_time_s']+segment['duration_s']
            before,after=sample_plan(plan,t-1e-7),sample_plan(plan,t+1e-7)
            for field in ('position','velocity','acceleration'):
                np.testing.assert_allclose(before[field],after[field],atol=1e-5)
        assert plan['motion_validation']['passed']
        assert plan['motion_validation']['continuous_upper_bounds']['speed_m_s'] <= .9*plan['limits']['speed_m_s']+1e-8
        assert plan['motion_validation']['continuous_upper_bounds']['acceleration_m_s2'] <= .7*plan['limits']['acceleration_m_s2']+1e-8
        assert np.linalg.norm(np.array(plan['target_world'])-plan['goal']) == pytest.approx(1.)
        assert plan['target_surface_safety']['passed']

def test_timing_controls_and_invalid_input():
    fast=build_static_plan([-5,-6,.12],[39.4,0,8])
    slow=build_static_plan([-5,-6,.12],[39.4,0,8],min_duration=fast['duration_s']*2)
    assert slow['duration_s']==pytest.approx(fast['duration_s']*2)
    assert slow['motion_validation']['continuous_upper_bounds']['speed_m_s'] < 1
    with pytest.raises(ValueError):build_static_plan([0,0,0],[40,0,2])


@pytest.mark.parametrize('kind', ['spline', 'right_angle'])
def test_takeoff_minimum_duration_slows_only_vertical_motion_and_preserves_C2_limits(kind):
    from jol_perception.observation_path import _evaluate
    original = build_static_plan([-5,-6,.12],[39.4,0,8],path_kind=kind)
    first = original['segments'][0]
    minimum = first['duration_s']*2.
    dilated = build_static_plan([-5,-6,.12],[39.4,0,8],path_kind=kind,
                               takeoff_min_duration=minimum)
    slow = dilated['segments'][0]
    assert slow['duration_s'] == pytest.approx(minimum)
    assert dilated['duration_s']-original['duration_s'] == pytest.approx(first['duration_s'])
    for before, after in zip(original['segments'], dilated['segments']):
        assert before['control_points'] == after['control_points']
    for before, after in zip(original['segments'][1:], dilated['segments'][1:]):
        assert before['duration_s'] == after['duration_s']
        assert after['start_time_s']-before['start_time_s'] == pytest.approx(first['duration_s'])
    for fraction in np.linspace(0., 1., 31):
        before = _evaluate(first, fraction*first['duration_s'])
        after = _evaluate(slow, fraction*minimum)
        np.testing.assert_allclose(after['position'], before['position'], atol=1e-12)
        for field, power, key in [('velocity',1,'speed_m_s'),
                                  ('acceleration',2,'acceleration_m_s2'),
                                  ('jerk',3,'jerk_m_s3')]:
            np.testing.assert_allclose(after[field], np.array(before[field])/2**power, atol=1e-12)
            assert np.linalg.norm(after[field]) <= slow['continuous_bounds'][key]+1e-9
    left, right = sample_plan(dilated, minimum-1e-7), sample_plan(dilated, minimum+1e-7)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(left[field], right[field], atol=1e-5)
    assert dilated['motion_validation']['passed']

def test_reject_missing_stale_untrusted_and_uncertain_coordinates():
    packet=dict(valid=True,frame_id='world',source='camera_bbox_lidar',stamp=10.,position=[40,0,8],covariance=(np.eye(3)*.01).tolist())
    assert valid_estimate(packet,10.1)
    assert not valid_estimate({},10.1)
    assert not valid_estimate(dict(packet,stamp=8),10.1)
    assert not valid_estimate(dict(packet,source='gazebo_ground_truth'),10.1)
    assert not valid_estimate(dict(packet,covariance=np.eye(3).tolist()),10.1)

def test_close_surface_arrival_reserves_body_and_coordinate_uncertainty():
    home,target=[-5,-6,.1],[39.4,0,8]
    cov=np.diag([.0229,.008,.008])
    plan=build_static_plan(home,target,target_covariance=cov)
    guard=plan['target_surface_safety']
    assert .85 < guard['required_plane_separation_m'] < 1.
    assert guard['continuous_minimum_plane_separation_m'] >= 1.-1e-8
    with pytest.raises(ValueError,match='standoff_too_small'):
        build_static_plan(home,target,standoff=.5,target_covariance=cov)
    with pytest.raises(ValueError,match='standoff_too_small'):
        build_static_plan(home,target,target_covariance=np.eye(3)*.25)
    with pytest.raises(ValueError,match='invalid_target_covariance'):
        build_static_plan(home,target,target_covariance=np.eye(3)*-1)


def test_composed_bounds_reduce_static_duration_without_changing_curve_or_sampling_safety():
    from jol_perception.observation_path import _segment, _evaluate
    plan = build_static_plan([-5,-6,.1],[37.18,13.59,8.1],sample_period=100.)
    fine = build_static_plan([-5,-6,.1],[37.18,13.59,8.1])
    assert plan['segments'] == fine['segments']  # Three saved points still certify the whole curve.
    assert 90. < plan['duration_s'] < 105.
    assert 'bernstein' in plan['motion_validation']['method']
    legacy_duration = 0.
    for segment in plan['segments']:
        legacy = _segment(segment['control_points'],0.,segment['phase'],1.35,.42,1.2)
        legacy_duration += legacy['duration_s']
        for fraction in (.03,.19,.53,.87,.99):
            actual = _evaluate(segment,segment['start_time_s']+fraction*segment['duration_s'])
            unchanged = _evaluate(legacy,fraction*legacy['duration_s'])
            np.testing.assert_allclose(actual['position'],unchanged['position'],atol=1e-11)
        for fraction in np.linspace(0.,1.,1001):
            value = _evaluate(segment,segment['start_time_s']+fraction*segment['duration_s'])
            for field,key in [('velocity','speed_m_s'),('acceleration','acceleration_m_s2'),('jerk','jerk_m_s3')]:
                assert np.linalg.norm(value[field]) <= segment['continuous_bounds'][key]+1e-9
    assert plan['duration_s'] < .6*legacy_duration


def test_reference_tailsitter_plan_keeps_vertical_c2_join_and_records_research_limits():
    with pytest.raises(ValueError, match='validated_Gazebo_controller_range'):
        build_static_plan([-5,-6,.1],[36.99,13.52,7.98],max_speed=9.)
    plan=build_static_plan([-5,-6,.1],[36.99,13.52,7.98],controller_profile='tailsitter20cm_reference',
        max_speed=9.,max_acceleration=3.5,max_jerk=12.,takeoff_min_duration=12.,body_radius=.12)
    assert plan['controller_profile']=='tailsitter20cm_reference'
    assert 'actual VTOL motion' in plan['controller_limit_scope']
    assert plan['segments'][0]['duration_s']==12.
    np.testing.assert_allclose(plan['segments'][0]['control_points'][-1],[-5,-6,3.1])
    left=sample_plan(plan,12.-1e-6);right=sample_plan(plan,12.+1e-6)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(left[field],right[field],atol=1e-6)
    assert 7.5 < max(np.linalg.norm(row['velocity']) for row in plan['samples']) <= 9.


def test_continuous_takeoff_has_exact_three_metre_vertical_and_nonzero_c2_join():
    import json
    from jol_perception.observation_path import _evaluate
    kwargs = dict(controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
                  takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
                  max_acceleration=3.5,max_jerk=12.,body_radius=.12)
    plan = json.loads(json.dumps(build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**kwargs)))
    first,second = plan['segments']
    assert first['duration_s'] == 4.
    assert plan['continuous_takeoff']
    assert plan['motion_validation']['passed']
    at_left = _evaluate(first,4.)
    at_right = _evaluate(second,4.)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(at_left[field],at_right[field],atol=1e-12)
    np.testing.assert_allclose(at_left['position'],[-2.,0.,3.1],atol=1e-12)
    np.testing.assert_allclose(at_left['velocity'],[0.,0.,1.],atol=1e-12)
    np.testing.assert_allclose(at_left['acceleration'],[0.,0.,0.],atol=1e-12)
    for stamp in np.linspace(0.,4.,51):
        np.testing.assert_allclose(sample_plan(plan,stamp)['position'][:2],[-2.,0.],atol=1e-12)
    for segment in plan['segments']:
        for stamp in np.linspace(segment['start_time_s'],segment['start_time_s']+segment['duration_s'],503):
            sample = _evaluate(segment,stamp)
            for field,key in [('velocity','speed_m_s'),('acceleration','acceleration_m_s2'),('jerk','jerk_m_s3')]:
                assert np.linalg.norm(sample[field]) <= segment['continuous_bounds'][key]+1e-9
    for stamp in np.linspace(.01,plan['duration_s']-.01,301):
        assert np.linalg.norm(sample_plan(plan,stamp)['velocity']) > 1e-7
    np.testing.assert_allclose(sample_plan(plan,plan['duration_s'])['position'],plan['goal'],atol=1e-12)
    np.testing.assert_allclose(sample_plan(plan,plan['duration_s'])['velocity'],[0.,0.,0.])


def test_continuous_takeoff_global_duration_scaling_preserves_c2_but_rejects_incompatible_join():
    from jol_perception.observation_path import _evaluate
    kwargs = dict(controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
                  takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
                  max_acceleration=3.5,max_jerk=12.,body_radius=.12)
    original = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**kwargs)
    slow = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],min_duration=original['duration_s']*2.,**kwargs)
    assert slow['duration_s'] == pytest.approx(original['duration_s']*2.)
    assert slow['takeoff_join_speed_m_s'] == pytest.approx(.5)
    join = slow['segments'][1]['start_time_s']
    left,right = _evaluate(slow['segments'][0],join),_evaluate(slow['segments'][1],join)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(left[field],right[field],atol=1e-12)
    assert np.linalg.norm(left['velocity']) == pytest.approx(.5)
    with pytest.raises(ValueError,match='incompatible'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(kwargs,takeoff_min_duration=12.))
    with pytest.raises(ValueError,match='cannot_meet_motion_limits'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(kwargs,takeoff_join_speed=1.2))
    with pytest.raises(ValueError,match='exceeds_command_limit'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(kwargs,takeoff_join_speed=9.))
    with pytest.raises(ValueError,match='must_be_boolean'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(kwargs,continuous_takeoff='yes'))
    # The stopped right-angle comparison and the default legacy remain unchanged.
    comparison = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(kwargs,path_kind='right_angle'))
    assert not comparison['continuous_takeoff']
    assert all('time_control_points' not in segment for segment in comparison['segments'])
    broken = dict(slow['segments'][0],time_control_points=[0.,0.,0.,.9,.8,1.])
    with pytest.raises(ValueError,match='monotone'):_evaluate(broken,1.)


def test_measured_path_projection_is_interpolated_monotone_and_jump_limited():
    from jol_perception.static_flight_core import path_progress
    plan = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],controller_profile='tailsitter20cm_reference',
                            continuous_takeoff=True,takeoff_join_speed=1.,takeoff_min_duration=4.,
                            max_speed=9.,max_acceleration=3.5,max_jerk=12.,body_radius=.12)
    # Between samples, progress follows the segment projection, not the closest vertex.
    left,right = plan['samples'][29:31]
    midpoint = (np.array(left['position'])+right['position'])/2.
    result = path_progress(plan,midpoint,previous_time=1.,max_advance=2.)
    assert result['trajectory_time_s'] == pytest.approx((left['time_s']+right['time_s'])/2.)
    assert result['cross_track_distance_m'] < 1e-10
    offpath = path_progress(plan,midpoint+np.array([1.,0.,0.]),1.,2.)
    assert offpath['trajectory_time_s'] == pytest.approx(result['trajectory_time_s'])
    assert offpath['cross_track_distance_m'] == pytest.approx(1.)
    # Backwards motion cannot improve progress; a leap to the goal cannot skip the window.
    backwards = path_progress(plan,plan['home'],previous_time=2.,max_advance=.2)
    assert backwards['trajectory_time_s'] == 2.
    assert backwards['cross_track_distance_m'] > 0.
    jumped = path_progress(plan,plan['goal'],previous_time=1.,max_advance=.2)
    assert 1. <= jumped['trajectory_time_s'] <= 1.2
    assert jumped['cross_track_distance_m'] > 30.
    assert jumped['advance_capped']
    # The exact 3m line/curve seam is a real projection endpoint.
    seam = path_progress(plan,plan['geometry']['lift'],previous_time=3.8,max_advance=.5)
    assert seam['trajectory_time_s'] == pytest.approx(4.)
    assert seam['cross_track_distance_m'] < 1e-10
    frozen = path_progress(plan,plan['goal'],previous_time=1.,max_advance=0.)
    assert frozen['trajectory_time_s'] == 1.
    end = path_progress(plan,plan['goal'],previous_time=plan['duration_s']-.2,max_advance=.5)
    assert end['progress_fraction'] == 1.
    assert end['cross_track_distance_m'] < 1e-10
    with pytest.raises(ValueError):path_progress(plan,[float('nan'),0.,0.])
    with pytest.raises(ValueError):path_progress(plan,plan['goal'],-1.)
    with pytest.raises(ValueError):path_progress(plan,plan['goal'],plan['duration_s']+1.)
    with pytest.raises(ValueError):path_progress(plan,plan['goal'],max_advance=-.1)
    with pytest.raises(ValueError):path_progress(dict(plan,frame_id='jol/flight_ned'),plan['goal'])


def test_virtual_sensor_waypoint_reaches_exact_sensor_target_with_c2_and_obstacle_checks():
    from jol_perception.observation_path import _evaluate
    home,target = [-2.,0.,.1],[36.99,13.52,8.]
    kwargs = dict(controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
                  takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,max_acceleration=3.5,
                  max_jerk=12.,body_radius=.12,target_is_virtual_waypoint=True,standoff=0.)
    plan = build_static_plan(home,target,**kwargs)
    np.testing.assert_allclose(plan['goal'],target,atol=0.)
    np.testing.assert_allclose(plan['geometry']['Pg'],target,atol=0.)
    np.testing.assert_allclose(sample_plan(plan,plan['duration_s'])['position'],target,atol=1e-12)
    assert plan['arrival_policy'] == 'virtual_sensor_waypoint'
    assert plan['target_is_virtual_waypoint'] and plan['standoff_m'] == 0.
    assert plan['target_surface_safety']['plane_check_applied'] is False
    assert 'passed' not in plan['target_surface_safety']
    assert 'required_plane_separation_m' not in plan['target_surface_safety']
    assert plan['motion_validation']['passed']
    left,right = _evaluate(plan['segments'][0],4.),_evaluate(plan['segments'][1],4.)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(left[field],right[field],atol=1e-12)
    with pytest.raises(ValueError,match='intersects an obstacle'):
        build_static_plan(home,target,**kwargs,
                          obstacles=[dict(min=[-2.2,-.2,1.],max=[-1.8,.2,2.])])
    for covariance in (np.full((3,3),np.nan),np.eye(3)*-1.,np.zeros((2,2))):
        with pytest.raises(ValueError,match='invalid_target_covariance'):
            build_static_plan(home,target,**kwargs,target_covariance=covariance)
    with pytest.raises(ValueError,match='must_be_boolean'):
        build_static_plan(home,target,**dict(kwargs,target_is_virtual_waypoint=1))
    with pytest.raises(ValueError,match='standoff'):
        build_static_plan(home,target,**dict(kwargs,target_is_virtual_waypoint=False))
    # Default physical surface remains a one-metre noncontact arrival.
    ordinary = build_static_plan(home,target)
    assert ordinary['standoff_m'] == 1.
    assert ordinary['arrival_policy'] == 'noncontact_hold_before_observed_target_surface'
    assert ordinary['target_surface_safety']['passed']


@pytest.mark.parametrize('kind', ['spline','right_angle'])
def test_sensor_gate_is_interior_checkpoint_with_motion_and_hold_behind(kind):
    from jol_perception.observation_path import _evaluate
    home,target = [-2.,0.,.1],[36.99,13.52,8.]
    plan = build_static_plan(home,target,path_kind=kind,
        controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
        takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
        max_acceleration=3.5,max_jerk=12.,body_radius=.16,
        target_is_virtual_waypoint=True,target_is_pass_through_gate=True,
        gate_width=1.2,gate_height=1.2,gate_exit_distance=20.)
    gate = plan['gate']; normal = np.array(gate['normal_world'])
    np.testing.assert_allclose(gate['center_world'],target,atol=0.)
    assert (np.array(plan['goal'])-target)@normal == pytest.approx(20.)
    assert plan['goal'][2] == target[2]
    crossing = sample_plan(plan,gate['planned_crossing_time_s'])
    np.testing.assert_allclose(crossing['position'],target,atol=1e-10)
    assert 0. < gate['planned_crossing_time_s'] < plan['duration_s']
    assert np.linalg.norm(crossing['velocity']) > 1.
    assert gate['dimensions_source'] == 'experiment_configuration'
    assert gate['center_source'] == 'locked_camera_bbox_lidar_estimate'
    assert plan['arrival_policy'] == 'pass_through_sensor_gate_then_hold_behind'
    assert plan['motion_validation']['passed']
    assert plan['target_surface_safety']['plane_check_applied'] is False
    for segment in plan['segments'][:-1]:
        t = segment['start_time_s']+segment['duration_s']
        left,right = sample_plan(plan,t-1e-7),sample_plan(plan,t+1e-7)
        for field in ('position','velocity','acceleration'):
            np.testing.assert_allclose(left[field],right[field],atol=1e-5)
    if kind == 'spline':
        seam = _evaluate(plan['segments'][0],4.)
        np.testing.assert_allclose(seam['velocity'],[0.,0.,1.],atol=1e-12)
        assert gate['solved_vertical_handle_m'] > 2.
    end = sample_plan(plan,plan['duration_s'])
    np.testing.assert_allclose(end['position'],plan['goal'],atol=1e-12)
    np.testing.assert_allclose(end['velocity'],[0.,0.,0.],atol=0.)


def test_gate_rejects_physical_target_too_small_rectangle_and_keeps_fixed_obstacle_guard():
    options = dict(controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
                   takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
                   max_acceleration=3.5,max_jerk=12.,body_radius=.16,
                   target_is_virtual_waypoint=True,target_is_pass_through_gate=True)
    with pytest.raises(ValueError,match='requires_virtual_target'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],
                          **dict(options,target_is_virtual_waypoint=False))
    with pytest.raises(ValueError,match='too_small'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(options,gate_width=.3))
    with pytest.raises(ValueError,match='boolean'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(options,target_is_pass_through_gate=1))
    plan = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**options)
    # The pass-through marker is virtual, but a real obstacle behind it still
    # invalidates the path instead of being silently ignored.
    hold = np.array(plan['goal'])
    with pytest.raises(ValueError,match='intersects an obstacle'):
        build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**options,
            obstacles=[dict(min=(hold-.2).tolist(),max=(hold+.2).tolist())])


def test_checkpoint_requires_actual_forward_intersection_inside_body_margin():
    from jol_perception.static_flight_core import checkpoint_crossing
    gate = dict(center_world=[4.,5.,8.],normal_world=[1.,0.,0.],
                right_world=[0.,1.,0.],up_world=[0.,0.,1.],
                width_m=1.2,height_m=1.2,body_radius_m=.16)
    passed = checkpoint_crossing(gate,[3.5,5.2,8.1],[4.5,5.4,8.3])
    assert passed['valid'] and passed['crossed'] and passed['passed']
    np.testing.assert_allclose(passed['intersection_world'],[4.,5.3,8.2],atol=1e-12)
    assert passed['fraction'] == .5
    for before,after in (([3.9,5.,8.],[3.99,5.,8.]), # close, still in front
                         ([4.1,5.,8.],[4.2,5.,8.]),  # initialized behind
                         ([4.5,5.,8.],[3.5,5.,8.]),  # backwards
                         ([4.,5.,8.],[4.1,5.,8.])):  # touching only, no front sample
        result = checkpoint_crossing(gate,before,after)
        assert result['valid'] and not result['crossed'] and not result['passed']
    lateral_miss = checkpoint_crossing(gate,[3.5,5.5,8.],[4.5,5.5,8.])
    assert lateral_miss['crossed'] and lateral_miss['missed'] and not lateral_miss['passed']
    vertical_miss = checkpoint_crossing(gate,[3.5,5.,8.5],[4.5,5.,8.5])
    assert vertical_miss['crossed'] and vertical_miss['missed']
    jumped = checkpoint_crossing(gate,[1.,5.,8.],[7.,5.,8.])
    assert not jumped['valid'] and not jumped['passed']
    assert jumped['reason'] == 'measured_position_jump'
    with pytest.raises(ValueError,match='basis'):
        checkpoint_crossing(dict(gate,up_world=[1.,0.,0.]),[3.5,5.,8.],[4.5,5.,8.])


def test_tailsitter_dynamics_rejects_force_attitude_rate_and_braking_infeasibility():
    import copy
    import json
    from pathlib import Path
    from jol_perception.static_flight_core import validate_tailsitter_plan_dynamics
    profile = json.loads((Path(__file__).parents[2]/'jol_sim/config/tailsitter_20cm_reference.json').read_text())
    # Pin this numerical case; tuning the runtime transition duration must not
    # silently change the expected braking-distance regression.
    profile['flight_parameters']['back_transition_duration_s'] = .45
    profile['px4_parameters']['VT_B_TRANS_DUR'] = .45
    profile['px4_parameters']['VT_TS_SMOOTH'] = 0
    profile['px4_parameters']['FW_AIRSPD_TRIM'] = 7.4
    options = dict(controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
        takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
        max_acceleration=3.5,max_jerk=12.,body_radius=.16,
        target_is_virtual_waypoint=True,target_is_pass_through_gate=True)
    plan = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**options)
    result = validate_tailsitter_plan_dynamics(plan,profile)
    assert result['passed'] and plan['dynamics_validation'] == result
    assert result['data_status'] == 'reference_scaled_not_measured'
    assert not result['full_vtol_feasibility_certified']
    assert not result['fixed_wing']['stall_margin_1_3_satisfied']
    assert 17. < result['post_gate_reserve']['required_distance_m'] < 17.2
    # A sparse saved preview cannot hide continuous acceleration/jerk peaks.
    sparse = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],sample_period=100.,**options)
    assert validate_tailsitter_plan_dynamics(sparse,profile)['nominal_mc'] == result['nominal_mc']
    for key,value,reason in [('MPC_THR_MAX',.1,'mc_maximum_thrust'),
                             ('MPC_TILTMAX_AIR',.5,'mc_nominal_tilt'),
                             ('MC_ROLLRATE_MAX',.1,'mc_nominal_thrust_axis_rate')]:
        changed = copy.deepcopy(profile); changed['px4_parameters'][key] = value
        with pytest.raises(ValueError,match=reason):
            validate_tailsitter_plan_dynamics(plan,changed)
        assert not plan['dynamics_validation']['passed']
    short = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],**dict(options,gate_exit_distance=2.))
    with pytest.raises(ValueError,match='post_gate_transition_and_jerk_limited_stop_distance'):
        validate_tailsitter_plan_dynamics(short,profile)
    assert not validate_tailsitter_plan_dynamics(short,profile,strict=False)['passed']
    uncovered = copy.deepcopy(profile); uncovered['motor']['max_rpm'] *= 2.
    with pytest.raises(ValueError,match='motor_data_does_not_cover'):
        validate_tailsitter_plan_dynamics(plan,uncovered)
    smoothed = copy.deepcopy(profile); smoothed['px4_parameters']['VT_TS_SMOOTH'] = 1
    with pytest.raises(ValueError,match='back_transition_reference_rate'):
        validate_tailsitter_plan_dynamics(plan,smoothed)  # .45s easing would demand250deg/s.
    smoothed['flight_parameters']['back_transition_duration_s'] = .55
    smoothed['px4_parameters']['VT_B_TRANS_DUR'] = .55
    approved = validate_tailsitter_plan_dynamics(plan,smoothed)
    assert approved['transition_reference']['back_reference_peak_rad_s'] < np.deg2rad(220.)


OPTIONS = dict(controller_profile='tailsitter20cm_reference',continuous_takeoff=True,
    takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,max_acceleration=3.5,
    max_jerk=12.,body_radius=.16,target_is_virtual_waypoint=True,
    target_is_pass_through_gate=True,gate_exit_distance=20.)
HOME,TARGET = [-2.,0.,.1],[36.99,13.52,8.]


def plan(**options):
    return build_static_plan(HOME,TARGET,**dict(OPTIONS,**options))


def test_reference_takeoff_is_faster_without_changing_cubic_or_c2_motion_limits():
    import json
    from pathlib import Path
    from jol_perception.static_flight_core import validate_tailsitter_plan_dynamics
    profile = json.loads((Path(__file__).parents[2]/'jol_sim/config/tailsitter_20cm_reference.json').read_text())
    options = dict(OPTIONS, **{key:value for key,value in profile['flight_parameters'].items() if key in OPTIONS})
    # Isolate the historical 4s -> 2.5s floor change from later join-speed tuning.
    options['takeoff_join_speed'] = 1.
    # The target starts to the camera's right, rather than already centered.
    target = [0.,-40.,8.]
    faster = build_static_plan(HOME,target,**options)
    previous = build_static_plan(HOME,target,**dict(options,takeoff_min_duration=4.))
    first,second = faster['segments']
    assert first['duration_s'] == pytest.approx(2.5)
    assert previous['duration_s']-faster['duration_s'] == pytest.approx(1.5)
    assert faster['limits'] == previous['limits']
    for old,new in zip(previous['segments'],faster['segments']):
        assert new['control_points'] == old['control_points']
    assert second['duration_s'] == previous['segments'][1]['duration_s']
    assert second['time_control_points'] == previous['segments'][1]['time_control_points']
    assert sample_plan(faster,1.)['position'][2]-HOME[2] > .6
    assert sample_plan(previous,1.)['position'][2]-HOME[2] < .2
    boundary = first['duration_s']
    left,right = _evaluate(first,boundary),_evaluate(second,boundary)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(left[field],right[field],atol=1e-12)
    np.testing.assert_allclose(left['position'],np.array(HOME)+[0.,0.,3.],atol=1e-12)
    np.testing.assert_allclose(left['velocity'],[0.,0.,1.],atol=1e-12)
    np.testing.assert_allclose(left['acceleration'],[0.,0.,0.],atol=1e-12)
    for time in np.linspace(0.,boundary,101):
        sample = _evaluate(first,time)
        np.testing.assert_allclose(sample['position'][:2],HOME[:2],atol=1e-12)
        for field,key in [('velocity','speed_m_s'),('acceleration','acceleration_m_s2'),('jerk','jerk_m_s3')]:
            assert np.linalg.norm(sample[field]) <= first['continuous_bounds'][key]+1e-9
    assert faster['motion_validation']['passed']
    assert validate_tailsitter_plan_dynamics(faster,profile)['passed']


def test_slightly_faster_vertical_join_improves_early_forward_motion_with_same_limits():
    import json
    from pathlib import Path
    from jol_perception.static_flight_core import validate_tailsitter_plan_dynamics
    profile = json.loads((Path(__file__).parents[2]/'jol_sim/config/tailsitter_20cm_reference.json').read_text())
    options = dict(OPTIONS, **{key:value for key,value in profile['flight_parameters'].items() if key in OPTIONS})
    target = [0.,-40.,8.]
    options['target_is_pass_through_gate']=True  # Explicit legacy gate comparison, independent of the NUB point-goal profile.
    baseline = build_static_plan(HOME,target,**dict(options,takeoff_join_speed=1.))
    candidate = build_static_plan(HOME,target,**dict(options,takeoff_join_speed=1.1))
    assert candidate['limits'] == baseline['limits']
    assert candidate['command_limits'] == baseline['command_limits']
    assert candidate['goal'] == baseline['goal']
    for old,new in zip(baseline['segments'],candidate['segments']):
        assert new['control_points'] == old['control_points']
    first,second = candidate['segments']
    boundary = first['duration_s']
    assert boundary == pytest.approx(2.5)
    left,right = _evaluate(first,boundary),_evaluate(second,boundary)
    for field in ('position','velocity','acceleration'):
        np.testing.assert_allclose(left[field],right[field],atol=1e-12)
    np.testing.assert_allclose(left['position'],np.array(HOME)+[0.,0.,3.],atol=1e-12)
    np.testing.assert_allclose(left['velocity'],[0.,0.,1.1],atol=1e-12)
    for time in np.linspace(0.,boundary,101):
        np.testing.assert_allclose(_evaluate(first,time)['position'][:2],HOME[:2],atol=1e-12)
    assert candidate['motion_validation']['passed']
    assert validate_tailsitter_plan_dynamics(candidate,profile)['passed']
    for segment in candidate['segments']:
        for key,bound in segment['continuous_bounds'].items():
            assert bound <= candidate['command_limits'][key]+1e-9
    baseline_speed = np.linalg.norm(sample_plan(baseline,5.)['velocity'][:2])
    candidate_speed = np.linalg.norm(sample_plan(candidate,5.)['velocity'][:2])
    assert candidate_speed >= 1.2*baseline_speed
    assert candidate['gate']['planned_crossing_time_s'] < baseline['gate']['planned_crossing_time_s']-.4


def test_gate_split_retimes_the_same_cubic_and_speeds_up_without_relaxing_limits():
    original = plan()
    optimized = plan(gate_crossing_speed=7.2,gate_crossing_tangential_acceleration=-.5)
    assert len(original['segments']) == 2 and len(optimized['segments']) == 3
    assert original['segments'][0] == optimized['segments'][0]
    assert optimized['duration_s'] < .9*original['duration_s']
    assert optimized['limits'] == original['limits']
    assert optimized['goal'] == original['goal']
    split = optimized['gate']['cubic_parameter_at_center']
    points = np.array(original['segments'][1]['control_points'])
    for part,low,high in ((optimized['segments'][1],0.,split),
                          (optimized['segments'][2],split,1.)):
        controls = np.array(part['control_points'])
        for u in np.linspace(0.,1.,31):
            exact = lambda p,q: (1-q)**3*p[0]+3*(1-q)**2*q*p[1]+3*(1-q)*q*q*p[2]+q**3*p[3]
            np.testing.assert_allclose(exact(controls,u),exact(points,low+(high-low)*u),atol=3e-14)
    crossing = sample_plan(optimized,optimized['gate']['planned_crossing_time_s'])
    np.testing.assert_allclose(crossing['position'],TARGET,atol=1e-11)
    velocity,acceleration = np.array(crossing['velocity']),np.array(crossing['acceleration'])
    assert np.linalg.norm(velocity) == pytest.approx(7.2,abs=1e-10)
    assert velocity@acceleration/np.linalg.norm(velocity) == pytest.approx(-.5,abs=1e-10)


def test_gate_split_keeps_nonzero_C2_joins_and_bounds_the_whole_flight():
    optimized = plan(gate_crossing_speed=7.2,gate_crossing_tangential_acceleration=-.5)
    for first,second in zip(optimized['segments'],optimized['segments'][1:]):
        time = second['start_time_s']
        left,right = _evaluate(first,time),_evaluate(second,time)
        for field in ('position','velocity','acceleration'):
            np.testing.assert_allclose(left[field],right[field],atol=2e-13)
        assert np.linalg.norm(left['velocity']) > .9
    for segment in optimized['segments']:
        assert np.all(np.diff(segment['time_control_points']) >= -1e-12)
        for time in np.linspace(segment['start_time_s'],segment['start_time_s']+segment['duration_s'],301):
            value = _evaluate(segment,time)
            for field,key in [('velocity','speed_m_s'),('acceleration','acceleration_m_s2'),('jerk','jerk_m_s3')]:
                assert np.linalg.norm(value[field]) <= segment['continuous_bounds'][key]+1e-9
    assert optimized['motion_validation']['passed']
    assert optimized['motion_validation']['continuous_upper_bounds']['speed_m_s'] <= 8.1+1e-8
    assert optimized['motion_validation']['continuous_upper_bounds']['acceleration_m_s2'] <= 2.45+1e-8
    assert optimized['motion_validation']['continuous_upper_bounds']['jerk_m_s3'] <= 12.+1e-8
    final = sample_plan(optimized,optimized['duration_s'])
    np.testing.assert_allclose(final['position'],optimized['goal'],atol=1e-12)
    np.testing.assert_allclose(final['velocity'],[0.,0.,0.],atol=0.)
    np.testing.assert_allclose(final['acceleration'],[0.,0.,0.],atol=0.)


@pytest.mark.parametrize('boundary',[
    dict(gate_crossing_speed=9.),dict(gate_crossing_speed=True),
    dict(gate_crossing_speed=-1.),dict(gate_crossing_speed=7.2,gate_crossing_tangential_acceleration=3.),
    dict(gate_crossing_speed=7.2,gate_crossing_tangential_acceleration=float('nan')),
    dict(gate_crossing_speed=7.2,gate_crossing_tangential_acceleration=True)])
def test_gate_split_rejects_invalid_boundaries(boundary):
    with pytest.raises(ValueError):plan(**boundary)


def test_gate_split_global_slowdown_preserves_C2_and_right_angle_comparison():
    optimized = plan(gate_crossing_speed=7.2)
    slow = plan(gate_crossing_speed=7.2,min_duration=2*optimized['duration_s'])
    for first,second in zip(slow['segments'],slow['segments'][1:]):
        for field in ('position','velocity','acceleration'):
            np.testing.assert_allclose(_evaluate(first,second['start_time_s'])[field],
                                       _evaluate(second,second['start_time_s'])[field],atol=2e-13)
    assert slow['gate']['planned_crossing_speed_m_s'] == pytest.approx(3.6)
    assert plan(path_kind='right_angle')['segments'] == plan(path_kind='right_angle',gate_crossing_speed=7.2)['segments']
