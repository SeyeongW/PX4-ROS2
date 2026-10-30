"""Static sensor waypoint mission geometry; no Gazebo ground-truth dependency."""
import math
import numpy as np
from .observation_path import _xyz, _positive, _segment, _finish_plan, sample_plan, _composed_derivative_bounds


def flight_frame(controller):
    if controller.get('controller_ready') is not True or controller.get('setpoint_frame') != 'jol/flight_ned':
        raise ValueError('NED_controller_not_ready')
    origin = _xyz(controller['origin_world'], 'controller origin')
    yaw = float(controller['north_yaw_world_rad'])
    if not math.isfinite(yaw): raise ValueError('invalid_north_reference')
    # North is an explicitly configured direction in Gazebo world, not vehicle yaw.
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c,s,0.], [s,-c,0.], [0.,0.,-1.]])
    return origin, rotation


def to_ned(position, controller, vector=False):
    origin, rotation = flight_frame(controller)
    return rotation @ (_xyz(position, 'world XYZ') - (0. if vector else origin))


def to_world(position, controller, vector=False):
    origin, rotation = flight_frame(controller)
    return rotation.T @ _xyz(position, 'NED XYZ') + (0. if vector else origin)


def validate_controller_envelope(plan, controller):
    origin, _ = flight_frame(controller)
    horizontal = _positive(controller['horizontal_limit_m'], 'controller horizontal limit')-.5
    ceiling = _positive(controller['height_limit_m'], 'controller height limit')-.5
    # The cylinder is convex: the Bezier control hull bounds the whole curve,
    # including a braking curve between samples.
    for segment in plan['segments']:
        for point in segment['control_points']:
            xyz = _xyz(point, 'controller path point')
            if np.linalg.norm(xyz[:2]-origin[:2]) > horizontal or not .05 <= xyz[2] <= ceiling:
                raise ValueError('path_outside_controller_envelope')


def validate_tailsitter_plan_dynamics(plan, profile, *, strict=True):
    """Reject impossible nominal inputs; this is not a VTOL transition proof.

    GCOPTER checks thrust, tilt and angular rate after its multicopter flatness
    map. Here continuous A/J hulls give conservative *nominal MC* bounds for
    F=m(a+g ez), tilt<=asin(A/g), |z_dot|<=J/(g-A). They do not model wing lift,
    propwash or the 90-degree transition. Those require recorded SITL evidence.
    """
    g = 9.80665
    mass = _positive(profile['mass_kg'],'aircraft mass')
    motor = profile['motor']; params = profile['px4_parameters']; flight = profile['flight_parameters']
    rpm = _positive(motor['max_rpm'],'maximum RPM')
    rows = [_xyz(row,'RPM thrust torque sample') for row in motor['rpm_thrust_torque_samples']]
    if (not rows or any(row[0] <= 0 or row[1] <= 0 or row[2] < 0 for row in rows)
            or rpm > max(row[0] for row in rows) or rpm < min(row[0] for row in rows)):
        raise ValueError('tailsitter_dynamics_motor_data_does_not_cover_maximum_RPM')
    axes = np.asarray(profile['rotor_axes_body'],dtype=float)
    if axes.shape != (4,3) or not np.allclose(axes,[[0.,0.,1.]]*4,atol=1e-9):
        raise ValueError('tailsitter_dynamics_requires_four_parallel_body_Z_rotors')
    # The lower covered bench point underestimates the declared maximum when
    # it falls between samples; do not extrapolate an unmeasured motor curve.
    total_thrust = 4*max(row[1] for row in rows if row[0] <= rpm)
    minimum_throttle = _positive(params.get('MPC_THR_MIN',.12),'MC minimum thrust',allow_zero=True)
    maximum_throttle = _positive(params.get('MPC_THR_MAX',1.),'MC maximum thrust')
    if not 0 <= minimum_throttle < maximum_throttle <= 1:
        raise ValueError('tailsitter_dynamics_invalid_MC_thrust_limits')
    bounds = plan['motion_validation']['continuous_upper_bounds']
    acceleration = _positive(bounds['acceleration_m_s2'],'continuous acceleration',allow_zero=True)
    jerk = _positive(bounds['jerk_m_s3'],'continuous jerk',allow_zero=True)
    tilt_limit = math.radians(_positive(params.get('MPC_TILTMAX_AIR',45.),'MC tilt limit'))
    rate_limit = math.radians(min(_positive(params.get(key,220.),key)
                                  for key in ('MC_ROLLRATE_MAX','MC_PITCHRATE_MAX')))
    tilt = math.asin(acceleration/g) if acceleration < g else math.pi
    rate = jerk/(g-acceleration) if acceleration < g else math.inf
    required_min,required_max = mass*(g-acceleration),mass*(g+acceleration)
    checks = dict(mc_positive_vertical_force=acceleration < g,
                  mc_minimum_thrust=required_min >= total_thrust*minimum_throttle,
                  mc_maximum_thrust=required_max <= total_thrust*maximum_throttle,
                  mc_nominal_tilt=tilt <= tilt_limit,
                  mc_nominal_thrust_axis_rate=rate <= rate_limit)
    rate_factor = 1.25 if params.get('VT_TS_SMOOTH',0) else 1.
    transition_rates = {}
    for phase,parameter in (('front','VT_F_TRANS_DUR'),('back','VT_B_TRANS_DUR')):
        duration = _positive(flight[phase+'_transition_duration_s'],phase+' transition duration')
        native_duration = _positive(params.get(parameter,duration),parameter)
        checks[phase+'_transition_duration_matches_firmware'] = math.isclose(duration,native_duration,abs_tol=1e-6)
        transition_rates[phase+'_reference_peak_rad_s'] = math.pi*.5*rate_factor/max(native_duration,.1)
        checks[phase+'_transition_reference_rate'] = transition_rates[phase+'_reference_peak_rad_s'] <= rate_limit
    aero = profile['aerodynamics']
    stall = math.sqrt(2*mass*g/(_positive(profile['air_density_kg_m3'],'air density')
        *_positive(aero['wing_area_m2'],'wing area')*_positive(aero['CL_max'],'maximum lift coefficient')))
    airspeed_min = _positive(flight['fw_min_airspeed_m_s'],'FW minimum airspeed')
    cruise = _positive(params['FW_AIRSPD_TRIM'],'FW trim airspeed')
    airspeed_max = _positive(params['FW_AIRSPD_MAX'],'FW maximum airspeed')
    checks.update(fw_minimum_above_model_stall=airspeed_min >= stall,
                  fw_cruise_within_airspeed_limits=airspeed_min <= cruise <= airspeed_max)
    gate = plan.get('gate'); reserve = None
    if gate:
        # Symmetric jerk-limited braking: triangular acceleration at low v,
        # otherwise trapezoidal. Native back-transition consumes distance too.
        # Use the allowed speed, not the slower nominal curve or pitot reading.
        speed = max(_positive(plan['limits']['speed_m_s'],'reserve speed'),cruise)
        limit_a = _positive(plan['limits']['acceleration_m_s2'],'braking acceleration')
        limit_j = _positive(plan['limits']['jerk_m_s3'],'braking jerk')
        stop_time = (2*math.sqrt(speed/limit_j) if speed <= limit_a**2/limit_j
                     else speed/limit_a+limit_a/limit_j)
        transition_distance = speed*_positive(flight['back_transition_duration_s'],'back transition duration')
        required = speed*stop_time/2+transition_distance+_positive(gate['body_radius_m'],'body radius')
        available = _positive(gate['exit_distance_m'],'gate exit distance')
        checks['post_gate_transition_and_jerk_limited_stop_distance'] = available >= required
        reserve = dict(speed_m_s=speed,braking_time_s=stop_time,
            jerk_limited_stopping_distance_m=speed*stop_time/2,
            transition_distance_m=transition_distance,required_distance_m=required,
            available_distance_m=available,assumptions='No wind; actual speed must stay within the reserve speed. Transition wing-lift transient is not bounded here.')
    result = dict(passed=all(checks.values()),checks={key:bool(value) for key,value in checks.items()},
        data_status=profile.get('data_status','not_declared'),
        nominal_mc=dict(required_thrust_bounds_N=[required_min,required_max],
            available_thrust_bounds_N=[total_thrust*minimum_throttle,total_thrust*maximum_throttle],
            tilt_bound_deg=math.degrees(tilt),tilt_limit_deg=math.degrees(tilt_limit),
            thrust_axis_rate_bound_rad_s=rate,roll_pitch_rate_limit_rad_s=rate_limit,
            scope='Zero-aerodynamic-force nominal MC command map; yaw rate and fixed-wing attitude are excluded.'),
        fixed_wing=dict(model_stall_speed_m_s=stall,minimum_airspeed_m_s=airspeed_min,
            trim_airspeed_m_s=cruise,stall_margin_1_3_satisfied=cruise >= 1.3*stall,
            planned_gate_speed_is_not_actual_airspeed=True),
        post_gate_reserve=reserve,full_vtol_feasibility_certified=False,
        transition_reference=dict(transition_rates,shape=('integrated_quintic_rate_ramps_each_20percent'
            if params.get('VT_TS_SMOOTH',0) else 'native_constant_angle_rate'),
            roll_pitch_rate_limit_rad_s=rate_limit),
        limitations=['Wing lift, aerodynamic pitch torque, propwash and attitude transitions require actual SITL validation.',
                     'Passing nominal input checks does not certify actual acceleration, jerk or tracking error.'],
        references=['https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/gcopter/include/gcopter/gcopter.hpp',
                    'https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/bspline_opt/src/uniform_bspline.cpp'])
    plan['dynamics_validation'] = result
    if strict and not result['passed']:
        raise ValueError('tailsitter_dynamics_'+','.join(key for key,value in checks.items() if not value))
    return result


def valid_estimate(estimate, now, max_age=.8):
    try:
        if (estimate.get('valid') is not True or estimate.get('frame_id') != 'world'
                or estimate.get('source') not in ('camera_bbox_lidar', 'camera_monocular_size')
                or not 0 <= now-float(estimate['stamp']) <= max_age):
            return False
        _xyz(estimate['position'], 'target XYZ')
        cov = np.asarray(estimate['covariance'], dtype=float).reshape(3,3)
        return bool(np.isfinite(cov).all() and np.allclose(cov,cov.T)
                    and np.min(np.linalg.eigvalsh(cov)) >= -1e-10
                    and np.max(np.diag(cov)) <= .25)
    except (KeyError,ValueError,TypeError): return False


def _target_covariance(covariance):
    covariance = np.eye(3)*.15**2 if covariance is None else np.asarray(covariance,dtype=float)
    if (covariance.shape != (3,3) or not np.isfinite(covariance).all()
            or not np.allclose(covariance,covariance.T)
            or np.min(np.linalg.eigvalsh(covariance)) < -1e-10):
        raise ValueError('invalid_target_covariance')
    return covariance


def target_surface_safety(home, target, covariance, standoff, body_radius=.36):
    home, target = _xyz(home,'home'), _xyz(target,'sensor target')
    standoff = _positive(standoff,'standoff')
    horizontal = target[:2]-home[:2]; distance = np.linalg.norm(horizontal)
    if distance <= standoff+2:raise ValueError('target_too_close_for_standoff')
    direction = np.r_[horizontal/distance,0.]
    covariance = _target_covariance(covariance)
    depth_margin = 2*math.sqrt(max(0.,float(direction@covariance@direction)))
    body_radius=_positive(body_radius,'body radius')
    margin = body_radius+.20+depth_margin
    if standoff <= margin:raise ValueError('standoff_too_small_for_vehicle_and_sensor_uncertainty')
    return direction, dict(direction_world=direction.tolist(),body_radius_m=body_radius,surface_clearance_m=.20,
        two_sigma_depth_margin_m=depth_margin,required_plane_separation_m=margin)


def _moving_segment(points, start, phase, speed, acceleration, jerk, entry_speed=0.,
                    exit_speed=0., minimum_duration=0., *,
                    entry_tangential_acceleration=None, exit_tangential_acceleration=None):
    """Quintic Hermite timing with optional endpoint tangential acceleration.

    For arc speed v=|B'|u_dot, tangential acceleration is
    a_t=|B'|u_ddot+(B' dot B'')u_dot**2/|B'|. Matching it across
    subdivisions of the same cubic preserves the full world acceleration.
    """
    points = np.asarray(points, dtype=float)
    entry_rate = entry_speed/(3*np.linalg.norm(points[1]-points[0])) if entry_speed else 0.
    exit_rate = exit_speed/(3*np.linalg.norm(points[-1]-points[-2])) if exit_speed else 0.
    constrained_acceleration = (entry_tangential_acceleration is not None
                                or exit_tangential_acceleration is not None)
    endpoint_acceleration = _xyz([
        0. if entry_tangential_acceleration is None else entry_tangential_acceleration,
        0. if exit_tangential_acceleration is None else exit_tangential_acceleration, 0.],
        'endpoint tangential acceleration')
    parameter_acceleration = []
    for index,rate in ((0,entry_rate),(1,exit_rate)):
        derivative = 3*(points[1]-points[0] if index == 0 else points[-1]-points[-2])
        second = 6*(points[2]-2*points[1]+points[0] if index == 0
                    else points[-1]-2*points[-2]+points[-3])
        norm = float(np.linalg.norm(derivative))
        parameter_acceleration.append((endpoint_acceleration[index]-rate**2*float(derivative@second)/norm)/norm
                                      if constrained_acceleration else 0.)
    # Monotone Bernstein timing controls certify u stays in [0,1]; the
    # geometric collision hull remains valid for every instant.
    maximum = math.inf if constrained_acceleration else 2.5/(entry_rate+exit_rate)
    duration = max(minimum_duration, np.linalg.norm(points[-1]-points[0])/speed, .1)
    if duration > maximum:
        raise ValueError('continuous_join_speed_and_minimum_duration_incompatible')
    def candidate(duration):
        m0,m1 = entry_rate*duration,exit_rate*duration
        timing = np.array([0.,m0/5.,2*m0/5.,1.-2*m1/5.,1.-m1/5.,1.])
        timing[2] += parameter_acceleration[0]*duration**2/20.
        timing[3] += parameter_acceleration[1]*duration**2/20.
        if np.any(np.diff(timing)<-1e-12):
            raise ValueError('continuous_join_speed_cannot_meet_motion_limits')
        numerators = _composed_derivative_bounds(points,timing)
        bounds = np.asarray(numerators)/duration**np.array([1.,2.,3.])
        ratio = max(bounds[0]/speed,math.sqrt(bounds[1]/acceleration),np.cbrt(bounds[2]/jerk))
        return timing,bounds,ratio
    last_failed_duration = None
    while True:
        timing,bounds,ratio = candidate(duration)
        if ratio <= 1.+1e-12:
            if constrained_acceleration and last_failed_duration is not None:
                # Refine only this feasible bracket; every retained candidate
                # still has continuous bounds and monotone timing controls.
                low,high = last_failed_duration,duration
                for _ in range(16):
                    middle = (low+high)/2.
                    proposed = candidate(middle)
                    if proposed[2] <= 1.+1e-12:
                        high = middle;timing,bounds,_ = proposed
                    else:low = middle
                duration = high
            return dict(phase=phase,start_time_s=start,duration_s=float(duration),
                        control_points=points.tolist(),time_control_points=timing.tolist(),
                        bounds_method='composed_degree15_bernstein_derivative_hull_six_subdivisions',
                        continuous_bounds=dict(zip(('speed_m_s','acceleration_m_s2','jerk_m_s3'),
                                                   map(float,bounds))))
        if duration >= maximum:
            raise ValueError('continuous_join_speed_cannot_meet_motion_limits')
        last_failed_duration = duration
        duration = min(maximum,duration*max(1.06,min(1.25,float(ratio)*1.01)))


def _split_cubic(points, parameter):
    """De Casteljau subdivision preserves exactly the original cubic geometry."""
    points = np.asarray(points,dtype=float)
    left,right = [points[0]],[points[-1]]
    while len(points)>1:
        points = (1.-parameter)*points[:-1]+parameter*points[1:]
        left.append(points[0]);right.append(points[-1])
    return np.asarray(left),np.asarray(right[::-1])


def build_static_plan(home, target, *, path_kind='spline', takeoff_height=3., standoff=1.,
                      vertical_handle=2., approach_handle=8., min_duration=0.,
                      max_speed=1.5, max_acceleration=.6, max_jerk=1.2,
                      obstacles=(), clearance=.75, sample_period=.05, target_covariance=None, body_radius=.36,
                      takeoff_min_duration=0., controller_profile='legacy',
                      continuous_takeoff=False, takeoff_join_speed=.8,
                      target_is_virtual_waypoint=False, target_is_pass_through_gate=False,
                      gate_width=1.2, gate_height=1.2, gate_exit_distance=20.,
                      gate_crossing_speed=None, gate_crossing_tangential_acceleration=-.5):
    """Exactly 3m vertical first; cubic transfer, quintic timing, C2 stopped joins.

    With continuous_takeoff the straight 3m ascent joins the cubic at nonzero
    vertical speed and zero acceleration; no stopped sample separates them.
    A virtual waypoint uses the sensor target itself without a standoff plane;
    callers must exclude only its display marker from physical collision.
    A virtual pass-through gate is an interior checkpoint, never the stopped
    endpoint. Its rectangle dimensions are experiment configuration, while its
    center is the locked camera/LiDAR estimate. The hold endpoint is behind it.
    `right_angle` uses the same speed/acceleration/jerk bounds and safety envelope,
    but stops at the vertical/horizontal corner instead of rounding it.
    """
    home, target = _xyz(home,'home'), _xyz(target,'sensor target')
    height = _positive(takeoff_height,'takeoff height')
    if not isinstance(target_is_virtual_waypoint,bool):raise ValueError('target_is_virtual_waypoint_must_be_boolean')
    if not isinstance(target_is_pass_through_gate,bool):raise ValueError('target_is_pass_through_gate_must_be_boolean')
    if target_is_pass_through_gate and not target_is_virtual_waypoint:
        raise ValueError('pass_through_requires_virtual_target_marker')
    standoff = _positive(standoff,'standoff',allow_zero=target_is_virtual_waypoint)
    if target_is_virtual_waypoint:standoff=0.
    speed = _positive(max_speed,'max speed'); accel = _positive(max_acceleration,'max acceleration')
    jerk = _positive(max_jerk,'max jerk'); period = _positive(sample_period,'sample period')
    if controller_profile not in ('legacy', 'tailsitter20cm_reference'):
        raise ValueError('unknown_static_controller_profile')
    if controller_profile == 'legacy' and (speed>1.8 or accel>.7 or jerk>1.4):
        raise ValueError('limits_exceed_validated_Gazebo_controller_range_1.8mps_0.7mps2_1.4mps3')
    if controller_profile == 'tailsitter20cm_reference' and (speed>12 or accel>6 or jerk>20):
        raise ValueError('limits_exceed_tailsitter_research_envelope_12mps_6mps2_20mps3')
    clearance = _positive(clearance,'clearance',allow_zero=True)
    minimum = _positive(min_duration,'minimum duration',allow_zero=True)
    if not isinstance(continuous_takeoff,bool):raise ValueError('continuous_takeoff_must_be_boolean')
    continuous = continuous_takeoff and path_kind == 'spline'
    join_speed = _positive(takeoff_join_speed,'takeoff join speed') if continuous else 0.
    if join_speed > speed*.9:raise ValueError('takeoff_join_speed_exceeds_command_limit')
    # Bound the selected physics body, including the PX4 proxy rotors. Stay on
    # the observed approach side; do not invent a hidden box center.
    distance = np.linalg.norm(target[:2]-home[:2])
    if target_is_virtual_waypoint:
        _target_covariance(target_covariance)
        body_radius = _positive(body_radius,'body radius')
        direction = np.r_[(target[:2]-home[:2])/distance,0.] if distance else np.zeros(3)
        safety = dict(plane_check_applied=False,covariance_validated=True,body_radius_m=body_radius,
                      scope='Virtual sensor waypoint; display-marker collision is excluded, fixed obstacles remain checked.')
    else:
        direction, safety = target_surface_safety(home,target,target_covariance,standoff,body_radius)
    goal = target-standoff*direction
    lift = home+np.array([0.,0.,height])
    if goal[2] <= lift[2]+.2: raise ValueError('goal_must_be_above_3m_takeoff')
    gate = None
    if target_is_pass_through_gate:
        width = _positive(gate_width,'gate width'); gate_height = _positive(gate_height,'gate height')
        exit_distance = _positive(gate_exit_distance,'gate exit distance')
        if distance <= 2.:raise ValueError('gate_too_close_for_vertical_takeoff')
        if min(width,gate_height) <= 2*body_radius:
            raise ValueError('gate_too_small_for_vehicle_body')
        goal = target+exit_distance*direction
        gate = dict(center_world=target.tolist(),normal_world=direction.tolist(),
                    right_world=np.cross([0.,0.,1.],direction).tolist(),up_world=[0.,0.,1.],
                    width_m=width,height_m=gate_height,body_radius_m=body_radius,
                    exit_distance_m=exit_distance,
                    center_source='locked_camera_bbox_lidar_estimate',
                    orientation_source='horizontal_takeoff_to_locked_sensor_target',
                    dimensions_source='experiment_configuration',
                    crossing_policy='forward_measured_segment_plane_intersection_with_body_margin')
    delta = lift-home
    # Feedback adds acceleration to the feed-forward command: reserve margin
    # rather than declaring a compliant planned curve to be a compliant flight.
    command_speed,command_accel = .9*speed,.7*accel
    takeoff_minimum=_positive(takeoff_min_duration,'takeoff minimum duration',allow_zero=True)
    line = [home,home+delta/3,home+2*delta/3,lift]
    segments = [(_moving_segment(line,0.,'takeoff',command_speed,command_accel,jerk,
                                 exit_speed=join_speed,minimum_duration=takeoff_minimum)
                 if continuous else
                 _segment(line,0.,'takeoff',command_speed,command_accel,jerk,composed_bounds=True))]
    if not continuous and takeoff_minimum>segments[0]['duration_s']:
        factor=takeoff_minimum/segments[0]['duration_s']
        segments[0]['duration_s']=takeoff_minimum
        for key,power in [('speed_m_s',1),('acceleration_m_s2',2),('jerk_m_s3',3)]:
            segments[0]['continuous_bounds'][key]/=factor**power
    start = segments[0]['duration_s']
    if path_kind == 'spline':
        h = min(_positive(vertical_handle,'vertical handle'),(goal[2]-lift[2])/(2 if continuous else 1))
        a = min(_positive(approach_handle,'approach handle'),(distance-standoff)*.8)
        if continuous:
            # B''(0)=6(P2-2P1+P0)=0. Nonzero entry velocity then
            # matches the straight ascent without a transverse acceleration jump.
            if gate:
                # Horizontal motion is (R+D)u^3 because both inner points are
                # vertical. Solve h so B(u_gate) is exactly the sensed gate
                # center and the terminal hold stays at that same altitude.
                # B''(0)=0 still preserves the nonzero vertical C2 join.
                u = (distance/(distance+exit_distance))**(1./3.)
                h = (target[2]-lift[2])*(1.-u**3)/(3.*(u-u**3))
                gate['cubic_parameter_at_center'] = float(u)
                gate['solved_vertical_handle_m'] = float(h)
            points = [lift,lift+np.array([0.,0.,h]),lift+np.array([0.,0.,2*h]),goal]
            if gate and gate_crossing_speed is not None:
                crossing_speed = _positive(gate_crossing_speed,'gate crossing speed')
                crossing_acceleration = float(_xyz([gate_crossing_tangential_acceleration,0.,0.],
                                                   'gate crossing tangential acceleration')[0])
                if crossing_speed>command_speed or abs(crossing_acceleration)>command_accel:
                    raise ValueError('gate_crossing_boundary_exceeds_command_limits')
                before,after = _split_cubic(points,gate['cubic_parameter_at_center'])
                for control,entry,exit,entry_accel,exit_accel in (
                        (before,join_speed,crossing_speed,0.,crossing_acceleration),
                        (after,crossing_speed,0.,crossing_acceleration,0.)):
                    segment = _moving_segment(control,start,'spline_tracking',command_speed,command_accel,
                        jerk,entry_speed=entry,exit_speed=exit,
                        entry_tangential_acceleration=entry_accel,exit_tangential_acceleration=exit_accel)
                    segments.append(segment);start += segment['duration_s']
                gate.update(timing_policy='same_cubic_split_quintic_C2_gate_boundary',
                    requested_crossing_speed_m_s=crossing_speed,
                    requested_crossing_tangential_acceleration_m_s2=crossing_acceleration)
            else:
                segments.append(_moving_segment(points,start,'spline_tracking',command_speed,command_accel,
                                                 jerk,entry_speed=join_speed))
        else:
            if gate:
                # A stopped vertical join does not require B''(0)=0, so the
                # same exact through-center geometry can be used unchanged.
                u = (distance/(distance+exit_distance))**(1./3.)
                h = (target[2]-lift[2])*(1.-u**3)/(3.*(u-u**3))
                points = [lift,lift+np.array([0.,0.,h]),lift+np.array([0.,0.,2*h]),goal]
                gate.update(cubic_parameter_at_center=float(u),solved_vertical_handle_m=float(h))
            else:
                points = [lift,lift+np.array([0.,0.,h]),goal-a*direction,goal]
            segments.append(_segment(points,start,'spline_tracking',command_speed,command_accel,jerk,
                                     composed_bounds=True))
    elif path_kind == 'right_angle':
        corner = np.array([home[0],home[1],goal[2]])
        for endpoint in (corner,goal):
            initial = np.asarray(segments[-1]['control_points'][-1]); delta = endpoint-initial
            segments.append(_segment([initial,initial+delta/3,initial+2*delta/3,endpoint],
                                     start,'spline_tracking',command_speed,command_accel,jerk,
                                     composed_bounds=True))
            start += segments[-1]['duration_s']
    else: raise ValueError('unknown_path_kind')
    duration = sum(p['duration_s'] for p in segments)
    if minimum > duration:
        factor = minimum/duration; start = 0.
        join_speed /= factor
        for segment in segments:
            segment['start_time_s'] = start; segment['duration_s'] *= factor
            for key,power in [('speed_m_s',1),('acceleration_m_s2',2),('jerk_m_s3',3)]:
                segment['continuous_bounds'][key] /= factor**power
            start += segment['duration_s']
    # A Bezier curve remains within its control-point convex hull. Checking
    # every control point certifies the entire curve against this half-space.
    if not target_is_virtual_waypoint:
        margin = min(float(-(np.asarray(point)-target)@direction)
                     for segment in segments for point in segment['control_points'])
        if margin < safety['required_plane_separation_m']-1e-9:
            raise ValueError('path_crosses_sensor_surface_approach_boundary')
        safety.update(continuous_minimum_plane_separation_m=margin,passed=True,
                      scope='Observed static surface approach; unknown hidden geometry is not reconstructed.')
    plan = _finish_plan(home,goal,height,segments,speed,accel,jerk,period,
                        list(obstacles),clearance)
    samples = plan['samples']; curvature = []
    for row in samples:
        v,a = np.array(row['velocity']),np.array(row['acceleration']); norm = np.linalg.norm(v)
        if norm > 1e-3: curvature.append(float(np.linalg.norm(np.cross(v,a))/norm**3))
    positions = np.array([row['position'] for row in samples])
    plan.update(path_kind=path_kind, valid=True, target_world=target.tolist(), standoff_m=standoff,
                controller_profile=controller_profile,
                controller_limit_scope=('Research hypotheses; actual VTOL motion must be recorded and evaluated.'
                    if controller_profile == 'tailsitter20cm_reference' else 'Legacy controller experiment limits'),
                arrival_policy=('pass_through_sensor_gate_then_hold_behind' if gate else
                                'virtual_sensor_waypoint' if target_is_virtual_waypoint
                                else 'noncontact_hold_before_observed_target_surface'),
                target_is_virtual_waypoint=target_is_virtual_waypoint,target_surface_safety=safety,
                command_limits=dict(speed_m_s=command_speed,acceleration_m_s2=command_accel,jerk_m_s3=jerk),
                feedback_margin=dict(speed_fraction=.9,acceleration_fraction=.7),
                takeoff_min_duration_s=takeoff_minimum,
                continuous_takeoff=continuous,takeoff_join_speed_m_s=join_speed,
                geometry=dict(P0=home.tolist(),lift=lift.tolist(),Pg=goal.tolist(),
                              vertical_handle_m=vertical_handle,approach_handle_m=approach_handle),
                trajectory_policy=('cubic_geometry_quintic_Hermite_timing_C2_nonzero_vertical_join'
                    if continuous else 'cubic_geometry_quintic_timing_C2_zero_velocity_acceleration_at_joins'),
                sampled_length_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
                sampled_max_curvature_m_inv=max(curvature,default=0.),
                curvature_note='Geometric curvature is undefined at a stopped sharp corner; not globally bounded for right_angle.')
    if gate:
        # Exact-curve bisection works for either timing law; no sampling-based
        # proximity or Gazebo target coordinates determine this checkpoint.
        low,high = segments[0]['duration_s'],plan['duration_s']
        for _ in range(60):
            middle = (low+high)/2.
            if (np.asarray(sample_plan(plan,middle)['position'])-target)@direction < 0:
                low = middle
            else:high = middle
        crossing = sample_plan(plan,(low+high)/2.)
        gate.update(planned_crossing_time_s=(low+high)/2.,
                    planned_crossing_world=crossing['position'],
                    planned_crossing_speed_m_s=float(np.linalg.norm(crossing['velocity'])),
                    hold_world=goal.tolist())
        plan.update(gate=gate,target_is_pass_through_gate=True)
    return plan


def checkpoint_crossing(gate, previous_position, current_position, max_segment_length=2.):
    """Test one fresh EKF segment against the locked virtual rectangle.

    Only a forward front-to-back intersection can pass. A nearby sample,
    initial position behind the gate, backward travel, or a large pose jump
    does not count. The caller must additionally reject stale timestamps.
    The whole body radius must fit inside the rectangle at the intersection.
    """
    center = _xyz(gate['center_world'],'gate center')
    normal = _xyz(gate['normal_world'],'gate normal')
    right = _xyz(gate['right_world'],'gate right'); up = _xyz(gate['up_world'],'gate up')
    axes = np.array([normal,right,up])
    if not np.allclose(axes@axes.T,np.eye(3),atol=1e-8):raise ValueError('invalid_gate_basis')
    width = _positive(gate['width_m'],'gate width'); height = _positive(gate['height_m'],'gate height')
    radius = _positive(gate.get('body_radius_m',0.),'gate body radius',allow_zero=True)
    if min(width,height) <= 2*radius:raise ValueError('gate_too_small_for_vehicle_body')
    before = _xyz(previous_position,'previous measured position')
    after = _xyz(current_position,'current measured position')
    length = float(np.linalg.norm(after-before))
    maximum = _positive(max_segment_length,'maximum measured segment length')
    d0,d1 = float((before-center)@normal),float((after-center)@normal)
    result = dict(crossed=False,passed=False,missed=False,
                  previous_plane_distance_m=d0,current_plane_distance_m=d1,
                  measured_segment_length_m=length)
    if length > maximum:
        return dict(result,valid=False,reason='measured_position_jump')
    if not d0 < 0. < d1:
        return dict(result,valid=True,reason='no_forward_plane_crossing')
    fraction = -d0/(d1-d0); point = before+fraction*(after-before)
    lateral,vertical = float((point-center)@right),float((point-center)@up)
    passed = abs(lateral)+radius <= width/2.+1e-10 and abs(vertical)+radius <= height/2.+1e-10
    return dict(result,valid=True,crossed=True,passed=bool(passed),missed=not bool(passed),
                fraction=float(fraction),intersection_world=point.tolist(),
                lateral_offset_m=lateral,vertical_offset_m=vertical,
                reason='rectangle_passed' if passed else 'rectangle_missed')


def stationary_sample(position):
    return dict(position=np.asarray(position,dtype=float).tolist(),velocity=[0.,0.,0.],acceleration=[0.,0.,0.])


def path_progress(plan, position, previous_time=0., max_advance=1.):
    """Project a measured world position onto a forward window of sampled path.

    This is geometric progress, not a clock or a prediction. The nearest line
    segment is clipped to the monotone time window before projection; moving
    backwards or teleporting ahead cannot rewind or skip that window. The
    returned distance is to the sampled polyline, not an exact Bezier distance.
    """
    position = _xyz(position,'measured aircraft position')
    previous = _positive(previous_time,'previous path time',allow_zero=True)
    advance = _positive(max_advance,'maximum path advance',allow_zero=True)
    try:
        duration = _positive(plan['duration_s'],'plan duration')
        samples = plan['samples']
        times = np.asarray([row['time_s'] for row in samples],dtype=float)
        points = np.asarray([row['position'] for row in samples],dtype=float)
        if (plan['frame_id'] != 'world' or len(times)<2 or points.shape != (len(times),3)
                or not np.isfinite(times).all() or not np.isfinite(points).all()
                or times[0] != 0. or np.any(np.diff(times)<=0.)
                or abs(times[-1]-duration)>1e-8 or previous>duration):
            raise ValueError('invalid_progress_path_samples_or_previous_time')
    except (KeyError,TypeError,OverflowError) as error:
        raise ValueError('invalid_progress_path') from error
    end = min(duration,previous+advance)
    indices = np.flatnonzero((times[1:]>=previous)&(times[:-1]<=end))
    t0,t1 = times[indices],times[indices+1]
    a,b = points[indices],points[indices+1]
    delta = b-a
    square_length = np.sum(delta*delta,axis=1)
    lower = np.maximum(0.,(previous-t0)/(t1-t0))
    upper = np.minimum(1.,(end-t0)/(t1-t0))
    fraction = np.divide(np.sum((position-a)*delta,axis=1),square_length,
                         out=np.zeros_like(square_length),where=square_length>0.)
    fraction = np.clip(fraction,lower,upper)
    projected = a+fraction[:,None]*delta
    distances = np.linalg.norm(projected-position,axis=1)
    closest = int(np.argmin(distances))
    clock = float(t0[closest]+fraction[closest]*(t1[closest]-t0[closest]))
    if not np.isfinite(distances[closest]) or not math.isfinite(clock):
        raise ValueError('nonfinite_path_progress_projection')
    clock = max(previous,min(end,clock))
    return dict(trajectory_time_s=clock,cross_track_distance_m=float(distances[closest]),
                projected_position_world=projected[closest].tolist(),progress_fraction=clock/duration,
                advance_capped=bool(clock>=end-1e-10 and end<duration),
                method='measured_world_position_projection_on_plan_sample_segments')
