"""Measured P/V/A -> certified quintic stop -> vertical landing, without teleport."""
import math

import numpy as np

from .observation_path import _xyz, _positive, _segment, _evaluate

MODEL = 'quintic_braking_cubic_landing_v1'
_FIELDS = (('velocity','speed_m_s'), ('acceleration','acceleration_m_s2'), ('jerk','jerk_m_s3'))


def _split(control):
    left, right = [control[0]], [control[-1]]
    while len(control) > 1:
        control = (control[:-1]+control[1:])*.5
        left.append(control[0]); right.append(control[-1])
    return np.asarray(left), np.asarray(right[::-1])


def _norm_bound(control, depth=6):
    if depth == 0:
        return float(np.max(np.linalg.norm(control, axis=1)))
    return max(_norm_bound(part, depth-1) for part in _split(control))


def _bounds(points, duration):
    control = points-points[0]
    result = {}
    for _, key in _FIELDS:
        control = (len(control)-1)*np.diff(control, axis=0)/duration
        rounding = 256*np.finfo(float).eps*max(1., float(np.max(np.linalg.norm(control, axis=1))))
        result[key] = _norm_bound(control)+rounding
    return result


def _cannot_clear(points, low, high, depth=0):
    if np.any(points.max(axis=0) < low) or np.any(points.min(axis=0) > high):
        return False
    if any(np.all(point >= low) and np.all(point <= high) for point in (points[0],points[-1])):
        return True
    if depth >= 24:
        return True
    return any(_cannot_clear(part,low,high,depth+1) for part in _split(points))


def build_braking_landing(position, velocity, acceleration, *, landing_z,
                          max_speed=1.5, max_acceleration=.6, max_jerk=1.2,
                          obstacles=(), clearance=.75, sample_period=.05, surface_boundary=None):
    """Return a world-frame stop/land plan or reject uncertifiable trajectories.

    The Hermite stop uses Pstop=P+.5*T*V+T²*A/12, with final V=A=0.
    Increasing duration is a bounded feasibility search, not obstacle avoidance.
    `surface_boundary` accepts target_world plus the original static plan's
    target_surface_safety fields; ground contact is outside free-flight limits.
    """
    p, v, a = (_xyz(value,name) for value,name in
               ((position,'position'),(velocity,'velocity'),(acceleration,'acceleration')))
    speed = _positive(max_speed,'max speed'); accel = _positive(max_acceleration,'max acceleration')
    jerk = _positive(max_jerk,'max jerk'); period = _positive(sample_period,'sample period')
    clearance = _positive(clearance,'clearance',allow_zero=True)
    z = _positive(landing_z,'landing height',allow_zero=True)
    if np.linalg.norm(v)>speed+1e-9 or np.linalg.norm(a)>accel+1e-9:
        raise ValueError('braking_initial_state_exceeds_motion_limits')
    if z>p[2]+1e-9:
        raise ValueError('landing_height_above_current_position')
    limits = dict(speed_m_s=speed,acceleration_m_s2=accel,jerk_m_s3=jerk)
    duration = max(.2,1.5*np.linalg.norm(v)/accel,
                   math.sqrt(6*np.linalg.norm(v)/jerk),2*np.linalg.norm(a)/jerk)
    for _ in range(100):
        stop = p+.5*duration*v+duration**2*a/12.
        points = np.array([p,p+duration*v/5.,p+2*duration*v/5.+duration**2*a/20.,stop,stop,stop])
        bounds = _bounds(points,duration)
        if all(math.isfinite(bounds[key]) and bounds[key]<=limits[key]*(1.+1e-10) for _,key in _FIELDS):
            break
        duration *= 1.08
        if duration>30.:break
    else:
        raise ValueError('no_motion_limited_quintic_braking_trajectory')
    if not all(math.isfinite(bounds[key]) and bounds[key]<=limits[key]*(1.+1e-10) for _,key in _FIELDS):
        raise ValueError('no_motion_limited_quintic_braking_trajectory')
    if stop[2]<z-1e-9:
        raise ValueError('braking_stop_below_landing_height')
    brake = dict(phase='abort_braking',kind='quintic_hermite',start_time_s=0.,duration_s=float(duration),
                 control_points=points.tolist(),continuous_bounds=bounds)
    segments = [brake]; goal = stop.copy(); goal[2] = z
    if np.linalg.norm(goal-stop)>1e-8:
        delta = goal-stop
        landing = _segment([stop,stop+delta/3.,stop+2*delta/3.,goal],float(duration),'abort_landing',
                           min(.4,speed),min(.3,accel),min(.6,jerk))
        landing['kind'] = 'cubic_quintic_timing'
        segments.append(landing)
    if surface_boundary is not None:
        target = _xyz(surface_boundary['target_world'],'surface target')
        direction = _xyz(surface_boundary['direction_world'],'surface direction')
        if not math.isclose(float(np.linalg.norm(direction)),1.,rel_tol=0.,abs_tol=1e-8):
            raise ValueError('surface_direction_must_be_unit_length')
        margin = _positive(surface_boundary['required_plane_separation_m'],'surface separation')
        if any(float(-(np.asarray(point)-target)@direction)<margin-1e-9
               for segment in segments for point in segment['control_points']):
            raise ValueError('braking_crosses_observed_target_surface_boundary')
    checked = []
    for obstacle in obstacles:
        low, high = _xyz(obstacle['min'],'obstacle min'),_xyz(obstacle['max'],'obstacle max')
        if np.any(low>high):raise ValueError('obstacle_min_exceeds_max')
        for segment in segments:
            control = np.asarray(segment['control_points'])
            padding = 32*np.finfo(float).eps*max(1.,float(np.max(np.abs(control))),
                                               float(np.max(np.abs(np.r_[low,high]))))
            if _cannot_clear(control,low-clearance-padding,high+clearance+padding):
                raise ValueError('braking_or_landing_intersects_obstacle_or_clearance_uncertified')
        checked.append(dict(min=low.tolist(),max=high.tolist()))
    total = segments[-1]['start_time_s']+segments[-1]['duration_s']
    counts = [math.ceil(segment['duration_s']/period) for segment in segments]
    if sum(counts)+1>20000:raise ValueError('braking_plan_exceeds_20000_samples')
    plan = dict(model=MODEL,frame_id='world',home=p.tolist(),goal=goal.tolist(),duration_s=total,
                initial_velocity=v.tolist(),initial_acceleration=a.tolist(),stop_world=stop.tolist(),
                segments=segments,limits=limits,obstacles=checked,clearance_m=clearance,
                trajectory_policy='measured_PVA_quintic_C2_stop_then_vertical_landing',
                collision_check='continuous_bernstein_hull_subdivision_against_inflated_aabbs',
                motion_validation=dict(passed=True,method='continuous_bernstein_derivative_hull_six_subdivisions',
                    continuous_upper_bounds={key:max(s['continuous_bounds'][key] for s in segments) for _,key in _FIELDS}))
    times = [0.]
    for segment,count in zip(segments,counts):
        times.extend((segment['start_time_s']+np.linspace(0.,segment['duration_s'],count+1)[1:]).tolist())
    plan['samples'] = [sample_braking_landing(plan,t) for t in times]
    return plan


def sample_braking_landing(plan, time_s):
    """Exact P/V/A/jerk evaluation; retain the measured initial V/A at t=0."""
    time_s = _positive(time_s,'braking time',allow_zero=True)
    if plan.get('model')!=MODEL or plan.get('frame_id')!='world':
        raise ValueError('invalid_braking_plan_model_or_frame')
    clamped = min(time_s,plan['duration_s'])
    segment = next((s for s in plan['segments'] if clamped<=s['start_time_s']+s['duration_s']),plan['segments'][-1])
    if segment['kind']=='cubic_quintic_timing':
        value = _evaluate(segment,clamped)
    elif segment['kind']=='quintic_hermite':
        u = (clamped-segment['start_time_s'])/segment['duration_s']
        control = np.asarray(segment['control_points']); value = dict(time_s=float(time_s),phase=segment['phase'])
        for field in ('position','velocity','acceleration','jerk'):
            evaluated = control.copy()
            while len(evaluated)>1:evaluated=(1.-u)*evaluated[:-1]+u*evaluated[1:]
            value[field] = evaluated[0].tolist()
            control = (len(control)-1)*np.diff(control,axis=0)/segment['duration_s']
    else:raise ValueError('unknown_braking_segment_kind')
    if time_s>=plan['duration_s']:
        for field,_ in _FIELDS:value[field]=[0.,0.,0.]
        value['phase']='hold'
    return value
