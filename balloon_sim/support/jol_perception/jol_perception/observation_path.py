"""고정 관찰점까지의 기하 경로만 만든다. 비행 명령이나 표적 추적 입력은 없다.

1 m는 home의 기체 참조점 대비 상승량이다. clearance는 제공된 장애물 AABB를
팽창시키는 거리이며 기체 크기·위치 오차·장애물 구성은 호출자가 정한다.
"""
import math

import numpy as np

MODEL = 'cubic_bezier_quintic_v1'
_H1, _H2, _H3 = 15./8., 10./math.sqrt(3.), 60.


def _xyz(value, name):
    try:
        if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3
                or any(isinstance(v, (bool, np.bool_)) or
                       not isinstance(v, (int, float, np.integer, np.floating)) for v in value)):
            raise ValueError(name+' must contain numeric XYZ, not boolean or string values')
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name+' must be finite XYZ') from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name+' must be finite XYZ')
    return result


def _positive(value, name, allow_zero=False):
    try:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(name+' must be numeric')
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(name+' must be finite') from error
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(name+' must be finite and '+('nonnegative' if allow_zero else 'positive'))
    return result


def _composed_derivative_bounds(points, time_control_points=None):
    """Bound B(quintic(t)) over the whole interval, independently of samples.

    Bernstein products give the exact degree-15 position polynomial. Derivative
    control hulls after six halvings bound each vector norm continuously.
    """
    def product(a, b):
        n, m = len(a)-1, len(b)-1
        result = np.zeros(n+m+1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                result[i+j] += x*y*math.comb(n, i)*math.comb(m, j)/math.comb(n+m, i+j)
        return result

    def bound(control, depth=6):
        if depth == 0:
            return float(np.max(np.linalg.norm(control, axis=1)))
        left, right = [control[0]], [control[-1]]
        while len(control) > 1:
            control = (control[:-1]+control[1:])*.5
            left.append(control[0]); right.append(control[-1])
        return max(bound(np.asarray(left), depth-1), bound(np.asarray(right[::-1]), depth-1))

    u = (np.array([0., 0., 0., 1., 1., 1.]) if time_control_points is None
         else np.asarray(time_control_points, dtype=float)); v = 1.-u
    basis = np.array([product(product(v,v),v), 3*product(product(v,v),u),
                      3*product(product(v,u),u), product(product(u,u),u)]).T
    # Translation cancels from derivatives; remove it before differencing.
    control = basis @ (points-points[0])
    numerators = []
    for _ in range(3):
        control = (len(control)-1)*np.diff(control, axis=0)
        rounding = 256*np.finfo(float).eps*max(1., float(np.max(np.linalg.norm(control, axis=1))))
        numerators.append(bound(control)+rounding)
    return numerators


def _segment(points, start, phase, speed, acceleration, jerk, *, composed_bounds=False):
    points = np.asarray(points, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        b1 = 3*float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        b2 = 6*float(np.max(np.linalg.norm(np.diff(points, n=2, axis=0), axis=1)))
        b3 = 6*float(np.linalg.norm(np.diff(points, n=3, axis=0)[0]))
        # Bezier 도함수의 convex-hull 상계와 시간함수의 최대 도함수를 결합한다.
        numerators = [b1*_H1, b2*_H1**2+b1*_H2,
                      b3*_H1**3+3*b2*_H1*_H2+b1*_H3]
        if composed_bounds and np.isfinite(numerators).all():
            # Both are continuous upper bounds; their minimum remains a bound.
            numerators = np.minimum(numerators, _composed_derivative_bounds(points))
        duration = max(numerators[0]/speed, math.sqrt(numerators[1]/acceleration),
                       float(np.cbrt(numerators[2]/jerk))) * (1.+1e-12)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Path scale or motion limits cannot produce a finite duration')
    inverse = 1./duration
    segment = dict(phase=phase, start_time_s=start, duration_s=duration,
                   control_points=points.tolist(), continuous_bounds=dict(
                    speed_m_s=numerators[0]*inverse,
                    acceleration_m_s2=numerators[1]*inverse**2,
                    jerk_m_s3=numerators[2]*inverse**3))
    if composed_bounds:
        segment['bounds_method'] = 'composed_degree15_bernstein_derivative_hull_six_subdivisions'
    return segment


def _evaluate(segment, time_s):
    points = np.asarray(segment['control_points'], dtype=float)
    duration = _positive(segment['duration_s'], 'segment duration')
    start = _positive(segment['start_time_s'], 'segment start', allow_zero=True)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        raise ValueError('Segment needs four finite XYZ control points')
    inverse = 1./duration
    tau = float(np.clip((time_s-start)*inverse, 0., 1.))
    if 'time_control_points' in segment:
        timing = np.asarray(segment['time_control_points'], dtype=float)
        if (timing.shape != (6,) or not np.isfinite(timing).all()
                or timing[0] != 0. or timing[-1] != 1. or np.any(np.diff(timing) < -1e-12)):
            raise ValueError('Timing needs six finite monotone Bernstein controls from zero to one')
        derivatives = []
        for order in range(4):
            degree = len(timing)-1
            derivatives.append(sum(math.comb(degree,i)*(1.-tau)**(degree-i)*tau**i*x
                                   for i,x in enumerate(timing))*inverse**order)
            timing = degree*np.diff(timing)
        u,du,ddu,dddu = derivatives
    else:
        u = tau**3*(10.-15.*tau+6.*tau**2)
        du = 30.*tau**2*(1.-tau)**2*inverse
        ddu = 60.*tau*(1.-tau)*(1.-2.*tau)*inverse**2
        dddu = (60.-360.*tau+360.*tau**2)*inverse**3
    v = 1.-u
    delta = np.diff(points, axis=0)
    second = np.diff(points, n=2, axis=0)
    position = v**3*points[0]+3*v*v*u*points[1]+3*v*u*u*points[2]+u**3*points[3]
    first = 3*(v*v*delta[0]+2*v*u*delta[1]+u*u*delta[2])
    curvature = 6*(v*second[0]+u*second[1])
    third = 6*np.diff(points, n=3, axis=0)[0]
    return dict(time_s=float(time_s), phase=segment['phase'], position=position.tolist(),
                velocity=(first*du).tolist(), acceleration=(curvature*du**2+first*ddu).tolist(),
                jerk=(third*du**3+3*curvature*du*ddu+first*dddu).tolist())


def sample_plan(plan, time_s):
    """build_plan의 결과를 정확 평가한다. 시작 전·완료 후에는 끝점에서 정지한다."""
    try:
        if isinstance(time_s, (bool, np.bool_)) or not isinstance(time_s, (int, float, np.integer, np.floating)):
            raise ValueError('Evaluation time must be numeric')
        time_s = float(time_s)
        if not math.isfinite(time_s) or plan['model'] != MODEL or plan['frame_id'] != 'world':
            raise ValueError('Invalid plan model, frame or time')
        duration = _positive(plan['duration_s'], 'plan duration')
        segments = plan['segments']
        if not isinstance(segments, list) or not segments:
            raise ValueError('Plan needs its exact curve segments')
        clamped = max(0., min(time_s, duration))
        segment = next((part for part in segments
                        if clamped <= part['start_time_s']+part['duration_s']), segments[-1])
        result = _evaluate(segment, clamped)
        if time_s < 0 or time_s >= duration:
            for field in ('velocity', 'acceleration', 'jerk'):
                result[field] = [0., 0., 0.]
        if time_s >= duration:
            result['phase'] = 'hold'
        return result
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError('Invalid exact-curve plan') from error


def _split(points):
    a = (points[:-1]+points[1:])*.5
    b = (a[:-1]+a[1:])*.5
    middle = (b[0]+b[1])*.5
    return np.array([points[0], a[0], b[0], middle]), np.array([middle, b[1], a[2], points[3]])


def _cannot_clear(points, low, high, depth=0):
    # 곡선 전체가 제어점 hull 안에 있다. 샘플 간격과 무관하게 분리된 부분만 승인한다.
    if np.any(points.max(axis=0) < low) or np.any(points.min(axis=0) > high):
        return False
    if any(np.all(point >= low) and np.all(point <= high) for point in (points[0], points[-1])):
        return True
    if depth >= 24:
        return True  # 경계 접촉 또는 수치상 분리 미확정도 보수적으로 거부한다.
    left, right = _split(points)
    return _cannot_clear(left, low, high, depth+1) or _cannot_clear(right, low, high, depth+1)


def build_plan(home, goal, *, takeoff_height=1., max_speed=2., max_acceleration=1.,
               max_jerk=2., sample_period=.05, obstacles=(), clearance=.75):
    """수직 이륙 정지 후 위쪽→수평 접선의 곡선으로 고정 관찰점에 도착한다.

    수평 이동 목표는 이륙 끝보다 높아야 한다. 장애물 회피 경로를 검색하지 않으며
    clearance를 만족하지 못하는 계획은 거부한다. 샘플 수는 20,000개 이하다.
    """
    home, goal = _xyz(home, 'home'), _xyz(goal, 'goal')
    height = _positive(takeoff_height, 'takeoff_height')
    speed = _positive(max_speed, 'max_speed')
    acceleration = _positive(max_acceleration, 'max_acceleration')
    jerk = _positive(max_jerk, 'max_jerk')
    period = _positive(sample_period, 'sample_period')
    clearance = _positive(clearance, 'clearance', allow_zero=True)
    lift = home+np.array([0., 0., height])
    if not np.isfinite(lift).all() or lift[2] <= home[2]:
        raise ValueError('Takeoff height is not representable at this home coordinate')
    if goal[2] < lift[2]:
        raise ValueError('Goal is below the takeoff endpoint')
    horizontal = goal[:2]-lift[:2]
    if np.any(horizontal != 0.) and goal[2] <= lift[2]:
        raise ValueError('A vertical-to-horizontal rising curve needs a higher goal')
    delta = lift-home
    segments = [_segment([home, home+delta/3., home+2*delta/3., lift], 0., 'takeoff',
                         speed, acceleration, jerk)]
    if not np.array_equal(goal, lift):
        if np.all(horizontal == 0.):
            delta = goal-lift
            points = [lift, lift+delta/3., lift+2*delta/3., goal]
        else:
            # 두 안쪽 제어점은 시작 접선 +Z, 끝 접선 수평을 고정한다.
            first = lift+np.array([0., 0., 2*(goal[2]-lift[2])/3.])
            second = goal-np.r_[horizontal/3., 0.]
            points = [lift, first, second, goal]
        segments.append(_segment(points, segments[0]['duration_s'], 'transfer', speed, acceleration, jerk))
    return _finish_plan(home, goal, height, segments, speed, acceleration, jerk, period, obstacles, clearance)


def build_corner_plan(home, goal, *, takeoff_height=1., turn_radius=1., max_speed=2.,
                      max_acceleration=1., max_jerk=2., sample_period=.05,
                      obstacles=(), clearance=.75):
    """P0 -> A (vertical), A -> B (cubic rounded 90-degree corner), B -> Pg.

    C=(P0.x,P0.y,Pg.z). Initial vertical length is at least takeoff_height.
    Each segment uses quintic time scaling: position, velocity and acceleration
    are continuous, with zero speed at joins. This is a stop-at-joins trajectory.
    """
    home, goal = _xyz(home, 'home'), _xyz(goal, 'goal')
    height = _positive(takeoff_height, 'takeoff_height')
    radius = _positive(turn_radius, 'turn_radius')
    speed = _positive(max_speed, 'max_speed')
    acceleration = _positive(max_acceleration, 'max_acceleration')
    jerk = _positive(max_jerk, 'max_jerk')
    period = _positive(sample_period, 'sample_period')
    clearance = _positive(clearance, 'clearance', allow_zero=True)
    rise = goal[2]-home[2]
    horizontal = goal-home; horizontal[2] = 0.
    distance = float(np.linalg.norm(horizontal))
    if not math.isfinite(rise) or rise < height:
        raise ValueError('Goal must allow the initial vertical takeoff')
    if distance == 0.:
        plan = build_plan(home, goal, takeoff_height=height, max_speed=speed,
                          max_acceleration=acceleration, max_jerk=jerk,
                          sample_period=period, obstacles=obstacles, clearance=clearance)
        plan['corner_geometry'] = dict(P0=home.tolist(), Pg=goal.tolist(), C=goal.tolist(),
                                      turn_angle_deg=None, turn_radius_m=0., vertical_only=True)
        return plan
    if rise <= height:
        raise ValueError('A rounded 90-degree turn needs height above the initial takeoff')
    radius = min(radius, rise-height, distance/2.)
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError('Corner radius is not representable')
    up = np.array([0., 0., 1.]); direction = horizontal/distance
    corner = np.array([home[0], home[1], goal[2]])
    entry, exit = corner-radius*up, corner+radius*direction
    kappa = 4.*(math.sqrt(2.)-1.)/3.
    delta = entry-home
    segments = [_segment([home, home+delta/3., home+2*delta/3., entry], 0., 'takeoff', speed, acceleration, jerk)]
    start = segments[0]['duration_s']
    segments.append(_segment([entry, entry+kappa*radius*up, exit-kappa*radius*direction, exit],
                             start, 'corner', speed, acceleration, jerk))
    start += segments[-1]['duration_s']
    delta = goal-exit
    segments.append(_segment([exit, exit+delta/3., exit+2*delta/3., goal],
                             start, 'approach', speed, acceleration, jerk))
    plan = _finish_plan(home, goal, height, segments, speed, acceleration, jerk, period, obstacles, clearance)
    plan['corner_geometry'] = dict(P0=home.tolist(), Pg=goal.tolist(), C=corner.tolist(),
                                  A=entry.tolist(), B=exit.tolist(), turn_angle_deg=90.,
                                  turn_radius_m=radius, kappa=kappa, vertical_only=False,
                                  vertical_length_m=float(entry[2]-home[2]))
    plan['trajectory_policy'] = 'quintic_time_scaling_per_segment_stop_at_joins_C2'
    return plan


def _finish_plan(home, goal, height, segments, speed, acceleration, jerk, period, obstacles, clearance):
    checked_obstacles = []
    try:
        for obstacle in obstacles:
            low, high = _xyz(obstacle['min'], 'obstacle min'), _xyz(obstacle['max'], 'obstacle max')
            if np.any(low > high):
                raise ValueError('Obstacle min must not exceed max')
            expanded_low, expanded_high = low-clearance, high+clearance
            if not np.isfinite(np.r_[expanded_low, expanded_high]).all():
                raise ValueError('Expanded obstacle is not finite')
            for segment in segments:
                points = np.asarray(segment['control_points'])
                padding = 32*np.finfo(float).eps*max(1., np.max(np.abs(points)),
                    np.max(np.abs(expanded_low)), np.max(np.abs(expanded_high)))
                if _cannot_clear(points, expanded_low-padding, expanded_high+padding):
                    raise ValueError('Curve intersects an obstacle or clearance cannot be certified')
            checked_obstacles.append(dict(min=low.tolist(), max=high.tolist()))
    except (TypeError, KeyError) as error:
        raise ValueError('Obstacles must contain min/max XYZ bounds') from error
    duration = segments[-1]['start_time_s']+segments[-1]['duration_s']
    counts = [segment['duration_s']/period for segment in segments]
    if not math.isfinite(duration) or any(not math.isfinite(count) or count > 19999 for count in counts):
        raise ValueError('Plan duration or sample count is too large')
    if sum(math.ceil(count) for count in counts)+1 > 20000:
        raise ValueError('Plan needs more than 20000 samples; increase sample_period')
    plan = dict(model=MODEL, frame_id='world', home=home.tolist(), goal=goal.tolist(),
                duration_s=duration, takeoff_height_m=height, clearance_m=clearance,
                limits=dict(speed_m_s=speed, acceleration_m_s2=acceleration, jerk_m_s3=jerk),
                obstacles=checked_obstacles, segments=segments,
                collision_check='continuous_bezier_hull_subdivision_against_inflated_aabbs')
    times = [0.]
    for segment, count in zip(segments, counts):
        times.extend((segment['start_time_s']+np.linspace(0., segment['duration_s'], math.ceil(count)+1)[1:]).tolist())
    plan['samples'] = [sample_plan(plan, time_s) for time_s in times]
    fields = [('velocity', 'speed_m_s'), ('acceleration', 'acceleration_m_s2'), ('jerk', 'jerk_m_s3')]
    plan['motion_validation'] = dict(
        passed=all(segment['continuous_bounds'][key] <= plan['limits'][key]*(1.+1e-10)
                   for segment in segments for _, key in fields),
        method=('composed_degree15_bernstein_derivative_hull_six_subdivisions'
                if all(segment.get('bounds_method') for segment in segments)
                else 'analytic_continuous_bezier_derivative_and_quintic_time_bounds'),
        continuous_upper_bounds={key:max(segment['continuous_bounds'][key] for segment in segments)
                                 for _,key in fields},
        sampled_peaks={key:max(float(np.linalg.norm(sample[field])) for sample in plan['samples'])
                       for field,key in fields})
    return plan
