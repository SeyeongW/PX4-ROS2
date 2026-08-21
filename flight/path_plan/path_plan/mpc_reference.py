"""Wang TrackingMPC's time-parameterised B-spline reference helpers."""

from __future__ import annotations

import math

import numpy as np


def _s_curve_stop_speed(remaining_m, acceleration_m_s2, jerk_m_s3):
    """Maximum speed that can stop in ``remaining_m`` with zero end accel."""
    remaining = max(0.0, float(remaining_m))
    acceleration = float(acceleration_m_s2)
    jerk = float(jerk_m_s3)
    if acceleration <= 0.0 or jerk <= 0.0:
        raise ValueError('acceleration and jerk limits must be positive')

    triangular_distance = acceleration ** 3 / jerk ** 2
    if remaining <= triangular_distance:
        return (remaining * math.sqrt(jerk)) ** (2.0 / 3.0)
    ratio = acceleration ** 2 / jerk
    return 0.5 * (math.sqrt(ratio ** 2 + 8.0 * acceleration * remaining)
                  - ratio)


def _relative_braking_path_speed(nominal_speed_m_s, tangent_xy,
                                 target_velocity_xy, range_xy_m,
                                 start_range_m, target_relative_speed_m_s):
    """Reduce moving-target relative speed without leaving the B-spline."""
    nominal = float(nominal_speed_m_s)
    tangent = np.asarray(tangent_xy, float)
    target_velocity = np.asarray(target_velocity_xy, float)
    distance = max(0.0, float(range_xy_m))
    start = float(start_range_m)
    target_relative = float(target_relative_speed_m_s)
    norm = float(np.linalg.norm(tangent))
    values = np.r_[nominal, tangent, target_velocity, distance, start,
                   target_relative]
    if (nominal < 0.0 or tangent.shape != (2,)
            or target_velocity.shape != (2,)
            or not np.all(np.isfinite(values))
            or start <= 0.0 or target_relative < 0.0 or norm <= 1.0e-9):
        raise ValueError('invalid moving-target B-spline braking input')
    if distance >= start or nominal == 0.0:
        return nominal

    direction = tangent / norm
    target_along = float(target_velocity @ direction)
    target_cross_sq = max(
        0.0, float(target_velocity @ target_velocity) - target_along ** 2)
    nominal_relative = nominal * direction - target_velocity
    relative_limit = min(
        float(np.linalg.norm(nominal_relative)), target_relative)
    if relative_limit ** 2 <= target_cross_sq:
        desired = target_along
    else:
        desired = target_along + math.sqrt(
            max(0.0, relative_limit ** 2 - target_cross_sq))
    return float(np.clip(desired, 0.0, nominal))


def limit_acceleration_slew(previous, desired, jerk_m_s3, elapsed_s):
    """Rate-limit the acceleration actually streamed through MAVROS."""
    old = np.asarray(previous, float)
    new = np.asarray(desired, float)
    limit = float(jerk_m_s3) * max(0.0, float(elapsed_s))
    if (old.shape != (3,) or new.shape != (3,)
            or not np.all(np.isfinite(np.r_[old, new]))
            or jerk_m_s3 <= 0.0):
        raise ValueError('invalid acceleration slew input')
    return old + np.clip(new - old, -limit, limit)


def path_reference_horizon(arc_m, path, progress_m, dt_s, horizon,
                           cruise_speed_m_s, acceleration_m_s2,
                           jerk_m_s3, target_velocity_xy=None,
                           target_range_xy_m=math.inf,
                           relative_brake_start_m=10.0,
                           target_relative_speed_m_s=0.3):
    """Time-parameterise geometry with Wang's jerk-aware endpoint braking."""
    arc = np.asarray(arc_m, float)
    points = np.asarray(path, float)
    dt = float(dt_s)
    count = int(horizon)
    speed_limit = float(cruise_speed_m_s)
    accel = float(acceleration_m_s2)
    jerk = float(jerk_m_s3)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) != len(arc)
            or len(points) < 2 or count < 1 or dt <= 0.0
            or speed_limit <= 0.0 or accel <= 0.0 or jerk <= 0.0
            or not np.all(np.isfinite(np.column_stack((arc, points))))
            or not np.all(np.diff(arc) > 0.0)):
        raise ValueError('invalid geometry path or MPC timing')

    distance = float(np.clip(progress_m, arc[0], arc[-1]))
    brake_range = max(0.0, float(target_range_xy_m))
    target_velocity = (None if target_velocity_xy is None else
                       np.asarray(target_velocity_xy, float))
    if (target_velocity is not None
            and (target_velocity.shape != (2,)
                 or not np.all(np.isfinite(target_velocity)))):
        raise ValueError('target_velocity_xy must contain two finite values')
    query = np.empty(count)
    speeds = np.empty(count)
    for index in range(count):
        remaining = max(0.0, float(arc[-1] - distance))
        speed = min(speed_limit, _s_curve_stop_speed(
            remaining, accel, jerk))
        if target_velocity is not None and brake_range < relative_brake_start_m:
            segment = int(np.clip(
                np.searchsorted(arc, distance, side='right') - 1,
                0, len(points) - 2))
            tangent_xy = points[segment + 1, :2] - points[segment, :2]
            speed = _relative_braking_path_speed(
                speed, tangent_xy, target_velocity, brake_range,
                relative_brake_start_m, target_relative_speed_m_s)
        distance = min(float(arc[-1]), distance + speed * dt)
        query[index] = distance
        speeds[index] = speed if distance < arc[-1] - 1.0e-9 else 0.0

    reference_positions = np.column_stack([
        np.interp(query, arc, points[:, axis]) for axis in range(3)])
    segment = np.clip(np.searchsorted(arc, query, side='right') - 1,
                      0, len(points) - 2)
    tangent = points[segment + 1] - points[segment]
    lengths = np.linalg.norm(tangent, axis=1)
    unit = np.divide(tangent, lengths[:, None],
                     out=np.zeros_like(tangent),
                     where=lengths[:, None] > 1.0e-9)
    reference_velocities = unit * speeds[:, None]
    return reference_positions, reference_velocities
