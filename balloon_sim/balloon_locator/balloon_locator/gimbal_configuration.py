"""Shared tracker defaults and checks, usable without ROS or device access."""
import math


TRACKER_DEFAULTS = {
    'image_width': 1920.0, 'image_height': 1080.0,
    'kp_yaw': 60.0, 'kp_pitch': 60.0,
    'yaw_sign': 1.0, 'pitch_sign': -1.0,
    'deadband': 0.03, 'max_rate': 60.0,
    'lost_timeout_sec': 0.5, 'zoom_gain': 1.0,
    'enable_search': False, 'lock_confirm_frames': 3,
    'attitude_timeout_sec': 0.5, 'search_resume_timeout_sec': 2.0,
    'search_yaw_rate': 18.0, 'search_pitch_rate': 8.0,
    'search_period_sec': 11.0, 'search_yaw_limit_deg': 40.0,
    'search_pitch_min_deg': 5.0, 'search_pitch_max_deg': 45.0,
}


def tracker_configuration_errors(parameters):
    """Match rclpy's declared types; reject invalid values before any commands."""
    values = {**TRACKER_DEFAULTS, **parameters}
    errors = []
    for key, default in TRACKER_DEFAULTS.items():
        value = values[key]
        if (type(value) is not type(default)
                or type(value) is float and not math.isfinite(value)):
            errors.append(f'invalid tracker parameter: {key}')
    if errors:
        return errors
    for key in ('image_width', 'image_height', 'lost_timeout_sec',
                'attitude_timeout_sec', 'lock_confirm_frames'):
        if values[key] <= 0:
            errors.append(f'invalid tracker parameter: {key}')
    for key in ('kp_yaw', 'kp_pitch', 'zoom_gain', 'search_resume_timeout_sec'):
        if values[key] < 0:
            errors.append(f'invalid tracker parameter: {key}')
    for key in ('yaw_sign', 'pitch_sign'):
        if values[key] not in (-1., 1.):
            errors.append(f'invalid tracker parameter: {key}')
    for key, valid in (
            ('deadband', 0 <= values['deadband'] < 1),
            ('max_rate', 0 < values['max_rate'] <= 100),
            ('search_yaw_rate', 0 <= values['search_yaw_rate'] <= 100),
            ('search_pitch_rate', 0 <= values['search_pitch_rate'] <= 100),
            ('search_period_sec', values['search_period_sec'] >= 2),
            ('search_yaw_limit_deg', 0 < values['search_yaw_limit_deg'] <= 180),
            ('search_pitch_bounds', -90 <= values['search_pitch_min_deg']
             < values['search_pitch_max_deg'] <= 90)):
        if not valid:
            errors.append(f'invalid tracker parameter: {key}')
    return errors
