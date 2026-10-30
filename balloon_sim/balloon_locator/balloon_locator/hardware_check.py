"""Offline configuration checks. No ROS node, camera, serial port or network IO."""
import argparse
import json
import math
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from .geometry import focal_px

SENSOR_NODES = ('siyi_camera_node', 'siyi_gimbal_node', 'balloon_detector_node',
                'gimbal_tracker_node', 'balloon_locator_node')


def load_parameters(path):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError('ROS parameter YAML mapping required')
    parameters = {}
    for name in (*SENSOR_NODES, 'mavlink_bridge_node'):
        if name in SENSOR_NODES and name not in document:
            raise ValueError(f'missing sensor parameter section: {name}')
        section = document.get(name, {}).get('ros__parameters', {})
        if not isinstance(section, dict):
            raise ValueError(f'{name}: ros__parameters mapping required')
        parameters[name] = dict(section)
    calibration = parameters['siyi_camera_node'].get('calibration_file', '')
    if calibration:
        p = Path(calibration).expanduser()
        parameters['siyi_camera_node']['calibration_file'] = str(
            (p if p.is_absolute() else path.parent / p).resolve())
    return parameters


def inspect_configuration(parameters, range_m=40., include_bridge=False):
    """Validate wiring and report an optimistic 40m uncertainty budget.

    `range_m` is camera-to-target distance. With a target 40m horizontally away
    and above the camera, 40m is an optimistic lower bound, not the slant range.
    Passing this check never certifies measured calibration or flight readiness.
    """
    camera, gimbal, detector, tracker, locator = (parameters[n] for n in SENSOR_NODES)
    bridge = parameters['mavlink_bridge_node']
    errors, blockers, warnings = [], [], []
    metrics = {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    def number(section, key, default, lower=None, upper=None, positive=False):
        value = section.get(key, default)
        # rclpy infers the static parameter type from the declared default.
        valid = type(value) is type(default) and math.isfinite(value)
        valid = valid and (not positive or value > 0)
        valid = valid and (lower is None or value >= lower) and (upper is None or value <= upper)
        require(valid, f'invalid parameter: {key}')
        return float(value) if valid else None

    require(type(range_m) in (int, float) and math.isfinite(range_m) and range_m > 0,
            'range_m must be finite and positive')
    metrics['evaluation_range_m'] = range_m if not errors else None
    for name in SENSOR_NODES:
        require(parameters[name].get('use_sim_time', False) is False, f'{name}: hardware requires use_sim_time=false')
    require(type(locator.get('calibration_valid', False)) is bool, 'calibration_valid must be boolean')
    require(locator.get('pose_source', 'gimbal') == 'gimbal', 'hardware requires pose_source=gimbal')
    require(locator.get('frame_id', 'local_ned') == 'local_ned', 'hardware requires local_ned frame')
    require(locator.get('require_camera_info', True) is True, 'hardware requires calibrated CameraInfo')
    require(locator.get('gimbal_rotation_order', 'zxy') == 'zxy', 'SIYI rotation order must be zxy (312)')
    require(camera.get('image_topic', '/camera/image_raw') == detector.get('image_topic', '/camera/image_raw'), 'camera/detector image topics differ')
    require(camera.get('camera_info_topic', '/camera/camera_info') == locator.get('camera_info_topic', '/camera/camera_info'), 'camera/locator CameraInfo topics differ')
    require(detector.get('detections_topic', '/balloon/detections') == '/balloon/detections', 'tracker/locator consume /balloon/detections')
    for section, key, default in ((camera, 'image_topic', '/camera/image_raw'),
            (camera, 'camera_info_topic', '/camera/camera_info'),
            (locator, 'position_topic', '/balloon/position_ned')):
        topic = section.get(key, default)
        require(isinstance(topic, str) and re.fullmatch(r'(?:/[A-Za-z_][A-Za-z_0-9]*)+', topic) is not None,
                f'absolute ROS topic required: {key}')
    for axis in ('yaw', 'pitch', 'roll'):
        sign = locator.get(axis + '_sign', 1.)
        require(type(sign) in (int, float) and sign in (-1., 1.), f'invalid {axis}_sign')
        number(locator, axis + '_zero_deg', 0.)
    number(locator, 'base_heading_deg', 0.)
    offset = locator.get('offset_ned', [0., 0., 0.])
    require(isinstance(offset, (list, tuple)) and len(offset) == 3 and
            all(type(v) in (float, int) and math.isfinite(v) for v in offset), 'invalid offset_ned')
    width = number(locator, 'image_width', 1920., positive=True)
    height = number(locator, 'image_height', 1080., positive=True)
    require(tracker.get('image_width', 1920.) == width and tracker.get('image_height', 1080.) == height,
            'locator/tracker image dimensions differ')
    age = number(locator, 'max_measurement_age_sec', .8, positive=True)
    latency = number(camera, 'latency_sec', 0., lower=0.)
    if age is not None and latency is not None:
        require(latency < age, 'camera latency exceeds accepted observation age')
    number(gimbal, 'attitude_latency_sec', 0., lower=0.)
    number(gimbal, 'poll_hz', 20., positive=True)
    number(gimbal, 'cmd_timeout_sec', .3, positive=True)
    for axis, default in (('yaw', 1.), ('pitch', -1.)):
        sign = tracker.get(axis + '_sign', default)
        require(type(sign) in (int, float) and sign in (-1., 1.), f'invalid tracker {axis}_sign')
        number(tracker, 'kp_' + axis, 60., lower=0.)
    number(tracker, 'zoom_gain', 1., lower=0.)
    deadband = number(tracker, 'deadband', .03, lower=0., upper=1.)
    require(deadband is not None and deadband < 1., 'tracker deadband must be below 1')
    number(tracker, 'max_rate', 60., positive=True, upper=100.)
    lost_timeout = number(tracker, 'lost_timeout_sec', .5, positive=True)
    if latency is not None and lost_timeout is not None:
        require(latency < lost_timeout, 'camera latency exceeds tracker observation timeout')
    for key, default in (('open_timeout_sec', 3.), ('read_timeout_sec', 1.),
                         ('reconnect_interval_sec', 1.), ('zoom_max_age_sec', 2.)):
        number(camera, key, default, positive=True)
    zoom = number(camera, 'calibration_zoom', 1., lower=1.)
    number(detector, 'process_scale', .5, lower=0., upper=1., positive=True)
    number(detector, 'max_detections', 3, positive=True)
    number(detector, 'min_circularity', .6, lower=0., upper=1.)
    number(detector, 'min_fill', .55, lower=0., upper=1.)
    minimum = number(detector, 'min_diameter_px', 6., positive=True)
    maximum = number(detector, 'max_diameter_px', 1000., positive=True)
    require(minimum is not None and maximum is not None and minimum <= maximum, 'invalid detector diameter bounds')
    try:
        from .balloon_detector import parse_hsv_ranges
        parse_hsv_ranges(detector.get('hsv_ranges', [0,120,70,10,255,255,170,120,70,180,255,255]))
    except (ValueError, TypeError, OverflowError):
        errors.append('invalid detector HSV ranges')
    try:
        url = urlsplit(camera.get('rtsp_url', 'rtsp://192.168.144.25:8554/main.264'))
        require(url.scheme in ('rtsp', 'rtsps') and bool(url.hostname), 'RTSP URL required')
        _ = url.port
    except (TypeError, ValueError, AttributeError):
        errors.append('invalid RTSP URL')
    require(isinstance(camera.get('frame_id', 'camera'), str) and bool(camera.get('frame_id', 'camera')), 'camera frame_id required')
    port = number(gimbal, 'port', 37260, lower=1, upper=65535)
    require(port is not None and port.is_integer(), 'gimbal port must be an integer')
    try:
        address = gimbal.get('ip', '192.168.144.25')
        if not isinstance(address, str):
            raise ValueError('IP must be a string')
        ipaddress.IPv4Address(address)
    except (ValueError, TypeError):
        errors.append('gimbal IPv4 address required')

    calibration = None
    calibration_path = camera.get('calibration_file', '')
    if not calibration_path:
        blockers.append('measured camera calibration_file is missing')
    else:
        try:
            from .camera_calibration import CameraCalibration
            calibration = CameraCalibration(calibration_path)
            require((width, height) == (calibration.width, calibration.height), 'calibration/locator/tracker dimensions differ')
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            errors.append('invalid camera calibration file: ' + type(error).__name__)
    if locator.get('calibration_valid', False) is not True:
        blockers.append('installation/optics/timing calibration is not confirmed')
    elif calibration is None:
        errors.append('calibration_valid=true requires a measured calibration file')

    diameter = number(locator, 'balloon_diameter_m', .30, positive=True)
    size_sigma = number(locator, 'diameter_rel_sigma', .05, lower=0.)
    focal_sigma = number(locator, 'focal_rel_sigma', .01, lower=0.)
    bias = number(locator, 'diameter_bias_sigma_px', .25, lower=0.)
    max_sigma = number(locator, 'max_sigma_m', .5, positive=True)
    min_pixels = number(locator, 'min_diameter_px', 8., positive=True)
    samples = number(locator, 'min_samples', 3, positive=True)
    require(samples is not None and samples.is_integer(), 'min_samples must be an integer')
    number(locator, 'pixel_sigma_px', 1., positive=True)
    for key, default in (('angle_sigma_deg', .2),
                         ('extrinsic_translation_std_m', .02), ('camera_time_std_s', .001)):
        number(locator, key, default, lower=0.)
    number(locator, 'max_range_m', 300., positive=True)
    number(locator, 'max_attitude_gap_sec', .3, positive=True)
    number(locator, 'forget', 1., positive=True, upper=1.)
    hfov = number(locator, 'hfov_deg_at_1x', 80., lower=.001, upper=179.999)
    if not errors:
        if calibration is None:
            focal = focal_px(width, hfov, zoom)
            warnings.append('pixel/uncertainty budget uses nominal FOV only; coordinates require measured CameraInfo')
        else:
            focal = float((calibration.p[0, 0] + calibration.p[1, 1]) / 2.)
        pixels = diameter * focal / range_m
        floor = range_m * math.sqrt(size_sigma**2 + focal_sigma**2 + (bias/pixels)**2)
        metrics.update(expected_diameter_px=pixels, common_radial_sigma_floor_m=floor,
                       accepted_sigma_m=max_sigma)
        if pixels < min_pixels:
            blockers.append('target is smaller than the locator minimum pixel diameter')
        if floor > max_sigma:
            blockers.append('systematic range uncertainty exceeds the coordinate acceptance limit')
    warnings.append('calibration/zoom/latency/axis signs are unmeasured until confirmed on the actual camera')
    warnings.append('hardware onboard control is separate from the Gazebo UI/3m/Bezier mission')
    if include_bridge:
        require(bridge.get('use_sim_time', False) is False, 'hardware bridge requires use_sim_time=false')
        require(offset == [0., 0., 0.], 'GPS bridge requires station offset_ned=[0,0,0]')
        require(bridge.get('topic', '/balloon/position_ned') == locator.get('position_topic', '/balloon/position_ned'), 'locator/bridge position topics differ')
        require(bridge.get('expect_frame_id', 'local_ned') == locator.get('frame_id', 'local_ned'), 'locator/bridge frames differ')
        for key, lower, upper in (('cam_lat_deg', -89.999999, 89.999999), ('cam_lon_deg', -180., 180.), ('cam_height_above_home_m', None, None)):
            number(bridge, key, float('nan'), lower=lower, upper=upper)
        number(bridge, 'declination_deg', 0.)
        for key, default in (('baud', 57600), ('min_samples', 30)):
            value = number(bridge, key, default, positive=True)
            require(value is not None and value.is_integer(), f'bridge {key} must be an integer')
        for key, default in (('rate_hz', 5.), ('max_age_sec', 1.), ('max_sigma_m', 2.)):
            number(bridge, key, default, positive=True)
        for key, default in (('target_system', 1), ('target_component', 191)):
            value = number(bridge, key, default, lower=1, upper=255)
            require(value is not None and value.is_integer(), f'{key} must be an integer')
    return dict(scope='offline_configuration_only', configuration_valid=not errors,
                sensor_estimation_configured=not errors and not blockers,
                hardware_connected=False, flight_ready=False, errors=errors,
                blockers=blockers, warnings=warnings, metrics=metrics)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='ROS parameter YAML; no device will be opened')
    parser.add_argument('--range-m', type=float, default=40., help='assumed camera-to-target distance for the optimistic error budget')
    parser.add_argument('--bridge', action='store_true', help='also check surveyed camera GPS and MAVLink routing')
    options = parser.parse_args(args)
    try:
        report = inspect_configuration(load_parameters(options.config), options.range_m, options.bridge)
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        parser.exit(2, 'Invalid configuration file. Check its path and ROS YAML structure.\n')
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report['sensor_estimation_configured'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
