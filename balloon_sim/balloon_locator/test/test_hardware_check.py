"""Hardware preparation checks have no device/network side effects."""
import json
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import pytest
import yaml
from launch import LaunchContext
from launch_ros.actions import Node

from balloon_locator.hardware_check import inspect_configuration, load_parameters, main

PACKAGE = Path(__file__).parents[1]
CONFIG = PACKAGE / 'config/balloon_locator.yaml'


def test_shipped_profile_reports_missing_measurements_and_40m_size_limit(capsys):
    with patch('socket.socket', side_effect=AssertionError('network forbidden')):
        assert main(['--config', str(CONFIG)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['configuration_valid']
    assert not report['sensor_estimation_configured']
    assert not report['hardware_connected'] and not report['flight_ready']
    assert report['metrics']['expected_diameter_px'] == pytest.approx(8.5806, rel=1e-4)
    assert report['metrics']['common_radial_sigma_floor_m'] > 2.
    assert any('uncertainty' in reason for reason in report['blockers'])


@pytest.mark.parametrize('node,key,value', [
    ('siyi_camera_node','use_sim_time', True),
    ('siyi_camera_node','latency_sec', 2.),
    ('siyi_gimbal_node','port', 37260.5),
    ('siyi_gimbal_node','port', 37260.0),
    ('siyi_gimbal_node','ip', 'not-an-ip'),
    ('siyi_gimbal_node','ip', 3232272409),
    ('siyi_camera_node','rtsp_url', 1),
    ('balloon_locator_node','require_camera_info', False),
    ('balloon_locator_node','max_sigma_m', float('nan')),
    ('balloon_locator_node','pose_source', 'tf'),
    ('gimbal_tracker_node','image_width', 640.),
    ('gimbal_tracker_node','yaw_sign', 0.),
    ('gimbal_tracker_node','max_rate', -1.),
    ('balloon_locator_node','pixel_sigma_px', 0.),
    ('balloon_detector_node','detections_topic', '/wrong/topic'),
    ('balloon_detector_node','min_circularity', float('nan')),
    ('balloon_detector_node','max_detections', 0),
])
def test_bad_wiring_and_values_are_rejected(node, key, value):
    params = load_parameters(CONFIG)
    params[node][key] = value
    report = inspect_configuration(params)
    assert report['errors'] and not report['configuration_valid']
    json.dumps(report, allow_nan=False)


def test_nan_range_can_be_reported_without_json_exception():
    report = inspect_configuration(load_parameters(CONFIG), float('nan'))
    assert report['errors'] and report['metrics']['evaluation_range_m'] is None
    json.dumps(report, allow_nan=False)


def test_bridge_requires_measured_gps_not_sensor_defaults():
    report = inspect_configuration(load_parameters(CONFIG), include_bridge=True)
    assert not report['configuration_valid']
    assert any('cam_lat_deg' in item for item in report['errors'])


def test_equal_but_empty_image_topics_cannot_start_hardware():
    params = load_parameters(CONFIG)
    for name in ('siyi_camera_node', 'balloon_detector_node'):
        params[name]['image_topic'] = ''
    assert not inspect_configuration(params)['configuration_valid']


def test_relative_calibration_path_is_relative_to_parameter_file(tmp_path):
    document = yaml.safe_load(CONFIG.read_text())
    document['siyi_camera_node']['ros__parameters']['calibration_file'] = 'measured.yaml'
    path = tmp_path/'hardware.yaml';path.write_text(yaml.safe_dump(document))
    params = load_parameters(path)
    assert params['siyi_camera_node']['calibration_file'] == str(tmp_path/'measured.yaml')


def test_missing_sensor_sections_do_not_silently_enable_default_hardware(tmp_path):
    path = tmp_path/'empty.yaml';path.write_text('{}')
    with pytest.raises(ValueError, match='missing sensor'):
        load_parameters(path)


def test_synthetic_calibrated_config_can_pass_without_claiming_real_flight_ready(tmp_path):
    # This is a synthetic test fixture, never an installed hardware calibration.
    camera = dict(image_width=1920, image_height=1080, distortion_model='plumb_bob',
        camera_matrix=dict(rows=3, cols=3, data=[6000.,0.,960.,0.,6000.,540.,0.,0.,1.]),
        distortion_coefficients=dict(rows=1, cols=5, data=[0.]*5),
        rectification_matrix=dict(rows=3, cols=3, data=[1.,0.,0.,0.,1.,0.,0.,0.,1.]),
        projection_matrix=dict(rows=3, cols=4, data=[6000.,0.,960.,0.,0.,6000.,540.,0.,0.,0.,1.,0.]))
    path = tmp_path/'synthetic.yaml';path.write_text(yaml.safe_dump(camera))
    params = load_parameters(CONFIG)
    params['siyi_camera_node']['calibration_file'] = str(path)
    params['balloon_locator_node'].update(calibration_valid=True, diameter_rel_sigma=.002,
                                         focal_rel_sigma=.001, diameter_bias_sigma_px=.1)
    report = inspect_configuration(params)
    assert report['sensor_estimation_configured'] and report['configuration_valid']
    assert not report['hardware_connected'] and not report['flight_ready']
    assert report['metrics']['expected_diameter_px'] == 45.


def test_ground_station_has_only_five_sensor_nodes_and_rejects_bad_config(tmp_path):
    module = run_path(str(PACKAGE/'launch/ground_station.launch.py'))
    context = LaunchContext();context.launch_configurations['config'] = str(CONFIG)
    actions = module['setup'](context)
    assert len([action for action in actions if isinstance(action, Node)]) == 5
    document = yaml.safe_load(CONFIG.read_text())
    document['balloon_locator_node']['ros__parameters']['pose_source'] = 'tf'
    path = tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(document))
    context.launch_configurations['config'] = str(path)
    with pytest.raises(ValueError, match='pose_source=gimbal'):
        module['setup'](context)
