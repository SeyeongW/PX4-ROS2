"""Checks for public Kalman motion estimates and non-mutating future forecasts."""
from collections import deque
import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

from jol_perception.tracking_core import (Config, Track, Tracker, predict_cv, forecast_model,
    CV_NOISE_MODEL, LEGACY_CV_NOISE_MODEL, ACCELERATION_REFERENCE_DT)


def test_motion_metadata_defaults_only_for_untagged_legacy_and_normalizes_malformed_numbers():
    assert forecast_model({}, legacy_default_std=8.) == (8., LEGACY_CV_NOISE_MODEL)
    for metadata in (None, {'prediction': None},
                     {'prediction': {'process_noise_model': LEGACY_CV_NOISE_MODEL}},
                     {'process_noise_model': LEGACY_CV_NOISE_MODEL, 'prediction': {}},
                     {'prediction': {'acceleration_std': 10**400}},
                     {'prediction': {'acceleration_std': 1e200}},
                     {'prediction': {'process_noise_model': CV_NOISE_MODEL, 'acceleration_std': 2.,
                       'acceleration_reference_dt_s': 10**400, 'acceleration_spectral_density_m2_s3': .4}}):
        with pytest.raises(ValueError):
            forecast_model(metadata, legacy_default_std=8.)


def test_prediction_covariance_and_purity():
    state = np.array([1., 2., 3., 3., 4., 2.])
    covariance = np.eye(6) * .4
    covariance[:3, 3:] = covariance[3:, :3] = np.eye(3) * .1
    original_state, original_covariance = state.copy(), covariance.copy()
    future, uncertainty = predict_cv(state, covariance, .5, 2.)
    assert np.allclose(future, [2.5, 4., 4., 3., 4., 2.])
    # 연속 백색가속도의 위치 분산 증가량은 q·h³/3이다.
    expected_position_variance = .4 + .5 * .2 + .5 ** 2 * .4 + 2. ** 2 * .1 * .5 ** 3 / 3
    assert np.allclose(uncertainty[:3, :3], np.eye(3) * expected_position_variance)
    assert np.linalg.eigvalsh(uncertainty).min() >= 0
    assert np.array_equal(state, original_state) and np.array_equal(covariance, original_covariance)
    now, now_cov = predict_cv(state, covariance, 0., 2.)
    assert np.array_equal(now, state) and np.array_equal(now_cov, covariance)
    assert not np.shares_memory(now, state) and not np.shares_memory(now_cov, covariance)
    for dt, sigma in [(-.1, 2.), (np.nan, 2.), (.5, 0.)]:
        try:
            predict_cv(state, covariance, dt, sigma)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid forecast parameters accepted')


def test_continuous_noise_matches_van_loan_and_is_independent_of_time_partition():
    # 별도 행렬지수 이산화로 폐형식 Q를 검사한다. 카메라 중간시각 삽입도 동일해야 한다.
    state, covariance = np.arange(6, dtype=float), np.eye(6)*.3
    generator = np.zeros((6, 6)); generator[:3, 3:] = np.eye(3)
    density = np.diag([0.]*3+[8.**2*ACCELERATION_REFERENCE_DT]*3)
    for dt in (.001, .1, .5, 1.):
        block = np.block([[generator, density], [np.zeros((6, 6)), -generator.T]])
        integrated = expm(block*dt)
        transition = integrated[:6, :6]
        expected = transition @ covariance @ transition.T + integrated[:6, 6:] @ transition.T
        direct = predict_cv(state, covariance, dt, 8.)
        assert np.allclose(direct[1], expected, atol=1e-12)
        split = predict_cv(*predict_cv(state, covariance, dt*.4, 8.), dt*.6, 8.)
        assert np.allclose(split[0], direct[0], atol=1e-12)
        assert np.allclose(split[1], direct[1], atol=1e-12)
    old = predict_cv(state, covariance, .5, 8., LEGACY_CV_NOISE_MODEL)
    assert not np.allclose(old[1], predict_cv(state, covariance, .5, 8.)[1])
    try:
        predict_cv(state, covariance, .1, 8., 'unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown process model accepted')


def test_nis_is_a_real_accepted_measurement_diagnostic_not_a_coast_measurement():
    tracker = Tracker(Config(min_hits=1))
    measurement = dict(position=np.array([20., 0., 8.]), size=np.ones(3),
        nearest_point=np.array([20., 0., 8.]), covariance=np.eye(3)*.04,
        sensor_origin=np.zeros(3), point_count=8, strong=True)
    tracker._associate([measurement], 1., 0.)
    assert tracker.tracks[0].as_dict(1.)['innovation_nis'] is None
    before = tracker.tracks[0]
    state, covariance = predict_cv(before.state, before.covariance, .1, before.acceleration_noise(tracker.config))
    measurement['position'] = np.array([20.1, .05, 8.])
    residual = measurement['position']-state[:3]
    expected = residual @ np.linalg.solve(covariance[:3, :3]+measurement['covariance'], residual)
    tracker._associate([measurement], 1.1, .1)
    row = tracker.tracks[0].as_dict(1.1)
    assert np.isclose(row['innovation_nis'], expected)
    assert row['innovation_stamp'] == 1.1 and row['innovation_dof'] == 3
    assert row['prediction']['process_noise_model'] == CV_NOISE_MODEL
    tracker._associate([], 1.2, .1)
    coast = tracker.tracks[0].as_dict(1.2)
    assert coast['innovation_nis'] is None and coast['innovation_stamp'] is None


def test_track_motion_schema_and_coast_time():
    state = np.array([1., 2., 3., 3., 4., 2.])
    covariance = np.eye(6) * .4
    track = Track(7, state.copy(), covariance.copy(), np.ones(3), born=1., last_seen=9.,
                  hits=4, is_confirmed=True, observed=False,
                  trail=deque([np.array([1., 2., 3.])]))
    packet = track.as_dict(10., config=Config(prediction_horizon=.5))
    assert packet['stamp'] == 10. and packet['last_seen'] == 9.
    assert packet['time_since_update'] == 1. and not packet['observed'] and packet['confirmed']
    assert packet['prediction']['stamp'] == 10.5  # Based on current state, not last measurement.
    assert packet['prediction']['position'] == [2.5, 4., 4.]
    assert packet['prediction']['acceleration_std'] == 8.
    tuned = track.as_dict(10., config=Config(acceleration_std=4., low_speed_acceleration_std=.7,
                                           low_speed_threshold=20.))
    # A confirmed coast cannot use an old low-speed estimate as evidence that
    # the object remained stationary after its last actual measurement.
    assert tuned['prediction']['acceleration_std'] == 4.
    expected_state, expected_covariance = predict_cv(state, covariance, .5, 4.)
    assert np.allclose(tuned['prediction']['position_covariance'], expected_covariance[:3, :3])
    track.observed = True
    observed = track.as_dict(10., config=Config(acceleration_std=4., low_speed_acceleration_std=.7,
                                              low_speed_threshold=20.))
    assert observed['prediction']['acceleration_std'] == .7
    track.observed = False
    assert packet['nearest_distance'] is None
    assert math.isclose(packet['speed_mps'], math.sqrt(29))
    assert np.allclose(packet['direction'], [3., 4., 2.] / np.sqrt(29))
    assert math.isclose(packet['heading_rad'], math.atan2(4, 3))
    assert math.isclose(packet['elevation_rad'], math.atan2(2, 5))
    assert np.shape(packet['state_covariance']) == (6, 6)
    assert np.shape(packet['velocity_covariance']) == (3, 3)
    assert np.array_equal(track.state, state) and np.array_equal(track.covariance, covariance)
    assert len(track.trail) == 1 and track.hits == 4 and track.last_seen == 9.
    json.dumps(packet, allow_nan=False)
    track.state[3:] = [.1, 0., 0.]
    slow = track.as_dict(10.)
    assert slow['direction'] is None and slow['heading_rad'] is None and slow['elevation_rad'] is None
    assert slow['velocity'] == [.1, 0., 0.] and slow['prediction']['position'][0] == 1.05
    track.state[3:] = [0., 0., 1.]
    vertical = track.as_dict(10.)
    assert vertical['direction'] == [0., 0., 1.] and vertical['heading_rad'] is None
    assert vertical['elevation_rad'] == np.pi / 2
    track.state[3:] = 0.
    assert track.as_dict(10., config=Config(direction_min_speed=0.))['direction'] is None
    for config in [Config(prediction_horizon=-1.), Config(direction_min_speed=np.nan)]:
        try:
            Tracker(config)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid prediction configuration accepted')


def test_forecast_from_tracking_and_missing_observation():
    rng = np.random.default_rng(814)
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    shape = rng.uniform([-.4, -.4, -.2], [.4, .4, .2], (80, 3))
    tracker = Tracker(Config(voxel_size=.02, prediction_horizon=.5))
    tracker.set_background(background)
    velocity = np.array([3., 4., 1.])
    for index in range(35):
        stamp = 1. + index * .1
        center = np.array([25., 0., 8.]) + velocity * (stamp - 1.)
        tracks, _ = tracker.process(np.vstack((background, center + shape)), stamp)
        if index >= 10:
            track = tracks[0]
            assert np.linalg.norm(np.array(track['velocity']) - velocity) < .03
            truth_future = center + velocity * .5
            assert np.linalg.norm(np.array(track['prediction']['position']) - truth_future) < .05
    track = tracks[0]
    before_covariance = np.array(track['position_covariance'])
    last_seen = track['last_seen']
    for index in range(1, 6):
        coasted, _ = tracker.process(background, stamp + index * .1)
    coasted = coasted[0]
    assert coasted['id'] == track['id'] and not coasted['observed']
    assert math.isclose(coasted['time_since_update'], .5) and coasted['last_seen'] == last_seen
    assert np.trace(coasted['position_covariance']) > np.trace(before_covariance)
    assert coasted['prediction']['stamp'] == coasted['stamp'] + .5
    assert coasted['nearest_point'] is None


def test_prediction_markers_preserve_existing_geometry():
    from geometry_msgs.msg import Point
    from std_msgs.msg import Header
    from jol_perception.tracker_node import TrackerNode
    track = Track(1, np.array([10., 3., 4., 5., 0., 0.]), np.eye(6), np.ones(3),
                  born=1., last_seen=2., hits=3, nearest_point=np.array([9.5, 3., 4.]),
                  trail=deque([np.array([9., 3., 4.]), np.array([10., 3., 4.])]))
    packet = track.as_dict(2.)
    markers = TrackerNode.make_markers(None, Header(frame_id='world'), [packet], np.array([1., 0., 2.]))
    by_namespace = {marker.ns: marker for marker in markers.markers}
    assert {'boxes', 'labels', 'trails', 'surface_distance', 'velocity_prediction',
            'predicted_position', 'prediction_labels'} <= set(by_namespace)
    assert by_namespace['surface_distance'].points[0] == Point(x=1., y=0., z=2.)
    assert by_namespace['velocity_prediction'].points[-1] == Point(x=12.5, y=3., z=4.)
    assert by_namespace['predicted_position'].pose.position == Point(x=12.5, y=3., z=4.)
    assert 'predicted' in by_namespace['prediction_labels'].text
