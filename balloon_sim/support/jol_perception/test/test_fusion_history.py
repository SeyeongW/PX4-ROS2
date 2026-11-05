"""Delayed bearing updates checked against an independent augmented Gaussian KF."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm

from jol_perception.fusion_core import fuse, project_tracks
from jol_perception.fusion_history import replay_camera_track, reanchor_camera_seed
from jol_perception.tracking_core import predict_cv, CV_NOISE_MODEL, LEGACY_CV_NOISE_MODEL


def update(state, covariance, measurement, observation, noise):
    residual = observation @ covariance @ observation.T + noise
    gain = covariance @ observation.T @ np.linalg.inv(residual)
    correction = np.eye(len(state))-gain @ observation
    return (state+gain @ (measurement-observation @ state),
            correction @ covariance @ correction.T+gain @ noise @ gain.T)


def fixture(exposure=10.04, noise_model=LEGACY_CV_NOISE_MODEL):
    state = np.array([0., 0., 10., .5, -.2, 0.])
    prior = np.diag([.4]*3+[.8]*3)
    prediction = dict(horizon_s=.5, acceleration_std=2.)
    if noise_model == CV_NOISE_MODEL:
        prediction.update(process_noise_model=noise_model, acceleration_reference_dt_s=.1,
                          acceleration_spectral_density_m2_s3=.4)
    anchor = dict(id=7, birth_stamp=9., position=state[:3].tolist(), velocity=state[3:].tolist(),
                  size=[.6, .6, .4], state_covariance=prior.tolist(), last_seen=10., observed=True,
                  prediction=prediction, prediction_start_stamp=9.9, process_acceleration_std=2.)
    if noise_model == CV_NOISE_MODEL:
        anchor['process_noise_model'] = noise_model
    history = [dict(stamp=10., frame_id='world', tracks=[anchor])]
    info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[], width=640, height=480)
    projection = project_tracks(history[0], exposure, np.eye(4), info)[0][0]
    center = np.asarray(projection['camera_position'])
    pixel = np.array([500*center[0]/center[2]+320, 500*center[1]/center[2]+240])
    detection = dict(bbox=[pixel[0]-10+2, pixel[1]-10-1, 20., 20.], class_name='drone', confidence=.9)
    seed = fuse([projection], [detection], np.eye(4), info)[0]
    assert seed['fusion_update']['accepted']
    for stamp in [10.1, 10.2]:
        state, prior = predict_cv(state, prior, .1, 2., noise_model)
        measurement = np.array([.5*(stamp-10)+.015, -.2*(stamp-10), 10.01])
        state, prior = update(state, prior, measurement, np.eye(3, 6), np.eye(3)*.12**2)
        track = dict(anchor, position=state[:3].tolist(), velocity=state[3:].tolist(),
                     state_covariance=prior.tolist(), measurement_stamp=stamp,
                     measurement_position=measurement.tolist(), measurement_covariance=(np.eye(3)*.12**2).tolist(),
                     prediction_start_stamp=stamp-.1, last_seen=stamp)
        history.append(dict(stamp=stamp, frame_id='world', tracks=[track]))
    return seed, history


def test_continuous_replay_validates_seed_anchor_and_later_prediction_metadata():
    for location in ('seed', 'anchor', 'latest'):
        seed, history = fixture(noise_model=CV_NOISE_MODEL)
        target = seed if location == 'seed' else history[0 if location == 'anchor' else -1]['tracks'][0]
        target['prediction'] = dict(target['prediction'], acceleration_spectral_density_m2_s3=4.)
        with pytest.raises(ValueError, match='density'):
            replay_camera_track(seed, history)


@pytest.mark.parametrize('exposure', [10., 10.04, 10.09])
def test_replay_matches_joint_gaussian_camera_then_raw_lidar(exposure):
    seed, history = fixture(exposure)
    anchor = history[0]['tracks'][0]
    state = np.r_[anchor['position'], anchor['velocity'], np.zeros(3)]
    uncertainty = np.zeros((9, 9))
    uncertainty[:6, :6] = anchor['state_covariance']
    uncertainty[6:, 6:] = np.eye(3)*4.
    offset = exposure-10.
    exposure_map = np.c_[np.block([[np.eye(3), np.eye(3)*offset], [np.zeros((3, 3)), np.eye(3)]]),
                         np.vstack((np.eye(3)*offset**2/2, np.eye(3)*offset))]
    evidence = seed['fusion_update']
    observation = np.asarray(evidence['observation_matrix']) @ exposure_map
    measurement = observation @ state+evidence['innovation_px']
    state, uncertainty = update(state, uncertainty, measurement, observation, np.array(evidence['measurement_covariance']))
    interval = .1
    end_map = np.c_[np.block([[np.eye(3), np.eye(3)*interval], [np.zeros((3, 3)), np.eye(3)]]),
                    np.vstack((np.eye(3)*interval**2/2, np.eye(3)*interval))]
    state, uncertainty = end_map @ state, end_map @ uncertainty @ end_map.T+np.eye(6)*1e-6
    for index, packet in enumerate(history[1:]):
        if index:
            state, uncertainty = predict_cv(state, uncertainty, .1, 2., LEGACY_CV_NOISE_MODEL)
        track = packet['tracks'][0]
        state, uncertainty = update(state, uncertainty, np.array(track['measurement_position']),
                                    np.eye(3, 6), np.array(track['measurement_covariance']))
    result = replay_camera_track(seed, history)
    assert np.allclose(np.r_[result['current_state']['position'], result['current_state']['velocity']], state, atol=1e-10)
    assert np.allclose(result['current_state']['state_covariance'], uncertainty, atol=1e-10)
    assert result['current_state_stamp'] == 10.2
    assert result['position'] == seed['position'] and result['stamp'] == exposure
    assert result['temporal_fusion']['replayed_measurement_stamps'] == [10.1, 10.2]


def test_duplicate_packets_and_repeated_calls_do_not_duplicate_information():
    seed, history = fixture()
    original = deepcopy((seed, history))
    expected = replay_camera_track(seed, history)
    assert replay_camera_track(seed, history+history[1:]) == expected
    assert replay_camera_track(seed, history) == expected
    assert (seed, history) == original
    # Later filtered posteriors must never be used as extra measurements.
    history[1]['tracks'][0]['position'] = [900., 900., 900.]
    history[1]['tracks'][0]['state_covariance'] = (np.eye(6)*900).tolist()
    assert replay_camera_track(seed, history) == expected


def test_gap_reused_identity_and_invalid_raw_measurement_are_rejected():
    seed, history = fixture()
    with pytest.raises(ValueError, match='transition_gap'):
        replay_camera_track(seed, [history[0], history[2]])
    history[1]['tracks'][0]['birth_stamp'] = 10.05
    with pytest.raises(ValueError, match='identity_reused'):
        replay_camera_track(seed, history)
    seed, history = fixture()
    history[1]['tracks'][0]['measurement_stamp'] = 10.
    with pytest.raises(ValueError, match='repeated_lidar_measurement'):
        replay_camera_track(seed, history)


def test_coasting_propagates_without_inventing_a_measurement():
    seed, history = fixture()
    history[-1]['tracks'][0].update(observed=False, measurement_stamp=None,
                                    measurement_position=None, measurement_covariance=None)
    result = replay_camera_track(seed, history)
    assert result['last_measurement_stamp'] == 10.1
    assert result['current_state_stamp'] == 10.2
    assert result['temporal_fusion']['replayed_measurements'] == 1


@pytest.mark.parametrize('exposure', [10., 10.04, 10.09])
def test_continuous_replay_matches_independent_joint_gaussian(exposure):
    # 노출과 다음 LiDAR 시각의 결합분포를 독립 행렬지수로 구성한다.
    seed, history = fixture(exposure, CV_NOISE_MODEL)
    anchor = history[0]['tracks'][0]
    generator = np.zeros((6, 6)); generator[:3, 3:] = np.eye(3)
    density = np.diag([0.]*3+[.4]*3)
    def discretize(dt):
        block = expm(np.block([[generator, density], [np.zeros((6, 6)), -generator.T]])*dt)
        return block[:6, :6], block[:6, 6:] @ block[:6, :6].T
    early, early_noise = discretize(exposure-10.)
    late, late_noise = discretize(10.1-exposure)
    prior_state = early @ np.r_[anchor['position'], anchor['velocity']]
    prior_covariance = early @ np.array(anchor['state_covariance']) @ early.T + early_noise
    cross = prior_covariance @ late.T
    joint_state = np.r_[prior_state, late @ prior_state]
    joint_covariance = np.block([[prior_covariance, cross],
                                 [cross.T, late @ prior_covariance @ late.T + late_noise]])
    evidence = seed['fusion_update']
    observation = np.c_[np.asarray(evidence['observation_matrix']), np.zeros((2, 6))]
    measurement = observation @ joint_state + evidence['innovation_px']
    joint_state, joint_covariance = update(joint_state, joint_covariance, measurement, observation,
                                          np.asarray(evidence['measurement_covariance']))
    state, uncertainty = joint_state[6:], joint_covariance[6:, 6:]
    for index, packet in enumerate(history[1:]):
        if index:
            transition, noise = discretize(.1)
            state, uncertainty = transition @ state, transition @ uncertainty @ transition.T + noise
        track = packet['tracks'][0]
        state, uncertainty = update(state, uncertainty, np.array(track['measurement_position']),
                                    np.eye(3, 6), np.array(track['measurement_covariance']))
    result = replay_camera_track(seed, history)
    assert np.allclose(np.r_[result['current_state']['position'], result['current_state']['velocity']], state, atol=1e-10)
    assert np.allclose(result['current_state']['state_covariance'], uncertainty, atol=1e-10)
    assert result['temporal_fusion']['process_bridge'] == 'continuous_markov_increments'
    assert result['current_state']['prediction']['process_noise_model'] == CV_NOISE_MODEL
    history[-1]['tracks'][0]['process_noise_model'] = LEGACY_CV_NOISE_MODEL
    with pytest.raises(ValueError, match='inconsistent process-noise model'):
        replay_camera_track(seed, history)
    history[-1]['tracks'][0]['process_noise_model'] = CV_NOISE_MODEL
    history[0]['tracks'][0]['process_noise_model'] = LEGACY_CV_NOISE_MODEL
    with pytest.raises(ValueError, match='inconsistent process-noise model'):
        replay_camera_track(seed, history)


def test_future_scan_is_retained_but_replayed_only_after_clock_catches_up():
    import json
    from types import SimpleNamespace, MethodType
    from jol_perception.camera_fusion import CameraFusion
    from std_msgs.msg import String

    seed, history = fixture()
    now, outputs = [10.04], []
    node = SimpleNamespace(history=history[:1], tolerance=.15, max_age=.8,
                           fusion_seed=dict(stamp=10.04, tracks=[seed]),
                           last_replay_key=(10.04, 0, 10.), replay_publications=0, fusion_context=None,
                           check_clock=lambda: now[0], timing_point=lambda: {'ros_s': now[0]},
                           check_calibration=lambda: None, health=lambda now: {'ready': True},
                           fused_pub=SimpleNamespace(publish=lambda msg: outputs.append(json.loads(msg.data))))
    node.publish_replay = MethodType(CameraFusion.publish_replay, node)
    node.replay_result = MethodType(CameraFusion.replay_result, node)
    CameraFusion.receive_tracks(node, String(data=json.dumps(history[1])))
    assert len(node.history) == 2 and not outputs
    now[0] = 10.1
    node.publish_replay(now[0])
    assert outputs[-1]['tracks'][0]['current_state_stamp'] == 10.1
    node.publish_replay(now[0])
    assert len(outputs) == 1


@pytest.mark.parametrize('exposure', [10.04, 10.1])
def test_late_causal_measurement_reanchors_from_lidar_only_and_preserves_original(exposure):
    seed, history = fixture(exposure)
    info = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[], width=640, height=480)
    # Inject a raw LiDAR measurement that existed before exposure but was not
    # available when inference completed. A new causal prior incorporates it.
    anchor = history[0]['tracks'][0]
    stamp = exposure
    state, uncertainty = predict_cv(np.r_[anchor['position'], anchor['velocity']],
                                    np.array(anchor['state_covariance']), stamp-10., 2., LEGACY_CV_NOISE_MODEL)
    measured = np.array([.5*(stamp-10)+.015, -.2*(stamp-10), 10.01])
    state, uncertainty = update(state, uncertainty, measured, np.eye(3, 6), np.eye(3)*.12**2)
    latest = dict(anchor, position=state[:3].tolist(), velocity=state[3:].tolist(),
                  state_covariance=uncertainty.tolist(), last_seen=stamp,
                  prediction_start_stamp=10., measurement_stamp=stamp,
                  measurement_position=measured.tolist(), measurement_covariance=(np.eye(3)*.12**2).tolist())
    arrived = [history[0], dict(stamp=stamp, frame_id='world', stabilized=True, tracks=[latest])]
    detection = dict(bbox=seed['camera_bbox'], class_name='drone', confidence=.9)
    original = dict(stamp=exposure, lidar_stamp=10., frame_id='world', tracks=[seed], fusion_revision=0)
    unchanged = deepcopy(original)
    revised = reanchor_camera_seed(original, arrived, np.eye(4), info, [detection])
    assert original == unchanged and revised['fusion_revision'] == 1
    assert revised['publication_kind'] == 'lidar_replay' and revised['stamp'] == exposure
    assert revised['lidar_stamp'] == exposure and revised['original_lidar_stamp'] == 10.
    expected = fuse(project_tracks(arrived[-1], exposure, np.eye(4), info)[0],
                    [detection], np.eye(4), info)[0]
    branch = replay_camera_track(revised['tracks'][0], arrived)
    assert branch['last_measurement_stamp'] == exposure
    assert branch['current_state_stamp'] == exposure  # Revision changes, time need not.
    assert np.allclose(branch['current_state']['position'], expected['position'])
    assert np.allclose(branch['current_state']['state_covariance'], expected['state_covariance'])
    assert branch['temporal_fusion']['replayed_measurements'] == 0  # Already in new prior.
    assert reanchor_camera_seed(revised, arrived, np.eye(4), info, [detection]) is revised
    # No accepted camera prior is reused: repeated recomputation has identical P.
    twice = reanchor_camera_seed(original, arrived, np.eye(4), info, [detection])
    assert twice == revised
    arrived[-1]['tracks'][0]['birth_stamp'] = exposure
    with pytest.raises(ValueError, match='identity_reused'):
        reanchor_camera_seed(original, arrived, np.eye(4), info, [detection])
