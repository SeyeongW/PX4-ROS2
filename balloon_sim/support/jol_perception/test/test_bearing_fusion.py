"""Independent bearing information, covariance, rotating-frame and clock checks."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from jol_perception.fusion_core import BearingConfig, association_score, bearing_update, fuse, project_tracks


INFO = dict(k=[500., 0., 320., 0., 500., 240., 0., 0., 1.], d=[0.]*5,
            width=640, height=480, frame_id='camera')


def fixture(position=(.35, -.25, 10.), stamp=1., exposure=1.):
    track = dict(id=3, position=list(position), velocity=[0., 0., 0.], size=[2., 2., 2.],
                 observed=True, confirmed=True, state_covariance=np.diag([.16]*3+[1.]*3).tolist(),
                 prediction=dict(acceleration_std=2., horizon_s=.5), nearest_point=[0., 0., 9.],
                 nearest_distance=9.)
    packet = dict(stamp=stamp, frame_id='world', sensor_frame='lidar', sensor_origin=[0., 0., 0.], tracks=[track])
    projections, _ = project_tracks(packet, exposure, np.eye(4), INFO)
    detection = dict(bbox=[310., 230., 20., 20.], class_name='quadcopter', confidence=.9,
                     proposal_source='full_image')
    return packet, projections, detection


def test_bearing_reduces_transverse_error_without_independent_depth():
    packet, projections, detection = fixture()
    untouched = deepcopy(packet)
    result = fuse(projections, [detection], np.eye(4), INFO)[0]
    assert result['fusion_update']['accepted']
    assert result['position_source'] == 'lidar_camera_bearing_ekf'
    assert np.allclose(result['camera_position'], result['position'])
    assert np.allclose(result['lidar_camera_position'], result['lidar_position'])
    assert not result['fusion_update']['independent_depth']
    assert np.linalg.norm(np.array(result['position'])-[0, 0, 10]) < .2
    covariance = np.array(result['state_covariance'])
    assert np.linalg.eigvalsh(covariance).min() >= 0
    assert result['nearest_distance'] == 9.
    assert result['nearest_point'] == [0., 0., 9.]
    assert result['nearest_distance_stamp'] == 1.
    assert result['prediction']['stamp'] == 1.5
    assert packet == untouched
    assert fuse(projections, [detection], np.eye(4), INFO)[0] == result
    # At the optical axis, a bearing update contains no radial information.
    _, projections, detection = fixture((0, 0, 10))
    _, covariance, update = bearing_update(projections[0], detection, np.eye(4), INFO)
    assert update['accepted'] and np.isclose(covariance[2, 2], .16)


def test_bearing_rejects_outlier_edge_and_invalid_covariance():
    _, projections, detection = fixture()
    detection['bbox'] = [510., 350., 20., 20.]
    state, _, update = bearing_update(projections[0], detection, np.eye(4), INFO)
    assert not update['accepted'] and update['reason'] == 'camera_innovation_gate'
    assert np.allclose(state[:3], projections[0]['position'])
    detection['bbox'] = [0., 230., 20., 20.]
    assert bearing_update(projections[0], detection, np.eye(4), INFO)[2]['reason'] == 'clipped_or_invalid_camera_box'
    # A crop boundary can truncate the object far from the full image boundary.
    detection.update(bbox=[310., 230., 20., 20.], box_clipped=True)
    state, _, update = bearing_update(projections[0], detection, np.eye(4), INFO)
    assert update['reason'] == 'clipped_or_invalid_camera_box'
    assert np.allclose(state[:3], projections[0]['position'])
    projections[0]['track']['state_covariance'][0][0] = -1
    assert bearing_update(projections[0], detection, np.eye(4), INFO)[2]['reason'] == 'invalid_state_covariance'


def test_continuous_camera_alignment_rejects_future_prior_and_unknown_model():
    # 새 모델은 노출 이후 상태를 소급 관측으로 쓰지 않는다. 구형 기록은 별도로 재현한다.
    for exposure, model in ((.9, 'continuous_white_acceleration'), (1.1, 'unknown')):
        _, projections, detection = fixture(exposure=exposure)
        projections[0]['track']['prediction'].update(process_noise_model=model,
            acceleration_reference_dt_s=.1, acceleration_spectral_density_m2_s3=.4)
        _, covariance, update = bearing_update(projections[0], detection, np.eye(4), INFO)
        assert not update['accepted'] and covariance is None
        assert update['reason'] == ('noncausal_or_unknown_motion_model' if exposure < 1.
                                    else 'invalid_motion_uncertainty')


def test_camera_refuses_inconsistent_continuous_noise_metadata():
    # 같은 σ인데 q만 달라진 메시지는 영상 보정의 사전 공분산으로 사용하지 않는다.
    for changed in ({'acceleration_spectral_density_m2_s3': 4.},
                    {'acceleration_reference_dt_s': 1.}, {'acceleration_std': 8.}):
        _, projections, detection = fixture(exposure=1.05)
        prediction = projections[0]['track']['prediction']
        prediction.update(process_noise_model='continuous_white_acceleration',
                          acceleration_reference_dt_s=.1, acceleration_spectral_density_m2_s3=.4)
        assert bearing_update(projections[0], detection, np.eye(4), INFO)[1] is not None
        prediction.update(changed)
        result = fuse(projections, [detection], np.eye(4), INFO)[0]
        assert result['state_covariance'] is None
        assert result['fusion_update']['reason'] == 'invalid_motion_uncertainty'
    # 잘못된 JSON 자료형이나 표현 범위 초과도 다음 영상 처리를 막지 않는다.
    for prediction in (None, [], {'acceleration_std': 1e200}):
        _, projections, detection = fixture()
        projections[0]['track']['prediction'] = prediction
        result = fuse(projections, [detection], np.eye(4), INFO)[0]
        assert result['state_covariance'] is None
        assert result['fusion_update']['reason'] == 'invalid_motion_uncertainty'


def test_signed_alignment_disabled_baseline_and_rotation_equivariance():
    packet, projections, detection = fixture(exposure=1.1)
    result = fuse(projections, [detection], np.eye(4), INFO, position_fusion_enabled=False)[0]
    assert not result['fusion_update']['accepted']
    assert np.allclose(result['position'], projections[0]['position'])
    assert result['state_covariance'][0][0] > packet['tracks'][0]['state_covariance'][0][0]
    assert np.isclose(result['prediction']['stamp'], 1.6)
    # Re-express the same scene in a rotated/translated world; camera pixels are unchanged.
    packet, projections, detection = fixture()
    original = fuse(projections, [detection], np.eye(4), INFO)[0]
    rotation = Rotation.from_euler('zyx', [.5, -.3, .2]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, [4., -3., 2.]
    track = packet['tracks'][0]
    track['position'] = (rotation@np.array(track['position'])+transform[:3, 3]).tolist()
    # Equal isotropic position/velocity blocks remain isotropic after rotation.
    projections, _ = project_tracks(packet, 1., np.linalg.inv(transform), INFO)
    updated = fuse(projections, [detection], np.linalg.inv(transform), INFO)[0]
    assert np.allclose(updated['position'], rotation@np.array(original['position'])+transform[:3, 3])
    assert np.allclose(updated['camera_position'], original['camera_position'])
    assert np.allclose(updated['position_covariance'], rotation@np.array(original['position_covariance'])@rotation.T)


def test_independent_noisy_bearing_improves_seeded_measurements():
    rng = np.random.default_rng(73)
    before, after, accepted = [], [], 0
    for _ in range(200):
        measured = np.array([0., 0., 10.])+rng.normal(0., .4, 3)
        _, projections, detection = fixture(measured)
        detection['bbox'][:2] = (np.array([310., 230.])+rng.normal(0., 2., 2)).tolist()
        state, _, update = bearing_update(projections[0], detection, np.eye(4), INFO,
                                           BearingConfig(pixel_std=2., box_fraction=0., calibration_std=0.))
        accepted += update['accepted']
        before.append(np.linalg.norm(measured-[0, 0, 10])**2)
        after.append(np.linalg.norm(state[:3]-[0, 0, 10])**2)
    # A finite 99% innovation gate may reject Gaussian tail observations.
    assert accepted >= 190 and np.mean(after) < .5*np.mean(before)


def test_completed_worker_from_previous_clock_epoch_cannot_publish():
    from jol_perception.camera_fusion import CameraFusion, PipelineAudit
    from sensor_msgs.msg import Image
    def forbidden():
        raise AssertionError('Old-epoch inference must not publish after clock reset')
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        last_clock=10., epoch=0, history=[1], latest=object(), last_stamp=10., last_result_stamp=10.,
        future=(SimpleNamespace(done=lambda: True, result=forbidden), dict(epoch=0, message=Image())), network=None,
        pipeline=PipelineAudit(lambda: dict(monotonic_ns=0, ros_s=1.)),
        check_calibration=lambda: None)
    node.check_clock = lambda: CameraFusion.check_clock(node)
    CameraFusion.tick(node)
    assert node.epoch == 1 and node.future is None and node.latest is None and not node.history


def sparse_surface_fixture():
    """부분 LiDAR 표면 상자는 작지만 영상 측정은 상태 불확실성 안에 있다."""
    packet, _, _ = fixture((0., 0., 50.))
    packet['tracks'][0].update(size=[.5, .1, .1], state_covariance=np.eye(6).tolist())
    projections, _ = project_tracks(packet, 1., np.eye(4), INFO)
    detection = dict(bbox=[307., 237., 12., 6.], class_name='quadcopter', confidence=.9,
                     semantic_family='drone', subtype_verified=False)
    assert association_score(projections[0]['bbox'], detection['bbox']) < .12
    return projections, detection


def test_unique_bearing_fallback_recovers_sparse_surface_and_keeps_primary_output():
    projections, detection = sparse_surface_fixture()
    expected, _, update = bearing_update(projections[0], detection, np.eye(4), INFO)
    result = fuse(projections, [detection], np.eye(4), INFO)
    assert len(result) == 1 and update['accepted']
    assert result[0]['association_method'] == 'unique_gated_bearing_fallback'
    assert np.allclose(result[0]['position'], expected[:3])
    assert not result[0]['fusion_update']['independent_depth']
    assert result[0]['class_name'] == 'quadcopter' and not result[0]['subtype_verified']
    assert result[0]['nearest_distance'] == 9.

    # 추가 후보가 있어도 기존 bbox 연결과 보정 결과는 그대로 유지한다.
    _, primary, primary_detection = fixture()
    before = fuse(primary, [primary_detection], np.eye(4), INFO)
    assert before and 'association_method' not in before[0]
    far = deepcopy(projections[0]); far['track']['id'] = far['id'] = 9
    far['bbox'] = [10., 10., 1., 1.]
    assert fuse([*primary, far], [primary_detection], np.eye(4), INFO) == before


def test_bearing_fallback_rejects_ambiguity_coast_outlier_clip_and_unknown_family():
    projections, detection = sparse_surface_fixture()
    second = dict(detection, bbox=[306., 237., 12., 6.])
    assert not fuse(projections, [detection, second], np.eye(4), INFO)
    duplicate = deepcopy(projections[0]); duplicate['track']['id'] = duplicate['id'] = 9
    assert not fuse([*projections, duplicate], [detection], np.eye(4), INFO)
    for changed in (dict(detection, box_clipped=True), dict(detection, semantic_family='person'),
                    dict(detection, bbox=[500., 237., 12., 6.])):
        assert not fuse(projections, [changed], np.eye(4), INFO)
    unknown = dict(detection); unknown.pop('semantic_family')
    assert not fuse(projections, [unknown], np.eye(4), INFO)
    assert not fuse(projections, [detection], np.eye(4), INFO, position_fusion_enabled=False)
    coast = deepcopy(projections); coast[0]['track']['observed'] = False
    assert not fuse(coast, [detection], np.eye(4), INFO)
    invalid = deepcopy(projections); invalid[0]['track']['state_covariance'][0][0] = -1.
    assert not fuse(invalid, [detection], np.eye(4), INFO)


def test_bearing_fallback_preserves_existing_bbox_ambiguity_rejection():
    _, projections, detection = fixture()
    detection['semantic_family'] = 'drone'
    duplicate = deepcopy(projections[0]); duplicate['track']['id'] = duplicate['id'] = 9
    assert not fuse([*projections, duplicate], [detection], np.eye(4), INFO)
