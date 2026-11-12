"""월드 좌표 예측과 시뮬레이션 센서·표시봉의 방향 역기구학.

OpenMANIPULATOR-X 공식 형상·관절 제한(Apache-2.0):
https://raw.githubusercontent.com/ROBOTIS-GIT/open_manipulator/84b68b219a0fa30b223ad7e3c268d1f11fa43577/open_manipulator_x_description/urdf/open_manipulator_x.urdf.xacro
기존 SciPy의 제한 조건 최소제곱 풀이:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

먼 물체의 위치까지 팔을 뻗지 않고 표시봉의 방향만 맞춘다.
"""
import math
import numpy as np
from scipy.optimize import least_squares
from .tracking_core import predict_cv, forecast_model, LEGACY_CV_NOISE_MODEL

ARM_LOWER = np.array([-math.pi, -1.5, -1.5, -1.7])
ARM_UPPER = np.array([math.pi, 1.5, 1.4, 1.97])
ARM_POSTURE = (-.6, .6)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def pose(xyz=(0., 0., 0.), yaw=0., pitch=0.):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cy*cp, -sy, cy*sp], [sy*cp, cy, sy*sp], [-sp, 0., cp]]
    matrix[:3, 3] = xyz
    return matrix


def arm_fk(joints, base=None, rod_length=.30):
    """관절 측정 시각의 베이스 변환으로 봉의 원점·끝점·방향을 구한다."""
    q = np.asarray(joints, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or rod_length <= 0:
        raise ValueError('Arm FK needs four finite joints and positive rod length')
    matrix = np.eye(4) if base is None else np.asarray(base, dtype=float).copy()
    matrix = matrix @ pose((.012, 0., .017), yaw=q[0])
    matrix = matrix @ pose((0., 0., .0595), pitch=q[1])
    matrix = matrix @ pose((.024, 0., .128), pitch=q[2])
    matrix = matrix @ pose((.124, 0., 0.), pitch=q[3])
    matrix = matrix @ pose((.126, 0., 0.))
    origin, direction = matrix[:3, 3], matrix[:3, 0]
    return origin, origin + rod_length*direction, direction


def turret_geometry(camera_mount='turret', lidar_mount_mode='payload'):
    """팔 카메라 모드에서는 수평 LiDAR와 독립적인 포탑 기계축을 조준한다."""
    if camera_mount not in ('turret', 'arm') or lidar_mount_mode not in ('payload', 'level'):
        raise ValueError('Invalid camera_mount or lidar_mount_mode')
    if camera_mount == 'turret':
        return dict(frame='jol/camera_optical', axis=2, offset=(.14, -.145, 1.0125))
    return dict(frame='jol/turret_pitch', axis=0, offset=(0., 0., 0.))


def turret_fk(joints, base=None, camera_mount='turret', lidar_mount_mode='payload'):
    """장착 방식에 맞는 실제 센서 또는 포탑 기계축의 원점·방향을 구한다."""
    q = np.asarray(joints, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('Turret FK needs finite yaw and pitch')
    geometry = turret_geometry(camera_mount, lidar_mount_mode)
    matrix = np.eye(4) if base is None else np.asarray(base, dtype=float).copy()
    matrix = matrix @ pose((0., 0., .7), yaw=q[0]) @ pose((0., 0., .3), pitch=q[1])
    matrix = matrix @ pose(geometry['offset'])
    # 광학 +Z와 센서 링크 +X는 같은 앞방향이다.
    return matrix[:3, 3], matrix[:3, 0]


def ray_error(origin, direction, target):
    origin, direction, target = map(lambda a: np.asarray(a, dtype=float), (origin, direction, target))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (origin, direction, target)):
        raise ValueError('Ray geometry requires finite XYZ vectors')
    norm = np.linalg.norm(direction)
    delta = target-origin
    if norm < 1e-10 or np.linalg.norm(delta) < 1e-6:
        raise ValueError('Singular ray or coincident target')
    direction = direction/norm
    depth = float(delta @ direction)
    perpendicular = float(np.linalg.norm(np.cross(delta, direction)))
    return dict(angle_error_deg=math.degrees(math.atan2(perpendicular, depth)),
                ray_depth_m=depth, cross_track_error_m=perpendicular)


def _target(target, base):
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Target must be finite world XYZ')
    local = np.linalg.inv(base) @ np.r_[target, 1.]
    if np.linalg.norm(local[:2]) < 1e-5 or np.linalg.norm(local[:3]) < .5:
        raise ValueError('Target at pan singularity or inside near-field exclusion')
    return target, local[:3]


def solve_turret(target, base=None, camera_mount='turret', lidar_mount_mode='payload'):
    base = np.eye(4) if base is None else np.asarray(base, dtype=float)
    target, local = _target(target, base)
    initial = [math.atan2(local[1], local[0]),
               np.clip(-math.atan2(local[2]-1., np.linalg.norm(local[:2])), -math.pi/6, math.pi/6)]

    def residual(q):
        origin, direction = turret_fk(q, base, camera_mount, lidar_mount_mode)
        delta = target-origin
        return direction-delta/np.linalg.norm(delta)

    fit = least_squares(residual, initial, bounds=([-np.inf, -math.pi/6], [np.inf, math.pi/6]),
                        max_nfev=25, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    joints = np.array([float(wrap(fit.x[0])), fit.x[1]])
    origin, direction = turret_fk(joints, base, camera_mount, lidar_mount_mode)
    error = ray_error(origin, direction, target)
    success = error['angle_error_deg'] < .1 and error['ray_depth_m'] > 0
    return dict(success=success, reason='ready' if success else 'turret_joint_limit_or_ik',
                joints=joints.tolist(), origin=origin.tolist(), direction=direction.tolist(), **error)


def solve_arm(target, base=None, rod_length=.30):
    """움직이는 베이스를 반영해 방향을 맞춘다. 베이스 생략 시 팔 좌표다."""
    base = np.eye(4) if base is None else np.asarray(base, dtype=float)
    target, local = _target(target, base)
    yaw = math.atan2(local[1], local[0]-.012)
    pitch = -math.atan2(local[2]-.2, np.linalg.norm(local[:2]))
    initial = np.clip([yaw, pitch], ARM_LOWER[[0, 3]]+1e-9, ARM_UPPER[[0, 3]]-1e-9)

    def joints(q):
        # ponytail: 방향 조준은 2축만 쓰고 어깨·팔꿈치는 고정한다. 접촉 작업은 전체 자세 IK가 필요하다.
        return np.array([q[0], *ARM_POSTURE, q[1]])

    def residual(q):
        origin, _, direction = arm_fk(joints(q), base, rod_length)
        delta = target-origin
        return direction-delta/np.linalg.norm(delta)

    fit = least_squares(residual, initial, bounds=(ARM_LOWER[[0, 3]], ARM_UPPER[[0, 3]]),
                        max_nfev=25, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    result = joints(fit.x)
    origin, tip, direction = arm_fk(result, base, rod_length)
    error = ray_error(origin, direction, target)
    success = error['angle_error_deg'] < .1 and error['ray_depth_m'] > rod_length
    return dict(success=success, reason='ready' if success else 'arm_joint_limit_or_ik',
                joints=result.tolist(), origin=origin.tolist(), tip=tip.tolist(),
                direction=direction.tolist(), **error)


def limit_step(current, desired, rate, dt, lower, upper, continuous_yaw=False):
    current, desired, rate = np.asarray(current, dtype=float), np.asarray(desired, dtype=float), np.asarray(rate, dtype=float)
    if (current.shape != desired.shape or not np.isfinite(current).all() or not np.isfinite(desired).all()
            or not np.isfinite(rate).all() or (rate <= 0).any() or not 0 <= dt <= .25):
        raise ValueError('Invalid joint slew inputs')
    error = desired-current
    if continuous_yaw:
        error[0] = wrap(error[0])
    command = current + np.clip(error, -rate*dt, rate*dt)
    if continuous_yaw:
        command[0] = wrap(command[0])
    return np.clip(command, lower, upper)


def feedback_step_interval(dt, now, joint_stamp):
    """현재 제어 주기 안의 측정 지연만 보상해 오래된 상태의 이동폭 증가를 막는다."""
    if not np.isfinite([dt, now, joint_stamp]).all() or dt < 0 or joint_stamp > now:
        raise ValueError('Invalid control interval or future joint feedback')
    age = now-joint_stamp
    return dt+age if age < dt-1e-9 else dt


def joint_slew_lower_bound(current, desired, rate, continuous_yaw=False):
    """표적·베이스가 정지했을 때 속도 제한만으로 구한 최소시간이며 도달 ETA는 아니다."""
    current, desired = np.asarray(current, dtype=float), np.asarray(desired, dtype=float)
    rate = np.broadcast_to(np.asarray(rate, dtype=float), current.shape)
    if (current.ndim != 1 or not current.size or current.shape != desired.shape
            or not np.isfinite(current).all() or not np.isfinite(desired).all()
            or not np.isfinite(rate).all() or (rate <= 0).any()):
        raise ValueError('Joint slew time needs finite joint vectors and positive rates')
    error = desired-current
    if continuous_yaw:
        error[0] = wrap(error[0])
    return float(np.max(np.abs(error)/rate))


def patrol_step(current, dt, elevation_deg=20., period=8., yaw_limit=1.2, pitch_limit=.8):
    """위쪽을 보며 yaw를 연속 회전한다. 실제 관절값을 쓰고 시간 정지 시 유지한다."""
    if (not np.isfinite([elevation_deg, period, yaw_limit, pitch_limit]).all()
            or not 0 <= elevation_deg <= 30 or period <= 0
            or not 0 < yaw_limit <= 1.2 or not 0 < pitch_limit <= .8
            or 2*math.pi/period > yaw_limit):
        raise ValueError('Patrol must fit turret elevation and joint-rate limits')
    current = np.asarray(current, dtype=float)
    if current.shape != (2,):
        raise ValueError('Patrol requires measured yaw and pitch')
    desired = [current[0]+2*math.pi/period*dt, -math.radians(elevation_deg)]
    return limit_step(current, desired, [yaw_limit, pitch_limit], dt,
                      [-math.pi, -math.pi/6], [math.pi, math.pi/6], continuous_yaw=True)


def forecast(track, stamp, now, horizon):
    dt = now + horizon-stamp
    if dt < 0 or not np.isfinite([dt, horizon]).all() or not 0 <= horizon <= 3.:
        raise ValueError('Prediction timestamp must be finite and not precede the filter state')
    state = np.r_[track['position'], track['velocity']]
    covariance = np.asarray(track['state_covariance'], dtype=float)
    if covariance.shape != (6, 6) or not np.allclose(covariance, covariance.T, atol=1e-7):
        raise ValueError('A symmetric world-state covariance is required')
    if np.linalg.eigvalsh(covariance).min() < -1e-7:
        raise ValueError('State covariance is not positive semidefinite')
    acceleration_std, noise_model = forecast_model(track)
    predicted, uncertainty = predict_cv(state, covariance, dt, acceleration_std, noise_model)
    return dict(model='constant_velocity', horizon_s=horizon, stamp=now+horizon,
                process_noise_model=noise_model,
                **{key: track['prediction'][key] for key in ('acceleration_reference_dt_s',
                    'acceleration_spectral_density_m2_s3') if key in track['prediction']},
                acceleration_std=acceleration_std, noise_source='tracker.prediction.acceleration_std',
                state_stamp=stamp, propagation_s=dt, position=predicted[:3].tolist(),
                position_covariance=uncertainty[:3, :3].tolist())


def choose_turret_state(track, lidar_stamp, packet, now, max_result_age=.35, max_lidar_age=.5,
                        enabled=True, selection_horizon=0., max_exposure_age=.8):
    """조준 시각의 불확실성이 줄 때만 신선한 카메라 융합 상태를 선택한다.

    획득·ID는 LiDAR가 맡으며 같은 카메라 상태를 필터에 중복 반영하지 않는다.
    상태 시각과 영상·LiDAR 실측 시각을 따로 검사한다.
    """
    lidar_uncertainty = float(np.trace(forecast(track, lidar_stamp, now, selection_horizon)['position_covariance']))
    detail = dict(control_state_source='lidar', fusion_selection_reason='waiting_for_fusion',
                  lidar_stamp=lidar_stamp, camera_stamp=None, fusion_lidar_stamp=None,
                  fusion_age_s=None, uncertainty_comparison_stamp=now+selection_horizon,
                  fusion_state_stamp=None, fusion_state_age_s=None,
                  fusion_last_measurement_stamp=None, fusion_last_measurement_age_s=None,
                  lidar_position_variance_m2=lidar_uncertainty,
                  fusion_position_variance_m2=None)

    def fallback(reason):
        detail['fusion_selection_reason'] = reason
        return track, lidar_stamp, detail

    if not enabled:
        return fallback('coordinate_fusion_disabled')
    if packet is None:
        return fallback('waiting_for_fusion')
    detail['fusion_revision'] = packet.get('fusion_revision', 0)
    # 거부한 후보의 시각도 남겨 데이터 지연 원인을 추적한다.
    for field, source in [('camera_stamp', 'stamp'), ('fusion_state_stamp', 'current_state_stamp'),
                          ('fusion_lidar_stamp', 'lidar_stamp'),
                          ('fusion_last_measurement_stamp', 'last_measurement_stamp')]:
        try:
            value = float(packet[source])
            if np.isfinite(value):
                detail[field] = value
        except (KeyError, TypeError, ValueError):
            pass
    if detail['camera_stamp'] is not None:
        detail['fusion_age_s'] = now-detail['camera_stamp']
    if detail['fusion_state_stamp'] is not None:
        detail['fusion_state_age_s'] = now-detail['fusion_state_stamp']
    if detail['fusion_last_measurement_stamp'] is not None:
        detail['fusion_last_measurement_age_s'] = now-detail['fusion_last_measurement_stamp']
    if (packet.get('frame_id') != 'world' or packet.get('stabilized') is not True
            or packet.get('calibration_valid') is not True or packet.get('fusion_reason') != 'ready'):
        return fallback('fusion_geometry_not_ready')
    try:
        stamp = float(packet['stamp'])
        if not np.isfinite(stamp) or stamp <= 0 or not 0 <= now-stamp <= max_exposure_age:
            return fallback('stale_or_future_fusion')
        detail.update(camera_stamp=stamp, fusion_age_s=now-stamp)
        candidates = [item for item in packet['tracks'] if int(item['id']) == int(track['id'])]
        if len(candidates) != 1:
            return fallback('selected_lidar_id_not_unique_in_fusion')
        posterior = candidates[0]
        delayed = 'temporal_fusion' in posterior
        if delayed and (not isinstance(posterior['temporal_fusion'], dict)
                        or posterior['temporal_fusion'].get('valid') is not True):
            return fallback('invalid_delayed_fusion')
        if (posterior.get('source') != 'camera_lidar'
                or posterior.get('position_source') != 'lidar_camera_bearing_ekf'
                or not isinstance(posterior.get('fusion_update'), dict)
                or posterior['fusion_update'].get('accepted') is not True):
            return fallback('camera_update_not_accepted')
        source_stamp = float(posterior['lidar_stamp'])
        exposure_stamp = float(posterior['stamp'])
        state_stamp = float(posterior['current_state_stamp']) if delayed else exposure_stamp
        last_measurement = float(posterior['last_measurement_stamp']) if delayed else source_stamp
        detail.update(fusion_state_stamp=state_stamp, fusion_state_age_s=now-state_stamp,
                      fusion_last_measurement_stamp=last_measurement,
                      fusion_last_measurement_age_s=now-last_measurement)
        if not 0 <= now-state_stamp <= max_result_age:
            return fallback('stale_or_future_fusion')
        if (not np.isfinite([state_stamp, source_stamp]).all() or source_stamp <= 0
                or abs(exposure_stamp-stamp) > 1e-6 or source_stamp > lidar_stamp+1e-6
                or state_stamp < exposure_stamp-1e-6 or last_measurement > max(state_stamp, source_stamp)+1e-6
                or last_measurement > lidar_stamp+1e-6 or not 0 <= now-last_measurement <= max_lidar_age
                or (delayed and state_stamp > max(lidar_stamp, exposure_stamp)+1e-6)
                or (delayed and posterior.get('birth_stamp') != track.get('birth_stamp'))):
            return fallback('invalid_fusion_source_stamp')
        detail['fusion_lidar_stamp'] = source_stamp
        if delayed and last_measurement < float(track['last_seen'])-1e-6:
            return fallback('fusion_missing_latest_lidar_measurement')
        selected = posterior['current_state'] if delayed else posterior
        # 구형·신형 잡음 모델의 공분산을 섞어 더 정밀한 상태라고 선택하지 않는다.
        try:
            _, selected_model = forecast_model(selected)
        except (KeyError, TypeError, ValueError, OverflowError):
            return fallback('invalid_fusion_motion_metadata')
        if selected_model != track['prediction'].get('process_noise_model', LEGACY_CV_NOISE_MODEL):
            return fallback('fusion_motion_model_mismatch')
        state = np.r_[selected['position'], selected['velocity']].astype(float)
        covariance = np.asarray(selected['state_covariance'], dtype=float)
        if (state.shape != (6,) or not np.isfinite(state).all() or covariance.shape != (6, 6)
                or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, atol=1e-8, rtol=0)
                or np.linalg.eigvalsh(covariance).min() < -1e-9):
            return fallback('invalid_fusion_state_or_covariance')
        for field, expected in [('position_covariance', covariance[:3, :3]),
                                ('velocity_covariance', covariance[3:, 3:])]:
            supplied = np.asarray(selected[field], dtype=float)
            if supplied.shape != (3, 3) or not np.allclose(supplied, expected, atol=1e-8, rtol=0):
                return fallback('inconsistent_fusion_covariance')
        fused_track = dict(track, position=state[:3].tolist(), velocity=state[3:].tolist(),
                           state_covariance=covariance.tolist(),
                           position_covariance=covariance[:3, :3].tolist(),
                           velocity_covariance=covariance[3:, 3:].tolist())
        if 'prediction' in selected:
            fused_track['prediction'] = selected['prediction']
        uncertainty = float(np.trace(forecast(fused_track, state_stamp, now, selection_horizon)['position_covariance']))
        if not np.isfinite(uncertainty):
            return fallback('invalid_fusion_state_or_covariance')
        detail['fusion_position_variance_m2'] = uncertainty
        if uncertainty >= lidar_uncertainty-1e-12:
            return fallback('lidar_has_lower_predicted_uncertainty')
        detail.update(control_state_source='camera_lidar', fusion_selection_reason='lower_predicted_uncertainty')
        return fused_track, state_stamp, detail
    except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return fallback('invalid_fusion_state_or_covariance')


class TargetSelector:
    """신선한 LiDAR 후보를 현재 시각으로 맞춘 뒤 가까운 대상을 선택한다.

    거리 동률이면 기존 ID를 유지한다. 최초 획득은 실측이 필요하고,
    예측만 있는 후보는 마지막 실측이 신선할 때만 유지한다.
    """
    def __init__(self, requested_id=0, min_speed=0.):
        self.requested_id, self.min_speed = requested_id, min_speed
        self.selected_id, self.last_seen = None, -math.inf
        self.detail = {}

    def request(self, identity):
        if identity < 0:
            raise ValueError('Target ID must be nonnegative; zero selects automatically')
        self.requested_id, self.selected_id, self.last_seen = identity, None, -math.inf
        self.detail = {}

    def choose(self, packet, now, max_age=.5, max_coast=1., origin=(0., 0., 0.)):
        previous = self.selected_id
        detail = self.detail = dict(
            policy='manual_id' if self.requested_id else 'nearest_recent_3d',
            reference_frame='world', origin=None, source_stamp=None, evaluated_stamp=now,
            distance_state_stamp=now, position_source='kalman_current_time',
            previous_target_id=previous, selected_id=None, distance_m=None,
            candidate_distances_m=[], rejected_candidates=[],
            observed_max_age_s=max_age, prediction_max_age_s=max_coast,
            tie_break='current_id_then_lowest_id')

        def finish(track, state, reason=None):
            detail.update(reason=reason or state, selected_id=self.selected_id)
            if track is not None:
                detail['distance_m'] = distances[self.selected_id]
            return track, state

        if packet is None:
            return finish(None, 'waiting_for_tracks')
        if not isinstance(packet, dict) or not isinstance(packet.get('tracks'), list):
            return finish(None, 'invalid_tracks_packet')
        if packet.get('frame_id') != 'world':
            return finish(None, 'requires_world_tracks')
        try:
            stamp = float(packet['stamp'])
        except (KeyError, TypeError, ValueError):
            return finish(None, 'invalid_tracks_packet')
        if not np.isfinite(stamp) or not 0 <= now-stamp <= max_age:
            return finish(None, 'stale_or_future_tracks')
        detail['source_stamp'] = stamp
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Selection origin must be a finite world XYZ vector')
        detail['origin'] = origin.tolist()
        tracks, distances, current_positions, seen_ids = {}, {}, {}, set()
        for track in packet['tracks']:
            identity, reason = None, None
            try:
                if not isinstance(track, dict):
                    raise ValueError('Track must be an object')
                identity = int(track['id'])
                if isinstance(track['id'], bool) or identity <= 0 or identity != track['id']:
                    raise ValueError('Track ID must be a positive integer')
                if identity in seen_ids:
                    tracks.pop(identity, None)
                    distances.pop(identity, None)
                    reason = 'duplicate_id'
                elif (track.get('confirmed') is not True or not isinstance(track.get('hits'), int)
                      or track['hits'] < 3):
                    reason = 'unconfirmed'
                else:
                    position = np.asarray(track['position'], dtype=float)
                    velocity = np.asarray(track['velocity'], dtype=float)
                    seen = float(track['last_seen'])
                    if (position.shape != (3,) or velocity.shape != (3,)
                            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
                            or not np.isfinite(seen) or track.get('observed') not in (True, False)
                            or not isinstance(track.get('observed'), bool)):
                        raise ValueError('Malformed track state')
                    # 새 패킷을 기다리는 제어 주기에도 같은 CV 모델로 현재 거리를 비교한다.
                    current_position = forecast(track, stamp, now, 0.)['position']
                    covariance = np.asarray(track['position_covariance'], dtype=float)
                    if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                            or not np.allclose(covariance, np.asarray(track['state_covariance'])[:3, :3])):
                        raise ValueError('Malformed position covariance')
                    observed = track['observed']
                    if seen > stamp+1e-6 or seen > now:
                        reason = 'future_measurement'
                    elif now-seen > (max_age if observed else max_coast):
                        reason = 'expired_measurement'
                    elif observed and np.linalg.norm(velocity) < self.min_speed and not self.requested_id:
                        reason = 'below_acquisition_speed'
                    else:
                        distance = float(np.linalg.norm(np.asarray(current_position)-origin))
                        if not np.isfinite(distance):
                            raise ValueError('Invalid distance')
                        tracks[identity], distances[identity] = track, distance
                        current_positions[identity] = current_position
            except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
                reason = 'invalid_track'
            if identity is not None:
                seen_ids.add(identity)
            if reason is not None:
                detail['rejected_candidates'].append(dict(id=identity, reason=reason))
        candidates = [track for track in tracks.values() if track['observed'] or previous is not None]
        detail['candidate_distances_m'] = [dict(id=int(track['id']),
            distance_m=distances[int(track['id'])], observed=track['observed'],
            position=current_positions[int(track['id'])], position_stamp=now,
            source_position=track['position'], last_seen=float(track['last_seen']))
            for track in sorted(candidates, key=lambda item: item['id'])]
        if self.requested_id:
            track = tracks.get(self.requested_id)
            if track is None:
                expired = any(item['id'] == self.requested_id and item['reason'] == 'expired_measurement'
                              for item in detail['rejected_candidates'])
                return finish(None, 'requested_target_expired' if expired else 'requested_target_unavailable')
            # 수동 ID도 실측 없이 처음 획득하지 않는다.
            if not track['observed'] and self.selected_id != self.requested_id:
                return finish(None, 'requested_target_unobserved')
            reason = 'manual_id'
        elif candidates:
            track = min(candidates, key=lambda item: (distances[int(item['id'])],
                        int(item['id']) != previous, int(item['id'])))
            reason = ('nearest_acquired' if previous is None else
                      'nearest_retained' if int(track['id']) == previous else 'nearest_handover')
        elif previous is not None and 0 <= now-self.last_seen <= max_coast:
            return finish(None, 'selected_target_missing')
        else:
            self.selected_id = None
            return finish(None, 'waiting_for_confirmed_target')
        self.selected_id, self.last_seen = int(track['id']), float(track['last_seen'])
        return finish(track, 'tracking' if track['observed'] else 'coasting', reason)
