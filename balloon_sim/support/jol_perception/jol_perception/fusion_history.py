"""늦게 도착한 영상 보정을 원래 노출 시각에 적용하고 이후 LiDAR 측정을 재계산한다.

같은 측정을 두 번 반영하지 않도록 영상 보정 결과를 다음 영상의 사전 상태로 재사용하지 않는다.
첫 구간의 가속도 불확실성은 tracking_core.predict_cv와 같은 모델을 쓴다."""
from copy import deepcopy

import numpy as np

from .tracking_core import predict_cv, forecast_model, CV_NOISE_MODEL
from .fusion_core import fuse, project_tracks


def reanchor_camera_seed(seed, history, matrix, info, detections, tolerance=.15,
                         bearing_config=None, position_fusion_enabled=True):
    """Re-evaluate the same image once from a newly arrived causal LiDAR prior.

    The existing exposure publication is immutable. This returns a new replay
    revision, using retained raw pixel detections and exact exposure calibration,
    never the previous camera posterior. Association and innovation gates run
    again because the newly available LiDAR prior can change both decisions.
    """
    causal = [packet for packet in history if packet['stamp'] <= seed['stamp']+1e-9]
    if not causal:
        return seed
    packet = max(causal, key=lambda packet: packet['stamp'])
    if packet['stamp'] <= seed['lidar_stamp']+1e-9:
        return seed
    if packet['frame_id'] != seed['frame_id']:
        raise ValueError('reanchor_frame_changed')
    births = {track['id']: track.get('birth_stamp') for track in seed['tracks']}
    if any(track['id'] in births and track.get('birth_stamp') != births[track['id']]
           for track in packet['tracks']):
        raise ValueError('reanchor_identity_reused')
    projections, reason = project_tracks(packet, seed['stamp'], matrix, info, tolerance)
    tracks = fuse(projections, detections, matrix, info, bearing_config=bearing_config,
                  position_fusion_enabled=position_fusion_enabled) if reason == 'ready' else []
    return dict(seed, tracks=tracks, lidar_stamp=packet['stamp'], fusion_reason=reason,
                fusion_revision=seed.get('fusion_revision', 0)+1,
                original_lidar_stamp=seed.get('original_lidar_stamp', seed['lidar_stamp']),
                source_exposure_stamp=seed['stamp'], publication_kind='lidar_replay',
                sensor_frame=packet.get('sensor_frame', packet.get('source_frame', packet['frame_id'])),
                sensor_origin=packet.get('sensor_origin'), stabilized=packet.get('stabilized', False),
                deskewed=packet.get('deskewed', False),
                time_difference_s=seed['stamp']-packet['stamp'],
                coordinate_updates=sum(t['fusion_update']['accepted'] for t in tracks))


def transition(dt):
    matrix = np.eye(6)
    matrix[:3, 3:] = np.eye(3) * dt
    acceleration = np.vstack((np.eye(3) * dt**2 / 2, np.eye(3) * dt))
    return matrix, acceleration


def covariance(value, size):
    result = np.asarray(value, dtype=float)
    if (result.shape != (size, size) or not np.isfinite(result).all()
            or not np.allclose(result, result.T, atol=1e-8, rtol=0)
            or np.linalg.eigvalsh(result).min() < -1e-9):
        raise ValueError('invalid_replay_covariance')
    return result


def replay_camera_track(seed, history):
    """Return a fresh posterior; reject missing transitions/ID reuse explicitly.

    history is a bounded sequence of LiDAR-only packets. Duplicate packet stamps
    are deduplicated, so repeated publication cannot repeat a measurement update.
    """
    result = deepcopy(seed)
    exposure, anchor_stamp = float(seed['stamp']), float(seed['lidar_stamp'])
    if not np.isfinite([exposure, anchor_stamp]).all() or not 0 < anchor_stamp <= exposure+1e-9:
        raise ValueError('noncausal_camera_prior')
    packets = {float(packet['stamp']): packet for packet in history}
    if anchor_stamp not in packets:
        raise ValueError('exposure_prior_evicted')

    def at(packet):
        matches = [track for track in packet['tracks'] if track['id'] == seed['id']]
        if len(matches) != 1:
            raise ValueError('replay_identity_missing_or_ambiguous')
        track = matches[0]
        if float(track['birth_stamp']) != float(seed['birth_stamp']):
            raise ValueError('replay_identity_reused')
        return track

    anchor = at(packets[anchor_stamp])
    _, noise_model = forecast_model(seed)
    if forecast_model(anchor)[1] != noise_model:
        raise ValueError('camera_replay_process_model_mismatch')
    state = np.r_[seed['position'], seed['velocity']].astype(float)
    uncertainty = covariance(seed['state_covariance'], 6)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError('invalid_replay_state')
    update = seed['fusion_update']
    if update.get('accepted') is not True:
        raise ValueError('camera_update_not_accepted')
    state_stamp = exposure
    last_measurement = float(anchor['last_seen'])
    if not np.isfinite(last_measurement) or not 0 < last_measurement <= anchor_stamp+1e-9:
        raise ValueError('invalid_anchor_measurement_stamp')
    previous = anchor_stamp
    measurements = []
    latest = anchor
    for stamp, packet in sorted(packets.items()):
        if stamp <= anchor_stamp:
            continue
        latest = at(packet)
        if forecast_model(latest)[1] != noise_model:
            raise ValueError('camera_replay_process_model_mismatch')
        if abs(float(latest['prediction_start_stamp'])-previous) > 1e-6:
            raise ValueError('replay_transition_gap')
        if stamp <= exposure:
            raise ValueError('exposure_prior_not_latest_causal_state')
        sigma = float(latest['process_acceleration_std'])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('invalid_replay_process_noise')
        if previous == anchor_stamp and exposure > anchor_stamp and noise_model == CV_NOISE_MODEL:
            # 연속 잡음의 독립 증분으로 노출→LiDAR 구간을 예측한다. 노출 사전값도 검사한다.
            _, expected = predict_cv(np.r_[anchor['position'], anchor['velocity']],
                                     covariance(anchor['state_covariance'], 6), exposure-anchor_stamp,
                                     sigma, noise_model)
            if not np.allclose(expected, update['prior_state_covariance'], atol=1e-8, rtol=1e-7):
                raise ValueError('camera_replay_process_model_mismatch')
            state, uncertainty = predict_cv(state, uncertainty, stamp-state_stamp, sigma, noise_model)
        elif previous == anchor_stamp and exposure > anchor_stamp:
            # Conditioning a split prediction independently undercounts process
            # noise. Share the original interval's acceleration latent instead.
            dt, offset = stamp-anchor_stamp, exposure-anchor_stamp
            full, full_accel = transition(dt)
            partial, partial_accel = transition(offset)
            base_state = np.r_[anchor['position'], anchor['velocity']].astype(float)
            base_covariance = covariance(anchor['state_covariance'], 6)
            predicted, predicted_covariance = predict_cv(base_state, base_covariance, dt, sigma, noise_model)
            exposure_covariance = partial @ base_covariance @ partial.T + partial_accel @ partial_accel.T * sigma**2
            if not np.allclose(exposure_covariance, update['prior_state_covariance'], atol=1e-8, rtol=1e-7):
                raise ValueError('camera_replay_process_model_mismatch')
            observation = np.asarray(update['observation_matrix'], dtype=float)
            noise = covariance(update['measurement_covariance'], 2)
            innovation = np.asarray(update['innovation_px'], dtype=float)
            if (observation.shape != (2, 6) or innovation.shape != (2,)
                    or not np.isfinite(observation).all() or not np.isfinite(innovation).all()):
                raise ValueError('invalid_camera_replay_observation')
            cross = full @ base_covariance @ partial.T + full_accel @ partial_accel.T * sigma**2
            residual = observation @ exposure_covariance @ observation.T + noise
            gain = np.linalg.solve(residual, observation @ cross.T).T
            state = predicted + gain @ innovation
            uncertainty = predicted_covariance - gain @ residual @ gain.T
        else:
            state, uncertainty = predict_cv(state, uncertainty, stamp-state_stamp, sigma, noise_model)
        if latest['observed']:
            measurement_stamp = float(latest['measurement_stamp'])
            measurement = np.asarray(latest['measurement_position'], dtype=float)
            noise = covariance(latest['measurement_covariance'], 3)
            if (abs(measurement_stamp-stamp) > 1e-6 or measurement_stamp <= last_measurement
                    or measurement.shape != (3,) or not np.isfinite(measurement).all()):
                raise ValueError('invalid_or_repeated_lidar_measurement')
            observation = np.eye(3, 6)
            residual = observation @ uncertainty @ observation.T + noise
            gain = np.linalg.solve(residual, observation @ uncertainty).T
            state += gain @ (measurement-observation @ state)
            correction = np.eye(6)-gain @ observation
            uncertainty = correction @ uncertainty @ correction.T + gain @ noise @ gain.T
            last_measurement = measurement_stamp
            measurements.append(measurement_stamp)
        elif any(latest.get(field) is not None for field in
                 ('measurement_stamp', 'measurement_position', 'measurement_covariance')):
            raise ValueError('coasted_state_claims_measurement')
        uncertainty = covariance((uncertainty+uncertainty.T)/2, 6)
        state_stamp, previous = stamp, stamp
    result['current_state'] = dict(position=state[:3].tolist(), velocity=state[3:].tolist(),
                                 state_covariance=uncertainty.tolist(),
                                 position_covariance=uncertainty[:3, :3].tolist(),
                                 velocity_covariance=uncertainty[3:, 3:].tolist())
    result.update(current_state_stamp=state_stamp,
                  source_exposure_stamp=exposure, last_measurement_stamp=last_measurement,
                  temporal_fusion=dict(valid=True, method='single_camera_fixed_lag_raw_lidar_replay',
                                       replayed_measurement_stamps=measurements,
                                       replayed_measurements=len(measurements),
                                       process_bridge=('continuous_markov_increments' if noise_model == CV_NOISE_MODEL
                                                       else 'shared_interval_acceleration')))
    horizon = float(latest['prediction']['horizon_s'])
    sigma, noise_model = forecast_model(latest)
    predicted, predicted_covariance = predict_cv(state, uncertainty, horizon, sigma, noise_model)
    result['current_state']['prediction'] = dict(latest['prediction'], stamp=state_stamp+horizon,
                                                position=predicted[:3].tolist(),
                                                position_covariance=predicted_covariance[:3, :3].tolist())
    return result
