"""ROS 없이 실행하는 영상 전처리·신경망 검출·투영·방향 EKF 보정 함수들이다.

단안 카메라는 방향만 추가하며 독립 거리 센서처럼 취급하지 않는다."""
import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracking_core import predict_cv, forecast_model, CV_NOISE_MODEL


@dataclass(frozen=True)
class BearingConfig:
    """Pixel uncertainty includes box-center/shape and extrinsic residuals.

    These are measurement-model assumptions, not calibrated accuracy claims.
    The camera supplies two angular constraints, never an independent depth.
    """
    pixel_std: float = 2.0
    box_fraction: float = .10
    calibration_std: float = 1.0
    innovation_gate: float = 9.21  # chi-square 99% for two bearing coordinates

    def __post_init__(self):
        values = [self.pixel_std, self.box_fraction, self.calibration_std, self.innovation_gate]
        if (not np.isfinite(values).all() or self.pixel_std <= 0 or self.box_fraction < 0
                or self.calibration_std < 0 or self.innovation_gate <= 0):
            raise ValueError('Invalid bearing measurement uncertainty/gate')


def calibration_error(info, width, height, frame):
    """Reject uncalibrated, incompatible, or unsupported raw-camera geometry."""
    k, d = np.asarray(info['k'], dtype=float), np.asarray(info['d'], dtype=float)
    if (info['frame_id'] != frame or info['width'] != width or info['height'] != height):
        return 'camera_info_mismatch'
    if (k.shape != (9,) or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0
            or not np.allclose(k[[3, 6, 7, 8]], [0, 0, 0, 1]) or abs(k[1]) > 1e-8
            or not 0 <= k[2] < width or not 0 <= k[5] < height):
        return 'invalid_camera_intrinsics'
    if (d.ndim != 1 or len(d) not in (0, 4, 5, 8, 12, 14) or not np.isfinite(d).all()
            or info.get('distortion_model', '') not in ('', 'plumb_bob', 'rational_polynomial')):
        return 'unsupported_camera_distortion'
    if info.get('binning_x', 0) > 1 or info.get('binning_y', 0) > 1 or info.get('roi_offset', [0, 0]) != [0, 0]:
        return 'unsupported_camera_roi'
    return ''


def image_bgr(message):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(message.encoding)
    if (channels is None or not 0 < message.width <= 8192 or not 0 < message.height <= 8192
            or message.step < channels * message.width or len(message.data) < message.step * message.height):
        raise ValueError('Unsupported image dimensions, encoding, or row stride')
    image = np.ndarray((message.height, message.width, channels), np.uint8,
                       buffer=message.data, strides=(message.step, channels, 1))
    conversion = {'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR,
                  'bgra8': cv2.COLOR_BGRA2BGR, 'mono8': cv2.COLOR_GRAY2BGR}.get(message.encoding)
    return cv2.cvtColor(image, conversion) if conversion is not None else image.copy()


class DroneNetwork:
    """YOLOv8 ONNX inference. Weights and provenance are independently checked."""
    def __init__(self, path, confidence=0.35, threads=2, provider='CPUExecutionProvider'):
        import onnxruntime as ort
        path = Path(path)
        if not 0 < confidence < 1 or not 1 <= threads <= 16:
            raise ValueError('Invalid network threshold/thread count')
        source = json.loads(path.with_name('drone_yolo_source.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['onnx_sha256']:
            raise ValueError('ONNX SHA-256 does not match model provenance')
        self.model = source['repository'] + '@' + source['revision']
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        if provider not in ('CPUExecutionProvider', 'CUDAExecutionProvider'):
            raise ValueError('Expected CPUExecutionProvider or CUDAExecutionProvider')
        if provider not in ort.get_available_providers():
            raise ValueError('Requested ONNX provider is unavailable: '+provider)
        if provider == 'CUDAExecutionProvider' and hasattr(ort, 'preload_dlls'):
            # ORT >=1.21 can load CUDA/cuDNN supplied by the isolated GPU venv.
            ort.preload_dlls(directory='')
        providers = ([provider] if provider == 'CPUExecutionProvider' else
                     [(provider, {'use_tf32': 0}), 'CPUExecutionProvider'])
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        self.provider_options = self.session.get_provider_options()
        if self.providers[0] != provider:
            raise ValueError('Requested ONNX provider did not initialize: '+provider)
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get('names', '{}'))
        if (not isinstance(self.names, dict) or set(self.names) != set(range(len(self.names)))
                or not self.names or any(n not in ('quadcopter', 'fixed-wing', 'drone') for n in self.names.values())):
            raise ValueError('Expected explicit drone class names')
        inp = self.session.get_inputs()[0]
        if inp.type != 'tensor(float)' or inp.shape[:2] != [1, 3] or len(inp.shape) != 4 or inp.shape[2:] != [640, 640]:
            raise ValueError('Expected float32 RGB [1,3,640,640] model')
        self.input_name, self.confidence = inp.name, confidence

    def infer(self, image, regions=()):
        height, width = image.shape[:2]
        results = []
        # ponytail: at most four projected crops; batch inference if CPU latency requires it.
        for proposal in [(0, 0, width, height), *regions[:4]]:
            region = proposal['region'] if isinstance(proposal, dict) else proposal
            source = (proposal['proposal_source'] if isinstance(proposal, dict) else
                      'full_image' if tuple(region) == (0, 0, width, height) else 'calibrated_lidar_roi')
            x, y, w, h = map(int, region)
            if min(w, h) < 1 or min(x, y) < 0 or x + w > width or y + h > height:
                raise ValueError('Inference region outside image')
            scale = min(640 / w, 640 / h)
            nw, nh = round(w * scale), round(h * scale)
            left, top = (640 - nw) // 2, (640 - nh) // 2
            canvas = np.full((640, 640, 3), 114, np.uint8)
            canvas[top:top + nh, left:left + nw] = cv2.resize(image[y:y + h, x:x + w], (nw, nh))
            tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
            raw = self.session.run(None, {self.input_name: tensor})[0]
            if raw.shape != (1, 4 + len(self.names), 8400):
                raise ValueError('Unexpected YOLO output shape')
            rows = raw[0].T
            class_ids = rows[:, 4:].argmax(axis=1)
            scores = rows[np.arange(len(rows)), class_ids + 4]
            for row, class_id, score in zip(rows[scores >= self.confidence], class_ids[scores >= self.confidence], scores[scores >= self.confidence]):
                if not np.isfinite(row).all() or min(row[2:4]) <= 0:
                    continue
                x0 = (row[0] - row[2] / 2 - left) / scale + x
                y0 = (row[1] - row[3] / 2 - top) / scale + y
                x1, y1 = x0 + row[2] / scale, y0 + row[3] / scale
                clipped = x0 < x or y0 < y or x1 > x+w or y1 > y+h
                # A crop never observes pixels outside its own input. Preserve
                # truncation so an asymmetric box cannot supply a center bearing.
                x0, y0, x1, y1 = np.clip([x0, y0, x1, y1], [x, y, x, y], [x+w, y+h, x+w, y+h])
                if x1 > x0 and y1 > y0:
                    results.append({'bbox': [float(x0), float(y0), float(x1-x0), float(y1-y0)],
                                    'class_name': self.names[int(class_id)], 'class_id': int(class_id),
                                    'confidence': float(score), 'inference_region': list(region),
                                    'proposal_source': source, 'box_clipped': bool(clipped),
                                    'class_scores': {name: float(row[4+index]) for index, name in self.names.items()}})
        if not results:
            return []
        # Both output classes are members of the same drone family: suppress duplicate subtypes.
        keep = cv2.dnn.NMSBoxes([r['bbox'] for r in results], [r['confidence'] for r in results], self.confidence, 0.45)
        kept = []
        for i in np.asarray(keep).reshape(-1)[:32]:
            winner = dict(results[int(i)])
            alternatives = {}
            for candidate in results:
                # association_score above .45 necessarily means IoU above .45;
                # retain conflicting neural evidence that family NMS suppressed.
                if association_score(winner['bbox'], candidate['bbox']) > .45:
                    for name, confidence in candidate['class_scores'].items():
                        if confidence >= self.confidence and (name not in alternatives or
                                                             confidence > alternatives[name]['confidence']):
                            alternatives[name] = dict(class_name=name, confidence=confidence,
                                                      proposal_source=candidate['proposal_source'])
            winner.update(semantic_family='drone', subtype_ambiguous=len(alternatives) > 1,
                          subtype_verified=False,
                          subtype_status='ambiguous' if len(alternatives) > 1 else 'unvalidated',
                          subtype_candidates=list(alternatives.values()))
            kept.append(winner)
        return kept


def transform_matrix(translation, quaternion):
    q = np.asarray(quaternion, dtype=float)
    t = np.asarray(translation, dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all() or abs(np.linalg.norm(q) - 1.) > .01:
        raise ValueError('Invalid TF translation/quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = t
    return result


def project_tracks(packet, image_stamp, matrix, info, tolerance=.15, *, search_max_age=0.):
    """Project boxes from the track frame using camera-from-track TF at exposure.

    Positions, velocities and AABB axes must share the packet frame. In the
    rotating-rig setup that frame is world, so sensor rotation never enters the
    velocity extrapolation. The supplied matrix already includes moving pivots,
    fixed lever arms and the camera optical-frame convention.
    """
    dt = image_stamp - packet['stamp']
    if not np.isfinite(dt) or abs(dt) > tolerance:
        return [], 'stale_lidar'
    k, d = np.array(info['k']).reshape(3, 3), np.array(info['d'])
    projections = []
    for track in packet['tracks']:
        search_only = not track.get('observed', False)
        if search_only:
            age = image_stamp-float(track.get('last_seen', float('-inf')))
            if not (search_max_age > 0 and track.get('confirmed', False)
                    and np.isfinite(age) and 0 <= age <= search_max_age):
                continue
        position, velocity, size = (np.asarray(track[name], dtype=float) for name in ('position', 'velocity', 'size'))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (position, velocity, size)) or (size < 0).any():
            continue
        position = position + dt * velocity
        corners = position + np.array(list(itertools.product((-0.5, .5), repeat=3))) * np.maximum(size, .10)
        camera = corners @ matrix[:3, :3].T + matrix[:3, 3]
        center = matrix[:3, :3] @ position + matrix[:3, 3]
        if (camera[:, 2] <= .05).any() or center[2] <= .05:
            continue
        pixels, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, d if len(d) else None)
        pixels = pixels.reshape(-1, 2)
        if not np.isfinite(pixels).all():
            continue
        lo, hi = pixels.min(axis=0), pixels.max(axis=0)
        if hi[0] < 0 or hi[1] < 0 or lo[0] >= info['width'] or lo[1] >= info['height']:
            continue
        lo, hi = np.maximum(lo, 0), np.minimum(hi, [info['width'], info['height']])
        projections.append({'id': track['id'], 'bbox': [*lo.tolist(), *(hi-lo).tolist()],
                            'search_only': search_only,
                            'camera_position': center.tolist(), 'position': position.tolist(), 'track': track,
                            'frame_id': packet['frame_id'], 'image_stamp': image_stamp,
                            'lidar_stamp': packet['stamp'],
                            'sensor_frame': packet.get('sensor_frame', packet.get('source_frame', packet['frame_id'])),
                            'sensor_origin': packet.get('sensor_origin')})
    return projections, 'ready'


def crop_regions(projections, width, height):
    regions = []
    for projection in projections:
        x, y, w, h = projection['bbox']
        if max(w, h) > 80:
            continue
        side = min(max(128, int(max(w, h) * 6)), width, height)
        region = (int(np.clip(x + w/2 - side/2, 0, width-side)),
                  int(np.clip(y + h/2 - side/2, 0, height-side)), side, side)
        if not any(abs(region[0]-r[0]) < side/2 and abs(region[1]-r[1]) < side/2 for r in regions):
            regions.append(region)
    return regions[:4]


def reacquisition_regions(packet, image_stamp, matrix, info, tolerance=.15, max_age=1.):
    """Camera search proposals only; coast predictions are never measurements.

    Require a fresh causal LiDAR packet and exposure-time TF supplied by the
    caller, while permitting confirmed tracks last observed up to one second
    ago. Project the current Kalman uncertainty into pixels for a 3-sigma ROI.
    """
    if not 0 <= image_stamp-packet['stamp'] <= tolerance:
        return []
    projections, _ = project_tracks(packet, image_stamp, matrix, info, tolerance,
                                    search_max_age=max_age)
    k, d = np.asarray(info['k']).reshape(3, 3), np.asarray(info['d'])
    regions = []
    for projection in projections:
        if not projection['search_only']:
            continue
        track = projection['track']
        covariance = np.asarray(track.get('state_covariance'), dtype=float)
        try:
            sigma, noise_model = forecast_model(track, legacy_default_std=8.)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (covariance.shape != (6, 6) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, atol=1e-8)
                or np.linalg.eigvalsh(covariance).min() < -1e-9
                or not np.isfinite(sigma) or sigma <= 0):
            continue
        _, covariance = predict_cv(np.r_[track['position'], track['velocity']], covariance,
                                   image_stamp-packet['stamp'], sigma, noise_model)
        center = np.asarray(projection['camera_position']).reshape(1, 3)
        pixel, jacobian = cv2.projectPoints(center, np.zeros(3), np.zeros(3), k, d if len(d) else None)
        observation = jacobian[:, 3:6] @ matrix[:3, :3]
        pixel_covariance = observation @ covariance[:3, :3] @ observation.T
        pixel_std = np.sqrt(np.maximum(np.diag(pixel_covariance), 0.))
        if not np.isfinite(pixel_std).all():
            continue
        size = np.asarray(projection['bbox'][2:])
        side = int(np.ceil(max(128., 6*size.max(), np.max(size+6*pixel_std))))
        # Search covers the retained short coast; full-frame inference remains
        # the fallback when uncertainty exceeds this bounded crop.
        side = min(side, info['width'], info['height'])
        origin = np.clip(pixel.reshape(2)-side/2, [0, 0],
                         [info['width']-side, info['height']-side]).astype(int)
        regions.append(dict(region=[int(origin[0]), int(origin[1]), side, side],
                            proposal_source='kalman_reacquisition_roi', track_id=track['id'],
                            last_observed_stamp=track['last_seen'],
                            observation_age_s=image_stamp-track['last_seen'],
                            pixel_std=pixel_std.tolist()))
    return regions[:4]


def association_score(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[:2]+a[2:], b[:2]+b[2:])
    intersection = float(np.prod(np.maximum(hi-lo, 0)))
    union = float(np.prod(a[2:]) + np.prod(b[2:]) - intersection)
    iou = intersection / union if union > 0 else 0.
    distance = np.linalg.norm((a[:2]+a[2:]/2) - (b[:2]+b[2:]/2))
    gate = max(5., np.linalg.norm(a[2:]+b[2:]) * .45)
    return max(iou, .3*(1-distance/gate)) if distance < gate else iou


def bearing_update(projection, detection, matrix, info, config=None, enabled=True):
    """One camera update of a fresh LiDAR prior, with no posterior feedback.

    h(x) is the distorted pixel projection at exposure-time camera TF. Its
    Jacobian has rank two: the camera contributes bearing, not depth borrowed
    from LiDAR and incorrectly counted as another independent range sample.
    """
    # 픽셀 오차와 불확실성으로 상태를 보정한다. 단안 영상은 깊이를 새로 측정하지 않는다.
    cfg = config or BearingConfig()
    track = projection['track']
    state = np.r_[projection['position'], track['velocity']].astype(float)
    result = dict(accepted=False, reason='missing_state_covariance',
                  measurement_model='camera_pixel_bearing', independent_depth=False)
    if 'state_covariance' not in track:
        return state, None, result
    covariance = np.asarray(track['state_covariance'], dtype=float)
    if (covariance.shape != (6, 6) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-8)
            or np.linalg.eigvalsh(covariance).min() < -1e-9):
        result['reason'] = 'invalid_state_covariance'
        return state, None, result
    dt = projection['image_stamp'] - projection['lidar_stamp']
    try:
        sigma, noise_model = forecast_model(track, legacy_default_std=8.)
    except (KeyError, TypeError, ValueError, OverflowError):
        result['reason'] = 'invalid_motion_uncertainty'
        return state, None, result
    if noise_model == CV_NOISE_MODEL:
        if dt < 0:
            result['reason'] = 'noncausal_or_unknown_motion_model'
            return state, None, result
        _, covariance = predict_cv(np.r_[track['position'], track['velocity']], covariance, dt, sigma, noise_model)
    else:
        # 구형 기록의 부호 있는 정렬·공분산 계산을 그대로 재현한다.
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        acceleration = np.vstack((np.eye(3) * dt * dt / 2, np.eye(3) * dt))
        covariance = transition @ covariance @ transition.T + acceleration @ acceleration.T * sigma**2
    result['prior_state_covariance'] = covariance.tolist()
    if not enabled:
        result['reason'] = 'coordinate_fusion_disabled'
        return state, covariance, result
    box = np.asarray(detection['bbox'], dtype=float)
    if (detection.get('box_clipped', False) or box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) < 2
            or min(box[:2]) <= 1 or box[0]+box[2] >= info['width']-1
            or box[1]+box[3] >= info['height']-1):
        result['reason'] = 'clipped_or_invalid_camera_box'
        return state, covariance, result
    center = np.asarray(projection['camera_position'], dtype=float)
    pixel, jacobian = cv2.projectPoints(center.reshape(1, 3), np.zeros(3), np.zeros(3),
                                      np.array(info['k']).reshape(3, 3),
                                      np.array(info['d']) if info['d'] else None)
    observation = np.zeros((2, 6))
    observation[:, :3] = jacobian[:, 3:6] @ matrix[:3, :3]
    innovation = box[:2] + box[2:]/2 - pixel.reshape(2)
    variance = cfg.pixel_std**2 + (cfg.box_fraction*box[2:])**2 + cfg.calibration_std**2
    if detection.get('proposal_source') in ('calibrated_lidar_roi', 'kalman_reacquisition_roi'):
        # A crop selected by LiDAR has conditional selection bias; do not
        # treat an enlarged crop as additional independent camera resolution.
        variance *= 1.5**2
    noise = np.diag(variance)
    residual = observation @ covariance @ observation.T + noise
    normalized = float(innovation @ np.linalg.solve(residual, innovation))
    result.update(innovation_px=innovation.tolist(), innovation_mahalanobis_sq=normalized,
                  observation_matrix=observation.tolist(), measurement_covariance=noise.tolist(),
                  measurement_std_px=np.sqrt(variance).tolist())
    if normalized > cfg.innovation_gate:
        result['reason'] = 'camera_innovation_gate'
        return state, covariance, result
    gain = np.linalg.solve(residual, observation @ covariance).T
    posterior = state + gain @ innovation
    correction = np.eye(6) - gain @ observation
    posterior_covariance = correction @ covariance @ correction.T + gain @ noise @ gain.T
    result.update(accepted=True, reason='bearing_update', correction_m=(posterior[:3]-state[:3]).tolist())
    return posterior, (posterior_covariance+posterior_covariance.T)/2, result


def fuse(projections, detections, matrix, info, ambiguity_margin=.05, bearing_config=None,
         position_fusion_enabled=True):
    projections = [p for p in projections if not p.get('search_only', False)
                   and p['track'].get('observed', False)]
    if not projections or not detections:
        return []
    scores = np.array([[association_score(p['bbox'], d['bbox']) for d in detections] for p in projections])
    rows, columns = linear_sum_assignment(-scores)
    # 기존 bbox 연결을 먼저 확정해 이미 발행하던 연결을 보존한다.
    pairs = []
    for row, column in zip(rows, columns):
        alternatives = np.delete(scores[:, column], row)
        if scores[row, column] >= .12 and not (len(alternatives) and
                scores[row, column]-alternatives.max() < ambiguity_margin):
            pairs.append((row, column, False))
    used_rows = {row for row, _, _ in pairs}
    used_columns = {column for _, column, _ in pairs}
    accepted = []
    if position_fusion_enabled:
        for row, projection in enumerate(projections):
            if row in used_rows or np.max(scores[row]) >= .12:
                continue
            for column, detection in enumerate(detections):
                if column in used_columns or np.max(scores[:, column]) >= .12:
                    continue
                if detection.get('semantic_family') != 'drone':
                    continue
                if bearing_update(projection, detection, matrix, info, bearing_config)[2]['accepted']:
                    accepted.append((row, column))
    # 양쪽 모두 유일한 게이트 통과 상대가 있어야 한다. 겹치는 후보는 추측하지 않는다.
    pairs += [(row, column, True) for row, column in accepted
              if sum(r == row for r, _ in accepted) == 1
              and sum(c == column for _, c in accepted) == 1]
    result = []
    inverse = np.linalg.inv(matrix)
    for row, column, fallback in pairs:
        score = scores[row, column]
        projection, detection = projections[row], detections[column]
        pixel = np.array(detection['bbox'][:2]) + np.array(detection['bbox'][2:])/2
        ray = cv2.undistortPoints(pixel.reshape(1, 1, 2), np.array(info['k']).reshape(3, 3),
                                 np.array(info['d']) if info['d'] else None).reshape(2)
        camera_point = np.r_[ray, 1.] * projection['camera_position'][2]
        bearing_depth_position = inverse[:3, :3] @ camera_point + inverse[:3, 3]
        track = projection['track']
        state, covariance, update = bearing_update(projection, detection, matrix, info,
                                                   bearing_config, position_fusion_enabled)
        supplied_prediction = track.get('prediction', {})
        prediction = dict(supplied_prediction) if isinstance(supplied_prediction, dict) else {}
        if covariance is not None:
            horizon = float(prediction.get('horizon_s', .5))
            sigma, noise_model = forecast_model(track, legacy_default_std=8.)
            predicted_state, predicted_covariance = predict_cv(state, covariance, horizon, sigma, noise_model)
            prediction = dict(prediction, model='constant_velocity', horizon_s=horizon, acceleration_std=sigma,
                              process_noise_model=noise_model,
                              stamp=projection['image_stamp']+horizon, position=predicted_state[:3].tolist(),
                              position_covariance=predicted_covariance[:3, :3].tolist())
        result.append({'id': track['id'], 'class_name': detection['class_name'], 'confidence': detection['confidence'],
                       'source': 'camera_lidar', 'stamp': projection['image_stamp'],
                       'birth_stamp': track.get('birth_stamp'),
                       'source_exposure_stamp': projection['image_stamp'],
                       'current_state_stamp': projection['image_stamp'],
                       'last_measurement_stamp': track.get('last_seen', projection['lidar_stamp']),
                       'lidar_stamp': projection['lidar_stamp'], 'position': state[:3].tolist(),
                       'lidar_position': projection['position'], 'lidar_velocity': track['velocity'],
                       'fusion_update': update,
                       'state_covariance': covariance.tolist() if covariance is not None else None,
                       'position_covariance': covariance[:3, :3].tolist() if covariance is not None else None,
                       'velocity_covariance': covariance[3:, 3:].tolist() if covariance is not None else None,
                       'prediction': prediction,
                       'semantic_family': 'drone', 'subtype_ambiguous': detection.get('subtype_ambiguous', False),
                       'subtype_verified': False,
                       'subtype_status': 'ambiguous' if detection.get('subtype_ambiguous') else 'unvalidated',
                       'subtype_candidates': detection.get('subtype_candidates', []),
                       'position_source': ('lidar_camera_bearing_ekf' if update['accepted']
                                           else 'lidar_cluster_center_time_aligned'), 'velocity': state[3:].tolist(),
                       'size': track['size'], 'position_frame_id': projection['frame_id'],
                       # Keep the measured range at the scan pose/time; do not
                       # recompute it from the extrapolated image-time position.
                       'nearest_distance': track.get('nearest_distance'),
                       'nearest_point': track.get('nearest_point'),
                       'nearest_point_frame_id': projection['frame_id'],
                       'nearest_distance_stamp': projection['lidar_stamp'] if track.get('nearest_distance') is not None else None,
                       'nearest_distance_sensor_frame': projection['sensor_frame'],
                       'nearest_distance_sensor_origin': projection['sensor_origin'],
                       'camera_position': (matrix[:3, :3] @ state[:3] + matrix[:3, 3]).tolist(),
                       'lidar_camera_position': projection['camera_position'],
                       'camera_bearing_lidar_depth_position': bearing_depth_position.tolist(),
                       'camera_bbox': detection['bbox'], 'association_score': float(score),
                       'proposal_source': detection.get('proposal_source', 'full_image')})
        if fallback:
            result[-1]['association_method'] = 'unique_gated_bearing_fallback'
    return result
