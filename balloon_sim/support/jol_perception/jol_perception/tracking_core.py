"""점군에서 배경 제거 → 군집화 → 객체 연결 → 칼만 위치·속도 추정을 수행한다.

좌표 단위는 m, 시간은 센서의 초 단위 시각이다. 시뮬레이터 정답은 입력하지 않는다."""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .background_surface import BackgroundSurface

# 기존 10Hz 가속도 설정을 보존하면서 연속 잡음 밀도 q=σ²·0.1s로 변환한다.
CV_NOISE_MODEL = 'continuous_white_acceleration'
LEGACY_CV_NOISE_MODEL = 'piecewise_constant_acceleration'
ACCELERATION_REFERENCE_DT = .1


@dataclass
class Config:
    min_range: float = 0.5
    max_range: float = 75.0
    voxel_size: float = 0.12
    background_distance: float = 0.40
    background_fast_filter: bool = True
    background_plane_filter: bool = True
    noise_radius: float = 0.85
    noise_min_neighbors: int = 2
    cluster_eps: float = 1.20
    cluster_min_points: int = 3
    weak_cluster_min_points: int = 2
    weak_recovery_seconds: float = 1.0
    sparse_temporal_enabled: bool = True
    sparse_min_hits: int = 4
    sparse_confirmation_window: float = 1.5
    sparse_min_span: float = 0.3
    sparse_max_residual: float = 0.45
    sparse_measurement_std: float = 0.5
    sparse_max_candidates: int = 100
    max_cluster_size: float = 8.0
    association_distance: float = 2.5
    association_chi2_gate: float = 25.902  # 3D chi-square, 99.999%; also enforce the distance cap.
    measurement_min_std: float = 0.12
    measurement_angular_std: float = 0.001
    max_speed: float = 18.0
    acceleration_std: float = 8.0
    low_speed_acceleration_std: float = 2.0
    low_speed_threshold: float = 2.0
    innovation_adaptation: bool = True
    prediction_horizon: float = 0.5
    direction_min_speed: float = 0.5
    track_timeout: float = 2.5
    min_hits: int = 3
    confirmation_window: float = 0.5
    tentative_timeout: float = 0.5
    max_time_gap: float = 3.0
    trail_length: int = 100


def forecast_model(track, legacy_default_std=None):
    """융합·조준·평가가 같은 잡음 계약을 검사한다. 무태그 구형 기록만 기본값을 허용한다."""
    if not isinstance(track, dict):
        raise ValueError('Track metadata must be an object')
    prediction = track.get('prediction', {})
    if not isinstance(prediction, dict):
        raise ValueError('Prediction metadata must be an object')
    try:
        model = prediction.get('process_noise_model', LEGACY_CV_NOISE_MODEL)
        untagged = 'process_noise_model' not in prediction and 'process_noise_model' not in track
        sigma = float(prediction.get('acceleration_std', legacy_default_std if untagged else None))
        variance = sigma**2
        if (not np.isfinite([sigma, variance]).all() or sigma <= 0 or model not in (CV_NOISE_MODEL, LEGACY_CV_NOISE_MODEL)
                or track.get('process_noise_model', model) != model):
            raise ValueError('Invalid or inconsistent process-noise model')
        if model == CV_NOISE_MODEL:
            reference = float(prediction['acceleration_reference_dt_s'])
            density = float(prediction['acceleration_spectral_density_m2_s3'])
            if (not np.isfinite([reference, density]).all()
                    or not np.isclose(reference, ACCELERATION_REFERENCE_DT, atol=1e-12, rtol=0)
                    or not np.isclose(density, variance*reference, atol=1e-12, rtol=1e-9)):
                raise ValueError('Continuous acceleration density does not match recorded sigma/reference')
    except (KeyError, TypeError, OverflowError) as error:
        # JSON의 큰 정수·누락값도 동일한 거부 경로로 보내 콜백을 중단시키지 않는다.
        raise ValueError('Invalid process-noise metadata') from error
    return sigma, model


def predict_cv(state, covariance, dt, acceleration_std, noise_model=CV_NOISE_MODEL):
    """[위치, 속도]와 공분산을 예측한다. 입력은 변경하지 않는다.

    연속 백색가속도 Q=q·[[dt³/3 I, dt²/2 I], [dt²/2 I, dt I]].
    같은 q라면 시간을 나눠도 같은 공분산이다. 구형 bag의 모델은 명시적으로 보존한다.
    공분산은 모델의 가정이며 실제 정확도 보증이 아니다.
    """
    state, covariance = np.asarray(state, dtype=float), np.asarray(covariance, dtype=float)
    if (state.shape != (6,) or covariance.shape != (6, 6)
            or not np.isfinite(state).all() or not np.isfinite(covariance).all()
            or not np.isfinite([dt, acceleration_std]).all() or dt < 0 or acceleration_std <= 0
            or noise_model not in (CV_NOISE_MODEL, LEGACY_CV_NOISE_MODEL)):
        raise ValueError('Prediction needs finite 6D state/covariance, nonnegative time and positive noise')
    if dt == 0:
        return state.copy(), covariance.copy()
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    if noise_model == CV_NOISE_MODEL:
        density = acceleration_std**2 * ACCELERATION_REFERENCE_DT
        process_noise = np.kron([[dt**3/3, dt**2/2], [dt**2/2, dt]], np.eye(3)) * density
    else:
        acceleration = np.vstack((np.eye(3) * dt * dt / 2, np.eye(3) * dt))
        process_noise = acceleration @ acceleration.T * acceleration_std ** 2 + np.eye(6) * 1e-6
    return transition @ state, transition @ covariance @ transition.T + process_noise


def voxel_filter(points, voxel_size, sensor_origin=None):
    """Keep one actual return per voxel, choosing the nearest to the sensor."""
    if len(points) == 0 or voxel_size <= 0:
        return points
    relative = points if sensor_origin is None else points - sensor_origin
    order = np.argsort(np.einsum('ij,ij->i', relative, relative), kind='stable')
    cells = np.floor(points[order] / voxel_size).astype(np.int64)
    cells -= cells.min(axis=0)
    dimensions = cells.max(axis=0) + 1
    if math.prod(map(int, dimensions)) <= np.iinfo(np.int64).max:
        # Collision-free mixed-radix key, including originally negative cells.
        keys = (cells[:, 0] * dimensions[1] + cells[:, 1]) * dimensions[2] + cells[:, 2]
        _, first = np.unique(keys, return_index=True)
    else:
        _, first = np.unique(cells, axis=0, return_index=True)
    return points[order[first]]


def dbscan(points, eps, min_points):
    """DBSCAN with a spatial index; noise has label -1.

    Neighborhoods are queried on demand so a dense cloud does not allocate an
    N-by-N neighborhood matrix. This implements DBSCAN, not a clustering package.
    """
    count = len(points)
    labels = np.full(count, -1, dtype=np.int32)
    if not count:
        return labels
    tree = cKDTree(points)
    visited = np.zeros(count, dtype=bool)
    cluster = 0
    for seed in range(count):
        if visited[seed]:
            continue
        visited[seed] = True
        neighbors = tree.query_ball_point(points[seed], eps)
        if len(neighbors) < min_points:
            continue
        labels[seed] = cluster
        pending = deque(neighbors)
        queued = set(neighbors)
        while pending:
            current = pending.popleft()
            if not visited[current]:
                visited[current] = True
                nearby = tree.query_ball_point(points[current], eps)
                if len(nearby) >= min_points:
                    for index in nearby:
                        if index not in queued:
                            pending.append(index)
                            queued.add(index)
            if labels[current] == -1:
                labels[current] = cluster
        cluster += 1
    return labels


def cluster_covariance(size, point_count, position, sensor_origin, config, reference_size=None):
    """Conservative center uncertainty, not a calibrated MID360 accuracy claim.

    Keep the existing 12 cm center-error floor. Uniform extent variance / number
    of returns models sparse surface sampling; cap effective returns at 16 since
    neighbouring beams are correlated. A tunable angular sampling term grows
    transversely with range. Weak matches use the previously observed extent so
    two nearby returns cannot claim the full object's center is well known.
    """
    extent = np.maximum(np.asarray(size, dtype=float), 0.1)
    if reference_size is not None:
        extent = np.maximum(extent, reference_size)
    count = min(max(int(point_count), 1), 16)
    noise = np.diag(config.measurement_min_std ** 2 + extent ** 2 / (12 * count))
    relative = np.asarray(position) - np.asarray(sensor_origin)
    distance = np.linalg.norm(relative)
    if distance > 1e-9:
        bearing = relative / distance
        noise += (distance * config.measurement_angular_std) ** 2 * (np.eye(3) - np.outer(bearing, bearing))
    return noise


@dataclass
class Track:
    id: int
    state: np.ndarray
    covariance: np.ndarray
    size: np.ndarray
    born: float
    last_seen: float
    hits: int = 1
    observed: bool = True
    nearest_point: np.ndarray = None
    point_count: int = 0
    trail: deque = field(default_factory=deque)
    confirmation_times: deque = field(default_factory=deque)
    is_confirmed: bool = False
    measurement_position: np.ndarray = None
    measurement_covariance: np.ndarray = None
    last_strong_seen: float = None
    measurement_strength: str = 'strong'
    sparse_initiated: bool = False
    sparse_history: deque = field(default_factory=deque)
    process_acceleration_std: float = 2.0
    maneuver_acceleration_std: float = 0.0
    prediction_start_stamp: float = 0.0
    innovation_nis: float = None

    def as_dict(self, stamp, sensor_origin=None, config=None):
        cfg = config or Config()
        point = self.nearest_point if self.observed else None
        speed = float(np.linalg.norm(self.state[3:]))
        direction = self.state[3:] / speed if speed >= cfg.direction_min_speed and speed > 1e-9 else None
        horizontal_speed = float(np.linalg.norm(self.state[3:5]))
        sigma = self.acceleration_noise(cfg)
        future_state, future_covariance = predict_cv(self.state, self.covariance, cfg.prediction_horizon, sigma)
        return {'id': self.id, 'stamp': stamp, 'state_stamp': stamp, 'birth_stamp': self.born,
                'measurement_stamp': self.last_seen if self.observed else None,
                'measurement_position': self.measurement_position.tolist()
                    if self.observed and self.measurement_position is not None else None,
                'measurement_covariance': (self.measurement_covariance.tolist()
                    if self.measurement_covariance is not None else (np.eye(3) * cfg.measurement_min_std**2).tolist())
                    if self.observed else None,
                'measurement_strength': self.measurement_strength if self.observed else None,
                'prediction_start_stamp': self.prediction_start_stamp,
                'process_acceleration_std': self.process_acceleration_std,
                'process_noise_model': CV_NOISE_MODEL,
                # 연결에 수락된 실측의 진단값이다. coast와 최초 생성에는 NIS가 없다.
                'innovation_nis': self.innovation_nis if self.observed else None,
                'innovation_dof': 3 if self.observed and self.innovation_nis is not None else None,
                'innovation_stamp': self.last_seen if self.observed and self.innovation_nis is not None else None,
                'position': self.state[:3].tolist(),
                'velocity': self.state[3:].tolist(), 'size': self.size.tolist(),
                'speed_mps': speed, 'direction': direction.tolist() if direction is not None else None,
                'heading_rad': float(np.arctan2(self.state[4], self.state[3]))
                               if direction is not None and horizontal_speed > 1e-9 else None,
                'elevation_rad': float(np.arctan2(self.state[5], horizontal_speed))
                                 if direction is not None else None,
                'position_covariance': self.covariance[:3, :3].tolist(),
                'velocity_covariance': self.covariance[3:, 3:].tolist(),
                'state_covariance': self.covariance.tolist(),
                'prediction': {'model': 'constant_velocity', 'horizon_s': cfg.prediction_horizon,
                               'acceleration_std': sigma,
                               'process_noise_model': CV_NOISE_MODEL,
                               'acceleration_reference_dt_s': ACCELERATION_REFERENCE_DT,
                               'acceleration_spectral_density_m2_s3': sigma**2 * ACCELERATION_REFERENCE_DT,
                               'stamp': stamp + cfg.prediction_horizon,
                               'position': future_state[:3].tolist(),
                               'position_covariance': future_covariance[:3, :3].tolist()},
                'nearest_distance': float(np.linalg.norm(point - (np.zeros(3) if sensor_origin is None else sensor_origin))) if point is not None else None,
                'nearest_point': point.tolist() if point is not None else None,
                'observed': self.observed, 'confirmed': self.is_confirmed,
                'last_seen': self.last_seen, 'time_since_update': max(0.0, stamp - self.last_seen),
                'age': stamp - self.born, 'hits': self.hits,
                'point_count': self.point_count if self.observed else 0,
                'trail': [p.tolist() for p in self.trail]}

    def acceleration_noise(self, config):
        # A last-observed low speed cannot establish continued stillness while
        # measurements are absent. Restore normal manoeuvre uncertainty during
        # confirmed coasts so a real restart can pass the association gate.
        speed = np.linalg.norm(self.state[3:])
        low_speed = (speed < config.low_speed_threshold
                     and (self.observed or not self.is_confirmed))
        if low_speed and self.is_confirmed and self.measurement_strength != 'strong':
            # Sparse surface returns cannot establish stillness from speed alone.
            # Enclose the model's 95% 3D velocity ellipsoid; keep the established
            # dense-cluster low-speed response to suppress surface jitter.
            variance = max(0., float(np.linalg.eigvalsh(self.covariance[3:, 3:]).max()))
            low_speed = speed + math.sqrt(7.815 * variance) < config.low_speed_threshold
        base = config.low_speed_acceleration_std if low_speed else config.acceleration_std
        return max(base, self.maneuver_acceleration_std) if config.innovation_adaptation else base


class Tracker:
    def __init__(self, config=None):
        self.config = config or Config()
        if (not np.isfinite([self.config.acceleration_std, self.config.low_speed_acceleration_std,
                             self.config.low_speed_threshold]).all()
                or min(self.config.acceleration_std, self.config.low_speed_acceleration_std) <= 0
                or self.config.low_speed_threshold < 0):
            raise ValueError('Acceleration noise must be positive and low-speed threshold nonnegative')
        if (not np.isfinite([self.config.prediction_horizon, self.config.direction_min_speed]).all()
                or min(self.config.prediction_horizon, self.config.direction_min_speed) < 0):
            raise ValueError('Prediction horizon and direction speed threshold must be finite and nonnegative')
        if (not isinstance(self.config.min_hits, int) or self.config.min_hits < 1
                or not np.isfinite([self.config.confirmation_window, self.config.tentative_timeout]).all()
                or min(self.config.confirmation_window, self.config.tentative_timeout) <= 0):
            raise ValueError('Confirmation needs a positive hit count, time window and tentative timeout')
        if (not isinstance(self.config.weak_cluster_min_points, int) or self.config.weak_cluster_min_points < 1
                or not np.isfinite([self.config.weak_recovery_seconds, self.config.association_chi2_gate,
                                    self.config.measurement_min_std, self.config.measurement_angular_std]).all()
                or min(self.config.weak_recovery_seconds, self.config.association_chi2_gate,
                       self.config.measurement_min_std) <= 0 or self.config.measurement_angular_std < 0):
            raise ValueError('Association and measurement uncertainty parameters must be finite and positive')
        if (not isinstance(self.config.sparse_min_hits, int) or self.config.sparse_min_hits < 4
                or not isinstance(self.config.sparse_max_candidates, int) or self.config.sparse_max_candidates < 1
                or not np.isfinite([self.config.sparse_confirmation_window, self.config.sparse_min_span,
                                    self.config.sparse_max_residual, self.config.sparse_measurement_std]).all()
                or min(self.config.sparse_confirmation_window, self.config.sparse_min_span,
                       self.config.sparse_max_residual, self.config.sparse_measurement_std) <= 0
                or self.config.sparse_min_span > self.config.sparse_confirmation_window):
            raise ValueError('Sparse acquisition needs four or more timed observations and positive finite bounds')
        self.background = np.empty((0, 3), dtype=np.float64)
        self.background_tree = None
        self.background_surface = None
        self.background_cells = None
        self.tracks = []
        self.last_stamp = None
        self.next_id = 1
        self.last_metrics = {}
        self.last_detections = []

    def clean(self, points, sensor_origin=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        origin = np.zeros(3) if sensor_origin is None else np.asarray(sensor_origin, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Sensor origin must have three finite coordinates')
        ranges = np.linalg.norm(points - origin, axis=1)
        keep = (ranges >= self.config.min_range) & (ranges <= self.config.max_range)
        return voxel_filter(points[keep], self.config.voxel_size, origin)

    def set_background(self, points, sensor_origin=None, already_cleaned=False):
        # Multi-pose capture was range-filtered at each acquisition origin.
        if already_cleaned:
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            cleaned = voxel_filter(points[np.isfinite(points).all(axis=1)], self.config.voxel_size)
        else:
            cleaned = self.clean(points, sensor_origin)
        if len(cleaned) < 3:
            raise ValueError('Background requires at least three finite, in-range returns')
        # Build before publishing any replacement: a failed map fit/load must
        # leave the previous background and its track lifecycle together.
        background_tree = cKDTree(cleaned)
        background_surface = BackgroundSurface.fit(cleaned) if self.config.background_plane_filter else None
        background_cells = None
        if self.config.background_fast_filter and self.config.background_distance > 0:
            # Every point in an occupied cell is within the original threshold:
            # cell diagonal < background_distance. This is an exact shortcut,
            # not an approximate replacement for the boundary KD-tree search.
            cell_size = self.config.background_distance / (math.sqrt(3) * (1 + 1e-10))
            scaled = np.floor(cleaned / cell_size)
            if np.max(np.abs(scaled)) < np.iinfo(np.int64).max / 4:
                cells = scaled.astype(np.int64)
                low, high = cells.min(axis=0), cells.max(axis=0)
                dimensions = high - low + 1
                if math.prod(map(int, dimensions)) <= np.iinfo(np.int64).max:
                    relative = cells - low
                    keys = ((relative[:, 0] * dimensions[1] + relative[:, 1])
                            * dimensions[2] + relative[:, 2])
                    background_cells = (cell_size, low, high, dimensions, np.unique(keys))
        self.background = cleaned
        self.background_tree = background_tree
        self.background_surface = background_surface
        self.background_cells = background_cells
        self.reset_tracks()

    def foreground_mask(self, points):
        """Exact background decision, with guaranteed-interior occupied cells."""
        foreground = (~self.background_surface.contains(points) if self.background_surface is not None
                      else np.ones(len(points), dtype=bool))
        if len(points) and self.background_cells is not None:
            size, low, high, dims, occupied = self.background_cells
            cells = np.floor(points / size)
            indices = np.flatnonzero(np.all((cells >= low) & (cells <= high), axis=1))
            relative = cells[indices].astype(np.int64) - low
            keys = ((relative[:, 0] * dims[1] + relative[:, 1]) * dims[2] + relative[:, 2])
            locations = np.searchsorted(occupied, keys)
            valid = locations < len(occupied)
            interior = np.zeros(len(keys), dtype=bool)
            interior[valid] = occupied[locations[valid]] == keys[valid]
            foreground[indices[interior]] = False
        remaining = np.flatnonzero(foreground)
        if len(remaining):
            distance, _ = self.background_tree.query(
                points[remaining], k=1,
                distance_upper_bound=np.nextafter(self.config.background_distance, np.inf))
            foreground[remaining] = distance > self.config.background_distance
        self.background_query_points = len(remaining)
        return foreground

    def save_background(self, filename, frame_id, sensor_pose=None,
                        map_id='jol_monitor', stabilized=True):
        path = Path(filename).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, points=self.background, frame_id=frame_id,
                                sensor_pose=np.asarray(sensor_pose if sensor_pose is not None else []),
                                map_id=map_id, stabilized=stabilized, version=2)
        temporary.replace(path)

    def load_background(self, filename, frame_id, sensor_pose=None,
                        position_tolerance=0.02, angle_tolerance=0.005,
                        map_id='jol_monitor', stabilized=True):
        with np.load(Path(filename).expanduser(), allow_pickle=False) as data:
            version = int(data['version'])
            if version not in (1, 2):
                raise ValueError('Unsupported background format version')
            if str(data['frame_id']) != frame_id:
                raise ValueError('Saved background frame does not match tracking frame')
            if stabilized:
                if version != 2 or not bool(data['stabilized']):
                    raise ValueError('Sensor-frame background cannot be loaded as a fixed-frame map; recapture')
                if str(data['map_id']) != map_id:
                    raise ValueError('Saved background map identity does not match this scene')
            else:
                if version == 2 and bool(data['stabilized']):
                    raise ValueError('Fixed-frame map cannot be loaded in legacy sensor-frame mode')
                if not pose_matches(data['sensor_pose'], sensor_pose,
                                    position_tolerance, angle_tolerance):
                    raise ValueError('Saved background sensor pose does not match current TF')
            self.set_background(data['points'], already_cleaned=True)

    def reset_tracks(self):
        self.tracks = []
        self.last_stamp = None
        self.last_detections = []
        # IDs are never reused after a time reset or background recapture.

    def process(self, points, stamp, sensor_origin=None):
        # 순서: 입력 검사 → 고정 배경 제거 → 군집 후보 → 기존 ID 연결.
        started = time.perf_counter()
        if not np.isfinite(stamp):
            raise ValueError('Timestamp must be finite')
        if self.background_tree is None:
            raise RuntimeError('Capture or load an empty-scene background first')
        if self.last_stamp is not None and stamp == self.last_stamp:
            self.last_metrics = {'status': 'duplicate_stamp', 'processing_ms': 0.0}
            return None, np.empty((0, 3))
        # Rejected input must not consume its timestamp or erase an old epoch's
        # tracks. A corrected cloud at the same timestamp can then be retried.
        origin = np.zeros(3) if sensor_origin is None else np.asarray(sensor_origin, dtype=float)
        cleaned = self.clean(points, origin)
        reset_reason = None
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.reset_tracks()
            reset_reason = 'backwards_stamp_reset'
        if self.last_stamp is not None and stamp - self.last_stamp > self.config.max_time_gap:
            self.reset_tracks()
            reset_reason = 'time_gap_reset'
        dt = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        foreground = cleaned[self.foreground_mask(cleaned)]
        isolated = np.empty((0, 3))
        if len(foreground):
            tree = cKDTree(foreground)
            counts = tree.query_ball_point(foreground, self.config.noise_radius,
                                          return_length=True)
            isolated = foreground[counts < self.config.noise_min_neighbors]
            foreground = foreground[counts >= self.config.noise_min_neighbors]
        labels = dbscan(foreground, self.config.cluster_eps, self.config.cluster_min_points)
        groups = [(foreground[labels == label], True)
                  for label in range(int(labels.max()) + 1 if len(labels) else 0)]
        # Strong clusters retain the original DBSCAN decision. Only remaining
        # noise can become a weak cluster; no return is assigned to two objects.
        if self.config.weak_cluster_min_points < self.config.cluster_min_points:
            residual = foreground[labels == -1]
            weak_labels = dbscan(residual, self.config.cluster_eps, self.config.weak_cluster_min_points)
            groups += [(residual[weak_labels == label], False)
                       for label in range(int(weak_labels.max()) + 1 if len(weak_labels) else 0)]
        detections = []
        for members, strong in groups:
            low, high = members.min(axis=0), members.max(axis=0)
            size = high - low
            if size.max() > self.config.max_cluster_size:
                continue
            nearest = members[np.argmin(np.linalg.norm(members - origin, axis=1))]
            detections.append({'position': (low + high) / 2, 'size': np.maximum(size, 0.1),
                               'nearest_point': nearest, 'point_count': len(members),
                               'strong': strong, 'sensor_origin': origin,
                               'covariance': cluster_covariance(size, len(members), (low + high) / 2,
                                                                 origin, self.config)})
        sparse_overflow = self.config.sparse_temporal_enabled and len(isolated) > self.config.sparse_max_candidates
        if self.config.sparse_temporal_enabled and not sparse_overflow:
            # A singleton is only evidence for a time-consistent tentative track;
            # it never bypasses DBSCAN and becomes an immediately published object.
            sparse_labels = dbscan(isolated, self.config.cluster_eps, 1)
            for label in range(int(sparse_labels.max())+1 if len(sparse_labels) else 0):
                members = isolated[sparse_labels == label]
                low, high = members.min(axis=0), members.max(axis=0)
                nearest = members[np.argmin(np.linalg.norm(members-origin, axis=1))]
                detections.append(dict(position=(low+high)/2, size=np.maximum(high-low, .1), nearest_point=nearest,
                    point_count=len(members), strong=False, sparse=True, sensor_origin=origin,
                    covariance=np.eye(3)*self.config.sparse_measurement_std**2))
        self._associate(detections, stamp, dt)
        self.last_detections = [dict(position=d['position'].tolist(), size=d['size'].tolist(),
                                    nearest_point=d['nearest_point'].tolist(), point_count=d['point_count'],
                                    measurement_covariance=d['covariance'].tolist(), strong=d['strong'],
                                    sparse=d.get('sparse', False))
                                for d in detections]
        output = [track.as_dict(stamp, origin, self.config) for track in self.tracks if track.is_confirmed]
        elapsed = 1000.0 * (time.perf_counter() - started)
        self.last_metrics = {'status': reset_reason or 'tracking', 'input_points': len(points),
                             'filtered_points': len(cleaned), 'foreground_points': len(foreground),
                             'background_query_points': self.background_query_points,
                             'clusters': len(detections), 'tracks': len(output),
                             'strong_clusters': sum(d['strong'] for d in detections),
                             'weak_clusters': sum(not d['strong'] for d in detections),
                             'sparse_candidates': sum(d.get('sparse', False) for d in detections),
                             'sparse_candidate_overflow': bool(sparse_overflow),
                             'weak_updates': sum(t['measurement_strength'] == 'weak' for t in output),
                             'observed_tracks': sum(t['observed'] for t in output),
                             'processing_ms': elapsed}
        return output, foreground

    def _associate(self, detections, stamp, dt):
        # 이전 상태를 예측한 뒤 강한 관측부터 연결해 약한 점군의 ID 탈취를 줄인다.
        cfg = self.config
        self.tracks = [track for track in self.tracks if stamp - track.last_seen <=
                       (cfg.track_timeout if track.is_confirmed else cfg.sparse_confirmation_window
                        if track.sparse_initiated else cfg.tentative_timeout)]
        for track in self.tracks:
            # Low-speed surface sampling changes should not imply high acceleration.
            # Keep the original manoeuvre response above this measured-speed threshold.
            sigma = track.acceleration_noise(cfg)
            track.prediction_start_stamp, track.process_acceleration_std = stamp-dt, sigma
            track.state, track.covariance = predict_cv(track.state, track.covariance, dt, sigma)
            track.observed = False
        matches, used_detections, used_tracks = [], set(), set()
        for strong, confirmed in ((True, True), (True, False), (False, True), (False, False)):
            # 99% 3D gate for established weak tracks tolerates sparse stop/start
            # residuals. Tentative births retain 95% and temporal confirmation;
            # all stages still enforce the independent physical distance cap.
            chi2_gate = cfg.association_chi2_gate if strong else (11.345 if confirmed else 7.815)
            track_indices = [i for i, t in enumerate(self.tracks) if i not in used_tracks and
                             t.is_confirmed == confirmed and
                             not (strong and not confirmed and t.sparse_initiated) and
                             (strong or (cfg.sparse_temporal_enabled and t.sparse_initiated) or
                              (t.is_confirmed and t.last_strong_seen is not None and
                               stamp - t.last_strong_seen <= cfg.weak_recovery_seconds))]
            detection_indices = [j for j, d in enumerate(detections) if j not in used_detections
                                 and d.get('strong', True) == strong]
            if not track_indices or not detection_indices:
                continue
            cost = np.full((len(track_indices), len(detection_indices)), 1e9)
            for row, i in enumerate(track_indices):
                track = self.tracks[i]
                gate = cfg.association_distance + cfg.max_speed * min(stamp - track.last_seen, 0.5)
                for column, j in enumerate(detection_indices):
                    detection = detections[j]
                    innovation = detection['position'] - track.state[:3]
                    if np.linalg.norm(innovation) > gate:
                        continue
                    if track.sparse_initiated and not track.is_confirmed:
                        previous_stamp, previous_point = track.sparse_history[-1]
                        if np.linalg.norm(detection['position'] - previous_point) > cfg.max_speed * (stamp-previous_stamp) + cfg.sparse_measurement_std:
                            continue
                    noise = self._measurement_noise(detection, track)
                    residual = track.covariance[:3, :3] + noise
                    mahalanobis = float(innovation @ np.linalg.solve(residual, innovation))
                    # 시간 연속성으로 확정한 단일점 트랙은 R로 희소성을 반영하고 공통 3D 게이트를 쓴다.
                    # 강군집의 잔여점 복구·미확정 후보는 기존의 좁은 게이트를 유지한다.
                    gate_nis = (cfg.association_chi2_gate if confirmed and track.sparse_initiated
                                and detection.get('sparse', False) else chi2_gate)
                    if mahalanobis <= gate_nis:
                        cost[row, column] = mahalanobis + 0.15 * np.linalg.norm(track.size - detection['size'])
            rows, columns = linear_sum_assignment(cost)
            stage_matches = [(track_indices[i], detection_indices[j]) for i, j in zip(rows, columns)
                             if cost[i, j] < 1e9]
            matches.extend(stage_matches)
            used_tracks.update(i for i, _ in stage_matches)
            used_detections.update(j for _, j in stage_matches)
        observation = np.eye(3, 6)
        for i, j in matches:
            track, detection = self.tracks[i], detections[j]
            strong = detection.get('strong', True)
            measurement_noise = self._measurement_noise(detection, track)
            innovation = detection['position'] - observation @ track.state
            residual_cov = observation @ track.covariance @ observation.T + measurement_noise
            nis = float(innovation @ np.linalg.solve(residual_cov, innovation))
            track.innovation_nis = nis
            gain = np.linalg.solve(residual_cov, observation @ track.covariance).T
            track.state += gain @ innovation
            # Joseph form keeps covariance positive despite floating-point roundoff.
            correction = np.eye(6) - gain @ observation
            track.covariance = (correction @ track.covariance @ correction.T
                                + gain @ measurement_noise @ gain.T)
            if strong:
                track.size = 0.7 * track.size + 0.3 * detection['size']
                track.last_strong_seen = stamp
                # A large *accepted real* innovation indicates a manoeuvre that
                # the low-speed model did not explain. Increase next interval's
                # Q, then decay; never tune from truth or a predicted observation.
                track.maneuver_acceleration_std *= 0.5
                if cfg.innovation_adaptation and nis > 16.266:
                    track.maneuver_acceleration_std = min(5 * cfg.acceleration_std,
                        max(track.maneuver_acceleration_std, cfg.acceleration_std * nis / 3))
            track.last_seen, track.observed = stamp, True
            track.nearest_point = detection['nearest_point']
            track.point_count = detection['point_count']
            track.measurement_position = detection['position'].copy()
            track.measurement_covariance = measurement_noise.copy()
            track.measurement_strength = 'strong' if strong else 'sparse' if detection.get('sparse') else 'weak'
            track.hits += 1
            if strong:
                track.confirmation_times.append(stamp)
            while track.confirmation_times and stamp-track.confirmation_times[0] > cfg.confirmation_window + 1e-9:
                track.confirmation_times.popleft()
            was_confirmed = track.is_confirmed
            track.is_confirmed |= len(track.confirmation_times) >= cfg.min_hits
            if track.sparse_initiated:
                track.sparse_history.append((stamp, detection['position'].copy()))
                while track.sparse_history and stamp-track.sparse_history[0][0] > cfg.sparse_confirmation_window:
                    track.sparse_history.popleft()
                track.is_confirmed |= self._sparse_confirmed(track.sparse_history)
                # Sparse returns cannot resolve an additional object inside the
                # same spatial resolution used by DBSCAN. Two surface returns
                # must not promote two IDs for one already confirmed object.
                if not was_confirmed and any(other is not track and other.is_confirmed and
                        np.linalg.norm(other.state[:3]-track.state[:3]) <= cfg.cluster_eps
                        for other in self.tracks):
                    track.is_confirmed = False
            used_detections.add(j)
        sparse_tentatives = sum(t.sparse_initiated and not t.is_confirmed for t in self.tracks)
        for j, detection in enumerate(detections):
            strong = detection.get('strong', True)
            if j in used_detections or (not strong and not cfg.sparse_temporal_enabled):
                continue
            if not strong:
                if any(track.observed and np.linalg.norm(track.state[:3]-detection['position']) <= cfg.cluster_eps
                       for track in self.tracks):
                    continue
                if sparse_tentatives >= cfg.sparse_max_candidates:
                    continue
                sparse_tentatives += 1
            noise = detection['covariance'].copy()
            if not strong:
                noise += np.eye(3)*max(0., cfg.sparse_measurement_std**2 - np.linalg.eigvalsh(noise).min())
            self.tracks.append(Track(self.next_id, np.r_[detection['position'], np.zeros(3)],
                                     np.diag([0.25] * 3 + [100.] * 3), detection['size'],
                                     stamp, stamp, nearest_point=detection['nearest_point'],
                                     point_count=detection['point_count'],
                                     measurement_position=detection['position'].copy(),
                                     measurement_covariance=noise,
                                     last_strong_seen=stamp if strong else None,
                                     measurement_strength='strong' if strong else 'sparse' if detection.get('sparse') else 'weak',
                                     sparse_initiated=not strong,
                                     sparse_history=deque([(stamp, detection['position'].copy())], maxlen=16),
                                     prediction_start_stamp=stamp,
                                     process_acceleration_std=cfg.low_speed_acceleration_std,
                                     confirmation_times=deque([stamp] if strong else []), is_confirmed=strong and cfg.min_hits == 1,
                                     trail=deque(maxlen=cfg.trail_length)))
            self.next_id += 1
        for track in self.tracks:
            if track.observed:
                track.trail.append(track.state[:3].copy())

    def _measurement_noise(self, detection, track):
        if detection['strong']:
            return detection['covariance']
        noise = cluster_covariance(detection['size'], detection['point_count'], detection['position'],
                                   detection['sensor_origin'], self.config, track.size)
        if detection.get('sparse') or track.sparse_initiated:
            noise += np.eye(3)*max(0., self.config.sparse_measurement_std**2-np.linalg.eigvalsh(noise).min())
        return noise

    def _sparse_confirmed(self, history):
        cfg = self.config
        if len(history) < cfg.sparse_min_hits or history[-1][0]-history[0][0] < cfg.sparse_min_span:
            return False
        times = np.asarray([t for t, _ in history])
        positions = np.asarray([p for _, p in history])
        design = np.column_stack((np.ones(len(times)), times-times.mean()))
        fit = np.linalg.lstsq(design, positions, rcond=None)[0]
        return bool(np.linalg.norm(fit[1]) <= cfg.max_speed and
                    np.max(np.linalg.norm(design @ fit-positions, axis=1)) <= cfg.sparse_max_residual)


def pose_matches(saved, current, position_tolerance=0.02, angle_tolerance=0.005):
    """Compare [tx, ty, tz, qx, qy, qz, qw], accepting quaternion sign flips."""
    saved, current = np.asarray(saved, dtype=float), np.asarray(current, dtype=float)
    if saved.shape != (7,) or current.shape != (7,) or not np.isfinite([saved, current]).all():
        return False
    if np.linalg.norm(saved[:3] - current[:3]) > position_tolerance:
        return False
    q1, q2 = saved[3:], current[3:]
    if min(np.linalg.norm(q1), np.linalg.norm(q2)) < 1e-8:
        return False
    dot = abs(np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return 2 * np.arccos(np.clip(dot, 0, 1)) <= angle_tolerance
