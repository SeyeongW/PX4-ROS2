"""Deterministic sensor-like NPZ replay checks, not Gazebo/physical ROSbag proof.

Run: PYTHONPATH=src/jol_perception python3 src/jol_perception/test/test_tracking.py
With --output DIR, keep the generated recordings and metrics for inspection.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from jol_perception.tracking_core import Config, Track, Tracker, cluster_covariance, dbscan, pose_matches, voxel_filter


def test_confirmed_sparse_restart_keeps_id_without_weak_clutter_stealing_dense_track():
    # 재출발의 실제 단일점은 받아들이되 강군집의 잔여 잡음까지 게이트를 넓히지 않는다.
    for sparse_birth, sparse_measurement, confirmed, expected in (
            (True, True, True, True), (False, True, True, False),
            (True, False, True, False), (True, True, False, False)):
        tracker = Tracker()
        tracker.next_id = 2
        track = Track(1, np.array([20., 0., 8., 0., 0., 0.]), np.eye(6)*.25,
                      np.ones(3), 1., 2.1, is_confirmed=confirmed, observed=False,
                      sparse_initiated=sparse_birth, last_strong_seen=2.1 if not sparse_birth else None,
                      sparse_history=deque([(2.1, np.array([20., 0., 8.]))]))
        tracker.tracks = [track]
        detection = dict(position=np.array([23., 0., 8.]), size=np.ones(3)*.1,
                         nearest_point=np.array([23., 0., 8.]), point_count=1,
                         strong=False, sparse=sparse_measurement, sensor_origin=np.zeros(3),
                         covariance=np.eye(3)*.25)
        tracker._associate([detection], 2.7, .1)
        assert track.observed is expected
        if expected:
            assert track.id == 1 and track.measurement_strength == 'sparse'
            assert 11.345 < track.innovation_nis < tracker.config.association_chi2_gate
        # 공분산이 커도 물리적 거리 상한 밖 관측은 같은 ID에 연결할 수 없다.
        track.covariance = np.eye(6)*1000
        detection['position'] = detection['nearest_point'] = np.array([70., 0., 8.])
        tracker._associate([detection], 2.8, .1)
        assert not track.observed


def make_recording(path, scenario):
    rng = np.random.default_rng(739)
    background = np.column_stack((rng.uniform(-65, 65, 1800),
                                  rng.uniform(-65, 65, 1800), np.full(1800, -2.047)))
    shape = rng.uniform([-0.9, -0.9, -0.3], [0.9, 0.9, 0.3], (100, 3))
    stamps = np.arange(140) * 0.1
    object_count = 2 if scenario in ('multi', 'crossing') else 1
    truth = np.full((len(stamps), object_count, 3), np.nan)
    visible = np.ones((len(stamps), object_count), dtype=bool)
    clouds, offsets = [], [0]
    for frame, stamp in enumerate(stamps):
        parts = [background + rng.normal(0, 0.012, background.shape)]
        for object_index in range(object_count):
            motion_time = stamp
            if scenario == 'stop':
                motion_time = min(stamp, 4.0) if stamp < 8 else stamp - 4
            angle = motion_time * 0.2 + object_index * 0.20
            center = np.array([50 * np.cos(angle), 50 * np.sin(angle), 8.0])
            if scenario == 'crossing':
                # Paths cross in plan view with 3 m vertical separation, preserving
                # resolvable clusters. Exactly overlapping bodies are unobservable.
                center = np.array([(-12 + 2 * stamp) * (1 if object_index == 0 else -1),
                                   48.0, 8.0 + object_index * 3.0])
            truth[frame, object_index] = center
            hidden = scenario == 'occlusion' and 45 <= frame < 58
            hidden |= scenario == 'long_reappearance' and 45 <= frame < 80
            visible[frame, object_index] = not hidden
            if not hidden:
                parts.append(center + shape + rng.normal(0, 0.012, shape.shape))
        parts.append(rng.uniform([-60, -60, 0], [60, 60, 30], (15, 3)))
        points = np.concatenate(parts).astype(np.float32)
        clouds.append(points)
        offsets.append(offsets[-1] + len(points))
    np.savez_compressed(path, points=np.concatenate(clouds), offsets=offsets,
                        stamps=stamps, background=background, truth=truth, visible=visible)


def replay(path):
    tracker = Tracker()
    history, latencies, speed_errors, position_errors = [], [], [], []
    false_positives = id_switches = missed_visible = 0
    last_ids = {}
    with np.load(path, allow_pickle=False) as recording:
        tracker.set_background(recording['background'])
        for frame, stamp in enumerate(recording['stamps']):
            start, end = recording['offsets'][frame:frame + 2]
            tracks, _ = tracker.process(recording['points'][start:end], float(stamp))
            history.append(tracks)
            latencies.append(tracker.last_metrics['processing_ms'])
            observed = [track for track in tracks if track['observed']]
            active_indices = np.flatnonzero(recording['visible'][frame])
            centers = recording['truth'][frame, active_indices]
            matched_tracks, matched_truth = set(), set()
            if observed and len(centers):
                positions = np.array([track['position'] for track in observed])
                distances = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
                rows, columns = linear_sum_assignment(distances)
                for row, column in zip(rows, columns):
                    if distances[row, column] > 2.0:
                        continue
                    object_index = int(active_indices[column])
                    track_id = observed[row]['id']
                    if object_index in last_ids and last_ids[object_index] != track_id:
                        id_switches += 1
                    last_ids[object_index] = track_id
                    matched_tracks.add(row)
                    matched_truth.add(column)
                    position_errors.append(float(distances[row, column]))
                    if frame > 10:
                        truth_velocity = (recording['truth'][frame, object_index]
                                          - recording['truth'][frame - 1, object_index]) / 0.1
                        speed_errors.append(float(abs(np.linalg.norm(observed[row]['velocity'])
                                                     - np.linalg.norm(truth_velocity))))
            false_positives += len(observed) - len(matched_tracks)
            if frame >= tracker.config.min_hits - 1:
                # Explicit initial confirmation warmup; later misses still count.
                missed_visible += len(centers) - len(matched_truth)
    return {'scenario': path.stem, 'frames': len(history), 'false_positive_detections': false_positives,
            'id_switches': id_switches, 'missed_visible_detections': missed_visible,
            'processing_ms_p50': float(np.median(latencies)),
            'processing_ms_p95': float(np.percentile(latencies, 95)),
            'position_error_m_p95': float(np.percentile(position_errors, 95)),
            'speed_error_mps_p95': float(np.percentile(speed_errors, 95))}, history


def test_recorded_scenarios(output_dir=None):
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix='jol-replay-') as temporary:
            return test_recorded_scenarios(Path(temporary))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for scenario in ['single', 'multi', 'stop', 'occlusion', 'crossing', 'long_reappearance']:
        recording_path = output_dir / f'{scenario}.npz'
        make_recording(recording_path, scenario)
        report, history = replay(recording_path)
        reports.append(report)
        assert report['false_positive_detections'] == 0, report
        assert report['id_switches'] == (1 if scenario == 'long_reappearance' else 0), report
        assert report['missed_visible_detections'] <= (Config().min_hits - 1 if scenario == 'long_reappearance' else 0), report
        assert report['position_error_m_p95'] < 0.25, report
        if scenario == 'stop':
            assert history[75][0]['observed'] and np.linalg.norm(history[75][0]['velocity']) < 0.1
        if scenario == 'occlusion':
            assert history[50][0]['id'] == history[60][0]['id']
            assert not history[50][0]['observed'] and history[50][0]['nearest_distance'] is None
        if scenario == 'long_reappearance':
            assert not history[75] and history[85][0]['id'] != history[40][0]['id']
    (output_dir / 'synthetic_replay_metrics.json').write_text(json.dumps(
        {'source': 'generated synthetic NPZ; not physical sensor or Gazebo ROSbag', 'results': reports}, indent=2))
    return reports


def test_guards_and_surface_range():
    tracker = Tracker(Config(voxel_size=0.1, noise_min_neighbors=2, cluster_min_points=3, min_hits=2))
    tracker.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
    points = np.array([[5., 0., 0.], [5.3, 0., 0.], [5.5, 0.2, 0.], [np.nan, 0, 0], [np.inf, 0, 0]])
    assert not tracker.process(points, 1.0)[0]
    tracks, _ = tracker.process(points, 1.1)
    original_id = tracks[0]['id']
    assert tracks[0]['nearest_distance'] == 5.0
    assert np.linalg.norm(tracks[0]['position']) > 5.0
    assert tracker.process(points, 1.1)[0] is None
    assert tracker.last_metrics['status'] == 'duplicate_stamp'
    assert tracker.process(points, 0.1)[0] == []
    assert tracker.last_metrics['status'] == 'backwards_stamp_reset'
    assert tracker.process(points, 0.2)[0][0]['id'] != original_id
    assert tracker.process(points, 10.0)[0] == []
    assert tracker.last_metrics['status'] == 'time_gap_reset'
    pose = [0, 0, 2.047, 0, 0, 0, 1]
    assert pose_matches(pose, [0, 0, 2.047, 0, 0, 0, -1])
    assert not pose_matches(pose, [1, 0, 2.047, 0, 0, 0, 1])
    with tempfile.TemporaryDirectory() as directory:
        saved = str(Path(directory) / 'background.npz')
        tracker.save_background(saved, 'jol/lidar', pose, stabilized=False)
        tracker.load_background(saved, 'jol/lidar', pose, stabilized=False)
        for frame, changed_pose in [('wrong_frame', pose), ('jol/lidar', [0, 0, 3, 0, 0, 0, 1])]:
            try:
                tracker.load_background(saved, frame, changed_pose, stabilized=False)
                raise AssertionError('Invalid metadata accepted')
            except ValueError:
                pass
    labels = dbscan(np.array([[0, 0, 0], [.1, 0, 0], [.2, 0, 0], [10, 0, 0]]), .15, 2)
    assert list(labels) == [0, 0, 0, -1]
    rng = np.random.default_rng(31)
    cloud = rng.uniform(-75, 75, (10000, 3))
    order = np.argsort(np.einsum('ij,ij->i', cloud, cloud), kind='stable')
    _, first = np.unique(np.floor(cloud[order] / .12).astype(np.int64), axis=0, return_index=True)
    assert np.array_equal(voxel_filter(cloud, .12), cloud[order[first]])
    boundary = Tracker(Config(background_distance=.5, voxel_size=.01,
                              noise_min_neighbors=1, cluster_min_points=1, min_hits=1))
    boundary.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
    assert not boundary.process(np.array([[3., 0., -1.5]]), 1.0)[0]
    assert len(boundary.process(np.array([[3., 0., -1.49]]), 1.1)[0]) == 1


def test_rejected_cloud_does_not_consume_timestamp_or_reset_tracks():
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    cloud = np.array([[20., 0., 8.], [20.2, 0., 8.], [20.4, .2, 8.]])
    # Include both kinds of legitimate epoch reset as well as an ordinary next
    # scan. Invalid geometry alone must never trigger any of those transitions.
    for stamp, bad_cloud, bad_origin in [(11., cloud, [np.nan, 0., 0.]),
                                         (1., cloud, [np.nan, 0., 0.]),
                                         (20., cloud, [np.nan, 0., 0.]),
                                         (11., np.array([[1., 2.]]), [0., 0., 0.])]:
        tracker = Tracker(Config(min_hits=1))
        tracker.set_background(background)
        tracker.process(cloud, 10.)
        track = tracker.tracks[0]
        state, covariance = track.state.copy(), track.covariance.copy()
        next_id, detections = tracker.next_id, tracker.last_detections.copy()
        try:
            tracker.process(bad_cloud, stamp, bad_origin)
            raise AssertionError('Invalid geometry was accepted')
        except ValueError:
            pass
        assert tracker.last_stamp == 10. and tracker.tracks[0] is track
        assert tracker.next_id == next_id and tracker.last_detections == detections
        assert np.array_equal(track.state, state) and np.array_equal(track.covariance, covariance)
        corrected, _ = tracker.process(cloud, stamp)
        assert len(corrected) == 1 and corrected[0]['observed']
        assert (corrected[0]['id'] == track.id) == (stamp == 11.)
        # A successfully consumed timestamp still rejects an actual duplicate.
        assert tracker.process(cloud, stamp)[0] is None


def test_failed_background_fit_preserves_complete_previous_map_and_track_epoch():
    from unittest.mock import patch
    tracker = Tracker(Config(min_hits=1))
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    cloud = np.array([[20., 0., 8.], [20.2, 0., 8.], [20.4, .2, 8.]])
    tracker.set_background(background)
    tracker.process(cloud, 10.)
    previous = (tracker.background, tracker.background_tree, tracker.background_surface,
                tracker.background_cells, tracker.tracks[0])
    try:
        with patch('jol_perception.tracking_core.BackgroundSurface.fit',
                   side_effect=ValueError('Simulated map-fit failure')):
            tracker.set_background(background+[10., 0., 0.])
        raise AssertionError('Failed map fit was accepted')
    except ValueError:
        pass
    assert all(old is new for old, new in zip(previous,
        (tracker.background, tracker.background_tree, tracker.background_surface,
         tracker.background_cells, tracker.tracks[0])))
    assert tracker.last_stamp == 10.
    assert tracker.process(cloud, 10.1)[0][0]['id'] == previous[-1].id
    tracker.set_background(background+[10., 0., 0.])
    assert tracker.tracks == [] and tracker.last_stamp is None
    assert tracker.process(cloud, 10.2)[0][0]['id'] > previous[-1].id


def test_low_speed_noise_and_real_slow_motion():
    rng = np.random.default_rng(1007)
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    shape = rng.uniform([-.4, -.4, -.2], [.4, .4, .2], (50, 3))
    adaptive = Tracker()
    original_noise = Tracker(Config(low_speed_threshold=0.0))
    for tracker in (adaptive, original_noise):
        tracker.set_background(background)
    speeds = [[], []]
    for index in range(100):
        # Common center perturbation models changing visible surface, not true motion.
        points = [35., 0., 8.] + shape + rng.normal(0, .12, 3)
        for slot, tracker in enumerate((adaptive, original_noise)):
            tracks, _ = tracker.process(points, 1. + index * .1)
            if index >= 20:
                speeds[slot].append(np.linalg.norm(tracks[0]['velocity']))
    assert np.mean(speeds[0]) < .65 * np.mean(speeds[1])
    adaptive.reset_tracks()
    for index in range(60):
        tracks, _ = adaptive.process([35. + .05 * index, 0., 8.] + shape, 20. + index * .1)
    # Low-speed covariance tuning must not clamp a genuinely moving target to zero.
    assert np.linalg.norm(np.array(tracks[0]['velocity']) - [.5, 0., 0.]) < .01


def test_two_hit_transients_never_publish_coasting_ghosts():
    tracker = Tracker()
    assert tracker.config.min_hits == 3
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    shape = np.array([[0., 0., 0.], [.2, 0., 0.], [.4, .2, 0.], [0., .2, .2]])
    persistent = [20., 0., 5.] + shape
    transient = [30., 5., 0.] + shape
    intermittent = [40., 5., 0.] + shape
    tracker.set_background(background)
    for index in range(20):
        parts = [persistent]
        if index < 2:
            parts.append(transient)
        if index in (0, 19):
            parts.append(intermittent)
        points = np.vstack(parts)
        tracks, _ = tracker.process(points, 1. + index * .1)
        assert not tracks if index < 2 else len(tracks) == 1
        assert all(np.linalg.norm(np.asarray(t['position']) - [20., 0., 5.]) < 1 for t in tracks)
    tracks, _ = tracker.process(background, 3.)
    assert len(tracks) == 1 and not tracks[0]['observed']


def test_confirmation_window_and_measurement_replay_contract():
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    shape = np.array([[20., 0., 4.], [20.2, 0., 4.], [20.4, .2, 4.]])
    tracker = Tracker()
    tracker.set_background(background)
    for frame in range(10, 59):
        stamp = frame / 10
        cloud = np.vstack((background, shape)) if frame in (10, 34, 58) else background
        assert tracker.process(cloud, stamp)[0] == []
    tracker.process(np.vstack((background, shape)), 5.9)
    tracks, _ = tracker.process(np.vstack((background, shape)), 6.)
    assert len(tracks) == 1 and tracks[0]['confirmed']
    row = tracks[0]
    assert row['hits'] == 3 and row['birth_stamp'] == 5.8
    assert row['measurement_stamp'] == row['state_stamp'] == 6.
    assert np.allclose(row['measurement_position'], (shape.min(axis=0) + shape.max(axis=0))/2)
    assert np.array_equal(row['measurement_covariance'], tracker.tracks[0].measurement_covariance)
    assert np.linalg.eigvalsh(row['measurement_covariance']).min() >= tracker.config.measurement_min_std**2
    assert abs(row['prediction_start_stamp']-5.9) < 1e-12
    assert row['process_acceleration_std'] == tracker.config.low_speed_acceleration_std
    coast = tracker.process(background, 6.1)[0][0]
    assert coast['confirmed'] and not coast['observed']
    assert coast['measurement_position'] is coast['measurement_stamp'] is coast['measurement_covariance'] is None
    assert coast['nearest_distance'] is None


def test_background_interior_shortcut_preserves_exact_boundary():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(61)
    background = rng.uniform(-4, 4, (2000, 3))
    tracker = Tracker(Config(voxel_size=0))
    tracker.set_background(background, already_cleaned=True)
    points = np.vstack((background, background+rng.uniform(-.3, .3, background.shape),
                        rng.uniform(-6, 6, (3000, 3))))
    expected = cKDTree(background).query(points)[0] > tracker.config.background_distance
    assert np.array_equal(tracker.foreground_mask(points), expected)
    assert tracker.background_query_points < len(points)-len(background)
    isolated = np.array([[3., 0., 0.], [6., 0., 0.], [9., 0., 0.]])
    tracker.set_background(isolated, already_cleaned=True)
    probes = np.array([[3., 0., .4], [3., 0., np.nextafter(.4, np.inf)], [3., 0., .401]])
    assert tracker.foreground_mask(probes).tolist() == [False, True, True]


def test_sparse_cluster_uncertainty_and_confident_outlier_gate():
    cfg = Config()
    near = cluster_covariance([1., 2., .5], 16, [10., 0., 0.], np.zeros(3), cfg)
    far = cluster_covariance([1., 2., .5], 16, [60., 0., 0.], np.zeros(3), cfg)
    sparse = cluster_covariance([1., 2., .5], 2, [60., 0., 0.], np.zeros(3), cfg)
    assert np.linalg.eigvalsh(near).min() >= cfg.measurement_min_std**2
    assert far[0, 0] == near[0, 0] and far[1, 1] > near[1, 1]
    assert np.all(np.diag(sparse) > np.diag(far))
    assert sparse[1, 1] > sparse[0, 0] > sparse[2, 2]
    # Large/uncertain clusters cannot bypass the independent physical distance cap.
    tracker = Tracker()
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    shape = np.array([[0., 0., 0.], [.2, 0., 0.], [.4, .2, 0.], [0., .2, .2]])
    tracker.set_background(background)
    for index in range(20):
        tracks, _ = tracker.process([25., 0., 8.] + shape, 1. + .1 * index)
    identity = tracks[0]['id']
    tracks, _ = tracker.process([27.4, 0., 8.] + shape, 3.)
    assert len(tracks) == 1 and tracks[0]['id'] == identity and not tracks[0]['observed']
    assert np.linalg.norm(np.asarray(tracks[0]['position']) - [25.2, .1, 8.1]) < .01


def test_weak_returns_recover_confirmed_track_without_birth_or_promotion():
    tracker = Tracker(Config(voxel_size=.05, sparse_temporal_enabled=False))
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    strong = np.array([[20., -.2, 8.], [20., .2, 8.], [19.8, 0., 8.], [20.2, 0., 8.]])
    weak = strong[:2]
    tracker.set_background(background)
    # Many weak returns over time still cannot create a new ID.
    for index in range(5):
        assert tracker.process(weak, .1*index)[0] == []
        assert tracker.tracks == [] and tracker.last_metrics['weak_clusters'] == 1
    # A tentative track cannot be promoted with weak support either.
    assert tracker.process(strong, 1.)[0] == []
    assert tracker.process(weak, 1.1)[0] == [] and tracker.tracks[0].hits == 1
    tracker.process(strong, 1.2)
    tracks, _ = tracker.process(strong, 1.3)
    identity, extent = tracks[0]['id'], tracks[0]['size']
    for index in range(1, 11):
        tracks, _ = tracker.process(np.vstack((weak, weak+[10., 0., 0.])), 1.3 + .1*index)
        assert len(tracks) == 1 and tracks[0]['id'] == identity and tracks[0]['observed']
        assert tracks[0]['measurement_strength'] == 'weak' and tracks[0]['point_count'] == 2
        assert tracks[0]['size'] == extent
        assert len(tracker.tracks) == 1
    tracks, _ = tracker.process(weak, 2.4)
    assert not tracks[0]['observed'] and tracks[0]['measurement_covariance'] is None
    assert tracks[0]['nearest_distance'] is None
    assert tracker.last_detections[0]['strong'] is False
    json.dumps(tracker.last_detections, allow_nan=False)


def test_sparse_temporal_confirmation_motion_stationary_and_coast():
    cfg = Config(sparse_temporal_enabled=True)
    for velocity in (np.zeros(3), np.array([5., 2., 1.])):
        tracker = Tracker(cfg)
        tracker.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
        identity = None
        for index in range(16):
            stamp = 1.+index*.2
            point = [30., 0., 8.]+velocity*(stamp-1.)
            tracks, _ = tracker.process(np.array([point]), stamp)
            if index < cfg.sparse_min_hits-1:
                assert tracks == []
                continue
            assert len(tracks) == 1 and tracks[0]['observed']
            if identity is None:
                identity = tracks[0]['id']
            assert tracks[0]['id'] == identity and tracks[0]['measurement_strength'] == 'sparse'
            assert np.linalg.eigvalsh(tracks[0]['measurement_covariance']).min() >= cfg.sparse_measurement_std**2-1e-12
            assert tracker.tracks[0].last_strong_seen is None
        assert np.linalg.norm(np.asarray(tracks[0]['velocity'])-velocity) < .05
        coast, _ = tracker.process(np.empty((0, 3)), stamp+.1)
        assert coast[0]['id'] == identity and not coast[0]['observed']
        assert coast[0]['measurement_position'] is coast[0]['nearest_point'] is None


def test_sparse_noise_empty_and_impossible_jump_do_not_confirm():
    rng = np.random.default_rng(159)
    tracker = Tracker(Config(sparse_temporal_enabled=True))
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    tracker.set_background(background)
    for index in range(150):
        noise = rng.uniform([-50., -50., 5.], [50., 50., 30.], (15, 3))
        tracks, _ = tracker.process(noise, 1.+index*.1)
        assert not tracks
        assert len(tracker.tracks) <= tracker.config.sparse_max_candidates
    tracker.reset_tracks()
    for index in range(20):
        assert tracker.process(background, 1.+index*.1)[0] == []
    # Repeated teleportation cannot pass the speed/linear-fit test even if all
    # points lie in the old broad association distance gate.
    history = [(i*.1, np.array([20.+(-1)**i*1.8, 0., 8.])) for i in range(4)]
    assert not tracker._sparse_confirmed(history)


def test_sparse_crossing_uses_observed_3d_separation():
    tracker = Tracker(Config(sparse_temporal_enabled=True))
    tracker.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
    identities = None
    for index in range(50):
        stamp = 1.+index*.1
        # Resolvable crossing, not two indistinguishable points at one location.
        points = np.array([[30.+(index-25)*.2, 0., 8.], [30.-(index-25)*.2, 0., 11.]])
        tracks, _ = tracker.process(points, stamp)
        if index < 4:
            continue
        observed = sorted((t for t in tracks if t['observed']), key=lambda t: t['position'][2])
        assert len(observed) == 2
        if identities is None:
            identities = [t['id'] for t in observed]
        assert [t['id'] for t in observed] == identities


def test_sparse_surface_returns_do_not_create_duplicate_id():
    tracker = Tracker(Config(sparse_temporal_enabled=True))
    tracker.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
    first = np.array([[30., 0., 8.]])
    for index in range(20):
        cloud = first if index < 5 else np.vstack((first, first+[0., 1., 0.]))
        tracks, _ = tracker.process(cloud, 1.+index*.1)
        if index >= 4:
            assert len(tracks) == 1 and tracks[0]['observed']
    assert len(tracker.tracks) == 1


def test_sparse_stop_restart_preserves_identity_and_real_observations():
    for speed in (5., 10.):
        tracker = Tracker()
        tracker.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
        identity = None
        for index in range(85):
            stamp = index * .1
            motion_time = min(stamp, 2.84) if stamp < 5.84 else stamp - 3.
            # One physical return per 400ms, with real empty scans between.
            # Stop shortly after a return, then restart between acquisitions.
            point = np.array([[5.+speed*motion_time, 0., 8.]])
            cloud = point if index % 4 == 0 else np.empty((0, 3))
            tracks, _ = tracker.process(cloud, stamp)
            if index < 12:
                assert tracks == []
                continue
            assert len(tracks) == 1
            row = tracks[0]
            identity = row['id'] if identity is None else identity
            assert row['id'] == identity
            assert row['observed'] == (index % 4 == 0)
            if not row['observed']:
                assert row['measurement_position'] is row['measurement_covariance'] is None
            if index == 56:
                assert row['speed_mps'] < .1
        assert abs(row['speed_mps']-speed) < .1


def test_sparse_crossing_with_gaps_and_outlier_returns():
    rng = np.random.default_rng(226)
    tracker = Tracker()
    tracker.set_background(np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]]))
    identities = {}
    for index in range(100):
        stamp = index * .1
        slot = (index % 4) // 2
        center = np.array([30.+(stamp-5.)*2.*(1 if slot == 0 else -1), 0., 8.+3.*slot])
        cloud = center[None, :] if index % 2 == 0 else np.empty((0, 3))
        # Independent changing clutter is physically nearby but inconsistent
        # with the two established 3D motions; it must not steal either ID.
        noise = rng.uniform([24., 6., 5.], [36., 18., 18.], (2, 3))
        tracks, _ = tracker.process(np.vstack((cloud, noise)), stamp)
        observed = [track for track in tracks if track['observed']]
        if index < 16:
            continue
        assert len(tracks) == 2
        assert len(observed) == (1 if index % 2 == 0 else 0)
        if observed:
            row = observed[0]
            identities.setdefault(slot, row['id'])
            assert row['id'] == identities[slot]
            assert np.linalg.norm(np.asarray(row['position'])-center) < .1


def test_uncertain_slow_estimate_accepts_real_restart_without_new_identity():
    tracker=Tracker()
    tracker.set_background(np.array([[3.,0.,-2.],[4.,0.,-2.],[5.,0.,-2.]]))
    # A sparse confirmed estimate reports low speed but retains substantial
    # velocity uncertainty. A 6m/s restart between real returns is below the
    # configured physical speed limit; it must not be mistaken for a new object.
    position=np.array([30.,0.,8.])
    track=Track(1,np.r_[position,[.2,0.,0.]],np.diag([.04]*3+[.5]*3),np.ones(3)*.1,
                born=.7,last_seen=1.,is_confirmed=True,observed=True,sparse_initiated=True,
                measurement_strength='sparse',
                nearest_point=position.copy(),point_count=1,measurement_position=position.copy(),
                sparse_history=deque((stamp,position.copy()) for stamp in [.7,.8,.9,1.]))
    tracker.tracks=[track]
    tracker.next_id=2
    tracker.last_stamp=1.
    tracks,_=tracker.process(np.array([[32.4,0.,8.]]),1.4)
    assert len(tracks)==1 and tracks[0]['id']==1
    assert tracks[0]['observed'] and tracks[0]['measurement_stamp']==1.4
    assert tracks[0]['measurement_position']==[32.4,0.,8.]
    assert len(tracker.tracks)==1
    # The subsequent empty frame remains explicitly unobserved.
    tracks,_=tracker.process(np.empty((0,3)),1.5)
    assert tracks[0]['id']==1 and not tracks[0]['observed']
    assert tracks[0]['measurement_position'] is None


def test_missing_track_does_not_recover_unrelated_singleton():
    background = np.array([[3., 0., -2.], [4., 0., -2.], [5., 0., -2.]])
    for missing, offset in [(2.4, 14.), (2.7, 10.)]:
        tracker = Tracker()
        tracker.set_background(background)
        for index in range(20):
            tracks, _ = tracker.process(np.array([[30., 0., 8.]]), index * .1)
        identity = tracks[0]['id']
        stamp = round(1.9 + missing, 6)
        for now in np.arange(2., stamp - .05, .1):
            tracker.process(np.empty((0, 3)), float(now))
        tracks, _ = tracker.process(np.array([[30. + offset, 0., 8.]]), stamp)
        # Neither broad spatial recovery nor extra hidden retention may turn a
        # single unrelated measurement into an observed old identity.
        assert not any(track['id'] == identity and track['observed'] for track in tracks)
        assert not any(track['observed'] for track in tracks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    test_guards_and_surface_range()
    test_low_speed_noise_and_real_slow_motion()
    print(json.dumps(test_recorded_scenarios(args.output), indent=2))
