"""Deterministic pan/tilt, lever arm, timestamps and map persistence regressions.

PYTHONPATH=src/jol_perception python3 src/jol_perception/test/test_motion_compensation.py
These mathematical checks do not replace a real sensor timing/extrinsic calibration.
"""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

from jol_perception.motion_compensation import register_cloud, self_filter_points, transform_points
from jol_perception.tracking_core import Config, Tracker, voxel_filter


def pose_at(t):
    rotation = Rotation.from_euler('ZY', [0.8 * t, 0.3 * np.sin(t)])
    # Both translated base and sensor lever arm are intentional.
    translation = np.array([3.0 + 0.4 * t, -4.0, 0.7]) + rotation.apply([0., 0., 1.347])
    return np.r_[translation, rotation.as_quat()]


def sensor_points(world, pose):
    return Rotation.from_quat(pose[3:]).inv().apply(world - pose[:3])


def test_pan_tilt_world_and_velocity():
    rng = np.random.default_rng(102)
    background = np.column_stack((rng.uniform(-30, 30, 600), rng.uniform(-30, 30, 600), np.zeros(600)))
    shape = rng.uniform([-.7, -.6, -.2], [.7, .6, .2], (90, 3))
    tracker = Tracker(Config(voxel_size=.04))
    tracker.set_background(background, already_cleaned=True)
    ids = set()
    max_error = 0.
    for stamp in np.arange(1., 4., .1):
        center = np.array([35., 10. * (stamp - 1), 8.])
        world = np.vstack((background, center + shape))
        raw = sensor_points(world, pose_at(stamp))
        registered, pose, deskewed = register_cloud(raw, stamp, pose_at)
        error = float(np.max(np.abs(registered - world)))
        max_error = max(error, max_error)
        assert error < 1e-12 and not deskewed
        tracks, _ = tracker.process(registered, stamp, pose[:3])
        if tracks:
            assert len(tracks) == 1
            ids.add(tracks[0]['id'])
            if stamp > 2:
                assert np.linalg.norm(np.array(tracks[0]['velocity']) - [0., 10., 0.]) < .03
            nearest = min(np.linalg.norm(center + shape - pose[:3], axis=1))
            assert abs(tracks[0]['nearest_distance'] - nearest) < 1e-10
    assert len(ids) == 1
    # Empty scene must stay empty while the same sensor changes pose.
    tracker.reset_tracks()
    for stamp in np.arange(4., 6., .1):
        world, pose, _ = register_cloud(sensor_points(background, pose_at(stamp)), stamp, pose_at)
        assert tracker.process(world, stamp, pose[:3])[0] == []
    return {'max_registration_error_m': max_error, 'moving_object_id_count': len(ids), 'target_world_speed_mps': 10.0}


def test_range_and_multi_pose_map():
    tracker = Tracker(Config(voxel_size=.01, min_hits=1, noise_min_neighbors=1, cluster_min_points=1))
    # A map can accumulate points outside the current view or 75m of world zero.
    map_points = np.array([[150., 0, 0], [185., 0, 0], [230., 0, 0]])
    tracker.set_background(map_points, already_cleaned=True)
    assert len(tracker.background) == 3
    cloud = np.array([[150., 0, 5], [150.2, 0, 5], [151., 0, 5]])
    origin = np.array([100., 0, 5])
    tracks, _ = tracker.process(cloud, 1., origin)
    assert tracks[0]['nearest_distance'] == 50.0
    assert tracks[0]['nearest_point'] == [150., 0., 5.]
    selected = voxel_filter(np.array([[1.1, 0, 0], [1.2, 0, 0]]), 1., np.array([5., 0, 0]))
    assert selected.tolist() == [[1.2, 0., 0.]]
    assert self_filter_points(np.array([[0., 0., 2.], [50., 0., 8.]])).tolist() == [[50., 0., 8.]]
    with tempfile.TemporaryDirectory() as directory:
        filename = str(Path(directory) / 'map.npz')
        tracker.save_background(filename, 'world', pose_at(1), map_id='test_map')
        tracker.load_background(filename, 'world', pose_at(9), map_id='test_map')
        assert len(tracker.background) == 3
        for args in ({'frame_id': 'jol/lidar', 'map_id': 'test_map'},
                     {'frame_id': 'world', 'map_id': 'another_map'},
                     {'frame_id': 'world', 'map_id': 'test_map', 'stabilized': False}):
            try:
                tracker.load_background(filename, **args)
            except ValueError:
                pass
            else:
                raise AssertionError('Incompatible map accepted')
        np.savez_compressed(filename, points=map_points, frame_id='world', sensor_pose=pose_at(1), version=1)
        try:
            tracker.load_background(filename, 'world', pose_at(1))
        except ValueError:
            pass
        else:
            raise AssertionError('Legacy map accepted as fixed-frame map')


def test_deskew_and_time_guards():
    rng = np.random.default_rng(42)
    world = rng.uniform([30., -20., 2.], [50., 20., 10.], (500, 3))
    offsets = np.linspace(-.1, 0., len(world))
    stamp = 2.
    raw = np.array([sensor_points(p[None], pose_at(stamp + offset))[0] for p, offset in zip(world, offsets)])
    corrected, pose, used = register_cloud(raw, stamp, pose_at, offsets, bin_seconds=.002)
    error = float(np.max(np.linalg.norm(corrected - world, axis=1)))
    assert used and error < 1e-5, error
    rigid, _, used = register_cloud(raw, stamp, pose_at)
    assert not used and np.max(np.linalg.norm(rigid - world, axis=1)) > 2
    nanoseconds = (stamp + offsets) * 1e9
    absolute, _, _ = register_cloud(raw, stamp, pose_at, nanoseconds,
                                     time_scale=1e-9, time_reference='absolute', bin_seconds=.002)
    assert np.allclose(corrected, absolute)
    for bad_stamp, bad_times in [(0., None), (np.nan, None), (stamp, np.full(len(raw), .8)),
                                  (stamp, np.full(len(raw), np.nan))]:
        try:
            register_cloud(raw, bad_stamp, pose_at, bad_times)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid acquisition timestamp accepted')
    calls = []
    def missing_tf(t):
        calls.append(t)
        raise LookupError('TF not yet available')
    try:
        register_cloud(raw, stamp, missing_tf)
    except LookupError:
        pass
    else:
        raise AssertionError('Missing TF silently accepted')
    assert calls == [stamp]  # No zero/latest retry.
    return {'deskew_max_error_m': error, 'whole_scan_max_error_m': float(np.max(np.linalg.norm(rigid - world, axis=1)))}


def test_fixed_deskew_keeps_acquisition_tf_coverage_without_per_point_slerp():
    from unittest.mock import patch
    points = np.array([[2., 1., .5], [3., -1., 2.], [4., 0., 3.]])
    pose = pose_at(1.)
    calls = []
    def fixed_pose(stamp):
        calls.append(stamp)
        return pose.copy()
    with patch('jol_perception.motion_compensation.Slerp', side_effect=AssertionError('fixed poses need no slerp')):
        world, header_pose, deskewed = register_cloud(points, 2., fixed_pose, [-.1, -.05, 0.])
    np.testing.assert_array_equal(world, transform_points(points, pose))
    np.testing.assert_array_equal(header_pose, pose)
    assert deskewed and len(calls) >= 11 and min(calls) == 1.9 and max(calls) == 2.
    def missing_middle(stamp):
        if 1.94 < stamp < 1.96:
            raise LookupError('missing acquisition TF')
        return pose
    try:
        register_cloud(points, 2., missing_middle, [-.1, -.05, 0.])
    except LookupError:
        pass
    else:
        raise AssertionError('A fixed mount still requires intermediate TF coverage')


def test_ros_queue_without_spinning():
    # Available when run in a sourced ROS environment. The actual node's queue
    # methods are exercised without a ROS graph or asynchronous scheduling.
    from jol_perception.tracker_node import TrackerNode, TransformException
    processed, dropped = [], []
    fake = SimpleNamespace(pending=deque([('cloud', time.monotonic())]), tf_wait_seconds=.2)
    def not_yet(msg, received):
        raise TransformException('future timestamp')
    fake._process_cloud = not_yet
    fake._drop = lambda msg, reason, error: dropped.append((msg, reason))
    TrackerNode._drain_pending(fake)
    assert len(fake.pending) == 1 and not dropped
    fake._process_cloud = lambda msg, received: processed.append(msg)
    TrackerNode._drain_pending(fake)
    assert not fake.pending and processed == ['cloud']
    fake.pending.append(('expired', time.monotonic() - 1))
    fake._process_cloud = not_yet
    TrackerNode._drain_pending(fake)
    assert not fake.pending and dropped == [('expired', 'tf_unavailable_drops')]


if __name__ == '__main__':
    import json
    result = test_pan_tilt_world_and_velocity()
    test_range_and_multi_pose_map()
    result.update(test_deskew_and_time_guards())
    try:
        import rclpy
    except ImportError:
        result['ros_queue_test'] = 'not run: source ROS setup first'
    else:
        test_ros_queue_without_spinning()
        result['ros_queue_test'] = 'passed'
    print(json.dumps(result, indent=2))
