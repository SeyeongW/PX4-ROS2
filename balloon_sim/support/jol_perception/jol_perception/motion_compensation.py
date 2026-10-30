"""점마다 다른 취득 시각의 센서 자세를 사용해 회전 중 점군 왜곡을 보정한다.

lookup_pose(seconds)는 [x,y,z,qx,qy,qz,qw]를 반환한다. 해당 시각의 TF가 없으면 실패시킨다."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform_points(points, pose):
    points, pose = np.asarray(points, dtype=float), np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError('TF pose must contain a finite translation and nonzero quaternion')
    return Rotation.from_quat(pose[3:]).apply(points) + pose[:3]


def self_filter_points(points, minimum=(-2.0, -2.0, -0.1), maximum=(2.0, 2.0, 3.0)):
    """Exclude the known installation envelope in the fixed tracking frame."""
    low, high = np.asarray(minimum), np.asarray(maximum)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or not np.all(low < high):
        raise ValueError('Self-filter bounds must define a finite fixed-frame box')
    points = np.asarray(points)
    return points[~np.all((points >= low) & (points <= high), axis=1)]


def observer_filter_points(points, stamp, own, scan_seconds=.2):
    """선택 실행의 협력 관찰 기체만 자기 odometry의 swept box로 제거한다."""
    if own is None or abs(stamp-own['stamp']) > .3:
        return points
    p, v = np.asarray(own['position']), np.asarray(own['velocity'])
    if not np.isfinite(np.r_[p, v]).all():
        return points
    ends = np.array([p+(stamp-own['stamp']-scan_seconds)*v, p+(stamp-own['stamp'])*v])
    low, high = ends.min(axis=0)-.8, ends.max(axis=0)+.8
    return points[~np.all((points >= low) & (points <= high), axis=1)]


def register_cloud(points, stamp, lookup_pose, point_times=None, time_scale=1.0,
                   time_reference='relative', max_scan_seconds=0.2, bin_seconds=0.01):
    """Return (fixed-frame XYZ, LiDAR pose at stamp, deskewed).

    Without an explicitly configured time field, treat the scan as instantaneous.
    With point times, query exact TF samples then linearly interpolate translation
    and slerp rotation at each acquisition time. ``bin_seconds`` bounds the TF
    sample spacing; acceleration within that spacing remains an approximation.
    Relative offsets are signed seconds from the PointCloud2 header after scaling.
    """
    if not np.isfinite(stamp) or stamp <= 0:
        raise ValueError('Cloud timestamp must be positive; zero would request latest TF')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pose = np.asarray(lookup_pose(float(stamp)), dtype=float)
    if point_times is None or not len(points):
        return transform_points(points, pose), pose, False
    times = np.asarray(point_times, dtype=float).reshape(-1) * time_scale
    if len(times) != len(points) or not np.isfinite(times).all():
        raise ValueError('Point time field must contain one finite value per point')
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError('point_time_scale must be finite and positive')
    if time_reference == 'relative':
        times += stamp
    elif time_reference != 'absolute':
        raise ValueError('point_time_reference must be relative or absolute')
    if not np.isfinite(bin_seconds) or bin_seconds <= 0 or not np.isfinite(max_scan_seconds) or max_scan_seconds <= 0:
        raise ValueError('Deskew scan duration and TF sample spacing must be positive')
    if np.any(times <= 0) or np.max(np.abs(times - stamp)) > max_scan_seconds:
        raise ValueError('Point times fall outside the configured acquisition interval')
    start, end = float(times.min()), float(times.max())
    intervals = int(np.ceil((end - start) / bin_seconds))
    if intervals > 1000:
        raise ValueError('Deskew would need more than 1001 TF samples per scan')
    if intervals == 0:
        return transform_points(points, lookup_pose(start)), pose, True
    sample_times = np.linspace(start, end, intervals + 1)
    poses = np.array([lookup_pose(float(t)) for t in sample_times])
    if poses.shape != (len(sample_times), 7) or not np.isfinite(poses).all():
        raise ValueError('Invalid TF pose samples')
    # A fixed mount still needs TF coverage over the acquisition interval,
    # but identical sampled poses need only one rigid transform, not N slerps.
    if np.all(poses == poses[0]):
        return transform_points(points, poses[0]), pose, True
    rotations = Slerp(sample_times, Rotation.from_quat(poses[:, 3:]))(times)
    translation = np.column_stack([np.interp(times, sample_times, poses[:, axis]) for axis in range(3)])
    return rotations.apply(points) + translation, pose, True
