import math
import threading
import time

import numpy as np
import pytest
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked

from autonomy_planner.gz_sensor_adapter import GazeboSensorAdapter


def _packed_cloud(x, y, z):
    x = np.asarray(x, dtype="<f4")
    y = np.asarray(y, dtype="<f4")
    z = np.asarray(z, dtype="<f4")
    assert x.shape == y.shape == z.shape
    height, width = x.shape
    packed = np.empty(
        x.shape,
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]),
    )
    packed["x"], packed["y"], packed["z"] = x, y, z
    message = PointCloudPacked()
    message.height = height
    message.width = width
    message.point_step = packed.dtype.itemsize
    message.row_step = width * packed.dtype.itemsize
    message.is_bigendian = False
    message.is_dense = False
    message.data = packed.tobytes()
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        field = message.field.add()
        field.name = name
        field.offset = offset
        field.datatype = PointCloudPacked.Field.FLOAT32
        field.count = 1
    return message


def _bare_adapter(stride=2):
    adapter = GazeboSensorAdapter.__new__(GazeboSensorAdapter)
    adapter._lock = threading.Lock()
    adapter._depth = None
    adapter._downward = None
    adapter._depth_roi_half_width_m = 1.8
    adapter._depth_roi_half_height_m = 1.5
    adapter._near_count_distance_m = 12.0
    adapter._depth_point_stride = stride
    return adapter


def test_xyz_stride_retains_uniform_organized_cloud_coverage():
    rows, columns = 6, 8
    x = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
    message = _packed_cloud(x, x + 100.0, x + 200.0)
    arrays = GazeboSensorAdapter._xyz_arrays(message, sample_stride=2)
    assert arrays is not None
    sampled_x, sampled_y, sampled_z = arrays
    assert sampled_x.dtype == np.dtype("<f4")
    assert sampled_x.shape == (3, 4)
    assert np.array_equal(sampled_x, x[::2, ::2])
    assert np.array_equal(sampled_y, (x + 100.0)[::2, ::2])
    assert np.array_equal(sampled_z, (x + 200.0)[::2, ::2])


def test_stride_two_does_not_miss_minimum_central_obstacle_and_keeps_percentile():
    rows, columns = 12, 16
    forward = np.full((rows, columns), 10.0, dtype=np.float32)
    lateral = np.tile(np.linspace(-2.5, 2.5, columns, dtype=np.float32),
                      (rows, 1))
    vertical = np.tile(np.linspace(-1.2, 1.2, rows, dtype=np.float32)[:, None],
                       (1, columns))
    # A 4x4-pixel obstacle remains represented after the 2x2 spatial stride.
    forward[4:8, 6:10] = 2.0
    adapter = _bare_adapter(stride=2)
    adapter._on_depth_cloud(_packed_cloud(forward, lateral, vertical))
    observation = adapter.depth()
    assert observation is not None
    assert observation.forward_axis == "x"
    assert observation.nearest_m == pytest.approx(2.0)
    assert observation.valid_points == (rows // 2) * (columns // 2)
    assert observation.left_near_points > 0
    assert observation.right_near_points > 0
    assert time.monotonic() - observation.stamp_monotonic < 0.1


def test_optical_z_forward_cloud_axis_detection_and_nan_filtering():
    rows, columns = 8, 10
    optical_x = np.tile(np.linspace(-1.0, 1.0, columns, dtype=np.float32),
                        (rows, 1))
    optical_y = np.zeros((rows, columns), dtype=np.float32)
    optical_z = np.full((rows, columns), 6.0, dtype=np.float32)
    optical_z[2:6, 4:8] = 3.0
    optical_z[0, 0] = math.nan
    adapter = _bare_adapter(stride=2)
    adapter._on_depth_cloud(_packed_cloud(optical_x, optical_y, optical_z))
    observation = adapter.depth()
    assert observation is not None
    assert observation.forward_axis == "z"
    assert observation.nearest_m == pytest.approx(3.0)
    assert observation.valid_points == (rows // 2) * (columns // 2) - 1


def test_down_lidar_filters_invalid_ranges_and_stamps_fresh_observation():
    adapter = _bare_adapter()
    scan = LaserScan()
    scan.range_min = 0.1
    scan.range_max = 100.0
    scan.ranges.extend([math.nan, math.inf, 0.05, 8.0, 4.0, 101.0])
    adapter._on_downward_scan(scan)
    observation = adapter.downward()
    assert observation is not None
    assert observation.range_m == pytest.approx(4.0)
    assert time.monotonic() - observation.stamp_monotonic < 0.1


def test_default_native_topic_contract_is_preserved_for_mission_adapter():
    defaults = GazeboSensorAdapter.__init__.__defaults__
    assert defaults is not None
    assert "/front_depth/image/points" in defaults
