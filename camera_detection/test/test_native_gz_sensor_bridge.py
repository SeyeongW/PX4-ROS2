import struct

import pytest
from gz.msgs10.camera_info_pb2 import CameraInfo as GzCameraInfo
from gz.msgs10.image_pb2 import Image as GzImage
import gz.msgs10.image_pb2 as gz_image_types
from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked as GzPointCloud
from sensor_msgs.msg import PointField

from camera_detection.native_gz_sensor_bridge import (
    convert_camera_info,
    convert_image,
    convert_laser_scan,
    convert_point_cloud,
)


def _stamp_and_frame(message, frame="sensor_optical_frame"):
    message.header.stamp.sec = 12
    message.header.stamp.nsec = 345_000_000
    datum = message.header.data.add()
    datum.key = "frame_id"
    datum.value.append(frame)


def test_rgb_and_depth_image_conversion_preserves_stamp_frame_and_payload():
    rgb = GzImage(width=2, height=1, step=6)
    _stamp_and_frame(rgb, "front_camera_optical_frame")
    rgb.pixel_format_type = gz_image_types.RGB_INT8
    rgb.data = bytes((1, 2, 3, 4, 5, 6))
    output = convert_image(rgb)
    assert (output.header.stamp.sec, output.header.stamp.nanosec) == (
        12, 345_000_000
    )
    assert output.header.frame_id == "front_camera_optical_frame"
    assert output.encoding == "rgb8"
    assert output.step == 6
    assert bytes(output.data) == rgb.data

    depth = GzImage(width=2, height=1, step=8)
    _stamp_and_frame(depth, "front_depth_optical_frame")
    depth.pixel_format_type = gz_image_types.R_FLOAT32
    depth.data = struct.pack("<2f", 1.25, 4.5)
    depth_output = convert_image(depth)
    assert depth_output.encoding == "32FC1"
    assert depth_output.header.frame_id == "front_depth_optical_frame"
    assert struct.unpack("<2f", bytes(depth_output.data)) == pytest.approx(
        (1.25, 4.5)
    )


def test_camera_info_conversion_preserves_calibration_and_optical_frame():
    message = GzCameraInfo(width=640, height=480)
    _stamp_and_frame(message, "down_camera_optical_frame")
    message.distortion.model = 0
    message.distortion.k.extend([0.1, -0.2, 0.01, 0.02, 0.0])
    message.intrinsics.k.extend([
        400.0, 0.0, 320.0,
        0.0, 401.0, 240.0,
        0.0, 0.0, 1.0,
    ])
    message.rectification_matrix.extend([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ])
    message.projection.p.extend([
        400.0, 0.0, 320.0, 0.0,
        0.0, 401.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ])
    output = convert_camera_info(message)
    assert output.header.frame_id == "down_camera_optical_frame"
    assert output.distortion_model == "plumb_bob"
    assert output.d == pytest.approx(message.distortion.k)
    assert output.k[0] == pytest.approx(400.0)
    assert output.k[4] == pytest.approx(401.0)
    assert output.p[2] == pytest.approx(320.0)


def test_point_cloud_conversion_maps_gz_datatypes_to_ros_values():
    message = GzPointCloud(height=1, width=1, point_step=12, row_step=12)
    _stamp_and_frame(message, "front_depth_optical_frame")
    message.data = struct.pack("<3f", 1.0, 2.0, 3.0)
    message.is_dense = True
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        field = message.field.add()
        field.name = name
        field.offset = offset
        field.datatype = GzPointCloud.Field.FLOAT32
        field.count = 1
    output = convert_point_cloud(message)
    assert output.header.frame_id == "front_depth_optical_frame"
    assert [field.datatype for field in output.fields] == [
        PointField.FLOAT32, PointField.FLOAT32, PointField.FLOAT32
    ]
    assert output.point_step == 12
    assert bytes(output.data) == message.data


def test_lidar_conversion_preserves_frame_ranges_and_nominal_scan_period():
    message = GzLaserScan(
        frame="lidar_sensor_link",
        angle_min=-0.5,
        angle_max=0.5,
        angle_step=0.25,
        range_min=0.2,
        range_max=50.0,
    )
    _stamp_and_frame(message, "lidar_sensor_link")
    message.ranges.extend([2.0, 3.0, 4.0])
    message.intensities.extend([0.1, 0.2, 0.3])
    output = convert_laser_scan(message, nominal_rate_hz=50.0)
    assert output.header.frame_id == "lidar_sensor_link"
    assert output.scan_time == pytest.approx(0.02)
    assert output.angle_increment == pytest.approx(0.25)
    assert output.ranges == pytest.approx([2.0, 3.0, 4.0])
