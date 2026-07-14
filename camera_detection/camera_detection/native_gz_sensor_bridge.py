#!/usr/bin/env python3
"""Gazebo Harmonic (transport13 / msgs10) sensor bridge for ROS 2 Humble.

Ubuntu 22.04's binary ``ros_gz_bridge`` is commonly linked against
ignition-transport11 / ignition-msgs8 and therefore discovers, but cannot
receive, gz-sim8 transport13 / msgs10 data.  This narrow bridge copies only
the payload contract used by this repository.  It performs no coordinate
reinterpretation and preserves every Gazebo timestamp and optical frame ID.
"""

from __future__ import annotations

from collections import defaultdict
import math
import os
import signal
import threading
import time
from typing import Callable

import numpy as np
from builtin_interfaces.msg import Time as RosTime
from gz.msgs10.camera_info_pb2 import CameraInfo as GzCameraInfo
from gz.msgs10.clock_pb2 import Clock as GzClock
from gz.msgs10.image_pb2 import Image as GzImage
import gz.msgs10.image_pb2 as gz_image_types
from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked as GzPointCloud
from gz.transport13 import Node as GzNode
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField


PIXEL_ENCODINGS = {
    gz_image_types.L_INT8: "mono8",
    gz_image_types.L_INT16: "mono16",
    gz_image_types.RGB_INT8: "rgb8",
    gz_image_types.RGBA_INT8: "rgba8",
    gz_image_types.BGRA_INT8: "bgra8",
    gz_image_types.BGR_INT8: "bgr8",
    gz_image_types.R_FLOAT16: "16FC1",
    gz_image_types.R_FLOAT32: "32FC1",
}
DISTORTION_MODELS = {
    0: "plumb_bob",
    1: "rational_polynomial",
    2: "equidistant",
}
FRONT_DEPTH_DISPLAY_TOPIC = "/front_depth/image"
FRONT_DEPTH_METRIC_TOPIC = "/front_depth/image_raw"
DOWN_DEPTH_DISPLAY_TOPIC = "/down_depth/image"
DOWN_DEPTH_METRIC_TOPIC = "/down_depth/image_raw"


def _ros_time(gz_time) -> RosTime:
    stamp = RosTime()
    stamp.sec = int(gz_time.sec)
    stamp.nanosec = int(gz_time.nsec)
    return stamp


def _frame_id(header, fallback: str = "") -> str:
    for datum in header.data:
        if datum.key in ("frame_id", "gz_frame_id") and datum.value:
            return str(datum.value[0])
    return fallback


def convert_image(message: GzImage) -> Image:
    output = Image()
    output.header.stamp = _ros_time(message.header.stamp)
    output.header.frame_id = _frame_id(message.header)
    output.width = int(message.width)
    output.height = int(message.height)
    try:
        output.encoding = PIXEL_ENCODINGS[int(message.pixel_format_type)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Gazebo pixel format {message.pixel_format_type}"
        ) from exc
    output.is_bigendian = 0
    output.step = int(message.step)
    output.data = message.data
    return output


def convert_depth_preview(
    message: GzImage,
    near_m: float,
    far_m: float,
) -> Image:
    """Convert metric ``R_FLOAT32`` depth into an rqt-safe mono8 preview.

    The raw ``32FC1`` image remains the source for planning and measurement.
    This display-only image masks NaN / Inf / out-of-range samples and maps
    every valid distance to at least intensity 32, so Humble rqt_image_view
    cannot collapse the frame to black when the raw image contains ``+Inf``.
    Near pixels are bright and far pixels are dark.
    """

    if int(message.pixel_format_type) != gz_image_types.R_FLOAT32:
        raise ValueError(
            "depth preview requires Gazebo R_FLOAT32, got "
            f"{message.pixel_format_type}"
        )
    width = int(message.width)
    height = int(message.height)
    source_step = int(message.step)
    if width <= 0 or height <= 0 or source_step < width * 4 or source_step % 4:
        raise ValueError(
            f"invalid depth image geometry {width}x{height} step={source_step}"
        )
    if len(message.data) < source_step * height:
        raise ValueError(
            f"short depth payload {len(message.data)} < {source_step * height}"
        )
    if not (math.isfinite(near_m) and math.isfinite(far_m) and far_m > near_m):
        raise ValueError(f"invalid depth preview range {near_m}..{far_m}")

    row_floats = source_step // 4
    depth = np.frombuffer(
        message.data,
        dtype="<f4",
        count=row_floats * height,
    ).reshape(height, row_floats)[:, :width]
    valid = np.isfinite(depth) & (depth >= near_m) & (depth <= far_m)
    preview = np.zeros((height, width), dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth[valid], near_m, far_m)
        inverse_range = (far_m - clipped) / (far_m - near_m)
        preview[valid] = np.rint(32.0 + 223.0 * inverse_range).astype(np.uint8)

    output = Image()
    output.header.stamp = _ros_time(message.header.stamp)
    output.header.frame_id = _frame_id(message.header)
    output.width = width
    output.height = height
    output.encoding = "mono8"
    output.is_bigendian = 0
    output.step = width
    output.data = preview.tobytes()
    return output


def convert_camera_info(message: GzCameraInfo) -> CameraInfo:
    output = CameraInfo()
    output.header.stamp = _ros_time(message.header.stamp)
    output.header.frame_id = _frame_id(message.header)
    output.width = int(message.width)
    output.height = int(message.height)
    output.distortion_model = DISTORTION_MODELS.get(
        int(message.distortion.model), "plumb_bob"
    )
    output.d = list(message.distortion.k)
    if len(message.intrinsics.k) == 9:
        output.k = list(message.intrinsics.k)
    if len(message.rectification_matrix) == 9:
        output.r = list(message.rectification_matrix)
    if len(message.projection.p) == 12:
        output.p = list(message.projection.p)
    return output


def convert_point_cloud(message: GzPointCloud) -> PointCloud2:
    output = PointCloud2()
    output.header.stamp = _ros_time(message.header.stamp)
    output.header.frame_id = _frame_id(message.header)
    output.height = int(message.height)
    output.width = int(message.width)
    output.is_bigendian = bool(message.is_bigendian)
    output.point_step = int(message.point_step)
    output.row_step = int(message.row_step)
    output.is_dense = bool(message.is_dense)
    output.data = message.data
    for source in message.field:
        field = PointField()
        field.name = str(source.name)
        field.offset = int(source.offset)
        # Gazebo uses 0..7 while sensor_msgs/PointField uses 1..8.
        field.datatype = int(source.datatype) + 1
        field.count = int(source.count)
        output.fields.append(field)
    return output


def convert_laser_scan(message: GzLaserScan, nominal_rate_hz: float) -> LaserScan:
    output = LaserScan()
    output.header.stamp = _ros_time(message.header.stamp)
    output.header.frame_id = _frame_id(message.header, str(message.frame))
    output.angle_min = float(message.angle_min)
    output.angle_max = float(message.angle_max)
    output.angle_increment = float(message.angle_step)
    output.time_increment = 0.0
    output.scan_time = 1.0 / max(float(nominal_rate_hz), 1e-6)
    output.range_min = float(message.range_min)
    output.range_max = float(message.range_max)
    output.ranges = list(message.ranges)
    output.intensities = list(message.intensities)
    return output


class NativeGazeboSensorBridge(Node):
    """Bridge the repository's fixed sensor topic contract."""

    def __init__(self) -> None:
        super().__init__("native_gz_sensor_bridge")
        world = str(self.declare_parameter("world", "applepark_city").value)
        model = str(
            self.declare_parameter("model", "x500_city_rgbd_lidar_0").value
        )
        self.publish_sensor_data_without_subscribers = bool(
            self.declare_parameter(
                "publish_sensor_data_without_subscribers", False
            ).value
        )
        self.lidar_rate_hz = float(
            self.declare_parameter("lidar_rate_hz", 50.0).value
        )

        defaults = {
            "front_rgb_image_gz_topic": "/front_camera/image",
            "front_rgb_info_gz_topic": "/front_camera/camera_info",
            "front_depth_image_gz_topic": "/front_depth/image",
            "front_depth_points_gz_topic": "/front_depth/image/points",
            "front_depth_info_gz_topic": "/front_depth/camera_info",
            "down_rgb_image_gz_topic": "/down_camera/image",
            "down_rgb_info_gz_topic": "/down_camera/camera_info",
            "down_depth_image_gz_topic": "/down_depth/image",
            "down_depth_points_gz_topic": "/down_depth/image/points",
            "down_depth_info_gz_topic": "/down_depth/camera_info",
            "lidar_gz_topic": (
                f"/world/{world}/model/{model}/link/lidar_sensor_link/"
                "sensor/lidar/scan"
            ),
        }
        topics = {
            name: str(self.declare_parameter(name, value).value)
            for name, value in defaults.items()
        }

        self._sensor_publishers = {
            "front_rgb": self.create_publisher(
                Image, "/front_camera/image", qos_profile_sensor_data
            ),
            "front_rgb_info": self.create_publisher(
                CameraInfo, "/front_camera/camera_info", qos_profile_sensor_data
            ),
            "front_depth": self.create_publisher(
                Image, FRONT_DEPTH_METRIC_TOPIC, qos_profile_sensor_data
            ),
            "front_depth_preview": self.create_publisher(
                Image, FRONT_DEPTH_DISPLAY_TOPIC, qos_profile_sensor_data
            ),
            "front_depth_points": self.create_publisher(
                PointCloud2, "/front_depth/points", qos_profile_sensor_data
            ),
            "front_depth_info": self.create_publisher(
                CameraInfo, "/front_depth/camera_info", qos_profile_sensor_data
            ),
            "down_rgb": self.create_publisher(
                Image, "/down_camera/image", qos_profile_sensor_data
            ),
            "down_rgb_info": self.create_publisher(
                CameraInfo, "/down_camera/camera_info", qos_profile_sensor_data
            ),
            "down_depth": self.create_publisher(
                Image, DOWN_DEPTH_METRIC_TOPIC, qos_profile_sensor_data
            ),
            "down_depth_preview": self.create_publisher(
                Image, DOWN_DEPTH_DISPLAY_TOPIC, qos_profile_sensor_data
            ),
            "down_depth_points": self.create_publisher(
                PointCloud2, "/down_depth/points", qos_profile_sensor_data
            ),
            "down_depth_info": self.create_publisher(
                CameraInfo, "/down_depth/camera_info", qos_profile_sensor_data
            ),
            "lidar": self.create_publisher(
                LaserScan, "/down_lidar", qos_profile_sensor_data
            ),
            "lidar_points": self.create_publisher(
                PointCloud2, "/down_lidar/points", qos_profile_sensor_data
            ),
        }
        clock_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.clock_publisher = self.create_publisher(Clock, "/clock", clock_qos)
        self.gz_node = GzNode()
        self._gz_topics: list[str] = []
        self._demand_subscriptions = {}
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._last_received: dict[str, float] = {}
        self._last_stats_wall_s = time.monotonic()

        self._subscribe(GzClock, "/clock", self._clock_callback, "clock")
        self._subscribe_image(topics["front_rgb_image_gz_topic"], "front_rgb")
        self._subscribe_info(topics["front_rgb_info_gz_topic"], "front_rgb_info")
        self._subscribe_image(
            topics["front_depth_image_gz_topic"],
            "front_depth",
            preview_key="front_depth_preview",
            preview_near_m=0.20,
            preview_far_m=30.0,
        )
        self._subscribe_cloud(
            topics["front_depth_points_gz_topic"], "front_depth_points"
        )
        self._subscribe_info(
            topics["front_depth_info_gz_topic"], "front_depth_info"
        )
        self._subscribe_image(topics["down_rgb_image_gz_topic"], "down_rgb")
        self._subscribe_info(topics["down_rgb_info_gz_topic"], "down_rgb_info")
        self._subscribe_image(
            topics["down_depth_image_gz_topic"],
            "down_depth",
            preview_key="down_depth_preview",
            preview_near_m=0.15,
            preview_far_m=80.0,
        )
        self._subscribe_cloud(
            topics["down_depth_points_gz_topic"], "down_depth_points"
        )
        self._subscribe_info(
            topics["down_depth_info_gz_topic"], "down_depth_info"
        )
        self._register_demand_subscription(
            GzLaserScan,
            topics["lidar_gz_topic"],
            self._lidar_callback,
            "lidar",
            self._sensor_publishers["lidar"],
        )
        self._subscribe_cloud(f"{topics['lidar_gz_topic']}/points", "lidar_points")
        self.create_timer(0.25, self._refresh_demand_subscriptions)
        self.create_timer(5.0, self._report_statistics)
        self.get_logger().info(
            "native Harmonic sensor bridge active "
            f"(world={world}, model={model}, sensor-data-on-demand="
            f"{not self.publish_sensor_data_without_subscribers})"
        )

    def _received(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1
            self._last_received[key] = time.monotonic()

    def _subscribe(
        self,
        message_type,
        topic: str,
        callback: Callable,
        key: str,
    ) -> None:
        if self.gz_node.subscribe(message_type, topic, callback) is False:
            raise RuntimeError(f"failed to subscribe to Gazebo topic {topic}")
        self._gz_topics.append(topic)
        self.get_logger().info(f"gz {topic} -> ROS sensor contract ({key})")

    def _register_demand_subscription(
        self,
        message_type,
        topic: str,
        callback: Callable,
        key: str,
        publisher,
    ) -> None:
        self._demand_subscriptions[topic] = (
            message_type, callback, key, publisher
        )

    def _refresh_demand_subscriptions(self) -> None:
        for topic, specification in self._demand_subscriptions.items():
            message_type, callback, key, publisher = specification
            desired = self._should_publish(publisher)
            active = topic in self._gz_topics
            if desired and not active:
                self._subscribe(message_type, topic, callback, key)
            elif not desired and active:
                if self.gz_node.unsubscribe(topic):
                    self._gz_topics.remove(topic)
                    self.get_logger().info(f"gz {topic} suspended (no ROS subscriber)")

    def stop_transport(self) -> None:
        """Detach transport callbacks before ROS publishers are destroyed."""

        for topic in reversed(self._gz_topics):
            try:
                self.gz_node.unsubscribe(topic)
            except Exception as exc:  # pragma: no cover - binding-specific cleanup
                self.get_logger().warning(f"failed to unsubscribe {topic}: {exc}")
        self._gz_topics.clear()

    def _subscribe_image(
        self,
        topic: str,
        key: str,
        *,
        preview_key: str | None = None,
        preview_near_m: float = 0.0,
        preview_far_m: float = 0.0,
    ) -> None:
        def callback(message: GzImage) -> None:
            try:
                publisher = self._sensor_publishers[key]
                if self._should_publish(publisher):
                    publisher.publish(convert_image(message))
                if preview_key is not None:
                    preview_publisher = self._sensor_publishers[preview_key]
                    if self._should_publish(preview_publisher):
                        preview_publisher.publish(
                            convert_depth_preview(
                                message,
                                preview_near_m,
                                preview_far_m,
                            )
                        )
                self._received(key)
            except ValueError as exc:
                self.get_logger().error(str(exc))

        demand_publishers = [self._sensor_publishers[key]]
        if preview_key is not None:
            demand_publishers.append(self._sensor_publishers[preview_key])
        self._register_demand_subscription(
            GzImage, topic, callback, key, tuple(demand_publishers)
        )

    def _subscribe_info(self, topic: str, key: str) -> None:
        def callback(message: GzCameraInfo) -> None:
            publisher = self._sensor_publishers[key]
            if self._should_publish(publisher):
                publisher.publish(convert_camera_info(message))
            self._received(key)

        self._register_demand_subscription(
            GzCameraInfo, topic, callback, key, self._sensor_publishers[key]
        )

    def _subscribe_cloud(self, topic: str, key: str) -> None:
        def callback(message: GzPointCloud) -> None:
            publisher = self._sensor_publishers[key]
            if self._should_publish(publisher):
                publisher.publish(convert_point_cloud(message))
            self._received(key)

        self._register_demand_subscription(
            GzPointCloud, topic, callback, key, self._sensor_publishers[key]
        )

    def _lidar_callback(self, message: GzLaserScan) -> None:
        publisher = self._sensor_publishers["lidar"]
        if self._should_publish(publisher):
            publisher.publish(convert_laser_scan(message, self.lidar_rate_hz))
        self._received("lidar")

    def _should_publish(self, publisher) -> bool:
        if isinstance(publisher, (tuple, list)):
            return self.publish_sensor_data_without_subscribers or any(
                item.get_subscription_count() > 0 for item in publisher
            )
        return (
            self.publish_sensor_data_without_subscribers
            or publisher.get_subscription_count() > 0
        )

    def _clock_callback(self, message: GzClock) -> None:
        output = Clock()
        output.clock = _ros_time(message.sim)
        self.clock_publisher.publish(output)
        self._received("clock")

    def _report_statistics(self) -> None:
        now = time.monotonic()
        with self._lock:
            elapsed = max(now - self._last_stats_wall_s, 1e-6)
            counts = dict(self._counts)
            ages = {
                key: now - received for key, received in self._last_received.items()
            }
            self._counts.clear()
            self._last_stats_wall_s = now
        keys = ("front_rgb", "front_depth", "down_rgb", "down_depth", "lidar")
        summary = " ".join(
            f"{key}={counts.get(key, 0) / elapsed:.1f}Hz/"
            f"{ages.get(key, math.inf):.3f}s"
            for key in keys
        )
        self.get_logger().info(f"sensor rates/ages: {summary}")


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    # See the teardown note below: do not let rclpy invalidate publishers
    # while gz worker threads are still entering Python callbacks.
    signal.signal(signal.SIGINT, lambda _signum, _frame: os._exit(0))
    signal.signal(signal.SIGTERM, lambda _signum, _frame: os._exit(0))
    node = NativeGazeboSensorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # gz-transport13 owns Python callbacks on native worker threads.  Its
        # Jammy binding can abort while interpreter-managed callbacks are
        # being destroyed (``scoped_acquire::dec_ref``).  This bridge is a
        # dedicated launch child, so exiting after ROS spin is the safe,
        # deterministic teardown path; the OS closes all transport sockets.
        os._exit(0)


if __name__ == "__main__":
    main()
