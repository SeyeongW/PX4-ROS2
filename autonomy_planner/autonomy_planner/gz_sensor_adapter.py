"""Read the mission sensors directly from Gazebo Harmonic Transport.

ROS 2 Humble installations often contain a ros_gz bridge built for an older
Gazebo ABI.  The autonomy node therefore consumes Harmonic's native
``gz.msgs10`` messages.  PX4 control still uses ROS 2 / px4_msgs through the
Micro XRCE-DDS Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.transport13 import Node as GazeboNode


@dataclass(frozen=True)
class DepthObservation:
    """Compact forward-obstacle summary derived from one depth cloud."""

    stamp_monotonic: float
    nearest_m: float
    left_near_points: int
    right_near_points: int
    valid_points: int
    forward_axis: str


@dataclass(frozen=True)
class DownwardRange:
    """Latest valid range along the downward lidar ray."""

    stamp_monotonic: float
    range_m: float


class GazeboSensorAdapter:
    """Thread-safe native Gazebo subscriber for depth and downward lidar."""

    def __init__(
        self,
        world_name: str,
        model_name: str,
        depth_topic: str = "/front_depth/image/points",
        depth_roi_half_width_m: float = 1.8,
        depth_roi_half_height_m: float = 1.5,
        near_count_distance_m: float = 12.0,
        depth_point_stride: int = 2,
    ) -> None:
        self._lock = threading.Lock()
        self._depth: DepthObservation | None = None
        self._downward: DownwardRange | None = None
        self._depth_roi_half_width_m = float(depth_roi_half_width_m)
        self._depth_roi_half_height_m = float(depth_roi_half_height_m)
        self._near_count_distance_m = float(near_count_distance_m)
        self._depth_point_stride = max(1, int(depth_point_stride))
        self.depth_topic = depth_topic
        self.lidar_topic = (
            f"/world/{world_name}/model/{model_name}/link/lidar_sensor_link/"
            "sensor/lidar/scan"
        )

        self._node = GazeboNode()
        depth_ok = self._node.subscribe(
            PointCloudPacked, self.depth_topic, self._on_depth_cloud
        )
        lidar_ok = self._node.subscribe(
            LaserScan, self.lidar_topic, self._on_downward_scan
        )
        if depth_ok is False:
            raise RuntimeError(f"failed to subscribe to {self.depth_topic}")
        if lidar_ok is False:
            raise RuntimeError(f"failed to subscribe to {self.lidar_topic}")

    def depth(self) -> DepthObservation | None:
        with self._lock:
            return self._depth

    def downward(self) -> DownwardRange | None:
        with self._lock:
            return self._downward

    @staticmethod
    def _xyz_arrays(
        message: PointCloudPacked,
        sample_stride: int = 1,
    ) -> tuple[np.ndarray, ...] | None:
        """Expose sampled XYZ float32 views without full-cloud float64 copies.

        Organized clouds are decimated in both image dimensions, which retains
        uniform field-of-view coverage.  A padded row may require a small
        float32 flattening copy, but never the former three float64 full-cloud
        conversions.
        """

        fields = {field.name.lower(): field for field in message.field}
        if not all(name in fields for name in ("x", "y", "z")):
            return None
        if message.point_step <= 0:
            return None
        width = int(message.width)
        height = max(int(message.height), 1)
        point_step = int(message.point_step)
        row_step = int(message.row_step) or width * point_step
        if width <= 0 or row_step < width * point_step:
            return None
        required = (height - 1) * row_step + width * point_step
        if required > len(message.data):
            return None
        stride = max(1, int(sample_stride))
        endian = ">" if message.is_bigendian else "<"
        arrays: list[np.ndarray] = []
        for name in ("x", "y", "z"):
            field = fields[name]
            if field.datatype != PointCloudPacked.Field.FLOAT32:
                return None
            if int(field.offset) + 4 > point_step:
                return None
            organized = np.ndarray(
                shape=(height, width),
                dtype=f"{endian}f4",
                buffer=message.data,
                offset=int(field.offset),
                strides=(row_step, point_step),
            )
            arrays.append(organized[::stride, ::stride])
        return tuple(arrays)

    @staticmethod
    def _choose_forward_axis(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> str:
        # Gazebo versions have emitted both camera-frame (+X forward) and
        # optical-frame (+Z forward) clouds.  Pick the convention whose depth
        # coordinate is most consistently positive and inside the sensor range.
        x_score = float(np.mean((x > 0.15) & (x < 25.0)))
        z_score = float(np.mean((z > 0.15) & (z < 25.0)))
        return "x" if x_score >= z_score else "z"

    def _on_depth_cloud(self, message: PointCloudPacked) -> None:
        xyz = self._xyz_arrays(message, self._depth_point_stride)
        if xyz is None:
            return
        x, y, z = xyz
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(finite):
            # A well-formed all-Inf cloud is a healthy open-sky / no-return
            # measurement, not a stale sensor.  Retain receipt freshness while
            # exposing an infinite nearest range and zero obstacle points.
            observation = DepthObservation(
                stamp_monotonic=time.monotonic(),
                nearest_m=math.inf,
                left_near_points=0,
                right_near_points=0,
                valid_points=0,
                forward_axis="no_return",
            )
            with self._lock:
                self._depth = observation
            return
        axis = self._choose_forward_axis(x, y, z)
        if axis == "x":
            forward = x
            left = y
            vertical = z
        else:
            forward = z
            # Optical coordinates use +X to the image right and +Y down.
            left = -x
            vertical = -y

        valid = finite & (forward > 0.15) & (forward < 25.0)
        central = (
            valid
            & (np.abs(left) <= self._depth_roi_half_width_m)
            & (np.abs(vertical) <= self._depth_roi_half_height_m)
        )
        central_depth = forward[central]
        if central_depth.size:
            # A low percentile is robust to a few invalid near-zero pixels but
            # still reacts to a real object occupying the flight envelope.
            nearest = float(np.percentile(central_depth, 3.0))
        else:
            nearest = math.inf

        near = valid & (forward <= self._near_count_distance_m)
        left_count = int(np.count_nonzero(near & (left >= 0.0)))
        right_count = int(np.count_nonzero(near & (left < 0.0)))
        observation = DepthObservation(
            stamp_monotonic=time.monotonic(),
            nearest_m=nearest,
            left_near_points=left_count,
            right_near_points=right_count,
            valid_points=int(np.count_nonzero(valid)),
            forward_axis=axis,
        )
        with self._lock:
            self._depth = observation

    def _on_downward_scan(self, message: LaserScan) -> None:
        minimum = float(message.range_min)
        maximum = float(message.range_max)
        if not (math.isfinite(minimum) and math.isfinite(maximum) and 0.0 < minimum < maximum):
            return
        finite = [
            float(value)
            for value in message.ranges
            if math.isfinite(value) and minimum <= float(value) <= maximum
        ]
        if finite:
            distance = min(finite)
        elif any(math.isinf(value) and value > 0.0 for value in message.ranges):
            # Positive infinity is the standard healthy no-return value.
            distance = maximum
        else:
            return
        reading = DownwardRange(time.monotonic(), distance)
        with self._lock:
            self._downward = reading
