"""Non-authoritative ROS2 sidecar that writes one run CSV and one summary."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path

import rclpy
from rcl_interfaces.msg import Log
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
    qos_profile_sensor_data,
)
from geometry_msgs.msg import PointStamped
from px4_msgs.msg import TrajectorySetpoint
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from .metrics import ClearanceMap, ExperimentAccumulator, SCHEMA_VERSION


SAMPLE_FIELDS = (
    "schema_version", "run_id", "sample_index", "timestamp_utc",
    "ros_time_s", "receive_ros_time_s", "elapsed_s", "mission_state",
    "vehicle_x_enu_m", "vehicle_y_enu_m", "vehicle_z_enu_m",
    "setpoint_x_enu_m", "setpoint_y_enu_m", "setpoint_z_enu_m",
    "cue_x_enu_m", "cue_y_enu_m", "cue_z_enu_m",
    "path_length_m", "path_length_xy_m", "tracking_error_m",
    "clearance_m", "clearance_residual_m", "relative_xy_error_m",
    "active_plan_seq", "replan_count", "aruco_detected",
    "aruco_detection_rate_pct", "latest_astar_plan_time_ms",
    "latest_sfc_generation_time_ms", "active_sfc_min_width_m",
    "active_sfc_avg_width_m", "active_sfc_corridor_count",
    "sfc_violation", "sfc_violation_count",
)


def _clean(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.15g}"
    return "" if value is None else value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _px4_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=5)


def _planned_path_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1)


class ExperimentLoggerNode(Node):
    def __init__(self):
        super().__init__("experiment_logger")
        default_dir = Path.home() / ".local/state/px4-ros2-jo/experiments"
        output_dir = Path(str(self.declare_parameter(
            "output_dir", str(default_dir)).value)).expanduser().resolve()
        map_yaml = str(self.declare_parameter("map_yaml", "").value).strip()
        requested_run_id = str(self.declare_parameter("run_id", "").value).strip()
        self.setpoint_max_age_s = float(self.declare_parameter(
            "setpoint_max_age_s", 0.5).value)
        if not math.isfinite(self.setpoint_max_age_s) \
                or self.setpoint_max_age_s <= 0.0:
            raise ValueError("setpoint_max_age_s must be finite and positive")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise ValueError(f"output_dir is not a directory: {output_dir}")

        token = _timestamp_token()
        self.run_id = requested_run_id or token
        self.started_utc = _utc_now()
        self.sample_path = output_dir / f"experiment_{token}.csv"
        self.summary_path = output_dir / f"experiment_{token}_summary.csv"
        self._sample_stream = self.sample_path.open(
            "x", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(
            self._sample_stream, fieldnames=SAMPLE_FIELDS)
        self._writer.writeheader()
        self._closed = False

        clearance_map = None
        if map_yaml:
            clearance_map = ClearanceMap.from_yaml(map_yaml)
        self.metrics = ExperimentAccumulator(clearance_map=clearance_map)

        self.create_subscription(
            PointStamped, "/mission/vehicle_position", self._on_position,
            qos_profile_sensor_data)
        self.create_subscription(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint",
            self._on_setpoint, _px4_qos())
        self.create_subscription(
            PointStamped, "/marker/cue", self._on_cue,
            qos_profile_sensor_data)
        self.create_subscription(String, "/mission/state", self._on_state, 10)
        self.create_subscription(
            Bool, "/aruco/detected", self._on_aruco,
            qos_profile_sensor_data)
        self.create_subscription(
            MarkerArray, "/mission/active_plan_markers", self._on_plan,
            _planned_path_qos())
        self.create_subscription(
            Log, "/rosout", self._on_rosout, QoSProfile(depth=1000))
        self.get_logger().info(
            f"read-only metrics CSV: {self.sample_path}; summary: "
            f"{self.summary_path}")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _stamp_s(self, stamp) -> float:
        value = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        return value if math.isfinite(value) and value > 0.0 else self._now_s()

    def _on_state(self, message: String) -> None:
        self.metrics.set_state(message.data)
        if self.metrics.state == "DONE":
            self.finalize()

    def _on_setpoint(self, message: TrajectorySetpoint) -> None:
        position = message.position
        if len(position) < 3:
            return
        # PX4 NED -> local ENU; the transform is self-inverse.
        source_time_s = float(message.timestamp) * 1.0e-6
        if not math.isfinite(source_time_s) or source_time_s <= 0.0:
            source_time_s = self._now_s()
        self.metrics.set_setpoint(
            float(position[1]), float(position[0]), -float(position[2]),
            source_time_s)

    def _on_cue(self, message: PointStamped) -> None:
        point = message.point
        self.metrics.set_cue(point.x, point.y, point.z)

    def _on_aruco(self, message: Bool) -> None:
        self.metrics.add_aruco(message.data)

    def _on_plan(self, message: MarkerArray) -> None:
        markers = list(message.markers)
        if not markers or markers[0].action != Marker.DELETEALL:
            return
        if len(markers) == 1:
            self.metrics.clear_active_sfc()
            return
        stamp = (markers[0].header.stamp.sec,
                 markers[0].header.stamp.nanosec)
        frame = markers[0].header.frame_id
        expected_frame = (self.metrics.clearance_map.frame_id
                          if self.metrics.clearance_map else frame)
        if not frame or frame != expected_frame:
            return
        if any(marker.header.frame_id != frame or
               (marker.header.stamp.sec, marker.header.stamp.nanosec) != stamp
               for marker in markers):
            return
        paths = [marker for marker in markers[1:]
                 if marker.ns == "active_path"]
        boxes = [marker for marker in markers[1:]
                 if marker.ns == "active_sfc"]
        if (len(paths) != 1 or not boxes
                or len(paths) + len(boxes) != len(markers) - 1
                or len({marker.id for marker in boxes}) != len(boxes)):
            return
        path = paths[0]
        if (path.type != Marker.LINE_STRIP or path.action != Marker.ADD
                or path.id <= 0 or len(path.points) < 2
                or not all(math.isfinite(value) for point in path.points
                           for value in (point.x, point.y, point.z))
                or not self._identity_orientation(path.pose.orientation)):
            return
        bounds = []
        for marker in boxes:
            values = (
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z),
                float(marker.scale.x), float(marker.scale.y),
                float(marker.scale.z))
            if (marker.type != Marker.CUBE or marker.action != Marker.ADD
                    or not all(math.isfinite(value) for value in values)
                    or any(value <= 0.0 for value in values[3:])
                    or not self._identity_orientation(
                        marker.pose.orientation)):
                return
            centre = values[:3]
            half = tuple(0.5 * value for value in values[3:])
            bounds.append((
                tuple(value - radius for value, radius in zip(centre, half)),
                tuple(value + radius for value, radius in zip(centre, half))))
        accepted = self.metrics.add_sfc_snapshot(path.id, bounds)
        if accepted and self._closed:
            self._write_summary()

    @staticmethod
    def _identity_orientation(orientation) -> bool:
        values = (float(orientation.x), float(orientation.y),
                  float(orientation.z), float(orientation.w))
        return (all(math.isfinite(value) for value in values)
                and max(abs(value) for value in values[:3]) <= 1.0e-9
                and abs(abs(values[3]) - 1.0) <= 1.0e-9)

    def _on_rosout(self, message: Log) -> None:
        changed = self.metrics.add_log(message.msg)
        if self._closed and changed:
            self._write_summary()

    def _on_position(self, message: PointStamped) -> None:
        if self._closed:
            return
        receive_time_s = self._now_s()
        now_s = self._stamp_s(message.header.stamp)
        point = message.point
        try:
            derived = self.metrics.add_position(
                point.x, point.y, point.z, now_s,
                setpoint_max_age_s=self.setpoint_max_age_s)
        except ValueError:
            return
        setpoint = self.metrics.latest_setpoint or (math.nan,) * 3
        cue = self.metrics.latest_cue or (math.nan,) * 3
        frames = self.metrics.aruco_frames
        aruco_rate = (100.0 * self.metrics.aruco_hits / frames
                      if frames else math.nan)
        latest_plan_ms = (self.metrics.plan_latencies_ms[-1]
                          if self.metrics.plan_latencies_ms else math.nan)
        latest_sfc_ms = (self.metrics.sfc_generation_times_ms[-1]
                         if self.metrics.sfc_generation_times_ms else math.nan)
        active_widths = [min(box[3] - box[0], box[4] - box[1])
                         for box in self.metrics.active_sfc_boxes]
        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "sample_index": self.metrics.sample_count,
            "timestamp_utc": _utc_now(),
            "ros_time_s": now_s,
            "receive_ros_time_s": receive_time_s,
            "elapsed_s": now_s - self.metrics.first_ros_time_s,
            "mission_state": self.metrics.state,
            **derived,
            "setpoint_x_enu_m": setpoint[0],
            "setpoint_y_enu_m": setpoint[1],
            "setpoint_z_enu_m": setpoint[2],
            "cue_x_enu_m": cue[0], "cue_y_enu_m": cue[1],
            "cue_z_enu_m": cue[2],
            "path_length_m": self.metrics.path_length_m,
            "path_length_xy_m": self.metrics.path_length_xy_m,
            "active_plan_seq": self.metrics.active_plan_seq,
            "replan_count": self.metrics.replan_count,
            "aruco_detected": self.metrics.latest_aruco_detected,
            "aruco_detection_rate_pct": aruco_rate,
            "latest_astar_plan_time_ms": latest_plan_ms,
            "latest_sfc_generation_time_ms": latest_sfc_ms,
            "active_sfc_min_width_m": (
                min(active_widths) if active_widths else math.nan),
            "active_sfc_avg_width_m": (
                sum(active_widths) / len(active_widths)
                if active_widths else math.nan),
            "active_sfc_corridor_count": len(active_widths),
            "sfc_violation_count": self.metrics.sfc_violation_events,
        }
        self._writer.writerow({key: _clean(row.get(key))
                               for key in SAMPLE_FIELDS})

    def _write_summary(self) -> None:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_utc": self.started_utc,
            "finished_utc": _utc_now(),
            "final_state": self.metrics.state,
            "sample_csv": str(self.sample_path),
            "samples": self.metrics.sample_count,
            **self.metrics.summary(),
            "map_yaml": (str(self.metrics.clearance_map.path)
                         if self.metrics.clearance_map else ""),
            "map_yaml_sha256": (self.metrics.clearance_map.sha256
                                if self.metrics.clearance_map else ""),
        }
        temporary = self.summary_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow({key: _clean(value)
                             for key, value in summary.items()})
        os.replace(temporary, self.summary_path)

    def finalize(self) -> None:
        if not self._closed:
            self._sample_stream.flush()
            self._sample_stream.close()
            self._closed = True
        self._write_summary()

    def destroy_node(self):
        self.finalize()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
