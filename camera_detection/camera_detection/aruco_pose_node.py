#!/usr/bin/env python3
"""Calibrated and quality-gated ArUco pose estimation for trailer landing.

Only RGB images, CameraInfo and (optionally) the corresponding depth image are
used as marker measurements.  Gazebo model pose is intentionally not an input.
The node supports separate front/down instances through fully parameterized
input and output topics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from typing import Optional, Sequence

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, Float32, Int32

try:
    from tf2_geometry_msgs import (
        do_transform_pose_stamped,
        do_transform_pose_with_covariance_stamped,
    )
    from tf2_ros import Buffer, TransformException, TransformListener

    TF2_AVAILABLE = True
except Exception:  # pragma: no cover - depends on ROS installation
    TF2_AVAILABLE = False
    Buffer = TransformListener = object  # type: ignore[assignment,misc]
    TransformException = Exception  # type: ignore[assignment]


@dataclass(frozen=True)
class SquarePoseEstimate:
    """One physically valid IPPE solution with quantitative quality."""

    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    ambiguity_ratio: float
    facing_cosine: float


@dataclass(frozen=True)
class DepthSample:
    """Robust depth statistics inside a detected marker polygon."""

    median_m: float
    mad_m: float
    valid_count: int


@dataclass(frozen=True)
class InnovationDecision:
    accepted: bool
    mahalanobis_squared: float


def marker_object_points(marker_size_m: float) -> np.ndarray:
    """Return IPPE_SQUARE points in required TL, TR, BR, BL order."""

    size = float(marker_size_m)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("marker_size_m must be finite and positive")
    half = size / 2.0
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def reprojection_rmse_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    """Compute 2-D corner reprojection RMSE, in pixels."""

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    residual = projected.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def estimate_square_marker_pose(
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_size_m: float,
    max_reprojection_error_px: float = 2.5,
    min_facing_cosine: float = 0.15,
    min_ambiguity_ratio: float = 1.15,
) -> tuple[Optional[SquarePoseEstimate], str]:
    """Solve both IPPE square poses and reject flipped / ambiguous results.

    The selected marker face normal must point toward the camera, all object
    corners must have positive camera Z, and the runner-up reprojection error
    must be sufficiently worse than the best solution.
    """

    image_points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    if not np.all(np.isfinite(image_points)):
        return None, "nonfinite_corners"
    if not np.all(np.isfinite(camera_matrix)) or camera_matrix[0, 0] <= 0.0 \
            or camera_matrix[1, 1] <= 0.0:
        return None, "invalid_intrinsics"

    object_points = marker_object_points(marker_size_m)
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None, "solvepnp_error"
    if len(result) < 3 or not result[0]:
        return None, "solvepnp_failed"

    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for raw_rvec, raw_tvec in zip(result[1], result[2]):
        rvec = np.asarray(raw_rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(raw_tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ object_points.T).T + tvec
        if np.min(camera_points[:, 2]) <= 0.01:
            continue
        distance = float(np.linalg.norm(tvec))
        if distance <= 1e-6:
            continue
        # Object +Z is the printed marker face normal.  A visible marker must
        # point approximately from the marker back toward the camera.
        facing = float(np.dot(rotation[:, 2], -tvec / distance))
        if facing < min_facing_cosine:
            continue
        error = reprojection_rmse_px(
            object_points, image_points, rvec, tvec, camera_matrix, distortion
        )
        if math.isfinite(error):
            candidates.append((error, facing, rvec, tvec))

    if not candidates:
        return None, "flipped_or_behind_camera"
    candidates.sort(key=lambda item: item[0])
    best_error, facing, best_rvec, best_tvec = candidates[0]
    if best_error > max_reprojection_error_px:
        return None, "reprojection_error"

    if len(candidates) == 1:
        ambiguity_ratio = math.inf
    else:
        ambiguity_ratio = (candidates[1][0] + 1e-6) / (best_error + 1e-6)
        if ambiguity_ratio < min_ambiguity_ratio:
            return None, "ippe_ambiguity"

    return (
        SquarePoseEstimate(
            rvec=best_rvec,
            tvec=best_tvec,
            reprojection_error_px=best_error,
            ambiguity_ratio=ambiguity_ratio,
            facing_cosine=facing,
        ),
        "accepted",
    )


def robust_marker_depth(
    depth_m: np.ndarray,
    rgb_corners_px: np.ndarray,
    rgb_shape: Sequence[int],
    min_depth_m: float,
    max_depth_m: float,
    minimum_samples: int = 12,
) -> Optional[DepthSample]:
    """Return median/MAD depth inside the marker, handling RGB-depth scaling."""

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        return None
    rgb_height, rgb_width = int(rgb_shape[0]), int(rgb_shape[1])
    if rgb_width <= 0 or rgb_height <= 0:
        return None
    scaled = np.asarray(rgb_corners_px, dtype=np.float64).reshape(4, 2).copy()
    scaled[:, 0] *= depth.shape[1] / rgb_width
    scaled[:, 1] *= depth.shape[0] / rgb_height
    polygon = np.rint(scaled).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, depth.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, depth.shape[0] - 1)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    # Avoid sampling the depth discontinuity immediately outside the marker.
    if np.count_nonzero(mask) >= 36:
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    values = depth[
        (mask != 0)
        & np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    ]
    if values.size < int(minimum_samples):
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return DepthSample(median, mad, int(values.size))


def covariance_and_quality(
    estimate: SquarePoseEstimate,
    camera_matrix: np.ndarray,
    marker_perimeter_px: float,
    depth_residual_m: Optional[float],
    max_reprojection_error_px: float,
    depth_tolerance_m: float,
) -> tuple[np.ndarray, float]:
    """Build conservative pose covariance and a normalized quality score."""

    focal = max(1.0, 0.5 * (camera_matrix[0, 0] + camera_matrix[1, 1]))
    depth = max(0.05, float(estimate.tvec[2]))
    pixel_sigma = max(0.20, estimate.reprojection_error_px)
    sigma_xy = max(0.003, depth * pixel_sigma / focal)
    sigma_z = max(0.008, 2.0 * sigma_xy)
    sigma_angle = max(math.radians(0.5), pixel_sigma / max(marker_perimeter_px, 4.0))
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[0, 0] = covariance[1, 1] = sigma_xy**2
    covariance[2, 2] = sigma_z**2
    covariance[3, 3] = covariance[4, 4] = covariance[5, 5] = sigma_angle**2

    reprojection_quality = math.exp(
        -((estimate.reprojection_error_px / max(max_reprojection_error_px, 1e-3)) ** 2)
    )
    ambiguity_quality = 1.0 if math.isinf(estimate.ambiguity_ratio) else min(
        1.0, max(0.0, (estimate.ambiguity_ratio - 1.0) / 1.5)
    )
    size_quality = min(1.0, max(0.0, marker_perimeter_px / 160.0))
    if depth_residual_m is None:
        depth_quality = 0.8
    else:
        depth_quality = math.exp(
            -((depth_residual_m / max(depth_tolerance_m, 1e-3)) ** 2)
        )
    quality = float(
        np.clip(
            reprojection_quality
            * (0.45 + 0.55 * ambiguity_quality)
            * (0.45 + 0.55 * size_quality)
            * depth_quality,
            0.0,
            1.0,
        )
    )
    covariance /= max(quality, 0.1)
    return covariance, quality


class PoseInnovationGate:
    """Timestamp-based constant-velocity Mahalanobis gate for pose outliers."""

    def __init__(
        self,
        threshold_squared: float = 16.27,
        process_variance_m2_s2: float = 0.25,
        reset_after_s: float = 1.0,
    ) -> None:
        self.threshold_squared = float(threshold_squared)
        self.process_variance = float(process_variance_m2_s2)
        self.reset_after_s = float(reset_after_s)
        self.position: Optional[np.ndarray] = None
        self.velocity = np.zeros(3, dtype=np.float64)
        self.variance = np.ones(3, dtype=np.float64)
        self.stamp_s: Optional[float] = None

    def reset(self) -> None:
        self.position = None
        self.velocity[:] = 0.0
        self.variance[:] = 1.0
        self.stamp_s = None

    def update(
        self,
        stamp_s: float,
        position_m: Sequence[float],
        measurement_variance_m2: Sequence[float],
    ) -> InnovationDecision:
        position = np.asarray(position_m, dtype=np.float64).reshape(3)
        measurement_variance = np.maximum(
            np.asarray(measurement_variance_m2, dtype=np.float64).reshape(3), 1e-8
        )
        stamp = float(stamp_s)
        if not math.isfinite(stamp) or not np.all(np.isfinite(position)):
            return InnovationDecision(False, math.inf)
        if self.position is None or self.stamp_s is None:
            self.position = position.copy()
            self.variance = measurement_variance.copy()
            self.stamp_s = stamp
            return InnovationDecision(True, 0.0)

        dt = stamp - self.stamp_s
        if dt <= 0.0:
            return InnovationDecision(False, math.inf)
        if dt > self.reset_after_s:
            self.reset()
            return self.update(stamp, position, measurement_variance)
        predicted = self.position + self.velocity * dt
        predicted_variance = self.variance + self.process_variance * dt * dt
        innovation = position - predicted
        innovation_variance = predicted_variance + measurement_variance
        mahalanobis_squared = float(np.sum(innovation * innovation / innovation_variance))
        if mahalanobis_squared > self.threshold_squared:
            return InnovationDecision(False, mahalanobis_squared)

        gain = predicted_variance / innovation_variance
        previous_position = self.position.copy()
        self.position = predicted + gain * innovation
        measured_velocity = (self.position - previous_position) / dt
        self.velocity = 0.7 * self.velocity + 0.3 * measured_velocity
        self.variance = np.maximum((1.0 - gain) * predicted_variance, 1e-8)
        self.stamp_s = stamp
        return InnovationDecision(True, mahalanobis_squared)


class DetectionStreak:
    """Require a time-contiguous run of accepted observations before output."""

    def __init__(self, minimum_frames: int = 3, maximum_gap_s: float = 0.2) -> None:
        self.minimum_frames = max(1, int(minimum_frames))
        self.maximum_gap_s = float(maximum_gap_s)
        self.count = 0
        self.marker_id: Optional[int] = None
        self.last_stamp_s: Optional[float] = None

    def update(self, accepted: bool, marker_id: Optional[int], stamp_s: float) -> bool:
        stamp = float(stamp_s)
        contiguous = (
            accepted
            and marker_id is not None
            and self.marker_id == marker_id
            and self.last_stamp_s is not None
            and 0.0 < stamp - self.last_stamp_s <= self.maximum_gap_s
        )
        if not accepted or marker_id is None:
            self.count = 0
            self.marker_id = None
            self.last_stamp_s = None
            return False
        self.count = self.count + 1 if contiguous else 1
        self.marker_id = int(marker_id)
        self.last_stamp_s = stamp
        return self.count >= self.minimum_frames


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _rotation_vector_to_quaternion(rvec: np.ndarray) -> tuple[float, ...]:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array([0.25 * scale,
                                   (rotation[0, 1] + rotation[1, 0]) / scale,
                                   (rotation[0, 2] + rotation[2, 0]) / scale,
                                   (rotation[2, 1] - rotation[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array([(rotation[0, 1] + rotation[1, 0]) / scale,
                                   0.25 * scale,
                                   (rotation[1, 2] + rotation[2, 1]) / scale,
                                   (rotation[0, 2] - rotation[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array([(rotation[0, 2] + rotation[2, 0]) / scale,
                                   (rotation[1, 2] + rotation[2, 1]) / scale,
                                   0.25 * scale,
                                   (rotation[1, 0] - rotation[0, 1]) / scale])
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


class ArucoPoseNode(Node):
    """Publish fresh, calibrated, depth-consistent marker observations."""

    def __init__(self) -> None:
        super().__init__("aruco_pose_node")
        self._declare_parameters()
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.marker_id = int(self.get_parameter("marker_id").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.depth_required = bool(self.get_parameter("depth_required").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.depth_absolute_tolerance_m = float(
            self.get_parameter("depth_absolute_tolerance_m").value
        )
        self.depth_relative_tolerance = float(
            self.get_parameter("depth_relative_tolerance").value
        )
        self.depth_sync_tolerance_s = float(
            self.get_parameter("depth_sync_tolerance_s").value
        )
        self.max_image_age_s = float(self.get_parameter("max_image_age_s").value)
        self.max_reprojection_error_px = float(
            self.get_parameter("max_reprojection_error_px").value
        )
        self.min_facing_cosine = float(self.get_parameter("min_facing_cosine").value)
        self.min_ambiguity_ratio = float(
            self.get_parameter("min_ambiguity_ratio").value
        )
        self.optical_frame_id = str(self.get_parameter("optical_frame_id").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)

        dictionary_name = str(self.get_parameter("aruco_dictionary").value)
        dictionary_values = {
            "DICT_4X4_50": aruco.DICT_4X4_50,
            "DICT_5X5_100": aruco.DICT_5X5_100,
            "DICT_6X6_250": aruco.DICT_6X6_250,
        }
        if dictionary_name not in dictionary_values:
            raise ValueError(f"unsupported aruco_dictionary: {dictionary_name}")
        dictionary = aruco.getPredefinedDictionary(dictionary_values[dictionary_name])
        try:
            detector_parameters = aruco.DetectorParameters()
        except AttributeError:  # OpenCV 4.5 used by Ubuntu 22.04
            detector_parameters = aruco.DetectorParameters_create()
        detector_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector_parameters.cornerRefinementWinSize = int(
            self.get_parameter("subpixel_window_px").value
        )
        try:
            detector = aruco.ArucoDetector(dictionary, detector_parameters)
            self._detect = detector.detectMarkers
        except AttributeError:  # OpenCV 4.5 used by Ubuntu 22.04
            self._detect = lambda gray: aruco.detectMarkers(
                gray, dictionary, parameters=detector_parameters
            )

        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None
        self.camera_info_received_s: Optional[float] = None
        self.calibration_from_file = False
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_depth_stamp_s: Optional[float] = None
        self.depth_history = deque(
            maxlen=max(2, int(self.get_parameter("depth_history_size").value))
        )
        self.pending_images = deque(maxlen=4)
        self.rejected_total = 0
        self.last_rejection_log_s = -math.inf
        calibration_file = str(self.get_parameter("calibration_file").value)
        if calibration_file:
            self._load_calibration_file(calibration_file)

        self.innovation_gate = PoseInnovationGate(
            threshold_squared=float(self.get_parameter("innovation_gate_squared").value),
            process_variance_m2_s2=float(
                self.get_parameter("innovation_process_variance").value
            ),
            reset_after_s=float(self.get_parameter("innovation_reset_after_s").value),
        )
        self.detection_streak = DetectionStreak(
            minimum_frames=int(self.get_parameter("minimum_detection_frames").value),
            maximum_gap_s=float(self.get_parameter("maximum_detection_gap_s").value),
        )

        self.detected_publisher = self.create_publisher(
            Bool, str(self.get_parameter("detected_topic").value), 10
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("pose_topic").value), 10
        )
        self.covariance_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter("pose_covariance_topic").value),
            10,
        )
        self.offset_publisher = self.create_publisher(
            Point, str(self.get_parameter("offset_topic").value), 10
        )
        self.quality_publisher = self.create_publisher(
            Float32, str(self.get_parameter("quality_topic").value), 10
        )
        self.reprojection_publisher = self.create_publisher(
            Float32, str(self.get_parameter("reprojection_error_topic").value), 10
        )
        self.rejected_publisher = self.create_publisher(
            Int32, str(self.get_parameter("rejected_candidates_topic").value), 10
        )
        if self.publish_debug:
            self.debug_publisher = self.create_publisher(
                CompressedImage, str(self.get_parameter("debug_topic").value), 10
            )

        image_topic = str(self.get_parameter("image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        self.create_subscription(
            CameraInfo, info_topic, self._camera_info_callback, qos_profile_sensor_data
        )
        self.create_subscription(Image, image_topic, self._image_callback,
                                 qos_profile_sensor_data)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._depth_callback,
                                     qos_profile_sensor_data)

        self.tf_buffer = None
        self.tf_listener = None
        if self.target_frame:
            if not TF2_AVAILABLE:
                raise RuntimeError("target_frame configured but tf2 is unavailable")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"calibrated ArUco ready: image={image_topic} depth={depth_topic or 'off'} "
            f"marker={self.marker_id} size={self.marker_size_m:.3f}m "
            f"output_frame={self.target_frame or self.optical_frame_id or 'image header'}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "image_topic": "/down_camera/image",
            "camera_info_topic": "/down_camera/camera_info",
            "depth_topic": "/down_depth/image_raw",
            "aruco_dictionary": "DICT_4X4_50",
            "marker_size_m": 1.0,
            "marker_id": 0,
            "calibration_file": "",
            "publish_debug": True,
            "subpixel_window_px": 5,
            "depth_required": True,
            "depth_scale": 0.001,
            "depth_min_m": 0.15,
            "depth_max_m": 80.0,
            "depth_absolute_tolerance_m": 0.35,
            "depth_relative_tolerance": 0.08,
            "depth_sync_tolerance_s": 0.10,
            "depth_history_size": 12,
            "max_image_age_s": 0.25,
            "camera_info_timeout_s": 2.0,
            "max_reprojection_error_px": 2.5,
            "min_facing_cosine": 0.15,
            "min_ambiguity_ratio": 1.15,
            "minimum_detection_frames": 3,
            "maximum_detection_gap_s": 0.75,
            "innovation_gate_squared": 16.27,
            "innovation_process_variance": 0.25,
            "innovation_reset_after_s": 1.0,
            "optical_frame_id": "down_camera_optical_frame",
            "target_frame": "base_link",
            "tf_timeout_s": 0.05,
            "detected_topic": "/perception/down/aruco_detected",
            "pose_topic": "/perception/down/marker_pose",
            "pose_covariance_topic": "/perception/down/marker_pose_covariance",
            "offset_topic": "/perception/down/aruco_offset",
            "quality_topic": "/perception/down/quality",
            "reprojection_error_topic": "/perception/down/reprojection_error_px",
            "rejected_candidates_topic": "/perception/down/rejected_candidates",
            "debug_topic": "/perception/down/debug/compressed",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_calibration_file(self, path: str) -> None:
        if not os.path.isfile(path):
            self.get_logger().error(f"calibration_file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        try:
            self.camera_matrix = np.asarray(
                data["camera_matrix"]["data"], dtype=np.float64
            ).reshape(3, 3)
            self.distortion = np.asarray(
                data["distortion_coefficients"]["data"], dtype=np.float64
            ).reshape(-1, 1)
            self.camera_info_received_s = self.get_clock().now().nanoseconds * 1e-9
            self.calibration_from_file = True
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"invalid calibration_file {path}: {exc}")

    def _camera_info_callback(self, message: CameraInfo) -> None:
        matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 \
                or matrix[1, 1] <= 0.0:
            return
        self.camera_matrix = matrix
        self.distortion = (
            np.asarray(message.d, dtype=np.float64).reshape(-1, 1)
            if message.d else np.zeros((5, 1), dtype=np.float64)
        )
        self.camera_info_received_s = self.get_clock().now().nanoseconds * 1e-9
        self.calibration_from_file = False

    @staticmethod
    def _image_to_bgr(message: Image) -> np.ndarray:
        channels = {
            "rgb8": (3, False), "bgr8": (3, False),
            "rgba8": (4, False), "bgra8": (4, False),
            "mono8": (1, False), "8UC1": (1, False),
        }
        if message.encoding not in channels:
            raise ValueError(f"unsupported RGB encoding: {message.encoding}")
        count, _ = channels[message.encoding]
        row_bytes = int(message.width) * count
        raw = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), int(message.step)
        )[:, :row_bytes]
        if count == 1:
            return cv2.cvtColor(raw.reshape(message.height, message.width),
                                cv2.COLOR_GRAY2BGR)
        image = raw.reshape(message.height, message.width, count)
        if message.encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if message.encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if message.encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(image)

    def _depth_to_metres(self, message: Image) -> np.ndarray:
        encoding = message.encoding.upper()
        if encoding in ("32FC1", "TYPE_32FC1"):
            dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
            item_size = 4
            scale = 1.0
        elif encoding in ("16UC1", "MONO16", "TYPE_16UC1"):
            dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
            item_size = 2
            scale = self.depth_scale
        else:
            raise ValueError(f"unsupported depth encoding: {message.encoding}")
        row_items = int(message.step) // item_size
        raw = np.frombuffer(message.data, dtype=dtype).reshape(
            int(message.height), row_items
        )[:, : int(message.width)]
        return raw.astype(np.float64) * scale

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = self._depth_to_metres(message)
        except (ValueError, TypeError) as exc:
            self._log_rejection(f"depth_decode:{exc}")
            return
        self.latest_depth_m = depth
        self.latest_depth_stamp_s = _stamp_seconds(message.header.stamp)
        self.depth_history.append((self.latest_depth_stamp_s, depth))
        self._process_pending_image()

    def _image_callback(self, message: Image) -> None:
        if not self.depth_required:
            self._process_image(message)
            return
        self.pending_images.append(message)
        self._process_pending_image()

    def _process_pending_image(self) -> None:
        if not self.pending_images or not self.depth_history:
            return
        best = None
        for index, message in enumerate(self.pending_images):
            stamp_s = _stamp_seconds(message.header.stamp)
            error = min(abs(stamp_s - sample[0]) for sample in self.depth_history)
            if error <= self.depth_sync_tolerance_s \
                    and (best is None or error < best[0]):
                best = (error, index, message)
        if best is None:
            newest_depth_s = self.depth_history[-1][0]
            while self.pending_images and (
                newest_depth_s - _stamp_seconds(self.pending_images[0].header.stamp)
                > self.depth_sync_tolerance_s
            ):
                self.pending_images.popleft()
            return
        _, index, message = best
        # Discard older RGB frames; processing the closest fresh sample keeps
        # latency bounded when rendering temporarily runs below real time.
        for _ in range(index + 1):
            self.pending_images.popleft()
        self._process_image(message)

    def _log_rejection(self, reason: str) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self.last_rejection_log_s >= 2.0:
            self.get_logger().warn(f"ArUco observation rejected: {reason}")
            self.last_rejection_log_s = now_s

    def _publish_rejected(self, reason: str, rejected_candidates: int) -> None:
        self.rejected_total += max(1, int(rejected_candidates))
        self.detected_publisher.publish(Bool(data=False))
        self.quality_publisher.publish(Float32(data=0.0))
        self.rejected_publisher.publish(Int32(data=self.rejected_total))
        self.detection_streak.update(False, None, 0.0)
        self._log_rejection(reason)

    def _fresh_image(self, stamp_s: float) -> bool:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if stamp_s <= 0.0 or now_s <= 0.0:
            return False
        age = now_s - stamp_s
        return -0.02 <= age <= self.max_image_age_s

    @staticmethod
    def _refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        refined = np.asarray(corners, dtype=np.float32).reshape(4, 1, 2).copy()
        cv2.cornerSubPix(
            gray,
            refined,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        return refined.reshape(4, 2)

    def _transform_outputs(
        self,
        pose: PoseStamped,
        covariance_pose: PoseWithCovarianceStamped,
    ) -> tuple[PoseStamped, PoseWithCovarianceStamped]:
        if not self.target_frame or self.target_frame == pose.header.frame_id:
            return pose, covariance_pose
        assert self.tf_buffer is not None
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            pose.header.frame_id,
            Time.from_msg(pose.header.stamp),
            timeout=Duration(seconds=self.tf_timeout_s),
        )
        return (
            do_transform_pose_stamped(pose, transform),
            do_transform_pose_with_covariance_stamped(covariance_pose, transform),
        )

    def _process_image(self, message: Image) -> None:
        stamp_s = _stamp_seconds(message.header.stamp)
        if not self._fresh_image(stamp_s):
            self._publish_rejected("stale_or_zero_image_stamp", 1)
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        info_timeout = float(self.get_parameter("camera_info_timeout_s").value)
        if self.camera_matrix is None or self.distortion is None \
                or self.camera_info_received_s is None \
                or (not self.calibration_from_file
                    and now_s - self.camera_info_received_s > info_timeout):
            self._publish_rejected("missing_or_stale_camera_info", 1)
            return
        try:
            frame = self._image_to_bgr(message)
        except (ValueError, TypeError) as exc:
            self._publish_rejected(f"image_decode:{exc}", 1)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detect(gray)
        rejected_count = len(rejected)
        if ids is None or len(ids) == 0:
            self._publish_rejected("marker_not_found", max(1, rejected_count))
            self._publish_debug(frame, corners, ids, message, "SEARCHING", None)
            return
        matches = np.where(ids.flatten() == self.marker_id)[0]
        if matches.size == 0:
            self._publish_rejected("configured_marker_id_not_found",
                                   max(1, rejected_count + len(ids)))
            self._publish_debug(frame, corners, ids, message, "WRONG ID", None)
            return
        try:
            selected = self._refine_corners(gray, corners[int(matches[0])])
        except cv2.error:
            self._publish_rejected("subpixel_refinement_failed",
                                   max(1, rejected_count))
            return

        center = np.mean(selected, axis=0)
        offset = Point()
        offset.x = float((center[0] - 0.5 * frame.shape[1]) / (0.5 * frame.shape[1]))
        offset.y = float((center[1] - 0.5 * frame.shape[0]) / (0.5 * frame.shape[0]))
        self.offset_publisher.publish(offset)

        estimate, reason = estimate_square_marker_pose(
            selected,
            self.camera_matrix,
            self.distortion,
            self.marker_size_m,
            self.max_reprojection_error_px,
            self.min_facing_cosine,
            self.min_ambiguity_ratio,
        )
        if estimate is None:
            self._publish_rejected(reason, max(1, rejected_count))
            self._publish_debug(frame, corners, ids, message, reason, None)
            return
        self.reprojection_publisher.publish(
            Float32(data=float(estimate.reprojection_error_px))
        )

        depth_sample = None
        depth_residual = None
        depth_tolerance = self.depth_absolute_tolerance_m
        synchronized_depth = None
        if self.depth_history:
            depth_stamp, candidate_depth = min(
                self.depth_history, key=lambda sample: abs(stamp_s - sample[0])
            )
            if abs(stamp_s - depth_stamp) <= self.depth_sync_tolerance_s:
                synchronized_depth = candidate_depth
        if synchronized_depth is not None:
            depth_sample = robust_marker_depth(
                synchronized_depth,
                selected,
                frame.shape[:2],
                self.depth_min_m,
                self.depth_max_m,
            )
        if depth_sample is not None:
            depth_residual = abs(depth_sample.median_m - float(estimate.tvec[2]))
            depth_tolerance = self.depth_absolute_tolerance_m \
                + self.depth_relative_tolerance * float(estimate.tvec[2])
            if depth_residual > depth_tolerance:
                self._publish_rejected("depth_pose_inconsistent", max(1, rejected_count))
                self._publish_debug(frame, corners, ids, message, "DEPTH GATE", estimate)
                return
        elif self.depth_required:
            self._publish_rejected("missing_synchronized_marker_depth",
                                   max(1, rejected_count))
            self._publish_debug(frame, corners, ids, message, "NO DEPTH", estimate)
            return

        perimeter = float(cv2.arcLength(selected.astype(np.float32), True))
        covariance, quality = covariance_and_quality(
            estimate,
            self.camera_matrix,
            perimeter,
            depth_residual,
            self.max_reprojection_error_px,
            depth_tolerance,
        )
        innovation = self.innovation_gate.update(
            stamp_s, estimate.tvec, np.diag(covariance)[:3]
        )
        if not innovation.accepted:
            self._publish_rejected(
                f"innovation_gate_d2={innovation.mahalanobis_squared:.2f}",
                max(1, rejected_count),
            )
            self._publish_debug(frame, corners, ids, message, "OUTLIER", estimate)
            return
        stable = self.detection_streak.update(True, self.marker_id, stamp_s)
        if not stable:
            self.detected_publisher.publish(Bool(data=False))
            self.quality_publisher.publish(Float32(data=float(quality)))
            self.rejected_publisher.publish(Int32(data=self.rejected_total))
            self._publish_debug(frame, corners, ids, message, "ACQUIRING", estimate)
            return

        source_frame = self.optical_frame_id or message.header.frame_id
        pose = PoseStamped()
        pose.header = message.header
        pose.header.frame_id = source_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            float(estimate.tvec[0]), float(estimate.tvec[1]), float(estimate.tvec[2])
        )
        quaternion = _rotation_vector_to_quaternion(estimate.rvec)
        pose.pose.orientation.x, pose.pose.orientation.y, \
            pose.pose.orientation.z, pose.pose.orientation.w = quaternion
        covariance_pose = PoseWithCovarianceStamped()
        covariance_pose.header = pose.header
        covariance_pose.pose.pose = pose.pose
        covariance_pose.pose.covariance = covariance.reshape(-1).tolist()
        try:
            pose, covariance_pose = self._transform_outputs(pose, covariance_pose)
        except TransformException as exc:
            self._publish_rejected(f"tf_lookup:{exc}", max(1, rejected_count))
            self._publish_debug(frame, corners, ids, message, "TF FAIL", estimate)
            return

        self.pose_publisher.publish(pose)
        self.covariance_publisher.publish(covariance_pose)
        self.detected_publisher.publish(Bool(data=True))
        self.quality_publisher.publish(Float32(data=float(quality)))
        self.rejected_publisher.publish(Int32(data=self.rejected_total))
        self._publish_debug(frame, corners, ids, message,
                            f"TRACK q={quality:.2f}", estimate)

    def _publish_debug(
        self,
        frame: np.ndarray,
        corners,
        ids,
        message: Image,
        status: str,
        estimate: Optional[SquarePoseEstimate],
    ) -> None:
        if not self.publish_debug:
            return
        if self.debug_publisher.get_subscription_count() == 0:
            return
        if ids is not None and len(ids):
            aruco.drawDetectedMarkers(frame, corners, ids)
        if estimate is not None and self.camera_matrix is not None:
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.distortion,
                estimate.rvec, estimate.tvec, self.marker_size_m * 0.5,
            )
            status += (
                f" err={estimate.reprojection_error_px:.2f}px "
                f"z={estimate.tvec[2]:.2f}m"
            )
        cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0) if status.startswith("TRACK") else (0, 170, 255),
                    2, cv2.LINE_AA)
        cv2.drawMarker(frame, (frame.shape[1] // 2, frame.shape[0] // 2),
                       (0, 0, 255), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        success, encoded = cv2.imencode(".jpg", frame)
        if success:
            debug = CompressedImage()
            debug.header = message.header
            debug.format = "jpeg"
            debug.data = encoded.tobytes()
            self.debug_publisher.publish(debug)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
