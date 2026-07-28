#!/usr/bin/env python3
"""
Calibrated and quality-gated ArUco pose estimation for trailer landing.

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

from aruco_landing.aruco_board import (
    BoardLayout,
    BoardPoseEstimate,
    estimate_board_pose,
    load_board_layout,
    marker_center_in_camera,
)
from geometry_msgs.msg import (
    Point,
    PoseStamped,
    PoseWithCovarianceStamped,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
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
class DepthConsistencyDecision:
    """Fail-closed agreement of every sampled marker with one PnP pose."""

    accepted: bool
    had_sample: bool
    residual_m: Optional[float]
    tolerance_m: float


@dataclass(frozen=True)
class InnovationDecision:
    accepted: bool
    mahalanobis_squared: float


def evaluate_marker_depth_consistency(
    measured_and_predicted_depths_m: Sequence[tuple[float, float]],
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> DepthConsistencyDecision:
    """
    Require every available board-marker depth to agree with the pose.

    Markers without a usable depth sample are omitted by the caller.  An empty
    sequence is therefore reported separately from an inconsistent sequence so
    ``depth_required`` can preserve its existing missing-depth policy.
    """
    absolute_tolerance_m = float(absolute_tolerance_m)
    relative_tolerance = float(relative_tolerance)
    if not math.isfinite(absolute_tolerance_m) \
            or not math.isfinite(relative_tolerance) \
            or absolute_tolerance_m < 0.0 \
            or relative_tolerance < 0.0:
        raise ValueError("depth tolerances must be finite and non-negative")
    if not measured_and_predicted_depths_m:
        return DepthConsistencyDecision(
            accepted=False,
            had_sample=False,
            residual_m=None,
            tolerance_m=absolute_tolerance_m,
        )

    residuals: list[float] = []
    tolerances: list[float] = []
    accepted = True
    for measured_depth_m, predicted_depth_m in measured_and_predicted_depths_m:
        measured_depth_m = float(measured_depth_m)
        predicted_depth_m = float(predicted_depth_m)
        tolerance_m = absolute_tolerance_m \
            + relative_tolerance * predicted_depth_m
        residual_m = abs(measured_depth_m - predicted_depth_m)
        finite_and_physical = (
            math.isfinite(measured_depth_m)
            and math.isfinite(predicted_depth_m)
            and measured_depth_m > 0.0
            and predicted_depth_m > 0.0
            and math.isfinite(tolerance_m)
            and math.isfinite(residual_m)
        )
        if not finite_and_physical or residual_m > tolerance_m:
            accepted = False
        residuals.append(residual_m)
        tolerances.append(tolerance_m)

    finite_residuals = [value for value in residuals if math.isfinite(value)]
    finite_tolerances = [value for value in tolerances if math.isfinite(value)]
    return DepthConsistencyDecision(
        accepted=accepted,
        had_sample=True,
        residual_m=(
            float(np.median(finite_residuals)) if finite_residuals else None
        ),
        tolerance_m=(
            float(np.median(finite_tolerances))
            if finite_tolerances else absolute_tolerance_m
        ),
    )


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
    """
    Solve both IPPE square poses and reject flipped / ambiguous results.

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
    estimate: SquarePoseEstimate | BoardPoseEstimate,
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
    if isinstance(estimate, BoardPoseEstimate):
        # Coplanar board PnP is much less certain in translation than its
        # sub-pixel corner residual suggests.  The 200 m SITL flights measured
        # 0.13--0.16 m intermittent translation innovations at a 5 m search
        # height while the old model reported only 0.013 m sigma.  Calibrate a
        # conservative range-proportional floor so timestamped fusion weights
        # those observations instead of treating planar scale as exact.
        sigma_xy = max(sigma_xy, 0.02 * depth)
        sigma_z = max(sigma_z, 0.03 * depth)
        if depth_residual_m is not None:
            sigma_z = max(sigma_z, float(depth_residual_m))
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

    def note_missed_detection(self, stamp_s: float) -> None:
        """Clear a stale track after a timestamped marker-detection gap."""
        stamp = float(stamp_s)
        if not math.isfinite(stamp) or self.stamp_s is None:
            return
        gap_s = stamp - self.stamp_s
        if gap_s < 0.0 or gap_s > self.reset_after_s:
            self.reset()

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
    """Debounce initial acquisition while retaining a bounded track latch."""

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
            # The rejected observation itself is never published.  Preserve
            # only the last independently accepted stamp so one dropped frame
            # does not hide another two good poses.  The next accepted sample
            # still resets acquisition when its timestamp gap is too large.
            return False
        self.count = self.count + 1 if contiguous else 1
        self.marker_id = int(marker_id)
        self.last_stamp_s = stamp
        return self.count >= self.minimum_frames


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def image_stamp_is_fresh(
    now_s: float,
    stamp_s: float,
    maximum_age_s: float,
    future_tolerance_s: float,
) -> bool:
    """Check one sensor stamp against the bounded ROS clock interval."""
    values = (now_s, stamp_s, maximum_age_s, future_tolerance_s)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if now_s <= 0.0 or stamp_s <= 0.0:
        return False
    if maximum_age_s <= 0.0 or future_tolerance_s < 0.0:
        return False
    age_s = float(now_s) - float(stamp_s)
    return -float(future_tolerance_s) <= age_s <= float(maximum_age_s)


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
        self.board_layout_file = str(
            self.get_parameter("board_layout_file").value
        )
        self.minimum_board_markers = int(
            self.get_parameter("minimum_board_markers").value
        )
        self.board_outlier_mad_scale = float(
            self.get_parameter("board_outlier_mad_scale").value
        )
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
        self.max_image_future_tolerance_s = float(
            self.get_parameter("max_image_future_tolerance_s").value
        )
        if (
            not math.isfinite(self.max_image_age_s)
            or self.max_image_age_s <= 0.0
            or not math.isfinite(self.max_image_future_tolerance_s)
            or self.max_image_future_tolerance_s < 0.0
            or self.max_image_future_tolerance_s > 0.25
        ):
            raise ValueError(
                "image age must be positive and future tolerance in [0, 0.25]"
            )
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
        self.board_layout: Optional[BoardLayout] = None
        if self.board_layout_file:
            self.board_layout = load_board_layout(self.board_layout_file)
            if self.board_layout.dictionary != dictionary_name:
                raise ValueError(
                    "board layout dictionary does not match aruco_dictionary"
                )
            if self.minimum_board_markers < 1 \
                    or self.minimum_board_markers > len(self.board_layout.markers):
                raise ValueError("minimum_board_markers is outside the board layout")
            if not math.isfinite(self.board_outlier_mad_scale) \
                    or self.board_outlier_mad_scale <= 0.0:
                raise ValueError("board_outlier_mad_scale must be finite and positive")
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

        # Perception is latest-sample data.  A reliable depth-10 FIFO can
        # deliver an old pose long after a brief executor/render stall, which
        # defeats capture-time interpolation and turns a valid marker into a
        # stale measurement at the controller.
        latest_sample_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.detected_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("detected_topic").value),
            latest_sample_qos,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("pose_topic").value),
            latest_sample_qos,
        )
        self.covariance_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter("pose_covariance_topic").value),
            latest_sample_qos,
        )
        self.offset_publisher = self.create_publisher(
            Point,
            str(self.get_parameter("offset_topic").value),
            latest_sample_qos,
        )
        self.quality_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("quality_topic").value),
            latest_sample_qos,
        )
        self.reprojection_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("reprojection_error_topic").value),
            latest_sample_qos,
        )
        self.rejected_publisher = self.create_publisher(
            Int32,
            str(self.get_parameter("rejected_candidates_topic").value),
            latest_sample_qos,
        )
        if self.publish_debug:
            # rqt_image_view requests RELIABLE by default. Keep that
            # compatibility while still preventing an image backlog.
            latest_debug_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.debug_publisher = self.create_publisher(
                CompressedImage,
                str(self.get_parameter("debug_topic").value),
                latest_debug_qos,
            )
            debug_raw_topic = str(
                self.get_parameter("debug_raw_topic").value
            ).strip()
            self.debug_raw_publisher = (
                self.create_publisher(Image, debug_raw_topic, latest_debug_qos)
                if debug_raw_topic
                else None
            )

        image_topic = str(self.get_parameter("image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        # Pose estimation must always consume the newest camera sample. A
        # deeper sensor queue accumulated old images whenever OpenCV briefly
        # exceeded the camera period, turning a clearly detected marker into
        # capture_time_stale rejections at the flight controller.
        self.create_subscription(
            CameraInfo, info_topic, self._camera_info_callback, qos_profile_sensor_data
        )
        self.create_subscription(Image, image_topic, self._image_callback,
                                 latest_sample_qos)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._depth_callback,
                                     latest_sample_qos)

        self.tf_buffer = None
        self.tf_listener = None
        if self.target_frame:
            if not TF2_AVAILABLE:
                raise RuntimeError("target_frame configured but tf2 is unavailable")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        target_description = (
            f"board:{self.board_layout.frame_id}"
            if self.board_layout is not None
            else f"marker:{self.marker_id}"
        )
        self.get_logger().info(
            f"calibrated ArUco ready: image={image_topic} depth={depth_topic or 'off'} "
            f"target={target_description} "
            f"output_frame={self.target_frame or self.optical_frame_id or 'image header'}"
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "image_topic": "/down_camera/image",
            "camera_info_topic": "/down_camera/camera_info",
            # NO DEPTH by default. This package is the REAL-VEHICLE detector
            # (the simulator has its own in landing_mpc), and the airframe's
            # down camera is a plain USB camera with no depth stream. Pointing
            # at a topic nobody publishes, with depth_required below, rejects
            # every detection and looks exactly like "the marker is not being
            # seen". Set both together if a registered depth source is added.
            "depth_topic": "",
            "aruco_dictionary": "DICT_4X4_50",
            # MEASURED marker edge, metres. solvePnP range is LINEAR in this,
            # so a wrong value scales every distance by the same ratio with no
            # other symptom. 0.18 m is the printed marker this airframe uses.
            "marker_size_m": 0.18,
            "marker_id": 0,
            "board_layout_file": "",
            "minimum_board_markers": 1,
            "board_outlier_mad_scale": 3.5,
            "calibration_file": "",
            # ON by default on the real vehicle: the annotated view is the
            # only way to see WHY a detection was rejected, and finding that out
            # in the field is the whole point of having it.
            "publish_debug": True,
            "subpixel_window_px": 5,
            "depth_required": False,
            "depth_scale": 0.001,
            "depth_min_m": 0.15,
            "depth_max_m": 80.0,
            "depth_absolute_tolerance_m": 0.35,
            "depth_relative_tolerance": 0.08,
            "depth_sync_tolerance_s": 0.10,
            "depth_history_size": 12,
            # MEASURED on this airframe: usb_cam frames arrive 0.58-0.60 s
            # after their capture stamp (145 frames, 1280x720 MJPG). At the old
            # 0.25 s every single frame was rejected as stale, which reads as
            # "the marker is never seen" and blocks preflight with no clue why.
            #
            # Raising it is safe here because the latency is COMPENSATED, not
            # ignored: the TF lookup uses the image's own capture stamp, so the
            # marker is placed with the vehicle pose from when the shutter
            # fired, not from now. The gate exists to catch a stalled camera,
            # and 1.0 s still does that while leaving margin over the measured
            # 0.6 s. Re-measure if the camera, format or resolution changes.
            "max_image_age_s": 1.0,
            # Gazebo sensor stamps may lead the most recent source-throttled
            # /clock tick by one bounded publish interval.  The original
            # capture stamp is retained; mission-side state interpolation and
            # future rejection remain authoritative before estimator update.
            "max_image_future_tolerance_s": 0.10,
            "camera_info_timeout_s": 2.0,
            "max_reprojection_error_px": 2.5,
            "min_facing_cosine": 0.15,
            "min_ambiguity_ratio": 1.15,
            "minimum_detection_frames": 3,
            "maximum_detection_gap_s": 1.0,
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
            "debug_raw_topic": "",
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
        return image_stamp_is_fresh(
            now_s,
            stamp_s,
            self.max_image_age_s,
            self.max_image_future_tolerance_s,
        )

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

    def _refine_board_observations(
        self,
        gray: np.ndarray,
        corners: Sequence[np.ndarray],
        ids: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], int]:
        """Refine each unique configured board detection and count rejections."""
        assert self.board_layout is not None
        configured_ids = set(self.board_layout.markers_by_id)
        detected_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        observations: dict[int, np.ndarray] = {}
        rejected_count = int(sum(item not in configured_ids for item in detected_ids))
        for marker_id in configured_ids:
            matches = np.where(detected_ids == marker_id)[0]
            if matches.size == 0:
                continue
            if matches.size != 1:
                rejected_count += int(matches.size)
                continue
            try:
                observations[marker_id] = self._refine_corners(
                    gray, corners[int(matches[0])]
                )
            except cv2.error:
                rejected_count += 1
        return observations, rejected_count

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
        # Gazebo's large, oblique marker can be visually crisp but still be
        # discarded by OpenCV 4.5's adaptive-threshold window sweep.  Keep the
        # normal path first, then retry only an otherwise empty result using a
        # deterministic Otsu binary image.  Pose and all downstream gates
        # continue to use the original sub-pixel image measurements.
        if ids is None or len(ids) == 0:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            fallback_corners, fallback_ids, fallback_rejected = self._detect(
                binary
            )
            if fallback_ids is not None and len(fallback_ids):
                corners, ids, rejected = (
                    fallback_corners,
                    fallback_ids,
                    fallback_rejected,
                )
        rejected_count = len(rejected)
        if ids is None or len(ids) == 0:
            self.innovation_gate.note_missed_detection(stamp_s)
            self._publish_rejected("marker_not_found", max(1, rejected_count))
            self._publish_debug(frame, corners, ids, message, "SEARCHING", None)
            return
        board_observations: dict[int, np.ndarray] = {}
        if self.board_layout is not None:
            board_observations, board_rejections = self._refine_board_observations(
                gray, corners, ids
            )
            rejected_count += board_rejections
            estimate, reason = estimate_board_pose(
                self.board_layout,
                board_observations,
                self.camera_matrix,
                self.distortion,
                self.max_reprojection_error_px,
                self.min_facing_cosine,
                self.min_ambiguity_ratio,
                self.minimum_board_markers,
                self.board_outlier_mad_scale,
            )
            if estimate is None:
                self._publish_rejected(reason, max(1, rejected_count))
                self._publish_debug(frame, corners, ids, message, reason, None)
                return
            rejected_count += len(estimate.rejected_marker_ids)
            center_projection, _ = cv2.projectPoints(
                np.zeros((1, 3), dtype=np.float64),
                estimate.rvec,
                estimate.tvec,
                self.camera_matrix,
                self.distortion,
            )
            center = center_projection.reshape(2)
            depth_corners = {
                marker_id: board_observations[marker_id]
                for marker_id in estimate.inlier_marker_ids
            }
            perimeter = sum(
                float(cv2.arcLength(value.astype(np.float32), True))
                for value in depth_corners.values()
            )
        else:
            matches = np.where(ids.flatten() == self.marker_id)[0]
            if matches.size == 0:
                self.innovation_gate.note_missed_detection(stamp_s)
                self._publish_rejected(
                    "configured_marker_id_not_found",
                    max(1, rejected_count + len(ids)),
                )
                self._publish_debug(frame, corners, ids, message, "WRONG ID", None)
                return
            try:
                selected = self._refine_corners(gray, corners[int(matches[0])])
            except cv2.error:
                self._publish_rejected(
                    "subpixel_refinement_failed", max(1, rejected_count)
                )
                return
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
            center = np.mean(selected, axis=0)
            depth_corners = {self.marker_id: selected}
            perimeter = float(cv2.arcLength(selected.astype(np.float32), True))

        offset = Point()
        offset.x = float((center[0] - 0.5 * frame.shape[1]) / (0.5 * frame.shape[1]))
        offset.y = float((center[1] - 0.5 * frame.shape[0]) / (0.5 * frame.shape[0]))
        self.offset_publisher.publish(offset)
        self.reprojection_publisher.publish(
            Float32(data=float(estimate.reprojection_error_px))
        )

        depth_residual = None
        depth_tolerance = self.depth_absolute_tolerance_m
        synchronized_depth = None
        if self.depth_history:
            depth_stamp, candidate_depth = min(
                self.depth_history, key=lambda sample: abs(stamp_s - sample[0])
            )
            if abs(stamp_s - depth_stamp) <= self.depth_sync_tolerance_s:
                synchronized_depth = candidate_depth
        depth_observations_m: list[tuple[float, float]] = []
        if synchronized_depth is not None:
            for marker_id, marker_corners in depth_corners.items():
                depth_sample = robust_marker_depth(
                    synchronized_depth,
                    marker_corners,
                    frame.shape[:2],
                    self.depth_min_m,
                    self.depth_max_m,
                )
                if depth_sample is None:
                    continue
                predicted_depth = float(estimate.tvec[2])
                if self.board_layout is not None:
                    predicted_depth = float(
                        marker_center_in_camera(
                            self.board_layout, estimate, marker_id
                        )[2]
                    )
                depth_observations_m.append(
                    (depth_sample.median_m, predicted_depth)
                )
        depth_decision = evaluate_marker_depth_consistency(
            depth_observations_m,
            self.depth_absolute_tolerance_m,
            self.depth_relative_tolerance,
        )
        if depth_decision.had_sample and not depth_decision.accepted:
            self._publish_rejected(
                "depth_pose_inconsistent", max(1, rejected_count)
            )
            self._publish_debug(
                frame, corners, ids, message, "DEPTH GATE", estimate
            )
            return
        if depth_decision.accepted:
            depth_residual = depth_decision.residual_m
            depth_tolerance = depth_decision.tolerance_m
        elif self.depth_required:
            reason = "missing_synchronized_marker_depth"
            self._publish_rejected(reason, max(1, rejected_count))
            self._publish_debug(frame, corners, ids, message, "DEPTH GATE", estimate)
            return

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
        if not self._fresh_image(stamp_s):
            self._publish_rejected("stale_after_pose_processing", 1)
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
        estimate: Optional[SquarePoseEstimate | BoardPoseEstimate],
    ) -> None:
        if not self.publish_debug:
            return
        compressed_subscribers = self.debug_publisher.get_subscription_count()
        raw_subscribers = (
            self.debug_raw_publisher.get_subscription_count()
            if self.debug_raw_publisher is not None
            else 0
        )
        if compressed_subscribers == 0 and raw_subscribers == 0:
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
        if raw_subscribers:
            raw_frame = np.ascontiguousarray(frame, dtype=np.uint8)
            raw_debug = Image()
            raw_debug.header = message.header
            raw_debug.height = int(raw_frame.shape[0])
            raw_debug.width = int(raw_frame.shape[1])
            raw_debug.encoding = "bgr8"
            raw_debug.is_bigendian = 0
            raw_debug.step = int(raw_frame.shape[1] * 3)
            raw_debug.data = raw_frame.tobytes()
            self.debug_raw_publisher.publish(raw_debug)

        success, encoded = (
            cv2.imencode(".jpg", frame)
            if compressed_subscribers
            else (False, None)
        )
        if success and encoded is not None:
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
