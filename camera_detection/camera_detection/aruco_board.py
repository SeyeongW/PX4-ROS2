#!/usr/bin/env python3
"""ROS-independent geometry and robust pose fitting for an ArUco landing board."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml


# A single square away from the board origin cannot constrain the landing
# centre robustly in a near-fronto-parallel view: a small IPPE tilt error is
# multiplied by the marker-to-origin lever arm.  Only a marker whose declared
# centre is numerically at the board origin may therefore be used alone.
_BOARD_ORIGIN_TOLERANCE_M = 1.0e-6


@dataclass(frozen=True)
class BoardMarker:
    """Describe one marker pose and physical size in the landing-board frame."""

    marker_id: int
    size_m: float
    position_board_m: tuple[float, float, float]
    yaw_board_rad: float = 0.0


@dataclass(frozen=True)
class BoardLayout:
    """Describe a uniquely identified collection of coplanar landing markers."""

    dictionary: str
    frame_id: str
    markers: tuple[BoardMarker, ...]

    @property
    def markers_by_id(self) -> dict[int, BoardMarker]:
        """Return markers indexed by dictionary ID."""
        return {marker.marker_id: marker for marker in self.markers}


@dataclass(frozen=True)
class BoardPoseEstimate:
    """Represent the refined camera-from-board pose and its fit diagnostics."""

    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    ambiguity_ratio: float
    facing_cosine: float
    inlier_marker_ids: tuple[int, ...]
    rejected_marker_ids: tuple[int, ...]
    marker_reprojection_errors_px: tuple[tuple[int, float], ...]
    marker_weights: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _MarkerPoseCandidate:
    marker_id: int
    rotation_camera_from_board: np.ndarray
    translation_camera_from_board_m: np.ndarray
    local_reprojection_error_px: float
    ambiguity_ratio: float
    facing_cosine: float
    perimeter_px: float


def load_board_layout(path: str | Path) -> BoardLayout:
    """Load and strictly validate a landing-board layout YAML file."""
    layout_path = Path(path).expanduser()
    data = yaml.safe_load(layout_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("board layout must be a YAML mapping")
    dictionary = data.get("dictionary")
    frame_id = data.get("frame_id", "landing_board")
    raw_markers = data.get("markers")
    if not isinstance(dictionary, str) or not dictionary:
        raise ValueError("board dictionary must be a non-empty string")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("board frame_id must be a non-empty string")
    if not isinstance(raw_markers, list) or not raw_markers:
        raise ValueError("board markers must be a non-empty list")

    markers: list[BoardMarker] = []
    marker_ids: set[int] = set()
    for index, raw_marker in enumerate(raw_markers):
        if not isinstance(raw_marker, dict):
            raise ValueError(f"board marker {index} must be a mapping")
        try:
            marker_id = int(raw_marker["id"])
            size_m = float(raw_marker["size_m"])
            position = tuple(float(value) for value in raw_marker["position_m"])
            yaw_rad = float(raw_marker.get("yaw_rad", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid board marker {index}: {exc}") from exc
        if marker_id < 0 or marker_id in marker_ids:
            raise ValueError(f"duplicate or negative board marker id: {marker_id}")
        if len(position) != 3 or not np.all(np.isfinite(position)):
            raise ValueError(f"marker {marker_id} position_m must contain 3 finite values")
        if not math.isfinite(size_m) or size_m <= 0.0:
            raise ValueError(f"marker {marker_id} size_m must be finite and positive")
        if not math.isfinite(yaw_rad):
            raise ValueError(f"marker {marker_id} yaw_rad must be finite")
        markers.append(BoardMarker(marker_id, size_m, position, yaw_rad))
        marker_ids.add(marker_id)
    return BoardLayout(dictionary, frame_id, tuple(markers))


def marker_object_points(marker_size_m: float) -> np.ndarray:
    """Return square points in OpenCV IPPE TL, TR, BR, BL order."""
    size = float(marker_size_m)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("marker_size_m must be finite and positive")
    half = 0.5 * size
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def marker_object_points_in_board(marker: BoardMarker) -> np.ndarray:
    """Return one marker's four object points expressed in the board frame."""
    cosine = math.cos(marker.yaw_board_rad)
    sine = math.sin(marker.yaw_board_rad)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    position = np.asarray(marker.position_board_m, dtype=np.float64)
    return (rotation @ marker_object_points(marker.size_m).T).T + position


def _reprojection_rmse_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    residual = projected.reshape(-1, 2) - np.asarray(image_points).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _marker_rotation_in_board(marker: BoardMarker) -> np.ndarray:
    cosine = math.cos(marker.yaw_board_rad)
    sine = math.sin(marker.yaw_board_rad)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _single_marker_candidate(
    marker: BoardMarker,
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    max_reprojection_error_px: float,
    min_facing_cosine: float,
    min_ambiguity_ratio: float,
) -> Optional[_MarkerPoseCandidate]:
    image_points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(image_points)):
        return None
    object_points = marker_object_points(marker.size_m)
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None
    if len(result) < 3 or not result[0]:
        return None

    solutions: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for raw_rvec, raw_tvec in zip(result[1], result[2]):
        rvec = np.asarray(raw_rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(raw_tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        rotation_camera_from_marker, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation_camera_from_marker @ object_points.T).T + tvec
        distance = float(np.linalg.norm(tvec))
        if np.min(camera_points[:, 2]) <= 0.01 or distance <= 1e-6:
            continue
        facing = float(
            np.dot(rotation_camera_from_marker[:, 2], -tvec / distance)
        )
        if facing < min_facing_cosine:
            continue
        error = _reprojection_rmse_px(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        if math.isfinite(error):
            solutions.append((error, facing, rotation_camera_from_marker, tvec))
    # OpenCV 4.5's IPPE_SQUARE can return a grossly tilted solution for an
    # exactly fronto-parallel square even with noise-free corners.  A direct
    # iterative solve supplies a board-safe seed for this common nadir-camera
    # geometry; IPPE candidates remain available for ambiguity comparison.
    try:
        iterative_success, iterative_rvec, iterative_tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        iterative_success = False
    if iterative_success:
        iterative_rvec = np.asarray(iterative_rvec, dtype=np.float64).reshape(3)
        iterative_tvec = np.asarray(iterative_tvec, dtype=np.float64).reshape(3)
        iterative_rotation, _ = cv2.Rodrigues(iterative_rvec)
        iterative_points = (
            iterative_rotation @ object_points.T
        ).T + iterative_tvec
        iterative_distance = float(np.linalg.norm(iterative_tvec))
        if (
            np.all(np.isfinite(iterative_rvec))
            and np.all(np.isfinite(iterative_tvec))
            and np.min(iterative_points[:, 2]) > 0.01
            and iterative_distance > 1.0e-6
        ):
            iterative_facing = float(
                np.dot(
                    iterative_rotation[:, 2],
                    -iterative_tvec / iterative_distance,
                )
            )
            iterative_error = _reprojection_rmse_px(
                object_points,
                image_points,
                iterative_rvec,
                iterative_tvec,
                camera_matrix,
                distortion,
            )
            if (
                iterative_facing >= min_facing_cosine
                and math.isfinite(iterative_error)
            ):
                solutions.append(
                    (
                        iterative_error,
                        iterative_facing,
                        iterative_rotation,
                        iterative_tvec,
                    )
                )
    if not solutions:
        return None
    solutions.sort(key=lambda item: item[0])
    distinct_solutions = []
    for solution in solutions:
        if any(
            np.linalg.norm(solution[2] - existing[2]) < 1.0e-3
            and np.linalg.norm(solution[3] - existing[3]) < 1.0e-3
            for existing in distinct_solutions
        ):
            continue
        distinct_solutions.append(solution)
    solutions = distinct_solutions
    best_error, facing, rotation_camera_from_marker, translation_camera_from_marker = (
        solutions[0]
    )
    if best_error > max_reprojection_error_px:
        return None
    ambiguity_ratio = math.inf
    if len(solutions) > 1:
        ambiguity_ratio = (solutions[1][0] + 1e-6) / (best_error + 1e-6)
        if ambiguity_ratio < min_ambiguity_ratio:
            return None

    rotation_board_from_marker = _marker_rotation_in_board(marker)
    rotation_camera_from_board = (
        rotation_camera_from_marker @ rotation_board_from_marker.T
    )
    marker_position_board = np.asarray(marker.position_board_m, dtype=np.float64)
    translation_camera_from_board = (
        translation_camera_from_marker
        - rotation_camera_from_board @ marker_position_board
    )
    perimeter = float(cv2.arcLength(image_points.astype(np.float32), True))
    return _MarkerPoseCandidate(
        marker.marker_id,
        rotation_camera_from_board,
        translation_camera_from_board,
        best_error,
        ambiguity_ratio,
        facing,
        perimeter,
    )


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion[:] = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quaternion[:] = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quaternion[:] = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quaternion[:] = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def weighted_board_pose_seed(
    rotations_camera_from_board: Sequence[np.ndarray],
    translations_camera_from_board_m: Sequence[np.ndarray],
    weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse board-pose candidates with a Markley rotation and weighted position."""
    if not rotations_camera_from_board or (
        len(rotations_camera_from_board) != len(translations_camera_from_board_m)
        or len(rotations_camera_from_board) != len(weights)
    ):
        raise ValueError("rotations, translations and weights must have equal nonzero length")
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0.0):
        raise ValueError("weights must be finite and positive")
    weight_array /= np.sum(weight_array)

    accumulator = np.zeros((4, 4), dtype=np.float64)
    for weight, rotation in zip(weight_array, rotations_camera_from_board):
        quaternion = _rotation_matrix_to_quaternion_wxyz(rotation)
        accumulator += weight * np.outer(quaternion, quaternion)
    _, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, -1]
    quaternion /= np.linalg.norm(quaternion)
    rotation = _quaternion_wxyz_to_rotation_matrix(quaternion)

    translations = np.asarray(
        translations_camera_from_board_m, dtype=np.float64
    ).reshape(-1, 3)
    if not np.all(np.isfinite(translations)):
        raise ValueError("translations must be finite")
    translation = np.sum(translations * weight_array[:, None], axis=0)
    return rotation, translation


def _marker_errors(
    layout: BoardLayout,
    observations: Mapping[int, np.ndarray],
    rotation_camera_from_board: np.ndarray,
    translation_camera_from_board_m: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[int, float]:
    rvec, _ = cv2.Rodrigues(rotation_camera_from_board)
    errors: dict[int, float] = {}
    markers_by_id = layout.markers_by_id
    for marker_id, corners in observations.items():
        marker = markers_by_id.get(marker_id)
        if marker is None:
            continue
        errors[marker_id] = _reprojection_rmse_px(
            marker_object_points_in_board(marker),
            corners,
            rvec,
            translation_camera_from_board_m,
            camera_matrix,
            distortion,
        )
    return errors


def _robust_error_limit(
    errors: Sequence[float], absolute_limit_px: float, mad_scale: float
) -> float:
    values = np.asarray(errors, dtype=np.float64)
    if values.size < 3:
        return float(absolute_limit_px)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scaled_mad_limit = median + mad_scale * max(1.4826 * mad, 0.25)
    relative_limit = 2.0 * median + 0.5
    return min(float(absolute_limit_px), max(scaled_mad_limit, relative_limit))


def _candidate_weights(
    candidates_by_id: Mapping[int, _MarkerPoseCandidate],
    seed_errors: Mapping[int, float],
    marker_ids: Sequence[int],
) -> np.ndarray:
    weights = []
    for marker_id in marker_ids:
        candidate = candidates_by_id[marker_id]
        variance_px2 = (
            0.25**2
            + candidate.local_reprojection_error_px**2
            + seed_errors[marker_id] ** 2
        )
        weights.append(max(candidate.perimeter_px, 1.0) / variance_px2)
    result = np.asarray(weights, dtype=np.float64)
    return result / np.sum(result)


def _observation_weights(
    observations: Mapping[int, np.ndarray],
    reprojection_errors: Mapping[int, float],
    marker_ids: Sequence[int],
) -> np.ndarray:
    """Weight every joint-PnP inlier, including non-seed observations."""
    weights = []
    for marker_id in marker_ids:
        corners = np.asarray(
            observations[marker_id], dtype=np.float32
        ).reshape(4, 2)
        perimeter_px = float(cv2.arcLength(corners, True))
        error_px = float(reprojection_errors[marker_id])
        variance_px2 = 0.25**2 + error_px**2
        weights.append(max(perimeter_px, 1.0) / variance_px2)
    result = np.asarray(weights, dtype=np.float64)
    return result / np.sum(result)


def _single_marker_is_at_board_origin(
    layout: BoardLayout, marker_ids: Sequence[int]
) -> bool:
    """Return true unless a sole inlier has a board-origin lever arm."""
    if len(marker_ids) != 1:
        return True
    marker = layout.markers_by_id[int(marker_ids[0])]
    offset_m = np.asarray(marker.position_board_m, dtype=np.float64)
    return float(np.linalg.norm(offset_m)) <= _BOARD_ORIGIN_TOLERANCE_M


def _refine_joint_pose(
    layout: BoardLayout,
    observations: Mapping[int, np.ndarray],
    marker_ids: Sequence[int],
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    markers_by_id = layout.markers_by_id
    object_points = np.concatenate(
        [marker_object_points_in_board(markers_by_id[item]) for item in marker_ids]
    ).astype(np.float64)
    image_points = np.concatenate(
        [np.asarray(observations[item], dtype=np.float64).reshape(4, 2)
         for item in marker_ids]
    )
    rvec, _ = cv2.Rodrigues(initial_rotation)
    tvec = np.asarray(initial_translation, dtype=np.float64).reshape(3, 1)
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rvec.reshape(3, 1),
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
    except cv2.error:
        return None
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        return None
    return rotation, translation


def estimate_board_pose(
    layout: BoardLayout,
    observations_px: Mapping[int, np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    max_reprojection_error_px: float = 2.5,
    min_facing_cosine: float = 0.15,
    min_ambiguity_ratio: float = 1.15,
    minimum_inlier_markers: int = 1,
    outlier_mad_scale: float = 3.5,
) -> tuple[Optional[BoardPoseEstimate], str]:
    """Fit the board center from mixed-size markers, rejecting marker outliers."""
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion_array = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    if (
        not np.all(np.isfinite(camera))
        or camera[0, 0] <= 0.0
        or camera[1, 1] <= 0.0
    ):
        return None, "invalid_intrinsics"
    if not math.isfinite(max_reprojection_error_px) \
            or max_reprojection_error_px <= 0.0:
        return None, "invalid_reprojection_limit"
    if minimum_inlier_markers < 1:
        return None, "invalid_minimum_inliers"

    markers_by_id = layout.markers_by_id
    observations: dict[int, np.ndarray] = {}
    for raw_marker_id, raw_corners in observations_px.items():
        marker_id = int(raw_marker_id)
        if marker_id not in markers_by_id:
            continue
        try:
            corners = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)
        except ValueError:
            continue
        if np.all(np.isfinite(corners)):
            observations[marker_id] = corners
    if not observations:
        return None, "configured_board_markers_not_found"
    observed_ids = sorted(observations)
    if not _single_marker_is_at_board_origin(layout, observed_ids):
        return None, "single_offset_marker_underconstrained"

    candidates_by_id: dict[int, _MarkerPoseCandidate] = {}
    for marker_id, corners in observations.items():
        candidate = _single_marker_candidate(
            markers_by_id[marker_id],
            corners,
            camera,
            distortion_array,
            max_reprojection_error_px,
            min_facing_cosine,
            min_ambiguity_ratio,
        )
        if candidate is not None:
            candidates_by_id[marker_id] = candidate
    # Per-marker IPPE candidates seed the board pose.  Once one physical seed
    # exists, the joint board geometry can disambiguate other observed squares;
    # those observations must not be discarded merely because their individual
    # fronto-parallel pose was ambiguous.
    if not candidates_by_id:
        return None, "insufficient_physical_marker_candidates"

    seed_options = []
    for candidate in candidates_by_id.values():
        errors = _marker_errors(
            layout,
            observations,
            candidate.rotation_camera_from_board,
            candidate.translation_camera_from_board_m,
            camera,
            distortion_array,
        )
        finite_errors = [value for value in errors.values() if math.isfinite(value)]
        support = sum(
            value <= max_reprojection_error_px for value in finite_errors
        )
        clipped_cost = float(
            np.median(np.minimum(finite_errors, 4.0 * max_reprojection_error_px))
        )
        seed_options.append((-support, clipped_cost, candidate.marker_id, errors))
    seed_options.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, seed_errors = seed_options[0]
    error_limit = _robust_error_limit(
        list(seed_errors.values()), max_reprojection_error_px, outlier_mad_scale
    )
    inlier_ids = sorted(
        marker_id
        for marker_id, error in seed_errors.items()
        if error <= error_limit
    )
    if len(inlier_ids) < minimum_inlier_markers:
        return None, "insufficient_reprojection_inliers"
    if not _single_marker_is_at_board_origin(layout, inlier_ids):
        return None, "single_offset_marker_underconstrained"

    seed_candidate_ids = [
        marker_id for marker_id in inlier_ids
        if marker_id in candidates_by_id
    ]
    if not seed_candidate_ids:
        return None, "insufficient_physical_marker_candidates"
    seed_weights = _candidate_weights(
        candidates_by_id, seed_errors, seed_candidate_ids
    )
    initial_rotation, initial_translation = weighted_board_pose_seed(
        [
            candidates_by_id[item].rotation_camera_from_board
            for item in seed_candidate_ids
        ],
        [
            candidates_by_id[item].translation_camera_from_board_m
            for item in seed_candidate_ids
        ],
        seed_weights,
    )
    refined = _refine_joint_pose(
        layout,
        observations,
        inlier_ids,
        initial_rotation,
        initial_translation,
        camera,
        distortion_array,
    )
    if refined is None:
        return None, "board_pnp_refinement_failed"
    rotation, translation = refined
    final_errors = _marker_errors(
        layout, observations, rotation, translation, camera, distortion_array
    )
    final_limit = _robust_error_limit(
        [final_errors[item] for item in inlier_ids],
        max_reprojection_error_px,
        outlier_mad_scale,
    )
    final_inlier_ids = [
        item for item in inlier_ids if final_errors[item] <= final_limit
    ]
    if len(final_inlier_ids) < minimum_inlier_markers:
        return None, "insufficient_refined_inliers"
    if not _single_marker_is_at_board_origin(layout, final_inlier_ids):
        return None, "single_offset_marker_underconstrained"
    if final_inlier_ids != inlier_ids:
        refined = _refine_joint_pose(
            layout,
            observations,
            final_inlier_ids,
            rotation,
            translation,
            camera,
            distortion_array,
        )
        if refined is None:
            return None, "board_pnp_refinement_failed"
        rotation, translation = refined
        final_errors = _marker_errors(
            layout, observations, rotation, translation, camera, distortion_array
        )

    physical_final_ids = [
        item for item in final_inlier_ids if item in candidates_by_id
    ]
    if not physical_final_ids:
        return None, "insufficient_physical_marker_candidates"
    weights = _observation_weights(
        observations, final_errors, final_inlier_ids
    )

    rvec, _ = cv2.Rodrigues(rotation)
    used_object_points = np.concatenate(
        [marker_object_points_in_board(markers_by_id[item])
         for item in final_inlier_ids]
    )
    camera_points = (rotation @ used_object_points.T).T + translation
    distance = float(np.linalg.norm(translation))
    if np.min(camera_points[:, 2]) <= 0.01 or distance <= 1e-6:
        return None, "board_behind_camera"
    facing = float(np.dot(rotation[:, 2], -translation / distance))
    if facing < min_facing_cosine:
        return None, "board_back_face"

    used_image_points = np.concatenate(
        [observations[item] for item in final_inlier_ids]
    )
    reprojection_error = _reprojection_rmse_px(
        used_object_points,
        used_image_points,
        rvec,
        translation,
        camera,
        distortion_array,
    )
    if not math.isfinite(reprojection_error) \
            or reprojection_error > max_reprojection_error_px:
        return None, "board_reprojection_error"

    observed_configured_ids = set(observations)
    rejected_ids = tuple(sorted(observed_configured_ids - set(final_inlier_ids)))
    ambiguity = min(
        candidates_by_id[item].ambiguity_ratio for item in physical_final_ids
    )
    normalized_weights = weights / np.sum(weights)
    return (
        BoardPoseEstimate(
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
            tvec=translation,
            reprojection_error_px=reprojection_error,
            ambiguity_ratio=ambiguity,
            facing_cosine=facing,
            inlier_marker_ids=tuple(final_inlier_ids),
            rejected_marker_ids=rejected_ids,
            marker_reprojection_errors_px=tuple(
                (item, float(final_errors[item]))
                for item in sorted(final_errors)
            ),
            marker_weights=tuple(
                (item, float(weight))
                for item, weight in zip(final_inlier_ids, normalized_weights)
            ),
        ),
        "accepted",
    )


def marker_center_in_camera(
    layout: BoardLayout,
    estimate: BoardPoseEstimate,
    marker_id: int,
) -> np.ndarray:
    """Transform one configured marker center into the camera frame."""
    marker = layout.markers_by_id.get(int(marker_id))
    if marker is None:
        raise KeyError(f"unknown board marker id: {marker_id}")
    rotation, _ = cv2.Rodrigues(np.asarray(estimate.rvec, dtype=np.float64))
    return (
        rotation @ np.asarray(marker.position_board_m, dtype=np.float64)
        + np.asarray(estimate.tvec, dtype=np.float64)
    )
