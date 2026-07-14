import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from camera_detection.aruco_pose_node import (
    DetectionStreak,
    PoseInnovationGate,
    covariance_and_quality,
    estimate_square_marker_pose,
    marker_object_points,
    robust_marker_depth,
)


ROOT = Path(__file__).resolve().parents[2]


def _synthetic_observation():
    camera = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    expected_rvec = np.array([math.pi - 0.2, 0.10, 0.05])
    expected_tvec = np.array([0.20, -0.10, 5.0])
    corners, _ = cv2.projectPoints(
        marker_object_points(1.0), expected_rvec, expected_tvec,
        camera, distortion,
    )
    return camera, distortion, corners.reshape(4, 2), expected_tvec


def test_ippe_pose_recovers_metric_translation_and_reprojection():
    camera, distortion, corners, expected_tvec = _synthetic_observation()
    estimate, reason = estimate_square_marker_pose(
        corners, camera, distortion, 1.0
    )
    assert reason == "accepted"
    assert estimate is not None
    assert np.allclose(estimate.tvec, expected_tvec, atol=1e-5)
    assert estimate.reprojection_error_px < 1e-3
    assert estimate.facing_cosine > 0.9


def test_ippe_rejects_marker_back_face():
    camera = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros(5)
    corners, _ = cv2.projectPoints(
        marker_object_points(1.0), np.zeros(3), np.array([0.0, 0.0, 5.0]),
        camera, distortion,
    )
    estimate, reason = estimate_square_marker_pose(
        corners.reshape(4, 2), camera, distortion, 1.0
    )
    assert estimate is None
    assert reason == "flipped_or_behind_camera"


def test_ippe_rejects_equal_error_ambiguity(monkeypatch):
    camera, distortion, _, _ = _synthetic_observation()
    rvec = np.array([[math.pi], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [5.0]])
    corners, _ = cv2.projectPoints(
        marker_object_points(1.0), rvec, tvec, camera, distortion
    )
    corners = corners.reshape(4, 2)

    def ambiguous_solver(*args, **kwargs):
        return 2, (rvec.copy(), rvec.copy()), (tvec.copy(), tvec.copy()), None

    monkeypatch.setattr(cv2, "solvePnPGeneric", ambiguous_solver)
    estimate, reason = estimate_square_marker_pose(
        corners, camera, distortion, 1.0, min_ambiguity_ratio=1.15
    )
    assert estimate is None
    assert reason == "ippe_ambiguity"


def test_depth_gate_uses_scaled_polygon_and_robust_median():
    depth = np.full((240, 320), np.nan, dtype=np.float32)
    rgb_corners = np.array(
        [[200, 160], [440, 160], [440, 320], [200, 320]], dtype=np.float32
    )
    depth[82:159, 102:219] = 5.0
    depth[100, 120] = 0.0
    depth[110, 130] = 75.0
    sample = robust_marker_depth(
        depth, rgb_corners, (480, 640), 0.2, 20.0
    )
    assert sample is not None
    assert sample.median_m == pytest.approx(5.0)
    assert sample.mad_m == pytest.approx(0.0)
    assert sample.valid_count > 1000


def test_covariance_quality_degrades_with_reprojection_and_depth_error():
    camera, distortion, corners, _ = _synthetic_observation()
    estimate, _ = estimate_square_marker_pose(corners, camera, distortion, 1.0)
    assert estimate is not None
    covariance_good, quality_good = covariance_and_quality(
        estimate, camera, 500.0, 0.01, 2.5, 0.5
    )
    degraded = type(estimate)(
        estimate.rvec, estimate.tvec, 2.0, 1.2, estimate.facing_cosine
    )
    covariance_bad, quality_bad = covariance_and_quality(
        degraded, camera, 40.0, 0.45, 2.5, 0.5
    )
    assert quality_good > quality_bad
    assert covariance_good[0, 0] < covariance_bad[0, 0]
    assert np.all(np.diag(covariance_good) > 0.0)


def test_timestamp_innovation_gate_rejects_outlier_and_resets_after_dropout():
    gate = PoseInnovationGate(
        threshold_squared=16.27,
        process_variance_m2_s2=4.0,
        reset_after_s=1.0,
    )
    variance = [0.01, 0.01, 0.02]
    assert gate.update(1.0, [0.0, 0.0, 5.0], variance).accepted
    assert gate.update(1.1, [0.30, 0.0, 5.0], variance).accepted
    outlier = gate.update(1.2, [30.0, -20.0, 2.0], variance)
    assert not outlier.accepted
    assert outlier.mahalanobis_squared > gate.threshold_squared
    assert gate.update(2.5, [5.0, 5.0, 5.0], variance).accepted


def test_detection_streak_requires_contiguous_frames_and_same_id():
    streak = DetectionStreak(minimum_frames=3, maximum_gap_s=0.2)
    assert not streak.update(True, 0, 1.0)
    assert not streak.update(True, 0, 1.1)
    assert streak.update(True, 0, 1.2)
    assert not streak.update(True, 1, 1.3)
    assert not streak.update(False, None, 1.4)
    assert not streak.update(True, 0, 2.0)


def test_front_and_down_configs_have_distinct_inputs_outputs_and_real_depth():
    configs = []
    for name in ("aruco_front.yaml", "aruco_down.yaml"):
        data = yaml.safe_load((ROOT / "camera_detection/config" / name).read_text())
        configs.append(next(iter(data.values()))["ros__parameters"])
    front, down = configs
    assert front["image_topic"] != down["image_topic"]
    assert front["depth_topic"] == "/front_depth/image"
    assert down["depth_topic"] == "/down_depth/image"
    assert front["pose_topic"] != down["pose_topic"]
    assert front["optical_frame_id"] != down["optical_frame_id"]
    assert front["depth_required"] and down["depth_required"]


def test_perception_node_never_imports_gazebo_ground_truth():
    source = (ROOT / "camera_detection/camera_detection/aruco_pose_node.py").read_text()
    forbidden = ("dynamic_pose", "gz.transport", "/trailer/pose", "ground_truth")
    assert not any(token in source for token in forbidden)
