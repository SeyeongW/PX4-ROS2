"""ROS-independent helpers for timestamped ArUco pose measurements.

The helpers deliberately keep capture time separate from estimator time.  The
capture stamp is used only to reject duplicate camera frames and for status;
the estimator continues to use ``time.monotonic()`` in the ROS node.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np


_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass
class CaptureStampGuard:
    """Accept each nonzero ROS capture timestamp at most once in sequence."""

    last_stamp_ns: int | None = None
    duplicate_count: int = 0
    capture_stamp_s: float | None = None

    def register(self, seconds: int, nanoseconds: int) -> bool:
        """Register a capture stamp, returning false for a duplicate."""

        if (
            isinstance(seconds, bool)
            or isinstance(nanoseconds, bool)
            or not isinstance(seconds, Integral)
            or not isinstance(nanoseconds, Integral)
        ):
            raise ValueError("capture timestamp fields must be integers")
        seconds = int(seconds)
        nanoseconds = int(nanoseconds)
        if seconds < 0 or not 0 <= nanoseconds < _NANOSECONDS_PER_SECOND:
            raise ValueError("capture timestamp fields are outside ROS Time bounds")
        stamp_ns = seconds * _NANOSECONDS_PER_SECOND + nanoseconds
        if stamp_ns <= 0:
            raise ValueError("capture timestamp must be nonzero")
        if stamp_ns == self.last_stamp_ns:
            self.duplicate_count += 1
            return False
        self.last_stamp_ns = stamp_ns
        self.capture_stamp_s = seconds + nanoseconds / _NANOSECONDS_PER_SECOND
        return True


def extract_pose_covariance_6x6(values: Sequence[float]) -> np.ndarray:
    """Validate and return a ``PoseWithCovariance`` 6-by-6 covariance."""

    try:
        covariance = np.asarray(values, dtype=float).reshape(6, 6)
    except (TypeError, ValueError) as exc:
        raise ValueError("pose covariance must contain exactly 36 values") from exc
    if not np.all(np.isfinite(covariance)):
        raise ValueError("pose covariance must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("pose covariance must be symmetric")
    if np.any(np.diag(covariance[:3, :3]) < 0.0):
        raise ValueError("pose position variances must be nonnegative")
    if covariance[5, 5] < 0.0:
        raise ValueError("pose yaw variance must be nonnegative")
    return covariance.copy()


def flu_to_enu_rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Return body-FLU to world-ENU rotation for a ROS quaternion.

    ``/mavros/local_position/pose`` expresses the body orientation in ENU as
    the ROS ``(x, y, z, w)`` quaternion used here.  Position and covariance
    transformations take the same matrix snapshot.
    """

    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("ROS quaternion must be a finite xyzw four-vector")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("ROS quaternion norm must be positive")
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def build_estimator_covariance(
    covariance_6x6: Sequence[float],
    rotation_enu_from_flu: Sequence[Sequence[float]],
    distance_m: float,
    camera: str,
) -> np.ndarray:
    """Build the estimator xyz/yaw covariance using measured values first.

    The historical distance model is retained solely as a diagonal lower
    bound.  It never replaces an invalid or larger measured covariance.
    """

    covariance = extract_pose_covariance_6x6(covariance_6x6)
    rotation = np.asarray(rotation_enu_from_flu, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("FLU-to-ENU rotation must be a finite 3x3 matrix")
    distance = float(distance_m)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("marker distance must be finite and nonnegative")
    distance = max(0.2, distance)

    camera_name = str(camera).strip().lower()
    if camera_name == "front":
        position_variance_floor = (0.015 + 0.006 * distance) ** 2
        yaw_variance_floor = math.radians(8.0) ** 2
    elif camera_name == "down":
        position_variance_floor = (0.01 + 0.004 * distance) ** 2
        yaw_variance_floor = math.radians(5.0) ** 2
    else:
        raise ValueError("camera must be front or down")

    position_enu = rotation @ covariance[:3, :3] @ rotation.T
    position_enu = 0.5 * (position_enu + position_enu.T)
    diagonal = np.diag_indices(3)
    position_enu[diagonal] = np.maximum(
        position_enu[diagonal], position_variance_floor
    )

    estimator_covariance = np.zeros((4, 4), dtype=float)
    estimator_covariance[:3, :3] = position_enu
    estimator_covariance[3, 3] = max(
        float(covariance[5, 5]), yaw_variance_floor
    )
    return estimator_covariance
