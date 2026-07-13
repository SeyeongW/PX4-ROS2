"""Paired execution metrics for PX4 L1-style versus B-spline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class MetricsContractError(ValueError):
    """Raised when two runs are not scientifically comparable."""


def route_sha256(waypoints_enu_m: Sequence[Sequence[float]]) -> str:
    """Hash a route after stable millimetre quantisation."""

    points = np.asarray(waypoints_enu_m, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise MetricsContractError("route must be an N x 3 array with N >= 2")
    if not np.all(np.isfinite(points)):
        raise MetricsContractError("route contains non-finite coordinates")
    quantised = np.round(points, 3).tolist()
    payload = json.dumps(quantised, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExperimentContract:
    """Fields that must match before a method comparison is meaningful."""

    guidance_method: str
    map_revision: str
    occupancy_sha256: str
    route_sha256: str
    planner_run_id: str
    px4_git_revision: str
    scenario_seed: int
    hard_radius_m: float
    minimum_surface_clearance_m: float
    wind_profile_id: str
    payload_profile_id: str
    initial_position_enu_m: tuple[float, float, float]

    def assert_valid(self) -> None:
        if self.guidance_method not in ("px4_mc_l1_style", "bspline_mpc"):
            raise MetricsContractError(f"unknown guidance method {self.guidance_method!r}")
        for name in (
            "map_revision",
            "occupancy_sha256",
            "route_sha256",
            "planner_run_id",
            "px4_git_revision",
            "wind_profile_id",
            "payload_profile_id",
        ):
            if not str(getattr(self, name)).strip():
                raise MetricsContractError(f"{name} is required")
        if not math.isfinite(self.hard_radius_m) or self.hard_radius_m <= 0.0:
            raise MetricsContractError("hard_radius_m must be positive")
        if (
            not math.isfinite(self.minimum_surface_clearance_m)
            or self.minimum_surface_clearance_m < 10.0
        ):
            raise MetricsContractError(
                "minimum_surface_clearance_m must be at least 10 m"
            )
        initial = np.asarray(self.initial_position_enu_m, dtype=float)
        if initial.shape != (3,) or not np.all(np.isfinite(initial)):
            raise MetricsContractError("initial_position_enu_m must be finite")


@dataclass(frozen=True)
class ExecutionTrace:
    """Time-aligned vehicle samples in one common Gazebo-world ENU frame."""

    timestamp_s: np.ndarray
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    clearance_m: np.ndarray

    @classmethod
    def from_sequences(
        cls,
        timestamp_s: Sequence[float],
        position_enu_m: Sequence[Sequence[float]],
        velocity_enu_m_s: Sequence[Sequence[float]],
        clearance_m: Sequence[float],
    ) -> "ExecutionTrace":
        trace = cls(
            timestamp_s=np.asarray(timestamp_s, dtype=float),
            position_enu_m=np.asarray(position_enu_m, dtype=float),
            velocity_enu_m_s=np.asarray(velocity_enu_m_s, dtype=float),
            clearance_m=np.asarray(clearance_m, dtype=float),
        )
        trace.validate()
        return trace

    def validate(self) -> None:
        count = len(self.timestamp_s)
        if count < 4:
            raise MetricsContractError("execution trace needs at least four samples")
        if self.timestamp_s.shape != (count,):
            raise MetricsContractError("timestamps must be one-dimensional")
        if self.position_enu_m.shape != (count, 3):
            raise MetricsContractError("positions must have shape N x 3")
        if self.velocity_enu_m_s.shape != (count, 3):
            raise MetricsContractError("velocities must have shape N x 3")
        if self.clearance_m.shape != (count,):
            raise MetricsContractError("clearances must have shape N")
        arrays = (
            self.timestamp_s,
            self.position_enu_m,
            self.velocity_enu_m_s,
            self.clearance_m,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise MetricsContractError("execution trace contains non-finite values")
        if np.any(np.diff(self.timestamp_s) <= 0.0):
            raise MetricsContractError("timestamps must be strictly increasing")
        if np.any(self.clearance_m < 0.0):
            raise MetricsContractError("clearance cannot be negative")


@dataclass(frozen=True)
class GuidanceRunMetrics:
    guidance_method: str
    duration_s: float
    reference_length_m: float
    flown_length_m: float
    final_error_m: float
    completed: bool
    rms_cross_track_error_m: float
    p95_cross_track_error_m: float
    max_cross_track_error_m: float
    min_clearance_m: float
    hard_clearance_violations: int
    max_speed_m_s: float
    p95_speed_m_s: float
    max_acceleration_m_s2: float
    p95_acceleration_m_s2: float
    max_jerk_m_s3: float
    p95_jerk_m_s3: float
    backtracking_distance_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _project_to_polyline(
    points: np.ndarray, route: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    segment_starts = route[:-1]
    segment_vectors = route[1:] - route[:-1]
    segment_length_sq = np.sum(segment_vectors * segment_vectors, axis=1)
    if np.any(segment_length_sq <= 1e-12):
        raise MetricsContractError("reference route has duplicate adjacent points")
    segment_lengths = np.sqrt(segment_length_sq)
    prefix = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distances = np.empty(len(points), dtype=float)
    progress = np.empty(len(points), dtype=float)
    for sample_index, point in enumerate(points):
        relative = point - segment_starts
        fraction = np.clip(
            np.sum(relative * segment_vectors, axis=1) / segment_length_sq,
            0.0,
            1.0,
        )
        projections = segment_starts + fraction[:, None] * segment_vectors
        errors = np.linalg.norm(projections - point, axis=1)
        best = int(np.argmin(errors))
        distances[sample_index] = errors[best]
        progress[sample_index] = prefix[best] + fraction[best] * segment_lengths[best]
    return distances, progress


def compute_guidance_metrics(
    contract: ExperimentContract,
    trace: ExecutionTrace,
    reference_route_enu_m: Sequence[Sequence[float]],
    *,
    completion_radius_m: float = 2.0,
) -> GuidanceRunMetrics:
    """Compute the same geometric/dynamic metrics for either guidance method."""

    contract.assert_valid()
    trace.validate()
    route = np.asarray(reference_route_enu_m, dtype=float)
    if route_sha256(route) != contract.route_sha256:
        raise MetricsContractError(
            "reference route does not match ExperimentContract.route_sha256"
        )
    initial_error = np.linalg.norm(
        trace.position_enu_m[0] - np.asarray(contract.initial_position_enu_m)
    )
    if initial_error > 2.0:
        raise MetricsContractError(
            "trace initial position differs from experiment contract by more than 2 m"
        )
    if not math.isfinite(completion_radius_m) or completion_radius_m <= 0.0:
        raise MetricsContractError("completion_radius_m must be positive")

    cross_track, progress = _project_to_polyline(trace.position_enu_m, route)
    route_segments = np.diff(route, axis=0)
    reference_length = float(np.sum(np.linalg.norm(route_segments, axis=1)))
    flown_length = float(
        np.sum(np.linalg.norm(np.diff(trace.position_enu_m, axis=0), axis=1))
    )
    final_error = float(np.linalg.norm(trace.position_enu_m[-1] - route[-1]))
    speed = np.linalg.norm(trace.velocity_enu_m_s, axis=1)
    acceleration_vectors = np.gradient(
        trace.velocity_enu_m_s, trace.timestamp_s, axis=0, edge_order=2
    )
    acceleration = np.linalg.norm(acceleration_vectors, axis=1)
    jerk_vectors = np.gradient(acceleration_vectors, trace.timestamp_s, axis=0, edge_order=2)
    jerk = np.linalg.norm(jerk_vectors, axis=1)
    progress_delta = np.diff(progress)

    return GuidanceRunMetrics(
        guidance_method=contract.guidance_method,
        duration_s=float(trace.timestamp_s[-1] - trace.timestamp_s[0]),
        reference_length_m=reference_length,
        flown_length_m=flown_length,
        final_error_m=final_error,
        completed=final_error <= completion_radius_m,
        rms_cross_track_error_m=float(math.sqrt(np.mean(cross_track * cross_track))),
        p95_cross_track_error_m=float(np.percentile(cross_track, 95.0)),
        max_cross_track_error_m=float(np.max(cross_track)),
        min_clearance_m=float(np.min(trace.clearance_m)),
        hard_clearance_violations=int(
            np.count_nonzero(trace.clearance_m < contract.hard_radius_m)
        ),
        max_speed_m_s=float(np.max(speed)),
        p95_speed_m_s=float(np.percentile(speed, 95.0)),
        max_acceleration_m_s2=float(np.max(acceleration)),
        p95_acceleration_m_s2=float(np.percentile(acceleration, 95.0)),
        max_jerk_m_s3=float(np.max(jerk)),
        p95_jerk_m_s3=float(np.percentile(jerk, 95.0)),
        backtracking_distance_m=float(-np.sum(progress_delta[progress_delta < 0.0])),
    )


def assert_paired_contracts(
    l1_contract: ExperimentContract,
    bspline_contract: ExperimentContract,
    *,
    initial_tolerance_m: float = 0.25,
) -> None:
    """Reject a comparison where map, route, firmware, seed, or conditions differ."""

    l1_contract.assert_valid()
    bspline_contract.assert_valid()
    if l1_contract.guidance_method != "px4_mc_l1_style":
        raise MetricsContractError("first contract must be px4_mc_l1_style")
    if bspline_contract.guidance_method != "bspline_mpc":
        raise MetricsContractError("second contract must be bspline_mpc")
    matched_fields = (
        "map_revision",
        "occupancy_sha256",
        "route_sha256",
        "planner_run_id",
        "px4_git_revision",
        "scenario_seed",
        "hard_radius_m",
        "minimum_surface_clearance_m",
        "wind_profile_id",
        "payload_profile_id",
    )
    mismatched = [
        field
        for field in matched_fields
        if getattr(l1_contract, field) != getattr(bspline_contract, field)
    ]
    initial_delta = np.linalg.norm(
        np.asarray(l1_contract.initial_position_enu_m)
        - np.asarray(bspline_contract.initial_position_enu_m)
    )
    if initial_delta > initial_tolerance_m:
        mismatched.append("initial_position_enu_m")
    if mismatched:
        raise MetricsContractError("unpaired experiment fields: " + ", ".join(mismatched))


def comparison_delta(
    l1_metrics: GuidanceRunMetrics,
    bspline_metrics: GuidanceRunMetrics,
) -> dict[str, float | bool]:
    """Return L1-minus-B-spline deltas; negative is better for error/dynamics."""

    if l1_metrics.guidance_method != "px4_mc_l1_style":
        raise MetricsContractError("l1_metrics has the wrong guidance_method")
    if bspline_metrics.guidance_method != "bspline_mpc":
        raise MetricsContractError("bspline_metrics has the wrong guidance_method")
    fields = (
        "duration_s",
        "flown_length_m",
        "final_error_m",
        "rms_cross_track_error_m",
        "p95_cross_track_error_m",
        "max_cross_track_error_m",
        "max_speed_m_s",
        "max_acceleration_m_s2",
        "max_jerk_m_s3",
        "backtracking_distance_m",
    )
    result: dict[str, float | bool] = {
        f"l1_minus_bspline_{field}": float(
            getattr(l1_metrics, field) - getattr(bspline_metrics, field)
        )
        for field in fields
    }
    result["l1_minus_bspline_min_clearance_m"] = (
        l1_metrics.min_clearance_m - bspline_metrics.min_clearance_m
    )
    result["both_completed"] = l1_metrics.completed and bspline_metrics.completed
    result["both_hard_clearance_safe"] = (
        l1_metrics.hard_clearance_violations == 0
        and bspline_metrics.hard_clearance_violations == 0
    )
    return result


def write_metrics_json(
    path: str | Path,
    contract: ExperimentContract,
    metrics: GuidanceRunMetrics,
) -> None:
    """Persist one reproducible result without serialising raw high-rate samples."""

    payload: Mapping[str, object] = {
        "experiment_contract": asdict(contract),
        "metrics": metrics.as_dict(),
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
