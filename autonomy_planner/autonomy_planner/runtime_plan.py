"""Runtime-safe adapter for the deterministic 3-D city planner.

The heavy planning primitives live in :mod:`autonomy_planner.planner_3d`.
This module provides the small contract needed by a mission executor:

* build the exact map-declared 1.45 m hard / 2.15 m preferred envelope;
* run hierarchical A* and SFC construction exactly once per start/goal pair;
* retain that common certificate while selecting either constrained B-spline
  samples or the PX4-native waypoint input;
* provide hard SFC bounds in PX4-local NED for every MPC horizon sample; and
* keep slow replanning outside a control callback with a latest-request worker.

There are deliberately no ROS imports.  Planner and trajectory coordinates
remain Gazebo-world ENU until :func:`map_horizon_references_to_ned` performs
the explicit PX4 boundary conversion.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
import threading
import time
from typing import Callable, Sequence

import numpy as np

from .planner_core import CityMap, FrameTransform
from .planner_3d import (
    AStar3DConfig,
    BSplinePlanResult3D,
    HierarchicalAStar3DConfig,
    HierarchicalAStarPlanner3D,
    PlanningCandidateMetrics3D,
    PlanningModeComparison3D,
    PlanningOutputMode,
    PrismOccupancy3D,
    SafeFlightCorridor3D,
    TrajectoryLimits3D,
    ValidatedWaypointPlan3D,
    VehicleEnvelope3D,
    compare_planning_output_modes_3d,
    plan_validated_waypoints_3d,
)


_EPS = 1.0e-9


class RuntimePlanningError(RuntimeError):
    """Base exception for a plan that cannot safely become active."""


class CorridorViolationError(RuntimePlanningError):
    """A horizon reference was not certified by any hard SFC box."""


def _point3_tuple(value: Sequence[float], name: str) -> tuple[float, float, float]:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite three-vector")
    return tuple(float(component) for component in point)


def _readonly(value: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RuntimePlanner3DConfig:
    """Explicit bounded settings for the runtime hierarchical planner.

    ``preferred_clearance`` in :class:`AStar3DConfig` is measured outside the
    already hard-inflated occupancy.  It is therefore derived as
    ``2.15 - 1.45 == 0.70 m`` instead of applying 2.15 m twice.
    """

    expected_hard_radius_m: float = 1.45
    expected_preferred_radius_m: float = 2.15
    envelope_tolerance_m: float = 1.0e-9
    minimum_z_m: float = 0.0
    maximum_z_m: float = 120.0
    ground_z_m: float = 0.0
    minimum_surface_clearance_m: float = 10.0
    minimum_center_buffer_m: float = 0.10

    coarse_resolution_m: tuple[float, float, float] = (6.0, 6.0, 4.0)
    fine_resolution_m: tuple[float, float, float] = (2.0, 2.0, 2.0)
    vertical_cost_weight: float = 3.0
    climb_cost_weight: float = 0.5
    clearance_cost_weight: float = 2.0
    coarse_timeout_s: float = 8.0
    fine_timeout_s: float = 15.0
    coarse_max_expanded_nodes: int = 200_000
    fine_max_expanded_nodes: int = 350_000
    coarse_edge_sample_step_m: float = 1.0
    fine_edge_sample_step_m: float = 0.4
    snap_radius_m: float = 8.0
    fine_search_band_m: float = 10.0
    fine_vertical_search_band_m: float = 3.0
    widened_search_band_scale: float = 1.75
    maximum_fine_attempts: int = 2
    astar_cache_entries: int = 16

    sfc_minimum_overlap_m: float = 0.5
    sfc_initial_margin_m: float = 0.5
    sfc_maximum_margin_m: float = 5.0
    sfc_expansion_step_m: float = 0.5
    sfc_maximum_segment_length_m: float = 6.0
    sfc_maximum_subdivision_depth: int = 14
    sfc_maximum_expansion_iterations: int = 10_000
    sfc_deadline_s: float = 5.0
    validation_step_m: float = 0.2

    trajectory_limits: TrajectoryLimits3D = field(
        default_factory=lambda: TrajectoryLimits3D(4.0, 2.0, 4.0)
    )
    nominal_speed_m_s: float = 3.0
    native_sample_spacing_m: float = 1.0
    bspline_sample_period_s: float = 0.10
    bspline_control_point_spacing_m: float = 6.0
    bspline_deadline_s: float = 8.0
    bspline_maximum_optimization_iterations: int = 400
    bspline_reference_weight: float = 1.0
    bspline_acceleration_weight: float = 10.0
    bspline_jerk_weight: float = 20.0
    prepared_cache_entries: int = 4

    def __post_init__(self) -> None:
        positive = (
            self.expected_hard_radius_m,
            self.expected_preferred_radius_m,
            self.envelope_tolerance_m,
            *self.coarse_resolution_m,
            *self.fine_resolution_m,
            self.vertical_cost_weight,
            self.clearance_cost_weight,
            self.coarse_timeout_s,
            self.fine_timeout_s,
            self.coarse_edge_sample_step_m,
            self.fine_edge_sample_step_m,
            self.snap_radius_m,
            self.fine_search_band_m,
            self.fine_vertical_search_band_m,
            self.widened_search_band_scale,
            self.sfc_minimum_overlap_m,
            self.sfc_initial_margin_m,
            self.sfc_maximum_margin_m,
            self.sfc_expansion_step_m,
            self.sfc_maximum_segment_length_m,
            self.sfc_deadline_s,
            self.validation_step_m,
            self.nominal_speed_m_s,
            self.native_sample_spacing_m,
            self.bspline_sample_period_s,
            self.bspline_control_point_spacing_m,
            self.bspline_deadline_s,
            self.bspline_reference_weight,
            self.bspline_acceleration_weight,
            self.bspline_jerk_weight,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("runtime planner lengths, weights, and deadlines must be positive")
        if not math.isfinite(self.climb_cost_weight) or self.climb_cost_weight < 0.0:
            raise ValueError("climb_cost_weight must be finite and non-negative")
        if (
            not math.isfinite(self.minimum_surface_clearance_m)
            or self.minimum_surface_clearance_m < 0.0
            or not math.isfinite(self.minimum_center_buffer_m)
            or self.minimum_center_buffer_m < 0.0
        ):
            raise ValueError("surface clearance and centre buffer must be non-negative")
        if not self.minimum_z_m <= self.ground_z_m < self.maximum_z_m:
            raise ValueError("vertical runtime geofence is invalid")
        if self.expected_preferred_radius_m < self.expected_hard_radius_m:
            raise ValueError("preferred radius cannot be smaller than hard radius")
        if self.sfc_maximum_margin_m < self.sfc_initial_margin_m:
            raise ValueError("SFC maximum margin cannot be smaller than its initial margin")
        integers = (
            self.coarse_max_expanded_nodes,
            self.fine_max_expanded_nodes,
            self.maximum_fine_attempts,
            self.sfc_maximum_subdivision_depth,
            self.sfc_maximum_expansion_iterations,
            self.bspline_maximum_optimization_iterations,
        )
        if any(value <= 0 for value in integers):
            raise ValueError("runtime planner iteration and node limits must be positive")
        if self.astar_cache_entries < 0 or self.prepared_cache_entries < 0:
            raise ValueError("runtime planner cache sizes cannot be negative")

    def hierarchical_config(
        self, envelope: VehicleEnvelope3D
    ) -> HierarchicalAStar3DConfig:
        residual_preferred = (
            envelope.preferred_horizontal_radius_m
            - envelope.hard_horizontal_radius_m
        )
        common = dict(
            connectivity=26,
            vertical_weight=self.vertical_cost_weight,
            climb_weight=self.climb_cost_weight,
            clearance_weight=self.clearance_cost_weight,
            safe_clearance_m=residual_preferred,
            preferred_clearance_m=residual_preferred,
            minimum_clearance_m=0.0,
            snap_radius_m=self.snap_radius_m,
        )
        return HierarchicalAStar3DConfig(
            coarse=AStar3DConfig(
                resolution_m=self.coarse_resolution_m,
                planning_timeout_s=self.coarse_timeout_s,
                max_expanded_nodes=self.coarse_max_expanded_nodes,
                edge_sample_step_m=self.coarse_edge_sample_step_m,
                **common,
            ),
            fine=AStar3DConfig(
                resolution_m=self.fine_resolution_m,
                planning_timeout_s=self.fine_timeout_s,
                max_expanded_nodes=self.fine_max_expanded_nodes,
                edge_sample_step_m=self.fine_edge_sample_step_m,
                **common,
            ),
            fine_search_band_m=self.fine_search_band_m,
            fine_vertical_search_band_m=self.fine_vertical_search_band_m,
            widened_search_band_scale=self.widened_search_band_scale,
            maximum_fine_attempts=self.maximum_fine_attempts,
            cache_entries=self.astar_cache_entries,
        )


@dataclass(frozen=True)
class PreparedPlan3D:
    """One common A* + SFC certificate and its two A/B candidates."""

    start_enu_m: tuple[float, float, float]
    goal_enu_m: tuple[float, float, float]
    waypoint_plan: ValidatedWaypointPlan3D
    comparison: PlanningModeComparison3D

    @property
    def corridor(self) -> SafeFlightCorridor3D:
        corridor = self.waypoint_plan.corridor
        if corridor is None:
            raise RuntimePlanningError("prepared plan has no valid SFC")
        return corridor


@dataclass(frozen=True)
class ActivePlanMetrics3D:
    mode: PlanningOutputMode
    sample_count: int
    sampled_path_length_m: float
    duration_s: float | None
    source: PlanningCandidateMetrics3D


@dataclass(frozen=True)
class HorizonCorridorBounds3D:
    """All-or-nothing hard SFC assignment for a sequence of references."""

    reference_enu_m: np.ndarray
    corridor_box_indices: tuple[int, ...]
    bounds_px4_local_ned_m: np.ndarray

    def __post_init__(self) -> None:
        references = np.asarray(self.reference_enu_m, dtype=float)
        bounds = np.asarray(self.bounds_px4_local_ned_m, dtype=float)
        if references.ndim != 2 or references.shape[1] != 3:
            raise ValueError("horizon references must have shape (N, 3)")
        if bounds.shape != (len(references), 6):
            raise ValueError("NED corridor bounds must have shape (N, 6)")
        if len(self.corridor_box_indices) != len(references):
            raise ValueError("every horizon reference needs one SFC box")
        if not np.all(np.isfinite(references)) or not np.all(np.isfinite(bounds)):
            raise ValueError("horizon corridor mapping must be finite")
        if np.any(bounds[:, 0] > bounds[:, 1]) or np.any(bounds[:, 2] > bounds[:, 3]) or np.any(bounds[:, 4] > bounds[:, 5]):
            raise ValueError("NED corridor bounds are not ordered")
        object.__setattr__(self, "reference_enu_m", _readonly(references))
        object.__setattr__(self, "bounds_px4_local_ned_m", _readonly(bounds))


def enu_box_to_px4_local_ned_bounds(
    minimum_enu_m: Sequence[float],
    maximum_enu_m: Sequence[float],
    frame_transform: FrameTransform,
) -> np.ndarray:
    """Convert one ENU AABB to ordered PX4-local NED AABB bounds.

    Transforming all eight corners avoids implicit axis/sign assumptions and
    makes the result exactly follow :class:`FrameTransform`'s configured
    origin.  The output order is ``[xmin, xmax, ymin, ymax, zmin, zmax]``.
    """

    minimum = np.asarray(_point3_tuple(minimum_enu_m, "ENU box minimum"))
    maximum = np.asarray(_point3_tuple(maximum_enu_m, "ENU box maximum"))
    if np.any(maximum <= minimum):
        raise ValueError("ENU box maximum must be greater than minimum")
    corners_ned = np.asarray(
        [
            frame_transform.enu_to_ned(corner)
            for corner in itertools.product(
                (minimum[0], maximum[0]),
                (minimum[1], maximum[1]),
                (minimum[2], maximum[2]),
            )
        ]
    )
    result = np.asarray(
        (
            np.min(corners_ned[:, 0]),
            np.max(corners_ned[:, 0]),
            np.min(corners_ned[:, 1]),
            np.max(corners_ned[:, 1]),
            np.min(corners_ned[:, 2]),
            np.max(corners_ned[:, 2]),
        ),
        dtype=float,
    )
    result.setflags(write=False)
    return result


def map_horizon_references_to_ned(
    reference_enu_m: Sequence[Sequence[float]],
    corridor: SafeFlightCorridor3D,
    frame_transform: FrameTransform,
) -> HorizonCorridorBounds3D:
    """Assign every reference to a hard SFC box, or fail without a result."""

    references = np.asarray(reference_enu_m, dtype=float)
    if references.ndim != 2 or references.shape[1] != 3 or len(references) == 0:
        raise ValueError("horizon references must have shape (N>=1, 3)")
    if not np.all(np.isfinite(references)):
        raise ValueError("horizon references must be finite")

    assignments: list[int] = []
    previous = 0
    for reference_index, point in enumerate(references):
        candidates = corridor.box_indices_containing(point)
        if not candidates:
            raise CorridorViolationError(
                "SFC_REFERENCE_OUTSIDE:"
                f"reference={reference_index},enu={tuple(float(v) for v in point)}"
            )
        # Stay in the previous overlapping cell when possible; otherwise move
        # to the closest numbered cell.  The tie-break is deterministic.
        selected = (
            previous
            if previous in candidates
            else min(candidates, key=lambda candidate: (abs(candidate - previous), candidate))
        )
        assignments.append(selected)
        previous = selected

    bounds = np.asarray(
        [
            enu_box_to_px4_local_ned_bounds(
                corridor.boxes[index].minimum,
                corridor.boxes[index].maximum,
                frame_transform,
            )
            for index in assignments
        ]
    )
    return HorizonCorridorBounds3D(references, tuple(assignments), bounds)


@dataclass(frozen=True)
class ActivePlan3D:
    """Immutable samples selected from one retained A* + SFC certificate."""

    mode: PlanningOutputMode
    waypoint_plan: ValidatedWaypointPlan3D
    corridor: SafeFlightCorridor3D
    sampled_position_enu_m: np.ndarray
    sampled_velocity_enu_m_s: np.ndarray | None
    sampled_acceleration_enu_m_s2: np.ndarray | None
    sample_time_s: np.ndarray | None
    cumulative_length_m: np.ndarray
    metrics: ActivePlanMetrics3D
    bspline_result: BSplinePlanResult3D | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.sampled_position_enu_m, dtype=float)
        cumulative = np.asarray(self.cumulative_length_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("active plan positions must have shape (N>=2, 3)")
        if not np.all(np.isfinite(positions)) or cumulative.shape != (len(positions),):
            raise ValueError("active plan positions/cumulative lengths are invalid")
        if not np.all(np.isfinite(cumulative)) or abs(cumulative[0]) > _EPS or np.any(np.diff(cumulative) < -_EPS):
            raise ValueError("active plan cumulative length must be finite and monotonic from zero")
        if not self.waypoint_plan.success or self.waypoint_plan.corridor is not self.corridor:
            raise RuntimePlanningError("active plan lost its common waypoint/SFC certificate")
        outside = [index for index, point in enumerate(positions) if not self.corridor.contains(point)]
        if outside:
            raise CorridorViolationError(
                f"ACTIVE_PLAN_OUTSIDE_SFC:first_reference={outside[0]}"
            )
        for field_name in ("sampled_velocity_enu_m_s", "sampled_acceleration_enu_m_s2"):
            value = getattr(self, field_name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.shape != positions.shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"{field_name} must match sampled positions")
                object.__setattr__(self, field_name, _readonly(array))
        if self.sample_time_s is not None:
            times = np.asarray(self.sample_time_s, dtype=float)
            if times.shape != (len(positions),) or not np.all(np.isfinite(times)) or np.any(np.diff(times) < -_EPS):
                raise ValueError("active plan sample times are invalid")
            object.__setattr__(self, "sample_time_s", _readonly(times))
        object.__setattr__(self, "sampled_position_enu_m", _readonly(positions))
        object.__setattr__(self, "cumulative_length_m", _readonly(cumulative))

    @property
    def path_length_m(self) -> float:
        return float(self.cumulative_length_m[-1])

    def horizon_sfc_bounds_ned(
        self,
        reference_enu_m: Sequence[Sequence[float]],
        frame_transform: FrameTransform,
    ) -> HorizonCorridorBounds3D:
        return map_horizon_references_to_ned(
            reference_enu_m, self.corridor, frame_transform
        )


def _sample_polyline(
    waypoints: Sequence[Sequence[float]], spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(waypoints, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > _EPS))
    unique = path[keep]
    if len(unique) < 2:
        raise RuntimePlanningError("validated waypoint plan has no path length")
    cumulative_waypoints = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(unique, axis=0), axis=1)))
    )
    count = max(2, int(math.ceil(cumulative_waypoints[-1] / spacing_m)) + 1)
    targets = np.linspace(0.0, cumulative_waypoints[-1], count)
    samples = np.column_stack(
        [
            np.interp(targets, cumulative_waypoints, unique[:, axis])
            for axis in range(3)
        ]
    )
    # Interpolation by global arclength can cut a corner when one target falls
    # on each side of a waypoint.  Include every certified waypoint so the
    # sampled polyline exactly retains the native PX4 mission geometry.
    samples = np.vstack((samples, unique))
    order_parameter = np.concatenate((targets, cumulative_waypoints))
    order = np.argsort(order_parameter, kind="stable")
    samples = samples[order]
    ordered_parameter = order_parameter[order]
    unique_rows = np.concatenate(
        ([True], np.diff(ordered_parameter) > _EPS)
    )
    samples = samples[unique_rows]
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1)))
    )
    return samples, cumulative


class RuntimePlanAdapter3D:
    """Prepare one common safe plan and activate either A/B output mode."""

    def __init__(
        self,
        city_map: CityMap,
        config: RuntimePlanner3DConfig | None = None,
    ) -> None:
        self.city_map = city_map
        self.config = config or RuntimePlanner3DConfig()
        self.envelope = VehicleEnvelope3D.from_city_map_defaults(city_map)
        tolerance = self.config.envelope_tolerance_m
        if abs(self.envelope.hard_horizontal_radius_m - self.config.expected_hard_radius_m) > tolerance:
            raise ValueError(
                "map hard radius does not match runtime contract: "
                f"{self.envelope.hard_horizontal_radius_m} != {self.config.expected_hard_radius_m}"
            )
        if abs(self.envelope.preferred_horizontal_radius_m - self.config.expected_preferred_radius_m) > tolerance:
            raise ValueError(
                "map preferred radius does not match runtime contract: "
                f"{self.envelope.preferred_horizontal_radius_m} != {self.config.expected_preferred_radius_m}"
            )
        self.occupancy = PrismOccupancy3D.from_vehicle_envelope(
            city_map,
            self.envelope,
            minimum_z_m=self.config.minimum_z_m,
            maximum_z_m=self.config.maximum_z_m,
            ground_z_m=self.config.ground_z_m,
            minimum_surface_clearance_m=self.config.minimum_surface_clearance_m,
        )
        self.frame_transform = FrameTransform.from_city_map(city_map)
        self.planner = HierarchicalAStarPlanner3D(
            self.occupancy, self.config.hierarchical_config(self.envelope)
        )
        self._prepared_cache: OrderedDict[
            tuple[tuple[float, float, float], tuple[float, float, float]],
            PreparedPlan3D,
        ] = OrderedDict()
        self._plan_lock = threading.RLock()

    @property
    def minimum_planning_center_z_m(self) -> float:
        """Lowest centre Z satisfying clearance and a feasible hard SFC box.

        The vehicle centre itself is safe at 10.5 m with the default envelope.
        SFC construction additionally needs symmetric vertical padding around
        that centre.  Including the configured padding here prevents accepting
        10.5 m only to fail later while boxing the otherwise valid A* path.
        """

        return (
            self.config.ground_z_m
            + self.config.minimum_surface_clearance_m
            + self.envelope.vertical_half_extent_m
            + max(
                self.config.sfc_initial_margin_m,
                0.5 * self.config.sfc_minimum_overlap_m,
                math.sqrt(3.0) * self.config.sfc_initial_margin_m,
                0.5
                * math.sqrt(3.0)
                * self.config.sfc_minimum_overlap_m,
            )
            + self.config.minimum_center_buffer_m
        )

    def prepare(
        self,
        start_enu_m: Sequence[float],
        goal_enu_m: Sequence[float],
        *,
        force_replan: bool = False,
    ) -> PreparedPlan3D:
        """Run A*+SFC once and derive both candidates from that certificate."""

        start = _point3_tuple(start_enu_m, "runtime plan start")
        goal = _point3_tuple(goal_enu_m, "runtime plan goal")
        minimum_center_z = self.minimum_planning_center_z_m
        if start[2] < minimum_center_z - _EPS or goal[2] < minimum_center_z - _EPS:
            raise RuntimePlanningError(
                "MINIMUM_SURFACE_CLEARANCE_REJECTED:"
                f"start_z={start[2]:.3f},goal_z={goal[2]:.3f},"
                f"minimum_center_z={minimum_center_z:.3f}"
            )
        key = (start, goal)
        with self._plan_lock:
            if not force_replan and key in self._prepared_cache:
                self._prepared_cache.move_to_end(key)
                return self._prepared_cache[key]
            common = plan_validated_waypoints_3d(
                self.planner,
                start,
                goal,
                validation_step_m=self.config.validation_step_m,
                sfc_minimum_overlap_m=self.config.sfc_minimum_overlap_m,
                sfc_initial_margin_m=self.config.sfc_initial_margin_m,
                sfc_maximum_margin_m=self.config.sfc_maximum_margin_m,
                sfc_expansion_step_m=self.config.sfc_expansion_step_m,
                sfc_maximum_segment_length_m=self.config.sfc_maximum_segment_length_m,
                sfc_maximum_subdivision_depth=self.config.sfc_maximum_subdivision_depth,
                sfc_maximum_expansion_iterations=self.config.sfc_maximum_expansion_iterations,
                sfc_deadline_s=self.config.sfc_deadline_s,
            )
            if not common.success or common.corridor is None:
                raise RuntimePlanningError(
                    f"A_STAR_SFC_REJECTED:{common.metrics.reason}"
                )
            comparison = compare_planning_output_modes_3d(
                common,
                self.occupancy,
                self.config.trajectory_limits,
                nominal_speed_m_s=self.config.nominal_speed_m_s,
                bspline_options={
                    "control_point_spacing_m": self.config.bspline_control_point_spacing_m,
                    "wall_clock_deadline_s": self.config.bspline_deadline_s,
                    "maximum_optimization_iterations": self.config.bspline_maximum_optimization_iterations,
                    "reference_weight": self.config.bspline_reference_weight,
                    "acceleration_weight": self.config.bspline_acceleration_weight,
                    "jerk_weight": self.config.bspline_jerk_weight,
                },
            )
            prepared = PreparedPlan3D(start, goal, common, comparison)
            if self.config.prepared_cache_entries:
                self._prepared_cache[key] = prepared
                self._prepared_cache.move_to_end(key)
                while len(self._prepared_cache) > self.config.prepared_cache_entries:
                    self._prepared_cache.popitem(last=False)
            return prepared

    def activate(
        self,
        prepared: PreparedPlan3D,
        mode: PlanningOutputMode | str,
    ) -> ActivePlan3D:
        """Select samples without rerunning A* or rebuilding the SFC."""

        selected_mode = PlanningOutputMode(mode)
        if prepared.waypoint_plan is not prepared.comparison.waypoint_plan:
            raise RuntimePlanningError("mode comparison does not retain the common plan")
        corridor = prepared.corridor
        if selected_mode is PlanningOutputMode.SFC_BSPLINE:
            result = prepared.comparison.bspline_result
            source = prepared.comparison.bspline_metrics
            if (
                result is None
                or not result.success
                or result.trajectory is None
                or not source.success
            ):
                reason = source.reason if source is not None else "missing_result"
                raise RuntimePlanningError(f"B_SPLINE_REJECTED:{reason}")
            samples = result.trajectory.sample(self.config.bspline_sample_period_s)
            position = samples.position_enu_m
            velocity = samples.velocity_enu_m_s
            acceleration = samples.acceleration_enu_m_s2
            sample_time = samples.time_s - samples.time_s[0]
            bspline_result = result
            duration = float(sample_time[-1])
        else:
            source = prepared.comparison.waypoint_metrics
            if not source.success:
                raise RuntimePlanningError(f"PX4_WAYPOINTS_REJECTED:{source.reason}")
            position, _ = _sample_polyline(
                prepared.waypoint_plan.waypoints_enu_m,
                self.config.native_sample_spacing_m,
            )
            velocity = None
            acceleration = None
            sample_time = None
            bspline_result = None
            duration = None

        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(position, axis=0), axis=1)))
        )
        metrics = ActivePlanMetrics3D(
            selected_mode,
            len(position),
            float(cumulative[-1]),
            duration,
            source,
        )
        return ActivePlan3D(
            selected_mode,
            prepared.waypoint_plan,
            corridor,
            position,
            velocity,
            acceleration,
            sample_time,
            cumulative,
            metrics,
            bspline_result,
        )

    def plan_active(
        self,
        start_enu_m: Sequence[float],
        goal_enu_m: Sequence[float],
        mode: PlanningOutputMode | str,
    ) -> ActivePlan3D:
        return self.activate(self.prepare(start_enu_m, goal_enu_m), mode)


class PlanWorkerStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PlanWorkerOutcome3D:
    request_id: int
    status: PlanWorkerStatus
    plan: ActivePlan3D | None = None
    error: str | None = None


@dataclass(frozen=True)
class PlanWorkerStats3D:
    submitted: int
    executed: int
    cache_hits: int
    coalesced: int
    cancelled: int


@dataclass(frozen=True)
class _PlanRequest3D:
    request_id: int
    start_enu_m: tuple[float, float, float]
    goal_enu_m: tuple[float, float, float]
    mode: PlanningOutputMode

    @property
    def cache_key(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        str,
    ]:
        return self.start_enu_m, self.goal_enu_m, self.mode.value


class LatestPlanWorker3D:
    """Single-thread, capacity-one, latest-request asynchronous planner.

    A newer submission supersedes both a queued request and the observable
    result of an in-flight request.  Python cannot forcibly stop an arbitrary
    SciPy/A* call, so an in-flight obsolete computation is allowed to finish
    but is discarded before it can become active.  This is cooperative,
    bounded cancellation without unsafe thread termination.
    """

    def __init__(
        self,
        plan_function: Callable[
            [Sequence[float], Sequence[float], PlanningOutputMode], ActivePlan3D
        ],
        *,
        cache_entries: int = 4,
        outcome_history: int = 64,
        thread_name: str = "latest-plan-3d",
    ) -> None:
        if cache_entries < 0 or outcome_history <= 0:
            raise ValueError("worker cache/history bounds are invalid")
        self._plan_function = plan_function
        self._cache_entries = cache_entries
        self._outcome_history = outcome_history
        self._condition = threading.Condition()
        self._pending: _PlanRequest3D | None = None
        self._active_request_id: int | None = None
        self._outcomes: OrderedDict[int, PlanWorkerOutcome3D] = OrderedDict()
        self._cache: OrderedDict[
            tuple[tuple[float, float, float], tuple[float, float, float], str],
            ActivePlan3D,
        ] = OrderedDict()
        self._next_request_id = 1
        self._latest_request_id = 0
        self._stopping = False
        self._submitted = 0
        self._executed = 0
        self._cache_hits = 0
        self._coalesced = 0
        self._cancelled = 0
        self._thread = threading.Thread(
            target=self._run, name=thread_name, daemon=True
        )
        self._thread.start()

    def _record_locked(self, outcome: PlanWorkerOutcome3D) -> None:
        if outcome.request_id in self._outcomes:
            return
        self._outcomes[outcome.request_id] = outcome
        self._outcomes.move_to_end(outcome.request_id)
        while len(self._outcomes) > self._outcome_history:
            self._outcomes.popitem(last=False)
        self._condition.notify_all()

    def _supersede_locked(self, request_id: int) -> None:
        if request_id not in self._outcomes:
            self._coalesced += 1
            self._record_locked(
                PlanWorkerOutcome3D(request_id, PlanWorkerStatus.SUPERSEDED)
            )

    def submit(
        self,
        start_enu_m: Sequence[float],
        goal_enu_m: Sequence[float],
        mode: PlanningOutputMode | str,
    ) -> int:
        start = _point3_tuple(start_enu_m, "worker start")
        goal = _point3_tuple(goal_enu_m, "worker goal")
        selected_mode = PlanningOutputMode(mode)
        with self._condition:
            if self._stopping:
                raise RuntimeError("plan worker is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            self._latest_request_id = request_id
            self._submitted += 1
            request = _PlanRequest3D(request_id, start, goal, selected_mode)
            if self._active_request_id is not None:
                self._supersede_locked(self._active_request_id)
            if self._pending is not None:
                self._supersede_locked(self._pending.request_id)
                self._pending = None
            cached = self._cache.get(request.cache_key)
            if cached is not None:
                self._cache.move_to_end(request.cache_key)
                self._cache_hits += 1
                self._record_locked(
                    PlanWorkerOutcome3D(
                        request_id, PlanWorkerStatus.SUCCEEDED, plan=cached
                    )
                )
            else:
                self._pending = request
                self._condition.notify_all()
            return request_id

    def cancel(self, request_id: int) -> bool:
        """Cancel a queued/in-flight request's observable result."""

        with self._condition:
            if request_id in self._outcomes:
                return False
            is_pending = self._pending is not None and self._pending.request_id == request_id
            is_active = self._active_request_id == request_id
            if not is_pending and not is_active:
                return False
            if is_pending:
                self._pending = None
            self._cancelled += 1
            self._record_locked(
                PlanWorkerOutcome3D(request_id, PlanWorkerStatus.CANCELLED)
            )
            return True

    def wait(self, request_id: int, timeout_s: float | None = None) -> PlanWorkerOutcome3D:
        if timeout_s is not None and timeout_s < 0.0:
            raise ValueError("worker wait timeout cannot be negative")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while request_id not in self._outcomes:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"plan request {request_id} did not complete")
                self._condition.wait(remaining)
            return self._outcomes[request_id]

    @property
    def latest_outcome(self) -> PlanWorkerOutcome3D | None:
        with self._condition:
            return self._outcomes.get(self._latest_request_id)

    @property
    def stats(self) -> PlanWorkerStats3D:
        with self._condition:
            return PlanWorkerStats3D(
                self._submitted,
                self._executed,
                self._cache_hits,
                self._coalesced,
                self._cancelled,
            )

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping and self._pending is None:
                    return
                request = self._pending
                self._pending = None
                assert request is not None
                self._active_request_id = request.request_id
                self._executed += 1
            try:
                plan = self._plan_function(
                    request.start_enu_m, request.goal_enu_m, request.mode
                )
                error = None
            except Exception as exception:  # surfaced as data; worker remains alive
                plan = None
                error = f"{type(exception).__name__}:{exception}"
            with self._condition:
                self._active_request_id = None
                if request.request_id not in self._outcomes:
                    if error is None and plan is not None:
                        if self._cache_entries:
                            self._cache[request.cache_key] = plan
                            self._cache.move_to_end(request.cache_key)
                            while len(self._cache) > self._cache_entries:
                                self._cache.popitem(last=False)
                        self._record_locked(
                            PlanWorkerOutcome3D(
                                request.request_id,
                                PlanWorkerStatus.SUCCEEDED,
                                plan=plan,
                            )
                        )
                    else:
                        self._record_locked(
                            PlanWorkerOutcome3D(
                                request.request_id,
                                PlanWorkerStatus.FAILED,
                                error=error or "planner returned no plan",
                            )
                        )
                self._condition.notify_all()

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0.0:
            raise ValueError("worker close timeout cannot be negative")
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            if self._pending is not None:
                request_id = self._pending.request_id
                self._pending = None
                self._cancelled += 1
                self._record_locked(
                    PlanWorkerOutcome3D(request_id, PlanWorkerStatus.CANCELLED)
                )
            if self._active_request_id is not None and self._active_request_id not in self._outcomes:
                self._cancelled += 1
                self._record_locked(
                    PlanWorkerOutcome3D(
                        self._active_request_id, PlanWorkerStatus.CANCELLED
                    )
                )
            self._condition.notify_all()
        self._thread.join(timeout_s)

    def __enter__(self) -> "LatestPlanWorker3D":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "ActivePlan3D",
    "ActivePlanMetrics3D",
    "CorridorViolationError",
    "HorizonCorridorBounds3D",
    "LatestPlanWorker3D",
    "PlanWorkerOutcome3D",
    "PlanWorkerStats3D",
    "PlanWorkerStatus",
    "PreparedPlan3D",
    "RuntimePlanAdapter3D",
    "RuntimePlanner3DConfig",
    "RuntimePlanningError",
    "enu_box_to_px4_local_ned_bounds",
    "map_horizon_references_to_ned",
]
