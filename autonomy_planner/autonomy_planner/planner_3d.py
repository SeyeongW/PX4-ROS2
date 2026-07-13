"""Deterministic 3-D global-planning primitives for the PX4 demo.

The city asset is a 2.5-D map: every building is represented by an exact
polygonal footprint (including holes) and a vertical interval.  This module
turns that representation into genuine ``(x, y, z)`` collision queries, a
voxel A* search, and a chain of convex 3-D safe-corridor boxes.  The common
safe result can feed either a constrained cubic B-spline or PX4's native
L1-style waypoint guidance so both modes can be measured against the same
map, envelope, and metrics.

There are intentionally no ROS imports here.  All coordinates are Gazebo
world ENU metres and all times are seconds.  Runtime code is expected to do
the ENU/NED conversion only at the PX4 boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
import heapq
import itertools
import math
import os
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
from scipy import interpolate, optimize, spatial
import yaml

from .planner_core import CityMap, PolygonPrism


_EPS = 1.0e-9


class PlanningOutputMode(str, Enum):
    """Runtime-selectable post-A* guidance contracts."""

    SFC_BSPLINE = "sfc_bspline"
    PX4_NATIVE_WAYPOINTS = "px4_native_waypoints"


def resolve_city_map_yaml(
    repository_root: str | Path | None = None,
    *,
    prefer_uav_map: bool = True,
) -> Path:
    """Resolve the planner map without hard-coding one checkout location.

    The UAV-spacing derivative is preferred when it exists.  ``PX4_ROS2_ROOT``
    is accepted for installed/relocated workspaces; otherwise the source-tree
    root is derived from this module.  Falling back to the original city map
    keeps a fresh checkout usable before the derivative asset is generated.
    """

    if repository_root is None:
        environment_root = os.environ.get("PX4_ROS2_ROOT")
        root = (
            Path(environment_root).expanduser().resolve()
            if environment_root
            else Path(__file__).resolve().parents[2]
        )
    else:
        root = Path(repository_root).expanduser().resolve()
    names = (
        ("city_coordinates_uav.yaml", "city_coordinates.yaml")
        if prefer_uav_map
        else ("city_coordinates.yaml", "city_coordinates_uav.yaml")
    )
    searched: list[Path] = []
    for name in names:
        for candidate in (root / "gazebo" / "maps" / name, root / name):
            searched.append(candidate)
            if candidate.is_file():
                return candidate
    rendered = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(f"no city coordinate map found; searched: {rendered}")


def load_preferred_city_map(
    repository_root: str | Path | None = None,
    *,
    prefer_uav_map: bool = True,
) -> CityMap:
    """Load ``city_coordinates_uav.yaml`` when available, else the base map."""

    return CityMap.from_yaml(
        resolve_city_map_yaml(repository_root, prefer_uav_map=prefer_uav_map)
    )


@dataclass(frozen=True)
class VehicleEnvelope3D:
    """Planning envelope and decomposed static uncertainty budget.

    The horizontal body is yaw invariant because yaw is not part of the A*
    state.  A vehicle smaller than one metre is therefore still represented by
    the circumcircle of a 1 m x 1 m square (about 0.7071 m radius).
    """

    body_length_x_m: float = 1.0
    body_width_y_m: float = 1.0
    body_height_z_m: float = 0.4
    map_margin_m: float = 0.10
    localization_margin_m: float = 0.15
    tracking_margin_m: float = 0.20
    control_margin_m: float = 0.15
    extra_margin_m: float = 0.1428932188134524
    preferred_extra_m: float = 0.70
    vertical_tracking_margin_m: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.body_length_x_m,
            self.body_width_y_m,
            self.body_height_z_m,
            self.map_margin_m,
            self.localization_margin_m,
            self.tracking_margin_m,
            self.control_margin_m,
            self.extra_margin_m,
            self.preferred_extra_m,
            self.vertical_tracking_margin_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vehicle-envelope parameters must be finite")
        if min(self.body_length_x_m, self.body_width_y_m, self.body_height_z_m) <= 0.0:
            raise ValueError("vehicle body dimensions must be positive")
        if min(values[3:]) < 0.0:
            raise ValueError("vehicle-envelope margins cannot be negative")

    @property
    def planning_length_x_m(self) -> float:
        return max(1.0, self.body_length_x_m)

    @property
    def planning_width_y_m(self) -> float:
        return max(1.0, self.body_width_y_m)

    @property
    def body_yaw_invariant_radius_m(self) -> float:
        return 0.5 * math.hypot(
            self.planning_length_x_m, self.planning_width_y_m
        )

    @property
    def hard_horizontal_radius_m(self) -> float:
        return (
            self.body_yaw_invariant_radius_m
            + self.map_margin_m
            + self.localization_margin_m
            + self.tracking_margin_m
            + self.control_margin_m
            + self.extra_margin_m
        )

    @property
    def preferred_horizontal_radius_m(self) -> float:
        return self.hard_horizontal_radius_m + self.preferred_extra_m

    @property
    def vertical_half_extent_m(self) -> float:
        return 0.5 * self.body_height_z_m + self.vertical_tracking_margin_m

    @classmethod
    def from_city_map_defaults(
        cls, city_map: CityMap, *, base: "VehicleEnvelope3D | None" = None
    ) -> "VehicleEnvelope3D":
        """Match aggregate hard/preferred radii recorded by a map asset."""

        template = base or cls()
        with city_map.source_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        defaults = document.get("planner_defaults", {})
        body = defaults.get(
            "vehicle_minimum_xy_size_m",
            [template.body_length_x_m, template.body_width_y_m],
        )
        if len(body) != 2:
            raise ValueError("vehicle_minimum_xy_size_m must contain two values")
        length = max(float(body[0]), template.body_length_x_m)
        width = max(float(body[1]), template.body_width_y_m)
        body_radius = 0.5 * math.hypot(max(1.0, length), max(1.0, width))
        fixed_margins = (
            template.map_margin_m
            + template.localization_margin_m
            + template.tracking_margin_m
            + template.control_margin_m
        )
        hard = float(
            defaults.get(
                "hard_inflation_radius_m",
                body_radius + fixed_margins + template.extra_margin_m,
            )
        )
        preferred = float(
            defaults.get(
                "preferred_clearance_m", hard + template.preferred_extra_m
            )
        )
        extra = hard - body_radius - fixed_margins
        if extra < -_EPS or preferred < hard - _EPS:
            raise ValueError("map planner defaults contain an invalid safety budget")
        result = replace(
            template,
            body_length_x_m=length,
            body_width_y_m=width,
            extra_margin_m=max(0.0, extra),
            preferred_extra_m=max(0.0, preferred - hard),
        )
        if abs(result.hard_horizontal_radius_m - hard) > 1.0e-9:
            raise ValueError("failed to reproduce map hard-inflation radius")
        return result


def _point3(value: Sequence[float], name: str = "point") -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _point2(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value[:2], dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError("point must contain at least two finite values")
    return result


def _point_segment_distance_2d(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    p = _point2(point)
    a = _point2(start)
    b = _point2(end)
    delta = b - a
    length_squared = float(np.dot(delta, delta))
    if length_squared <= _EPS:
        return float(np.linalg.norm(p - a))
    ratio = float(np.clip(np.dot(p - a, delta) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ratio * delta)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return abs(_orientation(a, b, point)) <= _EPS and bool(
        np.all(point >= np.minimum(a, b) - _EPS)
        and np.all(point <= np.maximum(a, b) + _EPS)
    )


def _segments_intersect_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if ((ab_c > _EPS and ab_d < -_EPS) or (ab_c < -_EPS and ab_d > _EPS)) and (
        (cd_a > _EPS and cd_b < -_EPS) or (cd_a < -_EPS and cd_b > _EPS)
    ):
        return True
    return (
        (abs(ab_c) <= _EPS and _point_on_segment_2d(c, a, b))
        or (abs(ab_d) <= _EPS and _point_on_segment_2d(d, a, b))
        or (abs(cd_a) <= _EPS and _point_on_segment_2d(a, c, d))
        or (abs(cd_b) <= _EPS and _point_on_segment_2d(b, c, d))
    )


def _ring_edges(ring: Sequence[Sequence[float]]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for index in range(len(ring)):
        yield _point2(ring[index]), _point2(ring[(index + 1) % len(ring)])


def _segment_segment_distance_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    if _segments_intersect_2d(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance_2d(a, c, d),
        _point_segment_distance_2d(b, c, d),
        _point_segment_distance_2d(c, a, b),
        _point_segment_distance_2d(d, a, b),
    )


def _distance_to_solid_footprint(building: PolygonPrism, point_xy: Sequence[float]) -> float:
    """Euclidean XY distance to the building's solid footprint.

    A point in a polygon hole is not occupied; its distance is measured to the
    hole boundary.  This makes atria remain usable until horizontal inflation
    closes them.
    """

    if building.contains_xy(point_xy):
        return 0.0
    rings = (building.outer, *building.holes)
    return min(
        _point_segment_distance_2d(point_xy, start, end)
        for ring in rings
        for start, end in _ring_edges(ring)
    )


def _rectangle_distance_to_footprint(
    building: PolygonPrism, minimum_xy: np.ndarray, maximum_xy: np.ndarray
) -> float:
    corners = (
        np.asarray((minimum_xy[0], minimum_xy[1])),
        np.asarray((maximum_xy[0], minimum_xy[1])),
        np.asarray((maximum_xy[0], maximum_xy[1])),
        np.asarray((minimum_xy[0], maximum_xy[1])),
    )
    if any(building.contains_xy(corner) for corner in corners):
        return 0.0

    def inside_rectangle(point: Sequence[float]) -> bool:
        p = _point2(point)
        return bool(np.all(p >= minimum_xy - _EPS) and np.all(p <= maximum_xy + _EPS))

    # An outer-ring vertex in the rectangle proves solid-area intersection.
    if any(inside_rectangle(vertex) for vertex in building.outer):
        return 0.0

    rectangle_edges = tuple(_ring_edges(corners))
    boundary_rings = (building.outer, *building.holes)
    minimum_distance = math.inf
    for ring in boundary_rings:
        for polygon_start, polygon_end in _ring_edges(ring):
            for rect_start, rect_end in rectangle_edges:
                minimum_distance = min(
                    minimum_distance,
                    _segment_segment_distance_2d(
                        polygon_start, polygon_end, rect_start, rect_end
                    ),
                )
                if minimum_distance <= _EPS:
                    return 0.0
    return float(minimum_distance)


@dataclass(frozen=True)
class PrismOccupancy3D:
    """Exact 2.5-D polygon-prism occupancy with 3-D query semantics.

    Points outside the geofence are treated as occupied by both occupancy
    methods.  ``vertical_inflation_up_m`` extends obstacle tops upward;
    ``vertical_inflation_down_m`` extends their bottoms downward.  The ground
    is a horizontal obstacle whose upward inflation is also applied.
    """

    city_map: CityMap
    minimum_z_m: float = 0.0
    maximum_z_m: float = 120.0
    ground_z_m: float = 0.0
    horizontal_inflation_m: float = 0.0
    vertical_inflation_up_m: float = 0.0
    vertical_inflation_down_m: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_z_m,
            self.maximum_z_m,
            self.ground_z_m,
            self.horizontal_inflation_m,
            self.vertical_inflation_up_m,
            self.vertical_inflation_down_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("occupancy parameters must be finite")
        if self.minimum_z_m >= self.maximum_z_m:
            raise ValueError("minimum_z_m must be smaller than maximum_z_m")
        if not self.minimum_z_m <= self.ground_z_m < self.maximum_z_m:
            raise ValueError("ground_z_m must lie inside the vertical geofence")
        if min(
            self.horizontal_inflation_m,
            self.vertical_inflation_up_m,
            self.vertical_inflation_down_m,
        ) < 0.0:
            raise ValueError("occupancy inflation cannot be negative")
        if self.city_map.buildings:
            bounds = np.asarray(
                [building.bounds_xy for building in self.city_map.buildings],
                dtype=float,
            )
            foundations = np.asarray(
                [building.foundation_z_m for building in self.city_map.buildings],
                dtype=float,
            )
            roofs = np.asarray(
                [building.roof_z_m for building in self.city_map.buildings],
                dtype=float,
            )
        else:
            bounds = np.empty((0, 4), dtype=float)
            foundations = np.empty(0, dtype=float)
            roofs = np.empty(0, dtype=float)
        object.__setattr__(self, "_building_bounds_xy", bounds)
        object.__setattr__(self, "_building_foundations_z_m", foundations)
        object.__setattr__(self, "_building_roofs_z_m", roofs)

    @classmethod
    def from_city_map(
        cls,
        city_map: CityMap,
        *,
        minimum_z_m: float = 0.0,
        maximum_z_m: float = 120.0,
        ground_z_m: float = 0.0,
        drone_collision_radius_m: float = 0.0,
        map_uncertainty_m: float = 0.0,
        localization_uncertainty_m: float = 0.0,
        tracking_error_margin_m: float = 0.0,
        control_error_margin_m: float = 0.0,
        configured_safety_margin_m: float = 0.0,
        vertical_inflation_up_m: float = 0.0,
        vertical_inflation_down_m: float = 0.0,
    ) -> "PrismOccupancy3D":
        horizontal_terms = (
            drone_collision_radius_m,
            map_uncertainty_m,
            localization_uncertainty_m,
            tracking_error_margin_m,
            control_error_margin_m,
            configured_safety_margin_m,
        )
        if min(horizontal_terms) < 0.0:
            raise ValueError("horizontal inflation components cannot be negative")
        return cls(
            city_map=city_map,
            minimum_z_m=float(minimum_z_m),
            maximum_z_m=float(maximum_z_m),
            ground_z_m=float(ground_z_m),
            horizontal_inflation_m=float(sum(horizontal_terms)),
            vertical_inflation_up_m=float(vertical_inflation_up_m),
            vertical_inflation_down_m=float(vertical_inflation_down_m),
        )

    @classmethod
    def from_vehicle_envelope(
        cls,
        city_map: CityMap,
        envelope: VehicleEnvelope3D,
        *,
        minimum_z_m: float = 0.0,
        maximum_z_m: float = 120.0,
        ground_z_m: float = 0.0,
        minimum_surface_clearance_m: float = 0.0,
    ) -> "PrismOccupancy3D":
        """Create hard occupancy from the complete vehicle safety envelope.

        ``minimum_surface_clearance_m`` expands the top of the ground and every
        building prism.  It therefore enforces clearance below the complete
        vehicle, rather than treating a world-z setpoint as AGL.
        """

        vertical = envelope.vertical_half_extent_m
        surface_clearance = float(minimum_surface_clearance_m)
        if not math.isfinite(surface_clearance) or surface_clearance < 0.0:
            raise ValueError("minimum surface clearance must be finite and non-negative")
        return cls(
            city_map=city_map,
            minimum_z_m=float(minimum_z_m),
            maximum_z_m=float(maximum_z_m),
            ground_z_m=float(ground_z_m),
            horizontal_inflation_m=envelope.hard_horizontal_radius_m,
            vertical_inflation_up_m=vertical + surface_clearance,
            vertical_inflation_down_m=vertical,
        )

    def is_inside_map(self, point: Sequence[float]) -> bool:
        """Return whether the *centre point* lies in the raw map geofence."""

        p = _point3(point)
        return bool(
            self.city_map.bounds_x_m[0] <= p[0] < self.city_map.bounds_x_m[1]
            and self.city_map.bounds_y_m[0] <= p[1] < self.city_map.bounds_y_m[1]
            and self.minimum_z_m <= p[2] < self.maximum_z_m
        )

    def is_inside_inflated_geofence(self, point: Sequence[float]) -> bool:
        """Return whether the full inflated vehicle envelope is in-bounds."""

        p = _point3(point)
        horizontal = self.horizontal_inflation_m
        return bool(
            self.city_map.bounds_x_m[0] + horizontal < p[0]
            < self.city_map.bounds_x_m[1] - horizontal
            and self.city_map.bounds_y_m[0] + horizontal < p[1]
            < self.city_map.bounds_y_m[1] - horizontal
            and self.minimum_z_m + self.vertical_inflation_up_m < p[2]
            < self.maximum_z_m - self.vertical_inflation_down_m
        )

    def _building_contains(
        self, building: PolygonPrism, point: np.ndarray, inflated: bool
    ) -> bool:
        horizontal = self.horizontal_inflation_m if inflated else 0.0
        lower = building.foundation_z_m - (
            self.vertical_inflation_down_m if inflated else 0.0
        )
        upper = building.roof_z_m + (
            self.vertical_inflation_up_m if inflated else 0.0
        )
        return bool(
            lower - _EPS <= point[2] <= upper + _EPS
            and _distance_to_solid_footprint(building, point[:2]) <= horizontal + _EPS
        )

    def _point_candidate_indices(
        self, point: np.ndarray, *, inflated: bool
    ) -> np.ndarray:
        if len(self.city_map.buildings) == 0:
            return np.empty(0, dtype=int)
        horizontal = self.horizontal_inflation_m if inflated else 0.0
        up = self.vertical_inflation_up_m if inflated else 0.0
        down = self.vertical_inflation_down_m if inflated else 0.0
        bounds = self._building_bounds_xy
        mask = (
            (point[0] >= bounds[:, 0] - horizontal - _EPS)
            & (point[0] <= bounds[:, 2] + horizontal + _EPS)
            & (point[1] >= bounds[:, 1] - horizontal - _EPS)
            & (point[1] <= bounds[:, 3] + horizontal + _EPS)
            & (point[2] >= self._building_foundations_z_m - down - _EPS)
            & (point[2] <= self._building_roofs_z_m + up + _EPS)
        )
        return np.flatnonzero(mask)

    def is_occupied(self, point: Sequence[float]) -> bool:
        p = _point3(point)
        if not self.is_inside_map(p):
            return True
        if p[2] <= self.ground_z_m + _EPS:
            return True
        return any(
            self._building_contains(self.city_map.buildings[index], p, False)
            for index in self._point_candidate_indices(p, inflated=False)
        )

    def is_inflated_occupied(self, point: Sequence[float]) -> bool:
        p = _point3(point)
        if not self.is_inside_inflated_geofence(p):
            return True
        if p[2] <= self.ground_z_m + self.vertical_inflation_up_m + _EPS:
            return True
        return any(
            self._building_contains(self.city_map.buildings[index], p, True)
            for index in self._point_candidate_indices(p, inflated=True)
        )

    def distance_to_obstacle(
        self,
        point: Sequence[float],
        *,
        inflated: bool = True,
        maximum_distance_m: float | None = None,
    ) -> float:
        """Distance to the nearest obstacle or geofence boundary.

        The return value is zero for occupied/out-of-bounds points.  Including
        the boundary in this ESDF-like query lets A* use the same clearance
        penalty for buildings, ground, ceiling, and lateral geofence limits.
        """

        p = _point3(point)
        if maximum_distance_m is not None and (
            not math.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0
        ):
            raise ValueError("maximum_distance_m must be finite and positive")
        occupied = self.is_inflated_occupied(p) if inflated else self.is_occupied(p)
        if occupied:
            return 0.0
        horizontal = self.horizontal_inflation_m if inflated else 0.0
        up = self.vertical_inflation_up_m if inflated else 0.0
        down = self.vertical_inflation_down_m if inflated else 0.0
        distances = [
            p[0] - (self.city_map.bounds_x_m[0] + horizontal),
            (self.city_map.bounds_x_m[1] - horizontal) - p[0],
            p[1] - (self.city_map.bounds_y_m[0] + horizontal),
            (self.city_map.bounds_y_m[1] - horizontal) - p[1],
            p[2] - (self.ground_z_m + up),
            (self.maximum_z_m - down) - p[2],
        ]
        best = float(min(distances))
        if maximum_distance_m is not None:
            best = min(best, float(maximum_distance_m))
        bounds = self._building_bounds_xy
        if len(bounds):
            dx = np.maximum.reduce((bounds[:, 0] - p[0], np.zeros(len(bounds)), p[0] - bounds[:, 2]))
            dy = np.maximum.reduce((bounds[:, 1] - p[1], np.zeros(len(bounds)), p[1] - bounds[:, 3]))
            horizontal_lower = np.maximum(0.0, np.hypot(dx, dy) - horizontal)
            lower = self._building_foundations_z_m - down
            upper = self._building_roofs_z_m + up
            vertical_lower = np.maximum.reduce((lower - p[2], np.zeros(len(bounds)), p[2] - upper))
            lower_bounds = np.hypot(horizontal_lower, vertical_lower)
            bounded_candidates = np.flatnonzero(lower_bounds < best - _EPS)
            candidate_order = bounded_candidates[
                np.argsort(lower_bounds[bounded_candidates])
            ]
        else:
            lower_bounds = np.empty(0)
            candidate_order = np.empty(0, dtype=int)
        for index in candidate_order:
            if lower_bounds[index] >= best - _EPS:
                break
            building = self.city_map.buildings[int(index)]
            horizontal_distance = max(
                0.0, _distance_to_solid_footprint(building, p[:2]) - horizontal
            )
            lower = building.foundation_z_m - down
            upper = building.roof_z_m + up
            if p[2] < lower:
                vertical_distance = lower - p[2]
            elif p[2] > upper:
                vertical_distance = p[2] - upper
            else:
                vertical_distance = 0.0
            best = min(best, math.hypot(horizontal_distance, vertical_distance))
        return max(0.0, best)

    def highest_surface_below(
        self, x_m: float, y_m: float, z_m: float, *, inflated: bool = False
    ) -> float:
        """Return the highest mapped horizontal surface not above ``z_m``."""

        point = np.asarray((x_m, y_m, z_m), dtype=float)
        if not np.all(np.isfinite(point)) or not (
            self.city_map.bounds_x_m[0] <= x_m < self.city_map.bounds_x_m[1]
            and self.city_map.bounds_y_m[0] <= y_m < self.city_map.bounds_y_m[1]
        ):
            raise ValueError("highest_surface_below query lies outside the XY map")
        horizontal = self.horizontal_inflation_m if inflated else 0.0
        up = self.vertical_inflation_up_m if inflated else 0.0
        surfaces = [self.ground_z_m + up]
        candidate_point = np.asarray((x_m, y_m, z_m), dtype=float)
        bounds = self._building_bounds_xy
        if len(bounds):
            candidate_mask = (
                (x_m >= bounds[:, 0] - horizontal - _EPS)
                & (x_m <= bounds[:, 2] + horizontal + _EPS)
                & (y_m >= bounds[:, 1] - horizontal - _EPS)
                & (y_m <= bounds[:, 3] + horizontal + _EPS)
            )
            candidates = np.flatnonzero(candidate_mask)
        else:
            candidates = np.empty(0, dtype=int)
        for index in candidates:
            building = self.city_map.buildings[int(index)]
            roof = building.roof_z_m + up
            if roof <= z_m + _EPS and _distance_to_solid_footprint(
                building, candidate_point[:2]
            ) <= horizontal + _EPS:
                surfaces.append(roof)
        return float(max(surface for surface in surfaces if surface <= z_m + _EPS))

    def segment_is_free(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        inflated: bool = True,
        sample_step_m: float = 0.25,
    ) -> bool:
        if sample_step_m <= 0.0:
            raise ValueError("sample_step_m must be positive")
        a = _point3(start, "segment start")
        b = _point3(end, "segment end")
        distance = float(np.linalg.norm(b - a))
        count = max(1, int(math.ceil(distance / sample_step_m)))
        query = self.is_inflated_occupied if inflated else self.is_occupied
        return all(
            not query(a + ratio * (b - a))
            for ratio in np.linspace(0.0, 1.0, count + 1)
        )

    def segment_minimum_clearance(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        inflated: bool = True,
        sample_step_m: float = 0.25,
    ) -> float:
        """Sample the ESDF-like clearance along a closed line segment."""

        if sample_step_m <= 0.0:
            raise ValueError("sample_step_m must be positive")
        a = _point3(start, "segment start")
        b = _point3(end, "segment end")
        count = max(1, int(math.ceil(np.linalg.norm(b - a) / sample_step_m)))
        return float(
            min(
                self.distance_to_obstacle(a + ratio * (b - a), inflated=inflated)
                for ratio in np.linspace(0.0, 1.0, count + 1)
            )
        )

    def box_is_free(
        self,
        minimum_xyz: Sequence[float],
        maximum_xyz: Sequence[float],
        *,
        inflated: bool = True,
    ) -> bool:
        """Conservatively test a closed axis-aligned box against the map."""

        minimum = _point3(minimum_xyz, "box minimum")
        maximum = _point3(maximum_xyz, "box maximum")
        if np.any(maximum <= minimum):
            raise ValueError("box maximum must be greater than its minimum")
        horizontal = self.horizontal_inflation_m if inflated else 0.0
        up = self.vertical_inflation_up_m if inflated else 0.0
        down = self.vertical_inflation_down_m if inflated else 0.0
        if not (
            minimum[0] > self.city_map.bounds_x_m[0] + horizontal + _EPS
            and maximum[0] < self.city_map.bounds_x_m[1] - horizontal - _EPS
            and minimum[1] > self.city_map.bounds_y_m[0] + horizontal + _EPS
            and maximum[1] < self.city_map.bounds_y_m[1] - horizontal - _EPS
            and minimum[2] > self.minimum_z_m + up + _EPS
            and maximum[2] < self.maximum_z_m - down - _EPS
        ):
            return False
        if minimum[2] <= self.ground_z_m + up + _EPS:
            return False
        bounds = self._building_bounds_xy
        if len(bounds):
            z_overlap = ~(
                (maximum[2] < self._building_foundations_z_m - down - _EPS)
                | (minimum[2] > self._building_roofs_z_m + up + _EPS)
            )
            xy_overlap = ~(
                (maximum[0] < bounds[:, 0] - horizontal - _EPS)
                | (minimum[0] > bounds[:, 2] + horizontal + _EPS)
                | (maximum[1] < bounds[:, 1] - horizontal - _EPS)
                | (minimum[1] > bounds[:, 3] + horizontal + _EPS)
            )
            candidates = np.flatnonzero(z_overlap & xy_overlap)
        else:
            candidates = np.empty(0, dtype=int)
        for index in candidates:
            building = self.city_map.buildings[int(index)]
            lower = building.foundation_z_m - down
            upper = building.roof_z_m + up
            # Mere contact is conservatively considered collision.
            if maximum[2] < lower - _EPS or minimum[2] > upper + _EPS:
                continue
            footprint_distance = _rectangle_distance_to_footprint(
                building, minimum[:2], maximum[:2]
            )
            if footprint_distance <= horizontal + _EPS:
                return False
        return True


@dataclass(frozen=True)
class AStar3DConfig:
    resolution_m: tuple[float, float, float] | float = (2.0, 2.0, 2.0)
    connectivity: int = 26
    length_weight: float = 1.0
    vertical_weight: float = 1.0
    clearance_weight: float = 1.0
    # Clearance is measured *outside* the hard-inflated vehicle envelope.
    # ``safe_clearance_m`` is retained as a backward-compatible alias.
    safe_clearance_m: float = 4.0
    preferred_clearance_m: float | None = None
    minimum_clearance_m: float = 0.0
    climb_weight: float = 0.0
    snap_radius_m: float = 4.0
    planning_timeout_s: float = 5.0
    max_expanded_nodes: int = 500_000
    edge_sample_step_m: float = 0.4

    def __post_init__(self) -> None:
        resolution = np.asarray(self.resolution_m, dtype=float)
        if resolution.ndim == 0:
            resolution = np.repeat(resolution, 3)
        if resolution.shape != (3,) or np.any(~np.isfinite(resolution)) or np.any(resolution <= 0.0):
            raise ValueError("resolution_m must be one or three positive values")
        object.__setattr__(self, "resolution_m", tuple(float(v) for v in resolution))
        preferred = (
            self.safe_clearance_m
            if self.preferred_clearance_m is None
            else self.preferred_clearance_m
        )
        if not math.isfinite(preferred):
            raise ValueError("preferred clearance must be finite")
        object.__setattr__(self, "preferred_clearance_m", float(preferred))
        if self.connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18, or 26")
        if self.length_weight <= 0.0 or self.vertical_weight <= 0.0:
            raise ValueError("length and vertical weights must be positive")
        if min(
            self.clearance_weight,
            preferred,
            self.minimum_clearance_m,
            self.climb_weight,
            self.snap_radius_m,
        ) < 0.0:
            raise ValueError("A* costs, clearances, and snap radius cannot be negative")
        if self.minimum_clearance_m > preferred and preferred > 0.0:
            raise ValueError("minimum clearance cannot exceed preferred clearance")
        if self.planning_timeout_s <= 0.0 or self.max_expanded_nodes <= 0:
            raise ValueError("planning limits must be positive")
        if self.edge_sample_step_m <= 0.0:
            raise ValueError("edge sample step must be positive")


@dataclass(frozen=True)
class AStar3DMetrics:
    success: bool
    reason: str
    elapsed_s: float
    expanded_nodes: int
    generated_nodes: int
    path_length_m: float
    minimum_clearance_m: float
    start_requested_enu_m: tuple[float, float, float]
    goal_requested_enu_m: tuple[float, float, float]
    start_used_enu_m: tuple[float, float, float] | None
    goal_used_enu_m: tuple[float, float, float] | None
    start_snapped: bool
    goal_snapped: bool
    open_set_peak: int = 0
    cache_hit: bool = False


class SearchBand3D:
    """Sparse fine-search restriction represented by a sampled path KD-tree."""

    def __init__(
        self,
        centerline_enu_m: Sequence[Sequence[float]],
        radius_m: float,
        *,
        sample_step_m: float | None = None,
        vertical_radius_m: float | None = None,
    ) -> None:
        path = np.asarray(centerline_enu_m, dtype=float)
        if (
            path.ndim != 2
            or path.shape[1] != 3
            or len(path) < 2
            or not np.all(np.isfinite(path))
        ):
            raise ValueError("search-band centerline must have shape (N>=2, 3)")
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("search-band radius must be positive")
        vertical_radius = radius_m if vertical_radius_m is None else vertical_radius_m
        if not math.isfinite(vertical_radius) or vertical_radius <= 0.0:
            raise ValueError("vertical search-band radius must be positive")
        step = min(radius_m * 0.25, 1.0) if sample_step_m is None else sample_step_m
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("search-band sample step must be positive")
        samples: list[np.ndarray] = [path[0]]
        for start, end in zip(path, path[1:]):
            distance = float(np.linalg.norm(end - start))
            count = max(1, int(math.ceil(distance / step)))
            samples.extend(
                start + ratio * (end - start)
                for ratio in np.linspace(0.0, 1.0, count + 1)[1:]
            )
        self.centerline_enu_m = path
        self.radius_m = float(radius_m)
        self.vertical_radius_m = float(vertical_radius)
        self.sample_step_m = float(step)
        self._z_scale = self.radius_m / self.vertical_radius_m
        scaled_samples = np.asarray(samples).copy()
        scaled_samples[:, 2] *= self._z_scale
        self._tree = spatial.cKDTree(scaled_samples)
        # Sampling error cannot exclude a point that is exactly radius_m from
        # the continuous polyline.
        self._query_radius_m = self.radius_m + 0.5 * self.sample_step_m

    def contains(self, point_enu_m: Sequence[float]) -> bool:
        point = _point3(point_enu_m).copy()
        point[2] *= self._z_scale
        distance, _ = self._tree.query(point, k=1)
        return bool(distance <= self._query_radius_m + _EPS)


@dataclass(frozen=True)
class AStar3DResult:
    path_enu_m: np.ndarray
    metrics: AStar3DMetrics

    @property
    def success(self) -> bool:
        return self.metrics.success


class AStarPlanner3D:
    """Heap-based 3-D A* with 6/18/26 connectivity and supercover checks."""

    def __init__(
        self,
        occupancy: PrismOccupancy3D,
        config: AStar3DConfig | None = None,
        *,
        search_band: SearchBand3D | None = None,
    ):
        self.occupancy = occupancy
        self.config = config or AStar3DConfig()
        self.search_band = search_band
        self.resolution = np.asarray(self.config.resolution_m, dtype=float)
        self.origin = np.asarray(
            (
                occupancy.city_map.bounds_x_m[0],
                occupancy.city_map.bounds_y_m[0],
                occupancy.minimum_z_m,
            ),
            dtype=float,
        )
        extents = np.asarray(
            (
                occupancy.city_map.bounds_x_m[1] - occupancy.city_map.bounds_x_m[0],
                occupancy.city_map.bounds_y_m[1] - occupancy.city_map.bounds_y_m[0],
                occupancy.maximum_z_m - occupancy.minimum_z_m,
            )
        )
        self.shape = tuple(int(math.ceil(v)) for v in extents / self.resolution)
        self._free_cache: dict[tuple[int, int, int], bool] = {}
        self._clearance_cache: dict[tuple[int, int, int], float] = {}
        candidates = []
        for delta in itertools.product((-1, 0, 1), repeat=3):
            changed = sum(component != 0 for component in delta)
            if changed == 0:
                continue
            if self.config.connectivity == 6 and changed != 1:
                continue
            if self.config.connectivity == 18 and changed == 3:
                continue
            candidates.append(delta)
        self.neighbors = tuple(sorted(candidates, key=lambda d: (sum(v != 0 for v in d), d)))

    def contains_voxel(self, voxel: Sequence[int]) -> bool:
        return all(0 <= int(voxel[axis]) < self.shape[axis] for axis in range(3))

    def world_to_voxel(self, point: Sequence[float]) -> tuple[int, int, int]:
        p = _point3(point)
        if not self.occupancy.is_inside_map(p):
            raise ValueError("point lies outside the 3-D planning geofence")
        index = np.floor((p - self.origin) / self.resolution).astype(int)
        index = np.minimum(index, np.asarray(self.shape) - 1)
        return tuple(int(v) for v in index)

    def voxel_to_world(self, voxel: Sequence[int]) -> np.ndarray:
        cell = tuple(int(v) for v in voxel)
        if not self.contains_voxel(cell):
            raise ValueError("voxel lies outside the 3-D planning grid")
        point = self.origin + (np.asarray(cell, dtype=float) + 0.5) * self.resolution
        maxima = np.asarray(
            (
                self.occupancy.city_map.bounds_x_m[1],
                self.occupancy.city_map.bounds_y_m[1],
                self.occupancy.maximum_z_m,
            )
        )
        return np.minimum(point, maxima - 1.0e-7)

    def voxel_is_free(self, voxel: Sequence[int]) -> bool:
        cell = tuple(int(v) for v in voxel)
        if not self.contains_voxel(cell):
            return False
        if cell not in self._free_cache:
            point = self.voxel_to_world(cell)
            self._free_cache[cell] = bool(
                (self.search_band is None or self.search_band.contains(point))
                and not self.occupancy.is_inflated_occupied(point)
            )
        return self._free_cache[cell]

    def voxel_clearance(self, voxel: Sequence[int]) -> float:
        cell = tuple(int(v) for v in voxel)
        if cell not in self._clearance_cache:
            clearance_cap = max(
                float(self.config.preferred_clearance_m),
                self.config.minimum_clearance_m,
            )
            self._clearance_cache[cell] = self.occupancy.distance_to_obstacle(
                self.voxel_to_world(cell),
                inflated=True,
                maximum_distance_m=clearance_cap if clearance_cap > 0.0 else None,
            )
        return self._clearance_cache[cell]

    def voxel_is_admissible(self, voxel: Sequence[int]) -> bool:
        return bool(
            self.voxel_is_free(voxel)
            and self.voxel_clearance(voxel)
            > self.config.minimum_clearance_m + _EPS
        )

    def _snap(self, requested: np.ndarray) -> tuple[tuple[int, int, int] | None, bool]:
        try:
            original = self.world_to_voxel(requested)
        except ValueError:
            return None, False
        requested_clearance = self.occupancy.distance_to_obstacle(
            requested, inflated=True
        )
        if (
            not self.occupancy.is_inflated_occupied(requested)
            and requested_clearance > self.config.minimum_clearance_m + _EPS
            and self.voxel_is_admissible(original)
        ):
            return original, False
        if self.config.snap_radius_m <= 0.0:
            return None, False
        radii = np.ceil(self.config.snap_radius_m / self.resolution).astype(int)
        candidates: list[tuple[float, tuple[int, int, int]]] = []
        for offset in itertools.product(
            range(-radii[0], radii[0] + 1),
            range(-radii[1], radii[1] + 1),
            range(-radii[2], radii[2] + 1),
        ):
            cell = tuple(original[axis] + offset[axis] for axis in range(3))
            if not self.contains_voxel(cell) or not self.voxel_is_admissible(cell):
                continue
            distance = float(np.linalg.norm(self.voxel_to_world(cell) - requested))
            if distance <= self.config.snap_radius_m + _EPS:
                candidates.append((distance, cell))
        if not candidates:
            return None, False
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1], True

    def _transition_is_free(
        self, current: tuple[int, int, int], neighbor: tuple[int, int, int]
    ) -> bool:
        delta = tuple(neighbor[axis] - current[axis] for axis in range(3))
        changed_axes = tuple(axis for axis, value in enumerate(delta) if value != 0)
        # Every proper non-empty subset is part of the 3-D supercover.  Requiring
        # those voxels to be free prevents both edge and corner cutting.
        if len(changed_axes) > 1:
            for subset_size in range(1, len(changed_axes)):
                for subset in itertools.combinations(changed_axes, subset_size):
                    intermediate = list(current)
                    for axis in subset:
                        intermediate[axis] += delta[axis]
                    if not self.voxel_is_admissible(intermediate):
                        return False
        step = min(
            self.config.edge_sample_step_m, float(np.min(self.resolution)) * 0.25
        )
        if not self.occupancy.segment_is_free(
            self.voxel_to_world(current),
            self.voxel_to_world(neighbor),
            inflated=True,
            sample_step_m=step,
        ):
            return False
        return bool(
            self.config.minimum_clearance_m <= 0.0
            or self.occupancy.segment_minimum_clearance(
                self.voxel_to_world(current),
                self.voxel_to_world(neighbor),
                inflated=True,
                sample_step_m=step,
            )
            > self.config.minimum_clearance_m + _EPS
        )

    def _failure(
        self,
        reason: str,
        started: float,
        expanded: int,
        generated: int,
        start: np.ndarray,
        goal: np.ndarray,
        start_cell: tuple[int, int, int] | None,
        goal_cell: tuple[int, int, int] | None,
        start_snapped: bool,
        goal_snapped: bool,
        open_set_peak: int = 0,
    ) -> AStar3DResult:
        return AStar3DResult(
            path_enu_m=np.empty((0, 3), dtype=float),
            metrics=AStar3DMetrics(
                success=False,
                reason=reason,
                elapsed_s=time.monotonic() - started,
                expanded_nodes=expanded,
                generated_nodes=generated,
                path_length_m=0.0,
                minimum_clearance_m=0.0,
                start_requested_enu_m=tuple(start),
                goal_requested_enu_m=tuple(goal),
                start_used_enu_m=(
                    tuple(self.voxel_to_world(start_cell)) if start_cell is not None else None
                ),
                goal_used_enu_m=(
                    tuple(self.voxel_to_world(goal_cell)) if goal_cell is not None else None
                ),
                start_snapped=start_snapped,
                goal_snapped=goal_snapped,
                open_set_peak=open_set_peak,
            ),
        )

    def plan(self, start_enu_m: Sequence[float], goal_enu_m: Sequence[float]) -> AStar3DResult:
        started = time.monotonic()
        start = _point3(start_enu_m, "A* start")
        goal = _point3(goal_enu_m, "A* goal")
        if not self.occupancy.is_inside_map(start):
            return self._failure("start_outside_map", started, 0, 0, start, goal, None, None, False, False)
        if not self.occupancy.is_inside_map(goal):
            return self._failure("goal_outside_map", started, 0, 0, start, goal, None, None, False, False)
        start_cell, start_snapped = self._snap(start)
        goal_cell, goal_snapped = self._snap(goal)
        if start_cell is None:
            return self._failure("start_snap_failed", started, 0, 0, start, goal, None, goal_cell, False, goal_snapped)
        if goal_cell is None:
            return self._failure("goal_snap_failed", started, 0, 0, start, goal, start_cell, None, start_snapped, False)

        def heuristic(cell: tuple[int, int, int]) -> float:
            delta = self.voxel_to_world(goal_cell) - self.voxel_to_world(cell)
            delta[2] *= self.config.vertical_weight
            return self.config.length_weight * float(np.linalg.norm(delta))

        g_score: dict[tuple[int, int, int], float] = {start_cell: 0.0}
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        queue: list[tuple[float, float, int, float, tuple[int, int, int]]] = []
        sequence = 0
        start_h = heuristic(start_cell)
        heapq.heappush(queue, (start_h, start_h, sequence, 0.0, start_cell))
        closed: set[tuple[int, int, int]] = set()
        expanded = 0
        generated = 1
        open_set_peak = 1
        found = False

        while queue:
            if time.monotonic() - started > self.config.planning_timeout_s:
                return self._failure("planning_timeout", started, expanded, generated, start, goal, start_cell, goal_cell, start_snapped, goal_snapped, open_set_peak)
            _, _, _, queued_g, current = heapq.heappop(queue)
            if current in closed or queued_g > g_score.get(current, math.inf) + _EPS:
                continue
            if current == goal_cell:
                found = True
                break
            if expanded >= self.config.max_expanded_nodes:
                return self._failure("expanded_node_limit", started, expanded, generated, start, goal, start_cell, goal_cell, start_snapped, goal_snapped, open_set_peak)
            closed.add(current)
            expanded += 1
            current_world = self.voxel_to_world(current)
            for delta in self.neighbors:
                neighbor = tuple(current[axis] + delta[axis] for axis in range(3))
                if neighbor in closed or not self.voxel_is_admissible(neighbor):
                    continue
                clearance = self.voxel_clearance(neighbor)
                if not self._transition_is_free(current, neighbor):
                    continue
                neighbor_world = self.voxel_to_world(neighbor)
                physical_delta = neighbor_world - current_world
                weighted_delta = physical_delta.copy()
                weighted_delta[2] *= self.config.vertical_weight
                edge_length = float(np.linalg.norm(weighted_delta))
                clearance_penalty = 0.0
                preferred = float(self.config.preferred_clearance_m)
                if preferred > _EPS and clearance < preferred:
                    ratio = (preferred - clearance) / preferred
                    clearance_penalty = self.config.clearance_weight * ratio * ratio * edge_length
                climb_penalty = self.config.climb_weight * max(0.0, physical_delta[2])
                tentative = (
                    g_score[current]
                    + self.config.length_weight * edge_length
                    + clearance_penalty
                    + climb_penalty
                )
                if tentative + _EPS < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative
                    parent[neighbor] = current
                    sequence += 1
                    neighbor_h = heuristic(neighbor)
                    heapq.heappush(
                        queue,
                        (tentative + neighbor_h, neighbor_h, sequence, tentative, neighbor),
                    )
                    open_set_peak = max(open_set_peak, len(queue))
                    generated += 1

        if not found:
            return self._failure("no_path", started, expanded, generated, start, goal, start_cell, goal_cell, start_snapped, goal_snapped, open_set_peak)

        cells = [goal_cell]
        while cells[-1] != start_cell:
            predecessor = parent.get(cells[-1])
            if predecessor is None:
                return self._failure("path_reconstruction_failed", started, expanded, generated, start, goal, start_cell, goal_cell, start_snapped, goal_snapped)
            cells.append(predecessor)
        cells.reverse()
        points = np.asarray([self.voxel_to_world(cell) for cell in cells])
        if not start_snapped:
            points[0] = start
        if not goal_snapped:
            points[-1] = goal
        if len(points) == 1 and not np.allclose(start, goal):
            points = np.vstack((start if not start_snapped else points[0], goal if not goal_snapped else points[0]))
        if any(
            (
                not self.occupancy.segment_is_free(
                    points[index],
                    points[index + 1],
                    inflated=True,
                    sample_step_m=self.config.edge_sample_step_m,
                )
                or (
                    self.config.minimum_clearance_m > 0.0
                    and self.occupancy.segment_minimum_clearance(
                        points[index],
                        points[index + 1],
                        inflated=True,
                        sample_step_m=self.config.edge_sample_step_m,
                    )
                    <= self.config.minimum_clearance_m + _EPS
                )
            )
            for index in range(len(points) - 1)
        ):
            return self._failure("final_segment_validation_failed", started, expanded, generated, start, goal, start_cell, goal_cell, start_snapped, goal_snapped)

        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        clearance_samples: list[float] = []
        for index in range(max(1, len(points) - 1)):
            a = points[min(index, len(points) - 1)]
            b = points[min(index + 1, len(points) - 1)]
            count = max(1, int(math.ceil(np.linalg.norm(b - a) / self.config.edge_sample_step_m)))
            clearance_samples.extend(
                self.occupancy.distance_to_obstacle(a + ratio * (b - a), inflated=True)
                for ratio in np.linspace(0.0, 1.0, count + 1)
            )
        minimum_clearance = float(min(clearance_samples)) if clearance_samples else 0.0
        metrics = AStar3DMetrics(
            success=True,
            reason="success",
            elapsed_s=time.monotonic() - started,
            expanded_nodes=expanded,
            generated_nodes=generated,
            path_length_m=path_length,
            minimum_clearance_m=minimum_clearance,
            start_requested_enu_m=tuple(start),
            goal_requested_enu_m=tuple(goal),
            start_used_enu_m=tuple(points[0]),
            goal_used_enu_m=tuple(points[-1]),
            start_snapped=start_snapped,
            goal_snapped=goal_snapped,
            open_set_peak=open_set_peak,
        )
        return AStar3DResult(path_enu_m=points, metrics=metrics)


@dataclass(frozen=True)
class HierarchicalAStar3DConfig:
    """Bounded coarse-to-fine search parameters for large sparse maps."""

    coarse: AStar3DConfig = AStar3DConfig(
        resolution_m=(6.0, 6.0, 6.0),
        connectivity=26,
        vertical_weight=3.0,
        climb_weight=0.5,
        safe_clearance_m=0.7,
        preferred_clearance_m=0.7,
        planning_timeout_s=8.0,
        max_expanded_nodes=200_000,
        edge_sample_step_m=1.5,
        # A 10 m underside-clearance contract places the first free coarse
        # layer at z=15 m while takeoff hands over near z=10.5 m.
        snap_radius_m=8.0,
    )
    fine: AStar3DConfig = AStar3DConfig(
        resolution_m=(2.0, 2.0, 2.0),
        connectivity=26,
        vertical_weight=3.0,
        climb_weight=0.5,
        safe_clearance_m=0.7,
        preferred_clearance_m=0.7,
        planning_timeout_s=15.0,
        max_expanded_nodes=350_000,
        edge_sample_step_m=0.5,
    )
    fine_search_band_m: float = 10.0
    fine_vertical_search_band_m: float = 1.5
    widened_search_band_scale: float = 1.75
    maximum_fine_attempts: int = 2
    cache_entries: int = 16

    def __post_init__(self) -> None:
        if self.coarse.connectivity != 26 or self.fine.connectivity != 26:
            raise ValueError("hierarchical planner requires 26-neighbor A*")
        if not math.isfinite(self.fine_search_band_m) or self.fine_search_band_m <= 0.0:
            raise ValueError("fine search band must be positive")
        if (
            not math.isfinite(self.fine_vertical_search_band_m)
            or self.fine_vertical_search_band_m <= 0.0
        ):
            raise ValueError("fine vertical search band must be positive")
        if (
            not math.isfinite(self.widened_search_band_scale)
            or self.widened_search_band_scale < 1.0
        ):
            raise ValueError("widened search-band scale must be at least one")
        if self.maximum_fine_attempts not in (1, 2):
            raise ValueError("maximum_fine_attempts must be one or two")
        if self.cache_entries < 0:
            raise ValueError("cache_entries cannot be negative")


@dataclass(frozen=True)
class HierarchicalAStar3DMetrics:
    success: bool
    reason: str
    elapsed_s: float
    coarse_elapsed_s: float
    fine_elapsed_s: float
    coarse_expanded_nodes: int
    fine_expanded_nodes: int
    fine_attempts: int
    final_search_band_m: float
    cache_hit: bool


@dataclass(frozen=True)
class HierarchicalAStar3DResult:
    path_enu_m: np.ndarray
    metrics: HierarchicalAStar3DMetrics
    coarse_result: AStar3DResult | None
    fine_result: AStar3DResult | None

    @property
    def success(self) -> bool:
        return self.metrics.success


class HierarchicalAStarPlanner3D:
    """Sparse two-stage 26-neighbor A* with bounded widening and LRU cache.

    The cache is intentionally planner-local: it is valid only while the
    supplied static occupancy object is unchanged.  Every hit is collision
    revalidated before reuse, so callers can safely invalidate or replace the
    occupancy when a static-map revision occurs.
    """

    def __init__(
        self,
        occupancy: PrismOccupancy3D,
        config: HierarchicalAStar3DConfig | None = None,
    ) -> None:
        self.occupancy = occupancy
        self.config = config or HierarchicalAStar3DConfig()
        self._cache: OrderedDict[
            tuple[tuple[int, int, int], tuple[int, int, int]], AStar3DResult
        ] = OrderedDict()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _cache_key(
        self, start: np.ndarray, goal: np.ndarray
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        planner = AStarPlanner3D(self.occupancy, self.config.fine)
        try:
            return planner.world_to_voxel(start), planner.world_to_voxel(goal)
        except ValueError:
            return None

    def _cached_result(
        self, key: tuple[tuple[int, int, int], tuple[int, int, int]], start: np.ndarray, goal: np.ndarray
    ) -> AStar3DResult | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        candidate = cached.path_enu_m.copy()
        if not cached.metrics.start_snapped:
            candidate[0] = start
        if not cached.metrics.goal_snapped:
            candidate[-1] = goal
        if any(
            (
                not self.occupancy.segment_is_free(
                    left,
                    right,
                    inflated=True,
                    sample_step_m=self.config.fine.edge_sample_step_m,
                )
                or (
                    self.config.fine.minimum_clearance_m > 0.0
                    and self.occupancy.segment_minimum_clearance(
                        left,
                        right,
                        inflated=True,
                        sample_step_m=self.config.fine.edge_sample_step_m,
                    )
                    <= self.config.fine.minimum_clearance_m + _EPS
                )
            )
            for left, right in zip(candidate, candidate[1:])
        ):
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        elapsed = 0.0
        metrics = replace(
            cached.metrics,
            reason="cache_hit",
            elapsed_s=elapsed,
            start_requested_enu_m=tuple(start),
            goal_requested_enu_m=tuple(goal),
            start_used_enu_m=tuple(candidate[0]),
            goal_used_enu_m=tuple(candidate[-1]),
            cache_hit=True,
        )
        return AStar3DResult(candidate, metrics)

    def _store(
        self,
        key: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
        result: AStar3DResult,
    ) -> None:
        if key is None or self.config.cache_entries == 0 or not result.success:
            return
        self._cache[key] = AStar3DResult(result.path_enu_m.copy(), result.metrics)
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.cache_entries:
            self._cache.popitem(last=False)

    def plan(
        self, start_enu_m: Sequence[float], goal_enu_m: Sequence[float]
    ) -> HierarchicalAStar3DResult:
        started = time.monotonic()
        start = _point3(start_enu_m, "hierarchical A* start")
        goal = _point3(goal_enu_m, "hierarchical A* goal")
        key = self._cache_key(start, goal)
        if key is not None:
            cached = self._cached_result(key, start, goal)
            if cached is not None:
                elapsed = time.monotonic() - started
                return HierarchicalAStar3DResult(
                    cached.path_enu_m,
                    HierarchicalAStar3DMetrics(
                        True,
                        "cache_hit",
                        elapsed,
                        0.0,
                        0.0,
                        0,
                        0,
                        0,
                        0.0,
                        True,
                    ),
                    None,
                    cached,
                )

        coarse = AStarPlanner3D(self.occupancy, self.config.coarse).plan(start, goal)
        if not coarse.success:
            return HierarchicalAStar3DResult(
                np.empty((0, 3)),
                HierarchicalAStar3DMetrics(
                    False,
                    f"coarse_{coarse.metrics.reason}",
                    time.monotonic() - started,
                    coarse.metrics.elapsed_s,
                    0.0,
                    coarse.metrics.expanded_nodes,
                    0,
                    0,
                    0.0,
                    False,
                ),
                coarse,
                None,
            )

        fine: AStar3DResult | None = None
        fine_elapsed = 0.0
        fine_expanded = 0
        final_band = self.config.fine_search_band_m
        attempts = 0
        for attempt in range(self.config.maximum_fine_attempts):
            attempts = attempt + 1
            final_band = self.config.fine_search_band_m * (
                self.config.widened_search_band_scale**attempt
            )
            band = SearchBand3D(
                coarse.path_enu_m,
                final_band,
                vertical_radius_m=self.config.fine_vertical_search_band_m
                * (self.config.widened_search_band_scale**attempt),
            )
            candidate = AStarPlanner3D(
                self.occupancy, self.config.fine, search_band=band
            ).plan(start, goal)
            fine_elapsed += candidate.metrics.elapsed_s
            fine_expanded += candidate.metrics.expanded_nodes
            fine = candidate
            if candidate.success:
                self._store(key, candidate)
                break
        assert fine is not None
        success = fine.success
        reason = "goal_reached" if success else f"fine_{fine.metrics.reason}"
        return HierarchicalAStar3DResult(
            fine.path_enu_m,
            HierarchicalAStar3DMetrics(
                success,
                reason,
                time.monotonic() - started,
                coarse.metrics.elapsed_s,
                fine_elapsed,
                coarse.metrics.expanded_nodes,
                fine_expanded,
                attempts,
                final_band,
                False,
            ),
            coarse,
            fine,
        )


@dataclass(frozen=True)
class CorridorBox3D:
    minimum_enu_m: tuple[float, float, float]
    maximum_enu_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _point3(self.minimum_enu_m, "corridor minimum")
        maximum = _point3(self.maximum_enu_m, "corridor maximum")
        if np.any(maximum <= minimum):
            raise ValueError("corridor box must have positive extent")

    @property
    def minimum(self) -> np.ndarray:
        return np.asarray(self.minimum_enu_m, dtype=float)

    @property
    def maximum(self) -> np.ndarray:
        return np.asarray(self.maximum_enu_m, dtype=float)

    def contains(self, point: Sequence[float], tolerance_m: float = 1.0e-7) -> bool:
        p = _point3(point)
        return bool(
            np.all(p >= self.minimum - tolerance_m)
            and np.all(p <= self.maximum + tolerance_m)
        )

    def overlap_extent(self, other: "CorridorBox3D") -> np.ndarray:
        return np.maximum(0.0, np.minimum(self.maximum, other.maximum) - np.maximum(self.minimum, other.minimum))

    def overlaps(self, other: "CorridorBox3D", minimum_extent_m: float = 0.0) -> bool:
        return bool(np.all(self.overlap_extent(other) + _EPS >= minimum_extent_m))


@dataclass(frozen=True)
class SFCMetrics3D:
    reason: str
    elapsed_s: float
    box_count: int
    subdivision_count: int
    expansion_attempts: int
    expansion_accepts: int
    minimum_actual_overlap_m: float


@dataclass(frozen=True)
class SafeFlightCorridor3D:
    boxes: tuple[CorridorBox3D, ...]
    centerline_enu_m: np.ndarray
    minimum_overlap_m: float
    hard_horizontal_radius_m: float | None = None
    hard_vertical_up_m: float | None = None
    hard_vertical_down_m: float | None = None
    metrics: SFCMetrics3D | None = None

    def __post_init__(self) -> None:
        if not self.boxes:
            raise ValueError("safe flight corridor needs at least one box")
        centerline = np.asarray(self.centerline_enu_m, dtype=float)
        if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 2:
            raise ValueError("corridor centerline must have shape (N>=2, 3)")
        if len(self.boxes) != len(centerline) - 1:
            raise ValueError("one corridor box is required per centerline segment")
        if self.minimum_overlap_m < 0.0:
            raise ValueError("minimum overlap cannot be negative")
        envelope = (
            self.hard_horizontal_radius_m,
            self.hard_vertical_up_m,
            self.hard_vertical_down_m,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in envelope):
            raise ValueError("corridor envelope values must be finite and non-negative")

    def contains(self, point: Sequence[float], tolerance_m: float = 1.0e-7) -> bool:
        return any(box.contains(point, tolerance_m) for box in self.boxes)

    def contains_path(
        self,
        path: Sequence[Sequence[float]],
        tolerance_m: float = 1.0e-7,
    ) -> bool:
        """Return whether every point lies in the union of the SFC boxes.

        Trajectory validation routinely checks several thousand samples
        against tens or hundreds of boxes.  Calling :meth:`contains` for each
        point made that exact same union test in Python, resulting in
        ``O(samples * boxes)`` Python/NumPy dispatch overhead.  The broadcast
        below evaluates the identical inclusive box inequalities in compiled
        NumPy code.  Chunking caps temporary storage for unusually long
        runtime trajectories without changing the predicate or tolerance.
        """

        points = np.asarray(path, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not math.isfinite(tolerance_m)
            or tolerance_m < 0.0
        ):
            return False
        if len(points) == 0:
            return True
        minimum = np.asarray([box.minimum_enu_m for box in self.boxes], dtype=float)
        maximum = np.asarray([box.maximum_enu_m for box in self.boxes], dtype=float)
        # About 1.5 MiB of boolean temporaries at the usual 99-box city SFC.
        chunk_size = max(1, min(len(points), 2048))
        for begin in range(0, len(points), chunk_size):
            chunk = points[begin : begin + chunk_size, None, :]
            inside = np.all(chunk >= minimum[None, :, :] - tolerance_m, axis=2)
            inside &= np.all(chunk <= maximum[None, :, :] + tolerance_m, axis=2)
            if not bool(np.all(np.any(inside, axis=1))):
                return False
        return True

    def box_indices_containing(self, point: Sequence[float]) -> tuple[int, ...]:
        return tuple(index for index, box in enumerate(self.boxes) if box.contains(point))

    def validate(self, occupancy: PrismOccupancy3D) -> tuple[bool, str]:
        expected = (
            occupancy.horizontal_inflation_m,
            occupancy.vertical_inflation_up_m,
            occupancy.vertical_inflation_down_m,
        )
        recorded = (
            self.hard_horizontal_radius_m,
            self.hard_vertical_up_m,
            self.hard_vertical_down_m,
        )
        if any(
            value is not None and abs(value - target) > _EPS
            for value, target in zip(recorded, expected)
        ):
            return False, "hard_envelope_mismatch"
        for index, box in enumerate(self.boxes):
            if not occupancy.box_is_free(box.minimum, box.maximum, inflated=True):
                return False, f"box_{index}_collides"
            if not box.contains(self.centerline_enu_m[index]) or not box.contains(
                self.centerline_enu_m[index + 1]
            ):
                return False, f"box_{index}_misses_centerline"
        for index, (left, right) in enumerate(zip(self.boxes, self.boxes[1:])):
            if not left.overlaps(right, self.minimum_overlap_m):
                return False, f"boxes_{index}_{index + 1}_lack_overlap"
        return True, "success"


def build_safe_flight_corridor_3d(
    path_enu_m: Sequence[Sequence[float]],
    occupancy: PrismOccupancy3D,
    *,
    minimum_overlap_m: float = 0.5,
    initial_margin_m: float = 0.5,
    maximum_margin_m: float = 5.0,
    expansion_step_m: float = 0.5,
    maximum_segment_length_m: float = 6.0,
    maximum_subdivision_depth: int = 14,
    maximum_expansion_iterations: int = 10_000,
    wall_clock_deadline_s: float = 5.0,
) -> SafeFlightCorridor3D:
    """Build collision-free overlapping axis-aligned convex boxes.

    Long or diagonal A* edges are subdivided before boxing.  If a padded box
    intersects an inflated obstacle, the segment is bisected recursively.  A
    failure after the configured depth means the requested overlap physically
    does not fit and is reported instead of silently creating an unsafe gap.
    """

    started = time.monotonic()
    points = np.asarray(path_enu_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("3-D SFC path must have shape (N>=2, 3)")
    if min(minimum_overlap_m, initial_margin_m) < 0.0:
        raise ValueError("corridor overlap and margin cannot be negative")
    if maximum_margin_m < initial_margin_m or expansion_step_m <= 0.0:
        raise ValueError("invalid corridor expansion parameters")
    if maximum_segment_length_m <= 0.0 or maximum_subdivision_depth < 0:
        raise ValueError("invalid corridor subdivision parameters")
    if maximum_expansion_iterations <= 0 or wall_clock_deadline_s <= 0.0:
        raise ValueError("SFC iteration and time limits must be positive")
    for index in range(len(points) - 1):
        if not occupancy.segment_is_free(points[index], points[index + 1], inflated=True):
            raise ValueError(f"path segment {index} collides with inflated occupancy")

    required_padding = max(initial_margin_m, minimum_overlap_m * 0.5)
    subdivision_count = 0
    expansion_attempts = 0
    expansion_accepts = 0

    def check_budget() -> None:
        if time.monotonic() - started > wall_clock_deadline_s:
            raise TimeoutError("SFC_TIMEOUT")

    def padded_bounds(a: np.ndarray, b: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
        return np.minimum(a, b) - padding, np.maximum(a, b) + padding

    def subdivide(a: np.ndarray, b: np.ndarray, depth: int) -> list[np.ndarray]:
        nonlocal subdivision_count
        check_budget()
        minimum, maximum = padded_bounds(a, b, required_padding)
        length = float(np.linalg.norm(b - a))
        if length <= maximum_segment_length_m and occupancy.box_is_free(minimum, maximum, inflated=True):
            return [a, b]
        if depth >= maximum_subdivision_depth:
            raise RuntimeError("SFC_INFEASIBLE:no corridor box fits required overlap")
        subdivision_count += 1
        midpoint = (a + b) * 0.5
        left = subdivide(a, midpoint, depth + 1)
        right = subdivide(midpoint, b, depth + 1)
        return left[:-1] + right

    centerline: list[np.ndarray] = [points[0]]
    for start, end in zip(points, points[1:]):
        subdivided = subdivide(start, end, 0)
        centerline.extend(subdivided[1:])
    dense = np.asarray(centerline)

    boxes: list[CorridorBox3D] = []
    for start, end in zip(dense, dense[1:]):
        check_budget()
        minimum, maximum = padded_bounds(start, end, required_padding)
        # Grow each face independently.  Every accepted enlargement is checked
        # against the exact inflated prism map.
        face_growth = np.full(6, required_padding, dtype=float)
        improved = True
        while improved:
            check_budget()
            improved = False
            for face in range(6):
                if face_growth[face] + expansion_step_m > maximum_margin_m + _EPS:
                    continue
                expansion_attempts += 1
                if expansion_attempts > maximum_expansion_iterations:
                    raise RuntimeError("SFC_MAX_ITERATIONS")
                trial_minimum = minimum.copy()
                trial_maximum = maximum.copy()
                axis = face // 2
                if face % 2 == 0:
                    trial_minimum[axis] -= expansion_step_m
                else:
                    trial_maximum[axis] += expansion_step_m
                if occupancy.box_is_free(trial_minimum, trial_maximum, inflated=True):
                    minimum, maximum = trial_minimum, trial_maximum
                    face_growth[face] += expansion_step_m
                    expansion_accepts += 1
                    improved = True
        boxes.append(CorridorBox3D(tuple(minimum), tuple(maximum)))

    if len(boxes) > 1:
        actual_overlap = min(
            float(np.min(left.overlap_extent(right)))
            for left, right in zip(boxes, boxes[1:])
        )
    else:
        actual_overlap = math.inf
    metrics = SFCMetrics3D(
        reason="CONVERGED",
        elapsed_s=time.monotonic() - started,
        box_count=len(boxes),
        subdivision_count=subdivision_count,
        expansion_attempts=expansion_attempts,
        expansion_accepts=expansion_accepts,
        minimum_actual_overlap_m=actual_overlap,
    )
    corridor = SafeFlightCorridor3D(
        tuple(boxes),
        dense,
        float(minimum_overlap_m),
        occupancy.horizontal_inflation_m,
        occupancy.vertical_inflation_up_m,
        occupancy.vertical_inflation_down_m,
        metrics,
    )
    valid, reason = corridor.validate(occupancy)
    if not valid:
        raise RuntimeError(f"generated 3-D corridor failed validation: {reason}")
    return corridor


def simplify_waypoint_polyline_3d(
    path_enu_m: Sequence[Sequence[float]],
    occupancy: PrismOccupancy3D,
    *,
    minimum_clearance_m: float = 0.0,
    validation_step_m: float = 0.2,
) -> np.ndarray:
    """Greedily remove voxel waypoints while preserving hard swept-volume safety."""

    path = np.asarray(path_enu_m, dtype=float)
    if (
        path.ndim != 2
        or path.shape[1] != 3
        or len(path) < 2
        or not np.all(np.isfinite(path))
    ):
        raise ValueError("waypoint path must have shape (N>=2, 3)")
    if minimum_clearance_m < 0.0 or validation_step_m <= 0.0:
        raise ValueError("waypoint clearance and validation step are invalid")

    def segment_valid(start: np.ndarray, end: np.ndarray) -> bool:
        return bool(
            occupancy.segment_is_free(
                start, end, inflated=True, sample_step_m=validation_step_m
            )
            and (
                minimum_clearance_m <= 0.0
                or occupancy.segment_minimum_clearance(
                    start,
                    end,
                    inflated=True,
                    sample_step_m=validation_step_m,
                )
                > minimum_clearance_m + _EPS
            )
        )

    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not segment_valid(path[anchor], path[candidate]):
            candidate -= 1
        if not segment_valid(path[anchor], path[candidate]):
            raise RuntimeError(f"WAYPOINT_CONSTRAINT_VIOLATION:segment_{anchor}_{candidate}")
        simplified.append(path[candidate])
        anchor = candidate
    return np.asarray(simplified)


@dataclass(frozen=True)
class WaypointPlanMetrics3D:
    success: bool
    reason: str
    elapsed_s: float
    astar_elapsed_s: float
    sfc_elapsed_s: float
    expanded_nodes: int
    raw_waypoint_count: int
    output_waypoint_count: int
    corridor_box_count: int
    path_length_m: float
    minimum_clearance_m: float
    cache_hit: bool
    postprocess_elapsed_s: float = 0.0


@dataclass(frozen=True)
class ValidatedWaypointPlan3D:
    waypoints_enu_m: np.ndarray
    corridor: SafeFlightCorridor3D | None
    astar_result: HierarchicalAStar3DResult
    metrics: WaypointPlanMetrics3D

    @property
    def success(self) -> bool:
        return self.metrics.success


def plan_validated_waypoints_3d(
    planner: HierarchicalAStarPlanner3D,
    start_enu_m: Sequence[float],
    goal_enu_m: Sequence[float],
    *,
    minimum_clearance_m: float | None = None,
    validation_step_m: float = 0.2,
    sfc_minimum_overlap_m: float = 0.5,
    sfc_initial_margin_m: float = 0.5,
    sfc_maximum_margin_m: float = 5.0,
    sfc_expansion_step_m: float = 0.5,
    sfc_maximum_segment_length_m: float = 6.0,
    sfc_maximum_subdivision_depth: int = 14,
    sfc_maximum_expansion_iterations: int = 10_000,
    sfc_deadline_s: float = 5.0,
) -> ValidatedWaypointPlan3D:
    """Return only A*+SFC validated waypoints for PX4 native guidance.

    No external curve generator is invoked here.  Multicopters consume these
    waypoints through PX4 PositionSmoothing/L1-style crossing, while fixed-wing
    aircraft use their separate NPFG path-following stack.
    """

    started = time.monotonic()
    astar = planner.plan(start_enu_m, goal_enu_m)
    astar_finished = time.monotonic()
    if not astar.success:
        metrics = WaypointPlanMetrics3D(
            False,
            astar.metrics.reason,
            time.monotonic() - started,
            astar.metrics.elapsed_s,
            0.0,
            astar.metrics.coarse_expanded_nodes + astar.metrics.fine_expanded_nodes,
            0,
            0,
            0,
            0.0,
            0.0,
            astar.metrics.cache_hit,
            0.0,
        )
        return ValidatedWaypointPlan3D(np.empty((0, 3)), None, astar, metrics)

    configured_clearance = (
        planner.config.fine.minimum_clearance_m
        if minimum_clearance_m is None
        else minimum_clearance_m
    )
    if configured_clearance < 0.0:
        raise ValueError("minimum waypoint clearance cannot be negative")
    # A centerline that merely avoids collision may graze an inflated obstacle
    # too closely for a finite-width SFC box. Preserve enough residual space
    # for the requested initial box padding and overlap during LOS reduction.
    residual_hard_clearance = max(
        configured_clearance,
        sfc_initial_margin_m * math.sqrt(3.0),
        0.5 * sfc_minimum_overlap_m * math.sqrt(3.0),
    )
    try:
        waypoints = simplify_waypoint_polyline_3d(
            astar.path_enu_m,
            planner.occupancy,
            minimum_clearance_m=residual_hard_clearance,
            validation_step_m=validation_step_m,
        )
        sfc_started = time.monotonic()
        corridor = build_safe_flight_corridor_3d(
            waypoints,
            planner.occupancy,
            minimum_overlap_m=sfc_minimum_overlap_m,
            initial_margin_m=sfc_initial_margin_m,
            maximum_margin_m=sfc_maximum_margin_m,
            expansion_step_m=sfc_expansion_step_m,
            maximum_segment_length_m=sfc_maximum_segment_length_m,
            maximum_subdivision_depth=sfc_maximum_subdivision_depth,
            maximum_expansion_iterations=sfc_maximum_expansion_iterations,
            wall_clock_deadline_s=sfc_deadline_s,
        )
        sfc_elapsed = time.monotonic() - sfc_started
    except (RuntimeError, TimeoutError) as error:
        reason = str(error)
        metrics = WaypointPlanMetrics3D(
            False,
            reason,
            time.monotonic() - started,
            astar.metrics.elapsed_s,
            locals().get("sfc_elapsed", 0.0),
            astar.metrics.coarse_expanded_nodes + astar.metrics.fine_expanded_nodes,
            len(astar.path_enu_m),
            0,
            0,
            0.0,
            0.0,
            astar.metrics.cache_hit,
            time.monotonic() - astar_finished,
        )
        return ValidatedWaypointPlan3D(np.empty((0, 3)), None, astar, metrics)

    path_length = float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))
    minimum_clearance = min(
        planner.occupancy.segment_minimum_clearance(
            left, right, inflated=True, sample_step_m=validation_step_m
        )
        for left, right in zip(waypoints, waypoints[1:])
    )
    valid, reason = corridor.validate(planner.occupancy)
    success = valid and corridor.contains_path(waypoints)
    metrics = WaypointPlanMetrics3D(
        success,
        "GOAL_REACHED" if success else reason,
        time.monotonic() - started,
        astar.metrics.elapsed_s,
        sfc_elapsed,
        astar.metrics.coarse_expanded_nodes + astar.metrics.fine_expanded_nodes,
        len(astar.path_enu_m),
        len(waypoints),
        len(corridor.boxes),
        path_length,
        minimum_clearance,
        astar.metrics.cache_hit,
        time.monotonic() - astar_finished,
    )
    return ValidatedWaypointPlan3D(
        waypoints if success else np.empty((0, 3)),
        corridor if success else None,
        astar,
        metrics,
    )


@dataclass(frozen=True)
class TrajectoryLimits3D:
    maximum_velocity_m_s: float = 4.0
    maximum_acceleration_m_s2: float = 2.0
    maximum_jerk_m_s3: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.maximum_velocity_m_s,
            self.maximum_acceleration_m_s2,
            self.maximum_jerk_m_s3,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("trajectory limits must be finite and positive")


@dataclass(frozen=True)
class TrajectorySamples3D:
    time_s: np.ndarray
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    acceleration_enu_m_s2: np.ndarray
    jerk_enu_m_s3: np.ndarray
    yaw_enu_rad: np.ndarray
    yaw_rate_rad_s: np.ndarray


@dataclass(frozen=True)
class BSplineTrajectory3D:
    knots_s: np.ndarray
    control_points_enu_m: np.ndarray
    degree: int
    assigned_corridor_boxes: tuple[int, ...]

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots_s, dtype=float)
        controls = np.asarray(self.control_points_enu_m, dtype=float)
        if self.degree != 3:
            raise ValueError("this planner requires a cubic B-spline")
        if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 4:
            raise ValueError("control points must have shape (N>=4, 3)")
        if len(knots) != len(controls) + self.degree + 1 or np.any(np.diff(knots) < 0.0):
            raise ValueError("invalid B-spline knot vector")
        if len(self.assigned_corridor_boxes) != len(controls):
            raise ValueError("every control point needs a corridor assignment")

    @property
    def start_time_s(self) -> float:
        return float(self.knots_s[self.degree])

    @property
    def end_time_s(self) -> float:
        return float(self.knots_s[-self.degree - 1])

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def _spline(self, derivative: int = 0) -> interpolate.BSpline:
        spline = interpolate.BSpline(self.knots_s, self.control_points_enu_m, self.degree, extrapolate=False)
        return spline.derivative(derivative) if derivative else spline

    def derivative_control_points(self, derivative: int) -> np.ndarray:
        if derivative < 0 or derivative > self.degree:
            raise ValueError("derivative order must be between zero and degree")
        coefficients = self.control_points_enu_m.copy()
        knots = self.knots_s.copy()
        degree = self.degree
        for _ in range(derivative):
            denominator = knots[degree + 1 : degree + len(coefficients)] - knots[1 : len(coefficients)]
            scale = np.divide(
                degree,
                denominator,
                out=np.zeros_like(denominator),
                where=np.abs(denominator) > _EPS,
            )
            coefficients = np.diff(coefficients, axis=0) * scale[:, None]
            knots = knots[1:-1]
            degree -= 1
        return coefficients

    def sample(self, sample_period_s: float = 0.05, minimum_samples: int = 2) -> TrajectorySamples3D:
        if sample_period_s <= 0.0 or minimum_samples < 2:
            raise ValueError("invalid trajectory sampling parameters")
        count = max(minimum_samples, int(math.ceil(self.duration_s / sample_period_s)) + 1)
        times = np.linspace(self.start_time_s, self.end_time_s, count)
        position = np.asarray(self._spline(0)(times))
        velocity = np.asarray(self._spline(1)(times))
        acceleration = np.asarray(self._spline(2)(times))
        jerk = np.asarray(self._spline(3)(times))
        horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
        yaw = np.zeros(len(times), dtype=float)
        last_yaw = 0.0
        for index in range(len(times)):
            if horizontal_speed[index] > 1.0e-5:
                last_yaw = math.atan2(velocity[index, 1], velocity[index, 0])
            yaw[index] = last_yaw
        yaw = np.unwrap(yaw)
        yaw_rate = np.gradient(yaw, times) if len(times) > 2 else np.zeros_like(yaw)
        return TrajectorySamples3D(times, position, velocity, acceleration, jerk, yaw, yaw_rate)


@dataclass(frozen=True)
class TrajectoryValidation3D:
    valid: bool
    reason: str
    collision_free: bool
    corridor_contained: bool
    control_points_contained: bool
    endpoint_constraints_satisfied: bool
    dynamic_limits_satisfied: bool
    maximum_velocity_m_s: float
    maximum_acceleration_m_s2: float
    maximum_jerk_m_s3: float


@dataclass(frozen=True)
class BSplinePlanResult3D:
    success: bool
    reason: str
    trajectory: BSplineTrajectory3D | None
    validation: TrajectoryValidation3D | None
    optimization_iterations: int
    time_scaling_iterations: int
    elapsed_s: float = 0.0
    final_duration_s: float = 0.0
    termination_reason: str = "UNKNOWN"
    maximum_constraint_violation: float = math.inf


def _open_uniform_knots(control_count: int, degree: int, duration_s: float) -> np.ndarray:
    spans = control_count - degree
    interior = np.linspace(0.0, duration_s, spans + 1)[1:-1]
    return np.concatenate(
        (np.zeros(degree + 1), interior, np.full(degree + 1, duration_s))
    )


def _resample_polyline(path: np.ndarray, count: int) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > _EPS))
    unique = path[keep]
    if len(unique) < 2:
        raise ValueError("trajectory path must contain two distinct points")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(unique, axis=0), axis=1))))
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, unique[:, axis]) for axis in range(3)]
    )


def _endpoint_conditioned_control_reference(
    path: np.ndarray,
    count: int,
    *,
    degree: int = 3,
    start_derivative_order: int = 2,
    terminal_derivative_order: int = 1,
) -> np.ndarray:
    """Build a polyline reference consistent with clamped endpoint states.

    For a clamped cubic spline, zero initial velocity and acceleration make
    the first three control points coincide.  Zero terminal velocity likewise
    makes the final two coincide.  A plain uniform control reference still
    pulls the first *free* point several nominal intervals away from the
    start.  That creates an artificial acceleration/jerk spike and previously
    stretched the 405 m city trajectory from about 135 s to over 200 s.

    Two additional endpoint guard references let the least-squares smoothness
    objective form the acceleration and braking ramps locally.  They do not
    constrain the result: all endpoint equalities, per-control SFC bounds,
    dense collision checks, and derivative-control-polygon limits remain the
    authoritative acceptance contract.
    """

    if degree < 1 or not 0 <= start_derivative_order <= degree:
        raise ValueError("invalid B-spline start derivative order")
    if not 0 <= terminal_derivative_order <= degree:
        raise ValueError("invalid B-spline terminal derivative order")
    start_guard_count = start_derivative_order + 1 + 2
    terminal_guard_count = terminal_derivative_order + 1 + 2
    minimum_count = start_guard_count + terminal_guard_count
    if count < minimum_count:
        raise ValueError(
            f"endpoint-conditioned reference needs at least {minimum_count} controls"
        )
    core_count = count - start_guard_count - terminal_guard_count + 2
    core = _resample_polyline(path, core_count)
    return np.vstack(
        (
            np.repeat(core[:1], start_guard_count - 1, axis=0),
            core,
            np.repeat(core[-1:], terminal_guard_count - 1, axis=0),
        )
    )


def _basis_row(knots: np.ndarray, control_count: int, degree: int, time_s: float, derivative: int) -> np.ndarray:
    basis = np.eye(control_count)
    spline = interpolate.BSpline(knots, basis, degree, extrapolate=False)
    if derivative:
        spline = spline.derivative(derivative)
    return np.asarray(spline(time_s), dtype=float)


def _derivative_operator(
    knots: np.ndarray, control_count: int, degree: int, derivative: int
) -> np.ndarray:
    """Linear map from position controls to derivative control points."""

    if derivative < 0 or derivative > degree:
        raise ValueError("invalid B-spline derivative order")
    operator = np.eye(control_count)
    active_knots = np.asarray(knots, dtype=float)
    active_degree = degree
    active_count = control_count
    for _ in range(derivative):
        denominator = (
            active_knots[active_degree + 1 : active_degree + active_count]
            - active_knots[1:active_count]
        )
        scale = np.divide(
            active_degree,
            denominator,
            out=np.zeros_like(denominator),
            where=np.abs(denominator) > _EPS,
        )
        difference = np.zeros((active_count - 1, active_count), dtype=float)
        indices = np.arange(active_count - 1)
        difference[indices, indices] = -scale
        difference[indices, indices + 1] = scale
        operator = difference @ operator
        active_knots = active_knots[1:-1]
        active_degree -= 1
        active_count -= 1
    return operator


def _assign_control_boxes(reference: np.ndarray, corridor: SafeFlightCorridor3D) -> tuple[int, ...]:
    count = len(reference)
    assignments: list[int] = []
    previous = 0
    for index, point in enumerate(reference):
        candidates = [candidate for candidate in corridor.box_indices_containing(point) if candidate >= previous]
        expected = int(round(index * (len(corridor.boxes) - 1) / max(1, count - 1)))
        if candidates:
            selected = min(candidates, key=lambda candidate: (abs(candidate - expected), candidate))
        else:
            # Reference points are generated from the SFC centerline, so this
            # branch normally only handles floating-point boundary noise.
            selected = min(
                range(previous, len(corridor.boxes)),
                key=lambda candidate: float(
                    np.linalg.norm(
                        point
                        - np.clip(
                            point,
                            corridor.boxes[candidate].minimum,
                            corridor.boxes[candidate].maximum,
                        )
                    )
                ),
            )
        assignments.append(selected)
        previous = selected
    # Endpoint p/v/a constraints need the first three controls in the first
    # convex cell.  Terminal p/v constraints need the final two in the last.
    assignments[:3] = [0] * min(3, count)
    assignments[-2:] = [len(corridor.boxes) - 1] * min(2, count)
    return tuple(assignments)


def validate_bspline_trajectory_3d(
    trajectory: BSplineTrajectory3D,
    occupancy: PrismOccupancy3D,
    corridor: SafeFlightCorridor3D,
    limits: TrajectoryLimits3D,
    *,
    expected_start_position_enu_m: Sequence[float],
    expected_start_velocity_enu_m_s: Sequence[float],
    expected_start_acceleration_enu_m_s2: Sequence[float],
    expected_goal_position_enu_m: Sequence[float],
    expected_goal_velocity_enu_m_s: Sequence[float] | None = None,
    sample_period_s: float = 0.03,
    spatial_sample_step_m: float = 0.15,
    tolerance: float = 2.0e-4,
) -> TrajectoryValidation3D:
    path_length = float(
        np.sum(np.linalg.norm(np.diff(trajectory.control_points_enu_m, axis=0), axis=1))
    )
    minimum_samples = max(500, int(math.ceil(path_length / spatial_sample_step_m)) + 1)
    samples = trajectory.sample(sample_period_s, minimum_samples)
    collision_free = all(not occupancy.is_inflated_occupied(point) for point in samples.position_enu_m)
    if collision_free:
        collision_free = all(
            occupancy.segment_is_free(
                samples.position_enu_m[index],
                samples.position_enu_m[index + 1],
                inflated=True,
                sample_step_m=spatial_sample_step_m,
            )
            for index in range(len(samples.position_enu_m) - 1)
        )
    corridor_contained = corridor.contains_path(samples.position_enu_m)
    control_points_contained = all(
        corridor.boxes[box_index].contains(point)
        for point, box_index in zip(
            trajectory.control_points_enu_m, trajectory.assigned_corridor_boxes
        )
    )
    start_position = samples.position_enu_m[0]
    start_velocity = samples.velocity_enu_m_s[0]
    start_acceleration = samples.acceleration_enu_m_s2[0]
    end_position = samples.position_enu_m[-1]
    endpoint_ok = (
        np.linalg.norm(start_position - _point3(expected_start_position_enu_m)) <= tolerance
        and np.linalg.norm(start_velocity - _point3(expected_start_velocity_enu_m_s)) <= tolerance
        and np.linalg.norm(start_acceleration - _point3(expected_start_acceleration_enu_m_s2)) <= tolerance
        and np.linalg.norm(end_position - _point3(expected_goal_position_enu_m)) <= tolerance
    )
    if expected_goal_velocity_enu_m_s is not None:
        endpoint_ok = endpoint_ok and bool(
            np.linalg.norm(samples.velocity_enu_m_s[-1] - _point3(expected_goal_velocity_enu_m_s))
            <= tolerance
        )
    velocity_control = trajectory.derivative_control_points(1)
    acceleration_control = trajectory.derivative_control_points(2)
    jerk_control = trajectory.derivative_control_points(3)
    max_velocity = float(max(np.max(np.linalg.norm(velocity_control, axis=1)), np.max(np.linalg.norm(samples.velocity_enu_m_s, axis=1))))
    max_acceleration = float(max(np.max(np.linalg.norm(acceleration_control, axis=1)), np.max(np.linalg.norm(samples.acceleration_enu_m_s2, axis=1))))
    max_jerk = float(max(np.max(np.linalg.norm(jerk_control, axis=1)), np.max(np.linalg.norm(samples.jerk_enu_m_s3, axis=1))))
    dynamic_ok = bool(
        max_velocity <= limits.maximum_velocity_m_s + tolerance
        and max_acceleration <= limits.maximum_acceleration_m_s2 + tolerance
        and max_jerk <= limits.maximum_jerk_m_s3 + tolerance
    )
    valid = collision_free and corridor_contained and control_points_contained and endpoint_ok and dynamic_ok
    failed = []
    if not collision_free:
        failed.append("collision")
    if not corridor_contained:
        failed.append("outside_corridor")
    if not control_points_contained:
        failed.append("control_point_constraint")
    if not endpoint_ok:
        failed.append("endpoint_constraint")
    if not dynamic_ok:
        failed.append("dynamic_limit")
    return TrajectoryValidation3D(
        valid=valid,
        reason="success" if valid else ",".join(failed),
        collision_free=collision_free,
        corridor_contained=corridor_contained,
        control_points_contained=control_points_contained,
        endpoint_constraints_satisfied=endpoint_ok,
        dynamic_limits_satisfied=dynamic_ok,
        maximum_velocity_m_s=max_velocity,
        maximum_acceleration_m_s2=max_acceleration,
        maximum_jerk_m_s3=max_jerk,
    )


def optimize_bspline_trajectory_3d(
    path_enu_m: Sequence[Sequence[float]],
    occupancy: PrismOccupancy3D,
    corridor: SafeFlightCorridor3D,
    limits: TrajectoryLimits3D,
    *,
    initial_velocity_enu_m_s: Sequence[float] = (0.0, 0.0, 0.0),
    initial_acceleration_enu_m_s2: Sequence[float] = (0.0, 0.0, 0.0),
    terminal_velocity_enu_m_s: Sequence[float] | None = (0.0, 0.0, 0.0),
    nominal_speed_m_s: float = 3.0,
    control_point_spacing_m: float = 3.0,
    minimum_duration_s: float = 2.0,
    maximum_duration_s: float = 600.0,
    maximum_time_scaling_iterations: int = 8,
    maximum_optimization_iterations: int = 400,
    optimization_tolerance: float = 1.0e-8,
    wall_clock_deadline_s: float = 8.0,
    reference_weight: float = 100.0,
    acceleration_weight: float = 1.0,
    jerk_weight: float = 1.0,
) -> BSplinePlanResult3D:
    """Optimize spatial controls and time for the selectable B-spline mode.

    Each control point has hard axis-aligned bounds from an assigned convex
    SFC box. Endpoint p/v/a conditions are linear equalities. Velocity,
    acceleration, and jerk are bounded using derivative control polygons and
    dense continuous validation. PX4-native waypoint mode bypasses this
    function and consumes :func:`plan_validated_waypoints_3d` instead.
    """

    started = time.monotonic()
    path = np.asarray(path_enu_m, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("B-spline path must have shape (N>=2, 3)")
    if not corridor.contains_path(path):
        raise ValueError("B-spline reference path lies outside its SFC")
    if any(not occupancy.segment_is_free(a, b, inflated=True) for a, b in zip(path, path[1:])):
        raise ValueError("B-spline reference path collides with inflated occupancy")
    if min(nominal_speed_m_s, control_point_spacing_m, minimum_duration_s) <= 0.0:
        raise ValueError("trajectory timing and spacing parameters must be positive")
    if maximum_duration_s < minimum_duration_s or maximum_time_scaling_iterations < 0:
        raise ValueError("invalid trajectory time-scaling limits")
    if (
        maximum_optimization_iterations <= 0
        or optimization_tolerance <= 0.0
        or wall_clock_deadline_s <= 0.0
    ):
        raise ValueError("B-spline iteration, tolerance, and time limits must be positive")
    if min(reference_weight, acceleration_weight, jerk_weight) < 0.0:
        raise ValueError("trajectory objective weights cannot be negative")
    if reference_weight + acceleration_weight + jerk_weight <= 0.0:
        raise ValueError("at least one trajectory objective weight must be positive")

    initial_velocity = _point3(initial_velocity_enu_m_s, "initial velocity")
    initial_acceleration = _point3(initial_acceleration_enu_m_s2, "initial acceleration")
    terminal_velocity = None if terminal_velocity_enu_m_s is None else _point3(
        terminal_velocity_enu_m_s, "terminal velocity"
    )
    if np.linalg.norm(initial_velocity) > limits.maximum_velocity_m_s + _EPS:
        return BSplinePlanResult3D(False, "initial_velocity_exceeds_limit", None, None, 0, 0, time.monotonic() - started, 0.0, "CONSTRAINT_VIOLATION")
    if np.linalg.norm(initial_acceleration) > limits.maximum_acceleration_m_s2 + _EPS:
        return BSplinePlanResult3D(False, "initial_acceleration_exceeds_limit", None, None, 0, 0, time.monotonic() - started, 0.0, "CONSTRAINT_VIOLATION")
    if terminal_velocity is not None and np.linalg.norm(terminal_velocity) > limits.maximum_velocity_m_s + _EPS:
        return BSplinePlanResult3D(False, "terminal_velocity_exceeds_limit", None, None, 0, 0, time.monotonic() - started, 0.0, "CONSTRAINT_VIOLATION")

    path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    # Ten controls leave room for the clamped p/v/a and p/v endpoint states,
    # two shaping guards at each end, and at least one interior reference.
    control_count = max(10, int(math.ceil(path_length / control_point_spacing_m)) + 3)
    # Avoid needlessly large nonlinear programs on long city routes while
    # preserving enough local support to follow every corridor transition.
    control_count = min(max(control_count, len(corridor.boxes) + 3), 160)
    stationary_endpoints = bool(
        np.linalg.norm(initial_velocity) <= _EPS
        and np.linalg.norm(initial_acceleration) <= _EPS
        and terminal_velocity is not None
        and np.linalg.norm(terminal_velocity) <= _EPS
    )
    # The guard construction models a hover-to-hover route.  Retain the
    # previous uniform reference for non-zero derivative boundary conditions;
    # their exact fixed controls depend on duration and must not be biased
    # toward coincident endpoint anchors.
    reference = (
        _endpoint_conditioned_control_reference(path, control_count)
        if stationary_endpoints
        else _resample_polyline(path, control_count)
    )
    assignments = _assign_control_boxes(reference, corridor)
    lower = np.asarray([corridor.boxes[index].minimum for index in assignments])
    upper = np.asarray([corridor.boxes[index].maximum for index in assignments])
    reference = np.clip(reference, lower, upper)
    duration = max(minimum_duration_s, path_length / nominal_speed_m_s)
    duration = min(duration, maximum_duration_s)
    optimization_iterations = 0

    class _OptimizationDeadline(RuntimeError):
        pass

    def enforce_deadline(_: np.ndarray | None = None) -> None:
        if time.monotonic() - started > wall_clock_deadline_s:
            raise _OptimizationDeadline("B_SPLINE_TIMEOUT")

    for scaling_iteration in range(maximum_time_scaling_iterations + 1):
        try:
            enforce_deadline()
        except _OptimizationDeadline:
            return BSplinePlanResult3D(
                False,
                "optimization_timeout",
                None,
                None,
                optimization_iterations,
                scaling_iteration,
                time.monotonic() - started,
                duration,
                "TIMEOUT",
            )
        knots = _open_uniform_knots(control_count, 3, duration)
        rows = [
            _basis_row(knots, control_count, 3, 0.0, 0),
            _basis_row(knots, control_count, 3, 0.0, 1),
            _basis_row(knots, control_count, 3, 0.0, 2),
            _basis_row(knots, control_count, 3, duration, 0),
        ]
        targets = [path[0], initial_velocity, initial_acceleration, path[-1]]
        if terminal_velocity is not None:
            rows.append(_basis_row(knots, control_count, 3, duration, 1))
            targets.append(terminal_velocity)
        equality_matrix = np.asarray(rows)
        target_matrix = np.asarray(targets)
        controls = np.empty((control_count, 3), dtype=float)
        optimization_failed = None
        fixed_indices = np.asarray(
            [0, 1, 2, control_count - 2, control_count - 1]
            if terminal_velocity is not None
            else [0, 1, 2, control_count - 1],
            dtype=int,
        )
        free_mask = np.ones(control_count, dtype=bool)
        free_mask[fixed_indices] = False
        free_indices = np.flatnonzero(free_mask)
        identity = np.eye(control_count)
        design_blocks: list[np.ndarray] = []
        target_blocks: list[np.ndarray] = []
        if reference_weight > 0.0:
            design_blocks.append(math.sqrt(reference_weight) * identity)
        if acceleration_weight > 0.0:
            design_blocks.append(
                math.sqrt(acceleration_weight) * np.diff(identity, n=2, axis=0)
            )
        if jerk_weight > 0.0:
            design_blocks.append(
                math.sqrt(jerk_weight) * np.diff(identity, n=3, axis=0)
            )
        design = np.vstack(design_blocks)

        for axis in range(3):
            enforce_deadline()
            axis_reference = reference[:, axis]
            if terminal_velocity is not None:
                fixed_matrix = equality_matrix[:, fixed_indices]
            else:
                fixed_matrix = equality_matrix[:, fixed_indices]
            try:
                fixed_values = np.linalg.solve(
                    fixed_matrix, target_matrix[:, axis]
                )
            except np.linalg.LinAlgError as error:
                optimization_failed = f"endpoint_constraint_singular:{error}"
                break
            if np.any(fixed_values < lower[fixed_indices, axis] - 2.0e-4) or np.any(
                fixed_values > upper[fixed_indices, axis] + 2.0e-4
            ):
                optimization_failed = f"axis_{axis}_endpoint_outside_corridor"
                break
            target_blocks.clear()
            if reference_weight > 0.0:
                target_blocks.append(math.sqrt(reference_weight) * axis_reference)
            if acceleration_weight > 0.0:
                target_blocks.append(np.zeros(control_count - 2))
            if jerk_weight > 0.0:
                target_blocks.append(np.zeros(control_count - 3))
            design_target = np.concatenate(target_blocks)
            reduced_target = design_target - design[:, fixed_indices] @ fixed_values
            try:
                result = optimize.lsq_linear(
                    design[:, free_indices],
                    reduced_target,
                    bounds=(lower[free_indices, axis], upper[free_indices, axis]),
                    tol=optimization_tolerance,
                    lsmr_tol=optimization_tolerance,
                    max_iter=maximum_optimization_iterations,
                    verbose=0,
                )
                enforce_deadline()
            except _OptimizationDeadline:
                return BSplinePlanResult3D(
                    False,
                    "optimization_timeout",
                    None,
                    None,
                    optimization_iterations,
                    scaling_iteration,
                    time.monotonic() - started,
                    duration,
                    "TIMEOUT",
                )
            optimization_iterations += int(getattr(result, "nit", 0) or 0)
            axis_controls = np.empty(control_count, dtype=float)
            axis_controls[fixed_indices] = fixed_values
            axis_controls[free_indices] = result.x
            if not result.success or np.max(
                np.abs(equality_matrix @ axis_controls - target_matrix[:, axis])
            ) > 2.0e-4:
                optimization_failed = f"axis_{axis}_optimization_failed:{result.message}"
                break
            controls[:, axis] = axis_controls
        if optimization_failed is not None:
            if (
                scaling_iteration < maximum_time_scaling_iterations
                and duration * 1.5 <= maximum_duration_s + _EPS
            ):
                duration *= 1.5
                continue
            return BSplinePlanResult3D(
                False,
                optimization_failed,
                None,
                None,
                optimization_iterations,
                scaling_iteration,
                time.monotonic() - started,
                duration,
                "MAX_ITERATIONS"
                if "Iteration limit" in optimization_failed
                else "INFEASIBLE",
            )

        trajectory = BSplineTrajectory3D(knots, controls, 3, assignments)
        validation = validate_bspline_trajectory_3d(
            trajectory,
            occupancy,
            corridor,
            limits,
            expected_start_position_enu_m=path[0],
            expected_start_velocity_enu_m_s=initial_velocity,
            expected_start_acceleration_enu_m_s2=initial_acceleration,
            expected_goal_position_enu_m=path[-1],
            expected_goal_velocity_enu_m_s=terminal_velocity,
        )
        if validation.valid:
            return BSplinePlanResult3D(
                True,
                "success",
                trajectory,
                validation,
                optimization_iterations,
                scaling_iteration,
                time.monotonic() - started,
                trajectory.duration_s,
                "CONVERGED",
                0.0,
            )
        spatial_failure = not (
            validation.collision_free
            and validation.corridor_contained
            and validation.control_points_contained
            and validation.endpoint_constraints_satisfied
        )
        if spatial_failure:
            return BSplinePlanResult3D(
                False,
                f"spatial_validation_failed:{validation.reason}",
                trajectory,
                validation,
                optimization_iterations,
                scaling_iteration,
                time.monotonic() - started,
                trajectory.duration_s,
                "CONSTRAINT_VIOLATION",
            )
        scale = max(
            validation.maximum_velocity_m_s / limits.maximum_velocity_m_s,
            math.sqrt(validation.maximum_acceleration_m_s2 / limits.maximum_acceleration_m_s2),
            np.cbrt(validation.maximum_jerk_m_s3 / limits.maximum_jerk_m_s3),
        )
        if not math.isfinite(scale) or scale <= 1.0 + 1.0e-6:
            return BSplinePlanResult3D(False, "dynamic_validation_failed", trajectory, validation, optimization_iterations, scaling_iteration, time.monotonic() - started, trajectory.duration_s, "CONSTRAINT_VIOLATION")
        duration *= float(scale) * 1.02
        if duration > maximum_duration_s + _EPS:
            violation = max(
                0.0,
                validation.maximum_velocity_m_s - limits.maximum_velocity_m_s,
                validation.maximum_acceleration_m_s2 - limits.maximum_acceleration_m_s2,
                validation.maximum_jerk_m_s3 - limits.maximum_jerk_m_s3,
            )
            return BSplinePlanResult3D(False, "maximum_duration_exceeded", trajectory, validation, optimization_iterations, scaling_iteration, time.monotonic() - started, trajectory.duration_s, "INFEASIBLE", violation)

    return BSplinePlanResult3D(
        False,
        "time_scaling_iteration_limit",
        trajectory,
        validation,
        optimization_iterations,
        maximum_time_scaling_iterations,
        time.monotonic() - started,
        trajectory.duration_s,
        "MAX_ITERATIONS",
    )


@dataclass(frozen=True)
class PlanningCandidateMetrics3D:
    mode: PlanningOutputMode
    success: bool
    reason: str
    collision_free: bool
    hard_clearance_violation_count: int
    minimum_clearance_m: float
    path_length_m: float
    estimated_flight_time_s: float
    maximum_velocity_m_s: float
    maximum_acceleration_m_s2: float
    maximum_jerk_m_s3: float
    astar_elapsed_s: float
    sfc_elapsed_s: float
    postprocess_elapsed_s: float
    astar_expanded_nodes: int
    postprocess_iterations: int


@dataclass(frozen=True)
class PlanningModeComparison3D:
    """Common-input offline comparison; PX4 dynamics are filled at runtime."""

    waypoint_plan: ValidatedWaypointPlan3D
    waypoint_metrics: PlanningCandidateMetrics3D
    bspline_result: BSplinePlanResult3D | None
    bspline_metrics: PlanningCandidateMetrics3D


def compare_planning_output_modes_3d(
    waypoint_plan: ValidatedWaypointPlan3D,
    occupancy: PrismOccupancy3D,
    limits: TrajectoryLimits3D,
    *,
    nominal_speed_m_s: float = 3.0,
    bspline_options: dict[str, object] | None = None,
) -> PlanningModeComparison3D:
    """Cross-check both post-A* modes against one accepted map/SFC result.

    Native-waypoint acceleration and jerk depend on PX4 PositionSmoothing (or
    fixed-wing NPFG) and are intentionally reported as NaN offline. Runtime
    telemetry must replace them rather than treating a guessed curve as PX4's
    actual response.
    """

    if nominal_speed_m_s <= 0.0:
        raise ValueError("nominal speed must be positive")
    base = waypoint_plan.metrics
    waypoint_metrics = PlanningCandidateMetrics3D(
        PlanningOutputMode.PX4_NATIVE_WAYPOINTS,
        waypoint_plan.success,
        base.reason,
        waypoint_plan.success,
        0 if waypoint_plan.success else 1,
        base.minimum_clearance_m,
        base.path_length_m,
        base.path_length_m / nominal_speed_m_s if waypoint_plan.success else math.inf,
        math.nan,
        math.nan,
        math.nan,
        base.astar_elapsed_s,
        base.sfc_elapsed_s,
        base.postprocess_elapsed_s,
        base.expanded_nodes,
        0,
    )
    if not waypoint_plan.success or waypoint_plan.corridor is None:
        failed = PlanningCandidateMetrics3D(
            PlanningOutputMode.SFC_BSPLINE,
            False,
            "common_waypoint_plan_failed",
            False,
            1,
            0.0,
            0.0,
            math.inf,
            math.nan,
            math.nan,
            math.nan,
            base.astar_elapsed_s,
            base.sfc_elapsed_s,
            0.0,
            base.expanded_nodes,
            0,
        )
        return PlanningModeComparison3D(waypoint_plan, waypoint_metrics, None, failed)

    options = dict(bspline_options or {})
    options.setdefault("nominal_speed_m_s", nominal_speed_m_s)
    result = optimize_bspline_trajectory_3d(
        waypoint_plan.waypoints_enu_m,
        occupancy,
        waypoint_plan.corridor,
        limits,
        **options,
    )
    if result.success and result.trajectory is not None and result.validation is not None:
        samples = result.trajectory.sample(0.05, 500)
        path_length = float(
            np.sum(np.linalg.norm(np.diff(samples.position_enu_m, axis=0), axis=1))
        )
        minimum_clearance = min(
            occupancy.segment_minimum_clearance(left, right, sample_step_m=0.15)
            for left, right in zip(samples.position_enu_m, samples.position_enu_m[1:])
        )
        validation = result.validation
        bspline_metrics = PlanningCandidateMetrics3D(
            PlanningOutputMode.SFC_BSPLINE,
            True,
            result.termination_reason,
            validation.collision_free,
            0,
            minimum_clearance,
            path_length,
            result.trajectory.duration_s,
            validation.maximum_velocity_m_s,
            validation.maximum_acceleration_m_s2,
            validation.maximum_jerk_m_s3,
            base.astar_elapsed_s,
            base.sfc_elapsed_s,
            result.elapsed_s,
            base.expanded_nodes,
            result.optimization_iterations + result.time_scaling_iterations,
        )
    else:
        bspline_metrics = PlanningCandidateMetrics3D(
            PlanningOutputMode.SFC_BSPLINE,
            False,
            result.reason,
            bool(result.validation and result.validation.collision_free),
            1,
            0.0,
            0.0,
            math.inf,
            math.nan,
            math.nan,
            math.nan,
            base.astar_elapsed_s,
            base.sfc_elapsed_s,
            result.elapsed_s,
            base.expanded_nodes,
            result.optimization_iterations + result.time_scaling_iterations,
        )
    return PlanningModeComparison3D(
        waypoint_plan, waypoint_metrics, result, bspline_metrics
    )


__all__ = [
    "AStar3DConfig",
    "AStar3DMetrics",
    "AStar3DResult",
    "AStarPlanner3D",
    "BSplinePlanResult3D",
    "BSplineTrajectory3D",
    "CorridorBox3D",
    "HierarchicalAStar3DConfig",
    "HierarchicalAStar3DMetrics",
    "HierarchicalAStar3DResult",
    "HierarchicalAStarPlanner3D",
    "PlanningCandidateMetrics3D",
    "PlanningModeComparison3D",
    "PlanningOutputMode",
    "PrismOccupancy3D",
    "SFCMetrics3D",
    "SafeFlightCorridor3D",
    "SearchBand3D",
    "TrajectoryLimits3D",
    "TrajectorySamples3D",
    "TrajectoryValidation3D",
    "ValidatedWaypointPlan3D",
    "VehicleEnvelope3D",
    "WaypointPlanMetrics3D",
    "build_safe_flight_corridor_3d",
    "compare_planning_output_modes_3d",
    "load_preferred_city_map",
    "optimize_bspline_trajectory_3d",
    "resolve_city_map_yaml",
    "plan_validated_waypoints_3d",
    "simplify_waypoint_polyline_3d",
    "validate_bspline_trajectory_3d",
]
