"""Pure planning and control primitives for the PX4 city-map demo.

This module deliberately has no ROS imports.  The ROS node can therefore keep
message conversion and PX4 timing separate from deterministic, unit-testable
planning code.

The global planner operates in Gazebo's local ENU frame.  It rasterizes the
building footprints from ``gazebo/maps/city_coordinates.yaml`` at a fixed
resolution, inflates them, runs 8-connected A*, simplifies the grid path with
line-of-sight tests, constructs a sequence of convex capsule corridors, and
only accepts a cubic B-spline when every sampled point is both collision free
and inside that corridor.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import interpolate, ndimage, optimize
import yaml


_EPS = 1.0e-9


def _as_point2(value: Sequence[float]) -> tuple[float, float]:
    if len(value) < 2:
        raise ValueError("a 2-D point needs at least two components")
    return float(value[0]), float(value[1])


def _cross(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Signed twice-area of triangle ``a, b, c``."""

    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _point_on_segment(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> bool:
    if abs(_cross(start, end, point)) > _EPS:
        return False
    return (
        min(float(start[0]), float(end[0])) - _EPS
        <= float(point[0])
        <= max(float(start[0]), float(end[0])) + _EPS
        and min(float(start[1]), float(end[1])) - _EPS
        <= float(point[1])
        <= max(float(start[1]), float(end[1])) + _EPS
    )


def _segments_intersect(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    if ((ab_c > _EPS and ab_d < -_EPS) or (ab_c < -_EPS and ab_d > _EPS)) and (
        (cd_a > _EPS and cd_b < -_EPS) or (cd_a < -_EPS and cd_b > _EPS)
    ):
        return True
    return (
        (abs(ab_c) <= _EPS and _point_on_segment(c, a, b))
        or (abs(ab_d) <= _EPS and _point_on_segment(d, a, b))
        or (abs(cd_a) <= _EPS and _point_on_segment(a, c, d))
        or (abs(cd_b) <= _EPS and _point_on_segment(b, c, d))
    )


def _point_in_ring(point: Sequence[float], ring: Sequence[Sequence[float]]) -> bool:
    """Return true inside a ring, treating the boundary as inside."""

    if len(ring) < 3:
        return False
    x, y = _as_point2(point)
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        pi = ring[i]
        pj = ring[j]
        if _point_on_segment((x, y), pj, pi):
            return True
        yi = float(pi[1])
        yj = float(pj[1])
        if (yi > y) != (yj > y):
            intersection_x = (float(pj[0]) - float(pi[0])) * (y - yi) / (
                yj - yi
            ) + float(pi[0])
            if x < intersection_x:
                inside = not inside
        j = i
    return inside


def _ring_edges(
    ring: Sequence[Sequence[float]],
) -> Iterable[tuple[Sequence[float], Sequence[float]]]:
    for index, point in enumerate(ring):
        yield point, ring[(index + 1) % len(ring)]


def _point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    p = np.asarray(point[:2], dtype=float)
    a = np.asarray(start[:2], dtype=float)
    b = np.asarray(end[:2], dtype=float)
    delta = b - a
    length_squared = float(np.dot(delta, delta))
    if length_squared <= _EPS:
        return float(np.linalg.norm(p - a))
    ratio = float(np.clip(np.dot(p - a, delta) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ratio * delta)))


@dataclass(frozen=True)
class PolygonPrism:
    """A building footprint and its vertical extent, expressed in ENU."""

    building_id: str
    outer: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...]
    foundation_z_m: float
    roof_z_m: float

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.outer]
        ys = [point[1] for point in self.outer]
        return min(xs), min(ys), max(xs), max(ys)

    def contains_xy(self, point: Sequence[float]) -> bool:
        if not _point_in_ring(point, self.outer):
            return False
        return not any(_point_in_ring(point, hole) for hole in self.holes)


@dataclass(frozen=True)
class CityMap:
    """Relevant contents of ``city_coordinates.yaml``."""

    bounds_x_m: tuple[float, float]
    bounds_y_m: tuple[float, float]
    buildings: tuple[PolygonPrism, ...]
    px4_origin_enu_m: tuple[float, float, float]
    default_inflation_m: float
    source_path: Path
    gazebo_world_name: str = "applepark_city"
    runtime_entity_name: str = "x500_city_rgbd_lidar_0"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CityMap":
        source_path = Path(yaml_path).expanduser().resolve()
        with source_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        try:
            bounds = document["map"]["bounds_enu_m"]
            origin = document["frames"]["px4_local"]["origin_enu_m"]
            building_documents = document["obstacles"]["buildings"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid city coordinate map: missing {exc}") from exc

        bounds_x = tuple(float(value) for value in bounds["x"])
        bounds_y = tuple(float(value) for value in bounds["y"])
        if len(bounds_x) != 2 or len(bounds_y) != 2:
            raise ValueError("map bounds must contain [minimum, maximum]")
        if not bounds_x[0] < bounds_x[1] or not bounds_y[0] < bounds_y[1]:
            raise ValueError("map bounds must be strictly increasing")

        buildings: list[PolygonPrism] = []
        for building in building_documents:
            footprint = building["footprint"]
            outer = tuple(_as_point2(point) for point in footprint["outer"])
            holes = tuple(
                tuple(_as_point2(point) for point in ring)
                for ring in footprint.get("holes", [])
            )
            if len(outer) < 3:
                raise ValueError(f"{building.get('id', '<unknown>')} has no polygon")
            buildings.append(
                PolygonPrism(
                    building_id=str(building["id"]),
                    outer=outer,
                    holes=holes,
                    foundation_z_m=float(building["foundation_z_m"]),
                    roof_z_m=float(building["roof_z_m"]),
                )
            )

        inflation = float(
            document.get("planner_defaults", {}).get("obstacle_inflation_m", 0.0)
        )
        if inflation < 0.0:
            raise ValueError("obstacle inflation cannot be negative")
        if len(origin) != 3:
            raise ValueError("PX4 origin must have three ENU components")
        return cls(
            bounds_x_m=(bounds_x[0], bounds_x[1]),
            bounds_y_m=(bounds_y[0], bounds_y[1]),
            buildings=tuple(buildings),
            px4_origin_enu_m=tuple(float(value) for value in origin),
            default_inflation_m=inflation,
            source_path=source_path,
            gazebo_world_name=str(
                document.get("map", {}).get("gazebo_world_name", "applepark_city")
            ),
            runtime_entity_name=str(
                document.get("px4_vehicle", {}).get(
                    "runtime_entity_name", "x500_city_rgbd_lidar_0"
                )
            ),
        )


def _cell_intersects_building(
    building: PolygonPrism,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> bool:
    corners = (
        (minimum_x, minimum_y),
        (maximum_x, minimum_y),
        (maximum_x, maximum_y),
        (minimum_x, maximum_y),
    )
    center = ((minimum_x + maximum_x) * 0.5, (minimum_y + maximum_y) * 0.5)
    if any(building.contains_xy(point) for point in (*corners, center)):
        return True

    def in_cell(point: Sequence[float]) -> bool:
        return (
            minimum_x - _EPS <= float(point[0]) <= maximum_x + _EPS
            and minimum_y - _EPS <= float(point[1]) <= maximum_y + _EPS
        )

    if any(in_cell(point) and building.contains_xy(point) for point in building.outer):
        return True

    cell_edges = tuple(_ring_edges(corners))
    for start, end in _ring_edges(building.outer):
        if any(_segments_intersect(start, end, a, b) for a, b in cell_edges):
            return True
    # A cell crossed by a hole boundary contains both free and solid area and is
    # conservatively occupied.  Cells completely inside a hole remain free.
    for hole in building.holes:
        for start, end in _ring_edges(hole):
            if any(_segments_intersect(start, end, a, b) for a, b in cell_edges):
                return True
    return False


@dataclass
class OccupancyGrid:
    """A cell-centred, inflated 2-D occupancy raster in ENU."""

    minimum_x_m: float
    minimum_y_m: float
    maximum_x_m: float
    maximum_y_m: float
    resolution_m: float
    base_occupied: np.ndarray
    occupied: np.ndarray
    clearance_m: np.ndarray
    inflation_m: float

    @classmethod
    def from_city_map(
        cls,
        city_map: CityMap,
        resolution_m: float = 2.0,
        inflation_m: float | None = None,
    ) -> "OccupancyGrid":
        if resolution_m <= 0.0:
            raise ValueError("grid resolution must be positive")
        inflation = city_map.default_inflation_m if inflation_m is None else inflation_m
        if inflation < 0.0:
            raise ValueError("obstacle inflation cannot be negative")
        width = city_map.bounds_x_m[1] - city_map.bounds_x_m[0]
        height = city_map.bounds_y_m[1] - city_map.bounds_y_m[0]
        columns = int(math.ceil(width / resolution_m))
        rows = int(math.ceil(height / resolution_m))
        base = np.zeros((rows, columns), dtype=bool)

        for building in city_map.buildings:
            bx0, by0, bx1, by1 = building.bounds_xy
            first_x = max(0, int(math.floor((bx0 - city_map.bounds_x_m[0]) / resolution_m)))
            last_x = min(
                columns - 1,
                int(math.floor((bx1 - city_map.bounds_x_m[0]) / resolution_m)),
            )
            first_y = max(0, int(math.floor((by0 - city_map.bounds_y_m[0]) / resolution_m)))
            last_y = min(
                rows - 1,
                int(math.floor((by1 - city_map.bounds_y_m[0]) / resolution_m)),
            )
            if first_x > last_x or first_y > last_y:
                continue
            for row in range(first_y, last_y + 1):
                cell_y0 = city_map.bounds_y_m[0] + row * resolution_m
                cell_y1 = min(cell_y0 + resolution_m, city_map.bounds_y_m[1])
                for column in range(first_x, last_x + 1):
                    cell_x0 = city_map.bounds_x_m[0] + column * resolution_m
                    cell_x1 = min(cell_x0 + resolution_m, city_map.bounds_x_m[1])
                    if _cell_intersects_building(
                        building, cell_x0, cell_y0, cell_x1, cell_y1
                    ):
                        base[row, column] = True

        if np.any(base):
            distance_from_building = ndimage.distance_transform_edt(~base) * resolution_m
            occupied = base | (distance_from_building <= float(inflation) + _EPS)
        else:
            occupied = base.copy()
        if np.any(occupied):
            obstacle_clearance = ndimage.distance_transform_edt(~occupied) * resolution_m
        else:
            obstacle_clearance = np.full(base.shape, np.inf, dtype=float)

        x_centers = city_map.bounds_x_m[0] + (
            np.arange(columns, dtype=float) + 0.5
        ) * resolution_m
        y_centers = city_map.bounds_y_m[0] + (
            np.arange(rows, dtype=float) + 0.5
        ) * resolution_m
        boundary_x = np.minimum(
            x_centers - city_map.bounds_x_m[0],
            city_map.bounds_x_m[1] - x_centers,
        )
        boundary_y = np.minimum(
            y_centers - city_map.bounds_y_m[0],
            city_map.bounds_y_m[1] - y_centers,
        )
        boundary_clearance = np.minimum(boundary_y[:, None], boundary_x[None, :])
        clearance = np.minimum(obstacle_clearance, boundary_clearance)
        clearance[occupied] = 0.0
        return cls(
            minimum_x_m=city_map.bounds_x_m[0],
            minimum_y_m=city_map.bounds_y_m[0],
            maximum_x_m=city_map.bounds_x_m[1],
            maximum_y_m=city_map.bounds_y_m[1],
            resolution_m=float(resolution_m),
            base_occupied=base,
            occupied=occupied,
            clearance_m=clearance,
            inflation_m=float(inflation),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    def contains_world(self, point: Sequence[float]) -> bool:
        x, y = _as_point2(point)
        return (
            self.minimum_x_m <= x < self.maximum_x_m
            and self.minimum_y_m <= y < self.maximum_y_m
        )

    def world_to_grid(self, point: Sequence[float]) -> tuple[int, int]:
        if not self.contains_world(point):
            raise ValueError(f"point {tuple(point[:2])} lies outside the map")
        x, y = _as_point2(point)
        column = min(
            self.shape[1] - 1,
            int(math.floor((x - self.minimum_x_m) / self.resolution_m)),
        )
        row = min(
            self.shape[0] - 1,
            int(math.floor((y - self.minimum_y_m) / self.resolution_m)),
        )
        return row, column

    def grid_to_world(self, cell: Sequence[int]) -> tuple[float, float]:
        row, column = int(cell[0]), int(cell[1])
        if not (0 <= row < self.shape[0] and 0 <= column < self.shape[1]):
            raise ValueError(f"grid cell {(row, column)} lies outside the map")
        return (
            self.minimum_x_m + (column + 0.5) * self.resolution_m,
            self.minimum_y_m + (row + 0.5) * self.resolution_m,
        )

    def is_free(self, point: Sequence[float]) -> bool:
        if not self.contains_world(point):
            return False
        row, column = self.world_to_grid(point)
        return not bool(self.occupied[row, column])

    def clearance_at(self, point: Sequence[float]) -> float:
        if not self.contains_world(point):
            return 0.0
        row, column = self.world_to_grid(point)
        return float(self.clearance_m[row, column])

    def segment_is_free(
        self,
        start: Sequence[float],
        end: Sequence[float],
        sample_step_m: float | None = None,
    ) -> bool:
        """Test a continuous segment and reject diagonal corner cutting."""

        if not self.is_free(start) or not self.is_free(end):
            return False
        a = np.asarray(start[:2], dtype=float)
        b = np.asarray(end[:2], dtype=float)
        distance = float(np.linalg.norm(b - a))
        step = sample_step_m or self.resolution_m * 0.2
        sample_count = max(1, int(math.ceil(distance / step)))
        previous_cell = self.world_to_grid(a)
        for sample_index in range(1, sample_count + 1):
            ratio = sample_index / sample_count
            point = a + ratio * (b - a)
            if not self.is_free(point):
                return False
            cell = self.world_to_grid(point)
            row_delta = cell[0] - previous_cell[0]
            column_delta = cell[1] - previous_cell[1]
            if row_delta != 0 and column_delta != 0:
                # Dense sampling ensures each delta is at most one cell.  Both
                # side cells must be free before a diagonal crossing is legal.
                side_a = (previous_cell[0] + row_delta, previous_cell[1])
                side_b = (previous_cell[0], previous_cell[1] + column_delta)
                if self.occupied[side_a] or self.occupied[side_b]:
                    return False
            previous_cell = cell
        return True

    def path_is_free(self, path: Sequence[Sequence[float]]) -> bool:
        return len(path) > 0 and all(
            self.segment_is_free(path[index], path[index + 1])
            for index in range(len(path) - 1)
        )


class AStarPlanner:
    """8-connected A* with safe diagonals and a clearance preference."""

    _NEIGHBORS = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    def __init__(self, grid: OccupancyGrid, clearance_weight: float = 1.0):
        if clearance_weight < 0.0:
            raise ValueError("clearance weight cannot be negative")
        self.grid = grid
        self.clearance_weight = float(clearance_weight)

    def plan(
        self, start_enu_m: Sequence[float], goal_enu_m: Sequence[float]
    ) -> np.ndarray:
        start = self.grid.world_to_grid(start_enu_m)
        goal = self.grid.world_to_grid(goal_enu_m)
        if self.grid.occupied[start]:
            raise ValueError("A* start is occupied after inflation")
        if self.grid.occupied[goal]:
            raise ValueError("A* goal is occupied after inflation")

        g_score = np.full(self.grid.shape, np.inf, dtype=float)
        g_score[start] = 0.0
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        queue: list[tuple[float, int, tuple[int, int]]] = []
        sequence = 0

        def heuristic(cell: tuple[int, int]) -> float:
            # Every edge cost is at least its Euclidean length, so this remains
            # admissible even with the non-negative clearance penalty.
            return self.grid.resolution_m * math.hypot(
                goal[0] - cell[0], goal[1] - cell[1]
            )

        heapq.heappush(queue, (heuristic(start), sequence, start))
        closed = np.zeros(self.grid.shape, dtype=bool)
        while queue:
            _, _, current = heapq.heappop(queue)
            if closed[current]:
                continue
            if current == goal:
                break
            closed[current] = True
            for row_delta, column_delta in self._NEIGHBORS:
                neighbor = (current[0] + row_delta, current[1] + column_delta)
                if not (
                    0 <= neighbor[0] < self.grid.shape[0]
                    and 0 <= neighbor[1] < self.grid.shape[1]
                ):
                    continue
                if self.grid.occupied[neighbor] or closed[neighbor]:
                    continue
                if row_delta != 0 and column_delta != 0:
                    if self.grid.occupied[current[0] + row_delta, current[1]] or self.grid.occupied[
                        current[0], current[1] + column_delta
                    ]:
                        continue
                step = self.grid.resolution_m * math.hypot(row_delta, column_delta)
                clearance = max(
                    float(self.grid.clearance_m[neighbor]), self.grid.resolution_m
                )
                tentative = g_score[current] + step * (
                    1.0 + self.clearance_weight / clearance
                )
                if tentative + _EPS < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    sequence += 1
                    heapq.heappush(
                        queue,
                        (tentative + heuristic(neighbor), sequence, neighbor),
                    )
        else:
            raise RuntimeError("A* could not find a path to the requested goal")

        cells = [goal]
        while cells[-1] != start:
            predecessor = came_from.get(cells[-1])
            if predecessor is None:
                raise RuntimeError("A* path reconstruction failed")
            cells.append(predecessor)
        cells.reverse()
        points = np.asarray([self.grid.grid_to_world(cell) for cell in cells])
        points[0] = np.asarray(start_enu_m[:2], dtype=float)
        points[-1] = np.asarray(goal_enu_m[:2], dtype=float)
        if not self.grid.path_is_free(points):
            raise RuntimeError("internal error: A* produced a colliding path")
        return points


def simplify_line_of_sight(
    path: Sequence[Sequence[float]], grid: OccupancyGrid
) -> np.ndarray:
    """Greedily retain the farthest visible waypoint from each anchor."""

    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("path must have shape (N, 2+) ")
    if len(points) <= 2:
        return points.copy()
    simplified = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1 and not grid.segment_is_free(
            points[anchor], points[candidate]
        ):
            candidate -= 1
        simplified.append(points[candidate])
        anchor = candidate
    result = np.asarray(simplified)
    if not grid.path_is_free(result):
        raise RuntimeError("line-of-sight simplification broke path safety")
    return result


@dataclass(frozen=True)
class SafeCorridorSegment:
    """A convex 2-D capsule (line segment Minkowski-summed with a disk)."""

    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    radius_m: float

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("safe-corridor radius must be positive")

    def contains(self, point: Sequence[float], tolerance_m: float = 1.0e-7) -> bool:
        return _point_segment_distance(point, self.start_xy_m, self.end_xy_m) <= (
            self.radius_m + tolerance_m
        )


@dataclass(frozen=True)
class SafeFlightCorridor:
    """Union of overlapping convex capsules along a global path."""

    segments: tuple[SafeCorridorSegment, ...]

    def contains(self, point: Sequence[float], tolerance_m: float = 1.0e-7) -> bool:
        return any(
            segment.contains(point, tolerance_m=tolerance_m)
            for segment in self.segments
        )

    def contains_path(self, path: Sequence[Sequence[float]]) -> bool:
        return all(self.contains(point) for point in path)


def _capsule_is_free(
    grid: OccupancyGrid,
    start: Sequence[float],
    end: Sequence[float],
    radius_m: float,
) -> bool:
    """Conservatively sample a capsule before admitting it as an SFC cell."""

    a = np.asarray(start[:2], dtype=float)
    b = np.asarray(end[:2], dtype=float)
    length = float(np.linalg.norm(b - a))
    along_count = max(1, int(math.ceil(length / (grid.resolution_m * 0.4))))
    angular_count = 32
    angles = np.linspace(0.0, 2.0 * math.pi, angular_count, endpoint=False)
    offsets = np.column_stack((np.cos(angles), np.sin(angles))) * radius_m
    for ratio in np.linspace(0.0, 1.0, along_count + 1):
        center = a + ratio * (b - a)
        if not grid.is_free(center):
            return False
        if any(not grid.is_free(center + offset) for offset in offsets):
            return False
    return True


def build_safe_flight_corridor(
    path: Sequence[Sequence[float]],
    grid: OccupancyGrid,
    maximum_radius_m: float = 8.0,
    minimum_radius_m: float = 0.1,
) -> SafeFlightCorridor:
    points = np.asarray(path, dtype=float)
    if len(points) < 2:
        raise ValueError("at least two path points are required for an SFC")
    if maximum_radius_m <= 0.0 or minimum_radius_m <= 0.0:
        raise ValueError("corridor radii must be positive")
    if minimum_radius_m > maximum_radius_m:
        raise ValueError("minimum corridor radius exceeds maximum")

    segments: list[SafeCorridorSegment] = []
    for index in range(len(points) - 1):
        start = points[index, :2]
        end = points[index + 1, :2]
        if not grid.segment_is_free(start, end):
            raise ValueError("cannot build an SFC around a colliding segment")
        distance = float(np.linalg.norm(end - start))
        sample_count = max(1, int(math.ceil(distance / (grid.resolution_m * 0.5))))
        centerline_clearance = min(
            grid.clearance_at(start + ratio * (end - start))
            for ratio in np.linspace(0.0, 1.0, sample_count + 1)
        )
        # The EDT values are centre-to-centre.  Subtracting a half-cell diagonal
        # turns them into a conservative lower bound to an occupied cell square.
        radius = min(
            maximum_radius_m,
            max(
                minimum_radius_m,
                centerline_clearance - grid.resolution_m / math.sqrt(2.0),
            ),
        )
        while radius > minimum_radius_m and not _capsule_is_free(
            grid, start, end, radius
        ):
            radius = max(minimum_radius_m, radius * 0.7)
        if not _capsule_is_free(grid, start, end, radius):
            raise RuntimeError("no collision-free capsule fits around the path")
        segments.append(
            SafeCorridorSegment(_as_point2(start), _as_point2(end), float(radius))
        )
    return SafeFlightCorridor(tuple(segments))


def arc_length_resample(
    path: Sequence[Sequence[float]], spacing_m: float
) -> np.ndarray:
    """Resample a polyline at approximately uniform arc-length intervals."""

    if spacing_m <= 0.0:
        raise ValueError("arc-length spacing must be positive")
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("path must be a non-empty 2-D array")
    if len(points) == 1:
        return points.copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > _EPS))
    points = points[keep]
    if len(points) == 1:
        return points.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    total = float(cumulative[-1])
    targets = np.arange(0.0, total, spacing_m)
    if len(targets) == 0 or total - targets[-1] > _EPS:
        targets = np.append(targets, total)
    targets[-1] = total
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, dimension]) for dimension in range(points.shape[1])]
    )


@dataclass(frozen=True)
class SmoothPathResult:
    points: np.ndarray
    used_bspline: bool
    message: str


def _samples_are_safe(
    samples: np.ndarray, grid: OccupancyGrid, corridor: SafeFlightCorridor
) -> bool:
    return (
        len(samples) > 0
        and all(grid.is_free(point[:2]) and corridor.contains(point[:2]) for point in samples)
        and grid.path_is_free(samples[:, :2])
    )


def smooth_path_in_corridor(
    path: Sequence[Sequence[float]],
    grid: OccupancyGrid,
    corridor: SafeFlightCorridor,
    output_spacing_m: float = 1.0,
    smoothing: float = 2.0,
) -> SmoothPathResult:
    """Try cubic B-spline smoothing and reject any unsafe candidate.

    The safe fallback is the original SFC centreline, arc-length resampled.  A
    caller therefore always receives a checked path even when SciPy cannot fit
    the spline or the cubic curve overshoots a corner.
    """

    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        raise ValueError("smoothing requires a path with at least two points")
    if not _samples_are_safe(points, grid, corridor):
        raise ValueError("input path is not free and contained by its SFC")

    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > _EPS))
    unique_points = points[keep]
    # Densifying the centreline before fitting keeps a cubic spline close to
    # long line-of-sight segments and makes corner overshoot small enough for a
    # narrow SFC to reject or accept on geometry, rather than control-point
    # spacing.  It does not relax any subsequent safety check.
    spline_spacing = min(1.0, output_spacing_m * 0.5, grid.resolution_m * 0.5)
    spline_points = arc_length_resample(unique_points, spline_spacing)
    errors: list[str] = []
    if len(spline_points) >= 4:
        chord = np.concatenate(
            (
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(spline_points, axis=0), axis=1)),
            )
        )
        chord /= chord[-1]
        trial_smoothing = []
        for value in (max(0.0, smoothing), max(0.0, smoothing) * 0.25, 0.0):
            if value not in trial_smoothing:
                trial_smoothing.append(value)
        for smoothness in trial_smoothing:
            try:
                spline, _ = interpolate.splprep(
                    spline_points.T,
                    u=chord,
                    k=3,
                    s=smoothness,
                )
                dense_count = max(400, len(spline_points) * 4)
                dense = np.column_stack(
                    interpolate.splev(np.linspace(0.0, 1.0, dense_count), spline)
                )
                dense[0] = spline_points[0]
                dense[-1] = spline_points[-1]
                if not _samples_are_safe(dense, grid, corridor):
                    errors.append(f"s={smoothness:g} left free space or SFC")
                    continue
                resampled = arc_length_resample(dense, output_spacing_m)
                if _samples_are_safe(resampled, grid, corridor):
                    return SmoothPathResult(
                        points=resampled,
                        used_bspline=True,
                        message=f"accepted cubic B-spline with s={smoothness:g}",
                    )
                errors.append(f"s={smoothness:g} failed after arc-length resampling")
            except (TypeError, ValueError, RuntimeError) as exc:
                errors.append(f"s={smoothness:g}: {exc}")
    else:
        errors.append("fewer than four unique points; cubic fit not possible")

    fallback = arc_length_resample(unique_points, output_spacing_m)
    if not _samples_are_safe(fallback, grid, corridor):
        raise RuntimeError("even the SFC centreline fallback failed safety validation")
    return SmoothPathResult(
        points=fallback,
        used_bspline=False,
        message="polyline fallback: " + "; ".join(errors),
    )


@dataclass(frozen=True)
class PlannedPath:
    raw_astar: np.ndarray
    simplified: np.ndarray
    corridor: SafeFlightCorridor
    trajectory: np.ndarray
    used_bspline: bool
    smoothing_message: str


def plan_smoothed_path(
    grid: OccupancyGrid,
    start_enu_m: Sequence[float],
    goal_enu_m: Sequence[float],
    clearance_weight: float = 1.0,
    corridor_radius_m: float = 8.0,
    output_spacing_m: float = 1.0,
    smoothing: float = 2.0,
) -> PlannedPath:
    raw = AStarPlanner(grid, clearance_weight=clearance_weight).plan(
        start_enu_m, goal_enu_m
    )
    simplified = simplify_line_of_sight(raw, grid)
    corridor = build_safe_flight_corridor(
        simplified, grid, maximum_radius_m=corridor_radius_m
    )
    smooth = smooth_path_in_corridor(
        simplified,
        grid,
        corridor,
        output_spacing_m=output_spacing_m,
        smoothing=smoothing,
    )
    return PlannedPath(
        raw_astar=raw,
        simplified=simplified,
        corridor=corridor,
        trajectory=smooth.points,
        used_bspline=smooth.used_bspline,
        smoothing_message=smooth.message,
    )


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class FrameTransform:
    """Gazebo ENU to PX4 local NED transform at a fixed local origin."""

    origin_enu_m: tuple[float, float, float]

    @classmethod
    def from_city_map(cls, city_map: CityMap) -> "FrameTransform":
        return cls(city_map.px4_origin_enu_m)

    def enu_to_ned(self, point_enu_m: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_enu_m, dtype=float)
        if point.shape != (3,):
            raise ValueError("ENU point must have shape (3,)")
        origin = np.asarray(self.origin_enu_m, dtype=float)
        delta = point - origin
        return np.asarray((delta[1], delta[0], -delta[2]), dtype=float)

    def ned_to_enu(self, point_ned_m: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_ned_m, dtype=float)
        if point.shape != (3,):
            raise ValueError("NED point must have shape (3,)")
        origin = np.asarray(self.origin_enu_m, dtype=float)
        return origin + np.asarray((point[1], point[0], -point[2]), dtype=float)

    @staticmethod
    def yaw_enu_to_ned(yaw_enu_rad: float) -> float:
        return normalize_angle(math.pi / 2.0 - yaw_enu_rad)

    @staticmethod
    def yaw_ned_to_enu(yaw_ned_rad: float) -> float:
        return normalize_angle(math.pi / 2.0 - yaw_ned_rad)


@dataclass(frozen=True)
class MPCResult:
    acceleration_m_s2: np.ndarray
    predicted_positions_m: np.ndarray
    predicted_velocities_m_s: np.ndarray
    success: bool
    used_fallback: bool
    message: str
    cost: float


class DoubleIntegratorMPC:
    """Finite-horizon 3-D double-integrator MPC using SciPy optimize.

    Acceleration limits are component-wise.  If optimization fails or returns a
    non-finite solution, a bounded position/velocity PD command is returned so
    the flight loop always has a finite local avoidance command.
    """

    def __init__(
        self,
        dt_s: float = 0.2,
        horizon_steps: int = 12,
        acceleration_limit_m_s2: float | Sequence[float] = 3.0,
        position_weight: float = 8.0,
        velocity_weight: float = 2.0,
        acceleration_weight: float = 0.15,
        acceleration_change_weight: float = 0.1,
        fallback_kp: float = 1.5,
        fallback_kd: float = 1.8,
    ):
        if dt_s <= 0.0 or horizon_steps <= 0:
            raise ValueError("MPC dt and horizon must be positive")
        limits = np.asarray(acceleration_limit_m_s2, dtype=float)
        if limits.ndim == 0:
            limits = np.repeat(float(limits), 3)
        if limits.shape != (3,) or np.any(limits <= 0.0):
            raise ValueError("acceleration limits must be a positive scalar or 3-vector")
        weights = (
            position_weight,
            velocity_weight,
            acceleration_weight,
            acceleration_change_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("MPC weights cannot be negative")
        self.dt_s = float(dt_s)
        self.horizon_steps = int(horizon_steps)
        self.acceleration_limit_m_s2 = limits
        self.position_weight = float(position_weight)
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.acceleration_change_weight = float(acceleration_change_weight)
        self.fallback_kp = float(fallback_kp)
        self.fallback_kd = float(fallback_kd)
        self._warm_start = np.zeros((self.horizon_steps, 3), dtype=float)

    def _reference(self, values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape == (3,):
            return np.repeat(array[None, :], self.horizon_steps, axis=0)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
            raise ValueError(f"{name} must have shape (3,) or (N, 3)")
        if len(array) >= self.horizon_steps:
            return array[: self.horizon_steps].copy()
        return np.vstack(
            (array, np.repeat(array[-1][None, :], self.horizon_steps - len(array), axis=0))
        )

    def _rollout(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        accelerations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.empty((self.horizon_steps, 3), dtype=float)
        velocities = np.empty((self.horizon_steps, 3), dtype=float)
        current_position = position.copy()
        current_velocity = velocity.copy()
        for index, acceleration in enumerate(accelerations):
            current_position = (
                current_position
                + self.dt_s * current_velocity
                + 0.5 * self.dt_s * self.dt_s * acceleration
            )
            current_velocity = current_velocity + self.dt_s * acceleration
            positions[index] = current_position
            velocities[index] = current_velocity
        return positions, velocities

    def _bounded_pd(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
    ) -> np.ndarray:
        command = self.fallback_kp * (reference_position - position) + self.fallback_kd * (
            reference_velocity - velocity
        )
        return np.clip(
            command,
            -self.acceleration_limit_m_s2,
            self.acceleration_limit_m_s2,
        )

    def solve(
        self,
        position_m: Sequence[float],
        velocity_m_s: Sequence[float],
        reference_positions_m: Sequence[float] | np.ndarray,
        reference_velocities_m_s: Sequence[float] | np.ndarray | None = None,
    ) -> MPCResult:
        position = np.asarray(position_m, dtype=float)
        velocity = np.asarray(velocity_m_s, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("MPC position and velocity must each have shape (3,)")
        reference_positions = self._reference(reference_positions_m, "position reference")
        if reference_velocities_m_s is None:
            reference_velocities = np.zeros_like(reference_positions)
        else:
            reference_velocities = self._reference(
                reference_velocities_m_s, "velocity reference"
            )

        def objective(flat_accelerations: np.ndarray) -> float:
            accelerations = flat_accelerations.reshape(self.horizon_steps, 3)
            positions, velocities = self._rollout(position, velocity, accelerations)
            position_error = positions - reference_positions
            velocity_error = velocities - reference_velocities
            changes = np.diff(
                np.vstack((self._warm_start[0], accelerations)), axis=0
            )
            return float(
                self.position_weight * np.sum(position_error * position_error)
                + self.velocity_weight * np.sum(velocity_error * velocity_error)
                + self.acceleration_weight * np.sum(accelerations * accelerations)
                + self.acceleration_change_weight * np.sum(changes * changes)
            )

        bounds = [
            (-float(self.acceleration_limit_m_s2[axis]), float(self.acceleration_limit_m_s2[axis]))
            for _ in range(self.horizon_steps)
            for axis in range(3)
        ]
        try:
            result = optimize.minimize(
                objective,
                self._warm_start.reshape(-1),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1.0e-8},
            )
            valid = bool(result.success) and np.all(np.isfinite(result.x))
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            result = None
            valid = False
            failure_message = str(exc)

        if valid and result is not None:
            accelerations = result.x.reshape(self.horizon_steps, 3)
            positions, velocities = self._rollout(position, velocity, accelerations)
            self._warm_start[:-1] = accelerations[1:]
            self._warm_start[-1] = accelerations[-1]
            return MPCResult(
                acceleration_m_s2=accelerations[0].copy(),
                predicted_positions_m=positions,
                predicted_velocities_m_s=velocities,
                success=True,
                used_fallback=False,
                message=str(result.message),
                cost=float(result.fun),
            )

        acceleration = self._bounded_pd(
            position,
            velocity,
            reference_positions[0],
            reference_velocities[0],
        )
        fallback_controls = np.repeat(acceleration[None, :], self.horizon_steps, axis=0)
        positions, velocities = self._rollout(position, velocity, fallback_controls)
        if result is not None:
            failure_message = str(result.message)
        return MPCResult(
            acceleration_m_s2=acceleration,
            predicted_positions_m=positions,
            predicted_velocities_m_s=velocities,
            success=False,
            used_fallback=True,
            message=f"bounded PD fallback: {failure_message}",
            cost=float("nan"),
        )
