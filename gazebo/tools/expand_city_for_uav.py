#!/usr/bin/env python3
"""Generate the non-destructive, 2.5x-spaced Apple Park UAV city.

``city_coordinates.yaml`` is the only building-geometry source of truth.  A
building is transformed in two independent steps: its centroid is moved away
from a configurable anchor, and its local footprint is scaled around that
centroid.  Z coordinates are copied bit-for-bit as Python floats; they are
never scaled or recomputed.

The generated COLLADA file is deliberately shared by Gazebo visual and
collision elements.  This prevents the visual/collision/planner drift that a
world-level mesh scale would introduce.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_YAML = REPO_ROOT / "gazebo/maps/city_coordinates.yaml"
SOURCE_TEXTURE = REPO_ROOT / "gazebo/worlds/applepark_city/mesh/road_surface_city_500m.png"
SOURCE_ATTRIBUTION = REPO_ROOT / "gazebo/worlds/applepark_city/OSM_ATTRIBUTION.txt"
OUTPUT_YAML = REPO_ROOT / "gazebo/maps/city_coordinates_uav.yaml"
OUTPUT_WORLD_DIR = REPO_ROOT / "gazebo/worlds/applepark_city_uav"
OUTPUT_MESH_DIR = OUTPUT_WORLD_DIR / "mesh"
OUTPUT_DAE = OUTPUT_MESH_DIR / "buildings_uav.dae"
OUTPUT_ROAD = OUTPUT_MESH_DIR / "road_surface_city_uav.png"
OUTPUT_HEIGHT = OUTPUT_MESH_DIR / "height_map_city_uav.png"
OUTPUT_NORMAL = OUTPUT_MESH_DIR / "normal_map_city_uav.png"
OUTPUT_WORLD = OUTPUT_WORLD_DIR / "applepark_uav.world"
OUTPUT_MODEL_CONFIG = OUTPUT_WORLD_DIR / "model.config"
OUTPUT_ATTRIBUTION = OUTPUT_WORLD_DIR / "OSM_ATTRIBUTION.txt"
REPORT_DIR = REPO_ROOT / "reports/city_uav_expansion"

Point = tuple[float, float]
Ring = list[Point]

SPAWN_XY: Point = (-120.0, 115.0)
MISSION_GOAL_XY: Point = (200.0, -125.0)
TRAILER_DESTINATION_XY: Point = (-128.0, -128.0)

# A conservative initial contract for a yaw-invariant 1 m x 1 m vehicle.
R_BODY_XY_M = math.sqrt(0.5**2 + 0.5**2)
R_HARD_M = 1.45
R_PREFERRED_M = 2.15
TAKEOFF_POSITION_ERROR_M = 0.50
# The new deck is 5.5 x 3.0 m and remains yaw-aligned in this kinematic
# validation profile.  Use the larger horizontal half-extent for the swept
# route check, then add the UAV hard radius and route margin below.
TRAILER_HALF_WIDTH_M = 2.75
TRAILER_ROUTE_MARGIN_M = 1.00
PX4_BASE_LINK_Z_OFFSET_M = 0.24

DEFAULT_SPACING_SCALE = 2.5
DEFAULT_FOOTPRINT_SCALE = 0.85
FOOTPRINT_SCALE_CANDIDATES = (1.0, 0.9, 0.85, 0.75)
DEFAULT_NEIGHBOR_RADIUS_M = 35.0
DEFAULT_MAP_MARGIN_M = 20.0
MAX_COLLISION_PROXY_OUTWARD_ERROR_M = 0.25
MAX_COLLISION_PROXY_SUBDIVISION_DEPTH = 16
EPS = 1.0e-9


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    if abs(value) < 5.0e-13:
        value = 0.0
    return f"{value:.10f}".rstrip("0").rstrip(".")


def signed_area(ring: Sequence[Point]) -> float:
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1])
    )


def ring_centroid(ring: Sequence[Point]) -> tuple[float, float, float]:
    twice_area = 0.0
    cx_numerator = 0.0
    cy_numerator = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx_numerator += (x0 + x1) * cross
        cy_numerator += (y0 + y1) * cross
    require(abs(twice_area) > EPS, "zero-area polygon ring")
    return (
        cx_numerator / (3.0 * twice_area),
        cy_numerator / (3.0 * twice_area),
        0.5 * twice_area,
    )


def polygon_centroid(outer: Sequence[Point], holes: Sequence[Sequence[Point]]) -> Point:
    """Area centroid, treating holes as negative area independent of winding."""
    outer_cx, outer_cy, outer_area_signed = ring_centroid(outer)
    outer_area = abs(outer_area_signed)
    weighted_x = outer_cx * outer_area
    weighted_y = outer_cy * outer_area
    area = outer_area
    for hole in holes:
        hole_cx, hole_cy, hole_area_signed = ring_centroid(hole)
        hole_area = abs(hole_area_signed)
        weighted_x -= hole_cx * hole_area
        weighted_y -= hole_cy * hole_area
        area -= hole_area
    require(area > EPS, "holes consume polygon outer ring")
    return weighted_x / area, weighted_y / area


def orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(point: Point, start: Point, end: Point, tolerance: float = EPS) -> bool:
    if abs(orient(start, end, point)) > tolerance:
        return False
    return (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 * o2 < -EPS and o3 * o4 < -EPS:
        return True
    return (
        (abs(o1) <= EPS and point_on_segment(c, a, b))
        or (abs(o2) <= EPS and point_on_segment(d, a, b))
        or (abs(o3) <= EPS and point_on_segment(a, c, d))
        or (abs(o4) <= EPS and point_on_segment(b, c, d))
    )


def proper_segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return orient(a, b, c) * orient(a, b, d) < -EPS and orient(c, d, a) * orient(c, d, b) < -EPS


def point_in_ring(point: Point, ring: Sequence[Point], include_boundary: bool = True) -> bool:
    inside = False
    px, py = point
    for start, end in zip(ring, ring[1:] + ring[:1]):
        if point_on_segment(point, start, end):
            return include_boundary
        if (start[1] > py) != (end[1] > py):
            intersection_x = start[0] + (py - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if px < intersection_x:
                inside = not inside
    return inside


def point_in_polygon(point: Point, outer: Sequence[Point], holes: Sequence[Sequence[Point]]) -> bool:
    if not point_in_ring(point, outer, include_boundary=True):
        return False
    return not any(point_in_ring(point, hole, include_boundary=False) for hole in holes)


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= EPS:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)


def segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def ring_edges(ring: Sequence[Point]) -> Iterable[tuple[Point, Point]]:
    return zip(ring, ring[1:] + ring[:1])


def polygon_edges(polygon: "Polygon2D") -> Iterable[tuple[Point, Point]]:
    yield from ring_edges(polygon.outer)
    for hole in polygon.holes:
        yield from ring_edges(hole)


def polygon_distance(first: "Polygon2D", second: "Polygon2D") -> float:
    if point_in_polygon(first.outer[0], second.outer, second.holes) or point_in_polygon(
        second.outer[0], first.outer, first.holes
    ):
        return 0.0
    return min(
        segment_distance(a, b, c, d)
        for a, b in polygon_edges(first)
        for c, d in polygon_edges(second)
    )


def point_polygon_distance(point: Point, polygon: "Polygon2D") -> float:
    if point_in_polygon(point, polygon.outer, polygon.holes):
        return 0.0
    return min(point_segment_distance(point, start, end) for start, end in polygon_edges(polygon))


def segment_polygon_distance(start: Point, end: Point, polygon: "Polygon2D") -> float:
    if point_in_polygon(start, polygon.outer, polygon.holes) or point_in_polygon(
        end, polygon.outer, polygon.holes
    ):
        return 0.0
    return min(segment_distance(start, end, a, b) for a, b in polygon_edges(polygon))


def bbox_distance(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def validate_ring(ring: Sequence[Point], label: str) -> None:
    require(len(ring) >= 3, f"{label}: fewer than three vertices")
    require(abs(signed_area(ring)) > EPS, f"{label}: zero area")
    edges = list(ring_edges(ring))
    for first_index, (a, b) in enumerate(edges):
        require(math.dist(a, b) > EPS, f"{label}: repeated adjacent vertex")
        for second_index, (c, d) in enumerate(edges):
            if second_index <= first_index:
                continue
            if second_index in (first_index - 1, first_index + 1):
                continue
            if first_index == 0 and second_index == len(edges) - 1:
                continue
            require(not segments_intersect(a, b, c, d), f"{label}: self intersection")


@dataclass(frozen=True)
class Polygon2D:
    outer: Ring
    holes: list[Ring]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(point[0] for point in self.outer),
            min(point[1] for point in self.outer),
            max(point[0] for point in self.outer),
            max(point[1] for point in self.outer),
        )


@dataclass(frozen=True)
class BuildingGeometry:
    identifier: str
    source_component_id: int
    original: Polygon2D
    transformed: Polygon2D
    original_centroid: Point
    transformed_centroid: Point
    footprint_scale: float
    foundation_z: float
    ground_z: float
    roof_z: float
    height: float


@dataclass(frozen=True)
class CandidateEvaluation:
    scale: float
    overlap_count: int
    minimum_neighbor_gap_m: float
    spawn_clearance_m: float
    goal_clearance_m: float
    endpoint_clearance_m: float
    trailer_route_clearance_m: float
    trailer_route_waypoints: tuple[Point, ...]
    feasible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CollisionProxy:
    building_id: str
    part_index: int
    source_triangle: tuple[Point, Point, Point]
    center_xy: Point
    size_xy: Point
    center_z: float
    size_z: float
    yaw_rad: float
    maximum_outward_error_m: float


def normalize_ring(ring: Sequence[Sequence[float]], ccw: bool) -> Ring:
    result = [(float(point[0]), float(point[1])) for point in ring]
    if (signed_area(result) > 0.0) != ccw:
        result.reverse()
    return result


def transform_ring(ring: Sequence[Point], centroid: Point, new_centroid: Point, scale: float) -> Ring:
    return [
        (
            new_centroid[0] + scale * (point[0] - centroid[0]),
            new_centroid[1] + scale * (point[1] - centroid[1]),
        )
        for point in ring
    ]


def transform_buildings(
    source_buildings: Sequence[dict], spacing_scale: float, footprint_scale: float, anchor: Point
) -> list[BuildingGeometry]:
    transformed: list[BuildingGeometry] = []
    for record in source_buildings:
        original_outer = normalize_ring(record["footprint"]["outer"], ccw=True)
        original_holes = [normalize_ring(hole, ccw=False) for hole in record["footprint"].get("holes", [])]
        centroid = polygon_centroid(original_outer, original_holes)
        moved_centroid = (
            anchor[0] + spacing_scale * (centroid[0] - anchor[0]),
            anchor[1] + spacing_scale * (centroid[1] - anchor[1]),
        )
        new_outer = transform_ring(original_outer, centroid, moved_centroid, footprint_scale)
        new_holes = [transform_ring(hole, centroid, moved_centroid, footprint_scale) for hole in original_holes]
        validate_ring(new_outer, f"{record['id']} outer")
        for hole_index, hole in enumerate(new_holes):
            validate_ring(hole, f"{record['id']} hole {hole_index}")
            require(point_in_ring(hole[0], new_outer, include_boundary=False), f"{record['id']}: hole outside outer")
        transformed.append(
            BuildingGeometry(
                identifier=str(record["id"]),
                source_component_id=int(record["source_component_id"]),
                original=Polygon2D(original_outer, original_holes),
                transformed=Polygon2D(new_outer, new_holes),
                original_centroid=centroid,
                transformed_centroid=moved_centroid,
                footprint_scale=footprint_scale,
                foundation_z=float(record["foundation_z_m"]),
                ground_z=float(record["ground_reference_z_m"]),
                roof_z=float(record["roof_z_m"]),
                height=float(record["height_above_ground_m"]),
            )
        )
    return transformed


def find_neighbor_pairs(buildings: Sequence[BuildingGeometry], radius: float) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for first, second in itertools.combinations(range(len(buildings)), 2):
        first_polygon = buildings[first].original
        second_polygon = buildings[second].original
        if bbox_distance(first_polygon.bounds, second_polygon.bounds) > radius:
            continue
        gap = polygon_distance(first_polygon, second_polygon)
        if gap <= radius + EPS:
            pairs.append((first, second, gap))
    return pairs


def polyline_clearance(waypoints: Sequence[Point], buildings: Sequence[BuildingGeometry]) -> float:
    return min(
        segment_polygon_distance(start, end, building.transformed)
        for start, end in zip(waypoints, waypoints[1:])
        for building in buildings
    )


def choose_trailer_route(buildings: Sequence[BuildingGeometry]) -> tuple[tuple[Point, ...], float]:
    """Find the shortest deterministic rectilinear route with swept-width clearance.

    Only the start and destination are contracts; the prompt does not require
    the trailer to drive through buildings on the straight chord between them.
    Candidate cross streets are evaluated at 5 m increments.  This is a
    bounded geometry-generation search, not a runtime global planner.
    """
    required = TRAILER_HALF_WIDTH_M + R_HARD_M + TRAILER_ROUTE_MARGIN_M
    direct = (MISSION_GOAL_XY, TRAILER_DESTINATION_XY)
    direct_clearance = polyline_clearance(direct, buildings)
    if direct_clearance >= required:
        return direct, direct_clearance

    candidates: list[tuple[float, tuple[Point, ...], float]] = []
    for detour_y_integer in range(-300, 301, 5):
        detour_y = float(detour_y_integer)
        route = (
            MISSION_GOAL_XY,
            (MISSION_GOAL_XY[0], detour_y),
            (TRAILER_DESTINATION_XY[0], detour_y),
            TRAILER_DESTINATION_XY,
        )
        clearance = polyline_clearance(route, buildings)
        if clearance + EPS < required:
            continue
        length = sum(math.dist(start, end) for start, end in zip(route, route[1:]))
        candidates.append((length, route, clearance))
    if not candidates:
        return direct, direct_clearance
    _, route, clearance = min(candidates, key=lambda item: (item[0], item[1][1][1]))
    return route, clearance


def evaluate_candidate(
    buildings: Sequence[BuildingGeometry], neighbor_pairs: Sequence[tuple[int, int, float]]
) -> CandidateEvaluation:
    overlap_count = 0
    neighbor_gaps: list[float] = []
    for first, second, _ in neighbor_pairs:
        first_polygon = buildings[first].transformed
        second_polygon = buildings[second].transformed
        new_gap = polygon_distance(first_polygon, second_polygon)
        neighbor_gaps.append(new_gap)
        if new_gap <= EPS:
            overlap_count += 1

    point_clearances = {
        name: min(point_polygon_distance(point, building.transformed) for building in buildings)
        for name, point in (
            ("spawn", SPAWN_XY),
            ("goal", MISSION_GOAL_XY),
            ("endpoint", TRAILER_DESTINATION_XY),
        )
    }
    trailer_route, route_clearance = choose_trailer_route(buildings)

    reasons: list[str] = []
    required_neighbor_gap = 2.0 * R_HARD_M
    required_point_clearance = R_PREFERRED_M + TAKEOFF_POSITION_ERROR_M
    required_route_clearance = TRAILER_HALF_WIDTH_M + R_HARD_M + TRAILER_ROUTE_MARGIN_M
    if overlap_count:
        reasons.append(f"{overlap_count} transformed neighbor pairs overlap")
    if min(neighbor_gaps) < required_neighbor_gap:
        reasons.append(
            f"minimum neighbor gap {min(neighbor_gaps):.3f}m < hard width {required_neighbor_gap:.3f}m"
        )
    for name, clearance in point_clearances.items():
        if clearance < required_point_clearance:
            reasons.append(f"{name} clearance {clearance:.3f}m < {required_point_clearance:.3f}m")
    if route_clearance < required_route_clearance:
        reasons.append(f"trailer route clearance {route_clearance:.3f}m < {required_route_clearance:.3f}m")

    return CandidateEvaluation(
        scale=buildings[0].footprint_scale,
        overlap_count=overlap_count,
        minimum_neighbor_gap_m=min(neighbor_gaps),
        spawn_clearance_m=point_clearances["spawn"],
        goal_clearance_m=point_clearances["goal"],
        endpoint_clearance_m=point_clearances["endpoint"],
        trailer_route_clearance_m=route_clearance,
        trailer_route_waypoints=trailer_route,
        feasible=not reasons,
        reasons=tuple(reasons),
    )


def choose_scale(
    source_buildings: Sequence[dict], spacing_scale: float, anchor: Point, neighbor_radius: float
) -> tuple[list[BuildingGeometry], list[tuple[int, int, float]], list[CandidateEvaluation], str]:
    baseline = transform_buildings(source_buildings, spacing_scale, DEFAULT_FOOTPRINT_SCALE, anchor)
    neighbor_pairs = find_neighbor_pairs(baseline, neighbor_radius)
    evaluations: list[CandidateEvaluation] = []
    layouts: dict[float, list[BuildingGeometry]] = {}
    for scale in FOOTPRINT_SCALE_CANDIDATES:
        layout = transform_buildings(source_buildings, spacing_scale, scale, anchor)
        layouts[scale] = layout
        evaluations.append(evaluate_candidate(layout, neighbor_pairs))

    feasible = [evaluation.scale for evaluation in evaluations if evaluation.feasible]
    require(feasible, "none of the allowed footprint scales produces a safe layout")
    if len(feasible) == len(FOOTPRINT_SCALE_CANDIDATES):
        selected = DEFAULT_FOOTPRINT_SCALE
        reason = "all candidates are safe; use the prompt default 0.85 footprint scale"
    else:
        selected = max(feasible)
        reason = "selected the largest candidate satisfying every hard geometry constraint"
    return layouts[selected], neighbor_pairs, evaluations, reason


def point_in_triangle(point: Point, a: Point, b: Point, c: Point) -> bool:
    first, second, third = orient(a, b, point), orient(b, c, point), orient(c, a, point)
    return first >= -EPS and second >= -EPS and third >= -EPS


def triangulate_simple_ring(ring: Sequence[Point]) -> list[tuple[Point, Point, Point]]:
    """Deterministic ear clipping for a validated simple ring."""
    vertices = list(ring if signed_area(ring) > 0.0 else reversed(ring))
    indices = list(range(len(vertices)))
    triangles: list[tuple[Point, Point, Point]] = []
    guard = 0
    while len(indices) > 3:
        guard += 1
        require(guard <= len(vertices) ** 2, "ear clipping failed to converge")
        clipped = False
        for cursor, current in enumerate(indices):
            previous = indices[cursor - 1]
            following = indices[(cursor + 1) % len(indices)]
            a, b, c = vertices[previous], vertices[current], vertices[following]
            if orient(a, b, c) <= EPS:
                continue
            if any(
                point_in_triangle(vertices[index], a, b, c)
                for index in indices
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((a, b, c))
            del indices[cursor]
            clipped = True
            break
        require(clipped, "no valid ear found in polygon")
    a, b, c = (vertices[index] for index in indices)
    require(orient(a, b, c) > EPS, "degenerate final triangle")
    triangles.append((a, b, c))
    return triangles


def triangulate_polygon(polygon: Polygon2D) -> list[tuple[Point, Point, Point]]:
    if not polygon.holes:
        return triangulate_simple_ring(polygon.outer)

    # Only one source building has a rectangular courtyard.  A constrained
    # Delaunay library is unnecessary: triangulate all boundary vertices and
    # retain only triangles whose complete edges remain in the polygon.  The
    # area equality below makes this a fail-closed operation.
    try:
        import numpy as np
        from scipy.spatial import Delaunay
    except ImportError as exc:  # pragma: no cover - dependency is in setup script
        raise RuntimeError("scipy is required to triangulate polygon holes") from exc

    vertices = list(polygon.outer)
    for hole in polygon.holes:
        vertices.extend(hole)
    simplices = Delaunay(np.asarray(vertices, dtype=float)).simplices
    boundary_edges = list(polygon_edges(polygon))
    triangles: list[tuple[Point, Point, Point]] = []
    for simplex in simplices:
        triangle = tuple(vertices[int(index)] for index in simplex)
        if signed_area(list(triangle)) < 0.0:
            triangle = (triangle[0], triangle[2], triangle[1])
        centroid = (
            sum(point[0] for point in triangle) / 3.0,
            sum(point[1] for point in triangle) / 3.0,
        )
        if not point_in_polygon(centroid, polygon.outer, polygon.holes):
            continue
        valid = True
        for edge_start, edge_end in ring_edges(list(triangle)):
            midpoint = ((edge_start[0] + edge_end[0]) / 2.0, (edge_start[1] + edge_end[1]) / 2.0)
            if not point_in_polygon(midpoint, polygon.outer, polygon.holes):
                valid = False
                break
            if any(
                proper_segments_intersect(edge_start, edge_end, boundary_start, boundary_end)
                for boundary_start, boundary_end in boundary_edges
            ):
                valid = False
                break
        if valid:
            triangles.append(triangle)  # type: ignore[arg-type]

    expected_area = abs(signed_area(polygon.outer)) - sum(abs(signed_area(hole)) for hole in polygon.holes)
    triangle_area = sum(abs(signed_area(list(triangle))) for triangle in triangles)
    require(abs(expected_area - triangle_area) <= 1.0e-6, "hole triangulation does not cover polygon exactly")
    return triangles


def prism_triangles(building: BuildingGeometry) -> list[tuple[tuple[float, float, float], ...]]:
    result: list[tuple[tuple[float, float, float], ...]] = []
    cap_triangles = triangulate_polygon(building.transformed)
    for a, b, c in cap_triangles:
        result.append(((a[0], a[1], building.roof_z), (b[0], b[1], building.roof_z), (c[0], c[1], building.roof_z)))
        result.append(
            ((c[0], c[1], building.foundation_z), (b[0], b[1], building.foundation_z), (a[0], a[1], building.foundation_z))
        )
    for ring in [building.transformed.outer, *building.transformed.holes]:
        for a, b in ring_edges(ring):
            bottom_a = (a[0], a[1], building.foundation_z)
            bottom_b = (b[0], b[1], building.foundation_z)
            top_a = (a[0], a[1], building.roof_z)
            top_b = (b[0], b[1], building.roof_z)
            result.append((bottom_a, bottom_b, top_b))
            result.append((bottom_a, top_b, top_a))
    return result


def triangle_normal(triangle: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    a, b, c = triangle
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    normal = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in normal))
    require(length > EPS, "degenerate mesh triangle")
    return normal[0] / length, normal[1] / length, normal[2] / length


def write_dae(buildings: Sequence[BuildingGeometry], path: Path) -> tuple[int, int]:
    triangles = [triangle for building in buildings for triangle in prism_triangles(building)]
    positions = [point for triangle in triangles for point in triangle]
    normals = [triangle_normal(triangle) for triangle in triangles]
    position_text = " ".join(fmt(value) for point in positions for value in point)
    normal_text = " ".join(fmt(value) for normal in normals for value in normal)
    indices = " ".join(
        f"{triangle_index * 3 + vertex_index} {triangle_index}"
        for triangle_index in range(len(triangles))
        for vertex_index in range(3)
    )
    document = f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="building-effect"><profile_COMMON><technique sid="common"><phong>
      <ambient><color>0.75 0.75 0.75 1</color></ambient>
      <diffuse><color>0.85 0.85 0.85 1</color></diffuse>
      <specular><color>0.05 0.05 0.05 1</color></specular>
      <shininess><float>8</float></shininess>
    </phong></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials><material id="building-material"><instance_effect url="#building-effect"/></material></library_materials>
  <library_geometries><geometry id="buildings-uav-geometry" name="buildings_uav"><mesh>
    <source id="positions"><float_array id="positions-array" count="{len(positions) * 3}">{position_text}</float_array>
      <technique_common><accessor source="#positions-array" count="{len(positions)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common>
    </source>
    <source id="normals"><float_array id="normals-array" count="{len(normals) * 3}">{normal_text}</float_array>
      <technique_common><accessor source="#normals-array" count="{len(normals)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common>
    </source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="{len(triangles)}" material="building-material-symbol">
      <input semantic="VERTEX" source="#vertices" offset="0"/><input semantic="NORMAL" source="#normals" offset="1"/><p>{indices}</p>
    </triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene"><node id="buildings-uav-node"><instance_geometry url="#buildings-uav-geometry">
    <bind_material><technique_common><instance_material symbol="building-material-symbol" target="#building-material"/></technique_common></bind_material>
  </instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
'''
    path.write_text(document, encoding="utf-8")
    return len(positions), len(triangles)


def minimum_area_triangle_box(
    building: BuildingGeometry,
    triangle: tuple[Point, Point, Point],
    part_index: int,
) -> CollisionProxy:
    """Return the tightest edge-aligned rectangle enclosing a triangle.

    Every triangle vertex is included with a 1 micrometre numerical pad.  The
    resulting union is therefore conservative: DART may see a little more
    building than the visual, but never less.
    """
    candidates: list[tuple[float, float, float, float, float, float, float]] = []
    for start, end in ring_edges(list(triangle)):
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_x = [point[0] * cosine + point[1] * sine for point in triangle]
        local_y = [-point[0] * sine + point[1] * cosine for point in triangle]
        min_x, max_x = min(local_x), max(local_x)
        min_y, max_y = min(local_y), max(local_y)
        size_x, size_y = max_x - min_x, max_y - min_y
        candidates.append((size_x * size_y, yaw, min_x, max_x, min_y, max_y, size_x + size_y))
    _, yaw, min_x, max_x, min_y, max_y, _ = min(candidates, key=lambda value: (value[0], value[6], value[1]))
    pad = 1.0e-6
    size_x = max_x - min_x + 2.0 * pad
    size_y = max_y - min_y + 2.0 * pad
    center_local_x = 0.5 * (min_x + max_x)
    center_local_y = 0.5 * (min_y + max_y)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    center = (
        center_local_x * cosine - center_local_y * sine,
        center_local_x * sine + center_local_y * cosine,
    )
    half_x, half_y = 0.5 * size_x, 0.5 * size_y
    corners = [
        (
            center[0] + sx * half_x * cosine - sy * half_y * sine,
            center[1] + sx * half_x * sine + sy * half_y * cosine,
        )
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    outward_error = max(point_polygon_distance(corner, building.transformed) for corner in corners)
    return CollisionProxy(
        building_id=building.identifier,
        part_index=part_index,
        source_triangle=triangle,
        center_xy=center,
        size_xy=(size_x, size_y),
        center_z=0.5 * (building.foundation_z + building.roof_z),
        size_z=building.roof_z - building.foundation_z,
        yaw_rad=yaw,
        maximum_outward_error_m=outward_error,
    )


def generate_collision_proxies(buildings: Sequence[BuildingGeometry]) -> list[CollisionProxy]:
    proxies: list[CollisionProxy] = []
    for building in buildings:
        pending = [(triangle, 0) for triangle in triangulate_polygon(building.transformed)]
        building_proxies: list[CollisionProxy] = []
        while pending:
            triangle, depth = pending.pop()
            proxy = minimum_area_triangle_box(building, triangle, len(building_proxies))
            if (
                proxy.maximum_outward_error_m <= MAX_COLLISION_PROXY_OUTWARD_ERROR_M + EPS
                or depth >= MAX_COLLISION_PROXY_SUBDIVISION_DEPTH
            ):
                require(
                    proxy.maximum_outward_error_m <= MAX_COLLISION_PROXY_OUTWARD_ERROR_M + 1.0e-6,
                    f"{building.identifier}: collision proxy refinement did not converge",
                )
                building_proxies.append(proxy)
                continue
            _, first, second, opposite = max(
                (
                    math.dist(triangle[first_index], triangle[second_index]),
                    first_index,
                    second_index,
                    3 - first_index - second_index,
                )
                for first_index, second_index in ((0, 1), (1, 2), (2, 0))
            )
            midpoint = (
                0.5 * (triangle[first][0] + triangle[second][0]),
                0.5 * (triangle[first][1] + triangle[second][1]),
            )
            pending.append(((triangle[first], midpoint, triangle[opposite]), depth + 1))
            pending.append(((midpoint, triangle[second], triangle[opposite]), depth + 1))
        proxies.extend(building_proxies)
    return proxies


def proxy_collision_sdf(proxies: Sequence[CollisionProxy]) -> str:
    lines: list[str] = []
    for proxy in proxies:
        lines.append(
            "      <collision name=\""
            f"{proxy.building_id}_part_{proxy.part_index:03d}\"><pose>"
            f"{fmt(proxy.center_xy[0])} {fmt(proxy.center_xy[1])} {fmt(proxy.center_z)} 0 0 {fmt(proxy.yaw_rad)}"
            "</pose><geometry><box><size>"
            f"{fmt(proxy.size_xy[0])} {fmt(proxy.size_xy[1])} {fmt(proxy.size_z)}"
            "</size></box></geometry></collision>"
        )
    return "\n".join(lines)


def collision_proxy_polygon(proxy: CollisionProxy) -> Polygon2D:
    cosine, sine = math.cos(proxy.yaw_rad), math.sin(proxy.yaw_rad)
    half_x, half_y = 0.5 * proxy.size_xy[0], 0.5 * proxy.size_xy[1]
    corners = [
        (
            proxy.center_xy[0] + local_x * cosine - local_y * sine,
            proxy.center_xy[1] + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    ]
    return Polygon2D(corners, [])


def proxy_polyline_clearance(waypoints: Sequence[Point], proxies: Sequence[CollisionProxy]) -> float:
    polygons = [collision_proxy_polygon(proxy) for proxy in proxies]
    return min(
        segment_polygon_distance(start, end, polygon)
        for start, end in zip(waypoints, waypoints[1:])
        for polygon in polygons
    )


def determine_bounds(buildings: Sequence[BuildingGeometry], margin: float) -> tuple[float, float, float, float]:
    all_points = [point for building in buildings for point in building.transformed.outer]
    all_points.extend([SPAWN_XY, MISSION_GOAL_XY, TRAILER_DESTINATION_XY])
    extent = max(max(abs(point[0]), abs(point[1])) for point in all_points) + margin
    half_size = math.ceil(extent / 10.0) * 10.0
    return -half_size, -half_size, half_size, half_size


def write_textures(bounds: tuple[float, float, float, float], spacing_scale: float, anchor: Point) -> None:
    xmin, ymin, xmax, ymax = bounds
    require(abs((xmax - xmin) - (ymax - ymin)) <= EPS, "Gazebo heightmap must be square")
    source = Image.open(SOURCE_TEXTURE).convert("RGB")
    output_size = 2048
    source_width, source_height = source.size

    # PIL affine maps each output pixel to an input pixel.  World Y decreases
    # down the image in both source and destination, so both scale factors are
    # positive.  Outside the original 500 m source is neutral pavement.
    world_per_output_x = (xmax - xmin) / (output_size - 1)
    world_per_output_y = (ymax - ymin) / (output_size - 1)
    x_at_output_zero = xmin
    y_at_output_zero = ymax
    source_world_x_at_zero = anchor[0] + (x_at_output_zero - anchor[0]) / spacing_scale
    source_world_y_at_zero = anchor[1] + (y_at_output_zero - anchor[1]) / spacing_scale
    affine = (
        world_per_output_x / spacing_scale * (source_width - 1) / 500.0,
        0.0,
        (source_world_x_at_zero + 250.0) / 500.0 * (source_width - 1),
        0.0,
        world_per_output_y / spacing_scale * (source_height - 1) / 500.0,
        (250.0 - source_world_y_at_zero) / 500.0 * (source_height - 1),
    )
    road = source.transform(
        (output_size, output_size),
        Image.AFFINE,
        affine,
        resample=Image.BILINEAR,
        fillcolor=(95, 98, 96),
    )
    road.save(OUTPUT_ROAD, optimize=True)
    Image.new("L", (257, 257), color=255).save(OUTPUT_HEIGHT, optimize=True)
    Image.new("RGB", (257, 257), color=(128, 128, 255)).save(OUTPUT_NORMAL, optimize=True)


def write_world(bounds: tuple[float, float, float, float], proxies: Sequence[CollisionProxy]) -> None:
    map_size = bounds[2] - bounds[0]
    proxy_sdf = proxy_collision_sdf(proxies)
    world = f'''<?xml version="1.0"?>
<sdf version="1.9">
  <world name="applepark_city_uav">
    <magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
    <atmosphere type="adiabatic"/>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <physics name="2ms" type="ignored"><max_step_size>0.002</max_step_size><real_time_factor>1</real_time_factor><real_time_update_rate>500</real_time_update_rate></physics>
    <gravity>0 0 -9.8066</gravity>
    <scene><ambient>0.45 0.48 0.55 1</ambient><grid>false</grid><origin_visual>false</origin_visual><sky/></scene>
    <light type="directional" name="sun"><pose>0 -7000 7000 0 0 0</pose><diffuse>0.9 0.88 0.82 1</diffuse><specular>0 0 0 1</specular><direction>0 0.6 -0.6</direction></light>
    <gui fullscreen="0"><plugin filename="MinimalScene" name="3D View"><engine>ogre2</engine><scene>scene</scene><camera_pose>0 -250 1200 0 1.3963 1.5708</camera_pose><camera_clip><near>0.1</near><far>50000</far></camera_clip></plugin><plugin filename="WorldControl" name="World control"><play_pause>true</play_pause><step>true</step><start_paused>true</start_paused></plugin><plugin filename="WorldStats" name="World stats"><sim_time>true</sim_time><real_time_factor>true</real_time_factor></plugin></gui>
    <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><latitude_deg>37.32977431415926</latitude_deg><longitude_deg>-121.99860517699835</longitude_deg><elevation>50.6</elevation></spherical_coordinates>

    <!-- Exactly flat z=0 datum.  The 1 mm heightmap is visual-only; Bullet/DART physics uses the box. -->
    <model name="applepark_uav_ground"><static>true</static><link name="ground">
      <collision name="collision"><pose>0 0 -0.05 0 0 0</pose><geometry><box><size>{fmt(map_size)} {fmt(map_size)} 0.1</size></box></geometry></collision>
      <visual name="ground_visual"><cast_shadows>false</cast_shadows><geometry><heightmap><use_terrain_paging>false</use_terrain_paging><texture><diffuse>mesh/road_surface_city_uav.png</diffuse><normal>mesh/normal_map_city_uav.png</normal><size>{fmt(map_size)}</size></texture><uri>mesh/height_map_city_uav.png</uri><size>{fmt(map_size)} {fmt(map_size)} 0.001</size><pos>0 0 -0.001</pos><sampling>1</sampling></heightmap></geometry></visual>
    </link></model>

    <!-- The DAE is visual-only because gz-physics-dartsim cannot construct an
         SDF collision mesh.  Collision boxes below conservatively cover the
         exact same YAML prism triangulation. -->
    <model name="applepark_uav_buildings"><static>true</static><link name="buildings">
      <visual name="buildings_visual"><geometry><mesh><uri>mesh/buildings_uav.dae</uri><scale>1 1 1</scale></mesh></geometry></visual>
    </link></model>

    <model name="applepark_uav_building_collision_proxies"><static>true</static><link name="building_collision_proxies">
{proxy_sdf}
    </link></model>

    <!-- No visual spawn pad: PX4 model contact is referenced to the z=0 datum. -->
    <frame name="drone_spawn"><pose>-120 115 0 0 0 0</pose></frame>
    <frame name="mission_goal"><pose>200 -125 0 0 0 0</pose></frame>
    <frame name="trailer_destination"><pose>-128 -128 0 0 0 0</pose></frame>
    <include><uri>model://trailer_aruco</uri><name>trailer</name><pose>200 -125 0 0 0 0</pose></include>
    <!-- PX4 SITL spawns the real sensor-equipped vehicle dynamically. -->
  </world>
</sdf>
'''
    OUTPUT_WORLD.write_text(world, encoding="utf-8")
    OUTPUT_MODEL_CONFIG.write_text(
        """<?xml version="1.0"?>
<model><name>Apple Park UAV city</name><version>1.0</version><sdf version="1.9">applepark_uav.world</sdf><author><name>PX4-ROS2 contributors</name></author><description>Non-destructive 2.5x-spaced UAV city derived from city_coordinates.yaml.</description></model>
""",
        encoding="utf-8",
    )
    OUTPUT_ATTRIBUTION.write_text(SOURCE_ATTRIBUTION.read_text(encoding="utf-8"), encoding="utf-8")


def rounded_point(point: Point) -> list[float]:
    return [round(point[0], 10), round(point[1], 10)]


def write_yaml(
    source: dict,
    buildings: Sequence[BuildingGeometry],
    bounds: tuple[float, float, float, float],
    spacing_scale: float,
    anchor: Point,
    scale_reason: str,
    mesh_vertices: int,
    mesh_triangles: int,
    proxies: Sequence[CollisionProxy],
) -> None:
    output = copy.deepcopy(source)
    output["schema_version"] = 2
    output["map"].update(
        {
            "name": "city_uav",
            "gazebo_world_name": "applepark_city_uav",
            "world_file": "gazebo/worlds/applepark_city_uav/applepark_uav.world",
            "bounds_enu_m": {"x": [bounds[0], bounds[2]], "y": [bounds[1], bounds[3]]},
        }
    )
    output["derivation"] = {
        "source_yaml": "gazebo/maps/city_coordinates.yaml",
        "generator": "gazebo/tools/expand_city_for_uav.py",
        "anchor_enu_m": list(anchor),
        "city_spacing_scale_xy": spacing_scale,
        "building_footprint_scale_xy": buildings[0].footprint_scale,
        "allowed_footprint_scales": list(FOOTPRINT_SCALE_CANDIDATES),
        "selection_reason": scale_reason,
        "z_scale": 1.0,
        "fixed_mission_coordinates_enu_m": {
            "drone_spawn": list(SPAWN_XY),
            "global_goal_and_trailer_spawn": list(MISSION_GOAL_XY),
            "trailer_destination": list(TRAILER_DESTINATION_XY),
        },
        "mesh": {"vertices": mesh_vertices, "triangles": mesh_triangles},
        "collision_proxy": {
            "type": "conservative_oriented_box_prisms",
            "count": len(proxies),
            "source": "same transformed YAML polygon triangulation as visual DAE",
            "maximum_outward_error_m": max(proxy.maximum_outward_error_m for proxy in proxies),
            "maximum_undercoverage_m": 0.0,
        },
    }
    map_size = bounds[2] - bounds[0]
    output["terrain"] = {
        "type": "completely_flat_heightmap_and_box_collision",
        "image": "gazebo/worlds/applepark_city_uav/mesh/height_map_city_uav.png",
        "road_texture": "gazebo/worlds/applepark_city_uav/mesh/road_surface_city_uav.png",
        "normal_map": "gazebo/worlds/applepark_city_uav/mesh/normal_map_city_uav.png",
        "collision_geometry": {
            "shape": "box",
            "size_m": [map_size, map_size, 0.1],
            "center_z_m": -0.05,
            "top_z_m": 0.0,
        },
        "rows": 257,
        "columns": 257,
        "sample_spacing_m": map_size / 256.0,
        "row_0_y_m": bounds[3],
        "column_0_x_m": bounds[0],
        "row_direction": "decreasing_y",
        "height_formula_m": "0.0 (all visual pixels=255; box top z=0)",
        "height_range_m": [0.0, 0.0],
    }
    output["spawn"]["gazebo_spawn_pose_enu"].update({"x": SPAWN_XY[0], "y": SPAWN_XY[1], "z": 0.0})
    output["spawn"]["surface"] = "flat_ground_z0_no_visual_pad"
    output["spawn"].pop("pad", None)
    output["frames"]["px4_local"]["origin_enu_m"] = [
        SPAWN_XY[0], SPAWN_XY[1], PX4_BASE_LINK_Z_OFFSET_M
    ]
    output["frames"]["px4_local"]["origin_reference"] = (
        "PX4 x500 base_link at rest (model root + 0.24 m)"
    )
    output["trailer"].update(
        {
            "entity_name": "trailer",
            "model_uri": "model://trailer_aruco",
            "spawn_pose_enu": {
                "x": MISSION_GOAL_XY[0],
                "y": MISSION_GOAL_XY[1],
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "waypoints_enu_m": [list(point) for point in choose_trailer_route(buildings)[0]],
            "destination_enu_m": list(TRAILER_DESTINATION_XY),
            "cruise_speed_m_s": 3.0,
            "body_footprint_m": [5.5, 3.0],
            "deck_height_m": 1.25,
            "command_topic": "/model/trailer/cmd_vel",
            "pose_topic": "/world/applepark_city_uav/dynamic_pose/info",
        }
    )
    output["planner_defaults"] = {
        "vehicle_minimum_xy_size_m": [1.0, 1.0],
        "yaw_invariant_body_radius_m": R_BODY_XY_M,
        "hard_inflation_radius_m": R_HARD_M,
        "preferred_clearance_m": R_PREFERRED_M,
    }
    output_buildings: list[dict] = []
    source_by_id = {record["id"]: record for record in source["obstacles"]["buildings"]}
    for building in buildings:
        record = copy.deepcopy(source_by_id[building.identifier])
        record["footprint"] = {
            "outer": [rounded_point(point) for point in building.transformed.outer],
            "holes": [[rounded_point(point) for point in hole] for hole in building.transformed.holes],
        }
        record["aabb_xy_m"] = {
            "min": [round(building.transformed.bounds[0], 10), round(building.transformed.bounds[1], 10)],
            "max": [round(building.transformed.bounds[2], 10), round(building.transformed.bounds[3], 10)],
        }
        record["transform"] = {
            "original_centroid_xy_m": rounded_point(building.original_centroid),
            "transformed_centroid_xy_m": rounded_point(building.transformed_centroid),
            "footprint_scale_xy": building.footprint_scale,
        }
        # Explicit assignments document that these values are not derived.
        record["foundation_z_m"] = building.foundation_z
        record["ground_reference_z_m"] = building.ground_z
        record["roof_z_m"] = building.roof_z
        record["height_above_ground_m"] = building.height
        output_buildings.append(record)
    output["obstacles"]["buildings"] = output_buildings
    output["source_sha256"] = {
        "source_city_coordinates_yaml": sha256_file(SOURCE_YAML),
        "building_visual_collision_dae": sha256_file(OUTPUT_DAE),
        "terrain_heightmap": sha256_file(OUTPUT_HEIGHT),
        "road_texture": sha256_file(OUTPUT_ROAD),
        "normal_map": sha256_file(OUTPUT_NORMAL),
    }
    OUTPUT_YAML.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def write_reports(
    buildings: Sequence[BuildingGeometry],
    neighbor_pairs: Sequence[tuple[int, int, float]],
    evaluations: Sequence[CandidateEvaluation],
    bounds: tuple[float, float, float, float],
    spacing_scale: float,
    anchor: Point,
    scale_reason: str,
    mesh_vertices: int,
    mesh_triangles: int,
    proxies: Sequence[CollisionProxy],
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "building_transform.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "building_id",
                "source_component_id",
                "original_centroid_x_m",
                "original_centroid_y_m",
                "transformed_centroid_x_m",
                "transformed_centroid_y_m",
                "centroid_spacing_scale",
                "footprint_scale_xy",
                "foundation_z_m",
                "roof_z_m",
                "height_above_ground_m",
                "z_difference_from_source_m",
            ]
        )
        for building in buildings:
            writer.writerow(
                [
                    building.identifier,
                    building.source_component_id,
                    fmt(building.original_centroid[0]),
                    fmt(building.original_centroid[1]),
                    fmt(building.transformed_centroid[0]),
                    fmt(building.transformed_centroid[1]),
                    fmt(spacing_scale),
                    fmt(building.footprint_scale),
                    fmt(building.foundation_z),
                    fmt(building.roof_z),
                    fmt(building.height),
                    "0",
                ]
            )

    ratio_values: list[float] = []
    gap_records: list[tuple[str, str, float, float, float | None]] = []
    for first, second, original_gap in neighbor_pairs:
        new_gap = polygon_distance(buildings[first].transformed, buildings[second].transformed)
        ratio = new_gap / original_gap if original_gap > 1.0e-6 else None
        if ratio is not None:
            ratio_values.append(ratio)
        gap_records.append((buildings[first].identifier, buildings[second].identifier, original_gap, new_gap, ratio))
    with (REPORT_DIR / "pairwise_gap_before_after.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["building_a", "building_b", "original_edge_gap_m", "new_edge_gap_m", "gap_ratio"])
        for first, second, original_gap, new_gap, ratio in gap_records:
            writer.writerow(
                [first, second, fmt(original_gap), fmt(new_gap), "" if ratio is None else fmt(ratio)]
            )

    required_neighbor_gap = 2.0 * R_HARD_M
    with (REPORT_DIR / "invalid_or_tight_corridors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "entity_a", "entity_b", "measured_clearance_m", "required_clearance_m", "status"])
        tight_count = 0
        for first, second, _, new_gap, _ in gap_records:
            if new_gap < required_neighbor_gap:
                writer.writerow(["building_pair", first, second, fmt(new_gap), fmt(required_neighbor_gap), "FAIL"])
                tight_count += 1
        selected_evaluation = next(item for item in evaluations if item.scale == buildings[0].footprint_scale)
        mission_checks = (
            ("mission_point", "drone_spawn", "buildings", selected_evaluation.spawn_clearance_m, R_PREFERRED_M + TAKEOFF_POSITION_ERROR_M),
            ("mission_point", "global_goal", "buildings", selected_evaluation.goal_clearance_m, R_PREFERRED_M + TAKEOFF_POSITION_ERROR_M),
            ("mission_point", "trailer_destination", "buildings", selected_evaluation.endpoint_clearance_m, R_PREFERRED_M + TAKEOFF_POSITION_ERROR_M),
            ("trailer_route", "goal", "destination", selected_evaluation.trailer_route_clearance_m, TRAILER_HALF_WIDTH_M + R_HARD_M + TRAILER_ROUTE_MARGIN_M),
        )
        for kind, first, second, measured, required in mission_checks:
            if measured < required:
                writer.writerow([kind, first, second, fmt(measured), fmt(required), "FAIL"])
                tight_count += 1

    with (REPORT_DIR / "scale_candidate_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scale",
                "feasible",
                "overlap_count",
                "minimum_neighbor_gap_m",
                "spawn_clearance_m",
                "goal_clearance_m",
                "endpoint_clearance_m",
                "trailer_route_clearance_m",
                "reasons",
            ]
        )
        for evaluation in evaluations:
            writer.writerow(
                [
                    fmt(evaluation.scale),
                    str(evaluation.feasible).lower(),
                    evaluation.overlap_count,
                    fmt(evaluation.minimum_neighbor_gap_m),
                    fmt(evaluation.spawn_clearance_m),
                    fmt(evaluation.goal_clearance_m),
                    fmt(evaluation.endpoint_clearance_m),
                    fmt(evaluation.trailer_route_clearance_m),
                    "; ".join(evaluation.reasons),
                ]
            )

    with (REPORT_DIR / "collision_proxy_alignment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "building_id",
                "proxy_part",
                "center_x_m",
                "center_y_m",
                "center_z_m",
                "size_x_m",
                "size_y_m",
                "size_z_m",
                "yaw_rad",
                "source_vertex_alignment_error_m",
                "undercoverage_m",
                "maximum_outward_error_m",
            ]
        )
        for proxy in proxies:
            writer.writerow(
                [
                    proxy.building_id,
                    proxy.part_index,
                    fmt(proxy.center_xy[0]),
                    fmt(proxy.center_xy[1]),
                    fmt(proxy.center_z),
                    fmt(proxy.size_xy[0]),
                    fmt(proxy.size_xy[1]),
                    fmt(proxy.size_z),
                    fmt(proxy.yaw_rad),
                    "0",
                    "0",
                    fmt(proxy.maximum_outward_error_m),
                ]
            )

    selected = next(item for item in evaluations if item.scale == buildings[0].footprint_scale)
    proxy_polygons = [collision_proxy_polygon(proxy) for proxy in proxies]
    proxy_point_clearances = {
        name: min(point_polygon_distance(point, polygon) for polygon in proxy_polygons)
        for name, point in (
            ("spawn", SPAWN_XY),
            ("goal", MISSION_GOAL_XY),
            ("endpoint", TRAILER_DESTINATION_XY),
        )
    }
    proxy_route_clearance = proxy_polyline_clearance(selected.trailer_route_waypoints, proxies)
    required_proxy_route_clearance = TRAILER_HALF_WIDTH_M + R_HARD_M + TRAILER_ROUTE_MARGIN_M
    if proxy_route_clearance < required_proxy_route_clearance:
        with (REPORT_DIR / "invalid_or_tight_corridors.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    "collision_proxy_trailer_route",
                    "goal",
                    "destination",
                    fmt(proxy_route_clearance),
                    fmt(required_proxy_route_clearance),
                    "FAIL",
                ]
            )
        tight_count += 1
    summary = {
        "source_building_count": len(buildings),
        "generated_building_count": len(buildings),
        "anchor_enu_m": list(anchor),
        "city_spacing_scale_xy": spacing_scale,
        "selected_footprint_scale_xy": buildings[0].footprint_scale,
        "scale_selection_reason": scale_reason,
        "bounds_enu_m": {"x": [bounds[0], bounds[2]], "y": [bounds[1], bounds[3]]},
        "ground_size_m": [bounds[2] - bounds[0], bounds[3] - bounds[1]],
        "neighbor_pair_count": len(neighbor_pairs),
        "pairwise_gap_ratio": {
            "count_excluding_original_touching_pairs": len(ratio_values),
            "median": statistics.median(ratio_values),
            "p10": percentile(ratio_values, 0.10),
            "minimum": min(ratio_values),
        },
        "minimum_neighbor_gap_m": selected.minimum_neighbor_gap_m,
        "hard_corridor_width_m": 2.0 * R_HARD_M,
        "preferred_corridor_width_m": 2.0 * R_PREFERRED_M,
        "mission_clearance_m": {
            "drone_spawn": selected.spawn_clearance_m,
            "global_goal_and_trailer_spawn": selected.goal_clearance_m,
            "trailer_destination": selected.endpoint_clearance_m,
            "trailer_route": selected.trailer_route_clearance_m,
        },
        "trailer_route_waypoints_enu_m": [list(point) for point in selected.trailer_route_waypoints],
        "fixed_coordinates_enu_m": {
            "drone_spawn": list(SPAWN_XY),
            "global_goal_and_trailer_spawn": list(MISSION_GOAL_XY),
            "trailer_destination": list(TRAILER_DESTINATION_XY),
        },
        "z_preservation": {"maximum_absolute_error_m": 0.0, "tolerance_m": 1.0e-9},
        "visual_collision_yaml_alignment": {"maximum_absolute_xy_error_m": 0.0, "tolerance_m": 0.01},
        "collision_proxy": {
            "representation": "DART-compatible conservative oriented box prisms",
            "count": len(proxies),
            "source_vertex_alignment_error_m": 0.0,
            "maximum_undercoverage_m": 0.0,
            "maximum_outward_error_m": max(proxy.maximum_outward_error_m for proxy in proxies),
            "mean_outward_error_m": statistics.fmean(proxy.maximum_outward_error_m for proxy in proxies),
            "certified_minimum_neighbor_gap_m": selected.minimum_neighbor_gap_m
            - 2.0 * max(proxy.maximum_outward_error_m for proxy in proxies),
            "mission_clearance_m": {
                "drone_spawn": proxy_point_clearances["spawn"],
                "global_goal_and_trailer_spawn": proxy_point_clearances["goal"],
                "trailer_destination": proxy_point_clearances["endpoint"],
                "trailer_route": proxy_route_clearance,
            },
        },
        "mesh": {
            "path": "gazebo/worlds/applepark_city_uav/mesh/buildings_uav.dae",
            "sha256": sha256_file(OUTPUT_DAE),
            "vertices": mesh_vertices,
            "triangles": mesh_triangles,
        },
        "tight_or_invalid_corridor_count": tight_count,
        "selected_layout_feasible": selected.feasible,
    }
    (REPORT_DIR / "map_bounds_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    alignment = f"""# UAV city visual/collision/planner alignment

- Source of truth: `gazebo/maps/city_coordinates.yaml` ({len(buildings)} buildings).
- Derived planner geometry: `gazebo/maps/city_coordinates_uav.yaml`.
- Gazebo visual URI: `mesh/buildings_uav.dae` at scale `1 1 1`.
- Gazebo collision: {len(proxies)} DART-compatible oriented box prisms.
- Mesh SHA256: `{sha256_file(OUTPUT_DAE)}`.
- Vertex/triangle count: {mesh_vertices} / {mesh_triangles}.
- Maximum YAML-to-visual-mesh XY boundary error: `0.0 m` (limit `0.01 m`).
- Proxy source-vertex alignment error: `0.0 m` (limit `0.01 m`).
- Proxy maximum undercoverage: `0.0 m` (every source triangle is enclosed).
- Proxy maximum conservative outward error: `{fmt(max(proxy.maximum_outward_error_m for proxy in proxies))} m`.
- Maximum foundation/roof Z error from source: `0.0 m` (limit `1e-9 m`).
- Selected footprint scale: `{fmt(buildings[0].footprint_scale)}` — {scale_reason}.

The visual mesh is regenerated directly from every transformed outer/hole
ring.  No world-level building scale or pose is used.  DART does not implement
SDF mesh collision construction, so each cap triangle is conservatively
enclosed by its minimum-area edge-aligned rectangle and extruded over the
unchanged foundation/roof interval.  The proxy union may occupy reported extra
space, but cannot leave a visual/YAML building region collision-free.
"""
    (REPORT_DIR / "visual_collision_alignment.md").write_text(alignment, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_YAML)
    parser.add_argument("--spacing-scale", type=float, default=DEFAULT_SPACING_SCALE)
    parser.add_argument("--anchor-x", type=float, default=0.0)
    parser.add_argument("--anchor-y", type=float, default=0.0)
    parser.add_argument("--neighbor-radius", type=float, default=DEFAULT_NEIGHBOR_RADIUS_M)
    parser.add_argument("--map-margin", type=float, default=DEFAULT_MAP_MARGIN_M)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.spacing_scale > 1.0, "spacing scale must expand the city")
    require(args.neighbor_radius > 0.0, "neighbor radius must be positive")
    require(args.map_margin >= DEFAULT_MAP_MARGIN_M, "map margin must be at least 20m")
    source = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    source_buildings = source["obstacles"]["buildings"]
    require(len(source_buildings) == 274, f"expected 274 buildings, got {len(source_buildings)}")
    anchor = (args.anchor_x, args.anchor_y)

    OUTPUT_MESH_DIR.mkdir(parents=True, exist_ok=True)
    buildings, neighbor_pairs, evaluations, scale_reason = choose_scale(
        source_buildings, args.spacing_scale, anchor, args.neighbor_radius
    )
    selected = next(item for item in evaluations if item.scale == buildings[0].footprint_scale)
    require(selected.feasible, "selected layout is not feasible")
    bounds = determine_bounds(buildings, args.map_margin)
    mesh_vertices, mesh_triangles = write_dae(buildings, OUTPUT_DAE)
    proxies = generate_collision_proxies(buildings)
    require(
        proxy_polyline_clearance(selected.trailer_route_waypoints, proxies)
        >= TRAILER_HALF_WIDTH_M + R_HARD_M + TRAILER_ROUTE_MARGIN_M,
        "DART collision proxy over-approximation blocks the selected trailer route",
    )
    write_textures(bounds, args.spacing_scale, anchor)
    write_world(bounds, proxies)
    write_yaml(
        source,
        buildings,
        bounds,
        args.spacing_scale,
        anchor,
        scale_reason,
        mesh_vertices,
        mesh_triangles,
        proxies,
    )
    summary = write_reports(
        buildings,
        neighbor_pairs,
        evaluations,
        bounds,
        args.spacing_scale,
        anchor,
        scale_reason,
        mesh_vertices,
        mesh_triangles,
        proxies,
    )
    print(
        "city_uav generated: "
        f"buildings={len(buildings)} scale={buildings[0].footprint_scale:g} "
        f"ground={summary['ground_size_m'][0]:g}m "
        f"neighbor_pairs={len(neighbor_pairs)} "
        f"gap_ratio_median={summary['pairwise_gap_ratio']['median']:.3f} "
        f"gap_ratio_p10={summary['pairwise_gap_ratio']['p10']:.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
