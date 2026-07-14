#!/usr/bin/env python3
"""Generate the rolled-back, 2.5x-spaced Apple Park UAV city.

``city_coordinates.yaml`` is the only building-geometry source of truth. Each
building centroid is moved away from a configurable anchor and each local XY
footprint is restored to the preceding ``jo`` 0.9 scale. Flight passages are
created by removing buildings, never by shrinking or moving retained buildings.
All Z coordinates are preserved;
they are copied bit-for-bit as Python floats and never scaled or recomputed.
The active profile deterministically removes 69 buildings, retaining 205
(exactly the nearest integer to three quarters) of the 274-building source
city.  A deterministic spatially-stratified random selector distributes those
removals across every part of the map.  It creates many possible passages
without moving or shrinking any retained building.

The generated COLLADA file is deliberately shared by Gazebo visual and
collision elements. This prevents visual/collision/YAML drift that a
world-level mesh scale would introduce.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

# Opposite diagonal road-end staging sites. The active city keeps the drone in
# the north-east and the larger trailer on the wider south-west asphalt strip.
# Passages and spawn safety come from empty road space, not moved buildings.
SPAWN_XY: Point = (587.0, 580.0)
MISSION_GOAL_XY: Point = (200.0, -125.0)
TRAILER_SPAWN_XY: Point = (-587.0, -512.0)
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
DEFAULT_FOOTPRINT_SCALE = 0.9
ACTIVE_HEIGHT_PROFILE = "deterministic_hash_rank_10_to_20m_v1"
FOOTPRINT_SCALE_CANDIDATES = (0.9,)
SOURCE_BUILDING_COUNT = 274
ACTIVE_BUILDING_COUNT = 205
REMOVED_BUILDING_COUNT = SOURCE_BUILDING_COUNT - ACTIVE_BUILDING_COUNT
# Exact quarter-reduction contract. A 5 x 5 source-coordinate grid receives a
# Hamilton quota proportional to its population. SHA-256(seed, building ID)
# provides a stable random order inside every cell, so the selection is both
# spatially distributed and exactly reproducible across Python versions.
REDUCTION_RANDOM_SEED = 7577
REDUCTION_GRID_SIZE = 5
REDUCTION_SOURCE_MIN_M = -250.0
REDUCTION_SOURCE_MAX_M = 250.0
REDUCTION_IDS_SHA256 = "041e0979aaad59280413eba5e758664b7b2653fe1981695633b18d51130dde6d"
REDUCTION_REMOVED_IDS = (
    "building_003", "building_006", "building_008", "building_013", "building_016",
    "building_017", "building_021", "building_023", "building_024", "building_025",
    "building_029", "building_030", "building_035", "building_037", "building_039",
    "building_041", "building_044", "building_052", "building_061", "building_066",
    "building_068", "building_071", "building_075", "building_077", "building_079",
    "building_080", "building_081", "building_085", "building_094", "building_096",
    "building_098", "building_101", "building_107", "building_112", "building_115",
    "building_122", "building_123", "building_138", "building_140", "building_149",
    "building_151", "building_152", "building_160", "building_163", "building_164",
    "building_168", "building_173", "building_177", "building_185", "building_192",
    "building_196", "building_199", "building_201", "building_203", "building_205",
    "building_222", "building_225", "building_230", "building_231", "building_232",
    "building_237", "building_239", "building_240", "building_246", "building_248",
    "building_261", "building_262", "building_264", "building_270",
)
REDUCTION_PROTECTED_IDS = (
    # Map-envelope extrema, the sole courtyard, historical jo sentinels and
    # the exact active 10 / 20 m height extrema are deliberately retained.
    "building_001", "building_009", "building_046", "building_047", "building_131",
    "building_141", "building_147", "building_171", "building_190", "building_202",
    "building_213",
)
REDUCTION_METHOD = (
    "25% deterministic spatial-random filter: 5x5 Hamilton quotas and "
    "SHA-256(seed, building ID) ranking; retained XYZ unchanged"
)
DEFAULT_NEIGHBOR_RADIUS_M = 35.0
DEFAULT_MAP_MARGIN_M = 20.0
# Gazebo Harmonic 8.14 / DART accepts the closed COLLADA prism through its
# AttachMesh fallback.  Using the same single DAE for visual and collision is
# exact, preserves the courtyard hole, and avoids hundreds of SDF entities.
COLLISION_GEOMETRY_TYPE = "shared_exact_dae_mesh"
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
    trailer_spawn_clearance_m: float
    endpoint_clearance_m: float
    trailer_route_clearance_m: float
    trailer_route_waypoints: tuple[Point, ...]
    feasible: bool
    reasons: tuple[str, ...]


def normalize_ring(ring: Sequence[Sequence[float]], ccw: bool) -> Ring:
    result = [(float(point[0]), float(point[1])) for point in ring]
    if (signed_area(result) > 0.0) != ccw:
        result.reverse()
    return result


def reduction_cell(point: Point) -> tuple[int, int]:
    """Return the clamped (row, column) of a source-city centroid."""
    width = (REDUCTION_SOURCE_MAX_M - REDUCTION_SOURCE_MIN_M) / REDUCTION_GRID_SIZE
    column = int(math.floor((point[0] - REDUCTION_SOURCE_MIN_M) / width))
    row = int(math.floor((point[1] - REDUCTION_SOURCE_MIN_M) / width))
    return (
        max(0, min(REDUCTION_GRID_SIZE - 1, row)),
        max(0, min(REDUCTION_GRID_SIZE - 1, column)),
    )


def deterministic_removal_ids(source_buildings: Sequence[dict]) -> tuple[str, ...]:
    """Select 69 spatially distributed IDs with a stable pseudo-random rank."""
    centroids: dict[str, Point] = {}
    cells: dict[str, tuple[int, int]] = {}
    for record in source_buildings:
        identifier = str(record["id"])
        outer = normalize_ring(record["footprint"]["outer"], ccw=True)
        holes = [normalize_ring(hole, ccw=False) for hole in record["footprint"].get("holes", [])]
        centroid = polygon_centroid(outer, holes)
        centroids[identifier] = centroid
        cells[identifier] = reduction_cell(centroid)

    counts = Counter(cells.values())
    require(len(counts) == REDUCTION_GRID_SIZE**2, "source city does not populate every removal grid cell")
    raw = {cell: REMOVED_BUILDING_COUNT * count / SOURCE_BUILDING_COUNT for cell, count in counts.items()}
    quotas = {cell: int(math.floor(value)) for cell, value in raw.items()}
    remaining = REMOVED_BUILDING_COUNT - sum(quotas.values())
    largest_remainders = sorted(counts, key=lambda cell: (-(raw[cell] - quotas[cell]), cell))
    for cell in largest_remainders[:remaining]:
        quotas[cell] += 1

    protected = set(REDUCTION_PROTECTED_IDS)
    selected: set[str] = set()
    for cell in sorted(counts):
        eligible = [
            identifier
            for identifier, identifier_cell in cells.items()
            if identifier_cell == cell and identifier not in protected
        ]
        ranked = sorted(
            eligible,
            key=lambda identifier: hashlib.sha256(
                f"{REDUCTION_RANDOM_SEED}:{identifier}".encode("utf-8")
            ).digest(),
        )
        require(len(ranked) >= quotas[cell], f"removal cell {cell} cannot meet its protected quota")
        selected.update(ranked[:quotas[cell]])

    # Serialize in canonical source order, not set or hash order.
    result = tuple(str(record["id"]) for record in source_buildings if str(record["id"]) in selected)
    require(len(result) == REMOVED_BUILDING_COUNT, "spatial-random selector did not return 69 IDs")
    return result


def removal_grid_audit(source_buildings: Sequence[dict], removed_ids: Sequence[str]) -> dict:
    """Return compact, machine-checkable spatial distribution evidence."""
    removed = set(removed_ids)
    source_counts: Counter[tuple[int, int]] = Counter()
    removed_counts: Counter[tuple[int, int]] = Counter()
    removed_centroids: list[Point] = []
    for record in source_buildings:
        outer = normalize_ring(record["footprint"]["outer"], ccw=True)
        holes = [normalize_ring(hole, ccw=False) for hole in record["footprint"].get("holes", [])]
        centroid = polygon_centroid(outer, holes)
        cell = reduction_cell(centroid)
        source_counts[cell] += 1
        if str(record["id"]) in removed:
            removed_counts[cell] += 1
            removed_centroids.append(centroid)
    quadrants = Counter(
        ("north" if y >= 0.0 else "south") + "_" + ("east" if x >= 0.0 else "west")
        for x, y in removed_centroids
    )
    return {
        "seed": REDUCTION_RANDOM_SEED,
        "grid_shape": [REDUCTION_GRID_SIZE, REDUCTION_GRID_SIZE],
        "source_bounds_m": [REDUCTION_SOURCE_MIN_M, REDUCTION_SOURCE_MAX_M],
        "populated_cells": len(source_counts),
        "cells_with_removals": sum(count > 0 for count in removed_counts.values()),
        "source_counts_by_row": [
            [source_counts[(row, column)] for column in range(REDUCTION_GRID_SIZE)]
            for row in range(REDUCTION_GRID_SIZE)
        ],
        "removed_counts_by_row": [
            [removed_counts[(row, column)] for column in range(REDUCTION_GRID_SIZE)]
            for row in range(REDUCTION_GRID_SIZE)
        ],
        "quadrant_counts": dict(sorted(quadrants.items())),
        "removed_ids_sha256": hashlib.sha256(
            "".join(f"{identifier}\n" for identifier in removed_ids).encode("utf-8")
        ).hexdigest(),
    }


def select_active_buildings(source_buildings: Sequence[dict]) -> tuple[list[dict], tuple[str, ...]]:
    """Apply the audited 69-building spatial-random reduction."""
    require(
        len(source_buildings) == SOURCE_BUILDING_COUNT,
        f"expected {SOURCE_BUILDING_COUNT} source buildings, got {len(source_buildings)}",
    )
    identifiers = {str(record["id"]) for record in source_buildings}
    generated_ids = deterministic_removal_ids(source_buildings)
    require(generated_ids == REDUCTION_REMOVED_IDS,
            "spatial-random removal result differs from the checked-in audit contract")
    require(len(REDUCTION_REMOVED_IDS) == REMOVED_BUILDING_COUNT,
            "checked-in removal set does not contain 69 IDs")
    require(len(set(REDUCTION_REMOVED_IDS)) == REMOVED_BUILDING_COUNT,
            "checked-in removal IDs are not unique")
    require(set(REDUCTION_REMOVED_IDS) <= identifiers, "building reduction IDs are missing")
    require(set(REDUCTION_PROTECTED_IDS) <= identifiers, "reduction protection IDs are missing")
    removed_set = set(REDUCTION_REMOVED_IDS)
    retained = [record for record in source_buildings if str(record["id"]) not in removed_set]
    removed = tuple(str(record["id"]) for record in source_buildings if str(record["id"]) in removed_set)
    require(len(retained) == ACTIVE_BUILDING_COUNT, "active building count differs from 205")
    require(len(removed) == REMOVED_BUILDING_COUNT, "removed building count differs from 69")
    require(not (set(REDUCTION_PROTECTED_IDS) & removed_set), "protected building was removed")
    audit = removal_grid_audit(source_buildings, removed)
    require(audit["cells_with_removals"] == REDUCTION_GRID_SIZE**2,
            "spatial-random removal does not cover every map region")
    require(audit["removed_ids_sha256"] == REDUCTION_IDS_SHA256,
            "spatial-random removal ID digest changed")
    return retained, removed


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
    direct = (TRAILER_SPAWN_XY, TRAILER_DESTINATION_XY)
    direct_clearance = polyline_clearance(direct, buildings)
    if direct_clearance >= required:
        return direct, direct_clearance

    candidates: list[tuple[float, tuple[Point, ...], float]] = []
    for detour_y_integer in range(-300, 301, 5):
        detour_y = float(detour_y_integer)
        route = (
            TRAILER_SPAWN_XY,
            (TRAILER_SPAWN_XY[0], detour_y),
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
            ("trailer_spawn", TRAILER_SPAWN_XY),
            ("endpoint", TRAILER_DESTINATION_XY),
        )
    }
    trailer_route, route_clearance = choose_trailer_route(buildings)

    reasons: list[str] = []
    required_neighbor_gap = 2.0 * R_HARD_M
    required_point_clearance = R_PREFERRED_M + TAKEOFF_POSITION_ERROR_M
    if overlap_count:
        reasons.append(f"{overlap_count} transformed neighbor pairs overlap")
    if min(neighbor_gaps) < required_neighbor_gap:
        reasons.append(
            f"minimum neighbor gap {min(neighbor_gaps):.3f}m < hard width {required_neighbor_gap:.3f}m"
        )
    for name, clearance in point_clearances.items():
        if clearance < required_point_clearance:
            reasons.append(f"{name} clearance {clearance:.3f}m < {required_point_clearance:.3f}m")

    return CandidateEvaluation(
        scale=buildings[0].footprint_scale,
        overlap_count=overlap_count,
        minimum_neighbor_gap_m=min(neighbor_gaps),
        spawn_clearance_m=point_clearances["spawn"],
        goal_clearance_m=point_clearances["goal"],
        trailer_spawn_clearance_m=point_clearances["trailer_spawn"],
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
    selected = DEFAULT_FOOTPRINT_SCALE
    reason = (
        "restore origin/jo building XY: 2.5x centroids and 0.9x footprints; "
        "create passages only through deterministic building removal"
    )
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


def determine_bounds(buildings: Sequence[BuildingGeometry], margin: float) -> tuple[float, float, float, float]:
    all_points = [point for building in buildings for point in building.transformed.outer]
    all_points.extend([SPAWN_XY, MISSION_GOAL_XY, TRAILER_SPAWN_XY, TRAILER_DESTINATION_XY])
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


def write_world(bounds: tuple[float, float, float, float]) -> None:
    map_size = bounds[2] - bounds[0]
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
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine><scene>scene</scene>
        <ambient_light>0.45 0.48 0.55</ambient_light>
        <background_color>0.55 0.62 0.70</background_color>
        <!-- Spawn-close fallback view. run_px4_map.sh reapplies this pose once
             and explicitly leaves CameraTracking disabled by default. -->
        <camera_pose>{fmt(SPAWN_XY[0] - 4.0)} {fmt(SPAWN_XY[1])} 3 0 0.45 0</camera_pose>
        <camera_clip><near>0.1</near><far>50000</far></camera_clip>
      </plugin>
      <!-- MinimalScene only owns the render texture.  GzSceneManager is what
           actually mirrors the world entities into that texture; omitting it
           produces an otherwise healthy but completely black 3D panel. -->
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="CameraTracking" name="Camera Tracking">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="EntityTree" name="Entity tree">
        <gz-gui>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control"><play_pause>true</play_pause><step>true</step><start_paused>true</start_paused></plugin>
      <plugin filename="WorldStats" name="World stats"><sim_time>true</sim_time><real_time_factor>true</real_time_factor></plugin>
    </gui>
    <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><latitude_deg>37.32977431415926</latitude_deg><longitude_deg>-121.99860517699835</longitude_deg><elevation>50.6</elevation></spherical_coordinates>

    <!-- Exactly flat z=0 datum.  The 1 mm heightmap is visual-only; Bullet/DART physics uses the box. -->
    <model name="applepark_uav_ground"><static>true</static><link name="ground">
      <collision name="collision"><pose>0 0 -0.05 0 0 0</pose><geometry><box><size>{fmt(map_size)} {fmt(map_size)} 0.1</size></box></geometry></collision>
      <visual name="ground_visual"><cast_shadows>false</cast_shadows><geometry><heightmap><use_terrain_paging>false</use_terrain_paging><texture><diffuse>mesh/road_surface_city_uav.png</diffuse><normal>mesh/normal_map_city_uav.png</normal><size>{fmt(map_size)}</size></texture><uri>mesh/height_map_city_uav.png</uri><size>{fmt(map_size)} {fmt(map_size)} 0.001</size><pos>0 0 -0.001</pos><sampling>1</sampling></heightmap></geometry></visual>
    </link></model>

    <!-- The same closed, triangulated DAE is used once for rendering and once
         for DART collision.  This keeps all 205 YAML prisms bit-aligned,
         preserves the courtyard opening and replaces hundreds of per-building
         collision entities with one static geometry. -->
    <model name="applepark_uav_buildings"><static>true</static><link name="buildings">
      <visual name="buildings_visual"><geometry><mesh><uri>mesh/buildings_uav.dae</uri><scale>1 1 1</scale></mesh></geometry></visual>
      <collision name="buildings_exact_shared_dae"><geometry><mesh><uri>mesh/buildings_uav.dae</uri><scale>1 1 1</scale></mesh></geometry></collision>
    </link></model>

    <!-- No visual spawn pad: PX4 model contact is referenced to the z=0 datum. -->
    <frame name="drone_spawn"><pose>{fmt(SPAWN_XY[0])} {fmt(SPAWN_XY[1])} 0 0 0 0</pose></frame>
    <frame name="mission_goal"><pose>{fmt(MISSION_GOAL_XY[0])} {fmt(MISSION_GOAL_XY[1])} 0 0 0 0</pose></frame>
    <frame name="trailer_spawn"><pose>{fmt(TRAILER_SPAWN_XY[0])} {fmt(TRAILER_SPAWN_XY[1])} 0 0 0 0</pose></frame>
    <frame name="trailer_destination"><pose>{fmt(TRAILER_DESTINATION_XY[0])} {fmt(TRAILER_DESTINATION_XY[1])} 0 0 0 0</pose></frame>
    <include><uri>model://trailer_aruco</uri><name>trailer</name><pose>{fmt(TRAILER_SPAWN_XY[0])} {fmt(TRAILER_SPAWN_XY[1])} 0 0 0 0</pose></include>
    <!-- PX4 SITL spawns the real sensor-equipped vehicle dynamically. -->
  </world>
</sdf>
'''
    OUTPUT_WORLD.write_text(world, encoding="utf-8")
    OUTPUT_MODEL_CONFIG.write_text(
        """<?xml version="1.0"?>
<model><name>Apple Park UAV city</name><version>1.0</version><sdf version="1.9">applepark_uav.world</sdf><author><name>PX4-ROS2 contributors</name></author><description>205-building UAV city with rolled-back jo 2.5x / 0.9x XY, 10-20 m skyline and one exact shared DART collision mesh.</description></model>
""",
        encoding="utf-8",
    )
    OUTPUT_ATTRIBUTION.write_text(SOURCE_ATTRIBUTION.read_text(encoding="utf-8"), encoding="utf-8")


def rounded_point(point: Point) -> list[float]:
    return [round(point[0], 10), round(point[1], 10)]


def write_yaml(
    source: dict,
    buildings: Sequence[BuildingGeometry],
    removed_building_ids: Sequence[str],
    bounds: tuple[float, float, float, float],
    spacing_scale: float,
    anchor: Point,
    scale_reason: str,
    mesh_vertices: int,
    mesh_triangles: int,
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
        "building_reduction": {
            "source_count": SOURCE_BUILDING_COUNT,
            "removed_count": len(removed_building_ids),
            "retained_count": len(buildings),
            "removed_fraction": len(removed_building_ids) / SOURCE_BUILDING_COUNT,
            "method": REDUCTION_METHOD,
            "protected_ids": list(REDUCTION_PROTECTED_IDS),
            "removed_ids": list(removed_building_ids),
            "spatial_random_audit": removal_grid_audit(
                source["obstacles"]["buildings"], removed_building_ids
            ),
        },
        "z_scale": 1.0,
        "fixed_mission_coordinates_enu_m": {
            "drone_spawn": list(SPAWN_XY),
            "global_goal": list(MISSION_GOAL_XY),
            "trailer_spawn": list(TRAILER_SPAWN_XY),
            "trailer_destination": list(TRAILER_DESTINATION_XY),
        },
        "mesh": {"vertices": mesh_vertices, "triangles": mesh_triangles},
        "collision_geometry": {
            "type": COLLISION_GEOMETRY_TYPE,
            "count": 1,
            "source": "same closed triangulated buildings_uav.dae used by the visual",
            "maximum_outward_error_m": 0.0,
            "maximum_undercoverage_m": 0.0,
            "courtyard_holes_preserved": True,
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
                "x": TRAILER_SPAWN_XY[0],
                "y": TRAILER_SPAWN_XY[1],
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "motion": "stationary_spawn_only",
            "waypoints_enu_m": [list(TRAILER_SPAWN_XY)],
            "body_footprint_m": [5.5, 3.0],
            "deck_height_m": 1.25,
            "command_topic": "/model/trailer/cmd_vel",
            "pose_topic": "/world/applepark_city_uav/dynamic_pose/info",
        }
    )
    output["trailer"].pop("cruise_speed_m_s", None)
    output["trailer"].pop("waypoint_tolerance_m", None)
    output["trailer"].pop("route_surface", None)
    output["trailer"]["surface"] = "exact_z0_flat_city_datum"
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
        minimum_x, minimum_y, maximum_x, maximum_y = building.transformed.bounds
        record["aabb_xyz_m"] = {
            "min": [round(minimum_x, 10), round(minimum_y, 10), building.foundation_z],
            "max": [round(maximum_x, 10), round(maximum_y, 10), building.roof_z],
            "center_enu_m": [
                round(0.5 * (minimum_x + maximum_x), 10),
                round(0.5 * (minimum_y + maximum_y), 10),
                0.5 * (building.foundation_z + building.roof_z),
            ],
            "size_xyz_m": [
                round(maximum_x - minimum_x, 10),
                round(maximum_y - minimum_y, 10),
                building.roof_z - building.foundation_z,
            ],
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
    shortest = min(output_buildings, key=lambda item: float(item["height_above_ground_m"]))
    tallest = max(output_buildings, key=lambda item: float(item["height_above_ground_m"]))
    minimum_gap = min(
        polygon_distance(first.transformed, second.transformed)
        for first, second in itertools.combinations(buildings, 2)
    )
    output["obstacles"]["summary"] = {
        "source_building_count": SOURCE_BUILDING_COUNT,
        "building_count": len(output_buildings),
        "removed_building_count": len(removed_building_ids),
        "active_height_profile": ACTIVE_HEIGHT_PROFILE,
        "foundation_z_range_m": [
            min(building.foundation_z for building in buildings),
            max(building.foundation_z for building in buildings),
        ],
        "roof_z_range_m": [
            min(building.roof_z for building in buildings),
            max(building.roof_z for building in buildings),
        ],
        "shortest_building": {
            "id": shortest["id"],
            "height_above_ground_m": shortest["height_above_ground_m"],
        },
        "tallest_building": {
            "id": tallest["id"],
            "height_above_ground_m": tallest["height_above_ground_m"],
        },
        "minimum_building_gap_m": minimum_gap,
        "minimum_required_drone_passage_m": 2.0,
        "all_building_gaps_meet_requirement": minimum_gap >= 2.0,
    }
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
    source_buildings: Sequence[dict],
    buildings: Sequence[BuildingGeometry],
    removed_building_ids: Sequence[str],
    neighbor_pairs: Sequence[tuple[int, int, float]],
    evaluations: Sequence[CandidateEvaluation],
    bounds: tuple[float, float, float, float],
    spacing_scale: float,
    anchor: Point,
    scale_reason: str,
    mesh_vertices: int,
    mesh_triangles: int,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "building_transform.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
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
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["building_a", "building_b", "original_edge_gap_m", "new_edge_gap_m", "gap_ratio"])
        for first, second, original_gap, new_gap, ratio in gap_records:
            writer.writerow(
                [first, second, fmt(original_gap), fmt(new_gap), "" if ratio is None else fmt(ratio)]
            )

    required_neighbor_gap = 2.0 * R_HARD_M
    with (REPORT_DIR / "invalid_or_tight_corridors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
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
            ("mission_point", "trailer_spawn", "buildings", selected_evaluation.trailer_spawn_clearance_m, R_PREFERRED_M + TAKEOFF_POSITION_ERROR_M),
        )
        for kind, first, second, measured, required in mission_checks:
            if measured < required:
                writer.writerow([kind, first, second, fmt(measured), fmt(required), "FAIL"])
                tight_count += 1

    with (REPORT_DIR / "scale_candidate_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "scale",
                "feasible",
                "overlap_count",
                "minimum_neighbor_gap_m",
                "spawn_clearance_m",
                "goal_clearance_m",
                "trailer_spawn_clearance_m",
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
                    fmt(evaluation.trailer_spawn_clearance_m),
                    fmt(evaluation.endpoint_clearance_m),
                    fmt(evaluation.trailer_route_clearance_m),
                    "; ".join(evaluation.reasons),
                ]
            )

    # The collision and visual are the same DAE.  Keep a compact per-building
    # audit instead of emitting hundreds of synthetic proxy records.
    stale_proxy_report = REPORT_DIR / "collision_proxy_alignment.csv"
    stale_proxy_report.unlink(missing_ok=True)
    with (REPORT_DIR / "collision_mesh_alignment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "building_id",
                "foundation_z_m",
                "roof_z_m",
                "visual_collision_shared_dae",
                "boundary_alignment_error_m",
                "undercoverage_m",
                "outward_error_m",
            ]
        )
        for building in buildings:
            writer.writerow(
                [
                    building.identifier,
                    fmt(building.foundation_z),
                    fmt(building.roof_z),
                    "true",
                    "0",
                    "0",
                    "0",
                ]
            )

    selected = next(item for item in evaluations if item.scale == buildings[0].footprint_scale)
    collision_point_clearances = {
        name: min(point_polygon_distance(point, building.transformed) for building in buildings)
        for name, point in (
            ("spawn", SPAWN_XY),
            ("goal", MISSION_GOAL_XY),
            ("trailer_spawn", TRAILER_SPAWN_XY),
            ("endpoint", TRAILER_DESTINATION_XY),
        )
    }
    summary = {
        "source_building_count": SOURCE_BUILDING_COUNT,
        "generated_building_count": len(buildings),
        "removed_building_count": len(removed_building_ids),
        "removed_building_ids": list(removed_building_ids),
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
            "global_goal": selected.goal_clearance_m,
            "trailer_spawn": selected.trailer_spawn_clearance_m,
        },
        "spatial_random_reduction": removal_grid_audit(source_buildings, removed_building_ids),
        "trailer_motion": "stationary_spawn_only",
        "fixed_coordinates_enu_m": {
            "drone_spawn": list(SPAWN_XY),
            "global_goal": list(MISSION_GOAL_XY),
            "trailer_spawn": list(TRAILER_SPAWN_XY),
            "trailer_destination": list(TRAILER_DESTINATION_XY),
        },
        "z_preservation": {"maximum_absolute_error_m": 0.0, "tolerance_m": 1.0e-9},
        "visual_collision_yaml_alignment": {"maximum_absolute_xy_error_m": 0.0, "tolerance_m": 0.01},
        "collision_geometry": {
            "representation": COLLISION_GEOMETRY_TYPE,
            "count": 1,
            "source": "same closed triangulated DAE as visual",
            "source_vertex_alignment_error_m": 0.0,
            "maximum_undercoverage_m": 0.0,
            "maximum_outward_error_m": 0.0,
            "certified_minimum_neighbor_gap_m": selected.minimum_neighbor_gap_m,
            "mission_clearance_m": {
                "drone_spawn": collision_point_clearances["spawn"],
                "global_goal": collision_point_clearances["goal"],
                "trailer_spawn": collision_point_clearances["trailer_spawn"],
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
    alignment = f"""# UAV city visual/collision/YAML alignment

- Source of truth: `gazebo/maps/city_coordinates.yaml` ({SOURCE_BUILDING_COUNT} buildings).
- Active reduction: {len(buildings)} retained / {len(removed_building_ids)} removed.
- Derived coordinate geometry: `gazebo/maps/city_coordinates_uav.yaml`.
- Gazebo visual URI: `mesh/buildings_uav.dae` at scale `1 1 1`.
- Gazebo collision: one static shared `mesh/buildings_uav.dae` triangle mesh.
- Mesh SHA256: `{sha256_file(OUTPUT_DAE)}`.
- Vertex/triangle count: {mesh_vertices} / {mesh_triangles}.
- Maximum YAML-to-visual-mesh XY boundary error: `0.0 m` (limit `0.01 m`).
- Collision source-vertex alignment error: `0.0 m` (limit `0.01 m`).
- Collision maximum undercoverage/outward error: `0.0 m`.
- Maximum foundation/roof Z error from source: `0.0 m` (limit `1e-9 m`).
- Selected footprint scale: `{fmt(buildings[0].footprint_scale)}` — {scale_reason}.

The closed mesh is regenerated directly from every transformed outer/hole
ring. No world-level building scale or pose is used. Gazebo visual and DART
collision both reference that same single file at scale `1 1 1`, so the
courtyard remains open and no per-building collision entities are needed.
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
    active_source_buildings, removed_building_ids = select_active_buildings(source_buildings)
    anchor = (args.anchor_x, args.anchor_y)

    OUTPUT_MESH_DIR.mkdir(parents=True, exist_ok=True)
    buildings, neighbor_pairs, evaluations, scale_reason = choose_scale(
        active_source_buildings, args.spacing_scale, anchor, args.neighbor_radius
    )
    selected = next(item for item in evaluations if item.scale == buildings[0].footprint_scale)
    require(selected.feasible, "selected layout is not feasible")
    bounds = determine_bounds(buildings, args.map_margin)
    mesh_vertices, mesh_triangles = write_dae(buildings, OUTPUT_DAE)
    require(selected.minimum_neighbor_gap_m >= 2.0, "shared DAE collision closes a required 2m corridor")
    write_textures(bounds, args.spacing_scale, anchor)
    write_world(bounds)
    write_yaml(
        source,
        buildings,
        removed_building_ids,
        bounds,
        args.spacing_scale,
        anchor,
        scale_reason,
        mesh_vertices,
        mesh_triangles,
    )
    summary = write_reports(
        source_buildings,
        buildings,
        removed_building_ids,
        neighbor_pairs,
        evaluations,
        bounds,
        args.spacing_scale,
        anchor,
        scale_reason,
        mesh_vertices,
        mesh_triangles,
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
