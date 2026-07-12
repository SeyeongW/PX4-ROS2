#!/usr/bin/env python3
"""Export exact Gazebo obstacle coordinates and deterministic 2-D references.

The YAML and figures are generated from the same collision assets used by
Gazebo.  Coordinates are world ENU; PX4's spawn-relative NED conversion is
recorded explicitly so path-planning code does not silently swap axes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from PIL import Image
import yaml

import align_city_building_foundations as city_dae


GAZEBO = Path(__file__).resolve().parents[1]
REPO = GAZEBO.parent
MAPS = GAZEBO / "maps"
OUTPUTS = GAZEBO / "validation" / "path_planning"
CITY_DAE = GAZEBO / "worlds" / "applepark_city" / "mesh" / "buildings.dae"
CITY_HEIGHT = GAZEBO / "worlds" / "applepark_city" / "mesh" / "height_map_city_500m.png"
CITY_ROADS = GAZEBO / "worlds" / "applepark_city" / "mesh" / "road_surface_city_500m.png"
CITY_COLLISION = GAZEBO / "worlds" / "applepark_city" / "mesh" / "city_terrain_collision.obj"
CITY_HEIGHTS = GAZEBO / "validation" / "city" / "building_height_scaling.csv"
CITY_FOUNDATIONS = GAZEBO / "validation" / "city" / "building_foundation_alignment.csv"
MOUNTAIN_HEIGHT = GAZEBO / "models" / "ugv_mou_terrain" / "materials" / "textures" / "mountain_height_300.png"
MOUNTAIN_COLLISION = GAZEBO / "models" / "ugv_mou_terrain" / "meshes" / "ugv_mou_terrain_collision.obj"
FOREST_SDF = GAZEBO / "models" / "ugv_mou_forest_obstacles" / "model.sdf"
TREE_SOURCE = GAZEBO / "models" / "ugv_mou_forest_obstacles" / "source" / "tree_layout.source.xml"

CITY_YAML = MAPS / "city_coordinates.yaml"
MOUNTAIN_YAML = MAPS / "mountain_coordinates.yaml"
CITY_FIGURE = OUTPUTS / "city_2d_path_planning_reference.png"
MOUNTAIN_FIGURE = OUTPUTS / "mountain_2d_path_planning_reference.png"
HOME_CITY_FIGURE = Path.home() / CITY_FIGURE.name
HOME_MOUNTAIN_FIGURE = Path.home() / MOUNTAIN_FIGURE.name

EXPECTED_BUILDINGS = 274
EXPECTED_CITY_RINGS = 275
EXPECTED_CITY_BOUNDARY_VERTICES = 2585
EXPECTED_TREES = 288
GRID_RESOLUTION_M = 0.25
SAFETY_INFLATION_M = 2.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def signed_area(ring: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
    )


def point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def canonical_ring(
    ring: list[tuple[float, float]], *, counter_clockwise: bool
) -> list[tuple[float, float]]:
    if (signed_area(ring) > 0.0) != counter_clockwise:
        ring = list(reversed(ring))
    start = min(range(len(ring)), key=lambda index: ring[index])
    return ring[start:] + ring[:start]


def component_boundary_rings(
    data: city_dae.DaeData, faces: np.ndarray
) -> list[list[tuple[float, float]]]:
    points = data.positions[data.position_indices[faces]]
    minimum_z = float(points[:, :, 2].min())
    bottom = points[np.all(np.isclose(points[:, :, 2], minimum_z, atol=1e-8), axis=1)]
    require(len(bottom) > 0, "building component has no bottom triangles")
    edge_counts: dict[
        tuple[tuple[float, float], tuple[float, float]], int
    ] = {}
    for triangle in bottom:
        vertices = [tuple(np.round(vertex[:2], 6)) for vertex in triangle]
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            edge = (first, second) if first <= second else (second, first)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = {edge for edge, count in edge_counts.items() if count == 1}
    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for first, second in boundary:
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    require(
        all(len(neighbours) == 2 for neighbours in adjacency.values()),
        "building footprint boundary is not a set of closed degree-2 rings",
    )

    remaining = set(boundary)
    rings: list[list[tuple[float, float]]] = []
    while remaining:
        seed = min(remaining)
        start = seed[0]
        previous: tuple[float, float] | None = None
        current = start
        ring: list[tuple[float, float]] = []
        while True:
            ring.append(current)
            candidates = sorted(adjacency[current])
            following = candidates[0] if candidates[0] != previous else candidates[1]
            edge = (current, following) if current <= following else (following, current)
            require(edge in remaining, "building footprint ring reused an edge")
            remaining.remove(edge)
            previous, current = current, following
            if current == start:
                break
        require(len(ring) >= 3, "building footprint ring has fewer than three vertices")
        rings.append(ring)
    return rings


def city_buildings() -> list[dict[str, object]]:
    data = city_dae.DaeData(CITY_DAE)
    components = city_dae.find_components(data)
    heights = read_csv(CITY_HEIGHTS)
    foundations = read_csv(CITY_FOUNDATIONS)
    require(len(components) == len(heights) == len(foundations) == EXPECTED_BUILDINGS,
            "city building source counts differ")
    buildings: list[dict[str, object]] = []
    ring_count = boundary_vertices = 0
    for ordinal, (faces, height, foundation) in enumerate(
        zip(components, heights, foundations), start=1
    ):
        require(int(height["output_component_ordinal"]) == ordinal,
                "city height audit order changed")
        require(int(foundation["output_component_ordinal"]) == ordinal,
                "city foundation audit order changed")
        rings = component_boundary_rings(data, faces)
        outer = max(rings, key=lambda ring: abs(signed_area(ring)))
        holes = [ring for ring in rings if ring is not outer]
        require(all(point_in_ring(hole[0], outer) for hole in holes),
                "building footprint has multiple exterior rings")
        outer = canonical_ring(outer, counter_clockwise=True)
        holes = [canonical_ring(ring, counter_clockwise=False) for ring in holes]
        holes.sort(key=lambda ring: ring[0])
        all_points = outer + [point for hole in holes for point in hole]
        ring_count += 1 + len(holes)
        boundary_vertices += len(all_points)
        source_id = int(height["post_height_component_ordinal"])
        require(source_id == int(foundation["post_height_component_ordinal"]),
                "city component identity differs between audits")
        buildings.append(
            {
                "id": f"building_{ordinal:03d}",
                "source_component_id": source_id,
                "shape": "polygon_prism",
                "footprint": {
                    "outer": [[float(x), float(y)] for x, y in outer],
                    "holes": [
                        [[float(x), float(y)] for x, y in hole] for hole in holes
                    ],
                },
                "aabb_xy_m": {
                    "min": [float(min(x for x, _ in all_points)), float(min(y for _, y in all_points))],
                    "max": [float(max(x for x, _ in all_points)), float(max(y for _, y in all_points))],
                },
                "foundation_z_m": float(height["foundation_base_z"]),
                "ground_reference_z_m": float(height["reference_base_z"]),
                "roof_z_m": float(height["new_roof_z"]),
                "height_above_ground_m": float(height["new_above_ground_height_m"]),
            }
        )
    require(ring_count == EXPECTED_CITY_RINGS, f"expected 275 city rings, got {ring_count}")
    require(boundary_vertices == EXPECTED_CITY_BOUNDARY_VERTICES,
            f"expected 2585 city boundary vertices, got {boundary_vertices}")
    return buildings


def mountain_trees() -> list[dict[str, object]]:
    source_types = {
        include.findtext("name", ""): include.findtext("uri", "").removeprefix("model://")
        for include in ET.parse(TREE_SOURCE).getroot().findall("./model/include")
        if include.findtext("uri", "") in {"model://pine_tree", "model://oak_tree"}
    }
    link = ET.parse(FOREST_SDF).getroot().find("./model/link")
    require(link is not None, "forest collision link is missing")
    require(not link.findall("./collision[box]"), "mountain maze box collisions remain")
    require(not any("maze_" in node.attrib.get("name", "") for node in link),
            "mountain maze runtime entity remains")
    trees: list[dict[str, object]] = []
    for collision in link.findall("collision"):
        name = collision.attrib.get("name", "")
        if not name.endswith("_trunk_collision"):
            continue
        tree_id = name.removesuffix("_trunk_collision")
        pose = [float(value) for value in collision.findtext("pose", "").split()]
        radius = float(collision.findtext("./geometry/cylinder/radius", "nan"))
        length = float(collision.findtext("./geometry/cylinder/length", "nan"))
        require(len(pose) == 6 and math.isfinite(radius) and math.isfinite(length),
                f"invalid tree collision: {name}")
        tree_type = source_types.get(tree_id)
        require(tree_type in {"pine_tree", "oak_tree"}, f"unknown tree source: {tree_id}")
        trees.append(
            {
                "id": tree_id,
                "type": tree_type,
                "shape": "cylinder",
                "center_enu_m": pose[:3],
                "radius_m": radius,
                "length_m": length,
                "base_z_m": pose[2] - length / 2.0,
                "top_z_m": pose[2] + length / 2.0,
                "yaw_rad": pose[5],
            }
        )
    trees.sort(key=lambda tree: str(tree["id"]))
    require(len(trees) == EXPECTED_TREES, f"expected 288 trees, got {len(trees)}")
    require(sum(tree["type"] == "pine_tree" for tree in trees) == 216,
            "pine tree count changed")
    require(sum(tree["type"] == "oak_tree" for tree in trees) == 72,
            "oak tree count changed")
    return trees


def frame_contract(spawn: list[float]) -> dict[str, object]:
    return {
        "gazebo_world": {
            "convention": "ENU",
            "x_axis": "east",
            "y_axis": "north",
            "z_axis": "up",
            "yaw": "counter_clockwise_from_east_rad",
            "units": "m",
        },
        "px4_local": {
            "convention": "NED",
            "origin_enu_m": spawn[:3],
            "conversion_from_gazebo_enu": {
                "x_north_m": "y_enu - origin_y",
                "y_east_m": "x_enu - origin_x",
                "z_down_m": "-(z_enu - origin_z)",
                "yaw_ned_rad": "normalize(pi/2 - yaw_enu)",
            },
        },
    }


def write_yaml(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def city_document(buildings: list[dict[str, object]]) -> dict[str, object]:
    spawn = [-120.0, 115.0, -3.019558902, 0.0, 0.0, 0.0]
    return {
        "schema_version": 1,
        "map": {
            "name": "city",
            "gazebo_world_name": "applepark_city",
            "world_file": "gazebo/worlds/applepark_city/applepark.world",
            "bounds_enu_m": {"x": [-250.0, 250.0], "y": [-250.0, 250.0]},
        },
        "frames": frame_contract(spawn),
        "px4_vehicle": {
            "airframe_autostart_id": 4014,
            "simulation_model": "gz_x500_mono_cam_down",
            "runtime_entity_name": "x500_mono_cam_down_0",
        },
        "terrain": {
            "type": "heightmap",
            "image": "gazebo/worlds/applepark_city/mesh/height_map_city_500m.png",
            "collision_mesh": "gazebo/worlds/applepark_city/mesh/city_terrain_collision.obj",
            "rows": 257,
            "columns": 257,
            "sample_spacing_m": 1.953125,
            "row_0_y_m": 250.0,
            "column_0_x_m": -250.0,
            "row_direction": "decreasing_y",
            "height_formula_m": "-15.0 + pixel/255*26.6",
        },
        "spawn": {
            "name": "city_drone_spawn",
            "gazebo_spawn_pose_enu": dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), spawn)),
            "pad": {"shape": "cylinder", "center_enu_m": [-120.0, 115.0, -3.269558902],
                    "radius_m": 4.0, "length_m": 0.5, "top_z_m": -3.019558902},
        },
        "planner_defaults": {"obstacle_inflation_m": SAFETY_INFLATION_M},
        "obstacles": {"buildings": buildings},
        "source_sha256": {
            "building_collision_dae": sha256(CITY_DAE),
            "terrain_heightmap": sha256(CITY_HEIGHT),
            "terrain_collision_mesh": sha256(CITY_COLLISION),
            "building_height_audit": sha256(CITY_HEIGHTS),
            "building_foundation_audit": sha256(CITY_FOUNDATIONS),
        },
    }


def mountain_document(trees: list[dict[str, object]]) -> dict[str, object]:
    spawn = [-80.0, -80.0, 0.16, 0.0, 0.0, 0.785398]
    return {
        "schema_version": 1,
        "map": {
            "name": "mountain",
            "gazebo_world_name": "ugv_drone_mountain_map",
            "world_file": "gazebo/worlds/ugv_drone_map.world",
            "bounds_enu_m": {"x": [-150.0, 150.0], "y": [-150.0, 150.0]},
        },
        "frames": frame_contract(spawn),
        "px4_vehicle": {
            "airframe_autostart_id": 4014,
            "simulation_model": "gz_x500_mono_cam_down",
            "runtime_entity_name": "x500_mono_cam_down_0",
        },
        "terrain": {
            "type": "heightmap_and_triangle_mesh",
            "image": "gazebo/models/ugv_mou_terrain/materials/textures/mountain_height_300.png",
            "collision_mesh": "gazebo/models/ugv_mou_terrain/meshes/ugv_mou_terrain_collision.obj",
            "rows": 257,
            "columns": 257,
            "sample_spacing_m": 1.171875,
            "row_0_y_m": 150.0,
            "column_0_x_m": -150.0,
            "row_direction": "decreasing_y",
            "height_formula_m": "pixel/255*40.0",
            "height_range_m": [0.0, 40.0],
        },
        "spawn": {
            "name": "mountain_drone_spawn",
            "gazebo_spawn_pose_enu": dict(zip(("x", "y", "z", "roll", "pitch", "yaw"), spawn)),
            "pad": {"shape": "box", "center_enu_m": [-80.0, -80.0, 0.08],
                    "size_m": [12.0, 12.0, 0.16], "top_z_m": 0.16},
        },
        "planner_defaults": {
            "obstacle_inflation_m": SAFETY_INFLATION_M,
            "tree_collision_scope": "trunk_only; canopy meshes are visual-only",
        },
        "obstacles": {"maze_walls": [], "trees": trees},
        "source_sha256": {
            "terrain_heightmap": sha256(MOUNTAIN_HEIGHT),
            "terrain_collision_mesh": sha256(MOUNTAIN_COLLISION),
            "forest_collision_sdf": sha256(FOREST_SDF),
            "tree_source_layout": sha256(TREE_SOURCE),
        },
    }


def pixel_ring(ring: list[list[float]], half_size: float, resolution: float) -> np.ndarray:
    points = np.asarray(ring, dtype=np.float64)
    cols = np.rint((points[:, 0] + half_size) / resolution)
    rows = np.rint((half_size - points[:, 1]) / resolution)
    limit = int(round(2.0 * half_size / resolution)) - 1
    return np.clip(np.column_stack((cols, rows)), 0, limit).astype(np.int32)


def city_occupancy(buildings: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    cells = int(round(500.0 / GRID_RESOLUTION_M))
    raw = np.zeros((cells, cells), dtype=np.uint8)
    for building in buildings:
        footprint = building["footprint"]
        assert isinstance(footprint, dict)
        cv2.fillPoly(raw, [pixel_ring(footprint["outer"], 250.0, GRID_RESOLUTION_M)], 255)
        for hole in footprint["holes"]:
            cv2.fillPoly(raw, [pixel_ring(hole, 250.0, GRID_RESOLUTION_M)], 0)
    radius = int(math.ceil(SAFETY_INFLATION_M / GRID_RESOLUTION_M))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return raw, cv2.dilate(raw, kernel)


def mountain_occupancy(trees: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    cells = int(round(300.0 / GRID_RESOLUTION_M))
    raw = np.zeros((cells, cells), dtype=np.uint8)
    for tree in trees:
        x, y, _ = tree["center_enu_m"]
        col = int(round((x + 150.0) / GRID_RESOLUTION_M))
        row = int(round((150.0 - y) / GRID_RESOLUTION_M))
        radius = int(math.ceil(float(tree["radius_m"]) / GRID_RESOLUTION_M))
        cv2.circle(raw, (col, row), radius, 255, thickness=-1)
    radius = int(math.ceil(SAFETY_INFLATION_M / GRID_RESOLUTION_M))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return raw, cv2.dilate(raw, kernel)


def compound_footprint(footprint: dict[str, object]) -> PlotPath:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in [footprint["outer"], *footprint["holes"]]:
        points = [tuple(point) for point in ring]
        vertices.extend(points + [points[0]])
        codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1) + [PlotPath.CLOSEPOLY])
    return PlotPath(vertices, codes)


def decorate(ax: plt.Axes, bounds: float, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.set_aspect("equal")
    ax.set_xlabel("East / Gazebo world X (m)")
    ax.set_ylabel("North / Gazebo world Y (m)")
    ax.grid(color="white", alpha=0.18, linewidth=0.4)
    x = bounds - 18.0
    ax.annotate("N", (x, bounds - 10.0), (x, bounds - 35.0), ha="center",
                arrowprops={"arrowstyle": "-|>", "color": "#1565c0", "lw": 2},
                color="#1565c0", fontweight="bold")
    ax.plot([-bounds + 15.0, -bounds + 65.0], [-bounds + 15.0] * 2, color="black", lw=4)
    ax.text(-bounds + 40.0, -bounds + 20.0, "50 m", ha="center", fontsize=8)


def plot_occupancy(ax: plt.Axes, raw: np.ndarray, inflated: np.ndarray, bounds: float,
                   spawn: tuple[float, float], title: str) -> None:
    display = np.full(raw.shape, 255, dtype=np.uint8)
    display[inflated > 0] = 170
    display[raw > 0] = 0
    ax.imshow(display, cmap="gray", vmin=0, vmax=255, origin="upper",
              extent=(-bounds, bounds, -bounds, bounds), interpolation="nearest")
    ax.scatter(*spawn, marker="*", s=130, c="#00c853", edgecolors="black", linewidths=0.8,
               zorder=5)
    decorate(ax, bounds, title)
    ax.legend(handles=[
        Line2D([0], [0], color="black", lw=8, label="Gazebo collision"),
        Line2D([0], [0], color="0.67", lw=8, label=f"+ {SAFETY_INFLATION_M:g} m inflation"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#00c853",
               markeredgecolor="black", markersize=11, label="PX4 spawn"),
    ], loc="upper left", fontsize=8, framealpha=0.92)


def plot_city(buildings: list[dict[str, object]], raw: np.ndarray, inflated: np.ndarray) -> None:
    with Image.open(CITY_ROADS) as image:
        roads = np.asarray(image.convert("RGB"))
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200, constrained_layout=True)
    axes[0].imshow(roads, origin="upper", extent=(-250, 250, -250, 250), alpha=0.82)
    for building in buildings:
        axes[0].add_patch(PathPatch(compound_footprint(building["footprint"]),
                                    facecolor="#5d6670", edgecolor="#20262c", linewidth=0.22))
    axes[0].scatter(-120, 115, marker="*", s=130, c="#00c853", edgecolors="black", zorder=5)
    decorate(axes[0], 250.0, "City semantic reference — exact 274 building footprints")
    axes[0].legend(handles=[
        Line2D([0], [0], color="#5d6670", lw=8, label="Building collision footprint"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#00c853",
               markeredgecolor="black", markersize=11, label="PX4 spawn"),
    ], loc="upper left", fontsize=8, framealpha=0.92)
    plot_occupancy(axes[1], raw, inflated, 250.0, (-120.0, 115.0),
                   "City SLAM-style occupancy reference (0.25 m/cell)")
    figure.suptitle("PX4-ROS2 city path-planning coordinates — Gazebo world ENU", fontsize=14)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(CITY_FIGURE, facecolor="white")
    plt.close(figure)
    shutil.copy2(CITY_FIGURE, HOME_CITY_FIGURE)


def plot_mountain(trees: list[dict[str, object]], raw: np.ndarray, inflated: np.ndarray) -> None:
    with Image.open(MOUNTAIN_HEIGHT) as image:
        heights = np.asarray(image, dtype=np.float64) / 255.0 * 40.0
    axis = np.linspace(-150.0, 150.0, heights.shape[0])
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200, constrained_layout=True)
    terrain = axes[0].imshow(heights, cmap="terrain", vmin=0, vmax=40, origin="upper",
                             extent=(-150, 150, -150, 150))
    axes[0].contour(axis, axis, np.flipud(heights), levels=np.arange(5, 41, 5),
                    colors="white", alpha=0.55, linewidths=0.45)
    for tree in trees:
        x, y, _ = tree["center_enu_m"]
        color = "#174d1b" if tree["type"] == "pine_tree" else "#7b3f00"
        axes[0].add_patch(Circle((x, y), float(tree["radius_m"]), facecolor=color,
                                 edgecolor="black", linewidth=0.18))
    axes[0].scatter(-80, -80, marker="*", s=130, c="#00c853", edgecolors="black", zorder=5)
    decorate(axes[0], 150.0, "Mountain elevation + 288 exact trunk collisions (maze removed)")
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#174d1b",
               markeredgecolor="black", markersize=7, label="Pine trunk (216)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7b3f00",
               markeredgecolor="black", markersize=7, label="Oak trunk (72)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#00c853",
               markeredgecolor="black", markersize=11, label="PX4 spawn"),
    ], loc="upper left", fontsize=8, framealpha=0.92)
    figure.colorbar(terrain, ax=axes[0], shrink=0.76, label="Terrain elevation (m)")
    plot_occupancy(axes[1], raw, inflated, 150.0, (-80.0, -80.0),
                   "Mountain SLAM-style occupancy reference (0.25 m/cell)")
    figure.suptitle("PX4-ROS2 mountain path-planning coordinates — Gazebo world ENU", fontsize=14)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(MOUNTAIN_FIGURE, facecolor="white")
    plt.close(figure)
    shutil.copy2(MOUNTAIN_FIGURE, HOME_MOUNTAIN_FIGURE)


def validate_spawn_free(raw: np.ndarray, half_size: float, spawn: tuple[float, float]) -> None:
    col = int(round((spawn[0] + half_size) / GRID_RESOLUTION_M))
    row = int(round((half_size - spawn[1]) / GRID_RESOLUTION_M))
    require(raw[row, col] == 0, "PX4 spawn lies inside a collision obstacle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="regenerate and verify invariants")
    parser.parse_args()
    buildings = city_buildings()
    trees = mountain_trees()
    write_yaml(CITY_YAML, city_document(buildings))
    write_yaml(MOUNTAIN_YAML, mountain_document(trees))
    city_raw, city_inflated = city_occupancy(buildings)
    mountain_raw, mountain_inflated = mountain_occupancy(trees)
    validate_spawn_free(city_raw, 250.0, (-120.0, 115.0))
    validate_spawn_free(mountain_raw, 150.0, (-80.0, -80.0))
    plot_city(buildings, city_raw, city_inflated)
    plot_mountain(trees, mountain_raw, mountain_inflated)
    print(f"city_yaml={CITY_YAML} buildings={len(buildings)} sha256={sha256(CITY_YAML)}")
    print(f"mountain_yaml={MOUNTAIN_YAML} trees={len(trees)} maze_walls=0 sha256={sha256(MOUNTAIN_YAML)}")
    print(f"city_reference={CITY_FIGURE} home_copy={HOME_CITY_FIGURE} sha256={sha256(CITY_FIGURE)}")
    print(f"mountain_reference={MOUNTAIN_FIGURE} home_copy={HOME_MOUNTAIN_FIGURE} sha256={sha256(MOUNTAIN_FIGURE)}")


if __name__ == "__main__":
    main()
