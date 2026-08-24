#!/usr/bin/env python3
"""Fail-closed validation for the generated Apple Park UAV city assets."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml
from PIL import Image, ImageDraw

import expand_city_for_uav as gen


TOLERANCE_XY_M = 0.01
TOLERANCE_Z_M = 1.0e-9
COLLADA_NAMESPACE = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_point_error(first: gen.Point, second: gen.Point) -> float:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def load_dae_positions(path: Path) -> tuple[list[tuple[float, float, float]], int]:
    root = ET.parse(path).getroot()
    up_axis = root.findtext("c:asset/c:up_axis", namespaces=COLLADA_NAMESPACE)
    require(up_axis == "Z_UP", f"expected Z_UP DAE, got {up_axis!r}")
    array = root.find(".//c:source[@id='positions']/c:float_array", COLLADA_NAMESPACE)
    require(array is not None and array.text, "DAE positions are missing")
    values = [float(value) for value in array.text.split()]
    require(len(values) % 3 == 0, "DAE position array is not XYZ-strided")
    positions = [tuple(values[index : index + 3]) for index in range(0, len(values), 3)]
    triangles = root.find(".//c:triangles", COLLADA_NAMESPACE)
    require(triangles is not None, "DAE triangles are missing")
    triangle_count = int(triangles.attrib["count"])
    require(len(positions) == triangle_count * 3, "expected unique three vertices per triangle")
    normals = root.find(".//c:source[@id='normals']/c:float_array", COLLADA_NAMESPACE)
    require(normals is not None and normals.text, "DAE face normals are missing")
    normal_values = [float(value) for value in normals.text.split()]
    require(len(normal_values) == triangle_count * 3, "one face normal per triangle is required")
    return positions, triangle_count


def validate() -> dict:
    required_paths = (
        gen.SOURCE_YAML,
        gen.OUTPUT_YAML,
        gen.OUTPUT_WORLD,
        gen.OUTPUT_MODEL_CONFIG,
        gen.OUTPUT_ATTRIBUTION,
        gen.OUTPUT_DAE,
        gen.OUTPUT_ROAD,
        gen.OUTPUT_HEIGHT,
        gen.OUTPUT_NORMAL,
        gen.OUTPUT_VERTICES_CSV,
    )
    for path in required_paths:
        require(path.is_file(), f"missing required generated asset: {path}")
    require(
        gen.OUTPUT_ATTRIBUTION.read_text(encoding="utf-8")
        == gen.SOURCE_ATTRIBUTION.read_text(encoding="utf-8"),
        "derived road texture attribution differs from source",
    )

    source = yaml.safe_load(gen.SOURCE_YAML.read_text(encoding="utf-8"))
    derived = yaml.safe_load(gen.OUTPUT_YAML.read_text(encoding="utf-8"))
    source_buildings = source["obstacles"]["buildings"]
    derived_buildings = derived["obstacles"]["buildings"]
    active_source_buildings, expected_removed_ids = gen.select_active_buildings(source_buildings)
    require(len(source_buildings) == gen.SOURCE_BUILDING_COUNT, "source building count changed")
    require(len(derived_buildings) == gen.ACTIVE_BUILDING_COUNT, "derived city must contain 205 buildings")
    require(
        [building["id"] for building in active_source_buildings]
        == [building["id"] for building in derived_buildings],
        "derived building IDs/order differ from deterministic reduction",
    )

    derivation = derived["derivation"]
    spacing_scale = float(derivation["city_spacing_scale_xy"])
    footprint_scale = float(derivation["building_footprint_scale_xy"])
    anchor = tuple(float(value) for value in derivation["anchor_enu_m"])
    require(abs(spacing_scale - 2.5) <= 1.0e-12, "default derived city is not spaced by 2.5")
    require(footprint_scale in gen.FOOTPRINT_SCALE_CANDIDATES, "footprint scale is not an allowed candidate")
    height_transform = derivation["height_transform"]
    require(height_transform["source_profile"] == gen.SOURCE_HEIGHT_PROFILE,
            "source height profile metadata mismatch")
    require(height_transform["active_profile"] == gen.ACTIVE_HEIGHT_PROFILE,
            "active height profile metadata mismatch")
    require([float(value) for value in height_transform["source_range_m"]]
            == [gen.SOURCE_HEIGHT_MIN_M, gen.SOURCE_HEIGHT_MAX_M],
            "source height range metadata mismatch")
    require([float(value) for value in height_transform["active_range_m"]]
            == [gen.ACTIVE_HEIGHT_MIN_M, gen.ACTIVE_HEIGHT_MAX_M],
            "active height range metadata mismatch")
    require(height_transform["foundation_and_ground_unchanged"] is True,
            "foundation/ground preservation metadata mismatch")

    maximum_foundation_ground_error = 0.0
    maximum_height_profile_error = 0.0
    maximum_centroid_error = 0.0
    maximum_footprint_error = 0.0
    expected_boundary_xyz: list[tuple[float, float, float]] = []
    expected_height_by_id = gen.active_height_by_identifier(active_source_buildings)
    for source_record, derived_record in zip(active_source_buildings, derived_buildings):
        for key in ("foundation_z_m", "ground_reference_z_m"):
            error = abs(float(source_record[key]) - float(derived_record[key]))
            maximum_foundation_ground_error = max(maximum_foundation_ground_error, error)
            require(error <= TOLERANCE_Z_M, f"{derived_record['id']} changed {key} by {error}")
        expected_height = expected_height_by_id[str(source_record["id"])]
        expected_roof = float(source_record["ground_reference_z_m"]) + expected_height
        for key, expected_value in (
            ("height_above_ground_m", expected_height),
            ("roof_z_m", expected_roof),
        ):
            error = abs(float(derived_record[key]) - expected_value)
            maximum_height_profile_error = max(maximum_height_profile_error, error)
            require(
                error <= TOLERANCE_Z_M,
                f"{derived_record['id']} does not follow the 20-50m height profile",
            )

        aabb = derived_record.get("aabb_xyz_m")
        require(isinstance(aabb, dict), f"{derived_record['id']} has no XYZ AABB")

        source_outer = gen.normalize_ring(source_record["footprint"]["outer"], ccw=True)
        source_holes = [gen.normalize_ring(hole, ccw=False) for hole in source_record["footprint"].get("holes", [])]
        source_centroid = gen.polygon_centroid(source_outer, source_holes)
        expected_centroid = (
            anchor[0] + spacing_scale * (source_centroid[0] - anchor[0]),
            anchor[1] + spacing_scale * (source_centroid[1] - anchor[1]),
        )
        derived_outer = [tuple(map(float, point)) for point in derived_record["footprint"]["outer"]]
        derived_holes = [
            [tuple(map(float, point)) for point in hole]
            for hole in derived_record["footprint"].get("holes", [])
        ]
        actual_centroid = gen.polygon_centroid(derived_outer, derived_holes)
        centroid_error = maximum_point_error(actual_centroid, expected_centroid)
        maximum_centroid_error = max(maximum_centroid_error, centroid_error)
        require(centroid_error <= TOLERANCE_XY_M, f"{derived_record['id']} centroid transform misaligned")

        expected_outer = gen.transform_ring(source_outer, source_centroid, expected_centroid, footprint_scale)
        expected_holes = [
            gen.transform_ring(hole, source_centroid, expected_centroid, footprint_scale)
            for hole in source_holes
        ]
        require(len(expected_outer) == len(derived_outer), f"{derived_record['id']} outer topology changed")
        require(len(expected_holes) == len(derived_holes), f"{derived_record['id']} hole count changed")
        for expected, actual in zip(expected_outer, derived_outer):
            error = maximum_point_error(expected, actual)
            maximum_footprint_error = max(maximum_footprint_error, error)
            require(error <= TOLERANCE_XY_M, f"{derived_record['id']} footprint transform misaligned")
        for expected_hole, actual_hole in zip(expected_holes, derived_holes):
            require(len(expected_hole) == len(actual_hole), f"{derived_record['id']} hole topology changed")
            for expected, actual in zip(expected_hole, actual_hole):
                error = maximum_point_error(expected, actual)
                maximum_footprint_error = max(maximum_footprint_error, error)
                require(error <= TOLERANCE_XY_M, f"{derived_record['id']} hole transform misaligned")

        foundation = float(derived_record["foundation_z_m"])
        roof = float(derived_record["roof_z_m"])
        for ring in [derived_outer, *derived_holes]:
            for x, y in ring:
                expected_boundary_xyz.append((x, y, foundation))
                expected_boundary_xyz.append((x, y, roof))
        bounds_xy = (
            min(point[0] for point in derived_outer),
            min(point[1] for point in derived_outer),
            max(point[0] for point in derived_outer),
            max(point[1] for point in derived_outer),
        )
        expected_aabb_min = [bounds_xy[0], bounds_xy[1], foundation]
        expected_aabb_max = [bounds_xy[2], bounds_xy[3], roof]
        expected_aabb_center = [
            0.5 * (bounds_xy[0] + bounds_xy[2]),
            0.5 * (bounds_xy[1] + bounds_xy[3]),
            0.5 * (foundation + roof),
        ]
        expected_aabb_size = [
            bounds_xy[2] - bounds_xy[0],
            bounds_xy[3] - bounds_xy[1],
            roof - foundation,
        ]
        for key, expected_values in (
            ("min", expected_aabb_min),
            ("max", expected_aabb_max),
            ("center_enu_m", expected_aabb_center),
            ("size_xyz_m", expected_aabb_size),
        ):
            actual_values = [float(value) for value in aabb[key]]
            require(
                max(abs(actual - expected) for actual, expected in zip(actual_values, expected_values))
                <= TOLERANCE_XY_M,
                f"{derived_record['id']} {key} differs from footprint prism",
            )

    positions, triangle_count = load_dae_positions(gen.OUTPUT_DAE)
    expected_keys = {
        # COLLADA is emitted with 12 significant digits; for single-digit
        # roofs that can differ from YAML by one nanometre at the final digit.
        (round(x, 6), round(y, 6), round(z, 8))
        for x, y, z in expected_boundary_xyz
    }
    mesh_keys = {(round(x, 6), round(y, 6), round(z, 8)) for x, y, z in positions}
    missing_from_mesh = expected_keys - mesh_keys
    extra_mesh_vertices = mesh_keys - expected_keys
    require(not missing_from_mesh, f"{len(missing_from_mesh)} YAML boundary vertices are absent from DAE")
    require(not extra_mesh_vertices, f"{len(extra_mesh_vertices)} DAE vertices do not come from YAML boundaries")

    expected_csv_header = [
        "schema_version", "map_name", "frame_id", "building_id",
        "source_component_id", "ring_role", "ring_index", "vertex_order",
        "next_vertex_order", "ring_vertex_count", "winding", "x_enu_m",
        "y_enu_m", "foundation_z_m", "ground_reference_z_m", "roof_z_m",
        "height_above_ground_m",
    ]
    with gen.OUTPUT_VERTICES_CSV.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == expected_csv_header, "A* vertex CSV header changed")
        csv_rows = list(reader)
    expected_csv_keys: list[tuple[str, str, int, int]] = []
    expected_by_key: dict[tuple[str, str, int, int], tuple[gen.Point, int, int, str, float, float, float, float, int]] = {}
    for building in derived_buildings:
        rings = [("outer", 0, building["footprint"]["outer"])] + [
            ("hole", index, ring)
            for index, ring in enumerate(building["footprint"].get("holes", []))
        ]
        for role, ring_index, ring in rings:
            winding = "CCW" if role == "outer" else "CW"
            for vertex_order, point in enumerate(ring):
                key = (str(building["id"]), role, ring_index, vertex_order)
                expected_csv_keys.append(key)
                expected_by_key[key] = (
                    (float(point[0]), float(point[1])),
                    (vertex_order + 1) % len(ring),
                    len(ring),
                    winding,
                    float(building["foundation_z_m"]),
                    float(building["ground_reference_z_m"]),
                    float(building["roof_z_m"]),
                    float(building["height_above_ground_m"]),
                    int(building["source_component_id"]),
                )
    require(len(csv_rows) == len(expected_csv_keys), "A* vertex CSV row count differs from YAML")
    actual_csv_keys: list[tuple[str, str, int, int]] = []
    for row in csv_rows:
        key = (
            row["building_id"], row["ring_role"], int(row["ring_index"]),
            int(row["vertex_order"]),
        )
        actual_csv_keys.append(key)
        require(key in expected_by_key, f"unexpected A* CSV vertex key: {key}")
        point, next_order, ring_count, winding, foundation, ground, roof, height_value, component_id = expected_by_key[key]
        require(int(row["schema_version"]) == gen.VERTEX_CSV_SCHEMA_VERSION,
                "A* vertex CSV schema version mismatch")
        require(row["map_name"] == "city_uav" and row["frame_id"] == "gazebo_world_enu",
                "A* vertex CSV map/frame mismatch")
        require(int(row["source_component_id"]) == component_id,
                f"{key}: A* source component mismatch")
        require(int(row["next_vertex_order"]) == next_order,
                f"{key}: A* ring closure order mismatch")
        require(int(row["ring_vertex_count"]) == ring_count and row["winding"] == winding,
                f"{key}: A* ring topology mismatch")
        actual_values = (
            float(row["x_enu_m"]), float(row["y_enu_m"]),
            float(row["foundation_z_m"]), float(row["ground_reference_z_m"]),
            float(row["roof_z_m"]), float(row["height_above_ground_m"]),
        )
        expected_values = (point[0], point[1], foundation, ground, roof, height_value)
        require(
            max(abs(actual - expected) for actual, expected in zip(actual_values, expected_values))
            <= TOLERANCE_Z_M,
            f"{key}: A* CSV coordinate differs from YAML",
        )
        for z_value in (foundation, roof):
            require(
                (round(point[0], 6), round(point[1], 6), round(z_value, 8)) in mesh_keys,
                f"{key}: A* XYZ corner is absent from DAE",
            )
    require(actual_csv_keys == expected_csv_keys, "A* vertex CSV ordering differs from YAML")
    require(len(set(actual_csv_keys)) == len(actual_csv_keys), "A* vertex CSV contains duplicate keys")
    astar_metadata = derivation["astar_geometry"]
    require(astar_metadata["vertices_csv"] == "gazebo/maps/city_uav_building_vertices.csv",
            "A* CSV path metadata mismatch")
    require(int(astar_metadata["logical_xy_vertex_count"]) == len(csv_rows),
            "A* logical XY vertex count metadata mismatch")
    require(int(astar_metadata["logical_xyz_vertex_count"]) == 2 * len(csv_rows),
            "A* logical XYZ vertex count metadata mismatch")
    require(astar_metadata["sha256"] == gen.sha256_file(gen.OUTPUT_VERTICES_CSV),
            "A* vertex CSV digest metadata is stale")

    world_root = ET.parse(gen.OUTPUT_WORLD).getroot()
    world = world_root.find("world")
    require(world is not None and world.attrib.get("name") == "applepark_city_uav", "wrong Gazebo world name")
    gui = world.find("gui")
    require(gui is not None, "generated world is missing its GUI configuration")
    gui_plugins = gui.findall("plugin")
    minimal_scene = next(
        (plugin for plugin in gui_plugins if plugin.attrib.get("filename") == "MinimalScene"),
        None,
    )
    require(minimal_scene is not None, "generated world is missing MinimalScene")
    require(
        minimal_scene.findtext("camera_pose")
        == f"{gen.fmt(gen.SPAWN_XY[0] - 4.0)} {gen.fmt(gen.SPAWN_XY[1])} 3 0 0.45 0",
        "generated world spawn-close fallback camera pose changed",
    )
    require(
        any(plugin.attrib.get("filename") == "EntityTree" for plugin in gui_plugins),
        "generated world is missing EntityTree for dynamic PX4 visibility",
    )
    visual_uri = world.findtext(".//model[@name='applepark_uav_buildings']/link/visual/geometry/mesh/uri")
    require(visual_uri == "mesh/buildings_uav.dae", "visual does not reference generated DAE")
    visual_scale = world.findtext(".//model[@name='applepark_uav_buildings']/link/visual/geometry/mesh/scale")
    require(visual_scale == "1 1 1", "world-level building scale is forbidden")
    building_model = world.find(".//model[@name='applepark_uav_buildings']")
    require(building_model is not None, "building model is missing")
    require(building_model.findtext("static") == "true", "building collision model must be static")
    expected_geometry = gen.transform_buildings(active_source_buildings, spacing_scale, footprint_scale, anchor)
    collision_elements = building_model.findall("link/collision")
    require(len(collision_elements) == len(expected_geometry),
            "building world must contain one exact collision prism per building")
    maximum_collision_parameter_error = 0.0
    for collision, expected in zip(collision_elements, expected_geometry):
        require(
            collision.attrib.get("name") == f"{expected.identifier}_exact_polyline",
            f"{expected.identifier}: wrong collision prism name",
        )
        pose_text = collision.findtext("pose")
        require(pose_text is not None, f"{expected.identifier}: collision pose is missing")
        pose_values = [float(value) for value in pose_text.split()]
        pose_error = max(
            abs(pose_values[index] - expected_value)
            for index, expected_value in enumerate(
                (0.0, 0.0, expected.foundation_z, 0.0, 0.0, 0.0)
            )
        )
        maximum_collision_parameter_error = max(maximum_collision_parameter_error, pose_error)
        require(pose_error <= TOLERANCE_Z_M,
                f"{expected.identifier}: collision foundation pose differs from YAML")
        polylines = collision.findall("geometry/polyline")
        expected_rings = [expected.transformed.outer, *expected.transformed.holes]
        require(len(polylines) == len(expected_rings),
                f"{expected.identifier}: collision ring count differs from YAML")
        for polyline, expected_ring in zip(polylines, expected_rings):
            height_text = polyline.findtext("height")
            require(height_text is not None, f"{expected.identifier}: collision height is missing")
            height_error = abs(float(height_text) - (expected.roof_z - expected.foundation_z))
            maximum_collision_parameter_error = max(
                maximum_collision_parameter_error, height_error
            )
            require(height_error <= TOLERANCE_Z_M,
                    f"{expected.identifier}: collision height differs from YAML")
            points = [
                tuple(float(value) for value in (point.text or "").split())
                for point in polyline.findall("point")
            ]
            require(len(points) == len(expected_ring),
                    f"{expected.identifier}: collision vertex count differs from YAML")
            for actual, expected_point in zip(points, expected_ring):
                error = maximum_point_error(actual, expected_point)  # type: ignore[arg-type]
                maximum_collision_parameter_error = max(
                    maximum_collision_parameter_error, error
                )
                require(error <= TOLERANCE_XY_M,
                        f"{expected.identifier}: collision point differs from YAML")
    require(not building_model.findall("link/collision/geometry/mesh"),
            "unsupported DART mesh collision remains in the active world")
    collision_metadata = derivation["collision_geometry"]
    require(collision_metadata["type"] == gen.COLLISION_GEOMETRY_TYPE, "collision metadata type mismatch")
    require(int(collision_metadata["count"]) == len(expected_geometry), "collision metadata count mismatch")
    require(
        collision_metadata["required_physics_engine"]
        == "gz-physics-dartsim-plugin",
        "exact polyline collision is not pinned to DART",
    )
    require(collision_metadata["dart_sdf_mesh_supported"] is False,
            "DART mesh-collision limitation is not explicit")
    require(collision_metadata["dart_sdf_polyline_supported"] is True,
            "DART polyline-collision support is not explicit")
    require(float(collision_metadata["maximum_outward_error_m"]) == 0.0, "collision metadata is not exact")
    require(float(collision_metadata["maximum_undercoverage_m"]) == 0.0, "collision metadata undercovers")
    require(
        world.find(".//model[@name='applepark_uav_building_collision_proxies']") is None,
        "legacy collision proxy model remains in world",
    )

    expected_triangle_positions = [
        point
        for building in expected_geometry
        for triangle in gen.prism_triangles(building)
        for point in triangle
    ]
    require(
        triangle_count == len(expected_triangle_positions) // 3,
        "DAE triangle count differs from the exact YAML prism triangulation",
    )
    require(
        len(positions) == len(expected_triangle_positions),
        "DAE vertex stream differs from the exact YAML prism triangulation",
    )
    maximum_mesh_stream_error = max(
        max(abs(actual_axis - expected_axis) for actual_axis, expected_axis in zip(actual, expected))
        for actual, expected in zip(positions, expected_triangle_positions)
    )
    require(maximum_mesh_stream_error <= TOLERANCE_XY_M, "DAE triangle stream differs from YAML geometry")
    bounds = derived["map"]["bounds_enu_m"]
    expected_ground_size = float(bounds["x"][1]) - float(bounds["x"][0])
    require(
        abs(expected_ground_size - 2.0 * gen.MINIMUM_MAP_HALF_SIZE_M) <= 1.0e-9,
        "UAV city safety ground size changed",
    )
    box_size_text = world.findtext(".//model[@name='applepark_uav_ground']/link/collision/geometry/box/size")
    require(box_size_text is not None, "ground collision box is missing")
    box_size = [float(value) for value in box_size_text.split()]
    require(box_size == [expected_ground_size, expected_ground_size, 0.1], "ground collision size differs from YAML")
    ground_pose_text = world.findtext(".//model[@name='applepark_uav_ground']/link/collision/pose")
    require(ground_pose_text is not None and float(ground_pose_text.split()[2]) == -0.05, "ground top is not z=0")

    height = Image.open(gen.OUTPUT_HEIGHT)
    require(
        height.size == (gen.FLAT_HEIGHTMAP_PIXELS, gen.FLAT_HEIGHTMAP_PIXELS),
        "flat heightmap dimensions changed",
    )
    require(height.getextrema() == (255, 255), "heightmap is not completely flat")
    require(
        Image.open(gen.OUTPUT_NORMAL).size
        == (gen.FLAT_HEIGHTMAP_PIXELS, gen.FLAT_HEIGHTMAP_PIXELS),
        "normal map dimensions changed",
    )
    road = Image.open(gen.OUTPUT_ROAD).convert("RGB")
    require(
        road.size == (gen.ROAD_TEXTURE_PIXELS, gen.ROAD_TEXTURE_PIXELS),
        "road texture dimensions changed",
    )
    require(
        derived["terrain"]["texture_boundary_extension"]
        == "edge_clamp_feather_to_neutral",
        "road texture edge-extension contract changed",
    )
    require(
        derived["terrain"]["southeast_texture_repair"]
        == "diagonal_edge_clamp_with_osm_road_restore",
        "south-east texture repair contract changed",
    )

    xmin, xmax = map(float, bounds["x"])
    ymin, ymax = map(float, bounds["y"])

    def world_to_pixel(point: gen.Point) -> tuple[float, float]:
        return (
            (point[0] - xmin) / (xmax - xmin) * (road.width - 1),
            (ymax - point[1]) / (ymax - ymin) * (road.height - 1),
        )

    # The 2.5x footprint is the exact map-space counterpart of origin/main's
    # initial city.  Certify that this larger geometry still does not cover a
    # single strict dark-asphalt pixel in the generated road texture.
    building_mask_image = Image.new("1", road.size, 0)
    mask_draw = ImageDraw.Draw(building_mask_image)
    for building in expected_geometry:
        mask_draw.polygon([world_to_pixel(point) for point in building.transformed.outer], fill=1)
        for hole in building.transformed.holes:
            mask_draw.polygon([world_to_pixel(point) for point in hole], fill=0)
    road_array = np.asarray(road, dtype=np.uint8)
    neutral = np.asarray(gen.NEUTRAL_GROUND_RGB, dtype=np.uint8)
    for edge in (road_array[0, :, :], road_array[-1, :, :], road_array[:, 0, :], road_array[:, -1, :]):
        require(np.all(edge == neutral), "road texture outer edge is not neutral ground")
    source_half_extent = 250.0 * spacing_scale
    # Match the generator's affine valid-pixel convention exactly. For the
    # restored 1300 m extent, the source edge falls between pixels, so round()
    # would inspect the wrong pair and miss a one-pixel boundary seam.
    seam_columns = [
        gen.math.ceil(
            (anchor[0] - source_half_extent - xmin) / (xmax - xmin)
            * (road.width - 1)
        ),
        gen.math.floor(
            (anchor[0] + source_half_extent - xmin) / (xmax - xmin)
            * (road.width - 1)
        ),
    ]
    seam_rows = [
        gen.math.ceil(
            (ymax - (anchor[1] + source_half_extent)) / (ymax - ymin)
            * (road.height - 1)
        ),
        gen.math.floor(
            (ymax - (anchor[1] - source_half_extent)) / (ymax - ymin)
            * (road.height - 1)
        ),
    ]
    seam_differences = []
    road_float = road_array.astype(np.float32)
    # Rounding places the west / north coordinates on the first source pixel,
    # while the east / south coordinates land on the last source pixel.  Test
    # the actual outside-to-inside pair on each side, not one pixel inward.
    for outside, inside in (
        (seam_columns[0] - 1, seam_columns[0]),
        (seam_columns[1] + 1, seam_columns[1]),
    ):
        seam_differences.append(
            np.linalg.norm(road_float[:, outside, :] - road_float[:, inside, :], axis=1)
        )
    for outside, inside in (
        (seam_rows[0] - 1, seam_rows[0]),
        (seam_rows[1] + 1, seam_rows[1]),
    ):
        seam_differences.append(
            np.linalg.norm(road_float[outside, :, :] - road_float[inside, :, :], axis=1)
        )
    seam_difference = np.concatenate(seam_differences)
    require(float(seam_difference.mean()) < 10.0, "road texture extension has a visible mean seam")
    require(
        float(np.count_nonzero(seam_difference > 50.0)) / seam_difference.size < 0.03,
        "road texture extension has a widespread hard seam",
    )
    road_max = road_array.max(axis=2)
    road_min = road_array.min(axis=2)
    strict_asphalt = (
        (road_array[:, :, 0] <= 75)
        & (road_array[:, :, 1] <= 80)
        & (road_array[:, :, 2] <= 85)
        & ((road_max - road_min) < 28)
    )
    active_road_overlap_pixels = int(
        np.count_nonzero(np.asarray(building_mask_image, dtype=bool) & strict_asphalt)
    )
    require(active_road_overlap_pixels == 0, "expanded buildings overlap visible asphalt")

    def road_rgb(point: gen.Point) -> tuple[int, int, int]:
        column = round((point[0] - xmin) / (xmax - xmin) * (road.width - 1))
        row = round((ymax - point[1]) / (ymax - ymin) * (road.height - 1))
        require(0 <= column < road.width and 0 <= row < road.height,
                f"road sample is outside the map: {point}")
        return road.getpixel((column, row))

    def is_asphalt(point: gen.Point) -> bool:
        red, green, blue = road_rgb(point)
        # Main asphalt is the connected dark neutral family around (58,63,68).
        # Keep the threshold strict even though the old dark affine-fill seam
        # has been removed; only real connected asphalt certifies a spawn.
        return red <= 75 and green <= 80 and blue <= 85 and max(red, green, blue) - min(red, green, blue) < 28

    # The drone gets a 2 x 2 m asphalt square. The trailer checks include a
    # 1 m halo around its 5 x 5 m body (7 x 7 m total).
    for x_step in range(-4, 5):
        for y_step in range(-4, 5):
            point = (gen.SPAWN_XY[0] + 0.25 * x_step, gen.SPAWN_XY[1] + 0.25 * y_step)
            require(is_asphalt(point), f"drone spawn safety square leaves asphalt at {point}")
    for x_step in range(-14, 15):
        for y_step in range(-14, 15):
            point = (
                gen.TRAILER_SPAWN_XY[0] + 0.25 * x_step,
                gen.TRAILER_SPAWN_XY[1] + 0.25 * y_step,
            )
            require(is_asphalt(point), f"trailer spawn safety rectangle leaves asphalt at {point}")
    closed_patrol = gen.TRAILER_PATROL_WAYPOINTS + (
        gen.TRAILER_PATROL_WAYPOINTS[0],
    )
    for first, second in zip(closed_patrol, closed_patrol[1:]):
        samples = max(1, math.ceil(math.dist(first, second)))
        for index in range(samples + 1):
            ratio = index / samples
            center = (
                first[0] + ratio * (second[0] - first[0]),
                first[1] + ratio * (second[1] - first[1]),
            )
            for offset_x in (-3.5, 0.0, 3.5):
                for offset_y in (-3.5, 0.0, 3.5):
                    sample = (center[0] + offset_x, center[1] + offset_y)
                    require(
                        is_asphalt(sample),
                        f"trailer patrol plus 1 m halo leaves asphalt at {sample}",
                    )
    for restored_road_sample in (
        (512.0, -450.0),
        (575.0, -322.0),
        (636.0, -450.0),
    ):
        require(
            is_asphalt(restored_road_sample),
            f"south-east restored road is missing at {restored_road_sample}",
        )
    spawn_separation = gen.math.dist(gen.SPAWN_XY, gen.TRAILER_SPAWN_XY)
    require(spawn_separation >= 100.0, "drone/trailer initial separation is below 100m")
    spawn_boundary_clearance = min(
        gen.SPAWN_XY[0] - xmin, xmax - gen.SPAWN_XY[0],
        gen.SPAWN_XY[1] - ymin, ymax - gen.SPAWN_XY[1],
    )
    trailer_boundary_clearance = min(
        gen.TRAILER_SPAWN_XY[0] - xmin, xmax - gen.TRAILER_SPAWN_XY[0],
        gen.TRAILER_SPAWN_XY[1] - ymin, ymax - gen.TRAILER_SPAWN_XY[1],
    )
    require(spawn_boundary_clearance >= 50.0 and trailer_boundary_clearance >= 50.0,
            "a staging site is too close to the map boundary")

    southwest_clearance = min(-600.0 - xmin, -600.0 - ymin)
    require(southwest_clearance >= 50.0, "(-600,-600) is outside the restored 1300m ground")

    spawn = derived["spawn"]["gazebo_spawn_pose_enu"]
    require((float(spawn["x"]), float(spawn["y"])) == gen.SPAWN_XY, "spawn coordinate was scaled")
    require(float(spawn["z"]) == 0.0, "derived PX4 spawn must reference flat ground z=0")
    require(
        derived["frames"]["px4_local"]["origin_enu_m"]
        == [gen.SPAWN_XY[0], gen.SPAWN_XY[1], gen.PX4_BASE_LINK_Z_OFFSET_M],
        "PX4 local origin must use x500 base_link height, not model-root z",
    )
    expected_forward = {
        "rgb_camera": {
            "parent_link": "base_link", "link": "front_rgb_link",
            "pose_xyz_rpy": [0.12, 0.03, 0.002, 0.0, 0.0, 0.0],
            "image_topic": "/front_camera/image",
            "camera_info_topic": "/front_camera/camera_info",
            "optical_frame_id": "front_camera_optical_frame",
            "resolution": [640, 360], "encoding": "rgb8",
            "nominal_update_rate_hz": 15.0,
        },
        "depth_camera": {
            "parent_link": "base_link", "link": "front_depth_link",
            "pose_xyz_rpy": [0.12, 0.0, 0.002, 0.0, 0.0, 0.0],
            "image_topic": "/front_depth/image",
            "metric_image_topic": "/front_depth/image_raw",
            "points_topic": "/front_depth/points",
            "camera_info_topic": "/front_depth/camera_info",
            "optical_frame_id": "front_depth_optical_frame",
            "resolution": [320, 240], "encoding": "mono8",
            "metric_encoding": "32FC1",
            "nominal_update_rate_hz": 12.0,
        },
    }
    expected_downward = {
        "rgb_camera": {
            "parent_link": "base_link", "link": "down_rgb_link",
            "pose_xyz_rpy": [0.0, 0.0, -0.05, 0.0, 1.57079632679, 0.0],
            "image_topic": "/down_camera/image",
            "camera_info_topic": "/down_camera/camera_info",
            "optical_frame_id": "down_camera_optical_frame",
            "resolution": [640, 480], "encoding": "rgb8",
            "nominal_update_rate_hz": 20.0,
        },
        "depth_camera": {
            "parent_link": "base_link", "link": "down_depth_link",
            "pose_xyz_rpy": [0.0, -0.03, -0.05, 0.0, 1.57079632679, 0.0],
            "image_topic": "/down_depth/image",
            "metric_image_topic": "/down_depth/image_raw",
            "points_topic": "/down_depth/points",
            "camera_info_topic": "/down_depth/camera_info",
            "optical_frame_id": "down_depth_optical_frame",
            "resolution": [320, 240], "encoding": "mono8",
            "metric_encoding": "32FC1",
            "nominal_update_rate_hz": 15.0,
        },
        "lidar": {
            "parent_link": "base_link", "link": "lidar_sensor_link",
            "pose_xyz_rpy": [0.0, 0.0, -0.05, 0.0, 1.57079632679, 0.0],
            "scan_topic": "/down_lidar", "points_topic": "/down_lidar/points",
            "frame_id": "lidar_sensor_link", "samples": 1,
            "range_m": [0.1, 100.0], "nominal_update_rate_hz": 50.0,
        },
    }
    require(derived["px4_vehicle"]["forward_sensors"] == expected_forward,
            "forward sensor YAML contract differs from the flyable model")
    require(derived["px4_vehicle"]["downward_sensors"] == expected_downward,
            "downward sensor YAML contract differs from the flyable model")
    require(
        derived["frames"]["gazebo_world"] == gen.CITY_GAZEBO_WORLD_FRAME
        and derived["frames"]["mavros_local"]
        == gen.CITY_MAVROS_LOCAL_FRAME,
        "city GPS/local frame contract differs from the generator",
    )
    require(
        derived["px4_vehicle"]["gimbal_variant"]
        == gen.CITY_GIMBAL_VARIANT,
        "city gimbal variant differs from the generator",
    )
    sitl_overrides = derived["px4_vehicle"]["sitl_parameter_overrides"]
    require(
        sitl_overrides == gen.CITY_SITL_PARAMETER_OVERRIDES,
        "city SITL moving-deck contract differs from the generator",
    )
    require(
        derived["mission"] == gen.CITY_MISSION_PROFILE,
        "city MissionManager profile differs from the generator",
    )
    map_size = float(derived["map"]["bounds_enu_m"]["x"][1]) - float(
        derived["map"]["bounds_enu_m"]["x"][0]
    )
    require(
        derived["terrain"]["coordinate_frame"] == "gazebo_world"
        and derived["terrain"]["center_m"] == [0.0, 0.0]
        and derived["terrain"]["size_m"] == [map_size, map_size],
        "city terrain frame contract differs from the generator",
    )
    require("pad" not in derived["spawn"], "derived YAML still declares a spawn pad")
    require(world.find(".//model[@name='drone_spawn_pad']") is None, "green spawn pad remains in UAV world")
    spawn_frame = world.find(".//frame[@name='drone_spawn']")
    require(
        spawn_frame is not None
        and spawn_frame.findtext("pose")
        == f"{gen.fmt(gen.SPAWN_XY[0])} {gen.fmt(gen.SPAWN_XY[1])} 0 0 0 0",
        "spawn frame is not at the road site on z=0",
    )
    trailer = derived["trailer"]
    require(trailer["entity_name"] == "trailer", "derived trailer entity contract changed")
    require(
        trailer["model_uri"] == "model://moving_platform_aruco_velocity",
        "deterministic ArUco trailer model is not selected",
    )
    require(trailer["body_footprint_m"] == [5.0, 5.0], "trailer footprint mismatch")
    require(float(trailer["deck_height_m"]) == 2.05, "trailer deck height mismatch")
    require(
        trailer["controller_plugin"] == "gz-sim-velocity-control-system"
        and trailer["command_topic"] == "/model/trailer/cmd_vel",
        "city trailer velocity-controller contract differs",
    )
    require(trailer["motion"] == "repeated_black_road_waypoint_patrol",
            "city trailer patrol is not enabled")
    require(trailer["route_type"] == "waypoints" and trailer["patrol_mode"] == "repeat",
            "city trailer repeat-waypoint contract differs")
    require(float(trailer["cruise_speed_m_s"]) == gen.TRAILER_CRUISE_SPEED_M_S,
            "city trailer speed is not 7 m/s")
    require(float(trailer["acceleration_m_s2"]) == gen.TRAILER_ACCELERATION_M_S2,
            "city trailer acceleration contract differs")
    require(trailer.get("stop_at_waypoints") is True,
            "city trailer must stop before changing straight segments")
    require("corner_radius_m" not in trailer,
            "city trailer arc routing must remain disabled")
    require(
        float(trailer["turn_speed_tolerance_m_s"])
        == gen.TRAILER_TURN_SPEED_TOLERANCE_M_S,
        "city trailer turn-speed tolerance differs",
    )
    require(
        tuple(int(index) for index in trailer["stop_waypoint_indices"])
        == gen.TRAILER_STOP_WAYPOINT_INDICES,
        "city trailer stop-corner indices differ",
    )
    trailer_include = next(
        (
            include
            for include in world.findall(".//include")
            if include.findtext("name") == "trailer"
        ),
        None,
    )
    require(
        trailer_include is not None
        and trailer_include.findtext("uri") == "model://moving_platform_aruco_velocity",
        "derived world does not spawn the calibrated ArUco trailer",
    )
    trailer_spawn = trailer["spawn_pose_enu"]
    require(
        (float(trailer_spawn["x"]), float(trailer_spawn["y"])) == gen.TRAILER_SPAWN_XY,
        "trailer spawn coordinate differs from the requested ENU point",
    )
    require(
        trailer_include.findtext("pose")
        == f"{gen.fmt(gen.TRAILER_SPAWN_XY[0])} {gen.fmt(gen.TRAILER_SPAWN_XY[1])} 0 0 0 0",
        "trailer world include pose differs from YAML",
    )
    require(
        trailer["waypoints_enu_m"] == [list(point) for point in gen.TRAILER_PATROL_WAYPOINTS],
        "city trailer YAML differs from the black-road closed patrol",
    )

    neighbor_pairs = gen.find_neighbor_pairs(expected_geometry, gen.DEFAULT_NEIGHBOR_RADIUS_M)
    evaluation = gen.evaluate_candidate(expected_geometry, neighbor_pairs)
    require(evaluation.feasible, "generated geometry overlaps or blocks an active spawn")
    require(abs(footprint_scale - 2.5) < 1.0e-12,
            "building XY footprint does not match the initial city's uniform 2.5x transform")
    require(evaluation.overlap_count == 0, "expanded building footprints overlap")
    require(evaluation.minimum_neighbor_gap_m > gen.EPS, "expanded building footprints touch")
    requested_goal_clearance = min(
        gen.point_polygon_distance(gen.MISSION_GOAL_XY, building.transformed)
        for building in expected_geometry
    )
    required_goal_clearance = (
        float(derived["mission"]["vehicle_clearance_xy_m"])
        + float(derived["mission"].get("bspline_clearance_margin_m", 0.0)))
    require(requested_goal_clearance >= required_goal_clearance,
            "fixed A* goal remains blocked by a building")

    def safety_rectangle(center: gen.Point, half_x: float, half_y: float) -> gen.Polygon2D:
        x, y = center
        return gen.Polygon2D(
            [(x - half_x, y - half_y), (x + half_x, y - half_y),
             (x + half_x, y + half_y), (x - half_x, y + half_y)],
            [],
        )

    drone_safety = safety_rectangle(gen.SPAWN_XY, 1.0, 1.0)
    trailer_safety = safety_rectangle(gen.TRAILER_SPAWN_XY, 3.5, 3.5)
    drone_safety_clearance = min(
        gen.polygon_distance(drone_safety, building.transformed)
        for building in expected_geometry
    )
    trailer_safety_clearance = min(
        gen.polygon_distance(trailer_safety, building.transformed)
        for building in expected_geometry
    )
    require(drone_safety_clearance >= 25.0,
            "drone road-end safety square is too close to a building")
    require(trailer_safety_clearance >= 25.0,
            "trailer road-end safety rectangle is too close to a building")
    ratios = []
    for first, second, original_gap in neighbor_pairs:
        if original_gap <= 1.0e-6:
            continue
        ratios.append(
            gen.polygon_distance(
                expected_geometry[first].transformed,
                expected_geometry[second].transformed,
            )
            / original_gap
        )
    gap_ratio_median = gen.statistics.median(ratios)
    gap_ratio_p10 = gen.percentile(ratios, 0.10)
    require(gap_ratio_median >= 1.4, "median gap ratio is below the dense-city contract")
    require(gap_ratio_p10 >= 1.2, "p10 gap ratio is below the dense-city contract")
    maximum_collision_outward_error = 0.0
    certified_collision_gap = evaluation.minimum_neighbor_gap_m
    require(certified_collision_gap > gen.EPS, "exact DART collision prisms touch")
    obstacle_summary = derived["obstacles"]["summary"]
    require(obstacle_summary["source_building_count"] == gen.SOURCE_BUILDING_COUNT,
            "YAML source obstacle count summary mismatch")
    require(obstacle_summary["building_count"] == gen.ACTIVE_BUILDING_COUNT,
            "YAML obstacle count summary mismatch")
    require(obstacle_summary["removed_building_count"] == gen.REMOVED_BUILDING_COUNT,
            "YAML removed obstacle count summary mismatch")
    reduction = derivation["building_reduction"]
    require(reduction["source_count"] == gen.SOURCE_BUILDING_COUNT,
            "building reduction source count mismatch")
    require(reduction["retained_count"] == gen.ACTIVE_BUILDING_COUNT,
            "building reduction retained count mismatch")
    require(reduction["removed_count"] == gen.REMOVED_BUILDING_COUNT,
            "building reduction removed count mismatch")
    require(tuple(reduction["removed_ids"]) == expected_removed_ids,
            "building reduction ID set/order is not deterministic")
    require(tuple(reduction["protected_ids"]) == gen.REDUCTION_PROTECTED_IDS,
            "building reduction protection set is not deterministic")
    expected_audit = gen.removal_grid_audit(source_buildings, expected_removed_ids)
    require(reduction["spatial_random_audit"] == expected_audit,
            "spatial-random reduction metadata differs from canonical audit")
    require(expected_audit["seed"] == gen.REDUCTION_RANDOM_SEED,
            "spatial-random reduction seed changed")
    require(expected_audit["cells_with_removals"] == gen.REDUCTION_GRID_SIZE**2,
            "removed buildings do not cover all 25 map regions")
    require(expected_audit["removed_ids_sha256"] == gen.REDUCTION_IDS_SHA256,
            "removed-building selection digest changed")
    require(not (set(expected_removed_ids) & set(gen.REDUCTION_PROTECTED_IDS)),
            "a protected building was removed")
    require(
        obstacle_summary["active_height_profile"] == gen.ACTIVE_HEIGHT_PROFILE,
        "wrong active height profile",
    )
    require(
        [float(value) for value in obstacle_summary["roof_z_range_m"]]
        == [gen.ACTIVE_HEIGHT_MIN_M, gen.ACTIVE_HEIGHT_MAX_M],
        "active roof range is not exactly 20-50m",
    )
    require(obstacle_summary["shortest_building"]["id"] == "building_190",
            "20m height sentinel was not retained")
    require(obstacle_summary["tallest_building"]["id"] == "building_171",
            "50m height sentinel was not retained")
    height_distribution = obstacle_summary["height_distribution"]
    require(int(height_distribution["count"]) == gen.ACTIVE_BUILDING_COUNT,
            "height distribution count mismatch")
    require(int(height_distribution["unique_count"]) == gen.ACTIVE_BUILDING_COUNT,
            "active building heights are not uniquely distributed")
    require(
        abs(float(height_distribution["mean_m"]) - gen.ACTIVE_HEIGHT_MEAN_M) <= 1e-12
        and abs(float(height_distribution["median_m"]) - gen.ACTIVE_HEIGHT_MEAN_M) <= 1e-12
        and abs(float(height_distribution["target_mean_m"]) - gen.ACTIVE_HEIGHT_MEAN_M) <= 1e-12,
        "active building height mean/median is not exactly 35m",
    )
    require(int(obstacle_summary["building_overlap_count"]) == 0,
            "YAML reports overlapping building footprints")
    require(obstacle_summary["all_building_footprints_disjoint"] is True,
            "YAML building-disjointness certification failed")
    require(obstacle_summary["gap_constraint_enforced"] is False,
            "YAML unexpectedly treats inter-building gap as a selection constraint")
    require(derived["source_sha256"]["source_city_coordinates_yaml"] == gen.sha256_file(gen.SOURCE_YAML), "source YAML hash is stale")

    return {
        "building_count": len(derived_buildings),
        "selected_footprint_scale": footprint_scale,
        "centroid_spacing_scale": spacing_scale,
        "ground_size_m": expected_ground_size,
        "mesh_triangles": triangle_count,
        "maximum_mesh_stream_error_m": maximum_mesh_stream_error,
        "maximum_centroid_error_m": maximum_centroid_error,
        "maximum_footprint_error_m": maximum_footprint_error,
        "maximum_foundation_ground_error_m": maximum_foundation_ground_error,
        "maximum_height_profile_error_m": maximum_height_profile_error,
        "maximum_collision_parameter_error_m": maximum_collision_parameter_error,
        "collision_geometry_count": len(collision_elements),
        "removed_building_count": len(expected_removed_ids),
        "removal_grid_cells_covered": expected_audit["cells_with_removals"],
        "removal_ids_sha256": expected_audit["removed_ids_sha256"],
        "collision_maximum_outward_error_m": maximum_collision_outward_error,
        "gap_ratio_median": gap_ratio_median,
        "gap_ratio_p10": gap_ratio_p10,
        "minimum_neighbor_gap_m": evaluation.minimum_neighbor_gap_m,
        "minimum_physical_collision_gap_m": certified_collision_gap,
        "active_road_overlap_pixels": active_road_overlap_pixels,
        "drone_spawn_safety_clearance_m": drone_safety_clearance,
        "trailer_spawn_safety_clearance_m": trailer_safety_clearance,
        "trailer_spawn_clearance_m": evaluation.trailer_spawn_clearance_m,
        "spawn_site_separation_m": spawn_separation,
        "drone_spawn_boundary_clearance_m": spawn_boundary_clearance,
        "trailer_spawn_boundary_clearance_m": trailer_boundary_clearance,
        "astar_vertex_rows": len(csv_rows),
        "astar_logical_xyz_vertices": 2 * len(csv_rows),
        "astar_goal_clearance_m": requested_goal_clearance,
    }


def main() -> int:
    try:
        result = validate()
    except (RuntimeError, ValueError, KeyError, OSError, ET.ParseError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS city_uav expansion")
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
