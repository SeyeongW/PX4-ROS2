#!/usr/bin/env python3
"""Fail-closed validation for the generated Apple Park UAV city assets."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from PIL import Image

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
    require(normals is not None and normals.text, "DART-safe DAE normals are missing")
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
        gen.REPORT_DIR / "building_transform.csv",
        gen.REPORT_DIR / "pairwise_gap_before_after.csv",
        gen.REPORT_DIR / "invalid_or_tight_corridors.csv",
        gen.REPORT_DIR / "collision_proxy_alignment.csv",
        gen.REPORT_DIR / "map_bounds_summary.json",
        gen.REPORT_DIR / "visual_collision_alignment.md",
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
    require(len(source_buildings) == 274, "source building count changed")
    require(len(derived_buildings) == 274, "derived city must contain 274 buildings")
    require(
        [building["id"] for building in source_buildings]
        == [building["id"] for building in derived_buildings],
        "derived building IDs/order differ from source",
    )

    derivation = derived["derivation"]
    spacing_scale = float(derivation["city_spacing_scale_xy"])
    footprint_scale = float(derivation["building_footprint_scale_xy"])
    anchor = tuple(float(value) for value in derivation["anchor_enu_m"])
    require(abs(spacing_scale - 2.5) <= 1.0e-12, "default derived city is not spaced by 2.5")
    require(footprint_scale in gen.FOOTPRINT_SCALE_CANDIDATES, "footprint scale is not an allowed candidate")

    maximum_z_error = 0.0
    maximum_centroid_error = 0.0
    maximum_footprint_error = 0.0
    expected_boundary_xyz: list[tuple[float, float, float]] = []
    for source_record, derived_record in zip(source_buildings, derived_buildings):
        for key in ("foundation_z_m", "ground_reference_z_m", "roof_z_m", "height_above_ground_m"):
            error = abs(float(source_record[key]) - float(derived_record[key]))
            maximum_z_error = max(maximum_z_error, error)
            require(error <= TOLERANCE_Z_M, f"{derived_record['id']} changed {key} by {error}")

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

    positions, triangle_count = load_dae_positions(gen.OUTPUT_DAE)
    expected_keys = {
        (round(x, 6), round(y, 6), round(z, 9))
        for x, y, z in expected_boundary_xyz
    }
    mesh_keys = {(round(x, 6), round(y, 6), round(z, 9)) for x, y, z in positions}
    missing_from_mesh = expected_keys - mesh_keys
    extra_mesh_vertices = mesh_keys - expected_keys
    require(not missing_from_mesh, f"{len(missing_from_mesh)} YAML boundary vertices are absent from DAE")
    require(not extra_mesh_vertices, f"{len(extra_mesh_vertices)} DAE vertices do not come from YAML boundaries")

    world_root = ET.parse(gen.OUTPUT_WORLD).getroot()
    world = world_root.find("world")
    require(world is not None and world.attrib.get("name") == "applepark_city_uav", "wrong Gazebo world name")
    visual_uri = world.findtext(".//model[@name='applepark_uav_buildings']/link/visual/geometry/mesh/uri")
    require(visual_uri == "mesh/buildings_uav.dae", "visual does not reference generated DAE")
    visual_scale = world.findtext(".//model[@name='applepark_uav_buildings']/link/visual/geometry/mesh/scale")
    require(visual_scale == "1 1 1", "world-level building scale is forbidden")
    require(not world.findall(".//collision/geometry/mesh"), "DART-incompatible mesh collision remains in world")

    expected_geometry = gen.transform_buildings(source_buildings, spacing_scale, footprint_scale, anchor)
    expected_proxies = gen.generate_collision_proxies(expected_geometry)
    proxy_elements = world.findall(
        ".//model[@name='applepark_uav_building_collision_proxies']/link/collision"
    )
    require(len(proxy_elements) == len(expected_proxies), "world collision proxy count differs from generator")
    maximum_proxy_parameter_error = 0.0
    for expected_proxy, element in zip(expected_proxies, proxy_elements):
        expected_name = f"{expected_proxy.building_id}_part_{expected_proxy.part_index:03d}"
        require(element.attrib.get("name") == expected_name, f"collision proxy order/name mismatch: {expected_name}")
        pose_text = element.findtext("pose")
        size_text = element.findtext("geometry/box/size")
        require(pose_text is not None and size_text is not None, f"{expected_name} is not a box proxy")
        actual_pose = [float(value) for value in pose_text.split()]
        actual_size = [float(value) for value in size_text.split()]
        expected_pose = [
            expected_proxy.center_xy[0],
            expected_proxy.center_xy[1],
            expected_proxy.center_z,
            0.0,
            0.0,
            expected_proxy.yaw_rad,
        ]
        expected_size = [expected_proxy.size_xy[0], expected_proxy.size_xy[1], expected_proxy.size_z]
        parameter_error = max(
            [abs(actual - expected) for actual, expected in zip(actual_pose, expected_pose)]
            + [abs(actual - expected) for actual, expected in zip(actual_size, expected_size)]
        )
        maximum_proxy_parameter_error = max(maximum_proxy_parameter_error, parameter_error)
        require(parameter_error <= TOLERANCE_XY_M, f"{expected_name} world proxy differs from generated geometry")

        cosine, sine = math.cos(actual_pose[5]), math.sin(actual_pose[5])
        for point in expected_proxy.source_triangle:
            dx, dy = point[0] - actual_pose[0], point[1] - actual_pose[1]
            local_x = dx * cosine + dy * sine
            local_y = -dx * sine + dy * cosine
            require(abs(local_x) <= 0.5 * actual_size[0] + 1.0e-8, f"{expected_name} under-covers source X")
            require(abs(local_y) <= 0.5 * actual_size[1] + 1.0e-8, f"{expected_name} under-covers source Y")
        require(
            expected_proxy.maximum_outward_error_m <= gen.MAX_COLLISION_PROXY_OUTWARD_ERROR_M + 1.0e-6,
            f"{expected_name} exceeds conservative proxy error budget",
        )

    bounds = derived["map"]["bounds_enu_m"]
    expected_ground_size = float(bounds["x"][1]) - float(bounds["x"][0])
    require(abs(expected_ground_size - 1260.0) <= 1.0e-9, "default expanded ground must be 1260m")
    box_size_text = world.findtext(".//model[@name='applepark_uav_ground']/link/collision/geometry/box/size")
    require(box_size_text is not None, "ground collision box is missing")
    box_size = [float(value) for value in box_size_text.split()]
    require(box_size == [expected_ground_size, expected_ground_size, 0.1], "ground collision size differs from YAML")
    ground_pose_text = world.findtext(".//model[@name='applepark_uav_ground']/link/collision/pose")
    require(ground_pose_text is not None and float(ground_pose_text.split()[2]) == -0.05, "ground top is not z=0")

    height = Image.open(gen.OUTPUT_HEIGHT)
    require(height.size == (257, 257), "flat heightmap dimensions changed")
    require(height.getextrema() == (255, 255), "heightmap is not completely flat")
    require(Image.open(gen.OUTPUT_NORMAL).size == (257, 257), "normal map dimensions changed")
    require(Image.open(gen.OUTPUT_ROAD).size == (2048, 2048), "road texture dimensions changed")

    spawn = derived["spawn"]["gazebo_spawn_pose_enu"]
    require((float(spawn["x"]), float(spawn["y"])) == gen.SPAWN_XY, "spawn coordinate was scaled")
    require(float(spawn["z"]) == 0.0, "derived PX4 spawn must reference flat ground z=0")
    require(
        derived["frames"]["px4_local"]["origin_enu_m"] == [-120.0, 115.0, 0.24],
        "PX4 local origin must use x500 base_link height, not model-root z",
    )
    require("pad" not in derived["spawn"], "derived YAML still declares a spawn pad")
    require(world.find(".//model[@name='drone_spawn_pad']") is None, "green spawn pad remains in UAV world")
    spawn_frame = world.find(".//frame[@name='drone_spawn']")
    require(spawn_frame is not None and spawn_frame.findtext("pose") == "-120 115 0 0 0 0", "spawn frame is not z=0")
    trailer = derived["trailer"]
    require(trailer["entity_name"] == "trailer", "derived trailer entity contract changed")
    require(trailer["model_uri"] == "model://trailer_aruco", "ArUco trailer model is not selected")
    require(trailer["command_topic"] == "/model/trailer/cmd_vel", "trailer command topic mismatch")
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
        and trailer_include.findtext("uri") == "model://trailer_aruco",
        "derived world does not spawn the calibrated ArUco trailer",
    )
    trailer_spawn = trailer["spawn_pose_enu"]
    require(
        (float(trailer_spawn["x"]), float(trailer_spawn["y"])) == gen.MISSION_GOAL_XY,
        "trailer/mission goal coordinate was scaled",
    )
    require(tuple(map(float, trailer["destination_enu_m"])) == gen.TRAILER_DESTINATION_XY, "trailer endpoint changed")

    summary = json.loads((gen.REPORT_DIR / "map_bounds_summary.json").read_text(encoding="utf-8"))
    require(summary["source_building_count"] == summary["generated_building_count"] == 274, "summary count mismatch")
    require(summary["selected_layout_feasible"] is True, "generator selected an infeasible layout")
    require(summary["tight_or_invalid_corridor_count"] == 0, "tight/invalid corridor report is not empty")
    require(summary["pairwise_gap_ratio"]["median"] >= 2.5, "median gap ratio is below 2.5")
    require(summary["pairwise_gap_ratio"]["p10"] >= 2.0, "p10 gap ratio is below 2.0")
    require(summary["minimum_neighbor_gap_m"] >= 2.0 * gen.R_HARD_M, "hard-width neighbor corridor failed")
    require(summary["mesh"]["sha256"] == gen.sha256_file(gen.OUTPUT_DAE), "reported DAE hash is stale")
    require(summary["mesh"]["triangles"] == triangle_count, "reported DAE triangle count is stale")
    require(summary["collision_proxy"]["count"] == len(expected_proxies), "reported proxy count is stale")
    require(summary["collision_proxy"]["maximum_undercoverage_m"] == 0.0, "collision proxy undercoverage reported")
    require(
        summary["collision_proxy"]["maximum_outward_error_m"]
        <= gen.MAX_COLLISION_PROXY_OUTWARD_ERROR_M + 1.0e-6,
        "collision proxy outward error exceeds budget",
    )
    require(
        summary["collision_proxy"]["certified_minimum_neighbor_gap_m"] >= 2.0 * gen.R_HARD_M,
        "conservative collision proxies close a hard-width building corridor",
    )
    require(
        summary["collision_proxy"]["mission_clearance_m"]["trailer_route"]
        >= gen.TRAILER_HALF_WIDTH_M + gen.R_HARD_M + gen.TRAILER_ROUTE_MARGIN_M,
        "collision proxies block the swept trailer route",
    )

    with (gen.REPORT_DIR / "invalid_or_tight_corridors.csv").open(newline="", encoding="utf-8") as stream:
        invalid_rows = list(csv.DictReader(stream))
    require(not invalid_rows, "invalid_or_tight_corridors.csv contains failures")
    require(derived["source_sha256"]["source_city_coordinates_yaml"] == gen.sha256_file(gen.SOURCE_YAML), "source YAML hash is stale")

    return {
        "building_count": len(derived_buildings),
        "selected_footprint_scale": footprint_scale,
        "centroid_spacing_scale": spacing_scale,
        "ground_size_m": expected_ground_size,
        "mesh_triangles": triangle_count,
        "maximum_centroid_error_m": maximum_centroid_error,
        "maximum_footprint_error_m": maximum_footprint_error,
        "maximum_z_error_m": maximum_z_error,
        "maximum_proxy_parameter_error_m": maximum_proxy_parameter_error,
        "collision_proxy_count": len(expected_proxies),
        "collision_proxy_maximum_outward_error_m": summary["collision_proxy"]["maximum_outward_error_m"],
        "gap_ratio_median": summary["pairwise_gap_ratio"]["median"],
        "gap_ratio_p10": summary["pairwise_gap_ratio"]["p10"],
        "minimum_neighbor_gap_m": summary["minimum_neighbor_gap_m"],
        "trailer_route_clearance_m": summary["mission_clearance_m"]["trailer_route"],
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
