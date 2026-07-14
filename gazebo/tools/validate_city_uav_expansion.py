#!/usr/bin/env python3
"""Fail-closed validation for the generated Apple Park UAV city assets."""

from __future__ import annotations

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

    maximum_z_error = 0.0
    maximum_centroid_error = 0.0
    maximum_footprint_error = 0.0
    expected_boundary_xyz: list[tuple[float, float, float]] = []
    for source_record, derived_record in zip(active_source_buildings, derived_buildings):
        for key in ("foundation_z_m", "ground_reference_z_m", "roof_z_m", "height_above_ground_m"):
            error = abs(float(source_record[key]) - float(derived_record[key]))
            maximum_z_error = max(maximum_z_error, error)
            require(error <= TOLERANCE_Z_M, f"{derived_record['id']} changed {key} by {error}")

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
    require(building_model.findtext("static") == "true", "shared concave building mesh must be static")
    collision_elements = building_model.findall("link/collision")
    require(len(collision_elements) == 1, "building world must contain one shared collision geometry")
    collision = collision_elements[0]
    require(collision.attrib.get("name") == "buildings_exact_shared_dae", "wrong shared collision name")
    collision_uri = collision.findtext("geometry/mesh/uri")
    collision_scale = collision.findtext("geometry/mesh/scale")
    require(collision_uri == visual_uri == "mesh/buildings_uav.dae", "visual/collision DAE URI differs")
    require(collision_scale == visual_scale == "1 1 1", "visual/collision DAE scale differs")
    collision_metadata = derivation["collision_geometry"]
    require(collision_metadata["type"] == gen.COLLISION_GEOMETRY_TYPE, "collision metadata type mismatch")
    require(int(collision_metadata["count"]) == 1, "collision metadata count mismatch")
    require(float(collision_metadata["maximum_outward_error_m"]) == 0.0, "collision metadata is not exact")
    require(float(collision_metadata["maximum_undercoverage_m"]) == 0.0, "collision metadata undercovers")
    require(not world.findall(".//collision/geometry/polyline"), "legacy polyline collision remains in world")
    require(
        world.find(".//model[@name='applepark_uav_building_collision_proxies']") is None,
        "legacy collision proxy model remains in world",
    )

    expected_geometry = gen.transform_buildings(active_source_buildings, spacing_scale, footprint_scale, anchor)
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
    maximum_collision_parameter_error = 0.0

    bounds = derived["map"]["bounds_enu_m"]
    expected_ground_size = float(bounds["x"][1]) - float(bounds["x"][0])
    require(abs(expected_ground_size - 1260.0) <= 1.0e-9, "rolled-back UAV city ground must be 1260m")
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
    road = Image.open(gen.OUTPUT_ROAD).convert("RGB")
    require(road.size == (2048, 2048), "road texture dimensions changed")

    xmin, xmax = map(float, bounds["x"])
    ymin, ymax = map(float, bounds["y"])

    def road_rgb(point: gen.Point) -> tuple[int, int, int]:
        column = round((point[0] - xmin) / (xmax - xmin) * (road.width - 1))
        row = round((ymax - point[1]) / (ymax - ymin) * (road.height - 1))
        require(0 <= column < road.width and 0 <= row < road.height,
                f"road sample is outside the map: {point}")
        return road.getpixel((column, row))

    def is_asphalt(point: gen.Point) -> bool:
        red, green, blue = road_rgb(point)
        # Main asphalt is the connected dark neutral family around (58,63,68).
        # The out-of-source neutral fill is (95,98,96), so a loose grayscale
        # threshold would incorrectly certify off-map-looking pavement.
        return red <= 75 and green <= 80 and blue <= 85 and max(red, green, blue) - min(red, green, blue) < 28

    # The drone gets a 2 x 2 m asphalt square. The trailer check includes a
    # 1 m halo around its 5.5 x 3 m yaw-aligned body (7.5 x 5 m total).
    for x_step in range(-4, 5):
        for y_step in range(-4, 5):
            point = (gen.SPAWN_XY[0] + 0.25 * x_step, gen.SPAWN_XY[1] + 0.25 * y_step)
            require(is_asphalt(point), f"drone spawn safety square leaves asphalt at {point}")
    for x_step in range(-15, 16):
        for y_step in range(-10, 11):
            point = (
                gen.TRAILER_SPAWN_XY[0] + 0.25 * x_step,
                gen.TRAILER_SPAWN_XY[1] + 0.25 * y_step,
            )
            require(is_asphalt(point), f"trailer spawn safety rectangle leaves asphalt at {point}")
    require(gen.SPAWN_XY[0] * gen.TRAILER_SPAWN_XY[0] < 0.0,
            "drone and trailer are not in opposite east/west halves")
    require(gen.SPAWN_XY[1] * gen.TRAILER_SPAWN_XY[1] < 0.0,
            "drone and trailer are not in opposite north/south halves")
    spawn_separation = gen.math.dist(gen.SPAWN_XY, gen.TRAILER_SPAWN_XY)
    require(spawn_separation >= 1500.0, "drone/trailer diagonal separation is below 1.5km")
    spawn_boundary_clearance = min(
        gen.SPAWN_XY[0] - xmin, xmax - gen.SPAWN_XY[0],
        gen.SPAWN_XY[1] - ymin, ymax - gen.SPAWN_XY[1],
    )
    trailer_boundary_clearance = min(
        gen.TRAILER_SPAWN_XY[0] - xmin, xmax - gen.TRAILER_SPAWN_XY[0],
        gen.TRAILER_SPAWN_XY[1] - ymin, ymax - gen.TRAILER_SPAWN_XY[1],
    )
    require(spawn_boundary_clearance >= 40.0 and trailer_boundary_clearance >= 40.0,
            "a diagonal staging site is too close to the map boundary")

    spawn = derived["spawn"]["gazebo_spawn_pose_enu"]
    require((float(spawn["x"]), float(spawn["y"])) == gen.SPAWN_XY, "spawn coordinate was scaled")
    require(float(spawn["z"]) == 0.0, "derived PX4 spawn must reference flat ground z=0")
    require(
        derived["frames"]["px4_local"]["origin_enu_m"]
        == [gen.SPAWN_XY[0], gen.SPAWN_XY[1], gen.PX4_BASE_LINK_Z_OFFSET_M],
        "PX4 local origin must use x500 base_link height, not model-root z",
    )
    require("pad" not in derived["spawn"], "derived YAML still declares a spawn pad")
    require(world.find(".//model[@name='drone_spawn_pad']") is None, "green spawn pad remains in UAV world")
    spawn_frame = world.find(".//frame[@name='drone_spawn']")
    require(
        spawn_frame is not None
        and spawn_frame.findtext("pose")
        == f"{gen.fmt(gen.SPAWN_XY[0])} {gen.fmt(gen.SPAWN_XY[1])} 0 0 0 0",
        "spawn frame is not at the diagonal road site on z=0",
    )
    trailer = derived["trailer"]
    require(trailer["entity_name"] == "trailer", "derived trailer entity contract changed")
    require(trailer["model_uri"] == "model://trailer_aruco", "ArUco trailer model is not selected")
    require(trailer["command_topic"] == "/model/trailer/cmd_vel", "trailer command topic mismatch")
    require(trailer["motion"] == "stationary_spawn_only", "city trailer must remain stationary")
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
        (float(trailer_spawn["x"]), float(trailer_spawn["y"])) == gen.TRAILER_SPAWN_XY,
        "trailer spawn coordinate differs from the requested ENU point",
    )
    require(
        trailer_include.findtext("pose")
        == f"{gen.fmt(gen.TRAILER_SPAWN_XY[0])} {gen.fmt(gen.TRAILER_SPAWN_XY[1])} 0 0 0 0",
        "trailer world include pose differs from YAML",
    )
    require(
        trailer["waypoints_enu_m"] == [list(gen.TRAILER_SPAWN_XY)],
        "stationary trailer YAML must contain only its spawn point",
    )

    neighbor_pairs = gen.find_neighbor_pairs(expected_geometry, gen.DEFAULT_NEIGHBOR_RADIUS_M)
    evaluation = gen.evaluate_candidate(expected_geometry, neighbor_pairs)
    require(evaluation.feasible, "generated geometry fails its hard clearance constraints")
    require(abs(footprint_scale - 0.9) < 1.0e-12, "building XY footprint is not rolled back to jo 0.9x")
    require(evaluation.minimum_neighbor_gap_m >= 2.0, "a building gap is narrower than 2m")
    require(evaluation.spawn_clearance_m >= 30.0,
            "drone road-end staging site is too close to a building")
    require(evaluation.trailer_spawn_clearance_m >= 40.0,
            "trailer road-end staging site is too close to a building")
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
    require(certified_collision_gap >= 2.0, "shared exact DAE collision closes a required 2m corridor")
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
    require([float(value) for value in obstacle_summary["roof_z_range_m"]] == [10.0, 20.0],
            "active roof range is not exactly 10-20m")
    require(obstacle_summary["shortest_building"]["id"] == "building_190",
            "10m height sentinel was not retained")
    require(obstacle_summary["tallest_building"]["id"] == "building_171",
            "20m height sentinel was not retained")
    require(float(obstacle_summary["minimum_building_gap_m"]) >= 2.0, "YAML reports a sub-2m gap")
    require(obstacle_summary["all_building_gaps_meet_requirement"] is True, "YAML gap certification failed")
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
        "maximum_z_error_m": maximum_z_error,
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
        "trailer_spawn_clearance_m": evaluation.trailer_spawn_clearance_m,
        "spawn_site_separation_m": spawn_separation,
        "drone_spawn_boundary_clearance_m": spawn_boundary_clearance,
        "trailer_spawn_boundary_clearance_m": trailer_boundary_clearance,
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
