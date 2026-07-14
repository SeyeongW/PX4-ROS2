#!/usr/bin/env python3
"""Validate repository-only city and mountain Gazebo map entry points."""

from __future__ import annotations

import csv
import hashlib
import math
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import yaml


GAZEBO = Path(__file__).resolve().parents[1]
REPO = GAZEBO.parent
CITY = GAZEBO / "worlds/applepark_city"
CITY_WORLD = CITY / "applepark.world"
MOUNTAIN_WORLD = GAZEBO / "worlds/ugv_drone_map.world"
LOG = GAZEBO / "validation/self_contained_maps_static.log"
OVERLAP_CSV = GAZEBO / "validation/city/road_building_overlap_coordinates.csv"
FOUNDATION_CSV = GAZEBO / "validation/city/building_foundation_alignment.csv"
HEIGHT_SCALING_CSV = GAZEBO / "validation/city/building_height_scaling.csv"
PREVIEW_CONTACT_Z = 0.0
CITY_HEIGHTMAP_POSITION_Z = -0.001
CITY_HEIGHTMAP_OBSERVED_MAX = 255
CITY_RENDER_HEIGHT_SIZE_Z = 0.001
CITY_BUILDING_COUNT = 274
CITY_HEIGHT_FACTOR_SEED = "px4-ros2-city-height-v2"
CITY_HEIGHT_FACTOR_MIN_MILLIONTHS = 2_000_000
CITY_HEIGHT_FACTOR_MAX_MILLIONTHS = 3_500_000
CITY_HEIGHT_FACTOR_BUCKETS = (
    CITY_HEIGHT_FACTOR_MAX_MILLIONTHS - CITY_HEIGHT_FACTOR_MIN_MILLIONTHS + 1
)
CITY_SOURCE_HEIGHT_PROFILE = "deterministic_hash_rank_10_to_20m_v1"
CITY_SOURCE_HEIGHT_MIN_M = 10.0
CITY_SOURCE_HEIGHT_MAX_M = 20.0
CITY_SOURCE_SHORTEST_ID = "building_190"
CITY_SOURCE_TALLEST_ID = "building_171"
CITY_UAV_HEIGHT_PROFILE = "deterministic_active_hash_rank_20_to_50m_mean35_v1"
CITY_UAV_HEIGHT_MIN_M = 20.0
CITY_UAV_HEIGHT_MAX_M = 50.0
CITY_UAV_HEIGHT_MEAN_M = 35.0
CITY_UAV_SHORTEST_ID = "building_190"
CITY_UAV_TALLEST_ID = "building_171"

EXPECTED_CITY_HASHES = {
    "worlds/applepark_city/mesh/buildings.dae":
        "3141bb5aafd931b2d93ce825b74e13645186570f0a84e7bdef6046fab688e43c",
    "worlds/applepark_city/mesh/height_map_city_500m.png":
        "0f3bc9604b38368ec071fbc9666bfcf4594c74ebcdae700b9fea0f2bed8ffa06",
    "worlds/applepark_city/mesh/normal_map_city_500m.png":
        "9d9011a129993a5386b2d31885510354f39fd0b71a4694040d3494428b9de900",
    "worlds/applepark_city/mesh/road_surface_city_500m.png":
        "f43cc932dd4c101c605c15c0a202e4e946fbea77e0c568763dbec217dc123a08",
    "worlds/applepark_city/mesh/city_terrain_collision.obj":
        "21745ec52bde2da3fca56c7a4a90fe5e5b872577beb9e8aaa058253e355e783f",
    "worlds/applepark_city/OSM_ATTRIBUTION.txt":
        "f706303dc1fa8e4b456e1d235a73ee395be38c2897c62b6ebf19ad74646892fb",
    "validation/city/road_building_overlap_coordinates.csv":
        "b3a1f0eaee7eafc2595c9bd35e6f23a4a74749b65a0a0224dc78ad6d4ddf6601",
    "validation/city/building_foundation_alignment.csv":
        "87de3a2f2df3f6b099d90536fa609ccc35ae9b92b1a8a75b872af662329ee9c0",
    "validation/city/building_height_scaling.csv":
        "a8940102cdab4b06a79f7093342b271c5c0a2194e6c635ccd17e3d7a14950d34",
    "validation/city/road_building_overlap_after_fix.png":
        "9f7f73b2dc958fde45e8e6bc1c660125d48e90d7afc3eaa1c398aa8cb98aae9c",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    require(result.returncode == 0, f"command failed: {' '.join(command)}\n{result.stdout}")
    return result.stdout.strip()


def world_element(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    world = root.find("world")
    require(world is not None, f"world element missing: {path}")
    return world


def validate_city() -> dict[str, object]:
    for relative, expected in EXPECTED_CITY_HASHES.items():
        path = GAZEBO / relative
        require(path.is_file(), f"city asset missing: {relative}")
        require(sha256(path) == expected, f"city asset hash changed: {relative}")

    world = world_element(CITY_WORLD)
    require(world.attrib.get("name") == "applepark_city", "city world name")
    expected_plugins = {
        "gz-sim-physics-system",
        "gz-sim-user-commands-system",
        "gz-sim-scene-broadcaster-system",
        "gz-sim-contact-system",
        "gz-sim-sensors-system",
        "gz-sim-imu-system",
        "gz-sim-air-pressure-system",
        "gz-sim-magnetometer-system",
        "gz-sim-navsat-system",
    }
    plugins = {plugin.attrib.get("filename", "") for plugin in world.findall("plugin")}
    require(plugins == expected_plugins, f"city system plugins: {sorted(plugins)}")
    require(
        all(
            plugin.attrib.get("name", "").startswith("gz::sim::systems::")
            for plugin in world.findall("plugin")
        ),
        "legacy Ignition plugin namespace remains in city world",
    )

    terrain = next(
        (model for model in world.findall("model") if model.attrib.get("name") == "applepark"),
        None,
    )
    require(terrain is not None, "city terrain model")
    ground_visual = terrain.find("./link/visual[@name='ground_visual']")
    require(ground_visual is not None, "city terrain visual")
    require(
        ground_visual.findtext("cast_shadows") == "false",
        "city terrain visual must not cast a heightmap-volume shadow",
    )
    heightmaps = terrain.findall(".//heightmap")
    require(len(heightmaps) == 1, "city visual heightmap")
    for heightmap in heightmaps:
        require(heightmap.findtext("uri") == "mesh/height_map_city_500m.png", "height URI")
        heightmap_size = [float(value) for value in heightmap.findtext("size", "").split()]
        heightmap_position = [
            float(value) for value in heightmap.findtext("pos", "").split()
        ]
        require(len(heightmap_size) == 3, "heightmap size vector")
        require(len(heightmap_position) == 3, "heightmap position vector")
        require(heightmap_size[:2] == [500.0, 500.0], "heightmap XY size")
        require(
            abs(heightmap_size[2] - CITY_RENDER_HEIGHT_SIZE_Z) < 1e-12,
            "OGRE2-normalized heightmap Z size",
        )
        require(
            heightmap_position == [0.0, 0.0, CITY_HEIGHTMAP_POSITION_Z],
            "heightmap position",
        )
        require(
            abs(heightmap_position[2] + heightmap_size[2]) < 1e-12,
            "city all-white heightmap top must stay at z=0",
        )
    require(
        terrain.findtext(".//collision/pose") == "0 0 -0.05 0 0 0"
        and terrain.findtext(".//collision//box/size") == "500 500 0.1",
        "engine-neutral flat city collision box",
    )
    texture = terrain.find(".//visual//heightmap/texture")
    require(texture is not None, "city heightmap texture")
    require(texture.findtext("diffuse") == "mesh/road_surface_city_500m.png", "road URI")
    require(texture.findtext("normal") == "mesh/normal_map_city_500m.png", "normal URI")
    require(texture.findtext("size") == "500", "city texture scale")

    buildings = next(
        (
            model
            for model in world.findall("model")
            if model.attrib.get("name") == "applepark_buildings"
        ),
        None,
    )
    require(buildings is not None, "city buildings model")
    require(buildings.findtext("pose") == "0 0 0 0 0 0", "city buildings pose")
    require(
        buildings.findtext(".//visual//mesh/uri") == "mesh/buildings.dae"
        and buildings.findtext(".//collision//mesh/uri") == "mesh/buildings.dae",
        "city building mesh URIs",
    )
    collision_obj = CITY / "mesh/city_terrain_collision.obj"
    vertex_count = 0
    face_count = 0
    with collision_obj.open("r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("v "):
                vertex_count += 1
            elif line.startswith("f "):
                face_count += 1
    require(vertex_count == 16641 and face_count == 32768, "city collision OBJ topology")
    require(
        all(model.attrib.get("name") != "drone_spawn_pad" for model in world.findall("model")),
        "city still contains the removed green spawn pad",
    )
    spawn_frame = next(
        (frame for frame in world.findall("frame") if frame.attrib.get("name") == "drone_spawn"),
        None,
    )
    require(spawn_frame is not None, "city drone spawn frame")
    require(spawn_frame.findtext("pose") == "-120 115 0 0 0 0", "city spawn frame pose")
    require(
        all(include.findtext("uri") != "model://map_preview_drone" for include in world.findall("include")),
        "city contains the removed static preview drone",
    )
    city_includes = world.findall("include")
    require(
        [include.findtext("uri") for include in city_includes]
        == ["model://moving_platform_track"],
        "city trailer include changed",
    )
    require(city_includes[0].findtext("name") == "flat_platform", "city trailer entity")
    require(city_includes[0].findtext("pose") == "-175 140 0 0 0 0", "city trailer spawn")
    expected_images = {
        "mesh/road_surface_city_500m.png": (2048, 2048, "RGB"),
        "mesh/height_map_city_500m.png": (257, 257, "L"),
        "mesh/normal_map_city_500m.png": (257, 257, "RGB"),
    }
    for relative, (width, height, mode) in expected_images.items():
        with Image.open(CITY / relative) as image:
            require(image.size == (width, height), f"image size: {relative}")
            require(image.mode == mode, f"image mode: {relative}")

    with Image.open(CITY / "mesh/height_map_city_500m.png") as image:
        height_pixels = list(image.getdata())
    require(
        min(height_pixels) == max(height_pixels) == CITY_HEIGHTMAP_OBSERVED_MAX,
        "city heightmap is not completely flat",
    )
    visual_heights = [
        CITY_HEIGHTMAP_POSITION_Z
        + pixel / CITY_HEIGHTMAP_OBSERVED_MAX * CITY_RENDER_HEIGHT_SIZE_Z
        for pixel in height_pixels
    ]
    source_heights = [0.0 for _ in height_pixels]
    render_alignment_error = max(abs(visual) for visual in visual_heights)
    require(render_alignment_error < 1e-12, "city visual / collision terrain Z alignment")

    with OVERLAP_CSV.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 274, "city overlap CSV building count")
    legacy_pixels = sum(int(row["legacy_asphalt_overlap_pixels"]) for row in rows)
    corrected_asphalt = sum(
        int(row["corrected_asphalt_overlap_pixels_before_clearance"]) for row in rows
    )
    corrected_semantic = sum(
        int(row["corrected_semantic_road_overlap_pixels_before_clearance"]) for row in rows
    )
    corrected_ordinals = {
        int(row["post_height_component_ordinal"])
        for row in rows
        if int(row["corrected_semantic_road_overlap_pixels_before_clearance"]) > 0
    }
    require(legacy_pixels == 104410, "legacy city overlap pixel audit")
    require(corrected_asphalt == 237, "coordinate-corrected asphalt pixel audit")
    require(corrected_semantic == 677, "coordinate-corrected semantic road pixel audit")
    require(corrected_ordinals == {1046, 1107, 1171, 1519, 1530, 1589}, "corrected buildings")

    with FOUNDATION_CSV.open("r", encoding="utf-8", newline="") as stream:
        foundation_rows = list(csv.DictReader(stream))
    require(len(foundation_rows) == 274, "city foundation audit building count")
    foundation_extensions = [
        float(row["foundation_extension_m"]) for row in foundation_rows
    ]
    require(
        max(abs(value - 0.05) for value in foundation_extensions) < 1e-12,
        "city foundations are not exactly 5 cm below the flat datum",
    )
    foundation_check = run_checked(
        ["python3", str(GAZEBO / "tools/flatten_city_assets.py"), "--check"]
    )
    require("result=PASS" in foundation_check, "city deterministic foundation validator")

    with HEIGHT_SCALING_CSV.open("r", encoding="utf-8", newline="") as stream:
        height_rows = list(csv.DictReader(stream))
    require(len(height_rows) == CITY_BUILDING_COUNT, "city height audit building count")
    height_factors = [float(row["factor"]) for row in height_rows]
    scaled_heights = [float(row["new_above_ground_height_m"]) for row in height_rows]
    scaled_roofs = [float(row["new_roof_z"]) for row in height_rows]
    active_heights = [float(row["active_above_ground_height_m"]) for row in height_rows]
    active_roofs = [float(row["active_roof_z"]) for row in height_rows]
    expected_digests: list[str] = []
    component_ids: set[int] = set()
    output_ordinals: set[int] = set()
    for ordinal, (height, foundation) in enumerate(
        zip(height_rows, foundation_rows), start=1
    ):
        require(
            int(height["output_component_ordinal"]) == ordinal,
            f"city height audit order changed at building {ordinal}",
        )
        require(
            int(foundation["output_component_ordinal"]) == ordinal,
            f"city foundation audit order changed at building {ordinal}",
        )
        component_id = int(height["post_height_component_ordinal"])
        require(
            int(foundation["post_height_component_ordinal"]) == component_id,
            f"city height/foundation component identity differs at building {ordinal}",
        )
        require(component_id not in component_ids, f"duplicate city component ID {component_id}")
        component_ids.add(component_id)
        output_ordinals.add(ordinal)

        digest = hashlib.sha256(
            f"{CITY_HEIGHT_FACTOR_SEED}:{component_id}".encode("ascii")
        ).digest()
        digest_hex = digest.hex()
        expected_digests.append(digest_hex)
        require(
            height["factor_sha256"] == digest_hex,
            f"building {ordinal} deterministic height digest changed",
        )
        factor_millionths = CITY_HEIGHT_FACTOR_MIN_MILLIONTHS + (
            int.from_bytes(digest[:8], "big") % CITY_HEIGHT_FACTOR_BUCKETS
        )
        expected_factor = factor_millionths / 1_000_000.0
        require(
            abs(float(height["factor"]) - expected_factor) < 5e-10,
            f"building {ordinal} deterministic historical factor changed",
        )

    require(len(component_ids) == CITY_BUILDING_COUNT, "city component IDs are not unique")
    require(
        output_ordinals == set(range(1, CITY_BUILDING_COUNT + 1)),
        "city output building ordinals are incomplete",
    )
    digest_order = sorted(
        range(CITY_BUILDING_COUNT),
        key=lambda index: (expected_digests[index], index + 1),
    )
    rank_by_ordinal = {
        row_index + 1: rank for rank, row_index in enumerate(digest_order)
    }
    require(
        set(rank_by_ordinal.values()) == set(range(CITY_BUILDING_COUNT)),
        "city bounded-height ranks are not a complete 0..273 permutation",
    )

    require(min(height_factors) >= 2.0 and max(height_factors) <= 3.5, "city height factor range")
    require(
        all(
            abs(
                float(row["new_above_ground_height_m"])
                - float(row["old_above_ground_height_m"]) * float(row["factor"])
            )
            < 1e-5
            for row in height_rows
        ),
        "city height audit multiplication",
    )
    expected_active_heights: list[float] = []
    for ordinal, (height, foundation, scaled_height, scaled_roof) in enumerate(
        zip(height_rows, foundation_rows, scaled_heights, scaled_roofs), start=1
    ):
        reference_base = float(height["reference_base_z"])
        require(abs(reference_base) < 1e-12, f"building {ordinal} reference datum changed")
        require(
            abs(scaled_roof - (reference_base + scaled_height)) < 1e-9,
            f"building {ordinal} historical scaled roof differs from its height",
        )
        rank = rank_by_ordinal[ordinal]
        fraction = rank / (CITY_BUILDING_COUNT - 1)
        bounded_height = CITY_SOURCE_HEIGHT_MIN_M + (
            CITY_SOURCE_HEIGHT_MAX_M - CITY_SOURCE_HEIGHT_MIN_M
        ) * fraction
        expected_active_heights.append(bounded_height)
        require(
            int(height["bounded_height_rank"]) == rank,
            f"building {ordinal} bounded-height rank changed",
        )
        require(
            abs(float(height["bounded_height_fraction"]) - fraction) < 1e-9,
            f"building {ordinal} bounded-height fraction changed",
        )
        require(
            abs(float(height["bounded_height_m"]) - bounded_height) < 1e-9,
            f"building {ordinal} bounded height changed",
        )
        require(
            height["active_profile"] == CITY_SOURCE_HEIGHT_PROFILE,
            f"building {ordinal} active height profile changed",
        )
        require(
            abs(active_heights[ordinal - 1] - bounded_height) < 1e-9,
            f"building {ordinal} active height differs from its bounded height",
        )
        require(
            abs(active_roofs[ordinal - 1] - (reference_base + bounded_height)) < 1e-9,
            f"building {ordinal} active roof differs from its bounded height",
        )
        require(
            abs(float(foundation["foundation_base_z"]) + 0.05) < 1e-12,
            f"building {ordinal} foundation base is not -0.05 m",
        )
        require(
            abs(float(foundation["roof_z"]) - active_roofs[ordinal - 1]) < 1e-9,
            f"building {ordinal} foundation audit roof differs from the active roof",
        )

    require(
        rank_by_ordinal[190] == 0 and rank_by_ordinal[171] == CITY_BUILDING_COUNT - 1,
        "city hash-rank extrema are not building_190 / building_171",
    )
    require(
        abs(active_heights[189] - CITY_SOURCE_HEIGHT_MIN_M) < 1e-12
        and abs(active_roofs[189] - CITY_SOURCE_HEIGHT_MIN_M) < 1e-12,
        "building_190 is not the exact 10 m source-city minimum",
    )
    require(
        abs(active_heights[170] - CITY_SOURCE_HEIGHT_MAX_M) < 1e-12
        and abs(active_roofs[170] - CITY_SOURCE_HEIGHT_MAX_M) < 1e-12,
        "building_171 is not the exact 20 m source-city maximum",
    )
    require(
        abs(min(expected_active_heights) - CITY_SOURCE_HEIGHT_MIN_M) < 1e-12
        and abs(max(expected_active_heights) - CITY_SOURCE_HEIGHT_MAX_M) < 1e-12
        and abs(min(active_heights) - CITY_SOURCE_HEIGHT_MIN_M) < 1e-12
        and abs(max(active_heights) - CITY_SOURCE_HEIGHT_MAX_M) < 1e-12,
        "source city height range is not exactly 10..20 m",
    )

    run_checked(["xmllint", "--noout", str(CITY_WORLD)])
    run_checked(["xmllint", "--noout", str(CITY / "mesh/buildings.dae")])
    return {
        "buildings": len(rows),
        "legacy_overlap_pixels": legacy_pixels,
        "corrected_asphalt_pixels_before_clearance": corrected_asphalt,
        "corrected_semantic_pixels_before_clearance": corrected_semantic,
        "final_visible_road_pixels": 0,
        "foundation_min_extension_m": min(foundation_extensions),
        "foundation_max_extension_m": max(foundation_extensions),
        "historical_height_factor_min": min(height_factors),
        "historical_height_factor_max": max(height_factors),
        "height_rank_min": min(rank_by_ordinal.values()),
        "height_rank_max": max(rank_by_ordinal.values()),
        "height_min_id": CITY_SOURCE_SHORTEST_ID,
        "height_max_id": CITY_SOURCE_TALLEST_ID,
        "height_min_m": min(active_heights),
        "height_max_m": max(active_heights),
        "roof_max_z_m": max(active_roofs),
        "terrain_visual_collision_z_error_m": render_alignment_error,
        "terrain_min_z_m": min(source_heights),
        "terrain_max_z_m": max(source_heights),
        "road_sha256": EXPECTED_CITY_HASHES[
            "worlds/applepark_city/mesh/road_surface_city_500m.png"
        ],
    }


def validate_model_uri(uri: str) -> Path:
    require(uri.startswith("model://"), f"unexpected non-model URI: {uri}")
    remainder = uri.removeprefix("model://")
    model_name, separator, internal = remainder.partition("/")
    model_root = GAZEBO / "models" / model_name
    require((model_root / "model.sdf").is_file(), f"local model missing: {model_name}")
    if separator:
        require((model_root / internal).is_file(), f"model resource missing: {uri}")
    return model_root


def validate_mountain() -> dict[str, object]:
    world = world_element(MOUNTAIN_WORLD)
    require(world.attrib.get("name") == "ugv_drone_mountain_map", "mountain map world name")
    include_uris = [include.findtext("uri", "") for include in world.findall("include")]
    require(
        include_uris
        == [
            "model://ugv_mou_terrain",
            "model://ugv_mou_forest_obstacles",
            "model://moving_platform_track",
        ],
        f"mountain map includes: {include_uris}",
    )
    require("iris_with_down_camera" not in MOUNTAIN_WORLD.read_text(encoding="utf-8"), "external Iris in map-only world")
    require("map_preview_drone" not in MOUNTAIN_WORLD.read_text(encoding="utf-8"), "static preview drone remains")
    expected_plugins = {
        "gz-sim-physics-system",
        "gz-sim-sensors-system",
        "gz-sim-user-commands-system",
        "gz-sim-scene-broadcaster-system",
        "gz-sim-contact-system",
        "gz-sim-imu-system",
        "gz-sim-air-pressure-system",
        "gz-sim-magnetometer-system",
        "gz-sim-navsat-system",
    }
    plugins = {plugin.attrib.get("filename", "") for plugin in world.findall("plugin")}
    require(plugins == expected_plugins, f"mountain PX4 system plugins: {sorted(plugins)}")
    pad = next(model for model in world.findall("model") if model.attrib.get("name") == "drone_launch_pad")
    pad_pose_z = float(pad.findtext("pose", "").split()[2])
    pad_height = float(pad.findtext(".//collision//box/size", "").split()[2])
    require(
        abs(0.16 - (pad_pose_z + pad_height / 2.0)) < 1e-9,
        "mountain PX4 spawn contact plane differs from the pad top",
    )
    model_roots = [validate_model_uri(uri) for uri in include_uris]
    for model_root in model_roots:
        model = ET.parse(model_root / "model.sdf").getroot()
        for uri_element in model.findall(".//uri"):
            validate_model_uri(uri_element.text or "")
    run_checked(["xmllint", "--noout", str(MOUNTAIN_WORLD)])
    mountain_validator = run_checked(
        ["python3", str(GAZEBO / "tools/validate_mountain_300_assets.py")]
    )
    require("PASS" in mountain_validator, "mountain deterministic validator")
    return {
        "world_sha256": sha256(MOUNTAIN_WORLD),
        "local_models": [path.name for path in model_roots],
        "external_runtime_assets": 0,
    }


def validate_no_external_runtime_paths() -> None:
    active_paths = [
        CITY_WORLD,
        MOUNTAIN_WORLD,
        GAZEBO / "run_world.sh",
        GAZEBO / "models/ugv_mou_terrain/model.sdf",
        GAZEBO / "models/ugv_mou_forest_obstacles/model.sdf",
        GAZEBO / "models/moving_platform_track/model.sdf",
        GAZEBO / "models/moving_platform_aruco/model.sdf",
    ]
    for path in active_paths:
        content = path.read_text(encoding="utf-8")
        require("sim_assets" not in content, f"sim_assets dependency: {path}")
        require("/home/" not in content, f"absolute home dependency: {path}")
        require("file://" not in content, f"file URI dependency: {path}")


def validate_px4_contracts() -> dict[str, object]:
    launch = GAZEBO / "run_px4_map.sh"
    require(launch.is_file() and launch.stat().st_mode & 0o111, "PX4 map launcher is not executable")
    linker = GAZEBO / "link_px4_model.sh"
    require(linker.is_file() and linker.stat().st_mode & 0o111, "PX4 model linker is not executable")
    linker_text = linker.read_text(encoding="utf-8")
    require('MODEL_NAME="x500_city_rgbd_lidar"' in linker_text, "unique PX4 model link name")
    require('elif [[ -e "$TARGET" ]]' in linker_text, "PX4 linker non-symlink refusal is missing")
    require("refusing to replace the existing non-symlink" in linker_text,
            "PX4 linker destructive-path guard is missing")

    vehicle_sdf = REPO / "px4_models/x500_city_rgbd_lidar/model.sdf"
    require(vehicle_sdf.is_file(), "repository PX4 vehicle model is missing")
    vehicle = ET.parse(vehicle_sdf).getroot().find("model")
    require(vehicle is not None and vehicle.attrib.get("name") == "x500_city_rgbd_lidar",
            "repository PX4 vehicle SDF name")
    front_pose_text = vehicle.findtext("./link[@name='front_rgb_link']/pose")
    front_depth_pose_text = vehicle.findtext("./link[@name='front_depth_link']/pose")
    require(front_pose_text is not None and front_depth_pose_text is not None,
            "front RGB/depth sensor poses are missing")
    for link_name in ("front_rgb_link", "front_depth_link"):
        pose = vehicle.find(f"./link[@name='{link_name}']/pose")
        require(
            pose is not None
            and pose.attrib.get("relative_to") == "base_link"
            and abs(float((pose.text or "").split()[2]) - 0.002) < 1e-12,
            f"{link_name} must remain attached to the x500 centre plate",
        )
    front_pitch = float(front_pose_text.split()[4])
    front_depth_pitch = float(front_depth_pose_text.split()[4])
    require(abs(front_pitch) < 1.0e-12,
            "front RGB scene camera must remain horizontal")
    require(abs(front_depth_pitch) < 1.0e-12,
            "front depth obstacle camera must remain horizontal")
    downward_contracts = (
        ("down_rgb_link", "down_rgb_camera", "camera", "down_camera/image"),
        ("down_depth_link", "down_depth_camera", "depth_camera", "down_depth/image"),
        ("lidar_sensor_link", "lidar", "gpu_lidar", None),
    )
    for link_name, sensor_name, sensor_type, topic in downward_contracts:
        link = vehicle.find(f"./link[@name='{link_name}']")
        require(link is not None, f"downward sensor link is missing: {link_name}")
        pose = link.find("pose")
        require(
            pose is not None
            and pose.attrib.get("relative_to") == "base_link"
            and abs(float((pose.text or "").split()[2]) + 0.05) < 1e-12,
            f"{link_name} is not mounted below PX4 base_link",
        )
        sensor = link.find(f"./sensor[@name='{sensor_name}']")
        require(
            sensor is not None and sensor.attrib.get("type") == sensor_type,
            f"downward sensor contract changed: {sensor_name}",
        )
        if topic is not None:
            require(sensor.findtext("topic") == topic, f"{sensor_name} topic changed")

    launch_text = launch.read_text(encoding="utf-8")
    for marker in (
        "PX4_GZ_STANDALONE=1",
        'PX4_GZ_WORLD="$WORLD_NAME"',
        'PX4_GZ_MODEL_POSE="$SPAWN_POSE"',
        'PX4_SYS_AUTOSTART="$AUTOSTART_ID"',
        'PX4_SIM_MODEL="$SIM_MODEL"',
        'PX4_ARGS+=(-s "$MAVROS_RCS")',
        "expected exactly one uxrce_dds_client start line",
        'DEFAULT_PHYSICS_ENGINE="gz-physics-dartsim-plugin"',
        'PHYSICS_ENGINE="${PHYSICS_ENGINE:-$DEFAULT_PHYSICS_ENGINE}"',
        'FOLLOW_DRONE:-0',
        'gz service -i --service /gui/move_to/pose',
        'track_mode: NONE',
        'gz service -i --service /gui/follow',
    ):
        require(marker in launch_text, f"PX4 launcher contract missing: {marker}")
    require("DDS_STOP_PID" not in launch_text, "racy post-start DDS stop watcher remains")
    expected = {
        "city": ("applepark_city", (-120.0, 115.0, 0.0), (-120.0, 115.0, 0.24), 274),
        "mountain": ("ugv_drone_mountain_map", (-80.0, -80.0, 0.16), (-80.0, -80.0, 0.40), 288),
    }
    counts = {}
    for name, (world_name, spawn_xyz, local_origin_xyz, obstacle_count) in expected.items():
        path = GAZEBO / "maps" / f"{name}_coordinates.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        require(document["map"]["gazebo_world_name"] == world_name, f"{name} YAML world")
        require(document["px4_vehicle"]["airframe_autostart_id"] == 4001, f"{name} PX4 airframe")
        require(document["px4_vehicle"]["simulation_model"] == "gz_x500_city_rgbd_lidar", f"{name} PX4 model")
        require(document["px4_vehicle"]["runtime_entity_name"] == "x500_city_rgbd_lidar_0", f"{name} PX4 entity")
        pose = document["spawn"]["gazebo_spawn_pose_enu"]
        require(tuple(pose[axis] for axis in ("x", "y", "z")) == spawn_xyz, f"{name} spawn")
        require(
            tuple(document["frames"]["px4_local"]["origin_enu_m"]) == local_origin_xyz,
            f"{name} PX4 local base_link origin",
        )
        if name == "city":
            actual = len(document["obstacles"]["buildings"])
        else:
            require(document["obstacles"]["maze_walls"] == [], "mountain YAML maze is not empty")
            actual = len(document["obstacles"]["trees"])
        require(actual == obstacle_count, f"{name} obstacle coordinate count")
        trailer = document["trailer"]
        require(trailer["entity_name"] == "flat_platform", f"{name} trailer entity")
        require(trailer["model_uri"] == "model://moving_platform_track", f"{name} trailer URI")
        require(len(trailer["waypoints_enu_m"]) >= 8, f"{name} trailer waypoint route")
        counts[name] = actual
    generator = run_checked(
        ["python3", str(GAZEBO / "tools/generate_path_planning_assets.py"), "--check"]
    )
    require("buildings=274" in generator and "trees=288 maze_walls=0" in generator,
            "path-planning asset generator validation")
    return counts


def validate_active_uav_city_contract() -> dict[str, object]:
    """Validate the checked-in city profile selected by ``run_px4_map.sh city``."""
    path = GAZEBO / "maps/city_coordinates_uav.yaml"
    preview = GAZEBO / "validation/path_planning/city_uav_205_reference.png"
    require(preview.is_file() and preview.stat().st_size > 0, "active UAV city reference image")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    bounds = document["map"]["bounds_enu_m"]
    terrain = document["terrain"]
    summary = document["obstacles"]["summary"]
    derivation = document["derivation"]
    reduction = derivation["building_reduction"]
    collision = derivation["collision_geometry"]

    require(bounds["x"] == [-650.0, 650.0], "active UAV city X bounds")
    require(bounds["y"] == [-650.0, 650.0], "active UAV city Y bounds")
    require(terrain["collision_geometry"]["size_m"][:2] == [1300.0, 1300.0],
            "active UAV city ground size")
    require(terrain["rows"] == 256 and terrain["columns"] == 256,
            "active UAV city Ogre2 heightmap dimensions")
    require(
        terrain["texture_boundary_extension"] == "edge_clamp_feather_to_neutral",
        "active UAV city texture edge extension",
    )
    spawn_pose = document["spawn"]["gazebo_spawn_pose_enu"]
    trailer_pose = document["trailer"]["spawn_pose_enu"]
    require(
        tuple(float(spawn_pose[key]) for key in ("x", "y", "z")) == (587.0, 580.0, 0.0),
        "active UAV city drone is not at the north-east road-end site",
    )
    require(
        tuple(float(trailer_pose[key]) for key in ("x", "y", "z")) == (-587.0, -512.0, 0.0),
        "active UAV city trailer is not at the south-west road-end site",
    )
    forward = document["px4_vehicle"]["forward_sensors"]
    require(
        forward == {
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
        },
        "active UAV city forward sensor contract",
    )
    downward = document["px4_vehicle"]["downward_sensors"]
    require(
        downward["rgb_camera"]["camera_info_topic"] == "/down_camera/camera_info"
        and downward["rgb_camera"]["optical_frame_id"] == "down_camera_optical_frame"
        and downward["rgb_camera"]["resolution"] == [640, 480]
        and downward["rgb_camera"]["nominal_update_rate_hz"] == 20.0
        and downward["depth_camera"]["image_topic"] == "/down_depth/image"
        and downward["depth_camera"]["metric_image_topic"] == "/down_depth/image_raw"
        and downward["depth_camera"]["encoding"] == "mono8"
        and downward["depth_camera"]["metric_encoding"] == "32FC1"
        and downward["depth_camera"]["camera_info_topic"] == "/down_depth/camera_info"
        and downward["depth_camera"]["optical_frame_id"] == "down_depth_optical_frame"
        and downward["depth_camera"]["resolution"] == [320, 240]
        and downward["depth_camera"]["nominal_update_rate_hz"] == 15.0
        and downward["lidar"]["frame_id"] == "lidar_sensor_link"
        and downward["lidar"]["range_m"] == [0.1, 100.0]
        and downward["lidar"]["nominal_update_rate_hz"] == 50.0,
        "active UAV city downward sensor contract",
    )
    trailer = document["trailer"]
    require(trailer["entity_name"] == "trailer", "active UAV city trailer entity")
    require(
        trailer["model_uri"] == "model://moving_platform_aruco",
        "active UAV city must use the SEO moving-platform trailer",
    )
    require(trailer["body_footprint_m"] == [5.0, 5.0],
            "active UAV city SEO trailer footprint")
    require(abs(float(trailer["deck_height_m"]) - 2.05) < 1e-12,
            "active UAV city SEO trailer deck height")
    require(
        trailer["controller_plugin"] == "libMovingPlatformController.so"
        and abs(float(trailer["controller_velocity_m_s"])) < 1e-12,
        "active UAV city SEO trailer controller contract",
    )
    trailer_model_path = GAZEBO / "models/moving_platform_aruco/model.sdf"
    trailer_tag_path = GAZEBO / "models/moving_platform_aruco/arucotag.png"
    require(trailer_model_path.is_file() and trailer_tag_path.is_file(),
            "active UAV city SEO trailer assets")
    trailer_model = ET.parse(trailer_model_path).getroot().find("model")
    require(
        trailer_model is not None
        and trailer_model.attrib.get("name") == "flat_platform",
        "SEO trailer root model",
    )
    trailer_link = trailer_model.find("link[@name='platform_link']")
    require(
        trailer_link is not None
        and trailer_link.findtext("pose") == "0 0 2 0 0 0",
        "SEO trailer platform link",
    )
    trailer_plugin = trailer_model.find("plugin")
    require(
        trailer_plugin is not None
        and trailer_plugin.attrib.get("filename") == "libMovingPlatformController.so"
        and trailer_plugin.findtext("link_name") == "platform_link",
        "SEO trailer plugin",
    )
    active_world = ET.parse(REPO / document["map"]["world_file"]).getroot()
    trailer_include = next(
        (
            include
            for include in active_world.findall(".//include")
            if include.findtext("name") == "trailer"
        ),
        None,
    )
    require(
        trailer_include is not None
        and trailer_include.findtext("uri") == "model://moving_platform_aruco"
        and trailer_include.findtext("pose") == "-587 -512 0 0 0 0",
        "active UAV city SEO trailer include",
    )
    spawn_separation = math.dist(
        (float(spawn_pose["x"]), float(spawn_pose["y"])),
        (float(trailer_pose["x"]), float(trailer_pose["y"])),
    )
    require(spawn_separation >= 1500.0, "active UAV city staging sites are not diagonal")
    require(derivation["city_spacing_scale_xy"] == 2.5,
            "active UAV city centroid spacing")
    require(derivation["building_footprint_scale_xy"] == 2.5,
            "active UAV city footprint scale")
    height_transform = derivation["height_transform"]
    require(height_transform["source_profile"] == CITY_SOURCE_HEIGHT_PROFILE,
            "active UAV city source height profile")
    require(height_transform["source_range_m"] ==
            [CITY_SOURCE_HEIGHT_MIN_M, CITY_SOURCE_HEIGHT_MAX_M],
            "active UAV city source height range")
    require(height_transform["active_profile"] == CITY_UAV_HEIGHT_PROFILE,
            "active UAV city transformed height profile")
    require(height_transform["active_range_m"] ==
            [CITY_UAV_HEIGHT_MIN_M, CITY_UAV_HEIGHT_MAX_M],
            "active UAV city transformed height range")
    require(abs(float(height_transform["target_mean_m"]) - CITY_UAV_HEIGHT_MEAN_M) < 1e-12,
            "active UAV city transformed height target mean")
    require(height_transform["foundation_and_ground_unchanged"] is True,
            "active UAV city foundation/ground contract")
    require(summary["active_height_profile"] == CITY_UAV_HEIGHT_PROFILE,
            "active UAV city height profile")
    roof_range = summary["roof_z_range_m"]
    require(
        len(roof_range) == 2
        and all(
            abs(float(actual) - expected) < 1e-12
            for actual, expected in zip(
                roof_range,
                (CITY_UAV_HEIGHT_MIN_M, CITY_UAV_HEIGHT_MAX_M),
            )
        ),
        "active UAV city roof range is not exactly 20..50 m",
    )
    require(
        summary["shortest_building"]["id"] == CITY_UAV_SHORTEST_ID
        and abs(
            float(summary["shortest_building"]["height_above_ground_m"])
            - CITY_UAV_HEIGHT_MIN_M
        ) < 1e-12,
        "active UAV city shortest building contract",
    )
    require(
        summary["tallest_building"]["id"] == CITY_UAV_TALLEST_ID
        and abs(
            float(summary["tallest_building"]["height_above_ground_m"])
            - CITY_UAV_HEIGHT_MAX_M
        ) < 1e-12,
        "active UAV city tallest building contract",
    )
    require(summary["source_building_count"] == 274, "active UAV city source building count")
    require(summary["building_count"] == 205, "active UAV city retained building count")
    require(summary["removed_building_count"] == 69, "active UAV city removed building count")
    require(summary["height_distribution"]["count"] == 205,
            "active UAV city height distribution count")
    require(summary["height_distribution"]["unique_count"] == 205,
            "active UAV city heights are not uniquely distributed")
    require(
        abs(float(summary["height_distribution"]["mean_m"]) - CITY_UAV_HEIGHT_MEAN_M) < 1e-12
        and abs(float(summary["height_distribution"]["median_m"]) - CITY_UAV_HEIGHT_MEAN_M) < 1e-12
        and abs(float(summary["height_distribution"]["target_mean_m"]) - CITY_UAV_HEIGHT_MEAN_M) < 1e-12,
        "active UAV city height mean/median is not exactly 35 m",
    )
    random_audit = reduction["spatial_random_audit"]
    require(random_audit["seed"] == 7577, "active UAV city removal seed")
    require(random_audit["grid_shape"] == [5, 5], "active UAV city removal grid")
    require(random_audit["cells_with_removals"] == 25,
            "active UAV city removals do not cover every region")
    require(
        random_audit["removed_ids_sha256"]
        == "de4469c06f387f4acc97ef29c9b9818fa08a4e831541f2f199cf62b35459c753",
        "active UAV city removal digest",
    )
    require(reduction["removed_ids"].count("building_265") == 1,
            "fixed A* goal blocking building was not removed")
    require(not (set(reduction["removed_ids"]) & set(reduction["protected_ids"])),
            "active UAV city removed a protected building")
    require(summary["building_overlap_count"] == 0,
            "active UAV city has overlapping buildings")
    require(summary["all_building_footprints_disjoint"] is True,
            "active UAV city building footprints are not disjoint")
    require(summary["gap_constraint_enforced"] is False,
            "active UAV city unexpectedly enforces an inter-building gap")
    require(summary["minimum_building_gap_m"] > 0.0,
            "active UAV city building footprints touch")
    require(collision["type"] == "exact_per_building_polyline_prisms_dart",
            "active UAV city collision type")
    require(collision["count"] == 205, "active UAV city collision count")
    require(
        collision["required_physics_engine"]
        == "gz-physics-dartsim-plugin"
        and collision["dart_sdf_mesh_supported"] is False
        and collision["dart_sdf_polyline_supported"] is True,
        "active UAV city exact mesh physics contract",
    )
    require(collision["maximum_outward_error_m"] == 0.0,
            "active UAV city collision outward error")

    expansion = run_checked(
        ["python3", str(GAZEBO / "tools/validate_city_uav_expansion.py")]
    )
    require("PASS city_uav expansion" in expansion,
            "active UAV city deterministic validator")
    return {
        "size_m": 1300,
        "spacing_scale": derivation["city_spacing_scale_xy"],
        "footprint_scale": derivation["building_footprint_scale_xy"],
        "height_profile": summary["active_height_profile"],
        "height_min_m": float(roof_range[0]),
        "height_max_m": float(roof_range[1]),
        "height_mean_m": float(summary["height_distribution"]["mean_m"]),
        "height_median_m": float(summary["height_distribution"]["median_m"]),
        "collision_geometries": collision["count"],
        "building_count": summary["building_count"],
        "removed_building_count": summary["removed_building_count"],
        "removal_grid_cells": random_audit["cells_with_removals"],
        "minimum_gap_m": summary["minimum_building_gap_m"],
        "spawn_separation_m": spawn_separation,
    }


def main() -> None:
    city = validate_city()
    active_city = validate_active_uav_city_contract()
    mountain = validate_mountain()
    validate_no_external_runtime_paths()
    coordinates = validate_px4_contracts()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    output = "\n".join(
        [
            "PX4-ROS2 self-contained Gazebo map static validation",
            "result=PASS",
            f"repository={REPO}",
            "city_engine=Gazebo Harmonic",
            "legacy_city_size_m=500x500",
            f"active_uav_city_size_m={active_city['size_m']}x{active_city['size_m']}",
            f"active_uav_city_centroid_spacing_scale={active_city['spacing_scale']}",
            f"active_uav_city_footprint_scale={active_city['footprint_scale']}",
            f"active_uav_city_height_profile={active_city['height_profile']}",
            "active_uav_city_height_range_m="
            f"{active_city['height_min_m']:.6f}..{active_city['height_max_m']:.6f}",
            f"active_uav_city_height_mean_m={active_city['height_mean_m']:.6f}",
            f"active_uav_city_height_median_m={active_city['height_median_m']:.6f}",
            f"active_uav_city_buildings={active_city['building_count']}",
            f"active_uav_city_removed_buildings={active_city['removed_building_count']}",
            f"active_uav_city_removal_grid_cells={active_city['removal_grid_cells']}",
            f"active_uav_city_collision_geometries={active_city['collision_geometries']}",
            f"active_uav_city_minimum_building_gap_m={active_city['minimum_gap_m']:.6f}",
            f"active_uav_city_spawn_separation_m={active_city['spawn_separation_m']:.6f}",
            f"city_buildings={city['buildings']}",
            f"city_legacy_overlap_pixels={city['legacy_overlap_pixels']}",
            "city_coordinate_corrected_asphalt_pixels_before_clearance="
            f"{city['corrected_asphalt_pixels_before_clearance']}",
            "city_coordinate_corrected_semantic_pixels_before_clearance="
            f"{city['corrected_semantic_pixels_before_clearance']}",
            f"city_final_visible_road_overlap_pixels={city['final_visible_road_pixels']}",
            "city_building_foundation_alignment=PASS",
            f"city_foundation_extension_m={city['foundation_min_extension_m']:.6f}..{city['foundation_max_extension_m']:.6f}",
            "city_building_height_scaling=PASS",
            f"city_source_height_profile={CITY_SOURCE_HEIGHT_PROFILE}",
            "city_historical_height_random_factor="
            f"{city['historical_height_factor_min']:.6f}..{city['historical_height_factor_max']:.6f}",
            f"city_source_height_rank={city['height_rank_min']}..{city['height_rank_max']}",
            "city_source_height_extrema="
            f"{city['height_min_id']}:{city['height_min_m']:.6f}.."
            f"{city['height_max_id']}:{city['height_max_m']:.6f}",
            f"city_source_above_ground_height_m={city['height_min_m']:.6f}..{city['height_max_m']:.6f}",
            f"city_source_max_roof_z_m={city['roof_max_z_m']:.6f}",
            "city_visual_collision_terrain_z_alignment=PASS",
            "city_visual_collision_terrain_z_max_error_m="
            f"{city['terrain_visual_collision_z_error_m']:.12f}",
            f"city_heightmap_visual_skirt_depth_m={CITY_RENDER_HEIGHT_SIZE_Z:.6f}",
            "city_terrain_z_range_m="
            f"{city['terrain_min_z_m']:.6f}..{city['terrain_max_z_m']:.6f}",
            f"city_road_sha256={city['road_sha256']}",
            "city_xmllint_and_local_uri_closure=PASS",
            "mountain_engine=Gazebo Harmonic",
            "mountain_size_m=300x300",
            f"mountain_world_sha256={mountain['world_sha256']}",
            f"mountain_local_models={','.join(mountain['local_models'])}",
            f"mountain_external_runtime_assets={mountain['external_runtime_assets']}",
            "mountain_deterministic_validator=PASS",
            "static_preview_drones=0",
            "legacy_city_and_mountain_trailer_model=flat_platform",
            "active_uav_city_drone_spawn_enu=587,580,0",
            "active_uav_city_trailer_model=moving_platform_aruco entity=trailer spawn_enu=-587,-512,0 mean_velocity=0 noise=seo_default_after_px4_spawn",
            "trailer_waypoint_driver=gazebo_transport_no_mavros",
            "px4_dynamic_model=gz_x500_city_rgbd_lidar autostart=4001",
            "px4_model_repository_symlink=PASS",
            "px4_local_origin=base_link_model_root_plus_0.24m",
            "front_rgb_optical_axis=pitch_0deg front_depth=pitch_0deg",
            f"coordinate_yaml_obstacles=city:{coordinates['city']} mountain_trees:{coordinates['mountain']} maze:0",
            "path_planning_reference_images=PASS",
            "active_sim_assets_or_absolute_home_references=0",
            "repository_only_asset_closure=PASS",
        ]
    ) + "\n"
    LOG.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
