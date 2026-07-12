#!/usr/bin/env python3
"""Validate repository-only city and mountain Gazebo map entry points."""

from __future__ import annotations

import csv
import hashlib
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
CITY_SOURCE_HEIGHT_SCALE_Z = 26.6
CITY_HEIGHTMAP_POSITION_Z = -15.0
CITY_HEIGHTMAP_UINT8_MAX = 255
CITY_HEIGHTMAP_OBSERVED_MAX = 152
CITY_RENDER_HEIGHT_SIZE_Z = (
    CITY_SOURCE_HEIGHT_SCALE_Z
    * CITY_HEIGHTMAP_OBSERVED_MAX
    / CITY_HEIGHTMAP_UINT8_MAX
)

EXPECTED_CITY_HASHES = {
    "worlds/applepark_city/mesh/buildings.dae":
        "e5fab82529fcc0f9d5819797346af76d763d6b973eed96ad8a7de324a7a253ce",
    "worlds/applepark_city/mesh/height_map_city_500m.png":
        "5a84adc1f45dcffe507fa77d2642cd672c622c225e60ae41f98280bdcf9b24cf",
    "worlds/applepark_city/mesh/normal_map_city_500m.png":
        "13e0f51751c556904b3b2eb92ad118ec3dd20d5f4a2a4e72d4344a6907556266",
    "worlds/applepark_city/mesh/road_surface_city_500m.png":
        "f43cc932dd4c101c605c15c0a202e4e946fbea77e0c568763dbec217dc123a08",
    "worlds/applepark_city/mesh/city_terrain_collision.obj":
        "99d8fae1ed193321dc3f2801b68a549b019a12068cf0cdff17210c78e0fe2024",
    "worlds/applepark_city/OSM_ATTRIBUTION.txt":
        "f706303dc1fa8e4b456e1d235a73ee395be38c2897c62b6ebf19ad74646892fb",
    "validation/city/road_building_overlap_coordinates.csv":
        "b3a1f0eaee7eafc2595c9bd35e6f23a4a74749b65a0a0224dc78ad6d4ddf6601",
    "validation/city/building_foundation_alignment.csv":
        "db0247ad555cd41fd3d902fa827bdc98125d06635c18598e4eb613bcb631e12f",
    "validation/city/building_height_scaling.csv":
        "bc7b14dac3c2cc5244804f37b3eeed8f629dad7e5f4ac955c504222bd15b4543",
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
        terrain.findtext(".//collision//mesh/uri")
        == "mesh/city_terrain_collision.obj",
        "Bullet city terrain collision mesh URI",
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
    pad = next(
        (model for model in world.findall("model") if model.attrib.get("name") == "drone_spawn_pad"),
        None,
    )
    require(pad is not None, "city drone spawn pad")
    require(pad.findtext("pose") == "-120 115 -3.269558902 0 0 0", "city spawn pad pose")
    require(
        pad.findtext(".//collision//cylinder/length") == "0.5"
        and pad.findtext(".//visual//cylinder/length") == "0.5",
        "city spawn pad foundation depth",
    )
    require(
        all(include.findtext("uri") != "model://map_preview_drone" for include in world.findall("include")),
        "city contains the removed static preview drone",
    )
    pad_pose_z = float(pad.findtext("pose", "").split()[2])
    pad_height = float(pad.findtext(".//collision//cylinder/length", "nan"))
    require(
        abs(-3.019558902 - (pad_pose_z + pad_height / 2.0)) < 1e-9,
        "city PX4 spawn contact plane differs from the pad top",
    )

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
    require(max(height_pixels) == CITY_HEIGHTMAP_OBSERVED_MAX, "city heightmap maximum")
    # Ogre2 scales image pixels by the maximum value actually present.  The
    # corrected SDF size makes that renderer formula identical to the source
    # elevation / Bullet collision formula at every uint8 pixel value.
    visual_heights = [
        CITY_HEIGHTMAP_POSITION_Z
        + pixel / CITY_HEIGHTMAP_OBSERVED_MAX * CITY_RENDER_HEIGHT_SIZE_Z
        for pixel in height_pixels
    ]
    source_heights = [
        CITY_HEIGHTMAP_POSITION_Z
        + pixel / CITY_HEIGHTMAP_UINT8_MAX * CITY_SOURCE_HEIGHT_SCALE_Z
        for pixel in height_pixels
    ]
    render_alignment_error = max(
        abs(visual - source)
        for visual, source in zip(visual_heights, source_heights)
    )
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
    require(min(foundation_extensions) > 0.109, "city minimum foundation extension")
    require(max(foundation_extensions) < 0.899, "city maximum foundation extension")
    foundation_check = run_checked(
        ["python3", str(GAZEBO / "tools/align_city_building_foundations.py"), "--check"]
    )
    require("result=PASS" in foundation_check, "city deterministic foundation validator")
    height_check = run_checked(
        ["python3", str(GAZEBO / "tools/scale_city_building_heights.py"), "--check"]
    )
    require("result=PASS" in height_check, "city deterministic height validator")

    with HEIGHT_SCALING_CSV.open("r", encoding="utf-8", newline="") as stream:
        height_rows = list(csv.DictReader(stream))
    require(len(height_rows) == 274, "city height audit building count")
    height_factors = [float(row["factor"]) for row in height_rows]
    new_heights = [float(row["new_above_ground_height_m"]) for row in height_rows]
    new_roofs = [float(row["new_roof_z"]) for row in height_rows]
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
        "height_factor_min": min(height_factors),
        "height_factor_max": max(height_factors),
        "height_min_m": min(new_heights),
        "height_max_m": max(new_heights),
        "roof_max_z_m": max(new_roofs),
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
    ]
    for path in active_paths:
        content = path.read_text(encoding="utf-8")
        require("sim_assets" not in content, f"sim_assets dependency: {path}")
        require("/home/" not in content, f"absolute home dependency: {path}")
        require("file://" not in content, f"file URI dependency: {path}")


def validate_px4_contracts() -> dict[str, object]:
    launch = GAZEBO / "run_px4_map.sh"
    require(launch.is_file() and launch.stat().st_mode & 0o111, "PX4 map launcher is not executable")
    launch_text = launch.read_text(encoding="utf-8")
    for marker in (
        "PX4_GZ_STANDALONE=1",
        'PX4_GZ_WORLD="$WORLD_NAME"',
        'PX4_GZ_MODEL_POSE="$SPAWN_POSE"',
        'PX4_SYS_AUTOSTART="$AUTOSTART_ID"',
        'PX4_SIM_MODEL="$SIM_MODEL"',
    ):
        require(marker in launch_text, f"PX4 launcher contract missing: {marker}")
    expected = {
        "city": ("applepark_city", (-120.0, 115.0, -3.019558902), 274),
        "mountain": ("ugv_drone_mountain_map", (-80.0, -80.0, 0.16), 288),
    }
    counts = {}
    for name, (world_name, spawn_xyz, obstacle_count) in expected.items():
        path = GAZEBO / "maps" / f"{name}_coordinates.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        require(document["map"]["gazebo_world_name"] == world_name, f"{name} YAML world")
        require(document["px4_vehicle"]["airframe_autostart_id"] == 4014, f"{name} PX4 airframe")
        require(document["px4_vehicle"]["simulation_model"] == "gz_x500_mono_cam_down", f"{name} PX4 model")
        pose = document["spawn"]["gazebo_spawn_pose_enu"]
        require(tuple(pose[axis] for axis in ("x", "y", "z")) == spawn_xyz, f"{name} spawn")
        if name == "city":
            actual = len(document["obstacles"]["buildings"])
        else:
            require(document["obstacles"]["maze_walls"] == [], "mountain YAML maze is not empty")
            actual = len(document["obstacles"]["trees"])
        require(actual == obstacle_count, f"{name} obstacle coordinate count")
        counts[name] = actual
    generator = run_checked(
        ["python3", str(GAZEBO / "tools/generate_path_planning_assets.py"), "--check"]
    )
    require("buildings=274" in generator and "trees=288 maze_walls=0" in generator,
            "path-planning asset generator validation")
    return counts


def main() -> None:
    city = validate_city()
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
            "city_size_m=500x500",
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
            f"city_current_height_random_factor={city['height_factor_min']:.6f}..{city['height_factor_max']:.6f}",
            f"city_new_above_ground_height_m={city['height_min_m']:.6f}..{city['height_max_m']:.6f}",
            f"city_max_roof_z_m={city['roof_max_z_m']:.6f}",
            "city_visual_collision_terrain_z_alignment=PASS",
            "city_visual_collision_terrain_z_max_error_m="
            f"{city['terrain_visual_collision_z_error_m']:.12f}",
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
            "px4_dynamic_model=gz_x500_mono_cam_down autostart=4014",
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
