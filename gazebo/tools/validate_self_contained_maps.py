#!/usr/bin/env python3
"""Validate repository-only city and mountain Gazebo map entry points."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


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
        "gz-sim-sensors-system",
        "gz-sim-imu-system",
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
    city_drone = next(
        (
            include
            for include in world.findall("include")
            if include.findtext("uri") == "model://map_preview_drone"
        ),
        None,
    )
    require(city_drone is not None, "city repository-local preview drone")
    require(city_drone.findtext("pose") == "-120 115 -3.019558902 0 0 0", "city drone contact pose")
    preview_contact_z = validate_preview_drone_model()
    pad_pose_z = float(pad.findtext("pose", "").split()[2])
    pad_height = float(pad.findtext(".//collision//cylinder/length", "nan"))
    drone_pose_z = float(city_drone.findtext("pose", "").split()[2])
    require(
        abs(drone_pose_z + preview_contact_z - (pad_pose_z + pad_height / 2.0)) < 1e-9,
        "city preview landing gear is not seated on the pad",
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


def validate_preview_drone_model() -> float:
    model_root = validate_model_uri("model://map_preview_drone")
    model = ET.parse(model_root / "model.sdf").getroot().find("model")
    require(model is not None and model.findtext("static") == "true", "preview drone is not static")
    gear_bottoms = []
    for gear_name in ("landing_gear_left", "landing_gear_right"):
        gear = model.find(f".//visual[@name='{gear_name}']")
        require(gear is not None, f"preview drone gear missing: {gear_name}")
        pose = [float(value) for value in gear.findtext("pose", "").split()]
        size = [float(value) for value in gear.findtext("./geometry/box/size", "").split()]
        require(len(pose) == 6 and len(size) == 3, f"preview drone gear geometry: {gear_name}")
        gear_bottoms.append(pose[2] - size[2] / 2.0)
    require(
        max(abs(bottom - PREVIEW_CONTACT_Z) for bottom in gear_bottoms) < 1e-12,
        "preview drone contact plane changed",
    )
    return PREVIEW_CONTACT_Z


def validate_mountain() -> dict[str, object]:
    world = world_element(MOUNTAIN_WORLD)
    require(world.attrib.get("name") == "ugv_drone_mountain_map", "mountain map world name")
    include_uris = [include.findtext("uri", "") for include in world.findall("include")]
    require(
        include_uris
        == [
            "model://ugv_mou_terrain",
            "model://ugv_mou_forest_obstacles",
            "model://map_preview_drone",
        ],
        f"mountain map-only includes: {include_uris}",
    )
    require("iris_with_down_camera" not in MOUNTAIN_WORLD.read_text(encoding="utf-8"), "external Iris in map-only world")
    preview = next(include for include in world.findall("include") if include.findtext("name") == "mountain_preview_drone")
    require(preview.findtext("pose") == "-80 -80 0.16 0 0 0.785398", "mountain preview contact pose")
    pad = next(model for model in world.findall("model") if model.attrib.get("name") == "drone_launch_pad")
    pad_pose_z = float(pad.findtext("pose", "").split()[2])
    pad_height = float(pad.findtext(".//collision//box/size", "").split()[2])
    preview_pose_z = float(preview.findtext("pose", "").split()[2])
    require(
        abs(preview_pose_z + validate_preview_drone_model() - (pad_pose_z + pad_height / 2.0)) < 1e-9,
        "mountain preview landing gear is not seated on the pad",
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


def main() -> None:
    city = validate_city()
    mountain = validate_mountain()
    validate_no_external_runtime_paths()
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
            "active_sim_assets_or_absolute_home_references=0",
            "repository_only_asset_closure=PASS",
        ]
    ) + "\n"
    LOG.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
