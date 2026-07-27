#!/usr/bin/env python3
"""Static/resource validation for the PX4-ROS2 300 m mountain world."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


GAZEBO_ROOT = Path(__file__).resolve().parents[1]
MAP_WORLD = GAZEBO_ROOT / "worlds" / "ugv_drone_map.world"
WORLD = MAP_WORLD
TERRAIN = GAZEBO_ROOT / "models" / "ugv_mou_terrain"
FOREST = GAZEBO_ROOT / "models" / "ugv_mou_forest_obstacles"
HEIGHTMAP = TERRAIN / "materials" / "textures" / "mountain_height_300.png"
TEXTURE = TERRAIN / "materials" / "textures" / "natural_ground.png"
VISUAL_OBJ = TERRAIN / "meshes" / "ugv_mou_terrain_visual.obj"
COLLISION_OBJ = TERRAIN / "meshes" / "ugv_mou_terrain_collision.obj"
TREE_LAYOUT = FOREST / "source" / "tree_layout.source.xml"
LOG = GAZEBO_ROOT / "validation" / "ugv_drone_mountain_300_static.log"
RUNTIME_LOG = GAZEBO_ROOT / "validation" / "runtime" / "mountain_px4_runtime_validation.log"
EXPECTED_HEIGHTMAP_SHA256 = (
    "dea9d2ca2f6a4f357037e0f626a90ab2bfb815d6b1185659cc8d53b1ed63548c"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terrain_height(pixels: np.ndarray, x: float, y: float) -> float:
    col = (x + 150.0) / 300.0 * 256.0
    row = (150.0 - y) / 300.0 * 256.0
    c0, r0 = int(math.floor(col)), int(math.floor(row))
    c1, r1 = min(c0 + 1, 256), min(r0 + 1, 256)
    tx, ty = col - c0, row - r0
    value = (
        (1.0 - tx) * (1.0 - ty) * float(pixels[r0, c0])
        + tx * (1.0 - ty) * float(pixels[r0, c1])
        + (1.0 - tx) * ty * float(pixels[r1, c0])
        + tx * ty * float(pixels[r1, c1])
    )
    return value / 255.0 * 40.0


def generated_heightmap() -> np.ndarray:
    """Load the repository generator and reproduce its canonical pixels."""

    path = GAZEBO_ROOT / "tools" / "build_mountain_300_assets.py"
    spec = importlib.util.spec_from_file_location("px4_mountain_builder", path)
    require(spec is not None and spec.loader is not None, "mountain generator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_heightmap()


def disk_terrain_samples(
    pixels: np.ndarray, x: float, y: float, radius: float
) -> list[float]:
    values = [terrain_height(pixels, x, y)]
    for sample_radius in np.linspace(radius / 4.0, radius, 4):
        for angle in np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False):
            values.append(
                terrain_height(
                    pixels,
                    x + float(sample_radius) * math.cos(float(angle)),
                    y + float(sample_radius) * math.sin(float(angle)),
                )
            )
    return values


def oriented_box_terrain_samples(
    pixels: np.ndarray, pose: list[float], size: list[float], step: float = 0.25
) -> list[float]:
    along = np.linspace(
        -size[0] / 2.0,
        size[0] / 2.0,
        max(2, int(math.ceil(size[0] / step)) + 1),
    )
    across = np.linspace(
        -size[1] / 2.0,
        size[1] / 2.0,
        max(2, int(math.ceil(size[1] / step)) + 1),
    )
    cosine, sine = math.cos(pose[5]), math.sin(pose[5])
    return [
        terrain_height(
            pixels,
            pose[0] + cosine * float(local_x) - sine * float(local_y),
            pose[1] + sine * float(local_x) + cosine * float(local_y),
        )
        for local_x in along
        for local_y in across
    ]


def point_to_oriented_box_gap(
    x: float,
    y: float,
    radius: float,
    pose: list[float],
    size: list[float],
) -> float:
    dx, dy = x - pose[0], y - pose[1]
    cosine, sine = math.cos(pose[5]), math.sin(pose[5])
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    outside_x = max(abs(local_x) - size[0] / 2.0, 0.0)
    outside_y = max(abs(local_y) - size[1] / 2.0, 0.0)
    return math.hypot(outside_x, outside_y) - radius


def obj_stats(path: Path) -> tuple[int, int, tuple[float, ...]]:
    vertices = faces = 0
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("v "):
                values = [float(value) for value in line.split()[1:4]]
                vertices += 1
                for axis, value in enumerate(values):
                    mins[axis] = min(mins[axis], value)
                    maxs[axis] = max(maxs[axis], value)
            elif line.startswith("f "):
                faces += 1
    return vertices, faces, (*mins, *maxs)


def validate() -> list[str]:
    lines = ["PX4-ROS2 300 m mountain static validation", "status=PASS"]
    world_tree = ET.parse(WORLD)
    world_root = world_tree.getroot()
    require(world_root.attrib.get("version") == "1.9", "world is not SDF 1.9")
    world = world_root.find("world")
    require(world is not None, "world element is missing")
    require(world.findtext("./physics/max_step_size") == "0.002", "physics step changed")
    require(world.findtext("./physics/real_time_update_rate") == "500", "physics rate changed")
    plugins = {plugin.attrib.get("filename") for plugin in world.findall("plugin")}
    required_plugins = {
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
    require(required_plugins <= plugins, "one or more Harmonic systems disappeared")
    includes = {include.findtext("uri") for include in world.findall("include")}
    require(
        {
            "model://ugv_mou_terrain",
            "model://ugv_mou_forest_obstacles",
            "model://moving_platform_track",
        }
        == includes,
        "mountain PX4 map resource includes changed",
    )
    require(world.findtext("./scene/background") == "0.42 0.42 0.42 1", "background is not neutral")
    require(world.find("./scene/sky") is None, "active sky can reintroduce blue outside the map")
    require(
        world.findtext("./gui/plugin[@name='3D View']/background_color") == "0.42 0.42 0.42",
        "MinimalScene background is not neutral",
    )
    require(world.findtext("./model[@name='drone_launch_pad']/pose") == "-80 -80 0.08 0 0 0", "pad moved")
    world_text = WORLD.read_text(encoding="utf-8").lower()
    require("blue_ground" not in world_text and "sim_assets" not in world_text, "world has a stale resource reference")
    require("map_preview_drone" not in world_text, "static preview drone remains in PX4 map")

    pixels = np.asarray(Image.open(HEIGHTMAP).convert("L"), dtype=np.uint8)
    require(pixels.shape == (257, 257), "heightmap is not 257x257")
    require(sha256(HEIGHTMAP) == EXPECTED_HEIGHTMAP_SHA256, "heightmap checksum mismatch")
    require(
        np.array_equal(pixels, generated_heightmap()),
        "heightmap differs from the deterministic retained-hill/ridge generator",
    )
    edge = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    require(int(edge.max()) == 0, "heightmap edge is nonzero")
    require(int(pixels.max()) == 255, "40 m summit is absent")
    launch_height = terrain_height(pixels, -80.0, -80.0)
    require(abs(launch_height) < 1e-9, "launch pad is not on flat z=0 terrain")
    main_peak = float(pixels.max()) / 255.0 * 40.0
    x_axis = np.linspace(-150.0, 150.0, 257)
    second_peak = float(pixels[:, x_axis > 0.0].max()) / 255.0 * 40.0
    require(main_peak > 39.8 and 19.5 < second_peak < 20.2, "40 m / 20 m summit check failed")
    nonzero_ratio = float(np.count_nonzero(pixels)) / float(pixels.size)
    unique_elevations = len(np.unique(pixels))
    require(0.65 < nonzero_ratio < 0.75, "mountain coverage no longer forms a ridge landscape")
    require(unique_elevations >= 240, "mountain elevation detail was lost")
    heights = pixels.astype(np.float64) / 255.0 * 40.0
    spacing = 300.0 / 256.0
    # Measure the actual two planes written for every OBJ cell.  A centered
    # node gradient can hide the steeper diagonal plane at a grid corner.
    triangle_one_slopes = np.hypot(
        (heights[:-1, 1:] - heights[:-1, :-1]) / spacing,
        (heights[:-1, :-1] - heights[1:, :-1]) / spacing,
    )
    triangle_two_slopes = np.hypot(
        (heights[1:, 1:] - heights[1:, :-1]) / spacing,
        (heights[:-1, 1:] - heights[1:, 1:]) / spacing,
    )
    slopes = np.concatenate((triangle_one_slopes.ravel(), triangle_two_slopes.ravel()))
    slope_p99 = float(np.percentile(slopes, 99.0))
    max_slope = float(slopes.max())
    require(slope_p99 < 0.68 and max_slope < 0.79, "terrain acquired an unintended cliff")
    y_axis = np.linspace(150.0, -150.0, 257)
    clearing_nodes = pixels[
        (x_axis[None, :] / 42.0) ** 2 + (y_axis[:, None] / 30.0) ** 2 <= 1.0
    ]
    require(int(clearing_nodes.max()) == 0, "central flight clearing is not z=0")

    pad_samples = [
        terrain_height(pixels, float(x), float(y))
        for x in np.linspace(-86.0, -74.0, 49)
        for y in np.linspace(-86.0, -74.0, 49)
    ]
    require(max(abs(value) for value in pad_samples) < 1e-12, "launch pad footprint is not flat z=0")

    rgb = np.asarray(Image.open(TEXTURE).convert("RGB"), dtype=np.uint8)
    blue_dominant = int(((rgb[:, :, 2] > rgb[:, :, 0]) & (rgb[:, :, 2] > rgb[:, :, 1])).sum())
    require(blue_dominant == 0, "terrain texture contains blue-dominant pixels")
    mtl_text = (TERRAIN / "meshes" / "ugv_mou_terrain.mtl").read_text(encoding="utf-8")
    require("natural_ground.png" in mtl_text and "blue_ground" not in mtl_text, "terrain MTL is stale")

    visual_stats = obj_stats(VISUAL_OBJ)
    collision_stats = obj_stats(COLLISION_OBJ)
    require(visual_stats == collision_stats, "visual and collision OBJ geometry differ")
    require(sha256(VISUAL_OBJ) == sha256(COLLISION_OBJ), "visual/collision OBJ bytes differ")
    vertices, faces, bounds = visual_stats
    require(vertices == 66049 and faces == 131072, "unexpected terrain mesh topology")
    require(bounds == (-150.0, -150.0, 0.0, 150.0, 150.0, 40.0), f"terrain bounds differ: {bounds}")

    terrain_sdf = ET.parse(TERRAIN / "model.sdf").getroot()
    require(terrain_sdf.attrib.get("version") == "1.9", "terrain is not SDF 1.9")
    terrain_uris = {node.text for node in terrain_sdf.findall(".//uri")}
    require(
        terrain_uris
        == {
            "model://ugv_mou_terrain/meshes/ugv_mou_terrain_collision.obj",
            "model://ugv_mou_terrain/meshes/ugv_mou_terrain_visual.obj",
        },
        "terrain model has a non-local URI",
    )

    forest_root = ET.parse(FOREST / "model.sdf").getroot()
    require(forest_root.attrib.get("version") == "1.9", "forest is not SDF 1.9")
    model = forest_root.find("model")
    require(model is not None and model.findtext("static") == "true", "forest is not static")
    links = model.findall("link")
    require(len(links) == 1 and links[0].attrib.get("name") == "obstacles", "obstacles are not one rooted link")
    obstacle_link = links[0]
    require(obstacle_link.find("pose") is None, "compound obstacle link must remain at model origin")
    collisions = obstacle_link.findall("collision")
    visuals = obstacle_link.findall("visual")
    collision_by_name = {node.attrib.get("name", ""): node for node in collisions}
    visual_by_name = {node.attrib.get("name", ""): node for node in visuals}
    require(len(collision_by_name) == len(collisions), "duplicate collision name")
    require(len(visual_by_name) == len(visuals), "duplicate visual name")

    tree_collisions = {
        name.removesuffix("_trunk_collision"): node
        for name, node in collision_by_name.items()
        if name.startswith("mountain_tree_") and name.endswith("_trunk_collision")
    }
    tree_branches = {
        name.removesuffix("_branch_visual"): node
        for name, node in visual_by_name.items()
        if name.startswith("mountain_tree_") and name.endswith("_branch_visual")
    }
    tree_barks = {
        name.removesuffix("_bark_visual"): node
        for name, node in visual_by_name.items()
        if name.startswith("mountain_tree_") and name.endswith("_bark_visual")
    }
    wall_collisions = {
        name.removesuffix("_collision"): node
        for name, node in collision_by_name.items()
        if name.startswith("maze_") and name.endswith("_collision")
    }
    wall_visuals = {
        name.removesuffix("_visual"): node
        for name, node in visual_by_name.items()
        if name.startswith("maze_") and name.endswith("_visual")
    }
    tree_names = set(tree_collisions)
    wall_names = set(wall_collisions)
    require(tree_names == set(tree_branches) == set(tree_barks), "tree collision/visual sets differ")
    require(not wall_names and not wall_visuals, "maze collision or visual remains")
    require(len(tree_names) == 288, "tree instance count differs")
    require(len(collisions) == 288, "compound link must contain only 288 trunk collisions")
    require(len(visuals) == 576, "compound link must contain only 576 tree visuals")
    require(
        set(collision_by_name) == {f"{name}_trunk_collision" for name in tree_names}
        ,
        "compound link has an unexpected collision",
    )
    require(
        set(visual_by_name) == {f"{name}_{part}_visual" for name in tree_names for part in ("branch", "bark")}
        ,
        "compound link has an unexpected visual",
    )

    tree_source = ET.parse(TREE_LAYOUT).getroot()
    expected_trees = {
        include.findtext("name", ""): (
            include.findtext("uri", "").split("model://", 1)[-1],
            [float(value) for value in include.findtext("pose", "").split()],
        )
        for include in tree_source.findall("./model/include")
        if include.findtext("uri", "") in {"model://pine_tree", "model://oak_tree"}
    }
    require(tree_names == set(expected_trees), "generated tree names differ from the source layout")

    pine_count = oak_count = 0
    z_errors = []
    tree_xy = []
    tree_trunks = []
    tree_disk_errors = []
    tree_disk_reliefs = []
    added_tree_pad_clearances = []
    for name in sorted(tree_names):
        branch = tree_branches[name]
        bark = tree_barks[name]
        collision = tree_collisions[name]
        pose = [float(value) for value in branch.findtext("pose", "").split()]
        bark_pose = [float(value) for value in bark.findtext("pose", "").split()]
        collision_pose = [float(value) for value in collision.findtext("pose", "").split()]
        require(len(pose) == len(bark_pose) == len(collision_pose) == 6, "tree has an invalid absolute pose")
        require(pose == bark_pose, "tree branch and bark poses differ")
        source_tree_type, source_pose = expected_trees[name]
        mesh_uri = branch.findtext("./geometry/mesh/uri", "")
        require(mesh_uri == bark.findtext("./geometry/mesh/uri", ""), "tree mesh pair differs")
        branch_scale = [float(value) for value in branch.findtext("./geometry/mesh/scale", "").split()]
        bark_scale = [float(value) for value in bark.findtext("./geometry/mesh/scale", "").split()]
        require(branch_scale == bark_scale == [2.0, 2.0, 2.0], "tree visual is not uniformly scaled 2x")
        if mesh_uri.endswith("pine_tree.dae"):
            require(source_tree_type == "pine_tree", "generated pine differs from source type")
            pine_count += 1
            base_z = pose[2] - 0.0001
            visual_lift, trunk_offset, radius, length = 0.0, 4.9999, 0.60, 10.0
        elif mesh_uri.endswith("oak_tree.dae"):
            require(source_tree_type == "oak_tree", "generated oak differs from source type")
            oak_count += 1
            base_z = pose[2] - 0.1406
            visual_lift, trunk_offset, radius, length = 0.0703, 6.4297, 0.90, 13.0
        else:
            raise RuntimeError("tree has an unexpected mesh")
        expected_visual_pose = source_pose.copy()
        expected_visual_pose[2] += visual_lift
        require(
            max(abs(pose[index] - expected_visual_pose[index]) for index in range(6)) < 1e-9,
            "tree visual pose differs from the terrain-seated 2x source layout",
        )
        require(branch.findtext("./geometry/mesh/submesh/name") == "Branch", "tree branch submesh changed")
        require(bark.findtext("./geometry/mesh/submesh/name") == "Bark", "tree bark submesh changed")
        require(abs(float(collision.findtext("./geometry/cylinder/radius", "nan")) - radius) < 1e-12, "tree trunk radius changed")
        require(abs(float(collision.findtext("./geometry/cylinder/length", "nan")) - length) < 1e-12, "tree trunk length changed")
        require(max(abs(source_pose[index] - collision_pose[index]) for index in (0, 1, 3, 4, 5)) < 1e-9, "tree trunk pose was not composed into the root link")
        require(abs(collision_pose[2] - (source_pose[2] + trunk_offset)) < 1e-9, "tree trunk height was not composed into the root link")
        collision_bottom = collision_pose[2] - length / 2.0
        disk_samples = disk_terrain_samples(pixels, pose[0], pose[1], radius)
        z_errors.append(max(0.0, min(disk_samples) - base_z))
        tree_disk_errors.append(max(0.0, collision_bottom - min(disk_samples)))
        tree_disk_reliefs.append(max(disk_samples) - min(disk_samples))
        tree_xy.append((pose[0], pose[1]))
        tree_trunks.append((pose[0], pose[1], radius))
        require(abs(pose[0]) <= 140.0 and abs(pose[1]) <= 140.0, "tree crossed the 10 m map boundary margin")
        require(not (-36.1 <= pose[0] <= 36.1 and -24.1 <= pose[1] <= 24.1), "tree intersects central flight clearing")
        tree_index = int(name.rsplit("_", 1)[-1])
        if tree_index >= 73:
            require(not (-42.0 <= pose[0] <= 42.0 and -30.0 <= pose[1] <= 30.0), "added tree intersects expanded flight clearing")
            added_tree_pad_clearances.append(math.hypot(pose[0] + 80.0, pose[1] + 80.0))
    require((pine_count, oak_count) == (216, 72), "pine/oak counts differ")
    require(max(z_errors) < 0.005, "a tree burial allowance exceeded 5 mm")
    require(max(tree_disk_errors) < 1e-5, "a trunk collision bottom floats above terrain")
    require(max(tree_disk_reliefs) < 1.40, "terrain relief under a trunk is excessive")
    require(len(added_tree_pad_clearances) == 216, "forest expansion tree count differs")
    require(min(added_tree_pad_clearances) >= 24.0, "added tree entered the launch-pad clearance")
    min_tree_spacing = min(
        math.hypot(x1 - x2, y1 - y2)
        for index, (x1, y1) in enumerate(tree_xy)
        for x2, y2 in tree_xy[index + 1 :]
    )
    require(min_tree_spacing >= 7.25, "tree spacing fell below 7.25 m")
    min_trunk_clearance = min(
        math.hypot(x1 - x2, y1 - y2) - radius1 - radius2
        for index, (x1, y1, radius1) in enumerate(tree_trunks)
        for x2, y2, radius2 in tree_trunks[index + 1 :]
    )
    require(min_trunk_clearance > 5.0, "2x tree trunk collisions overlap or lost clearance")

    pad_pose = [-80.0, -80.0, 0.08, 0.0, 0.0, 0.0]
    pad_size = [12.0, 12.0, 0.16]
    min_tree_pad_gap = min(
        point_to_oriented_box_gap(x, y, radius, pad_pose, pad_size)
        for x, y, radius in tree_trunks
    )
    require(min_tree_pad_gap > 5.0, "a tree trunk overlaps or crowds the launch pad")

    resource_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (WORLD, TERRAIN / "model.sdf", TERRAIN / "model.config", FOREST / "model.sdf", FOREST / "model.config")
    )
    require("/home/xogus/sim_assets" not in resource_text, "operational asset refers to sim_assets")

    runtime_status = "pending"
    if RUNTIME_LOG.is_file():
        runtime_text = RUNTIME_LOG.read_text(encoding="utf-8", errors="replace")
        current_runtime_markers = {
            "RUNTIME VALIDATION: PASS",
            "runtime_process_tree_stopped=PASS",
            f"world_sha256={sha256(MAP_WORLD)}",
            f"forest_sdf_sha256={sha256(FOREST / 'model.sdf')}",
            f"heightmap_sha256={sha256(HEIGHTMAP)}",
            f"terrain_visual_obj_sha256={sha256(VISUAL_OBJ)}",
            f"terrain_collision_obj_sha256={sha256(COLLISION_OBJ)}",
        }
        if all(marker in runtime_text for marker in current_runtime_markers):
            runtime_status = "PASS"

    lines.extend(
        (
            "world_sdf=1.9",
            "terrain_bounds_m=x[-150,150] y[-150,150] z[0,40]",
            f"summits_m=main:{main_peak:.6f} second:{second_peak:.6f}",
            f"terrain_nonzero_area_ratio={nonzero_ratio:.6f}",
            f"terrain_unique_elevations={unique_elevations}",
            f"terrain_triangle_slope_p99={slope_p99:.6f} terrain_triangle_slope_max={max_slope:.6f}",
            f"launch_pad_terrain_z_m={launch_height:.6f}",
            f"launch_pad_footprint_max_contact_error_m={max(abs(value) for value in pad_samples):.9f}",
            f"terrain_vertices={vertices}",
            f"terrain_triangles={faces}",
            f"heightmap_sha256={sha256(HEIGHTMAP)}",
            f"visual_obj_sha256={sha256(VISUAL_OBJ)}",
            f"collision_obj_sha256={sha256(COLLISION_OBJ)}",
            f"natural_texture_sha256={sha256(TEXTURE)}",
            f"natural_texture_blue_dominant_pixels={blue_dominant}",
            "scene_background=neutral_gray sky=disabled",
            "obstacle_links=1 compound_static=true",
            f"obstacle_collisions={len(collisions)} obstacle_visuals={len(visuals)}",
            f"forest_sdf_sha256={sha256(FOREST / 'model.sdf')}",
            f"trees_total={len(tree_names)} pine={pine_count} oak={oak_count} tree_mesh_visuals={len(tree_branches) + len(tree_barks)}",
            "tree_mesh_scale=2 2 2",
            f"tree_max_terrain_z_error_m={max(z_errors):.9f}",
            f"tree_disk_max_contact_error_m={max(tree_disk_errors):.9f}",
            f"tree_disk_max_relief_m={max(tree_disk_reliefs):.9f}",
            f"tree_min_spacing_m={min_tree_spacing:.6f}",
            f"tree_trunk_min_clearance_m={min_trunk_clearance:.6f}",
            f"tree_launch_pad_min_clearance_m={min_tree_pad_gap:.6f}",
            f"added_tree_launch_pad_clearance_m={min(added_tree_pad_clearances):.6f}",
            "maze_runtime_collisions=0 maze_runtime_visuals=0",
            "operational_sim_assets_references=0",
            f"runtime_gui_validation={runtime_status}",
        )
    )
    return lines


def main() -> None:
    lines = validate()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
