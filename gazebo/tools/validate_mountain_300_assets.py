#!/usr/bin/env python3
"""Static/resource validation for the PX4-ROS2 300 m mountain world."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


GAZEBO_ROOT = Path(__file__).resolve().parents[1]
WORLD = GAZEBO_ROOT / "worlds" / "ugv_drone.world"
TERRAIN = GAZEBO_ROOT / "models" / "ugv_mou_terrain"
FOREST = GAZEBO_ROOT / "models" / "ugv_mou_forest_obstacles"
HEIGHTMAP = TERRAIN / "materials" / "textures" / "mountain_height_300.png"
TEXTURE = TERRAIN / "materials" / "textures" / "natural_ground.png"
VISUAL_OBJ = TERRAIN / "meshes" / "ugv_mou_terrain_visual.obj"
COLLISION_OBJ = TERRAIN / "meshes" / "ugv_mou_terrain_collision.obj"
TREE_LAYOUT = FOREST / "source" / "tree_layout.source.xml"
MAZE_LAYOUT = FOREST / "source" / "maze_layout.source.xml"
LOG = GAZEBO_ROOT / "validation" / "ugv_drone_mountain_300_static.log"
RUNTIME_LOG = GAZEBO_ROOT / "validation" / "runtime" / "mountain_tree288_runtime.log"
EXPECTED_HEIGHTMAP_SHA256 = (
    "d25691a939651c845a4e7e0134b384d45be9eef7382913a2b63e6f2330e93f52"
)
MAZE_TRANSLATION = (-12.0, 0.0)
MAZE_YAW = 1.5708


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
        "gz-sim-imu-system",
        "gz-sim-navsat-system",
    }
    require(required_plugins <= plugins, "one or more Harmonic systems disappeared")
    includes = {include.findtext("uri") for include in world.findall("include")}
    require(
        {
            "model://ugv_mou_terrain",
            "model://ugv_mou_forest_obstacles",
            "model://iris_with_down_camera",
        }
        <= includes,
        "world resource include is missing",
    )
    require(world.findtext("./scene/background") == "0.42 0.42 0.42 1", "background is not neutral")
    require(world.find("./scene/sky") is None, "active sky can reintroduce blue outside the map")
    require(
        world.findtext("./gui/plugin[@name='3D View']/background_color") == "0.42 0.42 0.42",
        "MinimalScene background is not neutral",
    )
    require(world.findtext("./model[@name='drone_launch_pad']/pose") == "-80 -80 0.08 0 0 0", "pad moved")
    iris = next(include for include in world.findall("include") if include.findtext("name") == "mountain_iris")
    require(iris.findtext("pose") == "-80 -80 0.355 0 0 45", "drone spawn moved")
    world_text = WORLD.read_text(encoding="utf-8").lower()
    require("blue_ground" not in world_text and "sim_assets" not in world_text, "world has a stale resource reference")

    pixels = np.asarray(Image.open(HEIGHTMAP).convert("L"), dtype=np.uint8)
    require(pixels.shape == (257, 257), "heightmap is not 257x257")
    require(sha256(HEIGHTMAP) == EXPECTED_HEIGHTMAP_SHA256, "heightmap checksum mismatch")
    edge = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    require(int(edge.max()) == 0, "heightmap edge is nonzero")
    require(int(pixels.max()) == 255, "40 m summit is absent")
    launch_height = terrain_height(pixels, -80.0, -80.0)
    require(abs(launch_height) < 1e-9, "launch pad is not on flat z=0 terrain")
    main_peak = float(pixels.max()) / 255.0 * 40.0
    x_axis = np.linspace(-150.0, 150.0, 257)
    second_peak = float(pixels[:, x_axis > 0.0].max()) / 255.0 * 40.0
    require(main_peak > 39.8 and 19.5 < second_peak < 20.2, "40 m / 20 m summit check failed")

    rgb = np.asarray(Image.open(TEXTURE).convert("RGB"), dtype=np.uint8)
    blue_dominant = int(((rgb[:, :, 2] > rgb[:, :, 0]) & (rgb[:, :, 2] > rgb[:, :, 1])).sum())
    require(blue_dominant == 0, "terrain texture contains blue-dominant pixels")
    mtl_text = (TERRAIN / "meshes" / "ugv_mou_terrain.mtl").read_text(encoding="utf-8")
    require("natural_ground.png" in mtl_text and "blue_ground" not in mtl_text, "terrain MTL is stale")

    visual_stats = obj_stats(VISUAL_OBJ)
    collision_stats = obj_stats(COLLISION_OBJ)
    require(visual_stats == collision_stats, "visual and collision OBJ geometry differ")
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
    require(wall_names == set(wall_visuals), "wall collision/visual sets differ")
    require(len(tree_names) == 288 and len(wall_names) == 72, "obstacle instance counts differ")
    require(len(collisions) == 360, "compound link must contain 288 trunk and 72 wall collisions")
    require(len(visuals) == 648, "compound link must contain 576 tree and 72 wall visuals")
    require(
        set(collision_by_name) == {f"{name}_trunk_collision" for name in tree_names}
        | {f"{name}_collision" for name in wall_names},
        "compound link has an unexpected collision",
    )
    require(
        set(visual_by_name) == {f"{name}_{part}_visual" for name in tree_names for part in ("branch", "bark")}
        | {f"{name}_visual" for name in wall_names},
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

    maze_source = ET.parse(MAZE_LAYOUT).getroot()
    source_wall_entries = []
    expected_walls = {}
    source_signatures = set()
    for include in maze_source.findall("./model/include"):
        wall_type = include.findtext("uri", "").split("model://", 1)[-1]
        source_pose = tuple(float(value) for value in include.findtext("pose", "").split())
        source_wall_entries.append((wall_type, source_pose))
        signature = (wall_type, source_pose)
        if signature in source_signatures:
            continue
        source_signatures.add(signature)
        x, y, z, roll, pitch, yaw = source_pose
        world_x = MAZE_TRANSLATION[0] + math.cos(MAZE_YAW) * x - math.sin(MAZE_YAW) * y
        world_y = MAZE_TRANSLATION[1] + math.sin(MAZE_YAW) * x + math.cos(MAZE_YAW) * y
        world_yaw = math.atan2(math.sin(yaw + MAZE_YAW), math.cos(yaw + MAZE_YAW))
        expected_walls[f"maze_{include.findtext('name', 'wall')}"] = (
            wall_type,
            (world_x, world_y, z + 1.875, roll, pitch, world_yaw),
        )
    require(len(source_wall_entries) == 73, "maze source entry count differs")
    require(len(source_signatures) == 72, "maze source must contain one exact duplicate")
    require(wall_names == set(expected_walls), "generated wall names differ from deduplicated source layout")

    pine_count = oak_count = 0
    z_errors = []
    tree_xy = []
    tree_trunks = []
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
            visual_lift, trunk_offset, radius, length = 0.0, 5.0, 0.60, 10.0
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
        z_errors.append(abs(base_z - terrain_height(pixels, pose[0], pose[1])))
        tree_xy.append((pose[0], pose[1]))
        tree_trunks.append((pose[0], pose[1], radius))
        require(abs(pose[0]) <= 140.0 and abs(pose[1]) <= 140.0, "tree crossed the 10 m map boundary margin")
        require(not (-36.1 <= pose[0] <= 36.1 and -24.1 <= pose[1] <= 24.1), "tree intersects maze exclusion")
        tree_index = int(name.rsplit("_", 1)[-1])
        if tree_index >= 73:
            require(not (-42.0 <= pose[0] <= 42.0 and -30.0 <= pose[1] <= 30.0), "added tree intersects expanded maze clearance")
            added_tree_pad_clearances.append(math.hypot(pose[0] + 80.0, pose[1] + 80.0))
    require((pine_count, oak_count) == (216, 72), "pine/oak counts differ")
    require(max(z_errors) < 1e-5, "a tree is not seated on the terrain")
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

    wall_min_x = wall_min_y = math.inf
    wall_max_x = wall_max_y = -math.inf
    wall_signatures = set()
    for name in sorted(wall_names):
        collision = wall_collisions[name]
        visual = wall_visuals[name]
        pose = [float(value) for value in collision.findtext("pose", "").split()]
        visual_pose = [float(value) for value in visual.findtext("pose", "").split()]
        size = [float(value) for value in collision.findtext("./geometry/box/size", "").split()]
        visual_size = [float(value) for value in visual.findtext("./geometry/box/size", "").split()]
        require(len(pose) == 6 and pose == visual_pose, "wall collision and visual absolute poses differ")
        require(size == visual_size, "wall collision and visual sizes differ")
        require(abs(pose[2] - 1.875) < 1e-9 and max(abs(value) for value in pose[3:5]) < 1e-9, "wall local offset was not composed into the root link")
        _, expected_pose = expected_walls[name]
        require(max(abs(pose[index] - expected_pose[index]) for index in range(6)) < 1e-6, "wall absolute pose differs from composed source layout")
        signature = (tuple(round(value, 6) for value in pose), tuple(size))
        require(signature not in wall_signatures, "maze contains a duplicate physical wall")
        wall_signatures.add(signature)
        hx = abs(math.cos(pose[5])) * size[0] / 2.0 + abs(math.sin(pose[5])) * size[1] / 2.0
        hy = abs(math.sin(pose[5])) * size[0] / 2.0 + abs(math.cos(pose[5])) * size[1] / 2.0
        wall_min_x, wall_max_x = min(wall_min_x, pose[0] - hx), max(wall_max_x, pose[0] + hx)
        wall_min_y, wall_max_y = min(wall_min_y, pose[1] - hy), max(wall_max_y, pose[1] + hy)
        require(size[2] == 3.75, "maze wall height is not 3.75 m")
    require(wall_min_x > -28.1 and wall_max_x < 28.1, "maze x bounds changed")
    require(wall_min_y > -16.1 and wall_max_y < 16.1, "maze y bounds changed")

    resource_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (WORLD, TERRAIN / "model.sdf", TERRAIN / "model.config", FOREST / "model.sdf", FOREST / "model.config")
    )
    require("/home/xogus/sim_assets" not in resource_text, "operational asset refers to sim_assets")

    runtime_status = "pending"
    if RUNTIME_LOG.is_file():
        runtime_text = RUNTIME_LOG.read_text(encoding="utf-8", errors="replace")
        current_forest_marker = f"forest_sdf_sha256={sha256(FOREST / 'model.sdf')}"
        if "RUNTIME VALIDATION: PASS" in runtime_text and current_forest_marker in runtime_text:
            runtime_status = "PASS"

    lines.extend(
        (
            "world_sdf=1.9",
            "terrain_bounds_m=x[-150,150] y[-150,150] z[0,40]",
            f"summits_m=main:{main_peak:.6f} second:{second_peak:.6f}",
            f"launch_pad_terrain_z_m={launch_height:.6f}",
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
            f"tree_min_spacing_m={min_tree_spacing:.6f}",
            f"tree_trunk_min_clearance_m={min_trunk_clearance:.6f}",
            f"added_tree_launch_pad_clearance_m={min(added_tree_pad_clearances):.6f}",
            f"maze_source_entries=73 maze_unique_walls={len(wall_names)} height_m=3.75",
            f"maze_aabb_m=x[{wall_min_x:.6f},{wall_max_x:.6f}] y[{wall_min_y:.6f},{wall_max_y:.6f}]",
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
