#!/usr/bin/env python3
"""Build the self-contained 300 m Harmonic mountain assets.

The checked-in inputs live below ``gazebo/models``.  This script never reads
from the original ``sim_assets`` workspace, so the PX4-ROS2 copy remains
portable after the import is complete.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


GAZEBO_ROOT = Path(__file__).resolve().parents[1]
TERRAIN_ROOT = GAZEBO_ROOT / "models" / "ugv_mou_terrain"
FOREST_ROOT = GAZEBO_ROOT / "models" / "ugv_mou_forest_obstacles"
HEIGHTMAP = TERRAIN_ROOT / "materials" / "textures" / "mountain_height_300.png"
NATURAL_TEXTURE = TERRAIN_ROOT / "materials" / "textures" / "natural_ground.png"
VISUAL_OBJ = TERRAIN_ROOT / "meshes" / "ugv_mou_terrain_visual.obj"
COLLISION_OBJ = TERRAIN_ROOT / "meshes" / "ugv_mou_terrain_collision.obj"
MTL = TERRAIN_ROOT / "meshes" / "ugv_mou_terrain.mtl"
TREE_LAYOUT = FOREST_ROOT / "source" / "tree_layout.source.xml"
MAZE_LAYOUT = FOREST_ROOT / "source" / "maze_layout.source.xml"
FOREST_SDF = FOREST_ROOT / "model.sdf"

MAP_SIZE_M = 300.0
HEIGHT_SCALE_M = 40.0
EXPECTED_HEIGHTMAP_SHA256 = (
    "d25691a939651c845a4e7e0134b384d45be9eef7382913a2b63e6f2330e93f52"
)
MAZE_TRANSLATION = (-12.0, 0.0)
MAZE_YAW = 1.5708
WALL_SIZE = {
    "unit_wall": (4.0, 0.15, 3.75),
    "half_wall": (2.0, 0.15, 3.75),
    "long_wall": (32.0, 0.15, 3.75),
}
TREE_MESH_SCALE = (2.0, 2.0, 2.0)

Pose = tuple[float, float, float, float, float, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = text
    return node


def parse_pose(text: str, description: str) -> Pose:
    values = tuple(float(value) for value in text.split())
    if len(values) != 6:
        raise RuntimeError(f"invalid {description} pose: {text!r}")
    return values  # type: ignore[return-value]


def rotation_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, ...], ...]:
    """Return the SDF fixed-axis Rz(yaw) * Ry(pitch) * Rx(roll) matrix."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def compose_pose(parent: Pose, local: Pose) -> Pose:
    """Compose two SDF poses so child elements can live in one root link."""

    parent_rotation = rotation_matrix(*parent[3:])
    local_rotation = rotation_matrix(*local[3:])
    rotated_local = tuple(
        sum(parent_rotation[row][column] * local[column] for column in range(3))
        for row in range(3)
    )
    translation = tuple(parent[index] + rotated_local[index] for index in range(3))
    rotation = tuple(
        tuple(
            sum(parent_rotation[row][axis] * local_rotation[axis][column] for axis in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1e-10:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        yaw = 0.0
    return (*translation, roll, pitch, yaw)


def format_pose(pose: Pose) -> str:
    values = (0.0 if abs(value) < 0.5e-9 else value for value in pose)
    return " ".join(f"{value:.6f}" for value in values)


def load_heightmap() -> np.ndarray:
    if sha256(HEIGHTMAP) != EXPECTED_HEIGHTMAP_SHA256:
        raise RuntimeError("heightmap checksum differs from the validated 40 m / 20 m asset")
    pixels = np.asarray(Image.open(HEIGHTMAP).convert("L"), dtype=np.uint8)
    if pixels.shape != (257, 257):
        raise RuntimeError(f"expected a 257x257 heightmap, got {pixels.shape}")
    edge = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if int(edge.max()) != 0:
        raise RuntimeError("heightmap boundary must be zero to avoid a vertical map-edge skirt")
    return pixels


def write_mtl() -> None:
    MTL.write_text(
        "\n".join(
            (
                "newmtl ugv_mou_natural_ground",
                "Ka 0.34 0.38 0.22",
                "Kd 0.72 0.76 0.48",
                "Ks 0.02 0.02 0.02",
                "Ns 4.0",
                "d 1.0",
                "illum 2",
                "map_Kd ../materials/textures/natural_ground.png",
                "",
            )
        ),
        encoding="utf-8",
    )


def write_terrain_obj(path: Path, pixels: np.ndarray) -> None:
    rows, cols = pixels.shape
    spacing_x = MAP_SIZE_M / (cols - 1)
    spacing_y = MAP_SIZE_M / (rows - 1)
    heights = pixels.astype(np.float64) / 255.0 * HEIGHT_SCALE_M
    dz_drow, dz_dx = np.gradient(heights, spacing_y, spacing_x)
    dz_dy = -dz_drow
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(heights)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("mtllib ugv_mou_terrain.mtl\n")
        stream.write("o ugv_mou_terrain_300m\n")
        for row in range(rows):
            y = MAP_SIZE_M / 2.0 - row * spacing_y
            for col in range(cols):
                x = -MAP_SIZE_M / 2.0 + col * spacing_x
                stream.write(f"v {x:.6f} {y:.6f} {heights[row, col]:.6f}\n")
        # Repeat the 512 px natural texture every 12 m rather than stretching
        # a single image over the complete 300 m terrain.
        repeats = MAP_SIZE_M / 12.0
        for row in range(rows):
            v = (1.0 - row / (rows - 1)) * repeats
            for col in range(cols):
                u = col / (cols - 1) * repeats
                stream.write(f"vt {u:.6f} {v:.6f}\n")
        for row in range(rows):
            for col in range(cols):
                stream.write(f"vn {nx[row, col]:.8f} {ny[row, col]:.8f} {nz[row, col]:.8f}\n")
        stream.write("usemtl ugv_mou_natural_ground\n")
        stream.write("s 1\n")
        for row in range(rows - 1):
            for col in range(cols - 1):
                a = row * cols + col + 1
                b = a + 1
                d = (row + 1) * cols + col + 1
                e = d + 1
                stream.write(f"f {a}/{a}/{a} {d}/{d}/{d} {b}/{b}/{b}\n")
                stream.write(f"f {b}/{b}/{b} {d}/{d}/{d} {e}/{e}/{e}\n")


def add_surface(collision: ET.Element) -> None:
    surface = child(collision, "surface")
    friction = child(surface, "friction")
    ode = child(friction, "ode")
    child(ode, "mu", "0.95")
    child(ode, "mu2", "0.95")
    contact = child(surface, "contact")
    contact_ode = child(contact, "ode")
    child(contact_ode, "kp", "1000000")
    child(contact_ode, "kd", "1")
    child(contact_ode, "max_vel", "0.2")
    child(contact_ode, "min_depth", "0.001")


def add_pbr_material(visual: ET.Element, texture: str, branch: bool) -> None:
    material = child(visual, "material")
    if branch:
        child(material, "ambient", "0.18 0.34 0.12 1")
        child(material, "diffuse", "0.34 0.58 0.22 1")
    else:
        child(material, "ambient", "0.22 0.13 0.07 1")
        child(material, "diffuse", "0.38 0.23 0.12 1")
    child(material, "specular", "0.02 0.02 0.02 1")
    pbr = child(material, "pbr")
    metal = child(pbr, "metal")
    child(metal, "albedo_map", f"model://ugv_mou_forest_obstacles/materials/textures/{texture}")
    child(metal, "metalness", "0.0")
    child(metal, "roughness", "0.92")


def add_tree_obstacles(link: ET.Element, include: ET.Element) -> str:
    name = include.findtext("name", "tree")
    tree_type = include.findtext("uri", "").split("model://", 1)[-1]
    if tree_type not in {"pine_tree", "oak_tree"}:
        raise RuntimeError(f"unsupported tree type: {tree_type}")
    tree_pose = parse_pose(include.findtext("pose", ""), name)

    collision = child(link, "collision", name=f"{name}_trunk_collision")
    if tree_type == "pine_tree":
        trunk_pose: Pose = (0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        visual_pose = tree_pose
        radius, length = "0.60", "10.0"
    else:
        # The oak mesh extends 0.0703 m below its origin at 1x.  Its source
        # pose already compensates for that amount, so lift the 2x visual by
        # one additional 0.0703 m and measure the trunk from terrain contact.
        trunk_pose = (0.0, 0.0, 6.4297, 0.0, 0.0, 0.0)
        visual_pose = compose_pose(tree_pose, (0.0, 0.0, 0.0703, 0.0, 0.0, 0.0))
        radius, length = "0.90", "13.0"
    child(collision, "pose", format_pose(compose_pose(tree_pose, trunk_pose)))
    geometry = child(collision, "geometry")
    cylinder = child(geometry, "cylinder")
    child(cylinder, "radius", radius)
    child(cylinder, "length", length)
    add_surface(collision)

    mesh_name = "pine_tree.dae" if tree_type == "pine_tree" else "oak_tree.dae"
    parts = (
        ("branch", "Branch", f"{'pine' if tree_type == 'pine_tree' else 'oak'}_branch.png", True),
        ("bark", "Bark", f"{'pine' if tree_type == 'pine_tree' else 'oak'}_bark.png", False),
    )
    for visual_name, submesh_name, texture, is_branch in parts:
        visual = child(link, "visual", name=f"{name}_{visual_name}_visual")
        child(visual, "pose", format_pose(visual_pose))
        child(visual, "cast_shadows", "true")
        geometry = child(visual, "geometry")
        mesh = child(geometry, "mesh")
        child(mesh, "uri", f"model://ugv_mou_forest_obstacles/meshes/{mesh_name}")
        child(mesh, "scale", " ".join(f"{value:g}" for value in TREE_MESH_SCALE))
        submesh = child(mesh, "submesh")
        child(submesh, "name", submesh_name)
        child(submesh, "center", "false")
        add_pbr_material(visual, texture, is_branch)
    return tree_type


def add_wall_obstacles(link: ET.Element, include: ET.Element) -> str:
    name = include.findtext("name", "wall")
    wall_type = include.findtext("uri", "").split("model://", 1)[-1]
    if wall_type not in WALL_SIZE:
        raise RuntimeError(f"unsupported wall type: {wall_type}")
    source_pose = parse_pose(include.findtext("pose", ""), name)
    maze_pose: Pose = (*MAZE_TRANSLATION, 0.0, 0.0, 0.0, MAZE_YAW)
    wall_base_pose = compose_pose(maze_pose, source_pose)
    wall_geometry_pose = compose_pose(wall_base_pose, (0.0, 0.0, 1.875, 0.0, 0.0, 0.0))
    size = WALL_SIZE[wall_type]
    size_text = " ".join(f"{value:g}" for value in size)

    collision = child(link, "collision", name=f"maze_{name}_collision")
    child(collision, "pose", format_pose(wall_geometry_pose))
    geometry = child(collision, "geometry")
    box = child(geometry, "box")
    child(box, "size", size_text)
    add_surface(collision)
    visual = child(link, "visual", name=f"maze_{name}_visual")
    child(visual, "pose", format_pose(wall_geometry_pose))
    geometry = child(visual, "geometry")
    box = child(geometry, "box")
    child(box, "size", size_text)
    material = child(visual, "material")
    child(material, "ambient", "0.30 0.27 0.22 1")
    child(material, "diffuse", "0.48 0.43 0.34 1")
    child(material, "specular", "0.03 0.03 0.03 1")
    return wall_type


def write_forest_sdf() -> tuple[dict[str, int], dict[str, int], int]:
    root = ET.Element("sdf", version="1.9")
    model = child(root, "model", name="ugv_mou_forest_obstacles")
    child(model, "static", "true")
    # Bullet Featherstone accepts this static compound body as one rooted
    # collision tree.  Multiple unjointed links would be interpreted as
    # independent floating subtrees and disabled at runtime.
    obstacle_link = child(model, "link", name="obstacles")

    tree_counts = {"pine_tree": 0, "oak_tree": 0}
    tree_root = ET.parse(TREE_LAYOUT).getroot()
    for include in tree_root.findall("./model/include"):
        uri = include.findtext("uri", "")
        if uri not in {"model://pine_tree", "model://oak_tree"}:
            continue
        tree_type = add_tree_obstacles(obstacle_link, include)
        tree_counts[tree_type] += 1

    wall_counts = {key: 0 for key in WALL_SIZE}
    maze_root = ET.parse(MAZE_LAYOUT).getroot()
    seen_walls: set[tuple[str, tuple[float, ...]]] = set()
    source_wall_count = 0
    for include in maze_root.findall("./model/include"):
        source_wall_count += 1
        wall_type = include.findtext("uri", "").split("model://", 1)[-1]
        pose = tuple(float(value) for value in include.findtext("pose", "").split())
        signature = (wall_type, pose)
        if signature in seen_walls:
            continue
        seen_walls.add(signature)
        wall_type = add_wall_obstacles(obstacle_link, include)
        wall_counts[wall_type] += 1

    if tree_counts != {"pine_tree": 216, "oak_tree": 72}:
        raise RuntimeError(f"unexpected tree counts: {tree_counts}")
    if source_wall_count != 73:
        raise RuntimeError(f"unexpected source maze entry count: {source_wall_count}")
    if wall_counts != {"unit_wall": 64, "half_wall": 2, "long_wall": 6}:
        raise RuntimeError(f"unexpected maze wall counts: {wall_counts}")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(FOREST_SDF, encoding="UTF-8", xml_declaration=True)
    return tree_counts, wall_counts, source_wall_count


def main() -> None:
    if not NATURAL_TEXTURE.is_file():
        raise RuntimeError(f"missing natural terrain texture: {NATURAL_TEXTURE}")
    pixels = load_heightmap()
    write_mtl()
    write_terrain_obj(VISUAL_OBJ, pixels)
    shutil.copy2(VISUAL_OBJ, COLLISION_OBJ)
    tree_counts, wall_counts, source_wall_count = write_forest_sdf()
    print(f"heightmap={HEIGHTMAP} sha256={sha256(HEIGHTMAP)}")
    print(f"terrain_peak_m={pixels.max() / 255.0 * HEIGHT_SCALE_M:.6f}")
    print(f"visual_obj={VISUAL_OBJ} sha256={sha256(VISUAL_OBJ)}")
    print(f"collision_obj={COLLISION_OBJ} sha256={sha256(COLLISION_OBJ)}")
    print(f"trees={sum(tree_counts.values())} counts={tree_counts}")
    print(
        f"maze_walls_unique={sum(wall_counts.values())} "
        f"source_entries={source_wall_count} counts={wall_counts}"
    )
    print(f"forest_sdf={FOREST_SDF} sha256={sha256(FOREST_SDF)}")


if __name__ == "__main__":
    main()
