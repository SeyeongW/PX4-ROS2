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
from scipy.ndimage import distance_transform_edt


GAZEBO_ROOT = Path(__file__).resolve().parents[1]
TERRAIN_ROOT = GAZEBO_ROOT / "models" / "ugv_mou_terrain"
FOREST_ROOT = GAZEBO_ROOT / "models" / "ugv_mou_forest_obstacles"
HEIGHTMAP = TERRAIN_ROOT / "materials" / "textures" / "mountain_height_300.png"
NATURAL_TEXTURE = TERRAIN_ROOT / "materials" / "textures" / "natural_ground.png"
VISUAL_OBJ = TERRAIN_ROOT / "meshes" / "ugv_mou_terrain_visual.obj"
COLLISION_OBJ = TERRAIN_ROOT / "meshes" / "ugv_mou_terrain_collision.obj"
MTL = TERRAIN_ROOT / "meshes" / "ugv_mou_terrain.mtl"
TREE_LAYOUT = FOREST_ROOT / "source" / "tree_layout.source.xml"
FOREST_SDF = FOREST_ROOT / "model.sdf"

MAP_SIZE_M = 300.0
HEIGHT_SCALE_M = 40.0
PIXELS = 257
TERRAIN_SEED = 20260711
EXPECTED_HEIGHTMAP_SHA256 = (
    "dea9d2ca2f6a4f357037e0f626a90ab2bfb815d6b1185659cc8d53b1ed63548c"
)
MAIN_CENTER = (-75.0, 75.0)
MAIN_RADIUS = 75.0
MAIN_AMPLITUDE = 40.0
SECOND_CENTER = (55.0, 80.0)
SECOND_RADIUS = 70.0
SECOND_AMPLITUDE = 20.0
PROFILE_EXPONENT = 0.10
# Keep the already validated low central flight corridor after removing the
# artificial maze.  This is terrain shaping only; no wall visual/collision is
# generated anywhere in the runtime map.
CENTRAL_CLEARING_RADII_M = (42.0, 30.0)
LAUNCH_CENTER = (-80.0, -80.0)
LAUNCH_FLAT_RADIUS_M = 11.0
LAUNCH_TRANSITION_RADIUS_M = 24.0
EDGE_TAPER_M = 18.0
TREE_MESH_SCALE = (2.0, 2.0, 2.0)
TREE_GROUND_COMPENSATION = {"pine_tree": 0.0001, "oak_tree": 0.0703}
TREE_TRUNK_RADIUS = {"pine_tree": 0.60, "oak_tree": 0.90}

# Broad, low ridges surround the retained 40 m / 20 m summits.  Every analytic
# cone is gentler than the global cap below; the cap also guarantees a gradual
# connection to the exact-flat trailer corridor, launch pad and map boundary.
# (center_x, center_y, half_length, half_width, yaw, amplitude)
RIDGES = (
    (-5.0, 126.0, 132.0, 38.0, 0.02, 12.0),
    (-124.0, 24.0, 82.0, 39.0, 1.39, 15.0),
    (110.0, 43.0, 84.0, 38.0, 1.87, 13.0),
    (98.0, -72.0, 90.0, 42.0, 2.28, 12.0),
    (3.0, -124.0, 112.0, 36.0, 0.08, 10.0),
    (-132.0, -13.0, 70.0, 35.0, 1.48, 11.0),
)

# An elliptical exact-flat corridor avoids a sharp rectangular corner in the
# triangle mesh while preserving ample room around the complete cross route.
# The 40 m peak and zero boundary make roughly 0.54 rise/run unavoidable;
# 0.60 leaves a small deterministic margin without clipping either summit.
MAX_DESIGN_GRADE = 0.60

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


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def smooth_hill_profile(normalized_radius: np.ndarray) -> np.ndarray:
    """Compact C1 radial mound used by the original 40 m / 20 m hills."""

    radius = np.clip(normalized_radius, 0.0, 1.0)
    samples = np.linspace(0.0, 1.0, 16385)
    density = np.power(samples * (1.0 - samples), PROFILE_EXPONENT)
    cdf_samples = np.zeros_like(samples)
    cdf_samples[1:] = np.cumsum(
        0.5 * (density[:-1] + density[1:]) * np.diff(samples)
    )
    cdf_samples /= cdf_samples[-1]
    cdf = np.interp(radius, samples, cdf_samples)
    return np.where(normalized_radius < 1.0, 1.0 - cdf, 0.0)


def interpolated_value_noise(rng: np.random.Generator, controls: int) -> np.ndarray:
    """Pure-NumPy low-frequency value noise, stable across image libraries."""

    control = rng.uniform(-1.0, 1.0, size=(controls, controls))
    source_axis = np.linspace(0.0, PIXELS - 1.0, controls)
    target_axis = np.arange(PIXELS, dtype=np.float64)
    horizontal = np.vstack(
        [np.interp(target_axis, source_axis, row) for row in control]
    )
    return np.vstack(
        [np.interp(target_axis, source_axis, horizontal[:, column]) for column in range(PIXELS)]
    ).T


def terrain_height(heights: np.ndarray, x: float, y: float) -> float:
    """Bilinearly sample a 257x257 world-height array in metres."""

    column = (x + MAP_SIZE_M / 2.0) / MAP_SIZE_M * (PIXELS - 1)
    row = (MAP_SIZE_M / 2.0 - y) / MAP_SIZE_M * (PIXELS - 1)
    column = min(max(column, 0.0), PIXELS - 1.0)
    row = min(max(row, 0.0), PIXELS - 1.0)
    c0, r0 = int(math.floor(column)), int(math.floor(row))
    c1, r1 = min(c0 + 1, PIXELS - 1), min(r0 + 1, PIXELS - 1)
    tx, ty = column - c0, row - r0
    return float(
        (1.0 - tx) * (1.0 - ty) * heights[r0, c0]
        + tx * (1.0 - ty) * heights[r0, c1]
        + (1.0 - tx) * ty * heights[r1, c0]
        + tx * ty * heights[r1, c1]
    )


def source_tree_specs() -> list[tuple[str, float, float]]:
    specs = []
    root = ET.parse(TREE_LAYOUT).getroot()
    for include in root.findall("./model/include"):
        tree_type = include.findtext("uri", "").removeprefix("model://")
        if tree_type not in TREE_TRUNK_RADIUS:
            continue
        pose = parse_pose(include.findtext("pose", ""), include.findtext("name", "tree"))
        specs.append((tree_type, pose[0], pose[1]))
    if len(specs) != 288:
        raise RuntimeError(f"expected 288 tree source poses, got {len(specs)}")
    return specs


def build_heightmap() -> np.ndarray:
    """Generate retained summits, broad ridges and slope-bounded transitions."""

    x_axis = np.linspace(-MAP_SIZE_M / 2.0, MAP_SIZE_M / 2.0, PIXELS)
    y_axis = np.linspace(MAP_SIZE_M / 2.0, -MAP_SIZE_M / 2.0, PIXELS)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    main_distance = np.hypot(x_grid - MAIN_CENTER[0], y_grid - MAIN_CENTER[1])
    second_distance = np.hypot(x_grid - SECOND_CENTER[0], y_grid - SECOND_CENTER[1])
    main = np.maximum(MAIN_AMPLITUDE - 0.56 * main_distance, 0.0)
    second = np.maximum(SECOND_AMPLITUDE - 0.34 * second_distance, 0.0)
    retained_hills = np.maximum(main, second)

    ridges = np.zeros_like(retained_hills)
    for center_x, center_y, half_length, half_width, yaw, amplitude in RIDGES:
        dx, dy = x_grid - center_x, y_grid - center_y
        along = math.cos(yaw) * dx + math.sin(yaw) * dy
        across = -math.sin(yaw) * dx + math.cos(yaw) * dy
        radius = np.hypot(along / half_length, across / half_width)
        ridges = np.maximum(
            ridges,
            amplitude * np.maximum(1.0 - radius, 0.0),
        )
    # Nothing east of x=0 may eclipse the retained 20 m summit; likewise all
    # western additions remain below the retained 40 m summit.
    ridges = np.minimum(ridges, np.where(x_grid > 0.0, 19.0, 36.0))
    heights = np.maximum(retained_hills, ridges)

    radius_x, radius_y = CENTRAL_CLEARING_RADII_M
    launch_distance = np.hypot(x_grid - LAUNCH_CENTER[0], y_grid - LAUNCH_CENTER[1])
    protected_zero = (
        (((x_grid / radius_x) ** 2 + (y_grid / radius_y) ** 2) <= 1.0)
        | (launch_distance <= LAUNCH_FLAT_RADIUS_M)
    )
    protected_zero[0, :] = protected_zero[-1, :] = True
    protected_zero[:, 0] = protected_zero[:, -1] = True
    spacing = MAP_SIZE_M / (PIXELS - 1)
    distance_to_zero = distance_transform_edt(~protected_zero) * spacing
    heights = np.minimum(heights, MAX_DESIGN_GRADE * distance_to_zero)
    heights[protected_zero] = 0.0
    heights = np.clip(heights, 0.0, HEIGHT_SCALE_M)

    pixels = np.rint(heights / HEIGHT_SCALE_M * 255.0).astype(np.uint8)
    pixels[0, :] = pixels[-1, :] = 0
    pixels[:, 0] = pixels[:, -1] = 0
    # Preserve both canonical summit codes after all protective operations.
    main_column = int(round((MAIN_CENTER[0] + 150.0) / MAP_SIZE_M * 256.0))
    main_row = int(round((150.0 - MAIN_CENTER[1]) / MAP_SIZE_M * 256.0))
    pixels[main_row, main_column] = 255
    second_column = int(round((SECOND_CENTER[0] + 150.0) / MAP_SIZE_M * 256.0))
    second_row = int(round((150.0 - SECOND_CENTER[1]) / MAP_SIZE_M * 256.0))
    pixels[second_row, second_column] = 128
    return pixels


def write_heightmap(pixels: np.ndarray) -> None:
    Image.fromarray(pixels, mode="L").save(HEIGHTMAP, optimize=True)
    if EXPECTED_HEIGHTMAP_SHA256 and sha256(HEIGHTMAP) != EXPECTED_HEIGHTMAP_SHA256:
        raise RuntimeError("generated mountain heightmap checksum is not the pinned asset")
    edge = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if pixels.shape != (PIXELS, PIXELS) or int(edge.max()) != 0:
        raise RuntimeError("mountain heightmap shape or zero-height boundary is invalid")


def seat_tree_layout(pixels: np.ndarray) -> None:
    heights = pixels.astype(np.float64) / 255.0 * HEIGHT_SCALE_M
    document = ET.parse(TREE_LAYOUT)
    for include in document.getroot().findall("./model/include"):
        tree_type = include.findtext("uri", "").removeprefix("model://")
        if tree_type not in TREE_GROUND_COMPENSATION:
            continue
        name = include.findtext("name", "tree")
        pose = list(parse_pose(include.findtext("pose", ""), name))
        # Do not mutate the terrain into hundreds of tiny flat shelves: their
        # former 0.9 m blend bands created the map's worst artificial cliffs.
        # Instead, bury the static trunk bottom to the minimum terrain sampled
        # beneath its collision disk.  The visual remains naturally seated and
        # no part of the cylindrical collision can float above the slope.
        radius = TREE_TRUNK_RADIUS[tree_type]
        samples = [terrain_height(heights, pose[0], pose[1])]
        for sample_radius in np.linspace(radius / 4.0, radius, 4):
            for angle in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
                samples.append(
                    terrain_height(
                        heights,
                        pose[0] + float(sample_radius) * math.cos(float(angle)),
                        pose[1] + float(sample_radius) * math.sin(float(angle)),
                    )
                )
        # A 3 mm burial allowance covers extrema between the deterministic
        # radial samples without creating a visible gap.
        pose[2] = min(samples) - 0.003 + TREE_GROUND_COMPENSATION[tree_type]
        include.find("pose").text = format_pose(tuple(pose))  # type: ignore[arg-type]
    ET.indent(document, space="  ")
    document.write(TREE_LAYOUT, encoding="UTF-8", xml_declaration=True)


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
        # The visual mesh extends 0.1 mm below its origin.  The source pose
        # compensates for that visual extent, while this matching -0.1 mm
        # collision offset keeps the cylinder bottom on the exact terrain.
        trunk_pose: Pose = (0.0, 0.0, 4.9999, 0.0, 0.0, 0.0)
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


def write_forest_sdf() -> dict[str, int]:
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

    if tree_counts != {"pine_tree": 216, "oak_tree": 72}:
        raise RuntimeError(f"unexpected tree counts: {tree_counts}")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(FOREST_SDF, encoding="UTF-8", xml_declaration=True)
    return tree_counts


def main() -> None:
    if not NATURAL_TEXTURE.is_file():
        raise RuntimeError(f"missing natural terrain texture: {NATURAL_TEXTURE}")
    pixels = build_heightmap()
    write_heightmap(pixels)
    seat_tree_layout(pixels)
    write_mtl()
    write_terrain_obj(VISUAL_OBJ, pixels)
    shutil.copy2(VISUAL_OBJ, COLLISION_OBJ)
    tree_counts = write_forest_sdf()
    print(f"heightmap={HEIGHTMAP} sha256={sha256(HEIGHTMAP)}")
    print(f"terrain_peak_m={pixels.max() / 255.0 * HEIGHT_SCALE_M:.6f}")
    print(f"visual_obj={VISUAL_OBJ} sha256={sha256(VISUAL_OBJ)}")
    print(f"collision_obj={COLLISION_OBJ} sha256={sha256(COLLISION_OBJ)}")
    print(f"trees={sum(tree_counts.values())} counts={tree_counts}")
    print("maze_walls_unique=0")
    print(f"forest_sdf={FOREST_SDF} sha256={sha256(FOREST_SDF)}")


if __name__ == "__main__":
    main()
