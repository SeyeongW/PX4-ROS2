#!/usr/bin/env python3
"""Generate the dedicated ``drone_cju_track`` Gazebo model pair.

This generator intentionally writes only these new model directories:

* ``models/drone_cju_track_running_track``
* ``models/drone_cju_track_stadium``

The older ``running_track`` and ``cheongju_university_stadium`` models remain
untouched. Shared primitive-building helpers are imported from their generators
to keep dimensions consistent.

All geometry is generated locally. The running surface is one connected
procedural OBJ annulus, and the covered grandstand roof is a procedural, closed
OBJ solid with 32 curved sections. No downloaded mesh, image, or texture is
used.
"""

import math
from pathlib import Path

import gen_cju_stadium_model as base_stadium
import gen_track_model as base_track


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
TRACK_MODEL_NAME = "drone_cju_track_running_track"
STADIUM_MODEL_NAME = "drone_cju_track_stadium"
TRACK_MODEL_DIR = MODELS_DIR / TRACK_MODEL_NAME
STADIUM_MODEL_DIR = MODELS_DIR / STADIUM_MODEL_NAME
TRACK_MESH_RELATIVE_PATH = Path("meshes") / "running_track_surface.obj"
TRACK_MTL_RELATIVE_PATH = Path("meshes") / "running_track_surface.mtl"
ROOF_MESH_RELATIVE_PATH = Path("meshes") / "grandstand_roof.obj"
ROOF_MTL_RELATIVE_PATH = Path("meshes") / "grandstand_roof.mtl"

CONTINUOUS_TRACK_RED_RGBA = (0.56, 0.14, 0.10, 1.0)
TRACK_MATERIAL_NAME = "continuous_track_red"
ROOF_MATERIAL_NAME = "cju_roof_blue"

# Each half-circle gets 96 angular intervals. The two exact straight edges and
# both sampled half-circles form a single connected top surface with no
# overlapping chord boxes or gaps.
TRACK_ARC_SEGMENTS = 96
TRACK_SURFACE_Z_M = (
    base_track.SURFACE_Z_M + base_track.SURFACE_HEIGHT_M / 2.0
)

# Smooth grandstand canopy dimensions. There are 32 intervals and therefore
# 33 cross sections, comfortably above the requested 24-section minimum.
ROOF_SECTION_COUNT = 32
ROOF_X_MIN_M = 85.5
ROOF_X_MAX_M = 108.5
ROOF_Y_MIN_M = -29.0
ROOF_Y_MAX_M = +29.0
ROOF_EDGE_TOP_Z_M = 7.25
ROOF_ARCH_RISE_M = 1.85
ROOF_THICKNESS_M = 0.22


def build_mtl(material_name, rgba):
    """Return a local Wavefront material matching the SDF visual colour."""

    red, green, blue, alpha = rgba
    return f"""# Procedurally generated local material
newmtl {material_name}
Ka {red:.6f} {green:.6f} {blue:.6f}
Kd {red:.6f} {green:.6f} {blue:.6f}
Ks 0.040000 0.040000 0.040000
Ns 8.000000
d {alpha:.6f}
illum 2
"""


def make_track_surface_mesh():
    """Return one connected, non-overlapping stadium-ring top surface.

    The station sequence travels counter-clockwise along the lower straight,
    right half-circle, upper straight, and left half-circle. Each station has
    exactly one inner and one outer vertex. Adjacent station pairs therefore
    form one quad (two triangles), including the final-to-first closure.
    """

    straight_half = base_track.STRAIGHT_LENGTH_M / 2.0
    inner_radius = (
        base_track.CURVE_RADIUS_M - base_track.TRACK_HALF_WIDTH_M
    )
    outer_radius = (
        base_track.CURVE_RADIUS_M + base_track.TRACK_HALF_WIDTH_M
    )

    # Each item is (inner_xyz, outer_xyz).
    stations = [
        (
            (-straight_half, -inner_radius, TRACK_SURFACE_Z_M),
            (-straight_half, -outer_radius, TRACK_SURFACE_Z_M),
        ),
        (
            (+straight_half, -inner_radius, TRACK_SURFACE_Z_M),
            (+straight_half, -outer_radius, TRACK_SURFACE_Z_M),
        ),
    ]

    # Right curve: lower-right to upper-right, including the upper endpoint.
    for index in range(1, TRACK_ARC_SEGMENTS + 1):
        theta = -math.pi / 2.0 + index * math.pi / TRACK_ARC_SEGMENTS
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        stations.append(
            (
                (
                    +straight_half + inner_radius * cos_theta,
                    inner_radius * sin_theta,
                    TRACK_SURFACE_Z_M,
                ),
                (
                    +straight_half + outer_radius * cos_theta,
                    outer_radius * sin_theta,
                    TRACK_SURFACE_Z_M,
                ),
            )
        )

    # The upper straight ends at the upper-left curve tangent.
    stations.append(
        (
            (-straight_half, +inner_radius, TRACK_SURFACE_Z_M),
            (-straight_half, +outer_radius, TRACK_SURFACE_Z_M),
        )
    )

    # Left curve: upper-left toward lower-left. The final lower endpoint is the
    # first station and is deliberately omitted to avoid duplicate vertices.
    for index in range(1, TRACK_ARC_SEGMENTS):
        theta = math.pi / 2.0 + index * math.pi / TRACK_ARC_SEGMENTS
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        stations.append(
            (
                (
                    -straight_half + inner_radius * cos_theta,
                    inner_radius * sin_theta,
                    TRACK_SURFACE_Z_M,
                ),
                (
                    -straight_half + outer_radius * cos_theta,
                    outer_radius * sin_theta,
                    TRACK_SURFACE_Z_M,
                ),
            )
        )

    expected_station_count = 2 * TRACK_ARC_SEGMENTS + 2
    if len(stations) != expected_station_count:
        raise ValueError(
            f"track station count {len(stations)} != "
            f"{expected_station_count}"
        )

    vertices = []
    for inner_vertex, outer_vertex in stations:
        vertices.extend((inner_vertex, outer_vertex))

    faces = []
    station_count = len(stations)
    for station_index in range(station_count):
        next_station = (station_index + 1) % station_count
        inner = station_index * 2 + 1
        outer = station_index * 2 + 2
        next_inner = next_station * 2 + 1
        next_outer = next_station * 2 + 2

        # Winding points upward (+z) on both triangles.
        faces.extend(
            (
                (inner, outer, next_outer),
                (inner, next_outer, next_inner),
            )
        )

    topology = validate_track_surface_mesh(vertices, faces)
    lines = [
        "# Procedurally generated continuous 400 m running-track surface",
        "# One connected topological annulus; no external source asset",
        f"mtllib {TRACK_MTL_RELATIVE_PATH.name}",
        f"o {TRACK_MODEL_NAME}_continuous_surface",
        f"usemtl {TRACK_MATERIAL_NAME}",
    ]
    lines.extend(
        f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices
    )
    lines.extend(("vn 0.000000 0.000000 1.000000", "s 1"))
    lines.extend(
        f"f {first}//1 {second}//1 {third}//1"
        for first, second, third in faces
    )
    return "\n".join(lines) + "\n", vertices, faces, topology


def validate_track_surface_mesh(vertices, faces):
    """Validate connected annulus topology, triangle winding, and mesh area."""

    edge_counts = {}
    vertex_neighbors = {index: set() for index in range(1, len(vertices) + 1)}
    area_m2 = 0.0

    for face in faces:
        if len(set(face)) != 3:
            raise ValueError(f"degenerate track face indices: {face}")
        points = [vertices[index - 1] for index in face]
        edge_a = tuple(points[1][axis] - points[0][axis] for axis in range(3))
        edge_b = tuple(points[2][axis] - points[0][axis] for axis in range(3))
        cross = (
            edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
            edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
            edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
        )
        if cross[2] <= 1.0e-10:
            raise ValueError(f"track face is degenerate or faces down: {face}")
        area_m2 += 0.5 * math.sqrt(
            sum(component * component for component in cross)
        )

        for start, end in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((start, end)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            vertex_neighbors[start].add(end)
            vertex_neighbors[end].add(start)

    invalid_edge_counts = {
        edge: count
        for edge, count in edge_counts.items()
        if count not in (1, 2)
    }
    if invalid_edge_counts:
        raise ValueError(
            f"track has non-manifold edges: {invalid_edge_counts}"
        )

    boundary_edge_count = sum(
        count == 1 for count in edge_counts.values()
    )
    expected_boundary_edge_count = len(vertices)
    if boundary_edge_count != expected_boundary_edge_count:
        raise ValueError(
            f"track boundary edges {boundary_edge_count} != "
            f"{expected_boundary_edge_count}"
        )

    # One flood fill proves the ring is a single connected mesh component.
    visited = set()
    pending = [1]
    while pending:
        vertex = pending.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        pending.extend(vertex_neighbors[vertex] - visited)
    if len(visited) != len(vertices):
        raise ValueError(
            f"track has disconnected vertices: "
            f"{len(vertices) - len(visited)}"
        )

    euler_characteristic = len(vertices) - len(edge_counts) + len(faces)
    if euler_characteristic != 0:
        raise ValueError(
            f"track annulus Euler characteristic is "
            f"{euler_characteristic}, expected 0"
        )

    return {
        "area_m2": area_m2,
        "boundary_edges": boundary_edge_count,
        "edges": len(edge_counts),
        "euler_characteristic": euler_characteristic,
    }


def track_surface_mesh_visual():
    track_uri = (
        f"model://{TRACK_MODEL_NAME}/"
        f"{TRACK_MESH_RELATIVE_PATH.as_posix()}"
    )
    return f"""      <visual name="continuous_red_surface">
        <pose>0 0 0 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh>
            <uri>{track_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        {base_stadium.material_xml(CONTINUOUS_TRACK_RED_RGBA)}
      </visual>"""


def build_track_sdf():
    visuals = [track_surface_mesh_visual()]

    # Retain the west-side green apron from the CJU-inspired visual layout.
    visuals.append(
        base_track.box_visual(
            "green_west_apron",
            0.0,
            (
                base_track.CURVE_RADIUS_M
                + base_track.TRACK_HALF_WIDTH_M
                + 1.5
            ),
            base_track.SURFACE_Z_M,
            0.0,
            base_track.STRAIGHT_LENGTH_M,
            3.0,
            base_track.SURFACE_HEIGHT_M,
            base_track.GREEN_APRON_RGBA,
        )
    )

    # Eight 1.22 m lanes have nine white boundary lines.
    for boundary_index in range(base_track.LANE_COUNT + 1):
        offset_m = (
            -base_track.TRACK_HALF_WIDTH_M
            + boundary_index * base_track.LANE_WIDTH_M
        )
        for segment_index, (x, y, yaw, length) in enumerate(
            base_track.path_segments(offset_m)
        ):
            visuals.append(
                base_track.box_visual(
                    f"lane_{boundary_index}_{segment_index}",
                    x,
                    y,
                    base_track.LINE_Z_M,
                    yaw,
                    length,
                    base_track.LINE_WIDTH_M,
                    base_track.LINE_HEIGHT_M,
                    base_track.WHITE_RGBA,
                )
            )

    visual_body = "\n".join(visuals)
    centre_perimeter = (
        2.0 * base_track.STRAIGHT_LENGTH_M
        + 2.0 * math.pi * base_track.CURVE_RADIUS_M
    )
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated by gen_drone_cju_track_models.py. -->
  <!-- Dedicated model; no external mesh, image, or texture.
       Centre path perimeter: {centre_perimeter:.6f} m.
       Eight visual lanes, one connected procedural red track surface. -->
  <model name="{TRACK_MODEL_NAME}">
    <static>true</static>
    <link name="track_link">
{visual_body}
    </link>
  </model>
</sdf>
"""


def build_track_config():
    return f"""<?xml version="1.0"?>
<model>
  <name>{TRACK_MODEL_NAME}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Dedicated centred 400 m track with eight lanes and a continuous red surface.
  </description>
</model>
"""


def roof_top_z(x):
    normalized = (x - ROOF_X_MIN_M) / (ROOF_X_MAX_M - ROOF_X_MIN_M)
    return ROOF_EDGE_TOP_Z_M + ROOF_ARCH_RISE_M * math.sin(
        math.pi * normalized
    )


def roof_top_slope(x):
    normalized = (x - ROOF_X_MIN_M) / (ROOF_X_MAX_M - ROOF_X_MIN_M)
    return (
        ROOF_ARCH_RISE_M
        * math.pi
        / (ROOF_X_MAX_M - ROOF_X_MIN_M)
        * math.cos(math.pi * normalized)
    )


def normalized(vector):
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(component / magnitude for component in vector)


def make_roof_mesh():
    """Return ``(obj_text, vertices, faces, normal_count)`` for a closed roof.

    Four vertices are emitted per cross section:

    * top at y-min / y-max
    * bottom at y-min / y-max

    Top, bottom, both longitudinal sides, and both end caps are triangulated.
    The resulting solid is a closed two-manifold.
    """

    vertices = []
    top_normal_vectors = []
    bottom_normal_vectors = []

    for section_index in range(ROOF_SECTION_COUNT + 1):
        fraction = section_index / ROOF_SECTION_COUNT
        x = ROOF_X_MIN_M + fraction * (
            ROOF_X_MAX_M - ROOF_X_MIN_M
        )
        top_z = roof_top_z(x)
        bottom_z = top_z - ROOF_THICKNESS_M
        vertices.extend(
            (
                (x, ROOF_Y_MIN_M, top_z),
                (x, ROOF_Y_MAX_M, top_z),
                (x, ROOF_Y_MIN_M, bottom_z),
                (x, ROOF_Y_MAX_M, bottom_z),
            )
        )

        slope = roof_top_slope(x)
        top_normal_vectors.append(normalized((-slope, 0.0, 1.0)))
        bottom_normal_vectors.append(normalized((slope, 0.0, -1.0)))

    normals = []
    top_normal_indices = []
    bottom_normal_indices = []
    for top_normal, bottom_normal in zip(
        top_normal_vectors, bottom_normal_vectors
    ):
        normals.append(top_normal)
        top_normal_indices.append(len(normals))
        normals.append(bottom_normal)
        bottom_normal_indices.append(len(normals))

    y_min_normal_index = len(normals) + 1
    normals.append((0.0, -1.0, 0.0))
    y_max_normal_index = len(normals) + 1
    normals.append((0.0, +1.0, 0.0))
    x_min_normal_index = len(normals) + 1
    normals.append((-1.0, 0.0, 0.0))
    x_max_normal_index = len(normals) + 1
    normals.append((+1.0, 0.0, 0.0))

    def vertex_index(section_index, corner_index):
        # OBJ indices are one-based.
        return section_index * 4 + corner_index + 1

    # Each face is a tuple of (vertex_index, normal_index) pairs.
    faces = []
    for section_index in range(ROOF_SECTION_COUNT):
        next_section = section_index + 1
        top_here = top_normal_indices[section_index]
        top_next = top_normal_indices[next_section]
        bottom_here = bottom_normal_indices[section_index]
        bottom_next = bottom_normal_indices[next_section]

        top_y_min = vertex_index(section_index, 0)
        top_y_max = vertex_index(section_index, 1)
        bottom_y_min = vertex_index(section_index, 2)
        bottom_y_max = vertex_index(section_index, 3)
        next_top_y_min = vertex_index(next_section, 0)
        next_top_y_max = vertex_index(next_section, 1)
        next_bottom_y_min = vertex_index(next_section, 2)
        next_bottom_y_max = vertex_index(next_section, 3)

        # Curved top, outward normal +z.
        faces.extend(
            (
                (
                    (top_y_min, top_here),
                    (next_top_y_min, top_next),
                    (next_top_y_max, top_next),
                ),
                (
                    (top_y_min, top_here),
                    (next_top_y_max, top_next),
                    (top_y_max, top_here),
                ),
            )
        )
        # Curved bottom, outward normal -z.
        faces.extend(
            (
                (
                    (bottom_y_min, bottom_here),
                    (bottom_y_max, bottom_here),
                    (next_bottom_y_max, bottom_next),
                ),
                (
                    (bottom_y_min, bottom_here),
                    (next_bottom_y_max, bottom_next),
                    (next_bottom_y_min, bottom_next),
                ),
            )
        )
        # y-min side, outward normal -y.
        faces.extend(
            (
                (
                    (bottom_y_min, y_min_normal_index),
                    (next_bottom_y_min, y_min_normal_index),
                    (next_top_y_min, y_min_normal_index),
                ),
                (
                    (bottom_y_min, y_min_normal_index),
                    (next_top_y_min, y_min_normal_index),
                    (top_y_min, y_min_normal_index),
                ),
            )
        )
        # y-max side, outward normal +y.
        faces.extend(
            (
                (
                    (bottom_y_max, y_max_normal_index),
                    (top_y_max, y_max_normal_index),
                    (next_top_y_max, y_max_normal_index),
                ),
                (
                    (bottom_y_max, y_max_normal_index),
                    (next_top_y_max, y_max_normal_index),
                    (next_bottom_y_max, y_max_normal_index),
                ),
            )
        )

    # Closed end caps.
    first_top_y_min = vertex_index(0, 0)
    first_top_y_max = vertex_index(0, 1)
    first_bottom_y_min = vertex_index(0, 2)
    first_bottom_y_max = vertex_index(0, 3)
    last = ROOF_SECTION_COUNT
    last_top_y_min = vertex_index(last, 0)
    last_top_y_max = vertex_index(last, 1)
    last_bottom_y_min = vertex_index(last, 2)
    last_bottom_y_max = vertex_index(last, 3)

    faces.extend(
        (
            (
                (first_bottom_y_min, x_min_normal_index),
                (first_top_y_min, x_min_normal_index),
                (first_top_y_max, x_min_normal_index),
            ),
            (
                (first_bottom_y_min, x_min_normal_index),
                (first_top_y_max, x_min_normal_index),
                (first_bottom_y_max, x_min_normal_index),
            ),
            (
                (last_bottom_y_min, x_max_normal_index),
                (last_bottom_y_max, x_max_normal_index),
                (last_top_y_max, x_max_normal_index),
            ),
            (
                (last_bottom_y_min, x_max_normal_index),
                (last_top_y_max, x_max_normal_index),
                (last_top_y_min, x_max_normal_index),
            ),
        )
    )

    validate_roof_mesh(vertices, faces)

    lines = [
        "# Procedurally generated closed grandstand roof",
        "# No external source asset",
        f"mtllib {ROOF_MTL_RELATIVE_PATH.name}",
        f"o {STADIUM_MODEL_NAME}_grandstand_roof",
        f"usemtl {ROOF_MATERIAL_NAME}",
    ]
    lines.extend(
        f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices
    )
    lines.extend(
        f"vn {x:.8f} {y:.8f} {z:.8f}" for x, y, z in normals
    )
    lines.append("s 1")
    for face in faces:
        face_fields = " ".join(
            f"{vertex_index_value}//{normal_index_value}"
            for vertex_index_value, normal_index_value in face
        )
        lines.append(f"f {face_fields}")

    return "\n".join(lines) + "\n", vertices, faces, len(normals)


def validate_roof_mesh(vertices, faces):
    """Raise if the procedural roof is degenerate or not a closed manifold."""

    edge_counts = {}
    signed_volume_times_six = 0.0

    for face in faces:
        vertex_indices = [vertex_index for vertex_index, _ in face]
        if len(set(vertex_indices)) != 3:
            raise ValueError(f"degenerate face indices: {vertex_indices}")

        points = [vertices[index - 1] for index in vertex_indices]
        edge_a = tuple(points[1][axis] - points[0][axis] for axis in range(3))
        edge_b = tuple(points[2][axis] - points[0][axis] for axis in range(3))
        cross = (
            edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
            edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
            edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
        )
        if math.sqrt(sum(value * value for value in cross)) < 1.0e-10:
            raise ValueError(f"zero-area face: {vertex_indices}")

        p0, p1, p2 = points
        signed_volume_times_six += (
            p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
            + p0[1] * (p1[2] * p2[0] - p1[0] * p2[2])
            + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
        )

        for start, end in zip(
            vertex_indices,
            vertex_indices[1:] + vertex_indices[:1],
        ):
            edge = tuple(sorted((start, end)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    non_manifold_edges = {
        edge: count for edge, count in edge_counts.items() if count != 2
    }
    if non_manifold_edges:
        raise ValueError(
            f"roof mesh is not closed; edge counts: {non_manifold_edges}"
        )
    if abs(signed_volume_times_six) < 1.0e-8:
        raise ValueError("roof mesh has zero signed volume")


def roof_mesh_visual():
    roof_uri = (
        f"model://{STADIUM_MODEL_NAME}/"
        f"{ROOF_MESH_RELATIVE_PATH.as_posix()}"
    )
    return f"""      <visual name="north_smooth_blue_roof">
        <pose>0 0 0 0 0 0</pose>
        <cast_shadows>true</cast_shadows>
        <geometry>
          <mesh>
            <uri>{roof_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        {base_stadium.material_xml(base_stadium.ROOF_BLUE_RGBA)}
      </visual>"""


def build_stadium_sdf():
    visuals = [
        base_stadium.box_visual(
            "stadium_plaza",
            0.0,
            0.0,
            0.002,
            base_stadium.SITE_LENGTH_M,
            base_stadium.SITE_WIDTH_M,
            0.004,
            base_stadium.PLAZA_RGBA,
            cast_shadows=False,
        )
    ]
    collisions = []

    base_stadium.add_football_field(visuals)
    base_stadium.add_blue_court(visuals)
    base_stadium.add_stands(visuals, collisions)
    base_stadium.add_trees(visuals)
    base_stadium.add_floodlights(visuals)
    base_stadium.add_surroundings(visuals)

    # Replace the seven faceted box panels from the legacy visual with a single
    # smooth, closed procedural mesh. Columns and all seating remain.
    visuals = [
        visual
        for visual in visuals
        if 'name="north_blue_roof_panel_' not in visual
    ]
    visuals.append(roof_mesh_visual())

    visual_body = "\n".join(visuals)
    collision_body = "\n".join(collisions)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated by gen_drone_cju_track_models.py. -->
  <!-- Dedicated primitive stadium plus a locally generated closed roof OBJ. -->
  <model name="{STADIUM_MODEL_NAME}">
    <static>true</static>
    <link name="stadium_link">
{visual_body}
{collision_body}
    </link>
  </model>
</sdf>
"""


def build_stadium_config():
    return f"""<?xml version="1.0"?>
<model>
  <name>{STADIUM_MODEL_NAME}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Dedicated CJU-inspired stadium with a smooth procedural blue grandstand roof.
  </description>
</model>
"""


def visual_xy_bounds():
    """Return the generated track's exact box AABB in local x/y coordinates."""

    min_x = min_y = math.inf
    max_x = max_y = -math.inf

    def expand(x, y, yaw, length, width):
        nonlocal min_x, min_y, max_x, max_y
        half_x = (
            abs(math.cos(yaw)) * length / 2.0
            + abs(math.sin(yaw)) * width / 2.0
        )
        half_y = (
            abs(math.sin(yaw)) * length / 2.0
            + abs(math.cos(yaw)) * width / 2.0
        )
        min_x = min(min_x, x - half_x)
        max_x = max(max_x, x + half_x)
        min_y = min(min_y, y - half_y)
        max_y = max(max_y, y + half_y)

    for x, y, yaw, length in base_track.path_segments():
        expand(x, y, yaw, length, base_track.TRACK_WIDTH_M)
    expand(
        0.0,
        (
            base_track.CURVE_RADIUS_M
            + base_track.TRACK_HALF_WIDTH_M
            + 1.5
        ),
        0.0,
        base_track.STRAIGHT_LENGTH_M,
        3.0,
    )
    for boundary_index in range(base_track.LANE_COUNT + 1):
        offset_m = (
            -base_track.TRACK_HALF_WIDTH_M
            + boundary_index * base_track.LANE_WIDTH_M
        )
        for x, y, yaw, length in base_track.path_segments(offset_m):
            expand(x, y, yaw, length, base_track.LINE_WIDTH_M)

    return min_x, min_y, max_x, max_y


def main():
    (TRACK_MODEL_DIR / TRACK_MESH_RELATIVE_PATH.parent).mkdir(
        parents=True, exist_ok=True
    )
    (STADIUM_MODEL_DIR / ROOF_MESH_RELATIVE_PATH.parent).mkdir(
        parents=True, exist_ok=True
    )

    track_sdf = build_track_sdf()
    stadium_sdf = build_stadium_sdf()
    (
        track_obj,
        track_vertices,
        track_faces,
        track_topology,
    ) = make_track_surface_mesh()
    roof_obj, roof_vertices, roof_faces, roof_normal_count = make_roof_mesh()

    (TRACK_MODEL_DIR / "model.sdf").write_text(track_sdf, encoding="utf-8")
    (TRACK_MODEL_DIR / "model.config").write_text(
        build_track_config(), encoding="utf-8"
    )
    (TRACK_MODEL_DIR / TRACK_MESH_RELATIVE_PATH).write_text(
        track_obj, encoding="utf-8"
    )
    (TRACK_MODEL_DIR / TRACK_MTL_RELATIVE_PATH).write_text(
        build_mtl(TRACK_MATERIAL_NAME, CONTINUOUS_TRACK_RED_RGBA),
        encoding="utf-8",
    )
    (STADIUM_MODEL_DIR / "model.sdf").write_text(
        stadium_sdf, encoding="utf-8"
    )
    (STADIUM_MODEL_DIR / "model.config").write_text(
        build_stadium_config(), encoding="utf-8"
    )
    (STADIUM_MODEL_DIR / ROOF_MESH_RELATIVE_PATH).write_text(
        roof_obj, encoding="utf-8"
    )
    (STADIUM_MODEL_DIR / ROOF_MTL_RELATIVE_PATH).write_text(
        build_mtl(ROOF_MATERIAL_NAME, base_stadium.ROOF_BLUE_RGBA),
        encoding="utf-8",
    )

    min_x, min_y, max_x, max_y = visual_xy_bounds()
    track_mesh_min = [
        min(vertex[axis] for vertex in track_vertices) for axis in range(3)
    ]
    track_mesh_max = [
        max(vertex[axis] for vertex in track_vertices) for axis in range(3)
    ]
    roof_min_z = min(vertex[2] for vertex in roof_vertices)
    roof_max_z = max(vertex[2] for vertex in roof_vertices)
    print(f"Generated dedicated track: {TRACK_MODEL_DIR}")
    print(
        f"  one continuous red mesh visual; "
        f"{base_track.LANE_COUNT} lanes / "
        f"{track_sdf.count('<visual name=')} visuals"
    )
    print(
        f"  red mesh: {TRACK_ARC_SEGMENTS} segments/half-circle, "
        f"{len(track_vertices)} vertices, {len(track_faces)} triangles, "
        f"{track_topology['edges']} edges / "
        f"{track_topology['boundary_edges']} boundary edges, "
        f"Euler={track_topology['euler_characteristic']}, "
        f"area={track_topology['area_m2']:.3f} m^2"
    )
    print(
        f"  red mesh AABB: "
        f"x=[{track_mesh_min[0]:.3f}, {track_mesh_max[0]:.3f}], "
        f"y=[{track_mesh_min[1]:.3f}, {track_mesh_max[1]:.3f}], "
        f"z={track_mesh_min[2]:.3f} m"
    )
    print(
        f"  track AABB: x=[{min_x:.3f}, {max_x:.3f}], "
        f"y=[{min_y:.3f}, {max_y:.3f}], "
        f"z=[0.005, 0.018] m"
    )
    print(f"Generated dedicated stadium: {STADIUM_MODEL_DIR}")
    print(
        f"  stadium footprint: x=[-{base_stadium.SITE_LENGTH_M / 2:.1f}, "
        f"{base_stadium.SITE_LENGTH_M / 2:.1f}], "
        f"y=[-{base_stadium.SITE_WIDTH_M / 2:.1f}, "
        f"{base_stadium.SITE_WIDTH_M / 2:.1f}] m"
    )
    print(
        f"  roof: {ROOF_SECTION_COUNT} sections, "
        f"{len(roof_vertices)} vertices, {len(roof_faces)} triangle faces, "
        f"{roof_normal_count} normals"
    )
    print(
        f"  roof AABB: x=[{ROOF_X_MIN_M:.2f}, {ROOF_X_MAX_M:.2f}], "
        f"y=[{ROOF_Y_MIN_M:.2f}, {ROOF_Y_MAX_M:.2f}], "
        f"z=[{roof_min_z:.2f}, {roof_max_z:.2f}] m"
    )
    print(
        f"  stadium visuals/collisions: "
        f"{stadium_sdf.count('<visual name=')}/"
        f"{stadium_sdf.count('<collision name=')}"
    )


if __name__ == "__main__":
    main()
