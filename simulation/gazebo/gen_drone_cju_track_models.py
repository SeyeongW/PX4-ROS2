#!/usr/bin/env python3
"""Generate the CJU stadium track and its three requested facilities.

The ``stadium_endpoint`` origin is the south-east track tangent visible at the
bottom-right of a north-up map.  Local +x is approximately east and local +y
is approximately north, so the stadium extends left/up from (0, 0).
"""

import math
from pathlib import Path

import gen_cju_stadium_model as base_stadium


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
TRACK_MODEL_NAME = "drone_cju_track_running_track"
STADIUM_MODEL_NAME = "drone_cju_track_stadium"
TRACK_MODEL_DIR = MODELS_DIR / TRACK_MODEL_NAME
STADIUM_MODEL_DIR = MODELS_DIR / STADIUM_MODEL_NAME
TRACK_MESH_PATH = Path("meshes/running_track_surface.obj")
TRACK_MTL_PATH = Path("meshes/running_track_surface.mtl")

# OSM way 1374978221 is approximately 178.7 m by 87.4 m.  The 92 m tangent
# separation and 39 m trailer radius remain unchanged, keeping the moving
# platform centred on the rendered track.
TRACK_CENTER_M = (-44, 46)
STRAIGHT_LENGTH_M = 92.0
OUTER_HALF_WIDTH_M = 43.7
OUTER_END_RADIUS_M = 43.35
LANE_COUNT = 8
LANE_WIDTH_M = 1.22
TRACK_WIDTH_M = LANE_COUNT * LANE_WIDTH_M
LINE_WIDTH_M = 0.08
ARC_SEGMENTS = 48
# Keep the track above court/field markings where their rectangles
# meet the rounded infield.  This cleanly masks sub-metre corner overlap.
TRACK_SURFACE_Z_M = 0.016
LINE_SURFACE_Z_M = 0.020

TRACK_RGBA = (0.54, 0.18, 0.13, 1.0)
LINE_RGBA = (0.96, 0.96, 0.93, 1.0)
TRACK_MATERIAL_NAME = "cju_track_red"
LINE_MATERIAL_NAME = "cju_lane_white"


def _material(name, rgba):
    red, green, blue, alpha = rgba
    return f"""newmtl {name}
Ka {red:.6f} {green:.6f} {blue:.6f}
Kd {red:.6f} {green:.6f} {blue:.6f}
Ks 0.040000 0.040000 0.040000
Ns 8.000000
d {alpha:.6f}
illum 2
"""


def build_mtl():
    return (
        "# Procedurally generated CJU track materials\n"
        + _material(TRACK_MATERIAL_NAME, TRACK_RGBA)
        + "\n"
        + _material(LINE_MATERIAL_NAME, LINE_RGBA)
    )


def stadium_outline(offset_m=0.0):
    """Return one smooth clockwise-free stadium contour.

    ``offset_m`` moves the contour inward.  Every contour starts at the
    south-east tangent, continues north on the east straight, then closes via
    the north curve, west straight, and south curve.
    """

    radius_x = OUTER_HALF_WIDTH_M - offset_m
    radius_y = OUTER_END_RADIUS_M - offset_m
    if radius_x <= 0.0 or radius_y <= 0.0:
        raise ValueError("track offset is larger than the curve radius")

    centre_x = -OUTER_HALF_WIDTH_M
    east_x = centre_x + radius_x
    west_x = centre_x - radius_x
    points = [(east_x, 0.0), (east_x, STRAIGHT_LENGTH_M)]

    for index in range(1, ARC_SEGMENTS + 1):
        theta = math.pi * index / ARC_SEGMENTS
        points.append((
            centre_x + radius_x * math.cos(theta),
            STRAIGHT_LENGTH_M + radius_y * math.sin(theta),
        ))

    points.append((west_x, 0.0))
    for index in range(1, ARC_SEGMENTS):
        theta = math.pi + math.pi * index / ARC_SEGMENTS
        points.append((
            centre_x + radius_x * math.cos(theta),
            radius_y * math.sin(theta),
        ))
    return tuple(points)


def _append_annulus(vertices, sections, name, material, outer, inner, z):
    if len(outer) != len(inner):
        raise ValueError("track contours must have the same station count")
    first_vertex = len(vertices) + 1
    for inner_point, outer_point in zip(inner, outer):
        vertices.extend(((*inner_point, z), (*outer_point, z)))

    faces = []
    for station in range(len(outer)):
        next_station = (station + 1) % len(outer)
        inner_index = first_vertex + station * 2
        outer_index = inner_index + 1
        next_inner = first_vertex + next_station * 2
        next_outer = next_inner + 1
        faces.extend((
            (inner_index, outer_index, next_outer),
            (inner_index, next_outer, next_inner),
        ))
    sections.append((name, material, faces))


def _validate_layout():
    outer = stadium_outline()
    if outer[0] != (0.0, 0.0):
        raise ValueError("photo bottom-right track tangent must be (0, 0)")
    if len(outer) < 64:
        raise ValueError("track curve sampling is too coarse")
    values = (
        *TRACK_CENTER_M,
        *(value for point in outer for value in point),
        *(value for _, centre, _, _ in base_stadium.FACILITIES
          for value in centre),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("layout contains a non-finite coordinate")
    if any(value != round(value) for value in TRACK_CENTER_M):
        raise ValueError("mission-relevant track centre must be integer metres")
    for _, centre, _, _ in base_stadium.FACILITIES:
        if any(value != round(value) for value in centre):
            raise ValueError("facility centres must be integer metres")

    inner_radius_x = OUTER_HALF_WIDTH_M - TRACK_WIDTH_M
    inner_radius_y = OUTER_END_RADIUS_M - TRACK_WIDTH_M
    for name, centre, size, _ in base_stadium.FACILITIES:
        if not name.endswith("_court"):
            continue
        half_x = size[0] / 2.0 + base_stadium.LINE_WIDTH_M / 2.0
        half_y = size[1] / 2.0 + base_stadium.LINE_WIDTH_M / 2.0
        for x in (centre[0] - half_x, centre[0] + half_x):
            for y in (centre[1] - half_y, centre[1] + half_y):
                curve_y = min(max(y, 0.0), STRAIGHT_LENGTH_M)
                inside = (
                    ((x + OUTER_HALF_WIDTH_M) / inner_radius_x) ** 2
                    + ((y - curve_y) / inner_radius_y) ** 2
                ) <= 1.0
                if not inside:
                    raise ValueError(f"{name} extends into the running track")


def make_track_surface_mesh():
    """Return one red annulus with regulation-width eight-lane markings."""

    vertices = []
    sections = []
    _append_annulus(
        vertices,
        sections,
        "continuous_red_surface",
        TRACK_MATERIAL_NAME,
        stadium_outline(0.0),
        stadium_outline(TRACK_WIDTH_M),
        TRACK_SURFACE_Z_M,
    )

    for boundary in range(LANE_COUNT + 1):
        offset = boundary * LANE_WIDTH_M
        outer_offset = max(0.0, offset - LINE_WIDTH_M / 2.0)
        inner_offset = min(TRACK_WIDTH_M, offset + LINE_WIDTH_M / 2.0)
        if boundary == 0:
            inner_offset = LINE_WIDTH_M
        elif boundary == LANE_COUNT:
            outer_offset = TRACK_WIDTH_M - LINE_WIDTH_M
        _append_annulus(
            vertices,
            sections,
            f"lane_boundary_{boundary}",
            LINE_MATERIAL_NAME,
            stadium_outline(outer_offset),
            stadium_outline(inner_offset),
            LINE_SURFACE_Z_M,
        )

    lines = [
        "# Smooth OSM-scale CJU eight-lane running track",
        f"mtllib {TRACK_MTL_PATH.name}",
    ]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(("vn 0 0 1", "s 1"))
    face_count = 0
    for name, material, faces in sections:
        lines.extend((f"o {name}", f"usemtl {material}"))
        lines.extend(f"f {a}//1 {b}//1 {c}//1" for a, b, c in faces)
        face_count += len(faces)
    return "\n".join(lines) + "\n", vertices, face_count


def build_track_sdf():
    uri = f"model://{TRACK_MODEL_NAME}/{TRACK_MESH_PATH.as_posix()}"
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Smooth red 8-lane track; materials are embedded in the OBJ/MTL. -->
  <model name="{TRACK_MODEL_NAME}">
    <static>true</static>
    <link name="track_link">
      <visual name="continuous_red_surface">
        <pose>0 0 0 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry><mesh><uri>{uri}</uri><scale>1 1 1</scale></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>
"""


def build_stadium_sdf():
    return base_stadium.build_sdf(STADIUM_MODEL_NAME)


def build_config(name, description):
    return f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>{description}</description>
</model>
"""


def main():
    _validate_layout()
    (TRACK_MODEL_DIR / TRACK_MESH_PATH.parent).mkdir(
        parents=True, exist_ok=True
    )
    STADIUM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    track_obj, vertices, face_count = make_track_surface_mesh()
    (TRACK_MODEL_DIR / "model.sdf").write_text(
        build_track_sdf(), encoding="utf-8"
    )
    (TRACK_MODEL_DIR / "model.config").write_text(
        build_config(
            TRACK_MODEL_NAME,
            "OSM-scale red running track with eight 1.22 m lanes.",
        ),
        encoding="utf-8",
    )
    (TRACK_MODEL_DIR / TRACK_MESH_PATH).write_text(
        track_obj, encoding="utf-8"
    )
    (TRACK_MODEL_DIR / TRACK_MTL_PATH).write_text(
        build_mtl(), encoding="utf-8"
    )
    (STADIUM_MODEL_DIR / "model.sdf").write_text(
        build_stadium_sdf(), encoding="utf-8"
    )
    (STADIUM_MODEL_DIR / "model.config").write_text(
        build_config(
            STADIUM_MODEL_NAME,
            "CJU field, basketball court, and jokgu court only.",
        ),
        encoding="utf-8",
    )
    print(
        f"Generated CJU map: {len(vertices)} track vertices, "
        f"{face_count} triangles, {LANE_COUNT} lanes"
    )


if __name__ == "__main__":
    main()
