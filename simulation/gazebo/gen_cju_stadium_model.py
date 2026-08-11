#!/usr/bin/env python3
"""Generate the field, basketball court, and jokgu court for the CJU map."""

import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models" / "cheongju_university_stadium"

DIRT_RGBA = (0.62, 0.50, 0.34, 1.0)
BASKETBALL_RGBA = (0.07, 0.34, 0.58, 1.0)
JOKGU_RGBA = (0.12, 0.42, 0.24, 1.0)
WHITE_RGBA = (0.96, 0.96, 0.93, 1.0)
SURFACE_HEIGHT_M = 0.02
LINE_WIDTH_M = 0.08
LINE_Z_M = 0.012
LINE_HEIGHT_M = 0.004

# The centres are integer image/map coordinates.  The basketball court keeps
# its FIBA dimensions; the vertical jokgu court is enlarged for map visibility.
# (name, centre x/y, horizontal size x/y, colour)
FACILITIES = (
    ("stadium_field", (44, 48), (68, 105), DIRT_RGBA),
    ("basketball_court", (44, 113), (28, 15), BASKETBALL_RGBA),
    ("jokgu_court", (44, -18), (10, 20), JOKGU_RGBA),
)


def material_xml(rgba):
    red, green, blue, alpha = rgba
    return f"""<material>
          <ambient>{red} {green} {blue} {alpha}</ambient>
          <diffuse>{red} {green} {blue} {alpha}</diffuse>
          <specular>0.04 0.04 0.04 1</specular>
        </material>"""


def box_visual(
    name,
    centre,
    size,
    rgba,
    *,
    z=0.0,
    height=SURFACE_HEIGHT_M,
    yaw=0.0,
):
    x, y = centre
    size_x, size_y = size
    return f"""      <visual name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>
        <cast_shadows>false</cast_shadows>
        <geometry><box><size>{size_x:.4f} {size_y:.4f} {height:.4f}</size></box></geometry>
        {material_xml(rgba)}
      </visual>"""


def _rectangle_lines(visuals, name, centre, size, *, halfway=False):
    x, y = centre
    size_x, size_y = size
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    for suffix, line_centre, line_size in (
        ("east", (x + half_x, y), (LINE_WIDTH_M, size_y)),
        ("west", (x - half_x, y), (LINE_WIDTH_M, size_y)),
        ("north", (x, y + half_y), (size_x, LINE_WIDTH_M)),
        ("south", (x, y - half_y), (size_x, LINE_WIDTH_M)),
    ):
        visuals.append(box_visual(
            f"{name}_line_{suffix}", line_centre, line_size, WHITE_RGBA,
            z=LINE_Z_M, height=LINE_HEIGHT_M,
        ))
    if halfway:
        visuals.append(box_visual(
            f"{name}_line_halfway", centre, (size_x, LINE_WIDTH_M),
            WHITE_RGBA, z=LINE_Z_M, height=LINE_HEIGHT_M,
        ))


def _circle_lines(visuals, name, centre, radius, segments=24):
    chord = 2.0 * radius * math.sin(math.pi / segments) * 1.03
    for index in range(segments):
        theta = (index + 0.5) * 2.0 * math.pi / segments
        point = (
            centre[0] + radius * math.cos(theta),
            centre[1] + radius * math.sin(theta),
        )
        visuals.append(box_visual(
            f"{name}_line_circle_{index}", point,
            (chord, LINE_WIDTH_M), WHITE_RGBA,
            z=LINE_Z_M, height=LINE_HEIGHT_M,
            yaw=theta + math.pi / 2.0,
        ))


def facility_visuals():
    visuals = [box_visual(*facility) for facility in FACILITIES]
    configured = {name: (centre, size) for name, centre, size, _ in FACILITIES}

    field_centre, field_size = configured["stadium_field"]
    _rectangle_lines(
        visuals, "stadium_field", field_centre, field_size, halfway=True
    )
    _circle_lines(visuals, "stadium_field", field_centre, 9.15)

    basketball_centre, basketball_size = configured["basketball_court"]
    _rectangle_lines(
        visuals,
        "basketball_court",
        basketball_centre,
        basketball_size,
        halfway=False,
    )
    visuals.append(box_visual(
        "basketball_court_line_halfway",
        basketball_centre,
        (LINE_WIDTH_M, basketball_size[1]),
        WHITE_RGBA,
        z=LINE_Z_M,
        height=LINE_HEIGHT_M,
    ))
    _circle_lines(
        visuals, "basketball_court", basketball_centre, 1.8, segments=16
    )

    jokgu_centre, jokgu_size = configured["jokgu_court"]
    _rectangle_lines(
        visuals, "jokgu_court", jokgu_centre, jokgu_size, halfway=True
    )
    return visuals


def build_sdf(model_name="cheongju_university_stadium"):
    visual_body = "\n".join(facility_visuals())
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated requested facilities only; markings are visual-only. -->
  <model name="{model_name}">
    <static>true</static>
    <link name="stadium_link">
{visual_body}
    </link>
  </model>
</sdf>
"""


def build_config(model_name="cheongju_university_stadium"):
    return f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>CJU field, basketball court, and enlarged vertical jokgu court.</description>
</model>
"""


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "model.sdf").write_text(build_sdf(), encoding="utf-8")
    (MODEL_DIR / "model.config").write_text(build_config(), encoding="utf-8")
    print(f"Generated CJU facilities: {MODEL_DIR}")


if __name__ == "__main__":
    main()
