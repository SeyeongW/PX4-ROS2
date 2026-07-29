#!/usr/bin/env python3
"""Generate a lightweight Cheongju University main-stadium scene.

The model is an original low-poly interpretation based on publicly visible
facts and photographs of the Cheongju University main complex stadium:

* a beige dirt football field inside a 400 m oval track,
* concrete / natural-stone open seating,
* a blue-roofed covered stand,
* a blue outdoor court, trees, floodlights, and simple campus buildings.

No photograph, texture, logo, downloaded mesh, or other third-party asset is
embedded. All geometry is composed from SDF box, cylinder, and sphere
primitives. The companion ``running_track`` model supplies the track itself.

Coordinate convention:
    Stadium centre = local (0, 0)
    Stadium long axis = local x
    Covered north stand = local +x end
    South outdoor court / campus buildings = local -x end

Run:
    python3 simulation/gazebo/gen_cju_stadium_model.py
"""

import math
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    SCRIPT_DIR, "models", "cheongju_university_stadium"
)

# Overall paved stadium footprint, corresponding to the roughly 227 x 139 m
# stadium boundary visible in map imagery. It also defines the model x/y extent.
SITE_LENGTH_M = 226.0
SITE_WIDTH_M = 138.0

PLAZA_RGBA = (0.48, 0.48, 0.45, 1.0)
DIRT_RGBA = (0.62, 0.50, 0.34, 1.0)
FIELD_LINE_RGBA = (0.94, 0.94, 0.88, 1.0)
STONE_LIGHT_RGBA = (0.52, 0.50, 0.44, 1.0)
STONE_DARK_RGBA = (0.39, 0.38, 0.35, 1.0)
CONCRETE_RGBA = (0.57, 0.57, 0.55, 1.0)
ROOF_BLUE_RGBA = (0.07, 0.28, 0.58, 1.0)
COURT_BLUE_RGBA = (0.07, 0.34, 0.58, 1.0)
METAL_RGBA = (0.58, 0.61, 0.63, 1.0)
LAMP_RGBA = (0.96, 0.96, 0.82, 1.0)
TREE_TRUNK_RGBA = (0.24, 0.14, 0.07, 1.0)
TREE_GREEN_RGBA = (0.12, 0.31, 0.13, 1.0)
BUILDING_BEIGE_RGBA = (0.62, 0.57, 0.49, 1.0)
BUILDING_BRICK_RGBA = (0.48, 0.25, 0.18, 1.0)
WINDOW_RGBA = (0.12, 0.22, 0.27, 1.0)


def material_xml(rgba):
    red, green, blue, alpha = rgba
    return f"""<material>
          <ambient>{red} {green} {blue} {alpha}</ambient>
          <diffuse>{red} {green} {blue} {alpha}</diffuse>
          <specular>0.04 0.04 0.04 1</specular>
        </material>"""


def box_visual(
    name,
    x,
    y,
    z,
    size_x,
    size_y,
    size_z,
    rgba,
    *,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    cast_shadows=True,
):
    shadow_text = "true" if cast_shadows else "false"
    return f"""      <visual name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>
        <cast_shadows>{shadow_text}</cast_shadows>
        <geometry>
          <box><size>{size_x:.4f} {size_y:.4f} {size_z:.4f}</size></box>
        </geometry>
        {material_xml(rgba)}
      </visual>"""


def cylinder_visual(
    name,
    x,
    y,
    z,
    radius,
    length,
    rgba,
    *,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    cast_shadows=True,
):
    shadow_text = "true" if cast_shadows else "false"
    return f"""      <visual name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>
        <cast_shadows>{shadow_text}</cast_shadows>
        <geometry>
          <cylinder><radius>{radius:.4f}</radius><length>{length:.4f}</length></cylinder>
        </geometry>
        {material_xml(rgba)}
      </visual>"""


def sphere_visual(
    name,
    x,
    y,
    z,
    radius,
    rgba,
    *,
    cast_shadows=True,
):
    shadow_text = "true" if cast_shadows else "false"
    return f"""      <visual name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
        <cast_shadows>{shadow_text}</cast_shadows>
        <geometry><sphere><radius>{radius:.4f}</radius></sphere></geometry>
        {material_xml(rgba)}
      </visual>"""


def box_collision(name, x, y, z, size_x, size_y, size_z):
    return f"""      <collision name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
        <geometry>
          <box><size>{size_x:.4f} {size_y:.4f} {size_z:.4f}</size></box>
        </geometry>
      </collision>"""


def add_football_field(visuals):
    # The dirt infield is limited to the clear region inside the track. The
    # 105 x 60 m football marking is a compact visual approximation.
    visuals.append(
        box_visual(
            "dirt_infield",
            0.0,
            0.0,
            0.007,
            112.0,
            62.0,
            0.006,
            DIRT_RGBA,
            cast_shadows=False,
        )
    )

    line_z = 0.0145
    line_height = 0.002
    line_width = 0.09
    pitch_half_length = 52.5
    pitch_half_width = 30.0

    for name, x, y, size_x, size_y in (
        ("pitch_side_north", 0.0, +pitch_half_width, 105.0, line_width),
        ("pitch_side_south", 0.0, -pitch_half_width, 105.0, line_width),
        ("pitch_end_east", +pitch_half_length, 0.0, line_width, 60.0),
        ("pitch_end_west", -pitch_half_length, 0.0, line_width, 60.0),
        ("pitch_halfway", 0.0, 0.0, line_width, 60.0),
    ):
        visuals.append(
            box_visual(
                name,
                x,
                y,
                line_z,
                size_x,
                size_y,
                line_height,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )

    # Centre circle (9.15 m radius) from short overlapping box chords.
    circle_radius = 9.15
    circle_segments = 24
    circle_chord = (
        2.0 * circle_radius * math.sin(math.pi / circle_segments) * 1.04
    )
    for index in range(circle_segments):
        theta = (index + 0.5) * 2.0 * math.pi / circle_segments
        visuals.append(
            box_visual(
                f"pitch_centre_circle_{index}",
                circle_radius * math.cos(theta),
                circle_radius * math.sin(theta),
                line_z,
                circle_chord,
                line_width,
                line_height,
                FIELD_LINE_RGBA,
                yaw=theta + math.pi / 2.0,
                cast_shadows=False,
            )
        )
    visuals.append(
        cylinder_visual(
            "pitch_centre_spot",
            0.0,
            0.0,
            0.0145,
            0.12,
            0.002,
            FIELD_LINE_RGBA,
            cast_shadows=False,
        )
    )

    # Penalty and goal areas.
    for side_name, side_sign in (("east", +1.0), ("west", -1.0)):
        penalty_inner_x = side_sign * (pitch_half_length - 16.5)
        penalty_mid_x = side_sign * (pitch_half_length - 8.25)
        goal_inner_x = side_sign * (pitch_half_length - 5.5)
        goal_mid_x = side_sign * (pitch_half_length - 2.75)

        visuals.append(
            box_visual(
                f"{side_name}_penalty_inner",
                penalty_inner_x,
                0.0,
                line_z,
                line_width,
                40.32,
                line_height,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )
        for edge_name, y in (("top", +20.16), ("bottom", -20.16)):
            visuals.append(
                box_visual(
                    f"{side_name}_penalty_{edge_name}",
                    penalty_mid_x,
                    y,
                    line_z,
                    16.5,
                    line_width,
                    line_height,
                    FIELD_LINE_RGBA,
                    cast_shadows=False,
                )
            )
        visuals.append(
            box_visual(
                f"{side_name}_goal_area_inner",
                goal_inner_x,
                0.0,
                line_z,
                line_width,
                18.32,
                line_height,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )
        for edge_name, y in (("top", +9.16), ("bottom", -9.16)):
            visuals.append(
                box_visual(
                    f"{side_name}_goal_area_{edge_name}",
                    goal_mid_x,
                    y,
                    line_z,
                    5.5,
                    line_width,
                    line_height,
                    FIELD_LINE_RGBA,
                    cast_shadows=False,
                )
            )
        visuals.append(
            cylinder_visual(
                f"{side_name}_penalty_spot",
                side_sign * (pitch_half_length - 11.0),
                0.0,
                0.0145,
                0.10,
                0.002,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )

    # Primitive-only football goals. They remain visual-only so they cannot
    # unexpectedly obstruct the UAV experiment.
    for side_name, side_sign in (("east", +1.0), ("west", -1.0)):
        goal_x = side_sign * 53.0
        for post_name, y in (("north", +3.66), ("south", -3.66)):
            visuals.append(
                cylinder_visual(
                    f"goal_{side_name}_{post_name}_post",
                    goal_x,
                    y,
                    1.22,
                    0.055,
                    2.44,
                    FIELD_LINE_RGBA,
                )
            )
        visuals.append(
            box_visual(
                f"goal_{side_name}_crossbar",
                goal_x,
                0.0,
                2.44,
                0.11,
                7.43,
                0.11,
                FIELD_LINE_RGBA,
            )
        )


def add_blue_court(visuals):
    court_x = -98.0
    visuals.append(
        box_visual(
            "south_blue_court",
            court_x,
            0.0,
            0.007,
            24.0,
            14.0,
            0.006,
            COURT_BLUE_RGBA,
            cast_shadows=False,
        )
    )

    line_z = 0.0145
    line_height = 0.002
    line_width = 0.07
    for name, x, y, size_x, size_y in (
        ("court_side_north", court_x, +7.0, 24.0, line_width),
        ("court_side_south", court_x, -7.0, 24.0, line_width),
        ("court_end_east", court_x + 12.0, 0.0, line_width, 14.0),
        ("court_end_west", court_x - 12.0, 0.0, line_width, 14.0),
        ("court_halfway", court_x, 0.0, line_width, 14.0),
    ):
        visuals.append(
            box_visual(
                name,
                x,
                y,
                line_z,
                size_x,
                size_y,
                line_height,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )

    radius = 1.8
    segment_count = 16
    chord = 2.0 * radius * math.sin(math.pi / segment_count) * 1.04
    for index in range(segment_count):
        theta = (index + 0.5) * 2.0 * math.pi / segment_count
        visuals.append(
            box_visual(
                f"court_centre_circle_{index}",
                court_x + radius * math.cos(theta),
                radius * math.sin(theta),
                line_z,
                chord,
                line_width,
                line_height,
                FIELD_LINE_RGBA,
                yaw=theta + math.pi / 2.0,
                cast_shadows=False,
            )
        )


def add_stands(visuals, collisions):
    # East-side open stand: alternating stone tones approximate the planted
    # concrete/natural-stone terraces visible around the real stadium.
    for index in range(7):
        depth = 3.0
        y = -45.0 - index * depth
        height = 0.43 * (index + 1)
        colour = (
            STONE_LIGHT_RGBA if index % 2 == 0 else STONE_DARK_RGBA
        )
        visuals.append(
            box_visual(
                f"east_stone_step_{index}",
                0.0,
                y,
                height / 2.0,
                118.0,
                depth,
                height,
                colour,
            )
        )
    collisions.append(
        box_collision(
            "east_stand_collision", 0.0, -54.0, 1.50, 118.0, 21.0, 3.0
        )
    )

    # Smaller west-side concrete terrace.
    for index in range(5):
        depth = 3.0
        y = +45.0 + index * depth
        height = 0.40 * (index + 1)
        visuals.append(
            box_visual(
                f"west_concrete_step_{index}",
                -3.0,
                y,
                height / 2.0,
                96.0,
                depth,
                height,
                CONCRETE_RGBA,
            )
        )
    collisions.append(
        box_collision(
            "west_stand_collision", -3.0, 51.0, 1.0, 96.0, 15.0, 2.0
        )
    )

    # Covered stand at the local +x (north) end.
    for index in range(8):
        depth = 3.0
        x = 86.0 + index * depth
        height = 0.46 * (index + 1)
        visuals.append(
            box_visual(
                f"north_grandstand_step_{index}",
                x,
                0.0,
                height / 2.0,
                depth,
                54.0,
                height,
                CONCRETE_RGBA,
            )
        )
    collisions.append(
        box_collision(
            "north_grandstand_collision",
            96.5,
            0.0,
            1.84,
            24.0,
            54.0,
            3.68,
        )
    )

    # Columns and a faceted blue arched roof. Seven broad panels keep the
    # silhouette recognisable without a heavy mesh.
    for x in (87.0, 107.0):
        for y in (-26.0, -13.0, 0.0, 13.0, 26.0):
            visuals.append(
                cylinder_visual(
                    f"north_roof_column_{x:g}_{y:g}",
                    x,
                    y,
                    3.6,
                    0.15,
                    7.2,
                    METAL_RGBA,
                )
            )

    panel_count = 7
    panel_spacing = 3.25
    panel_size_x = 3.45
    roof_centre_x = 97.0
    for index in range(panel_count):
        normalized = (index - (panel_count - 1) / 2.0) / (
            (panel_count - 1) / 2.0
        )
        x = roof_centre_x + (index - 3) * panel_spacing
        z = 7.25 + 1.75 * (1.0 - normalized * normalized)
        pitch = -0.22 * normalized
        visuals.append(
            box_visual(
                f"north_blue_roof_panel_{index}",
                x,
                0.0,
                z,
                panel_size_x,
                58.0,
                0.16,
                ROOF_BLUE_RGBA,
                pitch=pitch,
            )
        )


def add_tree(visuals, index, x, y, height):
    trunk_height = height * 0.48
    crown_radius = min(2.35, height * 0.35)
    visuals.append(
        cylinder_visual(
            f"tree_{index}_trunk",
            x,
            y,
            trunk_height / 2.0,
            0.20,
            trunk_height,
            TREE_TRUNK_RGBA,
        )
    )
    visuals.append(
        sphere_visual(
            f"tree_{index}_crown",
            x,
            y,
            trunk_height + crown_radius * 0.72,
            crown_radius,
            TREE_GREEN_RGBA,
        )
    )


def add_trees(visuals):
    positions = []
    for index, x in enumerate((-72, -48, -24, 0, 24, 48, 72)):
        positions.append((float(x), -66.0, 5.0 + 0.30 * (index % 3)))
        positions.append((float(x), +66.0, 5.2 + 0.25 * ((index + 1) % 3)))
    positions.extend(
        (
            (+110.0, -43.0, 5.5),
            (+110.0, +43.0, 5.2),
            (-110.0, -52.0, 5.0),
            (-110.0, +52.0, 5.3),
        )
    )

    for index, (x, y, height) in enumerate(positions):
        add_tree(visuals, index, x, y, height)


def add_floodlight(visuals, index, x, y, crossbar_along_x=True):
    visuals.append(
        cylinder_visual(
            f"floodlight_{index}_mast",
            x,
            y,
            9.0,
            0.16,
            18.0,
            METAL_RGBA,
        )
    )
    size_x, size_y = (
        (4.0, 0.20) if crossbar_along_x else (0.20, 4.0)
    )
    visuals.append(
        box_visual(
            f"floodlight_{index}_crossbar",
            x,
            y,
            18.05,
            size_x,
            size_y,
            0.20,
            METAL_RGBA,
        )
    )
    for lamp_index, offset in enumerate((-1.5, -0.5, 0.5, 1.5)):
        lamp_x = x + offset if crossbar_along_x else x
        lamp_y = y if crossbar_along_x else y + offset
        visuals.append(
            box_visual(
                f"floodlight_{index}_lamp_{lamp_index}",
                lamp_x,
                lamp_y,
                17.82,
                0.48,
                0.32,
                0.30,
                LAMP_RGBA,
            )
        )


def add_floodlights(visuals):
    for index, (x, y, along_x) in enumerate(
        (
            (-65.0, -65.0, True),
            (+65.0, -65.0, True),
            (-65.0, +65.0, True),
            (+65.0, +65.0, True),
            (+109.0, -35.0, False),
            (+109.0, +35.0, False),
        )
    ):
        add_floodlight(visuals, index, x, y, along_x)


def add_building(
    visuals,
    name,
    x,
    y,
    size_x,
    size_y,
    height,
    wall_rgba,
):
    visuals.append(
        box_visual(
            f"{name}_body",
            x,
            y,
            height / 2.0,
            size_x,
            size_y,
            height,
            wall_rgba,
        )
    )
    visuals.append(
        box_visual(
            f"{name}_roof",
            x,
            y,
            height + 0.16,
            size_x + 0.5,
            size_y + 0.5,
            0.32,
            STONE_DARK_RGBA,
        )
    )

    # A few dark window bands are enough to provide scale from an aerial view.
    facade_x = x + size_x / 2.0 + 0.011
    for floor in range(1, max(2, int(height // 3.0))):
        visuals.append(
            box_visual(
                f"{name}_window_band_{floor}",
                facade_x,
                y,
                floor * 2.7,
                0.025,
                size_y * 0.72,
                0.65,
                WINDOW_RGBA,
            )
        )


def add_surroundings(visuals):
    # Simplified campus blocks south of the stadium; deliberately generic and
    # unlabelled so they provide visual context without copying architecture.
    add_building(
        visuals,
        "southwest_campus_block",
        -102.0,
        +34.0,
        18.0,
        22.0,
        12.0,
        BUILDING_BEIGE_RGBA,
    )
    add_building(
        visuals,
        "southeast_campus_block",
        -102.0,
        -34.0,
        18.0,
        20.0,
        10.0,
        BUILDING_BRICK_RGBA,
    )


def build_sdf():
    visuals = [
        # This thin visual-only slab supplies the stadium apron. Bottom and top
        # are z=0 and z=0.004 m respectively, so it does not alter physics.
        box_visual(
            "stadium_plaza",
            0.0,
            0.0,
            0.002,
            SITE_LENGTH_M,
            SITE_WIDTH_M,
            0.004,
            PLAZA_RGBA,
            cast_shadows=False,
        )
    ]
    collisions = []

    add_football_field(visuals)
    add_blue_court(visuals)
    add_stands(visuals, collisions)
    add_trees(visuals)
    add_floodlights(visuals)
    add_surroundings(visuals)

    visual_body = "\n".join(visuals)
    collision_body = "\n".join(collisions)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated by gen_cju_stadium_model.py; edit the generator, not this file. -->
  <!-- Original primitive-only interpretation. No external texture or mesh. -->
  <model name="cheongju_university_stadium">
    <static>true</static>
    <link name="stadium_link">
{visual_body}
{collision_body}
    </link>
  </model>
</sdf>
"""


def build_config():
    return """<?xml version="1.0"?>
<model>
  <name>cheongju_university_stadium</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Lightweight primitive-only interpretation of Cheongju University main
    complex stadium surroundings. Use together with model://running_track.
  </description>
</model>
"""


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    sdf_path = os.path.join(MODEL_DIR, "model.sdf")
    config_path = os.path.join(MODEL_DIR, "model.config")

    with open(sdf_path, "w", encoding="utf-8") as sdf_file:
        sdf_file.write(build_sdf())
    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(build_config())

    sdf_text = build_sdf()
    print(f"Generated: {sdf_path}")
    print(
        f"  site footprint / x-y extents: "
        f"{SITE_LENGTH_M:.1f} x {SITE_WIDTH_M:.1f} m "
        f"(x +/-{SITE_LENGTH_M / 2:.1f}, y +/-{SITE_WIDTH_M / 2:.1f})"
    )
    print(f"  visuals: {sdf_text.count('<visual name=')}")
    print(f"  simplified remote collisions: {sdf_text.count('<collision name=')}")
    print("  tallest elements: 18.2 m floodlight heads")


if __name__ == "__main__":
    main()
