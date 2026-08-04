#!/usr/bin/env python3
"""Generate a lightweight Cheongju University main-stadium scene.

The model is an original low-poly interpretation based on publicly visible
facts and photographs of the Cheongju University main complex stadium:

* a beige dirt football field inside a 400 m oval track,
* concrete / natural-stone open seating,
* the west-side roofed headquarters / royal box,
* blue basketball courts, goal equipment, green rails and a perimeter tree ring.

No photograph, texture, logo, downloaded mesh, or other third-party asset is
embedded. All geometry is composed from SDF box, cylinder, and sphere
primitives. The companion ``running_track`` model supplies the track itself.

Coordinate convention:
    Stadium centre = local (0, 0)
    Stadium long axis = local x
    Covered royal-box stand = local +y (west) straight
    North / south basketball courts = local +x / -x ends

Run:
    python3 simulation/gazebo/gen_cju_stadium_model.py
"""

import math
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    SCRIPT_DIR, "models", "cheongju_university_stadium"
)

# OSM way 431113163 / 1374978221 calibrated dimensions. The model origin is
# the running-track centre and the local +x axis points toward the north end.
SITE_LENGTH_M = 226.8
SITE_WIDTH_M = 128.3
FIELD_CENTER_X_M = 1.7
FIELD_CENTER_Y_M = -0.8
FIELD_LENGTH_M = 111.6
FIELD_WIDTH_M = 70.8
COURTS = (
    ("north", 67.6, -0.2, 17.2, 27.0),
    ("south", -64.0, 0.8, 17.9, 29.6),
)

PLAZA_RGBA = (0.48, 0.48, 0.45, 1.0)
DIRT_RGBA = (0.62, 0.50, 0.34, 1.0)
FIELD_LINE_RGBA = (0.94, 0.94, 0.88, 1.0)
STONE_LIGHT_RGBA = (0.52, 0.50, 0.44, 1.0)
STONE_DARK_RGBA = (0.39, 0.38, 0.35, 1.0)
CONCRETE_RGBA = (0.57, 0.57, 0.55, 1.0)
ROOF_BLUE_RGBA = (0.07, 0.28, 0.58, 1.0)
ROOF_LIGHT_RGBA = (0.52, 0.62, 0.58, 1.0)
COURT_BLUE_RGBA = (0.07, 0.34, 0.58, 1.0)
METAL_RGBA = (0.58, 0.61, 0.63, 1.0)
TREE_TRUNK_RGBA = (0.24, 0.14, 0.07, 1.0)
TREE_GREEN_RGBA = (0.12, 0.31, 0.13, 1.0)
WINDOW_RGBA = (0.12, 0.22, 0.27, 1.0)
RAIL_GREEN_RGBA = (0.08, 0.32, 0.15, 1.0)
RIM_ORANGE_RGBA = (0.90, 0.27, 0.04, 1.0)
NET_RGBA = (0.90, 0.90, 0.86, 0.72)


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
    # OSM way 431605916 is a roughly 111.6 x 70.8 m sand pitch. The track
    # visual is layered above it, trimming the small overlap at both sides.
    visuals.append(
        box_visual(
            "dirt_infield",
            FIELD_CENTER_X_M,
            FIELD_CENTER_Y_M,
            0.007,
            FIELD_LENGTH_M,
            FIELD_WIDTH_M,
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
        (
            "pitch_side_north",
            FIELD_CENTER_X_M,
            FIELD_CENTER_Y_M + pitch_half_width,
            105.0,
            line_width,
        ),
        (
            "pitch_side_south",
            FIELD_CENTER_X_M,
            FIELD_CENTER_Y_M - pitch_half_width,
            105.0,
            line_width,
        ),
        (
            "pitch_end_east",
            FIELD_CENTER_X_M + pitch_half_length,
            FIELD_CENTER_Y_M,
            line_width,
            60.0,
        ),
        (
            "pitch_end_west",
            FIELD_CENTER_X_M - pitch_half_length,
            FIELD_CENTER_Y_M,
            line_width,
            60.0,
        ),
        (
            "pitch_halfway",
            FIELD_CENTER_X_M,
            FIELD_CENTER_Y_M,
            line_width,
            60.0,
        ),
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
                FIELD_CENTER_X_M + circle_radius * math.cos(theta),
                FIELD_CENTER_Y_M + circle_radius * math.sin(theta),
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
            FIELD_CENTER_X_M,
            FIELD_CENTER_Y_M,
            0.0145,
            0.12,
            0.002,
            FIELD_LINE_RGBA,
            cast_shadows=False,
        )
    )

    # Penalty and goal areas.
    for side_name, side_sign in (("east", +1.0), ("west", -1.0)):
        penalty_inner_x = FIELD_CENTER_X_M + side_sign * (
            pitch_half_length - 16.5
        )
        penalty_mid_x = FIELD_CENTER_X_M + side_sign * (
            pitch_half_length - 8.25
        )
        goal_inner_x = FIELD_CENTER_X_M + side_sign * (
            pitch_half_length - 5.5
        )
        goal_mid_x = FIELD_CENTER_X_M + side_sign * (
            pitch_half_length - 2.75
        )

        visuals.append(
            box_visual(
                f"{side_name}_penalty_inner",
                penalty_inner_x,
                FIELD_CENTER_Y_M,
                line_z,
                line_width,
                40.32,
                line_height,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )
        for edge_name, y in (
            ("top", FIELD_CENTER_Y_M + 20.16),
            ("bottom", FIELD_CENTER_Y_M - 20.16),
        ):
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
                FIELD_CENTER_Y_M,
                line_z,
                line_width,
                18.32,
                line_height,
                FIELD_LINE_RGBA,
                cast_shadows=False,
            )
        )
        for edge_name, y in (
            ("top", FIELD_CENTER_Y_M + 9.16),
            ("bottom", FIELD_CENTER_Y_M - 9.16),
        ):
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
                FIELD_CENTER_X_M
                + side_sign * (pitch_half_length - 11.0),
                FIELD_CENTER_Y_M,
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
        goal_x = FIELD_CENTER_X_M + side_sign * 53.0
        for post_name, y in (
            ("north", FIELD_CENTER_Y_M + 3.66),
            ("south", FIELD_CENTER_Y_M - 3.66),
        ):
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
                FIELD_CENTER_Y_M,
                2.44,
                0.11,
                7.43,
                0.11,
                FIELD_LINE_RGBA,
            )
        )

        # A sparse rear net is enough to read as a real goal from the flight
        # camera without adding a texture or collision cage.
        net_x = goal_x + side_sign * 1.5
        for index, z in enumerate((0.45, 0.90, 1.35, 1.80, 2.25)):
            visuals.append(
                box_visual(
                    f"goal_{side_name}_net_horizontal_{index}",
                    net_x,
                    FIELD_CENTER_Y_M,
                    z,
                    0.025,
                    7.32,
                    0.025,
                    NET_RGBA,
                    cast_shadows=False,
                )
            )
        for index, y_offset in enumerate((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)):
            visuals.append(
                box_visual(
                    f"goal_{side_name}_net_vertical_{index}",
                    net_x,
                    FIELD_CENTER_Y_M + y_offset,
                    1.22,
                    0.025,
                    0.025,
                    2.44,
                    NET_RGBA,
                    cast_shadows=False,
                )
            )
        for post_name, y in (("north", FIELD_CENTER_Y_M + 3.66),
                             ("south", FIELD_CENTER_Y_M - 3.66)):
            visuals.append(
                box_visual(
                    f"goal_{side_name}_{post_name}_net_depth",
                    goal_x + side_sign * 0.75,
                    y,
                    1.22,
                    1.5,
                    0.025,
                    0.025,
                    NET_RGBA,
                    pitch=-side_sign * 0.28,
                    cast_shadows=False,
                )
            )


def add_blue_court(
    visuals, name, court_x, court_y, size_x, size_y
):
    """Add one OSM-calibrated blue basketball court."""

    half_x = size_x / 2.0
    half_y = size_y / 2.0
    visuals.append(
        box_visual(
            f"{name}_blue_court",
            court_x,
            court_y,
            0.007,
            size_x,
            size_y,
            0.006,
            COURT_BLUE_RGBA,
            cast_shadows=False,
        )
    )

    line_z = 0.0145
    line_height = 0.002
    line_width = 0.07
    for line_name, x, y, line_size_x, line_size_y in (
        ("side_west", court_x, court_y + half_y, size_x, line_width),
        ("side_east", court_x, court_y - half_y, size_x, line_width),
        ("end_north", court_x + half_x, court_y, line_width, size_y),
        ("end_south", court_x - half_x, court_y, line_width, size_y),
        ("halfway", court_x, court_y, line_width, size_y),
    ):
        visuals.append(
            box_visual(
                f"{name}_court_{line_name}",
                x,
                y,
                line_z,
                line_size_x,
                line_size_y,
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
                f"{name}_court_centre_circle_{index}",
                court_x + radius * math.cos(theta),
                court_y + radius * math.sin(theta),
                line_z,
                chord,
                line_width,
                line_height,
                FIELD_LINE_RGBA,
                yaw=theta + math.pi / 2.0,
                cast_shadows=False,
            )
        )

    # Lightweight backboards, supports and rims are visual-only, like the
    # football goals.
    for end_name, sign in (("north", +1.0), ("south", -1.0)):
        backboard_x = court_x + sign * (half_x - 0.8)
        visuals.append(
            box_visual(
                f"{name}_{end_name}_backboard",
                backboard_x,
                court_y,
                3.05,
                0.08,
                1.80,
                1.05,
                FIELD_LINE_RGBA,
            )
        )
        pole_x = backboard_x + sign * 0.75
        visuals.append(
            cylinder_visual(
                f"{name}_{end_name}_basket_pole",
                pole_x,
                court_y,
                1.55,
                0.07,
                3.10,
                METAL_RGBA,
            )
        )
        visuals.append(
            box_visual(
                f"{name}_{end_name}_basket_arm",
                backboard_x + sign * 0.38,
                court_y,
                3.05,
                0.75,
                0.07,
                0.07,
                METAL_RGBA,
            )
        )
        rim_x = backboard_x - sign * 0.38
        rim_radius = 0.23
        rim_segments = 12
        rim_chord = 2.0 * rim_radius * math.sin(math.pi / rim_segments) * 1.04
        for index in range(rim_segments):
            theta = (index + 0.5) * 2.0 * math.pi / rim_segments
            visuals.append(
                cylinder_visual(
                    f"{name}_{end_name}_rim_{index}",
                    rim_x + rim_radius * math.cos(theta),
                    court_y + rim_radius * math.sin(theta),
                    3.05,
                    0.015,
                    rim_chord,
                    RIM_ORANGE_RGBA,
                    pitch=math.pi / 2.0,
                    yaw=theta + math.pi / 2.0,
                )
            )


def add_blue_courts(visuals):
    for court in COURTS:
        add_blue_court(visuals, *court)


def add_stands(visuals, collisions):
    # The long east stand is the largest open concrete / natural-stone bank.
    for index in range(7):
        depth = 3.0
        y = -45.2 - index * depth
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
                142.0,
                depth,
                height,
                colour,
            )
        )
        collisions.append(
            box_collision(
                f"east_stone_step_{index}_collision",
                0.0,
                y,
                height / 2.0,
                142.0,
                depth,
                height,
            )
        )

    # The west home straight contains the roofed headquarters / royal box.
    for index in range(5):
        depth = 3.0
        # Leave the photographed green apron clear between the west straight
        # and the first terrace riser.
        y = +48.1 + index * depth
        height = 0.40 * (index + 1)
        visuals.append(
            box_visual(
                f"west_concrete_step_{index}",
                0.0,
                y,
                height / 2.0,
                118.0,
                depth,
                height,
                CONCRETE_RGBA,
            )
        )
        collisions.append(
            box_collision(
                f"west_concrete_step_{index}_collision",
                0.0,
                y,
                height / 2.0,
                118.0,
                depth,
                height,
            )
        )

    # Lower open terraces continue around both short ends.
    for end_name, sign in (("north", +1.0), ("south", -1.0)):
        for index in range(6):
            depth = 2.5
            x = sign * (91.5 + index * depth)
            height = 0.38 * (index + 1)
            visuals.append(
                box_visual(
                    f"{end_name}_curve_step_{index}",
                    x,
                    0.0,
                    height / 2.0,
                    depth,
                    70.0,
                    height,
                    CONCRETE_RGBA,
                )
            )
            collisions.append(
                box_collision(
                    f"{end_name}_curve_step_{index}_collision",
                    x,
                    0.0,
                    height / 2.0,
                    depth,
                    70.0,
                    height,
                )
            )

    # Two-storey headquarters / royal-box block under the west canopy.
    visuals.extend(
        (
            box_visual(
                "west_royal_box_body",
                0.0,
                59.0,
                3.0,
                32.0,
                9.0,
                6.0,
                CONCRETE_RGBA,
            ),
            box_visual(
                "west_royal_box_front_windows",
                0.0,
                54.48,
                3.5,
                26.0,
                0.04,
                1.4,
                WINDOW_RGBA,
            ),
        )
    )
    collisions.append(
        box_collision(
            "west_royal_box_collision", 0.0, 59.0, 3.0, 32.0, 9.0, 6.0
        )
    )

    for x in (-32.0, -16.0, 0.0, 16.0, 32.0):
        for y in (49.0, 62.0):
            visuals.append(
                cylinder_visual(
                    f"west_canopy_column_{x:g}_{y:g}",
                    x,
                    y,
                    3.55,
                    0.14,
                    7.10,
                    METAL_RGBA,
                )
            )

    panel_count = 7
    panel_spacing = 2.25
    panel_size_y = 2.40
    roof_centre_y = 55.5
    for index in range(panel_count):
        normalized = (index - (panel_count - 1) / 2.0) / (
            (panel_count - 1) / 2.0
        )
        y = roof_centre_y + (index - 3) * panel_spacing
        z = 7.15 + 1.15 * (1.0 - normalized * normalized)
        roll = 0.18 * normalized
        visuals.append(
            box_visual(
                f"west_canopy_roof_panel_{index}",
                0.0,
                y,
                z,
                68.0,
                panel_size_y,
                0.16,
                ROOF_LIGHT_RGBA,
                roll=roll,
            )
        )

    # Blue steel trim is the recognisable feature in the official photograph.
    for y in (48.6, 62.4):
        visuals.append(
            box_visual(
                f"west_canopy_blue_edge_{y:g}",
                0.0,
                y,
                7.25,
                69.0,
                0.22,
                0.24,
                ROOF_BLUE_RGBA,
            )
        )
    for x in (-34.0, 0.0, 34.0):
        visuals.append(
            box_visual(
                f"west_canopy_blue_rib_{x:g}",
                x,
                55.5,
                8.20,
                0.22,
                14.0,
                0.22,
                ROOF_BLUE_RGBA,
            )
        )


def add_safety_rails(visuals):
    """Add the low green rails visible along the stand fronts."""

    rail_z = 0.72
    for side_name, y, length in (("east", -43.5, 142.0),
                                 ("west", 47.0, 118.0)):
        visuals.append(
            cylinder_visual(
                f"{side_name}_front_rail",
                0.0,
                y,
                rail_z,
                0.035,
                length,
                RAIL_GREEN_RGBA,
                pitch=math.pi / 2.0,
            )
        )
        for index, x in enumerate(range(-56, 57, 14)):
            visuals.append(
                cylinder_visual(
                    f"{side_name}_front_rail_post_{index}",
                    float(x),
                    y,
                    rail_z / 2.0,
                    0.035,
                    rail_z,
                    RAIL_GREEN_RGBA,
                )
            )

    for end_name, x in (("north", 90.0), ("south", -90.0)):
        visuals.append(
            cylinder_visual(
                f"{end_name}_front_rail",
                x,
                0.0,
                rail_z,
                0.035,
                68.0,
                RAIL_GREEN_RGBA,
                pitch=math.pi / 2.0,
                yaw=math.pi / 2.0,
            )
        )
        for index, y in enumerate(range(-30, 31, 12)):
            visuals.append(
                cylinder_visual(
                    f"{end_name}_front_rail_post_{index}",
                    x,
                    float(y),
                    rail_z / 2.0,
                    0.035,
                    rail_z,
                    RAIL_GREEN_RGBA,
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


def add_trees(visuals, position_scale=1.0):
    positions = []
    for index, x in enumerate((-72, -48, -24, 0, 24, 48, 72)):
        positions.append((float(x), -62.0, 5.0 + 0.30 * (index % 3)))
    for index, x in enumerate((-82, -58, 58, 82)):
        positions.append((float(x), +62.0, 5.2 + 0.25 * (index % 3)))
    for end_x in (-108.0, +108.0):
        for index, y in enumerate((-46.0, -23.0, 0.0, 23.0, 46.0)):
            positions.append((end_x, y, 5.0 + 0.25 * (index % 3)))

    for index, (x, y, height) in enumerate(positions):
        add_tree(visuals, index, x * position_scale, y * position_scale, height)


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
    add_blue_courts(visuals)
    add_stands(visuals, collisions)
    add_safety_rails(visuals)
    add_trees(visuals)

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
    OSM-calibrated primitive interpretation of Cheongju University main
    complex stadium. Use together with model://running_track.
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
    print(f"  stand / royal-box collisions: {sdf_text.count('<collision name=')}")
    print("  tallest element: 8.3 m west royal-box canopy")


if __name__ == "__main__":
    main()
