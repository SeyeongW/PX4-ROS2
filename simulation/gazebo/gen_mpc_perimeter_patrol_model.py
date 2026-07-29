#!/usr/bin/env python3
"""Generate the 1,000 x 100 m moving-landing shuttle guide.

The model is deliberately lightweight and self-contained:

* a visible 1,000 m straight guide matching the trailer controller;
* two visible reversal markers;
* no collision or visual obstacles.

No downloaded mesh, image, or texture is used. The generator writes only
``models/mpc_perimeter_patrol_field``.
"""

from __future__ import annotations

import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "mpc_perimeter_patrol_field"
MODEL_DIR = SCRIPT_DIR / "models" / MODEL_NAME

MAP_X_MIN_M = 0.0
MAP_X_MAX_M = 1000.0
MAP_Y_MIN_M = 0.0
MAP_Y_MAX_M = 100.0
SHUTTLE_START_XY_M = (0.0, 50.0)
SHUTTLE_END_XY_M = (1000.0, 50.0)
SHUTTLE_DISTANCE_M = 1000.0
ROUTE_WIDTH_M = 0.35
ROUTE_Z_M = 0.010
ROUTE_HEIGHT_M = 0.008
ROUTE_RGBA = (0.08, 0.62, 0.95, 1.0)
WAYPOINT_RGBA = (1.0, 0.82, 0.05, 1.0)
OBSTACLE_ORANGE_RGBA = (0.95, 0.38, 0.05, 1.0)

OBSTACLE_HEIGHT_M = 30.0
DRONE_SPAWN_XY_M = (15.0, 40.0)
TRAILER_SPAWN_XY_M = SHUTTLE_START_XY_M
OBSTACLES: tuple[dict[str, object], ...] = ()


def material_xml(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, alpha = rgba
    return f"""        <material>
          <ambient>{red} {green} {blue} {alpha}</ambient>
          <diffuse>{red} {green} {blue} {alpha}</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
        </material>"""


def route_box(
    name: str,
    x: float,
    y: float,
    yaw: float,
    length: float,
) -> str:
    return f"""      <visual name="{name}">
        <pose>{x:.6f} {y:.6f} {ROUTE_Z_M:.6f} 0 0 {yaw:.8f}</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <box>
            <size>{length:.6f} {ROUTE_WIDTH_M:.6f} {ROUTE_HEIGHT_M:.6f}</size>
          </box>
        </geometry>
{material_xml(ROUTE_RGBA)}
      </visual>"""


def waypoint_marker(name: str, x: float, y: float) -> str:
    return f"""      <visual name="{name}">
        <pose>{x:.6f} {y:.6f} {ROUTE_Z_M:.6f} 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry><cylinder><radius>0.55</radius><length>0.012</length></cylinder></geometry>
{material_xml(WAYPOINT_RGBA)}
      </visual>"""


def obstacle_geometry(obstacle: dict[str, object]) -> str:
    if obstacle["shape"] == "box":
        return (
            "<box><size>"
            f"{obstacle['size_x']} {obstacle['size_y']} "
            f"{OBSTACLE_HEIGHT_M}"
            "</size></box>"
        )
    return (
        "<cylinder>"
        f"<radius>{obstacle['radius']}</radius>"
        f"<length>{OBSTACLE_HEIGHT_M}</length>"
        "</cylinder>"
    )


def obstacle_elements(obstacle: dict[str, object]) -> str:
    name = str(obstacle["name"])
    yaw = float(obstacle.get("yaw", 0.0))
    pose = (
        f"{obstacle['x']} {obstacle['y']} {OBSTACLE_HEIGHT_M / 2.0} "
        f"0 0 {yaw:.8f}"
    )
    geometry = obstacle_geometry(obstacle)
    rgba = obstacle["rgba"]
    assert isinstance(rgba, tuple)
    return f"""      <collision name="{name}_collision">
        <pose>{pose}</pose>
        <geometry>{geometry}</geometry>
      </collision>
      <visual name="{name}_visual">
        <pose>{pose}</pose>
        <cast_shadows>true</cast_shadows>
        <geometry>{geometry}</geometry>
{material_xml(rgba)}
      </visual>"""


def build_route_visuals() -> list[str]:
    route_center_x_m = (
        SHUTTLE_START_XY_M[0] + SHUTTLE_END_XY_M[0]
    ) / 2.0
    route_center_y_m = (
        SHUTTLE_START_XY_M[1] + SHUTTLE_END_XY_M[1]
    ) / 2.0
    visuals = [
        route_box(
            "shuttle_centerline",
            route_center_x_m,
            route_center_y_m,
            0.0,
            SHUTTLE_DISTANCE_M,
        ),
        waypoint_marker(
            "shuttle_reversal_west",
            *SHUTTLE_START_XY_M,
        ),
        waypoint_marker(
            "shuttle_reversal_east",
            *SHUTTLE_END_XY_M,
        ),
    ]
    return visuals


def obstacle_bounds(
    obstacle: dict[str, object],
) -> tuple[float, float, float, float]:
    x = float(obstacle["x"])
    y = float(obstacle["y"])
    if obstacle["shape"] == "cylinder":
        half_x = half_y = float(obstacle["radius"])
    else:
        half_size_x = 0.5 * float(obstacle["size_x"])
        half_size_y = 0.5 * float(obstacle["size_y"])
        yaw = float(obstacle.get("yaw", 0.0))
        half_x = (
            abs(math.cos(yaw)) * half_size_x
            + abs(math.sin(yaw)) * half_size_y
        )
        half_y = (
            abs(math.sin(yaw)) * half_size_x
            + abs(math.cos(yaw)) * half_size_y
        )
    return x - half_x, y - half_y, x + half_x, y + half_y


def segment_intersects_aabb(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    """Return whether a 2-D segment intersects an axis-aligned rectangle."""
    min_x, min_y, max_x, max_y = bounds
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    t_min = 0.0
    t_max = 1.0
    for direction, offset in (
        (-delta_x, start[0] - min_x),
        (delta_x, max_x - start[0]),
        (-delta_y, start[1] - min_y),
        (delta_y, max_y - start[1]),
    ):
        if abs(direction) < 1e-12:
            if offset < 0.0:
                return False
            continue
        fraction = offset / direction
        if direction < 0.0:
            t_min = max(t_min, fraction)
        else:
            t_max = min(t_max, fraction)
        if t_min > t_max:
            return False
    return True


def shuttle_route_samples(
    sample_count: int = 2000,
) -> list[tuple[float, float]]:
    """Sample the exact 1,000 m centreline for optional clearance checks."""
    return [
        (
            SHUTTLE_START_XY_M[0]
            + (SHUTTLE_END_XY_M[0] - SHUTTLE_START_XY_M[0])
            * index
            / sample_count,
            SHUTTLE_START_XY_M[1]
            + (SHUTTLE_END_XY_M[1] - SHUTTLE_START_XY_M[1])
            * index
            / sample_count,
        )
        for index in range(sample_count + 1)
    ]


def platform_to_obstacle_clearance(
    route_point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float:
    platform_half_extent_m = 2.5
    min_x, min_y, max_x, max_y = bounds
    platform_min_x = route_point[0] - platform_half_extent_m
    platform_max_x = route_point[0] + platform_half_extent_m
    platform_min_y = route_point[1] - platform_half_extent_m
    platform_max_y = route_point[1] + platform_half_extent_m
    gap_x = max(
        min_x - platform_max_x,
        platform_min_x - max_x,
        0.0,
    )
    gap_y = max(
        min_y - platform_max_y,
        platform_min_y - max_y,
        0.0,
    )
    return math.hypot(gap_x, gap_y)


def minimum_trailer_obstacle_clearance() -> float:
    if not OBSTACLES:
        return math.inf
    route_samples = shuttle_route_samples()
    return min(
        platform_to_obstacle_clearance(point, obstacle_bounds(obstacle))
        for point in route_samples
        for obstacle in OBSTACLES
    )


def validate_layout() -> None:
    map_length_m = MAP_X_MAX_M - MAP_X_MIN_M
    map_width_m = MAP_Y_MAX_M - MAP_Y_MIN_M
    if not math.isclose(map_length_m, 1000.0, abs_tol=1e-9):
        raise ValueError(f"map length is {map_length_m} m, expected 1,000 m")
    if not math.isclose(map_width_m, 100.0, abs_tol=1e-9):
        raise ValueError(f"map width is {map_width_m} m, expected 100 m")
    route_distance_m = math.dist(
        SHUTTLE_START_XY_M,
        SHUTTLE_END_XY_M,
    )
    if not math.isclose(
        route_distance_m,
        SHUTTLE_DISTANCE_M,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"shuttle leg is {route_distance_m} m, "
            f"expected {SHUTTLE_DISTANCE_M} m"
        )
    if (
        SHUTTLE_START_XY_M != (MAP_X_MIN_M, 50.0)
        or SHUTTLE_END_XY_M != (MAP_X_MAX_M, 50.0)
    ):
        raise ValueError("shuttle endpoints must coincide with the map ends")
    platform_half_width_m = 2.5
    if not (
        MAP_Y_MIN_M + platform_half_width_m
        <= SHUTTLE_START_XY_M[1]
        <= MAP_Y_MAX_M - platform_half_width_m
    ):
        raise ValueError("trailer centreline is outside the map width")
    if OBSTACLES:
        raise ValueError("the baseline shuttle course must remain obstacle-free")


def build_sdf() -> str:
    route_visuals = build_route_visuals()
    obstacles = [obstacle_elements(obstacle) for obstacle in OBSTACLES]
    body = "\n".join(route_visuals + obstacles)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated by gen_mpc_perimeter_patrol_model.py. -->
  <!-- Visual route guide; no obstacles are included. -->
  <model name="{MODEL_NAME}">
    <static>true</static>
    <link name="field_link">
{body}
    </link>
  </model>
</sdf>
"""


def build_config() -> str:
    return f"""<?xml version="1.0"?>
<model>
  <name>{MODEL_NAME}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    1,000 x 100 m straight shuttle guide without obstacles.
  </description>
</model>
"""


def main() -> None:
    validate_layout()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sdf = build_sdf()
    (MODEL_DIR / "model.sdf").write_text(sdf, encoding="utf-8")
    (MODEL_DIR / "model.config").write_text(
        build_config(), encoding="utf-8"
    )

    print(f"Generated: {MODEL_DIR}")
    print(
        f"  map: {MAP_X_MAX_M - MAP_X_MIN_M:.0f} x "
        f"{MAP_Y_MAX_M - MAP_Y_MIN_M:.0f} m; "
        f"shuttle leg={SHUTTLE_DISTANCE_M:.3f} m, "
        f"cycle={2.0 * SHUTTLE_DISTANCE_M:.3f} m"
    )
    print(
        f"  route visuals: {len(build_route_visuals())}; "
        f"obstacles: {len(OBSTACLES)}"
    )
    if not OBSTACLES:
        print("  obstacle clearance/direct-path checks: not applicable")
        return

    bounds = [obstacle_bounds(obstacle) for obstacle in OBSTACLES]
    direct_intersection_count = sum(
        segment_intersects_aabb(
            TRAILER_SPAWN_XY_M,
            DRONE_SPAWN_XY_M,
            obstacle_bounds(obstacle),
        )
        for obstacle in OBSTACLES
    )
    print(
        "  minimum ideal trailer-footprint clearance: "
        f"{minimum_trailer_obstacle_clearance():.3f} m; "
        f"direct-path intersections: {direct_intersection_count}"
    )
    print(
        "  obstacle AABB union: "
        f"x=[{min(bound[0] for bound in bounds):.3f}, "
        f"{max(bound[2] for bound in bounds):.3f}], "
        f"y=[{min(bound[1] for bound in bounds):.3f}, "
        f"{max(bound[3] for bound in bounds):.3f}]"
    )


if __name__ == "__main__":
    main()
