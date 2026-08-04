#!/usr/bin/env python3
"""청주대학교 종합운동장용 400 m 육상 트랙 모델 생성기.

OpenStreetMap 외곽선에서 확인되는 약 179 x 87 m 비율과 공식 자료의
400 m 트랙 설명을 함께 만족하도록 1번 레인 계측선을 구성한다. 운동장 모델과
쉽게 회전ㆍ배치할 수 있도록 장축은 로컬 x축, 경기장 중심은 로컬 원점에 둔다.

트랙 폭은 1.22 m 차선 8개(총 9.76 m)를 시각적으로 근사한다. 청주대학교
공식 시설 사진에서 보이는 어두운 회색 트랙, 붉은 직선 구간, 녹색 가장자리의
인상을 원본 이미지나 텍스처를 복제하지 않고 SDF box primitive로만 표현한다.

실행:
    python3 simulation/gazebo/gen_track_model.py

생성:
    simulation/gazebo/models/running_track/model.sdf
    simulation/gazebo/models/running_track/model.config
"""

import math
import os


# 8차선 시각 근사. 노면은 중심 경로 좌우로 절반씩 펼친다.
LANE_COUNT = 8
LANE_WIDTH_M = 1.22
TRACK_WIDTH_M = LANE_COUNT * LANE_WIDTH_M
TRACK_HALF_WIDTH_M = TRACK_WIDTH_M / 2.0
LINE_WIDTH_M = 0.08

# OSM 외곽 폭(약 87.1 m)에 맞춘 안쪽 경계 반지름이다. 1번 레인 계측선은
# 안쪽 경계에서 0.30 m 떨어져 있으며, 직선 길이는 그 계측선이 정확히 400 m가
# 되도록 계산한다. CURVE_RADIUS_M은 시각 노면의 가운데 반지름이다.
INNER_CURVE_RADIUS_M = 33.77
MEASUREMENT_LINE_OFFSET_M = 0.30
MEASUREMENT_CURVE_RADIUS_M = (
    INNER_CURVE_RADIUS_M + MEASUREMENT_LINE_OFFSET_M
)
STRAIGHT_LENGTH_M = (
    400.0 - 2.0 * math.pi * MEASUREMENT_CURVE_RADIUS_M
) / 2.0
CURVE_RADIUS_M = INNER_CURVE_RADIUS_M + TRACK_HALF_WIDTH_M

# 곡선 하나를 24개 box 현으로 근사한다. 8개 차선에서도 충분히 매끄럽고,
# 수백 개 수준의 visual만 생성하므로 시뮬레이션 부하가 작다.
ARC_SEGMENTS = 24

# 모든 노면 시각 요소의 최고점은 z=0.02 m 이하이다.
SURFACE_Z_M = 0.008
SURFACE_HEIGHT_M = 0.006
ACCENT_Z_M = 0.012
ACCENT_HEIGHT_M = 0.004
LINE_Z_M = 0.0165
LINE_HEIGHT_M = 0.003

DARK_TRACK_RGBA = (0.20, 0.21, 0.22, 1.0)
RED_STRAIGHT_RGBA = (0.54, 0.18, 0.13, 1.0)
GREEN_APRON_RGBA = (0.18, 0.38, 0.18, 1.0)
WHITE_RGBA = (0.94, 0.94, 0.91, 1.0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "running_track")


def path_segments(offset_m=0.0):
    """Return stadium-path box segments at an inward offset.

    ``offset_m > 0`` moves the path toward the stadium centre. The lower
    straight travels east (+x), the right curve travels to the upper straight,
    and the upper straight travels west (-x).
    """

    radius = CURVE_RADIUS_M - offset_m
    if radius <= 0.0:
        raise ValueError("path offset is larger than the curve radius")

    segments = [
        (0.0, -CURVE_RADIUS_M + offset_m, 0.0, STRAIGHT_LENGTH_M),
        (0.0, +CURVE_RADIUS_M - offset_m, math.pi, STRAIGHT_LENGTH_M),
    ]

    chord_length = 2.0 * radius * math.sin(math.pi / (2.0 * ARC_SEGMENTS))
    # A small overlap closes hairline gaps between adjacent chord boxes.
    chord_length *= 1.04

    for side, start_angle in (
        (+1.0, -math.pi / 2.0),
        (-1.0, +math.pi / 2.0),
    ):
        curve_centre_x = side * STRAIGHT_LENGTH_M / 2.0
        for index in range(ARC_SEGMENTS):
            theta = start_angle + (index + 0.5) * math.pi / ARC_SEGMENTS
            x = curve_centre_x + radius * math.cos(theta)
            y = radius * math.sin(theta)
            yaw = theta + math.pi / 2.0
            segments.append((x, y, yaw, chord_length))

    return segments


def box_visual(name, x, y, z, yaw, length, width, height, rgba):
    red, green, blue, alpha = rgba
    return f"""      <visual name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <box><size>{length:.4f} {width:.4f} {height:.4f}</size></box>
        </geometry>
        <material>
          <ambient>{red} {green} {blue} {alpha}</ambient>
          <diffuse>{red} {green} {blue} {alpha}</diffuse>
          <specular>0.04 0.04 0.04 1</specular>
        </material>
      </visual>"""


def build_sdf():
    visuals = []

    # 전체 트랙은 어두운 회색으로 표현한다.
    for index, (x, y, yaw, length) in enumerate(path_segments()):
        visuals.append(
            box_visual(
                f"track_surface_{index}",
                x,
                y,
                SURFACE_Z_M,
                yaw,
                length,
                TRACK_WIDTH_M,
                SURFACE_HEIGHT_M,
                DARK_TRACK_RGBA,
            )
        )

    # 공식 사진에서 확인되는 붉은 직선과 그 바깥 녹색 apron을 단순화한다.
    visuals.append(
        box_visual(
            "red_home_straight",
            0.0,
            +CURVE_RADIUS_M,
            ACCENT_Z_M,
            0.0,
            STRAIGHT_LENGTH_M,
            TRACK_WIDTH_M,
            ACCENT_HEIGHT_M,
            RED_STRAIGHT_RGBA,
        )
    )
    visuals.append(
        box_visual(
            "green_home_apron",
            0.0,
            +CURVE_RADIUS_M + TRACK_HALF_WIDTH_M + 1.5,
            SURFACE_Z_M,
            0.0,
            STRAIGHT_LENGTH_M,
            3.0,
            SURFACE_HEIGHT_M,
            GREEN_APRON_RGBA,
        )
    )

    # 8차선에는 안쪽/바깥쪽 경계를 포함한 9개의 흰 선이 필요하다.
    for boundary_index in range(LANE_COUNT + 1):
        lateral_from_outer = boundary_index * LANE_WIDTH_M
        offset_m = -TRACK_HALF_WIDTH_M + lateral_from_outer
        for segment_index, (x, y, yaw, length) in enumerate(
            path_segments(offset_m)
        ):
            visuals.append(
                box_visual(
                    f"lane_{boundary_index}_{segment_index}",
                    x,
                    y,
                    LINE_Z_M,
                    yaw,
                    length,
                    LINE_WIDTH_M,
                    LINE_HEIGHT_M,
                    WHITE_RGBA,
                )
            )

    body = "\n".join(visuals)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <!-- Generated by gen_track_model.py; edit the generator, not this file. -->
  <!-- Centred 8-lane visual approximation of the CJU 400 m athletic track.
       Lane-1 measurement line: 2 * {STRAIGHT_LENGTH_M:.5f} m +
       2 * pi * {MEASUREMENT_CURVE_RADIUS_M:.2f} m = 400 m.
       Local long axis: x. Overall track surface: approximately
       {STRAIGHT_LENGTH_M + 2.0 * (CURVE_RADIUS_M + TRACK_HALF_WIDTH_M):.2f} m
       by {2.0 * (CURVE_RADIUS_M + TRACK_HALF_WIDTH_M):.2f} m. -->
  <model name="running_track">
    <static>true</static>
    <link name="link">
{body}
    </link>
  </model>
</sdf>
"""


def build_config():
    return """<?xml version="1.0"?>
<model>
  <name>running_track</name>
  <version>2.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Centred, primitive-only 400 m athletic track with eight visual lanes.
  </description>
</model>
"""


def visual_xy_bounds():
    """Return the exact generated box AABB as (min_x, min_y, max_x, max_y)."""

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

    for x, y, yaw, length in path_segments():
        expand(x, y, yaw, length, TRACK_WIDTH_M)

    expand(
        0.0,
        +CURVE_RADIUS_M,
        0.0,
        STRAIGHT_LENGTH_M,
        TRACK_WIDTH_M,
    )
    expand(
        0.0,
        +CURVE_RADIUS_M + TRACK_HALF_WIDTH_M + 1.5,
        0.0,
        STRAIGHT_LENGTH_M,
        3.0,
    )

    for boundary_index in range(LANE_COUNT + 1):
        offset_m = (
            -TRACK_HALF_WIDTH_M + boundary_index * LANE_WIDTH_M
        )
        for x, y, yaw, length in path_segments(offset_m):
            expand(x, y, yaw, length, LINE_WIDTH_M)

    return min_x, min_y, max_x, max_y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    sdf_path = os.path.join(MODEL_DIR, "model.sdf")
    config_path = os.path.join(MODEL_DIR, "model.config")

    with open(sdf_path, "w", encoding="utf-8") as sdf_file:
        sdf_file.write(build_sdf())
    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(build_config())

    measurement_perimeter = (
        2.0 * STRAIGHT_LENGTH_M
        + 2.0 * math.pi * MEASUREMENT_CURVE_RADIUS_M
    )
    nominal_length = (
        STRAIGHT_LENGTH_M
        + 2.0 * (CURVE_RADIUS_M + TRACK_HALF_WIDTH_M)
    )
    nominal_width = 2.0 * (CURVE_RADIUS_M + TRACK_HALF_WIDTH_M)
    min_x, min_y, max_x, max_y = visual_xy_bounds()
    visual_count = len(path_segments()) * (LANE_COUNT + 1) + len(
        path_segments()
    ) + 2

    print(f"Generated: {sdf_path}")
    print(f"  lane-1 measurement perimeter: {measurement_perimeter:.3f} m")
    print(
        f"  nominal dark-track extents: "
        f"{nominal_length:.3f} x {nominal_width:.3f} m"
    )
    print(
        f"  full visual AABB: x=[{min_x:.3f}, {max_x:.3f}], "
        f"y=[{min_y:.3f}, {max_y:.3f}] "
        f"({max_x - min_x:.3f} x {max_y - min_y:.3f} m)"
    )
    print(
        f"  {LANE_COUNT} lanes x {LANE_WIDTH_M:.2f} m "
        f"= {TRACK_WIDTH_M:.2f} m; {visual_count} visuals"
    )


if __name__ == "__main__":
    main()
