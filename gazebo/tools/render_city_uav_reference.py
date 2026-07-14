#!/usr/bin/env python3
"""Render the active UAV city and its 69-building spatial-random audit."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
import yaml

import expand_city_for_uav as gen


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "gazebo/validation/path_planning/city_uav_205_reference.png"
DEFAULT_HOME_COPY = Path.home() / "city_uav_205_reference.png"


def compound_patch(
    outer: list[list[float]],
    holes: list[list[list[float]]],
    **kwargs: object,
) -> PathPatch:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in (outer, *holes):
        points = [(float(point[0]), float(point[1])) for point in ring]
        vertices.extend(points + [points[0]])
        codes.extend(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(points) - 1)
            + [MplPath.CLOSEPOLY]
        )
    return PathPatch(MplPath(vertices, codes), **kwargs)


def draw_context(ax: plt.Axes, document: dict, road: object) -> None:
    bounds = document["map"]["bounds_enu_m"]
    xmin, xmax = map(float, bounds["x"])
    ymin, ymax = map(float, bounds["y"])
    ax.imshow(road, extent=(xmin, xmax, ymin, ymax), origin="upper", alpha=0.90)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gazebo world X / East (m)")
    ax.set_ylabel("Gazebo world Y / North (m)")
    ax.grid(color="white", linewidth=0.25, alpha=0.20)

    spawn = document["spawn"]["gazebo_spawn_pose_enu"]
    goal = document["derivation"]["fixed_mission_coordinates_enu_m"]["global_goal"]
    trailer = document["trailer"]["spawn_pose_enu"]
    ax.scatter(spawn["x"], spawn["y"], marker="*", s=150, color="#28e77b", edgecolor="black", zorder=8)
    ax.scatter(goal[0], goal[1], marker="X", s=90, color="#ff5252", edgecolor="black", zorder=8)
    ax.scatter(trailer["x"], trailer["y"], marker="s", s=70, color="#3aa0ff", edgecolor="black", zorder=8)
    scale_x = xmin + 45.0
    scale_y = ymin + 35.0
    ax.plot([scale_x, scale_x + 100.0], [scale_y, scale_y], color="black", linewidth=4, zorder=9)
    ax.text(scale_x + 50.0, scale_y + 10.0, "100 m", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.annotate(
        "N",
        xy=(xmax - 35.0, ymax - 25.0),
        xytext=(xmax - 35.0, ymax - 95.0),
        ha="center",
        fontsize=11,
        weight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#1769aa", "lw": 2.0},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--home-copy", type=Path, default=DEFAULT_HOME_COPY)
    args = parser.parse_args()

    document = yaml.safe_load(gen.OUTPUT_YAML.read_text(encoding="utf-8"))
    buildings = document["obstacles"]["buildings"]
    reduction = document["derivation"]["building_reduction"]
    if len(buildings) != gen.ACTIVE_BUILDING_COUNT:
        raise RuntimeError(f"expected {gen.ACTIVE_BUILDING_COUNT} active buildings, got {len(buildings)}")
    if reduction["removed_count"] != gen.REMOVED_BUILDING_COUNT:
        raise RuntimeError("active city reduction metadata is stale")

    source = yaml.safe_load(gen.SOURCE_YAML.read_text(encoding="utf-8"))
    source_by_id = {record["id"]: record for record in source["obstacles"]["buildings"]}
    removed_geometry = gen.transform_buildings(
        [source_by_id[identifier] for identifier in reduction["removed_ids"]],
        float(document["derivation"]["city_spacing_scale_xy"]),
        float(document["derivation"]["building_footprint_scale_xy"]),
        tuple(map(float, document["derivation"]["anchor_enu_m"])),
    )
    road = plt.imread(gen.OUTPUT_ROAD)

    figure, axes = plt.subplots(1, 2, figsize=(17, 8.5), constrained_layout=True)
    for ax in axes:
        draw_context(ax, document, road)
        for building in buildings:
            ax.add_patch(
                compound_patch(
                    building["footprint"]["outer"],
                    building["footprint"].get("holes", []),
                    facecolor="#59636f",
                    edgecolor="#f1f1f1",
                    linewidth=0.30,
                    alpha=0.96,
                    zorder=4,
                )
            )

    axes[0].set_title(f"Active PX4 UAV city — {len(buildings)} retained buildings")
    axes[1].set_title(
        f"Reduction audit — {reduction['removed_count']} of {reduction['source_count']} removed"
    )
    for building in removed_geometry:
        axes[1].add_patch(
            compound_patch(
                [list(point) for point in building.transformed.outer],
                [[list(point) for point in hole] for hole in building.transformed.holes],
                facecolor="#ff3b30",
                edgecolor="#8b0000",
                linewidth=0.55,
                alpha=0.72,
                hatch="//",
                zorder=6,
            )
        )

    legend = [
        Patch(facecolor="#59636f", edgecolor="white", label="Retained building"),
        Patch(facecolor="#ff3b30", edgecolor="#8b0000", hatch="//", label="Removed building"),
        Line2D([], [], marker="*", markersize=13, color="none", markerfacecolor="#28e77b", markeredgecolor="black", label="PX4 spawn"),
        Line2D([], [], marker="X", markersize=9, color="none", markerfacecolor="#ff5252", markeredgecolor="black", label="Mission goal"),
        Line2D([], [], marker="s", markersize=8, color="none", markerfacecolor="#3aa0ff", markeredgecolor="black", label="Trailer"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.065))
    figure.suptitle(
        "PX4-ROS2 city map — rolled-back jo XY, 10–20 m skyline, spatial-random 25% reduction",
        fontsize=14,
        weight="bold",
        y=1.025,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="#eef1f4")
    plt.close(figure)
    args.home_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.output, args.home_copy)
    print(f"active_buildings={len(buildings)} removed_buildings={reduction['removed_count']}")
    print(f"output={args.output}")
    print(f"home_copy={args.home_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
