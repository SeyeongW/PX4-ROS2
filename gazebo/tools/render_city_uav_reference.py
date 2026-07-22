#!/usr/bin/env python3
"""Render the active UAV city and its 69-building spatial-random audit."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
import yaml

import expand_city_for_uav as gen


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "gazebo/validation/path_planning/city_uav_205_reference.png"
DEFAULT_HOME_COPY = Path.home() / "city_uav_mesh_collision_map.png"
COLLADA_NAMESPACE = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


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


def dae_xy_triangle_edges(path: Path) -> list[list[tuple[float, float]]]:
    """Read the actual checked-in DAE triangle stream for a mesh overlay."""
    root = ET.parse(path).getroot()
    array = root.find(".//c:source[@id='positions']/c:float_array", COLLADA_NAMESPACE)
    if array is None or not array.text:
        raise RuntimeError("DAE positions are missing")
    values = [float(value) for value in array.text.split()]
    if len(values) % 9:
        raise RuntimeError("DAE position stream is not triangle-aligned")
    edges: list[list[tuple[float, float]]] = []
    for offset in range(0, len(values), 9):
        triangle = [
            (values[offset + 3 * index], values[offset + 3 * index + 1])
            for index in range(3)
        ]
        edges.extend(
            [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ]
        )
    return edges


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
    heights = [float(building["height_above_ground_m"]) for building in buildings]
    height_min = min(heights)
    height_max = max(heights)
    height_norm = Normalize(vmin=height_min, vmax=height_max)
    height_cmap = plt.get_cmap("turbo")

    figure, axes = plt.subplots(1, 3, figsize=(24, 8.2), constrained_layout=True)
    for axis_index, ax in enumerate(axes):
        draw_context(ax, document, road)
        if axis_index == 1:
            continue
        for building in buildings:
            facecolor = (
                height_cmap(height_norm(float(building["height_above_ground_m"])))
                if axis_index == 0
                else "#59636f"
            )
            ax.add_patch(
                compound_patch(
                    building["footprint"]["outer"],
                    building["footprint"].get("holes", []),
                    facecolor=facecolor,
                    edgecolor="#f1f1f1",
                    linewidth=0.30,
                    alpha=0.96,
                    zorder=4,
                )
            )

    mesh_edges = dae_xy_triangle_edges(gen.OUTPUT_DAE)
    axes[1].add_collection(
        LineCollection(
            mesh_edges,
            colors="#00d5ff",
            linewidths=0.12,
            alpha=0.42,
            zorder=5,
            rasterized=True,
        )
    )
    for building in buildings:
        axes[1].add_patch(
            compound_patch(
                building["footprint"]["outer"],
                building["footprint"].get("holes", []),
                facecolor="none",
                edgecolor="#ffd400",
                linewidth=0.45,
                linestyle="--",
                alpha=0.95,
                zorder=6,
            )
        )

    axes[0].set_title(
        f"Active PX4 UAV city — height distribution {height_min:g}–{height_max:g} m"
    )
    axes[1].set_title(
        "DAE visual triangles (cyan) / YAML + DART collision rings (yellow)"
    )
    axes[2].set_title(
        f"Reduction audit — {reduction['removed_count']} of {reduction['source_count']} removed"
    )
    for building in removed_geometry:
        axes[2].add_patch(
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

    height_scalar = ScalarMappable(norm=height_norm, cmap=height_cmap)
    height_scalar.set_array([])
    colorbar = figure.colorbar(height_scalar, ax=axes[0], fraction=0.045, pad=0.018)
    colorbar.set_label("Building height above ground (m)")

    legend = [
        Patch(facecolor="#59636f", edgecolor="white", label="Retained building"),
        Patch(facecolor="#ff3b30", edgecolor="#8b0000", hatch="//", label="Removed building"),
        Line2D([], [], color="#00d5ff", lw=2, label="DAE visual triangles"),
        Line2D([], [], color="#ffd400", lw=2, ls="--", label="YAML / DART collision rings"),
        Line2D([], [], marker="*", markersize=13, color="none", markerfacecolor="#28e77b", markeredgecolor="black", label="PX4 spawn"),
        Line2D([], [], marker="X", markersize=9, color="none", markerfacecolor="#ff5252", markeredgecolor="black", label="Mission goal"),
        Line2D([], [], marker="s", markersize=8, color="none", markerfacecolor="#3aa0ff", markeredgecolor="black", label="Trailer"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=7, frameon=True, bbox_to_anchor=(0.5, -0.065))
    map_size = float(document["map"]["bounds_enu_m"]["x"][1]) - float(
        document["map"]["bounds_enu_m"]["x"][0]
    )
    figure.suptitle(
        f"PX4-ROS2 city map — {map_size:g} x {map_size:g} m, uniform 2.5x XY, exact DART collision prisms",
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
