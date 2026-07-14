#!/usr/bin/env python3
"""Render a binary A* occupancy image from the active city YAML contract.

The PNG contains only exact active-building footprints: free cells are white
(255), occupied cells are black (0).  Image row 0 is the YAML map's maximum
Y (north) and column 0 is its minimum X (west).  Building holes are kept free.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, PngImagePlugin
import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "gazebo/maps/city_coordinates_uav.yaml"
DEFAULT_OUTPUT = REPO / "gazebo/maps/city_uav_astar_2d_map.png"
DEFAULT_RESOLUTION_M = 0.5


def positive_integer_cells(span_m: float, resolution_m: float, axis: str) -> int:
    cells = span_m / resolution_m
    rounded = round(cells)
    if rounded <= 0 or not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{axis} span {span_m:g} m is not an integer number of "
            f"{resolution_m:g} m cells"
        )
    return int(rounded)


def ring_pixels(
    ring: list[list[float]],
    *,
    xmin: float,
    ymax: float,
    resolution_m: float,
    width: int,
    height: int,
) -> np.ndarray:
    if len(ring) < 3:
        raise ValueError("each footprint ring must contain at least three vertices")
    points = np.asarray(ring, dtype=np.float64)
    if points.shape != (len(ring), 2) or not np.isfinite(points).all():
        raise ValueError("footprint vertices must be finite XY pairs")
    pixels = np.rint(
        np.column_stack(
            (
                (points[:, 0] - xmin) / resolution_m,
                (ymax - points[:, 1]) / resolution_m,
            )
        )
    ).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return pixels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resolution-m-per-pixel",
        type=float,
        default=DEFAULT_RESOLUTION_M,
    )
    args = parser.parse_args()

    if args.resolution_m_per_pixel <= 0.0:
        raise ValueError("resolution must be positive")

    document = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    bounds = document["map"]["bounds_enu_m"]
    xmin, xmax = map(float, bounds["x"])
    ymin, ymax = map(float, bounds["y"])
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("map bounds must be strictly increasing")

    resolution_m = float(args.resolution_m_per_pixel)
    width = positive_integer_cells(xmax - xmin, resolution_m, "X")
    height = positive_integer_cells(ymax - ymin, resolution_m, "Y")

    buildings = document["obstacles"]["buildings"]
    declared_count = int(document["obstacles"]["summary"]["building_count"])
    if len(buildings) != declared_count:
        raise ValueError(
            f"YAML declares {declared_count} buildings but contains {len(buildings)}"
        )

    occupied = np.zeros((height, width), dtype=np.uint8)
    hole_count = 0
    coordinate_tolerance = 1e-6
    for building in buildings:
        footprint = building["footprint"]
        outer = footprint["outer"]
        holes = footprint.get("holes", [])
        for ring in [outer, *holes]:
            for x, y in ring:
                if not (
                    xmin - coordinate_tolerance <= float(x) <= xmax + coordinate_tolerance
                    and ymin - coordinate_tolerance <= float(y) <= ymax + coordinate_tolerance
                ):
                    raise ValueError(f"{building['id']} has a vertex outside the map bounds")

        building_mask = np.zeros_like(occupied)
        cv2.fillPoly(
            building_mask,
            [
                ring_pixels(
                    outer,
                    xmin=xmin,
                    ymax=ymax,
                    resolution_m=resolution_m,
                    width=width,
                    height=height,
                )
            ],
            255,
        )
        for hole in holes:
            cv2.fillPoly(
                building_mask,
                [
                    ring_pixels(
                        hole,
                        xmin=xmin,
                        ymax=ymax,
                        resolution_m=resolution_m,
                        width=width,
                        height=height,
                    )
                ],
                0,
            )
        occupied = cv2.bitwise_or(occupied, building_mask)
        hole_count += len(holes)

    image = np.where(occupied > 0, 0, 255).astype(np.uint8)
    if set(np.unique(image).tolist()) != {0, 255}:
        raise RuntimeError("rendered map is not binary")

    metadata = PngImagePlugin.PngInfo()
    try:
        source_label = args.input.resolve().relative_to(REPO).as_posix()
    except ValueError:
        source_label = str(args.input.resolve())
    metadata.add_text("source_yaml", source_label)
    metadata.add_text("frame_id", "gazebo_world_enu")
    metadata.add_text("bounds_enu_m", f"x:[{xmin:g},{xmax:g}], y:[{ymin:g},{ymax:g}]")
    metadata.add_text("resolution_m_per_pixel", f"{resolution_m:g}")
    metadata.add_text("raster_origin", f"row0_y={ymax:g}, col0_x={xmin:g}")
    metadata.add_text("pixel_values", "free=255, occupied=0")
    metadata.add_text("active_buildings", str(len(buildings)))
    metadata.add_text("preserved_holes", str(hole_count))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(args.output, pnginfo=metadata, optimize=True)

    rendered = np.asarray(Image.open(args.output))
    if rendered.shape != (height, width) or set(np.unique(rendered).tolist()) != {0, 255}:
        raise RuntimeError("saved PNG failed binary occupancy validation")
    occupied_cells = int(np.count_nonzero(rendered == 0))
    if occupied_cells == 0:
        raise RuntimeError("saved PNG contains no occupied cells")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"source={args.input}")
    print(f"output={args.output}")
    print(f"bounds_enu_m=x[{xmin:g},{xmax:g}] y[{ymin:g},{ymax:g}]")
    print(f"resolution_m_per_pixel={resolution_m:g}")
    print(f"image_size_px={width}x{height}")
    print(f"active_buildings={len(buildings)} preserved_holes={hole_count}")
    print(f"occupied_cells={occupied_cells} free_cells={rendered.size - occupied_cells}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
