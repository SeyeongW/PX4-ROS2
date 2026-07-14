#!/usr/bin/env python3
"""Build and validate the trailer-safe, completely flat city datum.

The road texture and every building XY footprint stay unchanged. Gazebo's
visual heightmap and engine-neutral collision box are both placed at world
z=0. Building prisms use a deterministic hash-rank skyline bounded to exactly
10--20 m, with a small buried foundation below the common datum. Historical
pre-scale and 2.0--3.5x heights remain in the audit CSV for traceability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import align_city_building_foundations as city_dae


GAZEBO = Path(__file__).resolve().parents[1]
CITY = GAZEBO / "worlds" / "applepark_city"
BUILDINGS = CITY / "mesh" / "buildings.dae"
HEIGHTMAP = CITY / "mesh" / "height_map_city_500m.png"
NORMALMAP = CITY / "mesh" / "normal_map_city_500m.png"
COLLISION = CITY / "mesh" / "city_terrain_collision.obj"
HEIGHT_AUDIT = GAZEBO / "validation" / "city" / "building_height_scaling.csv"
FOUNDATION_AUDIT = GAZEBO / "validation" / "city" / "building_foundation_alignment.csv"

DATUM_Z_M = 0.0
FOUNDATION_Z_M = -0.05
ACTIVE_HEIGHT_PROFILE = "deterministic_hash_rank_10_to_20m_v1"
ACTIVE_HEIGHT_MIN_M = 10.0
ACTIVE_HEIGHT_MAX_M = 20.0
FACTOR_SEED = "px4-ros2-city-height-v2"
EXPECTED_COMPONENTS = 274
EXPECTED_TRIANGLES = 9248
EXPECTED_VECTORS = 27744
EXPECTED_TERRAIN_VERTICES = 16641
EXPECTED_TERRAIN_FACES = 32768


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_float(value: float) -> str:
    if abs(value) < 5e-13:
        return "0"
    return format(float(value), ".12g")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def atomic_text(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def flat_collision_content() -> tuple[str, int, int]:
    lines: list[str] = []
    vertices = faces = 0
    with COLLISION.open("r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("v "):
                parts = line.split()
                require(len(parts) >= 4, "invalid city collision vertex")
                lines.append(f"v {float(parts[1]):.6f} {float(parts[2]):.6f} 0.000000\n")
                vertices += 1
            else:
                lines.append(line)
                if line.startswith("f "):
                    faces += 1
    require(vertices == EXPECTED_TERRAIN_VERTICES, "city collision vertex count changed")
    require(faces == EXPECTED_TERRAIN_FACES, "city collision face count changed")
    return "".join(lines), vertices, faces


def flatten_buildings(*, modify: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = city_dae.DaeData(BUILDINGS)
    components = city_dae.find_components(data)
    heights = read_csv(HEIGHT_AUDIT)
    foundations = read_csv(FOUNDATION_AUDIT)
    require(len(components) == len(heights) == len(foundations) == EXPECTED_COMPONENTS,
            "city building audit count changed")
    require(len(data.position_indices) == EXPECTED_TRIANGLES, "city triangle count changed")
    require(len(data.positions) == EXPECTED_VECTORS, "city vector count changed")

    # The existing factor digest is a stable SHA-256 of the source component.
    # Rank all unique digests so heights are spatially pseudo-random, uniformly
    # distributed, and include the requested exact 10 m / 20 m endpoints.
    for row in heights:
        component_id = int(row["post_height_component_ordinal"])
        expected_digest = hashlib.sha256(
            f"{FACTOR_SEED}:{component_id}".encode("ascii")
        ).hexdigest()
        require(row["factor_sha256"] == expected_digest,
                f"component {component_id} factor digest changed")
    digests = [row["factor_sha256"] for row in heights]
    require(len(set(digests)) == EXPECTED_COMPONENTS, "city height digests are not unique")
    ordered = sorted(
        heights,
        key=lambda row: (row["factor_sha256"], int(row["output_component_ordinal"])),
    )
    bounded_rank = {
        int(row["output_component_ordinal"]): rank for rank, row in enumerate(ordered)
    }

    new_heights: list[dict[str, object]] = []
    new_foundations: list[dict[str, object]] = []
    for ordinal, (faces, height, foundation) in enumerate(
        zip(components, heights, foundations), start=1
    ):
        require(int(height["output_component_ordinal"]) == ordinal,
                f"height audit order changed at {ordinal}")
        require(int(foundation["output_component_ordinal"]) == ordinal,
                f"foundation audit order changed at {ordinal}")
        require(height["post_height_component_ordinal"] ==
                foundation["post_height_component_ordinal"],
                f"component identity changed at {ordinal}")

        position_ids = data.position_indices[faces].reshape(-1)
        points = data.positions[position_ids]
        current_low = float(points[:, 2].min())
        current_high = float(points[:, 2].max())
        bottom = np.isclose(points[:, 2], current_low, atol=1e-6)
        roof = np.isclose(points[:, 2], current_high, atol=1e-6)
        require(np.all(bottom | roof), f"building {ordinal} is not a two-level prism")
        require(int(bottom.sum()) == int(roof.sum()), f"building {ordinal} prism is unbalanced")

        old_height = float(height["old_above_ground_height_m"])
        factor = float(height["factor"])
        scaled_height = float(height["new_above_ground_height_m"])
        require(abs(scaled_height - old_height * factor) < 1e-5,
                f"building {ordinal} random height factor changed")
        rank = bounded_rank[ordinal]
        fraction = rank / (EXPECTED_COMPONENTS - 1)
        bounded_height = ACTIVE_HEIGHT_MIN_M + (
            ACTIVE_HEIGHT_MAX_M - ACTIVE_HEIGHT_MIN_M
        ) * fraction
        above_ground = bounded_height

        if modify:
            data.positions[position_ids[bottom], 2] = FOUNDATION_Z_M
            data.positions[position_ids[roof], 2] = above_ground
        else:
            require(abs(current_low - FOUNDATION_Z_M) < 1e-9,
                    f"building {ordinal} foundation is not on the flat datum")
            require(abs(current_high - above_ground) < 1e-6,
                    f"building {ordinal} roof height changed")

        height_row: dict[str, object] = dict(height)
        height_row.update(
            reference_base_z="0",
            foundation_base_z=format_float(FOUNDATION_Z_M),
            old_roof_z=format_float(old_height),
            bounded_height_rank=rank,
            bounded_height_fraction=format_float(fraction),
            bounded_height_m=format_float(bounded_height),
            active_roof_z=format_float(above_ground),
            active_above_ground_height_m=format_float(above_ground),
            active_profile=ACTIVE_HEIGHT_PROFILE,
        )
        new_heights.append(height_row)

        foundation_row: dict[str, object] = dict(foundation)
        foundation_row.update(
            original_base_z="0",
            foundation_base_z=format_float(FOUNDATION_Z_M),
            visual_terrain_min_z="0",
            collision_terrain_min_z="0",
            foundation_extension_m=format_float(-FOUNDATION_Z_M),
            roof_z=format_float(above_ground),
        )
        new_foundations.append(foundation_row)

    if modify:
        descriptor, name = tempfile.mkstemp(
            dir=BUILDINGS.parent, prefix=".buildings.flat.", suffix=".dae"
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            data.write(temporary)
            temporary.chmod(BUILDINGS.stat().st_mode & 0o777)
            os.replace(temporary, BUILDINGS)
        finally:
            temporary.unlink(missing_ok=True)
    return new_heights, new_foundations


def validate_flat_assets() -> None:
    with Image.open(HEIGHTMAP) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    require(pixels.shape == (257, 257), "flat city heightmap shape changed")
    require(int(pixels.min()) == int(pixels.max()) == 255,
            "city visual heightmap is not a single plane")
    with Image.open(NORMALMAP) as image:
        normals = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(normals.shape == (257, 257, 3), "flat city normal map shape changed")
    require(np.all(normals == np.array([128, 128, 255], dtype=np.uint8)),
            "city normal map is not vertical everywhere")
    zs: list[float] = []
    vertices = faces = 0
    with COLLISION.open("r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("v "):
                zs.append(float(line.split()[3]))
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    require(vertices == EXPECTED_TERRAIN_VERTICES and faces == EXPECTED_TERRAIN_FACES,
            "flat city collision topology changed")
    require(zs and max(abs(z) for z in zs) < 1e-12,
            "city Bullet collision mesh is not z=0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()

    if not args.check:
        height_rows, foundation_rows = flatten_buildings(modify=True)
        Image.fromarray(np.full((257, 257), 255, dtype=np.uint8), mode="L").save(
            HEIGHTMAP, optimize=True
        )
        Image.fromarray(
            np.broadcast_to(np.array([128, 128, 255], dtype=np.uint8), (257, 257, 3)).copy(),
            mode="RGB",
        ).save(NORMALMAP, optimize=True)
        collision_content, _, _ = flat_collision_content()
        atomic_text(COLLISION, collision_content)
        atomic_text(HEIGHT_AUDIT, csv_text(height_rows, list(height_rows[0])))
        atomic_text(FOUNDATION_AUDIT, csv_text(foundation_rows, list(foundation_rows[0])))

    expected_heights, expected_foundations = flatten_buildings(modify=False)
    require(HEIGHT_AUDIT.read_text(encoding="utf-8") ==
            csv_text(expected_heights, list(expected_heights[0])),
            "flat city height audit differs from geometry")
    require(FOUNDATION_AUDIT.read_text(encoding="utf-8") ==
            csv_text(expected_foundations, list(expected_foundations[0])),
            "flat city foundation audit differs from geometry")
    validate_flat_assets()
    print("PX4-ROS2 completely flat city asset validation")
    print("result=PASS")
    print(f"datum_z_m={DATUM_Z_M:.6f} foundation_z_m={FOUNDATION_Z_M:.6f}")
    print(f"buildings={EXPECTED_COMPONENTS} terrain_vertices={EXPECTED_TERRAIN_VERTICES}")
    print(f"active_height_profile={ACTIVE_HEIGHT_PROFILE}")
    print(f"active_height_range_m={ACTIVE_HEIGHT_MIN_M:.6f}..{ACTIVE_HEIGHT_MAX_M:.6f}")
    print(f"buildings_dae_sha256={sha256(BUILDINGS)}")
    print(f"heightmap_sha256={sha256(HEIGHTMAP)}")
    print(f"normalmap_sha256={sha256(NORMALMAP)}")
    print(f"collision_sha256={sha256(COLLISION)}")


if __name__ == "__main__":
    main()
