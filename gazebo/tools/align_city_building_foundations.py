#!/usr/bin/env python3
"""Extend Apple Park building foundations below both city terrain surfaces.

The upstream terrain generator sampled one elevation at each footprint
centroid, then used that value for the building's complete flat base.  On a
slope this leaves the downhill wall floating and the uphill side buried.
This deterministic post-process lowers only each component's bottom vertices;
XY coordinates, roofs, normals, topology, roads and terrain assets stay fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


GAZEBO = Path(__file__).resolve().parents[1]
CITY = GAZEBO / "worlds" / "applepark_city"
BUILDINGS = CITY / "mesh" / "buildings.dae"
HEIGHTMAP = CITY / "mesh" / "height_map_city_500m.png"
COLLISION = CITY / "mesh" / "city_terrain_collision.obj"
BUILDING_CSV = GAZEBO / "validation" / "city" / "road_building_overlap_coordinates.csv"
AUDIT_CSV = GAZEBO / "validation" / "city" / "building_foundation_alignment.csv"
HEIGHT_AUDIT_CSV = GAZEBO / "validation" / "city" / "building_height_scaling.csv"

EXPECTED_ORIGINAL_DAE_SHA256 = (
    "9850924d0432da9df7985512dc649491c3ca1381f8a48c56673c606792a1d6c1"
)
EXPECTED_PRE_HEIGHT_ALIGNED_DAE_SHA256 = (
    "8e800314c1fce8009d7f862aa87ad68bbd847e80f80582901bd504c3ac703e7b"
)
EXPECTED_ALIGNED_DAE_SHA256 = (
    "e5fab82529fcc0f9d5819797346af76d763d6b973eed96ad8a7de324a7a253ce"
)
EXPECTED_HEIGHTMAP_SHA256 = (
    "5a84adc1f45dcffe507fa77d2642cd672c622c225e60ae41f98280bdcf9b24cf"
)
EXPECTED_COLLISION_SHA256 = (
    "99d8fae1ed193321dc3f2801b68a549b019a12068cf0cdff17210c78e0fe2024"
)
EXPECTED_BUILDING_CSV_SHA256 = (
    "b3a1f0eaee7eafc2595c9bd35e6f23a4a74749b65a0a0224dc78ad6d4ddf6601"
)
EXPECTED_HEIGHT_AUDIT_CSV_SHA256 = (
    "bc7b14dac3c2cc5244804f37b3eeed8f629dad7e5f4ac955c504222bd15b4543"
)
EXPECTED_FOUNDATION_AUDIT_CSV_SHA256 = (
    "db0247ad555cd41fd3d902fa827bdc98125d06635c18598e4eb613bcb631e12f"
)
FLAT_CITY_DAE_SHA256 = (
    "199c0a3dbe471d319b01582670c327d03e1668cdf007791b10929a435f6c7449"
)

NS = "http://www.collada.org/2005/11/COLLADASchema"
Q = lambda tag: f"{{{NS}}}{tag}"
ET.register_namespace("", NS)

MAP_SIZE_M = 500.0
POSITION_Z_M = -15.0
HEIGHT_SCALE_M = 26.6
FOUNDATION_MARGIN_M = 0.05
QUANTIZE_DECIMALS = 6
EXPECTED_COMPONENTS = 274
EXPECTED_TRIANGLES = 9248
EXPECTED_VECTORS = 27744
EXPECTED_BOTTOM_VECTORS = 13872


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


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


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


class DaeData:
    def __init__(self, path: Path):
        self.path = path
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        require(root.tag == Q("COLLADA"), "unexpected COLLADA namespace")
        mesh = root.find(f".//{Q('library_geometries')}/{Q('geometry')}/{Q('mesh')}")
        require(mesh is not None, "building mesh is missing")
        sources = {node.attrib.get("id"): node for node in mesh.findall(Q("source"))}
        require(set(sources) == {"verts-array", "normals-array"}, "building sources changed")
        self.position_array = sources["verts-array"].find(Q("float_array"))
        self.position_accessor = sources["verts-array"].find(
            f"{Q('technique_common')}/{Q('accessor')}"
        )
        require(self.position_array is not None, "position float array is missing")
        require(self.position_accessor is not None, "position accessor is missing")
        values = np.fromstring(self.position_array.text or "", sep=" ", dtype=np.float64)
        require(values.size % 3 == 0, "position float count is invalid")
        self.positions = values.reshape(-1, 3)

        triangles = mesh.findall(Q("triangles"))
        require(len(triangles) == 1, "expected one triangle set")
        self.triangles = triangles[0]
        inputs = sorted(
            self.triangles.findall(Q("input")), key=lambda item: int(item.attrib["offset"])
        )
        require(
            [(item.attrib.get("semantic"), item.attrib.get("offset")) for item in inputs]
            == [("VERTEX", "0"), ("NORMAL", "1")],
            "triangle input layout changed",
        )
        indices = np.fromstring(
            self.triangles.findtext(Q("p"), ""), sep=" ", dtype=np.int64
        )
        triangle_count = int(self.triangles.attrib["count"])
        require(indices.size == triangle_count * 6, "triangle index count changed")
        self.position_indices = indices.reshape(triangle_count, 3, 2)[:, :, 0]

    def write(self, path: Path) -> None:
        self.position_array.text = " ".join(
            format_float(value) for value in self.positions.reshape(-1)
        )
        self.tree.write(
            path, encoding="utf-8", xml_declaration=False, short_empty_elements=True
        )


def find_components(data: DaeData) -> list[np.ndarray]:
    """Group face-connected prisms using their quantized geometric edges."""

    face_points = np.round(data.positions[data.position_indices], QUANTIZE_DECIMALS)
    count = len(face_points)
    parent = np.arange(count, dtype=np.int64)
    size = np.ones(count, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    edge_owner: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
    for face_id, triangle in enumerate(face_points):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            a, b = tuple(first), tuple(second)
            edge = (a, b) if a <= b else (b, a)
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = face_id
            else:
                union(face_id, previous)

    groups: dict[int, list[int]] = {}
    for face_id in range(count):
        groups.setdefault(find(face_id), []).append(face_id)
    components = [np.asarray(faces, dtype=np.int64) for faces in groups.values()]
    components.sort(key=lambda faces: int(faces[0]))
    return components


def terrain_aabb_minimum(
    heights: np.ndarray, minimum_xy: np.ndarray, maximum_xy: np.ndarray
) -> float:
    """Conservative continuous-surface lower bound over a building AABB.

    Both bilinear heightmap interpolation and the collision OBJ triangles are
    convex combinations of their cell corners.  The minimum of every grid
    vertex touched by the AABB is therefore no higher than either surface at
    any point of the actual footprint inside that box.
    """

    cells = heights.shape[0] - 1
    columns = (np.asarray([minimum_xy[0], maximum_xy[0]]) + MAP_SIZE_M / 2.0) / MAP_SIZE_M * cells
    rows = (MAP_SIZE_M / 2.0 - np.asarray([maximum_xy[1], minimum_xy[1]])) / MAP_SIZE_M * cells
    c0 = max(0, int(np.floor(columns.min())))
    c1 = min(cells, int(np.ceil(columns.max())))
    r0 = max(0, int(np.floor(rows.min())))
    r1 = min(cells, int(np.ceil(rows.max())))
    require(c1 >= c0 and r1 >= r0, "building AABB missed the terrain grid")
    return float(heights[r0 : r1 + 1, c0 : c1 + 1].min())


def csv_output(records: list[dict[str, object]]) -> str:
    fields = [
        "output_component_ordinal",
        "post_height_component_ordinal",
        "original_base_z",
        "foundation_base_z",
        "visual_terrain_min_z",
        "collision_terrain_min_z",
        "foundation_extension_m",
        "roof_z",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def analyze(
    data: DaeData, *, modify: bool, scaled: bool
) -> tuple[list[dict[str, object]], int]:
    components = find_components(data)
    require(len(components) == EXPECTED_COMPONENTS, "building component count changed")
    with BUILDING_CSV.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    require(len(source_rows) == len(components), "building CSV/component count differs")
    height_rows: list[dict[str, str]] = []
    if scaled:
        require(HEIGHT_AUDIT_CSV.is_file(), "building height audit CSV is missing")
        with HEIGHT_AUDIT_CSV.open("r", encoding="utf-8", newline="") as stream:
            height_rows = list(csv.DictReader(stream))
        require(len(height_rows) == len(components), "height CSV/component count differs")

    pixels = np.asarray(Image.open(HEIGHTMAP).convert("L"), dtype=np.float64)
    require(pixels.shape == (257, 257), "city heightmap shape changed")
    visual_heights = POSITION_Z_M + pixels / 255.0 * HEIGHT_SCALE_M
    collision_heights = visual_heights[::2, ::2]
    records: list[dict[str, object]] = []
    bottom_vector_count = 0

    for ordinal, (faces, row) in enumerate(zip(components, source_rows), start=1):
        position_ids = data.position_indices[faces].reshape(-1)
        points = data.positions[position_ids]
        low, high = points.min(axis=0), points.max(axis=0)
        original_base = float(row["base_z"])
        expected_xy = np.array(
            [
                float(row["local_min_x"]),
                float(row["local_min_y"]),
                float(row["local_max_x"]),
                float(row["local_max_y"]),
            ]
        )
        actual_xy = np.array([low[0], low[1], high[0], high[1]])
        require(np.max(np.abs(actual_xy - expected_xy)) < 1e-6, f"building {ordinal} XY changed")
        expected_roof = float(row["top_z"])
        if scaled:
            height_row = height_rows[ordinal - 1]
            require(
                int(height_row["output_component_ordinal"]) == ordinal,
                f"building {ordinal} height audit order changed",
            )
            require(
                height_row["post_height_component_ordinal"]
                == row["post_height_component_ordinal"],
                f"building {ordinal} height audit identity changed",
            )
            require(
                abs(float(height_row["old_roof_z"]) - expected_roof) < 1e-9,
                f"building {ordinal} pre-scale roof reference changed",
            )
            require(
                abs(float(height_row["reference_base_z"]) - original_base) < 1e-9,
                f"building {ordinal} height reference base changed",
            )
            expected_roof = float(height_row["new_roof_z"])
        require(
            abs(high[2] - expected_roof) < 1e-6,
            f"building {ordinal} roof is {high[2]:.9f}, expected {expected_roof:.9f}",
        )

        current_base = float(low[2])
        triangles = data.positions[data.position_indices[faces]]
        bottom_mask = np.all(
            np.isclose(triangles[:, :, 2], current_base, atol=1e-6), axis=1
        )
        bottom_triangles = triangles[bottom_mask, :, :2]
        require(len(bottom_triangles) > 0, f"building {ordinal} bottom faces missing")
        visual_min = terrain_aabb_minimum(visual_heights, low[:2], high[:2])
        collision_min = terrain_aabb_minimum(collision_heights, low[:2], high[:2])
        target = min(visual_min, collision_min) - FOUNDATION_MARGIN_M
        target = float(format_float(target))

        bottom_vertices = np.isclose(points[:, 2], current_base, atol=1e-6)
        require(int(bottom_vertices.sum()) * 2 == len(points), f"building {ordinal} is not a prism")
        bottom_vector_count += int(bottom_vertices.sum())
        if modify:
            data.positions[position_ids[bottom_vertices], 2] = target
        else:
            require(
                abs(current_base - target) < 1e-9,
                f"building {ordinal} foundation is {current_base:.9f}, expected {target:.9f}",
            )
        require(target <= visual_min - 0.049, f"building {ordinal} visual clearance")
        require(target <= collision_min - 0.049, f"building {ordinal} collision clearance")
        records.append(
            {
                "output_component_ordinal": ordinal,
                "post_height_component_ordinal": row["post_height_component_ordinal"],
                "original_base_z": format_float(original_base),
                "foundation_base_z": format_float(target),
                "visual_terrain_min_z": format_float(visual_min),
                "collision_terrain_min_z": format_float(collision_min),
                "foundation_extension_m": format_float(original_base - target),
                "roof_z": format_float(high[2]),
            }
        )
    require(bottom_vector_count == EXPECTED_BOTTOM_VECTORS, "bottom vector count changed")
    return records, bottom_vector_count


def write_aligned(data: DaeData, records: list[dict[str, object]]) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=BUILDINGS.parent, prefix=".buildings.foundation.", suffix=".dae"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        data.write(temporary)
        require(ET.parse(temporary).getroot().tag == Q("COLLADA"), "written DAE is invalid")
        temporary.chmod(BUILDINGS.stat().st_mode & 0o777)
        os.replace(temporary, BUILDINGS)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write(AUDIT_CSV, csv_output(records))


def validate_inputs(*, check: bool) -> None:
    require(sha256(HEIGHTMAP) == EXPECTED_HEIGHTMAP_SHA256, "heightmap hash changed")
    require(sha256(COLLISION) == EXPECTED_COLLISION_SHA256, "collision OBJ hash changed")
    require(sha256(BUILDING_CSV) == EXPECTED_BUILDING_CSV_SHA256, "building CSV hash changed")
    current = sha256(BUILDINGS)
    if check:
        require(bool(EXPECTED_ALIGNED_DAE_SHA256), "aligned DAE hash is not configured")
        require(current == EXPECTED_ALIGNED_DAE_SHA256, "aligned building DAE hash changed")
        require(HEIGHT_AUDIT_CSV.is_file(), "building height audit CSV is missing")
        require(AUDIT_CSV.is_file(), "building foundation audit CSV is missing")
        require(
            sha256(HEIGHT_AUDIT_CSV) == EXPECTED_HEIGHT_AUDIT_CSV_SHA256,
            "building height audit CSV changed",
        )
        require(
            sha256(AUDIT_CSV) == EXPECTED_FOUNDATION_AUDIT_CSV_SHA256,
            "building foundation audit CSV changed",
        )
    else:
        allowed = {
            EXPECTED_ORIGINAL_DAE_SHA256,
            EXPECTED_PRE_HEIGHT_ALIGNED_DAE_SHA256,
        }
        if EXPECTED_ALIGNED_DAE_SHA256:
            allowed.add(EXPECTED_ALIGNED_DAE_SHA256)
        require(current in allowed, "building DAE is neither the source nor aligned asset")
        if current == EXPECTED_ALIGNED_DAE_SHA256:
            require(HEIGHT_AUDIT_CSV.is_file(), "building height audit CSV is missing")
            require(
                sha256(HEIGHT_AUDIT_CSV) == EXPECTED_HEIGHT_AUDIT_CSV_SHA256,
                "building height audit CSV changed",
            )


def report(records: list[dict[str, object]], bottom_vectors: int, result: str) -> None:
    extensions = [float(row["foundation_extension_m"]) for row in records]
    foundations = [float(row["foundation_base_z"]) for row in records]
    print("PX4-ROS2 city building foundation alignment")
    print(f"result={result}")
    print(f"components={len(records)} triangles={EXPECTED_TRIANGLES} vectors={EXPECTED_VECTORS}")
    print(f"bottom_vectors_adjusted={bottom_vectors}")
    print(f"foundation_margin_m={FOUNDATION_MARGIN_M:.2f}")
    print(
        "extension_m="
        f"min:{min(extensions):.6f} mean:{np.mean(extensions):.6f} "
        f"median:{np.median(extensions):.6f} max:{max(extensions):.6f}"
    )
    print(f"foundation_global_min_z={min(foundations):.9f}")
    print(f"buildings_dae_sha256={sha256(BUILDINGS)}")
    print(f"audit_csv_sha256={sha256(AUDIT_CSV)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()
    if sha256(BUILDINGS) == FLAT_CITY_DAE_SHA256:
        # The final trailer-safe map supersedes the old sloped-foundation
        # stage. Preserve this historical entry point as a compatible alias.
        from flatten_city_assets import main as flat_city_main

        print("NOTICE: city datum is flat; delegating to flatten_city_assets.py")
        flat_city_main()
        return
    validate_inputs(check=args.check)
    data = DaeData(BUILDINGS)
    require(len(data.position_indices) == EXPECTED_TRIANGLES, "triangle count changed")
    require(len(data.positions) == EXPECTED_VECTORS, "position vector count changed")

    current = sha256(BUILDINGS)
    scaled = current == EXPECTED_ALIGNED_DAE_SHA256
    if args.check:
        records, bottom_vectors = analyze(data, modify=False, scaled=True)
        require(AUDIT_CSV.is_file(), "foundation audit CSV is missing")
        require(
            AUDIT_CSV.read_text(encoding="utf-8") == csv_output(records),
            "foundation audit CSV differs from geometry",
        )
        report(records, bottom_vectors, "PASS")
        return

    records, _ = analyze(
        data,
        modify=current == EXPECTED_ORIGINAL_DAE_SHA256,
        scaled=scaled,
    )
    if current == EXPECTED_ORIGINAL_DAE_SHA256:
        write_aligned(data, records)
    else:
        require(AUDIT_CSV.read_text(encoding="utf-8") == csv_output(records), "audit CSV changed")
    checked_current = sha256(BUILDINGS)
    checked_records, bottom_vectors = analyze(
        DaeData(BUILDINGS),
        modify=False,
        scaled=checked_current == EXPECTED_ALIGNED_DAE_SHA256,
    )
    require(csv_output(checked_records) == csv_output(records), "post-write foundation drift")
    report(checked_records, bottom_vectors, "PASS")


if __name__ == "__main__":
    main()
