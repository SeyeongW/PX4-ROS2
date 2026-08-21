#!/usr/bin/env python3
"""Deterministically scale the current Apple Park building heights.

The committed city buildings have already been seated into the sloped terrain.
This post-process therefore changes only each prism's roof vectors.  The
foundation vectors, XY footprint, triangle topology and normals are preserved
exactly.  A stable crop component identifier drives the pseudo-random factor,
so applying the same seed does not depend on Python's random implementation or
on mutable vertex coordinates.
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

import align_city_building_foundations as foundation


GAZEBO = Path(__file__).resolve().parents[1]
CITY = GAZEBO / "worlds" / "applepark_city"
BUILDINGS = CITY / "mesh" / "buildings.dae"
OVERLAP_CSV = GAZEBO / "validation" / "city" / "road_building_overlap_coordinates.csv"
FOUNDATION_CSV = GAZEBO / "validation" / "city" / "building_foundation_alignment.csv"
HEIGHT_AUDIT_CSV = GAZEBO / "validation" / "city" / "building_height_scaling.csv"

INPUT_DAE_SHA256 = "8e800314c1fce8009d7f862aa87ad68bbd847e80f80582901bd504c3ac703e7b"
OUTPUT_DAE_SHA256 = "e5fab82529fcc0f9d5819797346af76d763d6b973eed96ad8a7de324a7a253ce"
OVERLAP_CSV_SHA256 = "b3a1f0eaee7eafc2595c9bd35e6f23a4a74749b65a0a0224dc78ad6d4ddf6601"
INPUT_FOUNDATION_CSV_SHA256 = (
    "251cc93369133f1e31a353b4587fc00025b39e94fad843a1cb8d1148dd8368e3"
)
OUTPUT_FOUNDATION_CSV_SHA256 = (
    "db0247ad555cd41fd3d902fa827bdc98125d06635c18598e4eb613bcb631e12f"
)
HEIGHT_AUDIT_CSV_SHA256 = (
    "bc7b14dac3c2cc5244804f37b3eeed8f629dad7e5f4ac955c504222bd15b4543"
)
FLAT_CITY_DAE_SHA256 = (
    "3141bb5aafd931b2d93ce825b74e13645186570f0a84e7bdef6046fab688e43c"
)

FACTOR_SEED = "px4-ros2-city-height-v2"
MIN_FACTOR_MILLIONTHS = 2_000_000
MAX_FACTOR_MILLIONTHS = 3_500_000
FACTOR_BUCKETS = MAX_FACTOR_MILLIONTHS - MIN_FACTOR_MILLIONTHS + 1
EXPECTED_COMPONENTS = 274
EXPECTED_TRIANGLES = 9248
EXPECTED_VECTORS = 27744
EXPECTED_ROOF_VECTORS = 13872

HEIGHT_FIELDS = [
    "output_component_ordinal",
    "post_height_component_ordinal",
    "face_start",
    "face_end",
    "face_count",
    "local_min_x",
    "local_min_y",
    "local_max_x",
    "local_max_y",
    "reference_base_z",
    "foundation_base_z",
    "old_roof_z",
    "old_above_ground_height_m",
    "factor_sha256",
    "factor",
    "new_roof_z",
    "new_above_ground_height_m",
    "roof_vector_count",
]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def deterministic_factor(component_id: int) -> tuple[float, str]:
    digest = hashlib.sha256(
        f"{FACTOR_SEED}:{component_id}".encode("ascii")
    ).digest()
    millionths = MIN_FACTOR_MILLIONTHS + (
        int.from_bytes(digest[:8], "big") % FACTOR_BUCKETS
    )
    return millionths / 1_000_000.0, digest.hex()


def csv_output(records: list[dict[str, object]], fields: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def atomic_write(path: Path, content: str) -> None:
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


def analyze(
    data: foundation.DaeData,
    overlap_rows: list[dict[str, str]],
    foundation_rows: list[dict[str, str]],
    *,
    phase: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    components = foundation.find_components(data)
    require(len(components) == EXPECTED_COMPONENTS, "building component count changed")
    require(len(data.position_indices) == EXPECTED_TRIANGLES, "triangle count changed")
    require(len(data.positions) == EXPECTED_VECTORS, "position vector count changed")
    require(len(overlap_rows) == EXPECTED_COMPONENTS, "overlap CSV row count changed")
    require(len(foundation_rows) == EXPECTED_COMPONENTS, "foundation CSV row count changed")

    touched = np.zeros(len(data.positions), dtype=bool)
    height_records: list[dict[str, object]] = []
    updated_foundations: list[dict[str, object]] = []
    roof_vector_count = 0
    component_ids: set[int] = set()

    for ordinal, (faces, source, seated) in enumerate(
        zip(components, overlap_rows, foundation_rows), start=1
    ):
        require(
            int(source["output_component_ordinal"]) == ordinal,
            f"overlap CSV component order changed at {ordinal}",
        )
        require(
            int(seated["output_component_ordinal"]) == ordinal,
            f"foundation CSV component order changed at {ordinal}",
        )
        component_id = int(source["post_height_component_ordinal"])
        require(
            int(seated["post_height_component_ordinal"]) == component_id,
            f"component identity differs at {ordinal}",
        )
        require(component_id not in component_ids, f"duplicate component ID {component_id}")
        component_ids.add(component_id)

        position_ids = data.position_indices[faces].reshape(-1)
        require(
            len(np.unique(position_ids)) == len(position_ids),
            f"component {ordinal} no longer has face-expanded vectors",
        )
        require(
            not touched[position_ids].any(),
            f"component {ordinal} shares position vectors",
        )
        touched[position_ids] = True
        points = data.positions[position_ids]
        low, high = points.min(axis=0), points.max(axis=0)
        expected_xy = np.asarray(
            [
                float(source["local_min_x"]),
                float(source["local_min_y"]),
                float(source["local_max_x"]),
                float(source["local_max_y"]),
            ]
        )
        actual_xy = np.asarray([low[0], low[1], high[0], high[1]])
        require(
            np.max(np.abs(actual_xy - expected_xy)) < 1e-9,
            f"component {ordinal} XY footprint changed",
        )

        reference_base = float(source["base_z"])
        foundation_base = float(seated["foundation_base_z"])
        old_roof = float(source["top_z"])
        require(
            abs(float(seated["original_base_z"]) - reference_base) < 1e-12,
            f"component {ordinal} reference base differs",
        )
        require(
            abs(low[2] - foundation_base) < 1e-9,
            f"component {ordinal} foundation Z changed",
        )
        require(
            foundation_base < reference_base < old_roof,
            f"component {ordinal} invalid base / roof order",
        )

        factor, factor_digest = deterministic_factor(component_id)
        old_height = old_roof - reference_base
        new_height = factor * old_height
        new_roof = reference_base + new_height
        expected_roof = old_roof if phase == "input" else new_roof
        require(
            abs(high[2] - expected_roof) < 1e-6,
            f"component {ordinal} roof is {high[2]:.9f}, expected {expected_roof:.9f}",
        )

        bottom_mask = np.isclose(points[:, 2], foundation_base, atol=1e-9)
        roof_mask = np.isclose(points[:, 2], expected_roof, atol=1e-6)
        require(
            np.all(bottom_mask | roof_mask),
            f"component {ordinal} has an unexpected intermediate Z level",
        )
        require(
            int(bottom_mask.sum()) == int(roof_mask.sum()),
            f"component {ordinal} is not a two-level prism",
        )
        roof_vector_count += int(roof_mask.sum())
        if phase == "input":
            data.positions[position_ids[roof_mask], 2] = new_roof

        height_records.append(
            {
                "output_component_ordinal": ordinal,
                "post_height_component_ordinal": component_id,
                "face_start": int(faces[0]),
                "face_end": int(faces[-1]),
                "face_count": len(faces),
                "local_min_x": source["local_min_x"],
                "local_min_y": source["local_min_y"],
                "local_max_x": source["local_max_x"],
                "local_max_y": source["local_max_y"],
                "reference_base_z": foundation.format_float(reference_base),
                "foundation_base_z": seated["foundation_base_z"],
                "old_roof_z": foundation.format_float(old_roof),
                "old_above_ground_height_m": foundation.format_float(old_height),
                "factor_sha256": factor_digest,
                "factor": f"{factor:.6f}",
                "new_roof_z": foundation.format_float(new_roof),
                "new_above_ground_height_m": foundation.format_float(new_height),
                "roof_vector_count": int(roof_mask.sum()),
            }
        )
        updated = dict(seated)
        updated["roof_z"] = foundation.format_float(new_roof)
        updated_foundations.append(updated)

    require(touched.all(), "not all DAE position vectors belong to a building")
    require(len(component_ids) == EXPECTED_COMPONENTS, "component IDs are not unique")
    require(roof_vector_count == EXPECTED_ROOF_VECTORS, "roof vector count changed")
    return height_records, updated_foundations, roof_vector_count


def write_dae(data: foundation.DaeData) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=BUILDINGS.parent, prefix=".buildings.height.", suffix=".dae"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        data.write(temporary)
        require(
            sha256(temporary) == OUTPUT_DAE_SHA256,
            "generated building DAE hash differs from the locked output",
        )
        temporary.chmod(BUILDINGS.stat().st_mode & 0o777)
        os.replace(temporary, BUILDINGS)
    finally:
        temporary.unlink(missing_ok=True)


def validate_static_inputs() -> None:
    require(sha256(OVERLAP_CSV) == OVERLAP_CSV_SHA256, "overlap CSV hash changed")
    require(
        sha256(FOUNDATION_CSV)
        in {INPUT_FOUNDATION_CSV_SHA256, OUTPUT_FOUNDATION_CSV_SHA256},
        "foundation CSV hash is neither the input nor final audit",
    )
    require(
        sha256(BUILDINGS) in {INPUT_DAE_SHA256, OUTPUT_DAE_SHA256},
        "building DAE is neither the locked input nor final output",
    )


def report(records: list[dict[str, object]], roof_vectors: int) -> None:
    factors = np.asarray([float(row["factor"]) for row in records])
    old_heights = np.asarray(
        [float(row["old_above_ground_height_m"]) for row in records]
    )
    new_heights = np.asarray(
        [float(row["new_above_ground_height_m"]) for row in records]
    )
    new_roofs = np.asarray([float(row["new_roof_z"]) for row in records])
    foundations = np.asarray([float(row["foundation_base_z"]) for row in records])
    print("PX4-ROS2 deterministic city building height scaling")
    print("result=PASS")
    print(f"seed={FACTOR_SEED}")
    print(f"components={len(records)} triangles={EXPECTED_TRIANGLES} vectors={EXPECTED_VECTORS}")
    print(f"roof_vectors_scaled={roof_vectors} foundation_vectors_preserved={roof_vectors}")
    print(
        "factor="
        f"min:{factors.min():.6f} mean:{factors.mean():.6f} "
        f"median:{np.median(factors):.6f} max:{factors.max():.6f}"
    )
    print(f"old_above_ground_height_m={old_heights.min():.6f}..{old_heights.max():.6f}")
    print(f"new_above_ground_height_m={new_heights.min():.6f}..{new_heights.max():.6f}")
    print(f"new_roof_z_m={new_roofs.min():.6f}..{new_roofs.max():.6f}")
    print(f"foundation_z_m={foundations.min():.6f}..{foundations.max():.6f}")
    print(f"buildings_dae_sha256={sha256(BUILDINGS)}")
    print(f"height_audit_sha256={sha256(HEIGHT_AUDIT_CSV)}")
    print(f"foundation_audit_sha256={sha256(FOUNDATION_CSV)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()
    if sha256(BUILDINGS) == FLAT_CITY_DAE_SHA256:
        # Height factors are already preserved by the final flat-datum pass.
        # Keep the former public checker working after that pipeline change.
        from flatten_city_assets import main as flat_city_main

        print("NOTICE: city datum is flat; delegating to flatten_city_assets.py")
        flat_city_main()
        return
    validate_static_inputs()
    current_hash = sha256(BUILDINGS)
    if args.check:
        require(current_hash == OUTPUT_DAE_SHA256, "height-scaled building DAE is missing")
        require(HEIGHT_AUDIT_CSV.is_file(), "height audit CSV is missing")
        require(
            sha256(HEIGHT_AUDIT_CSV) == HEIGHT_AUDIT_CSV_SHA256,
            "height audit CSV hash changed",
        )
        require(
            sha256(FOUNDATION_CSV) == OUTPUT_FOUNDATION_CSV_SHA256,
            "final foundation audit CSV hash changed",
        )

    overlap_rows = read_csv(OVERLAP_CSV)
    foundation_rows = read_csv(FOUNDATION_CSV)
    data = foundation.DaeData(BUILDINGS)
    phase = "input" if current_hash == INPUT_DAE_SHA256 else "final"
    records, updated_foundations, roof_vectors = analyze(
        data, overlap_rows, foundation_rows, phase=phase
    )
    height_content = csv_output(records, HEIGHT_FIELDS)
    foundation_fields = list(foundation_rows[0])
    foundation_content = csv_output(updated_foundations, foundation_fields)

    require(
        hashlib.sha256(height_content.encode("utf-8")).hexdigest()
        == HEIGHT_AUDIT_CSV_SHA256,
        "generated height audit differs from the locked output",
    )
    require(
        hashlib.sha256(foundation_content.encode("utf-8")).hexdigest()
        == OUTPUT_FOUNDATION_CSV_SHA256,
        "generated foundation audit differs from the locked output",
    )

    if args.check:
        require(
            HEIGHT_AUDIT_CSV.read_text(encoding="utf-8") == height_content,
            "height audit CSV differs from final geometry",
        )
        require(
            FOUNDATION_CSV.read_text(encoding="utf-8") == foundation_content,
            "foundation audit CSV differs from final geometry",
        )
    else:
        if phase == "input":
            write_dae(data)
        atomic_write(HEIGHT_AUDIT_CSV, height_content)
        atomic_write(FOUNDATION_CSV, foundation_content)

    checked_records, checked_foundations, checked_roof_vectors = analyze(
        foundation.DaeData(BUILDINGS), overlap_rows, read_csv(FOUNDATION_CSV), phase="final"
    )
    require(
        csv_output(checked_records, HEIGHT_FIELDS) == height_content,
        "post-write height geometry drift",
    )
    require(
        csv_output(checked_foundations, foundation_fields) == foundation_content,
        "post-write foundation geometry drift",
    )
    require(checked_roof_vectors == roof_vectors, "post-write roof vector drift")
    report(checked_records, checked_roof_vectors)


if __name__ == "__main__":
    main()
