#!/usr/bin/env python3
"""Baking a map pack from a coordinate contract.

The point of these tests is **swappability**: a contract the baker has never seen
must produce a loadable pack with correct georeferencing, without touching the
city map's numbers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from drone_gcs.map_pack import MapPack

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BAKER = PACKAGE_ROOT / "tools" / "bake_map.py"


def synthetic_contract(tmp_path: Path, **overrides) -> Path:
    """A minimal version-2 contract: two buildings on a 200 x 100 m field."""
    doc = {
        "schema_version": 2,
        "map": {
            "name": "unittest_field",
            "gazebo_world_name": "unittest_field",
            "bounds_enu_m": {"x": [-100.0, 100.0], "y": [-50.0, 50.0]},
        },
        "frames": {"px4_local": {"origin_enu_m": [10.0, -20.0, 0.5]}},
        "spawn": {"gazebo_spawn_pose_enu": {"x": 10.0, "y": -20.0, "z": 0.0}},
        "trailer": {
            "entity_name": "test_trailer",
            "body_footprint_m": [4.0, 2.0],
        },
        "obstacles": {
            "buildings": [
                {
                    "id": "b1",
                    "footprint": {"outer": [[0, 0], [20, 0], [20, 10], [0, 10]], "holes": []},
                    "foundation_z_m": -0.05,
                    "roof_z_m": 30.0,
                },
                {
                    "id": "b2",
                    "footprint": {
                        "outer": [[-60, -30], [-40, -30], [-40, -10], [-60, -10]],
                        "holes": [[[-55, -25], [-45, -25], [-45, -15], [-55, -15]]],
                    },
                    "foundation_z_m": -0.05,
                    "roof_z_m": 10.0,
                },
            ]
        },
        "derivation": {
            "fixed_mission_coordinates_enu_m": {
                "drone_spawn": [10.0, -20.0],
                "global_goal": [-80.0, 40.0],
            }
        },
    }
    doc.update(overrides)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def run_baker(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(BAKER), *args],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise AssertionError(f"baker failed:\n{result.stdout}\n{result.stderr}")
    return result


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """Bake the synthetic contract once; several tests read the result."""
    tmp_path = tmp_path_factory.mktemp("bake")
    contract = synthetic_contract(tmp_path)
    out = tmp_path / "pack"
    result = run_baker(
        "--world-yaml", str(contract),
        "--out", str(out),
        "--no-ground-texture",
        "--basemap-px", "400",
        "--cruise-z", "15.0",
        "--occupancy-res-m", "1.0",
    )
    return out, result.stdout


def test_synthetic_contract_yields_a_loadable_pack(baked):
    out, _ = baked
    pack = MapPack.load(out)

    assert pack.name == "unittest_field"
    assert (pack.bounds.x_min, pack.bounds.x_max) == (-100.0, 100.0)
    assert (pack.bounds.y_min, pack.bounds.y_max) == (-50.0, 50.0)
    assert len(pack.buildings) == 2
    assert pack.px4_local_origin_enu_m == (10.0, -20.0, 0.5)
    assert pack.spawn_enu_m == (10.0, -20.0)


def test_synthesised_basemap_keeps_the_world_aspect(baked):
    """A 200 x 100 m field must not be baked into a square image."""
    out, _ = baked
    pack = MapPack.load(out)

    assert (pack.basemap.width_px, pack.basemap.height_px) == (400, 200)
    with Image.open(pack.basemap.path) as image:
        assert image.size == (400, 200)
    # Square pixels: both axes at the same metres-per-pixel.
    assert pack.basemap.m_per_px_x == pytest.approx(pack.basemap.m_per_px_y)
    assert pack.basemap.m_per_px_x == pytest.approx(0.5)


def test_holes_survive_the_bake(baked):
    out, _ = baked
    pack = MapPack.load(out)
    holed = [b for b in pack.buildings if b.holes]

    assert [b.id for b in holed] == ["b2"]
    assert len(holed[0].holes[0]) == 4


def test_markers_come_from_the_contract(baked):
    out, _ = baked
    markers = {m.name: m.enu_m for m in MapPack.load(out).markers}

    assert markers == {"drone_spawn": (10.0, -20.0), "global_goal": (-80.0, 40.0)}


def test_trailer_entity_comes_from_the_contract(baked):
    out, _ = baked
    trailer = MapPack.load(out).entity("test_trailer")

    assert trailer is not None
    assert trailer.footprint_m == (4.0, 2.0)
    # No sibling pursuit scenario for this map, so no route to draw.
    assert trailer.route_yaml is None


def test_occupancy_reflects_the_planner_model_not_raw_roofs(baked):
    """At z=15 m both buildings block: b1's roof is 30 m, and b2 is inflated.

    b2's roof is only 10 m, but the planner's default 10 m roof clearance lifts
    its obstacle top to ~20 m, so the cruise plane at 15 m still cuts it.  This
    is exactly the discrepancy the pack must not hide.
    """
    out, stdout = baked
    pack = MapPack.load(out)

    assert pack.occupancy is not None
    assert pack.occupancy_z_m == pytest.approx(15.0)
    assert "2 of 2 obstacles cut z=15.0 m" in stdout

    grid = Image.open(pack.occupancy.path).convert("L")
    assert grid.size == (200, 100)

    def blocked(x, y):
        col, row = pack.occupancy.enu_to_px(x, y)
        return grid.getpixel((int(col), int(row))) == 0

    assert blocked(10.0, 5.0)  # inside b1
    assert blocked(-50.0, -20.0)  # inside b2 (courtyard is inside its AABB)
    assert not blocked(90.0, 40.0)  # open field
    # Wall inflation pushes the obstacle past the raw footprint edge.
    assert blocked(20.5, 5.0)
    assert not blocked(60.0, 5.0)


def test_baker_refuses_a_non_version_2_contract(tmp_path):
    contract = synthetic_contract(tmp_path, schema_version=1)
    result = subprocess.run(
        [sys.executable, str(BAKER), "--world-yaml", str(contract), "--out", str(tmp_path / "p")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "schema_version" in result.stderr


def test_baker_rejects_a_missing_ground_texture(tmp_path):
    contract = synthetic_contract(tmp_path)
    result = subprocess.run(
        [
            sys.executable, str(BAKER),
            "--world-yaml", str(contract),
            "--out", str(tmp_path / "p"),
            "--ground-texture", str(tmp_path / "nope.png"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr
