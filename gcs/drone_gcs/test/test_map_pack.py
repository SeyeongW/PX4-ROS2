#!/usr/bin/env python3
"""Map pack loading, georeferencing, and frame conversion.

Run from the package root:

    PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from drone_gcs.map_pack import Bounds, MapPack, MapPackError, RasterLayer

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CITY_PACK = PACKAGE_ROOT / "maps" / "city_uav"

# One-pixel PNG, so raster tests need no image generation.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


def write_pack(root: Path, descriptor: dict, *, basemap=True, buildings=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if basemap:
        (root / "basemap.png").write_bytes(_PNG_1X1)
    if buildings is not None:
        (root / "buildings.json").write_text(json.dumps(buildings), encoding="utf-8")
    (root / "map.yaml").write_text(yaml.safe_dump(descriptor, allow_unicode=True), encoding="utf-8")
    return root


def minimal_descriptor(**overrides) -> dict:
    doc = {
        "name": "unit",
        "bounds_enu_m": {"x": [-100.0, 100.0], "y": [-50.0, 50.0]},
        "basemap": {"file": "basemap.png", "size_px": [200, 100]},
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------- bounds
def test_bounds_geometry():
    bounds = Bounds.from_dict({"x": [-650.0, 650.0], "y": [-650.0, 650.0]})
    assert bounds.width_m == 1300.0
    assert bounds.height_m == 1300.0
    assert bounds.center_m == (0.0, 0.0)
    assert bounds.contains(0.0, 0.0)
    assert bounds.contains(-650.0, 650.0)  # corners are inside
    assert not bounds.contains(650.1, 0.0)


@pytest.mark.parametrize(
    "node",
    [
        {"x": [10.0, 10.0], "y": [-1.0, 1.0]},  # zero-width
        {"x": [10.0, -10.0], "y": [-1.0, 1.0]},  # inverted
        {"x": [1.0], "y": [-1.0, 1.0]},  # wrong arity
        {"y": [-1.0, 1.0]},  # missing axis
    ],
)
def test_bounds_rejects_degenerate(node):
    with pytest.raises(MapPackError):
        Bounds.from_dict(node)


# --------------------------------------------------------------------- rasters
def test_raster_corners_and_roundtrip():
    """Row 0 is max Y (north), column 0 is min X (west)."""
    bounds = Bounds(-650.0, 650.0, -650.0, 650.0)
    raster = RasterLayer(Path("x.png"), 2048, 2048, bounds)

    assert raster.m_per_px_x == pytest.approx(0.634765625)
    assert raster.enu_to_px(-650.0, 650.0) == pytest.approx((0.0, 0.0))
    assert raster.enu_to_px(650.0, -650.0) == pytest.approx((2048.0, 2048.0))
    assert raster.enu_to_px(0.0, 0.0) == pytest.approx((1024.0, 1024.0))

    for x, y in [(123.4, -56.7), (-650.0, -650.0), (649.9, 649.9)]:
        assert raster.px_to_enu(*raster.enu_to_px(x, y)) == pytest.approx((x, y))


def test_raster_handles_non_square_pixels():
    """A raster whose aspect differs from the bounds keeps per-axis scales."""
    raster = RasterLayer(Path("x.png"), 200, 50, Bounds(-100.0, 100.0, -50.0, 50.0))
    assert raster.m_per_px_x == pytest.approx(1.0)
    assert raster.m_per_px_y == pytest.approx(2.0)
    assert raster.enu_to_px(0.0, 0.0) == pytest.approx((100.0, 25.0))


# ---------------------------------------------------------------------- loading
def test_load_minimal_pack(tmp_path):
    """An image plus bounds is a valid pack — everything else is optional."""
    root = write_pack(tmp_path / "unit", minimal_descriptor())
    pack = MapPack.load(root)

    assert pack.name == "unit"
    assert pack.buildings == []
    assert pack.occupancy is None
    assert pack.markers == []
    assert pack.entities == []
    assert pack.overfly_allowed is True
    assert pack.px4_local_origin_enu_m == (0.0, 0.0, 0.0)


def test_load_accepts_descriptor_path_directly(tmp_path):
    root = write_pack(tmp_path / "unit", minimal_descriptor())
    assert MapPack.load(root / "map.yaml").name == "unit"


def test_missing_bounds_is_an_error(tmp_path):
    doc = minimal_descriptor()
    del doc["bounds_enu_m"]
    root = write_pack(tmp_path / "unit", doc)
    with pytest.raises(MapPackError, match="bounds_enu_m"):
        MapPack.load(root)


def test_missing_basemap_file_is_an_error(tmp_path):
    root = write_pack(tmp_path / "unit", minimal_descriptor(), basemap=False)
    with pytest.raises(MapPackError, match="does not exist"):
        MapPack.load(root)


def test_size_px_falls_back_to_reading_the_image(tmp_path):
    doc = minimal_descriptor(basemap={"file": "basemap.png"})
    pack = MapPack.load(write_pack(tmp_path / "unit", doc))
    assert (pack.basemap.width_px, pack.basemap.height_px) == (1, 1)


def test_future_schema_is_refused(tmp_path):
    root = write_pack(tmp_path / "unit", minimal_descriptor(schema_version=99))
    with pytest.raises(MapPackError, match="schema_version"):
        MapPack.load(root)


def test_discover_skips_broken_packs(tmp_path):
    write_pack(tmp_path / "good", minimal_descriptor(name="good"))
    broken = minimal_descriptor(name="broken")
    del broken["bounds_enu_m"]
    write_pack(tmp_path / "broken", broken)
    (tmp_path / "not_a_pack").mkdir()

    assert [p.name for p in MapPack.discover(tmp_path)] == ["good"]


# --------------------------------------------------------------------- contents
def test_buildings_and_overfly_policy(tmp_path):
    buildings = {
        "buildings": [
            {
                "id": "b1",
                "outer": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "roof_z_m": 12.0,
                "foundation_z_m": -0.05,
            }
        ]
    }
    root = write_pack(
        tmp_path / "unit",
        minimal_descriptor(overfly_allowed=False, cruise_band_m=[20.0, 30.0]),
        buildings=buildings,
    )
    pack = MapPack.load(root)
    (building,) = pack.buildings

    assert building.aabb_xy == (0.0, 0.0, 10.0, 5.0)
    # Raw geometry says a 12 m roof is clear at 25 m ...
    assert building.blocks_at(25.0) is False
    # ... but the planner ran with overfly disabled, so it is still an obstacle.
    assert pack.blocks_at(building, 25.0) is True
    assert pack.cruise_z_m == 25.0


def test_malformed_buildings_json_is_an_error(tmp_path):
    root = write_pack(
        tmp_path / "unit", minimal_descriptor(), buildings={"buildings": [{"id": "b1"}]}
    )
    with pytest.raises(MapPackError, match="malformed building"):
        _ = MapPack.load(root).buildings


def test_entities_and_markers(tmp_path):
    doc = minimal_descriptor(
        entities=[{"name": "trailer", "label": "트레일러", "footprint_m": [5.0, 5.0]}],
        markers=[{"name": "goal", "enu_m": [200.0, -128.0], "color": "#f03e3e"}],
    )
    pack = MapPack.load(write_pack(tmp_path / "unit", doc))

    spec = pack.entity("trailer")
    assert spec is not None and spec.label == "트레일러" and spec.footprint_m == (5.0, 5.0)
    assert spec.trail is True  # defaults on
    assert pack.entity("nope") is None
    assert pack.markers[0].enu_m == (200.0, -128.0)
    assert pack.markers[0].label == "goal"  # falls back to the name


def test_entity_without_name_is_an_error(tmp_path):
    root = write_pack(tmp_path / "unit", minimal_descriptor(entities=[{"label": "x"}]))
    with pytest.raises(MapPackError, match="need a 'name'"):
        MapPack.load(root)


# ----------------------------------------------------------------------- frames
def test_mavros_frame_conversion_is_a_translation(tmp_path):
    doc = minimal_descriptor(px4_local_origin_enu_m=[587.0, 580.0, 0.24])
    pack = MapPack.load(write_pack(tmp_path / "unit", doc))

    # MAVROS reports its local ENU origin as zero; that is the vehicle spawn.
    assert pack.mavros_to_enu(0.0, 0.0, 0.0) == pytest.approx((587.0, 580.0, 0.24))
    assert pack.mavros_to_enu(-10.0, 20.0, 5.0) == pytest.approx((577.0, 600.0, 5.24))
    assert pack.enu_to_mavros(*pack.mavros_to_enu(1.0, 2.0, 3.0)) == pytest.approx(
        (1.0, 2.0, 3.0)
    )


# ------------------------------------------------------- the real baked city pack
@pytest.mark.skipif(not CITY_PACK.is_dir(), reason="city_uav pack not baked yet")
class TestBakedCityPack:
    """Guards the baked artefact against the numbers in the coordinate contract."""

    @pytest.fixture(scope="class")
    def pack(self):
        return MapPack.load(CITY_PACK)

    def test_georeference_matches_the_world(self, pack):
        assert pack.bounds == Bounds(-650.0, 650.0, -650.0, 650.0)
        # 2048 px over the world's 1300 m ground plane.
        assert pack.basemap.width_px == 2048
        assert pack.basemap.m_per_px_x == pytest.approx(0.634765625)
        assert pack.basemap.enu_to_px(0.0, 0.0) == pytest.approx((1024.0, 1024.0))

    def test_contract_numbers(self, pack):
        assert len(pack.buildings) == 205
        assert pack.spawn_enu_m == (587.0, 580.0)
        assert pack.px4_local_origin_enu_m == (587.0, 580.0, 0.24)
        assert pack.cruise_band_m == (20.0, 30.0)
        assert pack.overfly_allowed is False
        roofs = [b.roof_z_m for b in pack.buildings]
        assert min(roofs) == pytest.approx(20.0)
        assert max(roofs) == pytest.approx(50.0)
        # The contract's re-ranked skyline has an exact 35 m mean.
        assert sum(roofs) / len(roofs) == pytest.approx(35.0, abs=1e-6)
        assert sum(1 for b in pack.buildings if b.holes) == 1  # one courtyard

    def test_every_building_is_inside_the_geofence(self, pack):
        for building in pack.buildings:
            x0, y0, x1, y1 = building.aabb_xy
            assert pack.bounds.contains(x0, y0)
            assert pack.bounds.contains(x1, y1)

    def test_trailer_entity_and_markers(self, pack):
        trailer = pack.entity("trailer")
        assert trailer is not None
        assert trailer.footprint_m == (5.0, 5.0)
        markers = {m.name: m.enu_m for m in pack.markers}
        assert markers["drone_spawn"] == (587.0, 580.0)
        assert markers["global_goal"] == (200.0, -128.0)
        assert markers["trailer_spawn"] == (-587.0, -512.0)

    def test_occupancy_layer_shares_the_basemap_georeference(self, pack):
        assert pack.occupancy is not None
        assert pack.occupancy_z_m == pytest.approx(25.0)
        # Different resolution, same world extent: a given ENU point must land on
        # the same *place* in both rasters.
        for x, y in [(0.0, 0.0), (587.0, 580.0), (-587.0, -512.0)]:
            base = pack.basemap.enu_to_px(x, y)
            occ = pack.occupancy.enu_to_px(x, y)
            assert base[0] / pack.basemap.width_px == pytest.approx(
                occ[0] / pack.occupancy.width_px
            )
            assert base[1] / pack.basemap.height_px == pytest.approx(
                occ[1] / pack.occupancy.height_px
            )
