#!/usr/bin/env python3
"""Bake a GCS map pack from a simulation coordinate contract.

    python3 tools/bake_map.py --world-yaml simulation/gazebo/maps/city_coordinates_uav.yaml

Reads a `schema_version: 2` city contract (buildings, geofence, PX4 local origin,
spawn, trailer, terrain textures) and writes a self-contained pack:

    maps/<name>/map.yaml  basemap.png  buildings.json  [occupancy_z<z>.png]

Everything in the pack is derived from the contract — nothing is hardcoded here,
so a new or edited world becomes a new map by re-running this script.

The occupancy layer is generated through `path_plan.world_model.WorldModel`, the
same loader the planner uses, so the GUI shows exactly the free space A* searched
rather than a lookalike recomputed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
DEFAULT_WORLD_YAML = REPO / "simulation/gazebo/maps/city_coordinates_uav.yaml"
DEFAULT_PLANNER_CONFIG = REPO / "flight/path_plan/config/city_uav.yaml"
DEFAULT_MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

# Obstacle-model keys the GCS must copy from the planner rather than guess.  The
# planner's own defaults differ from its city config (notably `overfly_allowed`,
# which is False for this map), so a pack baked with library defaults would show
# free space A* never had.
_PLANNER_KEYS = (
    "inflation_xy_m",
    "roof_clearance_m",
    "cruise_floor_m",
    "cruise_ceiling_m",
    "overfly_allowed",
)

# The contract's own paths were written when `gazebo/` sat at the repo root; it
# now lives under `simulation/`.  Try both rather than rewriting the contract.
_PATH_PREFIXES = ("", "simulation/")


# ------------------------------------------------------------------ contract IO
def load_contract(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a YAML mapping")
    version = int(doc.get("schema_version", 0))
    if version != 2:
        raise SystemExit(
            f"{path}: schema_version {version} is not the version-2 city contract "
            "this baker reads"
        )
    return doc


def load_planner_params(path: Path | None) -> dict:
    """Read the A* node's obstacle model out of a path_plan config file.

    The GCS must draw the same free space the planner searched, so these values
    are copied from the planner's config instead of being restated here.
    """
    if path is None or not path.is_file():
        print(f"WARN: planner config {path} not found; using WorldModel defaults")
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = (doc.get("/astar_planner") or {}).get("ros__parameters") or {}
    picked = {key: params[key] for key in _PLANNER_KEYS if key in params}
    if picked:
        print(f"planner model from {path.name}: {picked}")
    if "goal_enu_m" in params:
        picked["goal_enu_m"] = params["goal_enu_m"]
    return picked


def find_buildings(node):
    """Return the `buildings` list wherever it sits in the contract tree.

    Mirrors `path_plan.world_model._find_buildings` so both agree on which list
    is authoritative when a contract nests obstacles.
    """
    if isinstance(node, dict):
        if isinstance(node.get("buildings"), list):
            return node["buildings"]
        for value in node.values():
            found = find_buildings(value)
            if found is not None:
                return found
    return None


def resolve_repo_path(relative: str) -> Path | None:
    for prefix in _PATH_PREFIXES:
        candidate = REPO / (prefix + relative)
        if candidate.is_file():
            return candidate
    return None


# ----------------------------------------------------------------- georeference
def verify_texture_georef(contract: dict, bounds: dict, texture: Path) -> None:
    """Cross-check the ground texture against the world SDF before trusting it.

    A texture is only a valid basemap if the world stretches exactly one tile of
    it across the whole ground plane.  Gazebo says so with two numbers: the
    heightmap's `<size>` (the plane's extent in metres) and the texture's
    `<size>` (metres covered by one tile).  If they disagree with each other or
    with the geofence, the image is not a 1:1 map of the bounds and using it
    would silently misplace every building.
    """
    world_rel = contract.get("map", {}).get("world_file")
    if not world_rel:
        print("WARN: contract has no map.world_file; texture georef unverified")
        return
    world_path = resolve_repo_path(str(world_rel))
    if world_path is None:
        print(f"WARN: world file {world_rel} not found; texture georef unverified")
        return

    span_x = bounds["x"][1] - bounds["x"][0]
    span_y = bounds["y"][1] - bounds["y"][0]
    root = ET.parse(world_path).getroot()
    for heightmap in root.iter("heightmap"):
        size = heightmap.find("size")
        tex_size = heightmap.find("texture/size")
        if size is None or size.text is None or tex_size is None or tex_size.text is None:
            continue
        plane = [float(v) for v in size.text.split()]
        tile = float(tex_size.text.split()[0])
        if not (math.isclose(plane[0], span_x) and math.isclose(plane[1], span_y)):
            continue
        if not math.isclose(tile, plane[0]):
            raise SystemExit(
                f"{world_path.name}: ground texture tiles every {tile} m over a "
                f"{plane[0]} m plane, so {texture.name} is not a 1:1 basemap. "
                "Re-bake with --no-ground-texture to synthesise one instead."
            )
        print(
            f"georef verified: {texture.name} covers {plane[0]}x{plane[1]} m "
            f"(1 tile) == geofence {span_x}x{span_y} m"
        )
        return
    print(
        f"WARN: no ground heightmap matching the {span_x}x{span_y} m geofence in "
        f"{world_path.name}; texture georef unverified"
    )


# --------------------------------------------------------------------- basemaps
def write_basemap(texture: Path | None, buildings, bounds, out: Path, size_px: int):
    """Copy the ground texture as the basemap, or synthesise one from footprints."""
    if texture is not None:
        with Image.open(texture) as image:
            image = image.convert("RGB")
            if max(image.size) != size_px:
                image = image.resize((size_px, size_px), Image.LANCZOS)
            image.save(out)
        return out.name, image.size

    # No texture: draw a plain ground with building footprints so the operator
    # still sees the city outline.  Deliberately flat and unstyled — the vector
    # layer is what the GUI actually shades.
    from PIL import ImageDraw

    span_x = bounds["x"][1] - bounds["x"][0]
    span_y = bounds["y"][1] - bounds["y"][0]
    height_px = int(round(size_px * span_y / span_x))
    image = Image.new("RGB", (size_px, height_px), (58, 62, 68))
    draw = ImageDraw.Draw(image)
    m_per_px_x = span_x / size_px
    m_per_px_y = span_y / height_px
    for b in buildings:
        ring = [
            (
                (px - bounds["x"][0]) / m_per_px_x,
                (bounds["y"][1] - py) / m_per_px_y,
            )
            for px, py in b["footprint"]["outer"]
        ]
        draw.polygon(ring, fill=(126, 132, 140), outline=(90, 95, 102))
    image.save(out)
    return out.name, image.size


def write_occupancy(
    world_yaml: Path, bounds, cruise_z: float, out: Path, res_m: float, planner: dict
):
    """Render what the *planner* treats as blocked at `cruise_z`.

    Uses `WorldModel.from_city_yaml` with the planner's own obstacle-model
    parameters, so wall inflation, roof clearance and the overfly policy match A*
    exactly.  Free is 255, blocked is 0 — the same convention as
    `simulation/gazebo/tools/render_city_uav_astar_map.py`.
    """
    world_model = _import_world_model()
    if world_model is None:
        return None, None

    kwargs = {key: planner[key] for key in ("inflation_xy_m", "roof_clearance_m") if key in planner}
    if "overfly_allowed" in planner:
        kwargs["overfly_allowed"] = bool(planner["overfly_allowed"])
    if "cruise_ceiling_m" in planner:
        kwargs["ceiling_m"] = float(planner["cruise_ceiling_m"])
    model = world_model.from_city_yaml(world_yaml, **kwargs)
    span_x = bounds["x"][1] - bounds["x"][0]
    span_y = bounds["y"][1] - bounds["y"][0]
    width = int(round(span_x / res_m))
    height = int(round(span_y / res_m))
    grid = np.full((height, width), 255, dtype=np.uint8)

    # A box blocks this altitude when the cruise plane cuts through it.
    lows, highs = model.boxes_min, model.boxes_max
    hit = (lows[:, 2] <= cruise_z) & (highs[:, 2] >= cruise_z)
    for lo, hi in zip(lows[hit], highs[hit]):
        col0 = int(np.floor((lo[0] - bounds["x"][0]) / res_m))
        col1 = int(np.ceil((hi[0] - bounds["x"][0]) / res_m))
        row0 = int(np.floor((bounds["y"][1] - hi[1]) / res_m))
        row1 = int(np.ceil((bounds["y"][1] - lo[1]) / res_m))
        grid[max(row0, 0):max(row1, 0), max(col0, 0):max(col1, 0)] = 0

    Image.fromarray(grid, mode="L").save(out)
    blocked = int((grid == 0).sum())
    print(
        f"occupancy: {width}x{height} @ {res_m} m/px, {int(hit.sum())} of "
        f"{len(lows)} obstacles cut z={cruise_z} m ({100.0 * blocked / grid.size:.1f}% blocked)"
    )
    return out.name, (width, height)


def _import_world_model():
    """Import the planner's WorldModel so occupancy matches A* exactly."""
    sys.path.insert(0, str(REPO / "flight/path_plan"))
    try:
        from path_plan.world_model import WorldModel

        return WorldModel
    except ImportError as exc:
        print(
            f"WARN: cannot import path_plan.world_model ({exc}); skipping the "
            "occupancy layer.  The pack still renders without it."
        )
        return None


# ------------------------------------------------------------------- vector data
def write_buildings(buildings, contract: dict, out: Path) -> dict:
    records = []
    for b in buildings:
        footprint = b["footprint"]
        records.append(
            {
                "id": str(b["id"]),
                "roof_z_m": float(b["roof_z_m"]),
                "foundation_z_m": float(b.get("foundation_z_m", 0.0)),
                "outer": [[float(p[0]), float(p[1])] for p in footprint["outer"]],
                "holes": [
                    [[float(p[0]), float(p[1])] for p in ring]
                    for ring in footprint.get("holes", []) or []
                ],
            }
        )
    roofs = [r["roof_z_m"] for r in records]
    doc = {
        "schema_version": 1,
        "map_name": contract.get("map", {}).get("name"),
        "count": len(records),
        "roof_z_range_m": [min(roofs), max(roofs)] if roofs else None,
        "holes_count": sum(1 for r in records if r["holes"]),
        "buildings": records,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def cross_check_vertices(records_count: int, contract_name: str) -> None:
    """Compare the building count against the exported A* vertex CSV, if present."""
    # Only this map's own export — never another map's, which would compare
    # a synthetic or new contract against the city's numbers.
    csv_path = resolve_repo_path(f"gazebo/maps/{contract_name}_building_vertices.csv")
    if csv_path is None:
        return
    ids = set()
    with csv_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        try:
            column = header.index("building_id")
        except ValueError:
            return
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) > column:
                ids.add(fields[column])
    if len(ids) != records_count:
        print(
            f"WARN: {csv_path.name} lists {len(ids)} buildings but the contract "
            f"has {records_count}; one of them is stale"
        )
    else:
        print(f"cross-check: {csv_path.name} agrees on {records_count} buildings")


# ---------------------------------------------------------------------- entities
def entity_specs(contract: dict, world_yaml: Path) -> list[dict]:
    """Derive dynamic-object specs from the contract's own sections.

    Today that is the trailer; anything else the contract grows (rover, second
    aircraft) should be added here rather than in the GUI.
    """
    out = []
    trailer = contract.get("trailer")
    if isinstance(trailer, dict) and trailer.get("entity_name"):
        footprint = trailer.get("body_footprint_m") or [5.0, 5.0]
        spec = {
            "name": str(trailer["entity_name"]),
            "label": "트레일러",
            "footprint_m": [float(footprint[0]), float(footprint[1])],
            "color": "#f59f00",
            "trail": True,
        }
        # A pursuit scenario file, when one sits beside the contract, carries the
        # trailer's intended route so the GUI can draw where it is headed.
        loop = world_yaml.parent / f"{contract['map']['name']}_trailer_loop.yaml"
        if loop.is_file():
            spec["route_yaml"] = str(loop.relative_to(REPO))
        out.append(spec)
    return out


# -------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-yaml", type=Path, default=DEFAULT_WORLD_YAML)
    parser.add_argument(
        "--planner-config",
        type=Path,
        default=DEFAULT_PLANNER_CONFIG,
        help="path_plan config to copy the obstacle model from",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="pack directory (default: maps/<contract map name>)",
    )
    parser.add_argument(
        "--ground-texture",
        type=Path,
        default=None,
        help="basemap image (default: the contract's terrain.road_texture)",
    )
    parser.add_argument(
        "--no-ground-texture",
        action="store_true",
        help="synthesise the basemap from footprints instead of using a texture",
    )
    parser.add_argument("--basemap-px", type=int, default=2048)
    parser.add_argument(
        "--cruise-z",
        type=float,
        default=None,
        help="altitude for the occupancy layer (default: mid cruise band)",
    )
    parser.add_argument("--occupancy-res-m", type=float, default=1.0)
    args = parser.parse_args()

    contract = load_contract(args.world_yaml)
    planner = load_planner_params(args.planner_config)
    map_node = contract.get("map", {})
    name = str(map_node.get("name") or args.world_yaml.stem)
    bounds = map_node.get("bounds_enu_m")
    if not bounds:
        raise SystemExit(f"{args.world_yaml}: map.bounds_enu_m is required")

    buildings = find_buildings(contract) or []
    out_dir = args.out or (DEFAULT_MAPS_DIR / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- basemap
    texture = None
    if not args.no_ground_texture:
        texture = args.ground_texture
        if texture is None:
            rel = contract.get("terrain", {}).get("road_texture")
            texture = resolve_repo_path(str(rel)) if rel else None
        if texture is not None and not Path(texture).is_file():
            raise SystemExit(f"ground texture {texture} does not exist")
        if texture is not None:
            verify_texture_georef(contract, bounds, Path(texture))
        else:
            print("no ground texture in the contract; synthesising a basemap")
    basemap_file, basemap_size = write_basemap(
        None if texture is None else Path(texture),
        buildings,
        bounds,
        out_dir / "basemap.png",
        args.basemap_px,
    )

    # --- vector obstacles
    doc = write_buildings(buildings, contract, out_dir / "buildings.json")
    cross_check_vertices(doc["count"], name)

    # --- planner occupancy at cruise altitude
    band = _cruise_band(contract, planner)
    cruise_z = args.cruise_z if args.cruise_z is not None else 0.5 * (band[0] + band[1])
    occ_file, occ_size = write_occupancy(
        args.world_yaml,
        bounds,
        cruise_z,
        out_dir / f"occupancy_z{cruise_z:g}.png",
        args.occupancy_res_m,
        planner,
    )

    # --- descriptor
    frames = contract.get("frames", {}).get("px4_local", {})
    origin = frames.get("origin_enu_m") or [0.0, 0.0, 0.0]
    spawn = contract.get("spawn", {}).get("gazebo_spawn_pose_enu") or {}
    descriptor = {
        "schema_version": 1,
        "name": name,
        "source_world_yaml": str(args.world_yaml.relative_to(REPO))
        if args.world_yaml.is_relative_to(REPO)
        else str(args.world_yaml),
        "bounds_enu_m": {
            "x": [float(bounds["x"][0]), float(bounds["x"][1])],
            "y": [float(bounds["y"][0]), float(bounds["y"][1])],
        },
        "basemap": {"file": basemap_file, "size_px": [basemap_size[0], basemap_size[1]]},
        "px4_local_origin_enu_m": [float(v) for v in origin],
        "cruise_band_m": [float(band[0]), float(band[1])],
        # The GUI colours buildings by whether the planner may fly over them, so
        # it needs the same policy the planner ran with.
        "overfly_allowed": bool(planner.get("overfly_allowed", True)),
        "planner_model": {
            key: planner[key] for key in ("inflation_xy_m", "roof_clearance_m") if key in planner
        },
        "layers": ["basemap", "occupancy", "buildings", "geofence", "markers"],
        "markers": static_markers(contract),
        "entities": entity_specs(contract, args.world_yaml),
    }
    if spawn:
        descriptor["spawn_enu_m"] = [float(spawn.get("x", 0.0)), float(spawn.get("y", 0.0))]
    if occ_file:
        descriptor["occupancy"] = {
            "file": occ_file,
            "size_px": [occ_size[0], occ_size[1]],
            "cruise_z_m": cruise_z,
        }
    else:
        descriptor["layers"].remove("occupancy")
    goal = _fixed_goal(contract, planner)
    if goal is not None:
        descriptor["default_goal_enu_m"] = [goal[0], goal[1], cruise_z]

    (out_dir / "map.yaml").write_text(
        yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(
        f"\nbaked map pack '{name}' -> {out_dir}\n"
        f"  basemap    {basemap_file} {basemap_size[0]}x{basemap_size[1]} px "
        f"({(bounds['x'][1] - bounds['x'][0]) / basemap_size[0]:.9g} m/px)\n"
        f"  buildings  {doc['count']} ({doc['holes_count']} with courtyards), "
        f"roofs {doc['roof_z_range_m'][0]:g}..{doc['roof_z_range_m'][1]:g} m\n"
        f"  occupancy  {occ_file or '(skipped)'}\n"
        f"  entities   {[e['name'] for e in descriptor['entities']] or '(none)'}"
    )
    return 0


def _cruise_band(contract: dict, planner: dict) -> tuple[float, float]:
    """Cruise band from the planner config, else the contract, else 10-20 m."""
    lo = planner.get("cruise_floor_m")
    hi = planner.get("cruise_ceiling_m")
    if lo is None or hi is None:
        drone = contract.get("drone") or {}
        lo = drone.get("cruise_floor_m", lo)
        hi = drone.get("cruise_ceiling_m", hi)
    if lo is None or hi is None:
        return (10.0, 20.0)
    return (float(lo), float(hi))


def _fixed_goal(contract: dict, planner: dict) -> tuple[float, float] | None:
    """The route's goal: what the planner is configured to fly to, else the contract's."""
    goal = planner.get("goal_enu_m")
    if goal:
        return (float(goal[0]), float(goal[1]))
    fixed = _fixed_coordinates(contract)
    if "global_goal" in fixed:
        return fixed["global_goal"]
    return None


def _fixed_coordinates(contract: dict) -> dict[str, tuple[float, float]]:
    derivation = contract.get("derivation")
    if not isinstance(derivation, dict):
        return {}
    node = derivation.get("fixed_mission_coordinates_enu_m") or {}
    out = {}
    for key, value in node.items():
        try:
            out[str(key)] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError, IndexError):
            continue
    return out


# Labels for the contract's fixed mission coordinates, in draw order.
_MARKER_LABELS = {
    "drone_spawn": ("드론 스폰", "#4dabf7"),
    "global_goal": ("목표", "#f03e3e"),
    "trailer_spawn": ("트레일러 스폰", "#f59f00"),
    "trailer_destination": ("트레일러 목적지", "#ae3ec9"),
}


def static_markers(contract: dict) -> list[dict]:
    """Fixed points of interest the contract already pins down."""
    out = []
    for key, (x, y) in _fixed_coordinates(contract).items():
        label, color = _MARKER_LABELS.get(key, (key, "#adb5bd"))
        out.append({"name": key, "label": label, "enu_m": [x, y], "color": color})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
