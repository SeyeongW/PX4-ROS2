#!/usr/bin/env python3
"""Map pack loading — the GUI's only view of "which world am I looking at".

A **map pack** is one directory that fully describes a flyable world in ground-
station terms.  Swapping maps means pointing at a different directory; no GUI
code changes and no constants move.

    maps/<name>/
        map.yaml            the descriptor (the only file the GUI reads)
        basemap.png         top-down background, georeferenced by `bounds_enu_m`
        buildings.json      obstacle rings + roof heights (vector overlay)
        occupancy_z<z>.png  optional: what the planner considers blocked at z

`tools/bake_map.py` generates packs from a simulation coordinate contract
(`simulation/gazebo/maps/city_coordinates_*.yaml`).  A pack can also be written
by hand: an image plus `bounds_enu_m` is enough to render, everything else is
optional.

Nothing here imports Qt or ROS, so it is fully unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMA_VERSION = 1

DESCRIPTOR_NAME = "map.yaml"


class MapPackError(Exception):
    """A map pack is missing, malformed, or self-inconsistent."""


# --------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Bounds:
    """Axis-aligned ENU extent of a map, in metres."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width_m(self) -> float:
        return self.x_max - self.x_min

    @property
    def height_m(self) -> float:
        return self.y_max - self.y_min

    @property
    def center_m(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @staticmethod
    def from_dict(node: dict) -> "Bounds":
        try:
            xb = [float(v) for v in node["x"]]
            yb = [float(v) for v in node["y"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise MapPackError(f"bounds_enu_m must be {{x: [lo, hi], y: [lo, hi]}}: {exc}")
        if len(xb) != 2 or len(yb) != 2:
            raise MapPackError("bounds_enu_m x and y must each hold exactly two values")
        if xb[1] <= xb[0] or yb[1] <= yb[0]:
            raise MapPackError(f"bounds_enu_m must be increasing, got x={xb} y={yb}")
        return Bounds(xb[0], xb[1], yb[0], yb[1])

    def to_dict(self) -> dict:
        return {"x": [self.x_min, self.x_max], "y": [self.y_min, self.y_max]}


@dataclass(frozen=True)
class RasterLayer:
    """A georeferenced image layer: pixel grid stretched over the map bounds.

    Row 0 is the map's maximum Y (north) and column 0 its minimum X (west) —
    the same convention as `render_city_uav_astar_map.py`, so a baked occupancy
    image and the ground texture stack without a flip.
    """

    path: Path
    width_px: int
    height_px: int
    bounds: Bounds

    @property
    def m_per_px_x(self) -> float:
        return self.bounds.width_m / self.width_px

    @property
    def m_per_px_y(self) -> float:
        return self.bounds.height_m / self.height_px

    def enu_to_px(self, x: float, y: float) -> tuple[float, float]:
        """ENU metres -> fractional pixel coordinates in this raster."""
        return (
            (x - self.bounds.x_min) / self.m_per_px_x,
            (self.bounds.y_max - y) / self.m_per_px_y,
        )

    def px_to_enu(self, col: float, row: float) -> tuple[float, float]:
        """Fractional pixel coordinates -> ENU metres."""
        return (
            self.bounds.x_min + col * self.m_per_px_x,
            self.bounds.y_max - row * self.m_per_px_y,
        )


# --------------------------------------------------------------------- contents
@dataclass(frozen=True)
class Building:
    """One obstacle prism: a footprint ring (with optional holes) plus heights."""

    id: str
    outer: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]
    roof_z_m: float
    foundation_z_m: float

    def blocks_at(self, z_m: float) -> bool:
        """True when a vehicle at altitude `z_m` cannot pass over this building.

        Raw geometry only — no planner inflation.  The occupancy raster carries
        the inflated truth; this is for colouring the vector overlay.
        """
        return z_m <= self.roof_z_m

    @property
    def aabb_xy(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.outer]
        ys = [p[1] for p in self.outer]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class Marker:
    """A fixed point of interest: spawn, goal, trailer destination."""

    name: str
    label: str
    enu_m: tuple[float, float]
    color: str = "#adb5bd"


@dataclass(frozen=True)
class EntitySpec:
    """A dynamic object to draw on the map (trailer, rover, second aircraft).

    `name` is the key the pose source reports — for the Gazebo bridge that is
    the simulation entity name.  The map pack never says *how* the pose arrives;
    that is the ROS link's business.
    """

    name: str
    label: str
    footprint_m: tuple[float, float]
    color: str = "#f59f00"
    trail: bool = True
    route_yaml: str | None = None


@dataclass
class MapPack:
    """A loaded map pack.  `buildings` is read lazily on first access."""

    name: str
    root: Path
    bounds: Bounds
    basemap: RasterLayer | None = None
    occupancy: RasterLayer | None = None
    occupancy_z_m: float | None = None
    px4_local_origin_enu_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_enu_m: tuple[float, float] | None = None
    cruise_band_m: tuple[float, float] | None = None
    default_goal_enu_m: tuple[float, float, float] | None = None
    layers: list[str] = field(default_factory=list)
    entities: list[EntitySpec] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    # The planner's overfly policy.  False means every building is a full-height
    # no-fly column regardless of its roof, so the GUI must not tint tall and
    # short buildings differently — the planner routes around all of them.
    overfly_allowed: bool = True
    planner_model: dict = field(default_factory=dict)
    source_world_yaml: str | None = None
    _buildings: list[Building] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def load(path: str | Path) -> "MapPack":
        """Load a pack from its directory or directly from its `map.yaml`."""
        path = Path(path).expanduser()
        descriptor = path / DESCRIPTOR_NAME if path.is_dir() else path
        if not descriptor.is_file():
            raise MapPackError(f"no map descriptor at {descriptor}")
        root = descriptor.parent

        try:
            doc = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise MapPackError(f"{descriptor} is not valid YAML: {exc}")
        if not isinstance(doc, dict):
            raise MapPackError(f"{descriptor} must contain a YAML mapping")

        version = int(doc.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise MapPackError(
                f"{descriptor} declares schema_version {version}; this build "
                f"understands up to {SCHEMA_VERSION}"
            )

        if "bounds_enu_m" not in doc:
            raise MapPackError(f"{descriptor} is missing the required bounds_enu_m")
        bounds = Bounds.from_dict(doc["bounds_enu_m"])

        basemap = _raster(doc.get("basemap"), root, bounds, "basemap")
        occupancy = _raster(doc.get("occupancy"), root, bounds, "occupancy")
        occ_z = doc.get("occupancy", {}).get("cruise_z_m") if occupancy else None

        return MapPack(
            name=str(doc.get("name") or root.name),
            root=root,
            bounds=bounds,
            basemap=basemap,
            occupancy=occupancy,
            occupancy_z_m=None if occ_z is None else float(occ_z),
            px4_local_origin_enu_m=_xyz(doc.get("px4_local_origin_enu_m"), (0.0, 0.0, 0.0)),
            spawn_enu_m=_xy(doc.get("spawn_enu_m")),
            cruise_band_m=_pair(doc.get("cruise_band_m")),
            default_goal_enu_m=_xyz(doc.get("default_goal_enu_m"), None),
            layers=[str(v) for v in doc.get("layers", []) or []],
            entities=[_entity(e) for e in doc.get("entities", []) or []],
            markers=[_marker(m) for m in doc.get("markers", []) or []],
            overfly_allowed=bool(doc.get("overfly_allowed", True)),
            planner_model=dict(doc.get("planner_model") or {}),
            source_world_yaml=_opt_str(doc.get("source_world_yaml")),
        )

    @staticmethod
    def discover(maps_dir: str | Path) -> list["MapPack"]:
        """Load every pack under `maps_dir`, sorted by name.  Bad packs are skipped."""
        maps_dir = Path(maps_dir).expanduser()
        packs = []
        for descriptor in sorted(maps_dir.glob(f"*/{DESCRIPTOR_NAME}")):
            try:
                packs.append(MapPack.load(descriptor))
            except MapPackError:
                continue
        return packs

    # ----------------------------------------------------------------- contents
    @property
    def buildings(self) -> list[Building]:
        """Vector obstacle overlay; empty when the pack ships no buildings.json."""
        if self._buildings is None:
            self._buildings = _load_buildings(self.root / "buildings.json")
        return self._buildings

    @property
    def cruise_z_m(self) -> float:
        """Representative cruise altitude: mid-band, else the occupancy z, else 0."""
        if self.cruise_band_m is not None:
            return 0.5 * (self.cruise_band_m[0] + self.cruise_band_m[1])
        if self.occupancy_z_m is not None:
            return self.occupancy_z_m
        return 0.0

    def entity(self, name: str) -> EntitySpec | None:
        for spec in self.entities:
            if spec.name == name:
                return spec
        return None

    def blocks_at(self, building: Building, z_m: float) -> bool:
        """Whether the planner treats `building` as an obstacle at altitude `z_m`.

        With `overfly_allowed: false` every building is a full-height column, so
        roof height does not decide passability — matching the planner instead of
        the raw geometry keeps the map from promising a route over a short roof
        that A* will never take.
        """
        if not self.overfly_allowed:
            return True
        return building.blocks_at(z_m)

    # ------------------------------------------------------------------- frames
    def mavros_to_enu(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """MAVROS local ENU (origin at the PX4 local origin) -> map ENU.

        MAVROS already publishes ENU, so this is a pure translation by the
        vehicle's local origin.  The NED conversion in the coordinate contract
        stays where it belongs: inside PX4/MAVROS.
        """
        ox, oy, oz = self.px4_local_origin_enu_m
        return (x + ox, y + oy, z + oz)

    def enu_to_mavros(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        ox, oy, oz = self.px4_local_origin_enu_m
        return (x - ox, y - oy, z - oz)


# ---------------------------------------------------------------------- helpers
def _raster(node, root: Path, bounds: Bounds, what: str) -> RasterLayer | None:
    if not node:
        return None
    if not isinstance(node, dict) or "file" not in node:
        raise MapPackError(f"{what} must be a mapping with a 'file' key")
    path = root / str(node["file"])
    if not path.is_file():
        raise MapPackError(f"{what} image {path} does not exist")
    size = node.get("size_px")
    if size is None:
        width, height = _image_size(path)
    else:
        try:
            width, height = (int(size[0]), int(size[1]))
        except (TypeError, ValueError, IndexError):
            raise MapPackError(f"{what} size_px must be [width, height]")
    if width <= 0 or height <= 0:
        raise MapPackError(f"{what} size_px must be positive, got {width}x{height}")
    return RasterLayer(path=path, width_px=width, height_px=height, bounds=bounds)


def _image_size(path: Path) -> tuple[int, int]:
    """Read an image's pixel size.  Only used when the descriptor omits size_px."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow ships with the sim tools
        raise MapPackError(
            f"{path} has no size_px in the descriptor and Pillow is unavailable: {exc}"
        )
    with Image.open(path) as image:
        return image.size


def _load_buildings(path: Path) -> list[Building]:
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MapPackError(f"{path} is not valid JSON: {exc}")
    out = []
    for item in doc.get("buildings", []):
        try:
            out.append(
                Building(
                    id=str(item["id"]),
                    outer=[(float(p[0]), float(p[1])) for p in item["outer"]],
                    holes=[
                        [(float(p[0]), float(p[1])) for p in ring]
                        for ring in item.get("holes", []) or []
                    ],
                    roof_z_m=float(item["roof_z_m"]),
                    foundation_z_m=float(item.get("foundation_z_m", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise MapPackError(f"{path}: malformed building entry {item!r}: {exc}")
    return out


def _entity(node) -> EntitySpec:
    if not isinstance(node, dict) or "name" not in node:
        raise MapPackError(f"entity entries need a 'name': {node!r}")
    footprint = _pair(node.get("footprint_m")) or (2.0, 2.0)
    return EntitySpec(
        name=str(node["name"]),
        label=str(node.get("label") or node["name"]),
        footprint_m=footprint,
        color=str(node.get("color", "#f59f00")),
        trail=bool(node.get("trail", True)),
        route_yaml=_opt_str(node.get("route_yaml")),
    )


def _marker(node) -> Marker:
    if not isinstance(node, dict) or "enu_m" not in node:
        raise MapPackError(f"marker entries need an 'enu_m': {node!r}")
    point = _pair(node["enu_m"])
    assert point is not None  # _pair only returns None for a None input
    name = str(node.get("name") or node.get("label") or "marker")
    return Marker(
        name=name,
        label=str(node.get("label") or name),
        enu_m=point,
        color=str(node.get("color", "#adb5bd")),
    )


def _pair(node) -> tuple[float, float] | None:
    if node is None:
        return None
    try:
        return (float(node[0]), float(node[1]))
    except (TypeError, ValueError, IndexError):
        raise MapPackError(f"expected two numbers, got {node!r}")


def _xy(node) -> tuple[float, float] | None:
    return _pair(node)


def _xyz(node, default):
    if node is None:
        return default
    try:
        return (float(node[0]), float(node[1]), float(node[2]))
    except (TypeError, ValueError, IndexError):
        raise MapPackError(f"expected three numbers, got {node!r}")


def _opt_str(node) -> str | None:
    return None if node is None else str(node)
