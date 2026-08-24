"""Shared obstacle / free-space model for the path_plan pipeline.

All geometry is expressed in the Gazebo ENU world frame, metres.

Buildings are modelled as axis-aligned bounding boxes (AABBs).  The city YAML
stores each building as a polygon prism (``footprint.outer`` + ``roof_z_m``);
we take the xy bounding box of the footprint and the ``[foundation_z, roof_z]``
span.  This is a conservative over-approximation that keeps A* occupancy checks
and SFC box tests fast and easy to reason about.

Occupancy is queried against obstacles so a point/segment is treated as the
vehicle centre.  Callers may supply already-inflated AABBs, or keep the
physical AABBs and set ``xy_clearance_m`` for an exact horizontal Euclidean
clearance:

    free(p)  <=>  p in world bounds
                  and  p_z >= ground_z + ground_clearance
                  and  p_z <= ceiling_z
                  and  for every obstacle B:  p not in  B (+) inflation

where ``(+) inflation`` is a rectangle with rounded corners in XY, extruded
over the obstacle's existing Z span.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def _find_buildings(node):
    """Return the ``buildings`` list wherever it lives in the YAML tree."""
    if isinstance(node, dict):
        if isinstance(node.get("buildings"), list):
            return node["buildings"]
        for value in node.values():
            found = _find_buildings(value)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class WorldModel:
    """AABB obstacle field with O(N) vectorised free-space queries.

    Attributes
    ----------
    boxes_min, boxes_max : (N, 3) float arrays
        Lower / upper corners of the base obstacle AABBs. They may already be
        inflated by the caller.
    bounds_min, bounds_max : (3,) float arrays
        Planning geofence corners (ENU).  ``bounds_min[2]`` encodes the ground
        clearance floor and ``bounds_max[2]`` the ceiling.
    xy_clearance_m : float
        Optional exact Euclidean XY dilation around each base AABB. The default
        zero preserves the historical AABB contract.
    """

    boxes_min: np.ndarray
    boxes_max: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    xy_clearance_m: float = 0.0

    def __post_init__(self):
        try:
            clearance = float(self.xy_clearance_m)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "xy_clearance_m must be a finite non-negative scalar") from exc
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "xy_clearance_m must be a finite non-negative scalar")
        object.__setattr__(self, "xy_clearance_m", clearance)

    # ---------------------------------------------------------------- builders
    @staticmethod
    def from_city_yaml(
        yaml_path: str | Path,
        *,
        xy_clearance_m: float = 1.5,
        vertical_margin_m: float = 0.4,
        roof_clearance_m: float = 10.0,
        ground_clearance_m: float = 0.0,
        ceiling_m: float = 30.0,
        overfly_allowed: bool = True,
    ) -> "WorldModel":
        """Load building AABB obstacles from a city coordinate YAML.

        ``xy_clearance_m`` is owned by the vehicle: the YAML building AABBs
        remain physical/raw and free-space queries keep the path centre at
        least this Euclidean distance from them.  When ``overfly_allowed`` is
        False every building is treated as a full-height no-fly column (top set
        well above the ceiling), so the vehicle must route around *all*
        buildings laterally regardless of their height.
        """
        document = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        buildings = _find_buildings(document) or []
        lows, highs = [], []
        for b in buildings:
            outer = b["footprint"]["outer"]
            pts = np.asarray(outer, dtype=float)          # (M, 2) xy polygon
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            z0 = float(b.get("foundation_z_m", 0.0))
            z1 = float(b["roof_z_m"])
            top = (z1 + vertical_margin_m + roof_clearance_m if overfly_allowed
                   else ceiling_m + 1.0e4)               # full-height column
            lows.append((x0, y0, z0 - vertical_margin_m))
            highs.append((x1, y1, top))
        bounds = document.get("map", {}).get("bounds_enu_m", {})
        xb = bounds.get("x", [-1e4, 1e4])
        yb = bounds.get("y", [-1e4, 1e4])
        ground = float(document.get("terrain", {})
                       .get("collision_geometry", {}).get("top_z_m", 0.0))
        return WorldModel(
            boxes_min=np.asarray(lows, dtype=float).reshape(-1, 3),
            boxes_max=np.asarray(highs, dtype=float).reshape(-1, 3),
            bounds_min=np.asarray(
                [xb[0], yb[0], ground + ground_clearance_m], float),
            bounds_max=np.asarray([xb[1], yb[1], ceiling_m], float),
            xy_clearance_m=xy_clearance_m,
        )

    @staticmethod
    def from_boxes(
        boxes_min: np.ndarray,
        boxes_max: np.ndarray,
        bounds_min,
        bounds_max,
        *,
        xy_clearance_m: float = 0.0,
    ) -> "WorldModel":
        return WorldModel(
            np.asarray(boxes_min, float).reshape(-1, 3),
            np.asarray(boxes_max, float).reshape(-1, 3),
            np.asarray(bounds_min, float),
            np.asarray(bounds_max, float),
            xy_clearance_m,
        )

    # ------------------------------------------------------------ free queries
    def in_bounds(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points)
        return np.all(
            (p >= self.bounds_min) & (p <= self.bounds_max), axis=1)

    def is_free(self, points: np.ndarray) -> np.ndarray:
        """Vectorised point-in-free-space test.  Returns a boolean array."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        free = self.in_bounds(p)
        if self.boxes_min.size:
            xy_gap = (np.maximum(
                self.boxes_min[None, :, :2] - p[:, None, :2], 0.0)
                + np.maximum(
                    p[:, None, :2] - self.boxes_max[None, :, :2], 0.0))
            inside_xy = np.einsum("pni,pni->pn", xy_gap, xy_gap) \
                <= self.xy_clearance_m ** 2
            inside_z = ((p[:, None, 2] >= self.boxes_min[None, :, 2])
                        & (p[:, None, 2] <= self.boxes_max[None, :, 2]))
            inside = inside_xy & inside_z
            free &= ~np.any(inside, axis=1)
        return free

    def segment_is_free(self, a, b, step_m: float = 0.5) -> bool:
        """Require the entire a->b segment to stay in bounds and miss every AABB.

        ``step_m`` remains in the public signature for compatibility, but an
        exact slab intersection is used.  Sampling can miss a thin obstacle or
        a corner crossing between adjacent samples.
        """
        del step_m
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if (a.shape != (3,) or b.shape != (3,)
                or not np.all(np.isfinite(a))
                or not np.all(np.isfinite(b))
                or not bool(np.all(self.in_bounds(np.vstack((a, b)))))):
            return False

        direction = b - a
        clearance = self.xy_clearance_m
        segment_low_x = min(a[0], b[0]) - clearance
        segment_high_x = max(a[0], b[0]) + clearance
        segment_low_y = min(a[1], b[1]) - clearance
        segment_high_y = max(a[1], b[1]) + clearance
        clearance_squared = clearance ** 2
        for low, high in zip(self.boxes_min, self.boxes_max):
            # Strict separation keeps exact clearance tangency inclusive.
            if (high[0] < segment_low_x or low[0] > segment_high_x
                    or high[1] < segment_low_y or low[1] > segment_high_y):
                continue
            clipped = _clip_segment_to_z_span(a, direction, low[2], high[2])
            if clipped is None:
                continue
            start_xy = a[:2] + clipped[0] * direction[:2]
            end_xy = a[:2] + clipped[1] * direction[:2]
            if (_segment_aabb_distance_squared_xy(
                    start_xy, end_xy, low[:2], high[:2])
                    <= clearance_squared):
                return False
        return True

    def clearance(self, point) -> float:
        """Euclidean distance from ``point`` to the nearest obstacle box (0 if inside).

        Point-to-AABB distance per box: ``‖max(lo−p, 0) + max(p−hi, 0)‖`` (the
        component-wise gap outside the box; zero on axes where p is within the
        box span). The minimum over all boxes is the clearance. Used by the A*
        cost function to reward routes that keep away from building walls.
        """
        p = np.asarray(point, dtype=float)
        if not self.boxes_min.size:
            return float("inf")
        xy_gap = (np.maximum(self.boxes_min[:, :2] - p[:2], 0.0)
                  + np.maximum(p[:2] - self.boxes_max[:, :2], 0.0))
        xy_distance = np.sqrt(np.einsum("ij,ij->i", xy_gap, xy_gap))
        xy_distance = np.maximum(xy_distance - self.xy_clearance_m, 0.0)
        z_gap = (np.maximum(self.boxes_min[:, 2] - p[2], 0.0)
                 + np.maximum(p[2] - self.boxes_max[:, 2], 0.0))
        return float(np.hypot(xy_distance, z_gap).min())

    def box_is_free(self, lo: np.ndarray, hi: np.ndarray) -> bool:
        """True iff the query AABB [lo, hi] overlaps no obstacle and stays in bounds.

        AABB overlap test (separating-axis on axis-aligned boxes):
            overlap  <=>  for all axes:  lo <= B.max  and  hi >= B.min
        """
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        if np.any(lo < self.bounds_min) or np.any(hi > self.bounds_max):
            return False
        if not self.boxes_min.size:
            return True
        z_overlap = ((lo[2] <= self.boxes_max[:, 2])
                     & (hi[2] >= self.boxes_min[:, 2]))
        xy_gap = (np.maximum(self.boxes_min[:, :2] - hi[:2], 0.0)
                  + np.maximum(lo[:2] - self.boxes_max[:, :2], 0.0))
        xy_overlap = np.einsum("ij,ij->i", xy_gap, xy_gap) \
            <= self.xy_clearance_m ** 2
        overlap = z_overlap & xy_overlap
        return not bool(np.any(overlap))


def _clip_segment_to_z_span(a, direction, low_z, high_z):
    """Return the segment parameter interval inside an inclusive Z span."""
    if direction[2] == 0.0:
        return (0.0, 1.0) if low_z <= a[2] <= high_z else None
    first = (low_z - a[2]) / direction[2]
    second = (high_z - a[2]) / direction[2]
    enter = max(0.0, min(first, second))
    leave = min(1.0, max(first, second))
    return None if enter > leave else (enter, leave)


def _point_segment_distance_squared_xy(point, a, b):
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    px = float(point[0] - a[0])
    py = float(point[1] - a[1])
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return px * px + py * py
    fraction = max(0.0, min(1.0,
                            (px * dx + py * dy) / length_squared))
    delta_x = float(point[0]) - (float(a[0]) + fraction * dx)
    delta_y = float(point[1]) - (float(a[1]) + fraction * dy)
    return delta_x * delta_x + delta_y * delta_y


def _point_aabb_distance_squared_xy(point, low, high):
    gap = np.maximum(low - point, 0.0) + np.maximum(point - high, 0.0)
    return float(gap @ gap)


def _segment_aabb_distance_squared_xy(a, b, low, high):
    """Exact squared distance between a 2-D segment and an AABB rectangle."""
    direction = b - a
    enter, leave = 0.0, 1.0
    for axis in range(2):
        if direction[axis] == 0.0:
            if a[axis] < low[axis] or a[axis] > high[axis]:
                break
            continue
        first = (low[axis] - a[axis]) / direction[axis]
        second = (high[axis] - a[axis]) / direction[axis]
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
        if enter > leave:
            break
    else:
        return 0.0

    distance_squared = min(
        _point_aabb_distance_squared_xy(a, low, high),
        _point_aabb_distance_squared_xy(b, low, high),
    )
    for corner in (
            np.array([low[0], low[1]]),
            np.array([low[0], high[1]]),
            np.array([high[0], low[1]]),
            np.array([high[0], high[1]])):
        distance_squared = min(
            distance_squared,
            _point_segment_distance_squared_xy(corner, a, b))
    return distance_squared
