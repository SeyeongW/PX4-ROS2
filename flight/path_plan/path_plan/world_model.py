"""Shared obstacle / free-space model for the path_plan pipeline.

All geometry is expressed in the Gazebo ENU world frame, metres.

Buildings are modelled as axis-aligned bounding boxes (AABBs).  The city YAML
stores each building as a polygon prism (``footprint.outer`` + ``roof_z_m``);
we take the xy bounding box of the footprint and the ``[foundation_z, roof_z]``
span.  This is a conservative over-approximation that keeps A* occupancy checks
and SFC box tests fast and easy to reason about.

Occupancy is queried against *inflated* obstacles so a point/segment is treated
as the vehicle centre.  Inflation is the Minkowski sum of the obstacle with the
vehicle radius:

    free(p)  <=>  p in world bounds
                  and  p_z >= ground_z + ground_clearance
                  and  p_z <= ceiling_z
                  and  for every obstacle B:  p not in  B (+) inflation

where ``(+) inflation`` grows B by ``inflation_xy`` horizontally, by
``vertical_margin`` downward and by ``vertical_margin + roof_clearance`` upward.
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
    """Inflated AABB obstacle field with O(N) vectorised free-space queries.

    Attributes
    ----------
    boxes_min, boxes_max : (N, 3) float arrays
        Lower / upper corners of the *already inflated* obstacle AABBs.
    bounds_min, bounds_max : (3,) float arrays
        Planning geofence corners (ENU).  ``bounds_min[2]`` encodes the ground
        clearance floor and ``bounds_max[2]`` the ceiling.
    """

    boxes_min: np.ndarray
    boxes_max: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    # ---------------------------------------------------------------- builders
    @staticmethod
    def from_city_yaml(
        yaml_path: str | Path,
        *,
        inflation_xy_m: float = 1.45,
        vertical_margin_m: float = 0.4,
        roof_clearance_m: float = 10.0,
        ground_clearance_m: float = 0.0,
        ceiling_m: float = 30.0,
        overfly_allowed: bool = True,
    ) -> "WorldModel":
        """Load building AABB obstacles from a city coordinate YAML.

        ``inflation_xy_m`` is the horizontal wall clearance (path-centre keeps at
        least this far from any wall).  When ``overfly_allowed`` is False every
        building is treated as a full-height no-fly column (top set well above the
        ceiling), so the vehicle must route around *all* buildings laterally
        regardless of their height instead of flying over short ones.
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
            lows.append((x0 - inflation_xy_m, y0 - inflation_xy_m, z0 - vertical_margin_m))
            highs.append((x1 + inflation_xy_m, y1 + inflation_xy_m, top))
        bounds = document.get("map", {}).get("bounds_enu_m", {})
        xb = bounds.get("x", [-1e4, 1e4])
        yb = bounds.get("y", [-1e4, 1e4])
        ground = float(document.get("terrain", {})
                       .get("collision_geometry", {}).get("top_z_m", 0.0))
        return WorldModel(
            boxes_min=np.asarray(lows, dtype=float).reshape(-1, 3),
            boxes_max=np.asarray(highs, dtype=float).reshape(-1, 3),
            bounds_min=np.asarray([xb[0], yb[0], ground + ground_clearance_m], float),
            bounds_max=np.asarray([xb[1], yb[1], ceiling_m], float),
        )

    @staticmethod
    def from_boxes(
        boxes_min: np.ndarray,
        boxes_max: np.ndarray,
        bounds_min,
        bounds_max,
    ) -> "WorldModel":
        return WorldModel(
            np.asarray(boxes_min, float).reshape(-1, 3),
            np.asarray(boxes_max, float).reshape(-1, 3),
            np.asarray(bounds_min, float),
            np.asarray(bounds_max, float),
        )

    # ------------------------------------------------------------- free queries
    def in_bounds(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points)
        return np.all((p >= self.bounds_min) & (p <= self.bounds_max), axis=1)

    def is_free(self, points: np.ndarray) -> np.ndarray:
        """Vectorised point-in-free-space test.  Returns a boolean array."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        free = self.in_bounds(p)
        if self.boxes_min.size:
            # inside box i  <=>  all axes:  min_i <= p <= max_i
            # p: (P,1,3), boxes: (1,N,3)  ->  (P,N)
            inside = np.all(
                (p[:, None, :] >= self.boxes_min[None, :, :])
                & (p[:, None, :] <= self.boxes_max[None, :, :]),
                axis=2,
            )
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
        for low, high in zip(self.boxes_min, self.boxes_max):
            enter, leave = 0.0, 1.0
            for axis in range(3):
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
        gap = np.maximum(self.boxes_min - p, 0.0) + np.maximum(p - self.boxes_max, 0.0)
        return float(np.sqrt(np.einsum("ij,ij->i", gap, gap)).min())

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
        overlap = np.all(
            (lo[None, :] <= self.boxes_max) & (hi[None, :] >= self.boxes_min),
            axis=1,
        )
        return not bool(np.any(overlap))
