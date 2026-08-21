"""Safe Flight Corridor (SFC) — a **guaranteed-free** AABB per seed point.

Each box is seeded at a *collision-free point* (a tiny cube ``p ± eps``, free
because ``p`` is free) and then inflated face by face while it stays
collision-free (:meth:`WorldModel.box_is_free`) and within ``max_extent``:

    while box_is_free(grow(box, f, step)) and extent_f < max_extent:
        box <- grow(box, f, step)

Seeding from a free point (instead of a whole A* segment's AABB) is the fix for
the old bug where a long segment's seed box already contained buildings and was
returned unchecked.  Every returned box passes ``box_is_free`` by construction.

The ego-planner optimizer (`bspline_optimizer.py`) uses **one box per control
point** as the soft corridor constraint, so :meth:`boxes_for_points` is the main
entry point; :meth:`build` resamples an A* polyline into free points for
standalone corridor visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world_model import WorldModel

_FACES = (
    (0, -1), (0, +1),   # -x, +x
    (1, -1), (1, +1),   # -y, +y
    (2, -1), (2, +1),   # -z, +z
)


@dataclass
class Corridor:
    """Ordered free boxes.  ``boxes_min/max`` are (M, 3)."""
    boxes_min: np.ndarray
    boxes_max: np.ndarray

    def __len__(self) -> int:
        return len(self.boxes_min)


class SafeFlightCorridor:
    def __init__(self, world: WorldModel, *, step_m: float = 0.5,
                 max_extent_m: float = 8.0, seed_spacing_m: float = 3.0,
                 seed_eps_m: float = 0.05):
        self.world = world
        self.step = float(step_m)
        self.max_extent = float(max_extent_m)
        self.seed_spacing = float(seed_spacing_m)
        self.eps = float(seed_eps_m)

    def _inflate(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = lo.astype(float).copy()
        hi = hi.astype(float).copy()
        centre = 0.5 * (lo + hi)
        for axis, direction in _FACES:
            for _ in range(int(self.max_extent / self.step) + 1):
                trial_lo, trial_hi = lo.copy(), hi.copy()
                if direction < 0:
                    trial_lo[axis] -= self.step
                    reached = centre[axis] - trial_lo[axis] >= self.max_extent
                else:
                    trial_hi[axis] += self.step
                    reached = trial_hi[axis] - centre[axis] >= self.max_extent
                if not self.world.box_is_free(trial_lo, trial_hi):
                    break
                lo, hi = trial_lo, trial_hi
                if reached:
                    break
        return lo, hi

    def _box_for_point(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Free box around a single point.  ``p`` should be collision-free; if
        the tiny seed is not free (rare), fall back to a degenerate box at ``p``
        so the optimizer's rebound penalty still pulls control points toward it."""
        p = np.asarray(p, float)
        lo = p - self.eps
        hi = p + self.eps
        if not self.world.box_is_free(lo, hi):
            return p.copy(), p.copy()
        return self._inflate(lo, hi)

    def boxes_for_points(self, seed_points: np.ndarray) -> Corridor:
        """One guaranteed-free box per seed point (used per B-spline control point)."""
        seeds = np.atleast_2d(np.asarray(seed_points, float))
        lows = np.empty_like(seeds)
        highs = np.empty_like(seeds)
        for i, p in enumerate(seeds):
            lows[i], highs[i] = self._box_for_point(p)
        return Corridor(lows, highs)

    def cover_polyline(
            self, points_m: np.ndarray, *, max_depth: int = 8,
    ) -> tuple[np.ndarray, Corridor]:
        """Cover every segment of a collision-free polyline with free boxes.

        A diagonal segment can be safe while its axis-aligned bounding box is
        blocked. Split only those segments, then require every returned chord
        to fit in at least one free convex box. A colliding or uncertifiable
        path raises instead of producing a partial corridor.
        """
        points = np.asarray(points_m, float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.all(np.isfinite(points)) or max_depth < 0):
            raise ValueError("SFC polyline must be finite Nx3 with N >= 2")
        keep = np.r_[True, np.linalg.norm(
            np.diff(points, axis=0), axis=1) > 1.0e-9]
        points = points[keep]
        if len(points) < 2:
            raise ValueError("SFC polyline has no spatial extent")

        def split(a: np.ndarray, b: np.ndarray, depth: int) -> list[np.ndarray]:
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            if (np.all(hi - lo <= 2.0 * self.max_extent)
                    and self.world.box_is_free(lo, hi)):
                return [a, b]
            if not self.world.segment_is_free_exact(a, b):
                raise ValueError("SFC polyline contains a colliding segment")
            if depth >= max_depth:
                raise ValueError("SFC segment cannot be boxed safely")
            mid = 0.5 * (a + b)
            left = split(a, mid, depth + 1)
            right = split(mid, b, depth + 1)
            return [*left[:-1], *right]

        refined = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            refined.extend(split(a, b, 0)[1:])
        refined = np.asarray(refined, float)

        lows, highs = [], []
        index = 0
        while index < len(refined) - 1:
            lo = refined[index].copy()
            hi = refined[index].copy()
            last = None
            for candidate in range(index + 1, len(refined)):
                trial_lo = np.minimum(lo, refined[candidate])
                trial_hi = np.maximum(hi, refined[candidate])
                if (np.any(trial_hi - trial_lo > 2.0 * self.max_extent)
                        or not self.world.box_is_free(trial_lo, trial_hi)):
                    break
                lo, hi, last = trial_lo, trial_hi, candidate
            if last is None:
                raise ValueError("SFC could not cover a polyline segment")
            box_lo, box_hi = self._inflate(lo, hi)
            if (not self.world.box_is_free(box_lo, box_hi)
                    or np.any(box_lo > lo) or np.any(box_hi < hi)):
                raise ValueError("SFC inflation produced an invalid box")
            lows.append(box_lo)
            highs.append(box_hi)
            index = last

        corridor = Corridor(np.asarray(lows, float), np.asarray(highs, float))
        for a, b in zip(refined[:-1], refined[1:]):
            covered = np.any(np.all(
                (a >= corridor.boxes_min - 1.0e-9)
                & (a <= corridor.boxes_max + 1.0e-9)
                & (b >= corridor.boxes_min - 1.0e-9)
                & (b <= corridor.boxes_max + 1.0e-9), axis=1))
            if not covered:
                raise ValueError("SFC left a polyline segment uncovered")
        return refined, corridor

    def build(self, waypoints_m: np.ndarray) -> Corridor:
        """Resample an A* polyline at ``seed_spacing`` and box each free sample."""
        wp = np.asarray(waypoints_m, dtype=float)
        if len(wp) < 2:
            raise ValueError("SFC needs at least two waypoints")
        d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))))
        count = max(2, int(d[-1] / self.seed_spacing) + 1)
        targets = np.linspace(0.0, d[-1], count)
        seeds = np.column_stack([np.interp(targets, d, wp[:, k]) for k in range(3)])
        return self.boxes_for_points(seeds)

    @staticmethod
    def assign_boxes(corridor: Corridor, points_m: np.ndarray) -> np.ndarray:
        """Index of a box containing each point (else nearest box centre)."""
        pts = np.atleast_2d(np.asarray(points_m, float))
        lo, hi = corridor.boxes_min, corridor.boxes_max
        out = np.empty(len(pts), dtype=int)
        centres = 0.5 * (lo + hi)
        for k, p in enumerate(pts):
            inside = np.where(np.all((p >= lo) & (p <= hi), axis=1))[0]
            out[k] = inside[0] if inside.size else int(
                np.argmin(np.linalg.norm(centres - p, axis=1)))
        return out
