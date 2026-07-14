"""A* — a **generic callback core** plus a 3-D grid adapter.

Design follows BigZaphod/AStar (https://github.com/BigZaphod/AStar): the search
core is domain-agnostic and knows nothing about grids or cities.  The caller
injects the problem through callbacks — exactly the ``ASPathNodeSource`` idea:

    BigZaphod (C)                 here (Python)
    ----------------------------  -------------------------------------------
    nodeNeighbors + edgeCost      neighbors(node) -> [(neighbor, edge_cost), ...]
    pathCostHeuristic             heuristic(node) -> float   (to goal)
    nodeComparator                Python hashable/`==` on the node object
    ASPathCreate                  a_star_search(...)

``a_star_search`` is the classic best-first search minimising

    f(n) = g(n) + h(n)

where ``g`` accumulates the injected ``edge_cost`` and ``h`` is the injected
heuristic.  The *cost function* is therefore fully pluggable.

The bundled :class:`AStarPlanner3D` adapter supplies a 26-connected grid whose
edge cost is not just distance but a **shaped cost** (see :meth:`_edge_cost`):

    cost(n→m) = step·(1 + w_clear·clearance_shortfall(m) + w_alt·|z_m − z_pref|)
                + w_climb·max(0, z_m − z_n)

so the route is short **and** keeps clear of walls, holds the preferred cruise
altitude, and avoids needless climbing.  All extra terms are ≥ 0, so the
straight-line heuristic stays admissible and A* stays optimal for this richer
objective.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .world_model import WorldModel

# 26-connected neighbour offsets (exclude the zero vector).
_NEIGHBORS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dz in (-1, 0, 1)
     if (dx, dy, dz) != (0, 0, 0)],
    dtype=int,
)
_NEIGHBOR_COST = np.linalg.norm(_NEIGHBORS, axis=1)  # 1, sqrt2 or sqrt3


@dataclass
class SearchResult:
    success: bool
    nodes: list                      # node path from start to goal (inclusive)
    cost: float
    expanded: int
    message: str = ""


def a_star_search(
    start,
    goal,
    neighbors: Callable[[object], Iterable[tuple[object, float]]],
    heuristic: Callable[[object], float],
    *,
    max_expanded: int = 400_000,
    weight: float = 1.0,
) -> SearchResult:
    """Domain-agnostic A*.  ``start``/``goal`` are any hashable, ``==``-comparable
    nodes; ``neighbors`` yields ``(node, edge_cost)``; ``heuristic`` estimates the
    remaining cost to ``goal`` (must be admissible for an optimal path).

    ``weight`` >= 1 runs *weighted A\\** (f = g + weight·h): the search is greedier
    toward the goal, expanding far fewer nodes for a large speed-up, at the cost of
    a path at most ``weight``× longer than optimal (bounded suboptimality)."""
    open_heap: list[tuple[float, int, object]] = []
    counter = 0
    heapq.heappush(open_heap, (weight * heuristic(start), counter, start))
    g_score = {start: 0.0}
    came_from: dict = {}
    closed: set = set()
    expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1
        if current == goal:
            return SearchResult(True, _reconstruct(came_from, current),
                                g_score[current], expanded, "ok")
        if expanded > max_expanded:
            return SearchResult(False, [], math.inf, expanded,
                                "expansion budget exhausted")
        for neighbor, edge_cost in neighbors(current):
            if neighbor in closed:
                continue
            tentative = g_score[current] + edge_cost
            if tentative < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                heapq.heappush(open_heap,
                               (tentative + weight * heuristic(neighbor), counter, neighbor))
    return SearchResult(False, [], math.inf, expanded, "no path found")


def _reconstruct(came_from: dict, node) -> list:
    path = [node]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


@dataclass
class AStarResult:
    success: bool
    waypoints_m: np.ndarray          # (K, 3) ENU path incl. start and goal
    expanded: int
    message: str = ""


class AStarPlanner3D:
    """Grid adapter that plugs a shaped 3-D cost into :func:`a_star_search`."""

    def __init__(self, world: WorldModel, resolution_m: float = 2.0,
                 max_expanded: int = 400_000,
                 *,
                 clearance_weight: float = 2.0,
                 clearance_pref_m: float = 10.0,
                 altitude_weight: float = 0.05,
                 altitude_pref_m: float | None = None,
                 climb_weight: float = 0.5,
                 heuristic_weight: float = 1.0):
        self.world = world
        self.res = float(resolution_m)
        self.origin = world.bounds_min.astype(float)
        self.max_expanded = int(max_expanded)
        self.h_weight = float(heuristic_weight)      # weighted A* (>=1 = faster)
        self.w_clear = float(clearance_weight)
        self.clear_pref = float(clearance_pref_m)
        self.w_alt = float(altitude_weight)
        # default preferred altitude = middle of the vertical geofence band
        self.z_pref = float(altitude_pref_m if altitude_pref_m is not None
                            else 0.5 * (world.bounds_min[2] + world.bounds_max[2]))
        self.w_climb = float(climb_weight)

    # --------------------------------------------------------- grid <-> world
    def _to_cell(self, p) -> tuple[int, int, int]:
        idx = np.round((np.asarray(p, float) - self.origin) / self.res).astype(int)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def _to_world(self, cell) -> np.ndarray:
        return self.origin + np.asarray(cell, float) * self.res

    def _cell_free(self, cell) -> bool:
        return bool(self.world.is_free(self._to_world(cell))[0])

    # ------------------------------------------------------------------ search
    def plan(self, start_m, goal_m) -> AStarResult:
        start = self._to_cell(start_m)
        goal = self._to_cell(goal_m)
        goal_w = self._to_world(goal)
        if not self._cell_free(start):
            return AStarResult(False, np.empty((0, 3)), 0, "start cell blocked")
        if not self._cell_free(goal):
            return AStarResult(False, np.empty((0, 3)), 0, "goal cell blocked")

        clear_cache: dict = {}

        def clearance(cell) -> float:
            value = clear_cache.get(cell)
            if value is None:
                value = self.world.clearance(self._to_world(cell))
                clear_cache[cell] = value
            return value

        def heuristic(cell) -> float:
            return float(np.linalg.norm(self._to_world(cell) - goal_w))

        def neighbors(cell):
            base = np.asarray(cell, int)
            cw = self._to_world(cell)
            out = []
            for offset, step_cells in zip(_NEIGHBORS, _NEIGHBOR_COST):
                nb = tuple(base + offset)
                if not self._cell_free(nb):
                    continue
                out.append((nb, self._edge_cost(cw, nb, clearance, step_cells)))
            return out

        result = a_star_search(start, goal, neighbors, heuristic,
                               max_expanded=self.max_expanded, weight=self.h_weight)
        if not result.success:
            return AStarResult(False, np.empty((0, 3)), result.expanded, result.message)
        path = np.array([self._to_world(c) for c in result.nodes], dtype=float)
        return AStarResult(True, self.shortcut(path), result.expanded, "ok")

    def plan_through(self, points) -> AStarResult:
        """Plan A* through an ordered list of via-points (start … goal).

        Each consecutive pair is planned independently and the results are
        concatenated (dropping the shared joint), so the returned path is
        *forced* to pass through every via-point.  Use this to shape a route —
        e.g. a slalom through tight gaps — that a single start→goal search would
        otherwise straighten out.
        """
        pts = [np.asarray(p, float) for p in points]
        if len(pts) < 2:
            raise ValueError("plan_through needs at least two points")
        path = None
        expanded = 0
        for a, b in zip(pts[:-1], pts[1:]):
            leg = self.plan(a, b)
            expanded += leg.expanded
            if not leg.success:
                return AStarResult(False, np.empty((0, 3)), expanded,
                                   f"leg {np.round(a,1)}->{np.round(b,1)}: {leg.message}")
            path = leg.waypoints_m if path is None else np.vstack(
                (path, leg.waypoints_m[1:]))
        return AStarResult(True, path, expanded, "ok")

    def _edge_cost(self, from_world, to_cell, clearance, step_cells) -> float:
        """Shaped edge cost (the pluggable 'reward' function)."""
        tw = self._to_world(to_cell)
        step = step_cells * self.res                      # base geometric length
        shortfall = max(0.0, self.clear_pref - clearance(to_cell))  # near-wall penalty
        alt_dev = abs(tw[2] - self.z_pref)                # off-band altitude penalty
        climb = max(0.0, tw[2] - from_world[2])           # upward-move penalty
        return (step * (1.0 + self.w_clear * shortfall + self.w_alt * alt_dev)
                + self.w_climb * climb)

    def shortcut(self, path: np.ndarray) -> np.ndarray:
        """Greedy string-pulling: drop waypoints a straight segment can skip."""
        if len(path) <= 2:
            return path
        kept = [path[0]]
        anchor = 0
        for probe in range(2, len(path)):
            if not self.world.segment_is_free(path[anchor], path[probe],
                                               step_m=0.5 * self.res):
                kept.append(path[probe - 1])
                anchor = probe - 1
        kept.append(path[-1])
        return np.asarray(kept, dtype=float)
