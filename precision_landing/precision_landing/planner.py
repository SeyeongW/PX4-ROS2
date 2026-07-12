"""A*/grid search + Safe Flight Corridor (SFC) generation -- front-end/back-end
for the MPC upgrade in the dynamic-path-planning roadmap (see research_direction
memory). This module is intentionally standalone and dependency-free (stdlib +
yaml only): it does NOT touch ROS or the MPC/QP solver. It exists to get the
front-end (A*) and back-end (corridor inflation) geometrically right and
independently testable BEFORE wiring a solver on top -- see
tools/plan_demo.py for an offline visual check against obstacle_map.yaml.

Pipeline (matches the roadmap note in research_direction):
  1. astar()            -- coarse, collision-free grid path start -> goal.
  2. simplify_path()     -- string-pull the jagged grid path down to a few
                             line-of-sight waypoints (corridors are generated
                             per waypoint, so fewer/straighter segments makes
                             for cleaner, more overlapping boxes).
  3. build_corridor()    -- inflate an axis-aligned box at each waypoint (and
                             evenly-spaced points along each segment, so
                             consecutive boxes overlap) as large as possible
                             without touching an (inflated) obstacle. This
                             convex box CHAIN is what turns the MPC's
                             obstacle-avoidance constraint from non-convex
                             (raw distance-to-obstacle) into a stack of linear
                             box constraints solvable as a QP.

Obstacles come from gazebo/config/obstacle_map.yaml (Method A: known map),
same source apf.py reads -- but here as axis-aligned boxes (their real
footprint), not apf.py's bounding-circle simplification, since box-vs-box
occupancy/inflation is both cheaper and a tighter fit for a grid search.
"""
import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml


@dataclass
class Box:
    e_min: float
    e_max: float
    n_min: float
    n_max: float

    def inflated(self, margin: float) -> "Box":
        return Box(self.e_min - margin, self.e_max + margin,
                   self.n_min - margin, self.n_max + margin)

    def contains(self, e: float, n: float) -> bool:
        return self.e_min <= e <= self.e_max and self.n_min <= n <= self.n_max

    def intersects(self, other: "Box") -> bool:
        return (self.e_min < other.e_max and self.e_max > other.e_min and
                self.n_min < other.n_max and self.n_max > other.n_min)


def load_obstacle_boxes(path: str) -> List[Box]:
    """Obstacles as their real axis-aligned footprint (not apf.py's bounding
    circle) -- a tighter, more natural fit for grid occupancy / box inflation."""
    with open(path) as f:
        data = yaml.safe_load(f)
    fw, fd = data['footprint']
    boxes = []
    for o in data['obstacles']:
        # Per-obstacle footprint override -- needed for maps like
        # obstacle_map_city.yaml where each building has its own real
        # footprint instead of one size shared by every obstacle.
        ow, od = o.get('footprint', (fw, fd))
        hw, hd = ow / 2.0, od / 2.0
        boxes.append(Box(o['e'] - hw, o['e'] + hw, o['n'] - hd, o['n'] + hd))
    return boxes


# --- A* front-end -------------------------------------------------------
def _blocked(e: float, n: float, inflated_obs: List[Box]) -> bool:
    return any(b.contains(e, n) for b in inflated_obs)


def _segment_blocked(a: Tuple[float, float], b: Tuple[float, float],
                      inflated_obs: List[Box], step: float = 0.5) -> bool:
    """Sampled line-of-sight check (simple and robust enough at these scales;
    a proper line-AABB intersection would be exact but this is plenty for
    metre-scale obstacles/grids)."""
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n_steps = max(1, int(dist / step))
    for i in range(n_steps + 1):
        t = i / n_steps
        e = a[0] + t * (b[0] - a[0])
        n = a[1] + t * (b[1] - a[1])
        if _blocked(e, n, inflated_obs):
            return True
    return False


def astar(start: Tuple[float, float], goal: Tuple[float, float],
          obstacles: List[Box], bounds: Box, grid_res: float = 1.0,
          safety_margin: float = 1.5) -> Optional[List[Tuple[float, float]]]:
    """8-connected grid A* on the (E,N) plane. Returns a list of (e,n)
    waypoints from start to goal, or None if no path exists (goal/start
    unreachable, or one of them is itself inside an inflated obstacle).
    safety_margin is added on top of each obstacle's real footprint --
    keep it >= drone half-width + a little slack so the corridor step later
    has room to place a box the drone can actually fit through."""
    inflated = [o.inflated(safety_margin) for o in obstacles]
    if _blocked(*start, inflated) or _blocked(*goal, inflated):
        return None

    def to_cell(p):
        return (round((p[0] - bounds.e_min) / grid_res),
                round((p[1] - bounds.n_min) / grid_res))

    def to_world(c):
        return (bounds.e_min + c[0] * grid_res, bounds.n_min + c[1] * grid_res)

    start_c, goal_c = to_cell(start), to_cell(goal)
    n_e = int((bounds.e_max - bounds.e_min) / grid_res) + 1
    n_n = int((bounds.n_max - bounds.n_min) / grid_res) + 1

    def in_bounds(c):
        return 0 <= c[0] < n_e and 0 <= c[1] < n_n

    def octile(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return grid_res * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))

    neighbours = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if (dx, dy) != (0, 0)]

    open_set = [(0.0, start_c)]
    came_from = {}
    g_score = {start_c: 0.0}
    visited = set()
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal_c:
            path = [cur]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return [to_world(c) for c in reversed(path)]
        for dx, dy in neighbours:
            nb = (cur[0] + dx, cur[1] + dy)
            if not in_bounds(nb) or nb in visited:
                continue
            if _blocked(*to_world(nb), inflated):
                continue
            step_cost = grid_res * (math.sqrt(2) if dx and dy else 1.0)
            tentative = g_score[cur] + step_cost
            if tentative < g_score.get(nb, math.inf):
                g_score[nb] = tentative
                came_from[nb] = cur
                heapq.heappush(open_set, (tentative + octile(nb, goal_c), nb))
    return None


def simplify_path(path: List[Tuple[float, float]], obstacles: List[Box],
                  safety_margin: float = 1.5) -> List[Tuple[float, float]]:
    """String-pull: from each kept point, jump to the FARTHEST point with a
    clear line of sight, dropping everything in between. Turns a jagged
    grid-cell path into a few straight, corridor-friendly segments."""
    if len(path) <= 2:
        return list(path)
    inflated = [o.inflated(safety_margin) for o in obstacles]
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and _segment_blocked(path[i], path[j], inflated):
            j -= 1
        out.append(path[j])
        i = j
    return out


# --- Safe Flight Corridor back-end --------------------------------------
def _inflate_box(seed: Tuple[float, float], obstacles: List[Box], bounds: Box,
                 step: float = 0.5, max_iter: int = 500) -> Box:
    """Grow an axis-aligned box outward from `seed` one side at a time (in
    `step` increments) until every side is blocked by an obstacle or the
    world bounds. Classic greedy SFC box inflation (Liu Sikang-style):
    cheap, and good enough at these obstacle scales/spacing -- an optimal
    (maximum-area) inflation isn't needed, just "big enough to give MPC
    room and to overlap the neighbouring boxes."""
    e, n = seed
    box = Box(e, e, n, n)
    sides = ['e_max', 'e_min', 'n_max', 'n_min']
    blocked = {s: False for s in sides}
    it = 0
    while not all(blocked.values()) and it < max_iter:
        it += 1
        for side in sides:
            if blocked[side]:
                continue
            trial = Box(box.e_min, box.e_max, box.n_min, box.n_max)
            if side == 'e_max':
                trial.e_max += step
            elif side == 'e_min':
                trial.e_min -= step
            elif side == 'n_max':
                trial.n_max += step
            else:
                trial.n_min -= step
            hits_bounds = (trial.e_min < bounds.e_min or trial.e_max > bounds.e_max or
                          trial.n_min < bounds.n_min or trial.n_max > bounds.n_max)
            hits_obstacle = any(trial.intersects(o) for o in obstacles)
            if hits_bounds or hits_obstacle:
                blocked[side] = True
            else:
                box = trial
    return box


def build_corridor(waypoints: List[Tuple[float, float]], obstacles: List[Box],
                   bounds: Box, safety_margin: float = 1.5,
                   seed_spacing: float = 3.0) -> List[Box]:
    """Inflate one box per waypoint AND per evenly-spaced point along each
    segment (seed_spacing apart) so consecutive boxes always overlap --
    a corridor with gaps is useless as a chain of convex constraints. Boxes
    are inflated against obstacles already padded by safety_margin, same
    margin astar() used, so the corridor is consistent with the path that
    was actually searched for."""
    inflated = [o.inflated(safety_margin) for o in obstacles]
    seeds: List[Tuple[float, float]] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n_steps = max(1, int(dist / seed_spacing))
        for i in range(n_steps):
            t = i / n_steps
            seeds.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    seeds.append(waypoints[-1])
    return [_inflate_box(s, inflated, bounds) for s in seeds]


def plan(start: Tuple[float, float], goal: Tuple[float, float],
        obstacle_map_path: str, bounds: Optional[Box] = None,
        grid_res: float = 1.0, safety_margin: float = 1.5,
        seed_spacing: float = 3.0):
    """High-level entry point: obstacle_map.yaml + start/goal -> (raw grid
    path, simplified waypoints, corridor box chain). None-path if unreachable.
    bounds defaults to a generous box around every obstacle plus start/goal."""
    obstacles = load_obstacle_boxes(obstacle_map_path)
    if bounds is None:
        es = [o.e_min for o in obstacles] + [o.e_max for o in obstacles] + \
            [start[0], goal[0]]
        ns = [o.n_min for o in obstacles] + [o.n_max for o in obstacles] + \
            [start[1], goal[1]]
        pad = 10.0
        bounds = Box(min(es) - pad, max(es) + pad, min(ns) - pad, max(ns) + pad)
    raw = astar(start, goal, obstacles, bounds, grid_res, safety_margin)
    if raw is None:
        return None, None, None
    simplified = simplify_path(raw, obstacles, safety_margin)
    corridor = build_corridor(simplified, obstacles, bounds, safety_margin,
                              seed_spacing)
    return raw, simplified, corridor
