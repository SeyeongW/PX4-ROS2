#!/usr/bin/env python3
"""Generate gazebo/config/trailer_route_city.yaml: a varied, corner-to-corner
loop for the trailer's moving_marker_node pattern='route' driver, guaranteed
obstacle-free by construction.

Rather than hand-placing points and hoping they miss all 274 buildings, this:
  1. finds a genuinely OPEN spot (maximum clearance from every building, not
     just barely-free) near each rough compass target spread across the city,
  2. reuses precision_landing/planner.py's own A* (the SAME algorithm already
     validated for the drone) to connect each consecutive pair of open spots,
     and concatenates the resulting string-pulled waypoint chains,
  3. re-verifies every final stitched segment at fine sampling resolution
     against the trailer's real hard-safety floor before writing anything out.

Every leg is therefore a real, obstacle-free route through the actual street
network, not just a straight line that happens to miss buildings.
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "precision_landing/precision_landing"))
import planner  # noqa: E402

OBSTACLE_MAP = REPO / "gazebo/config/obstacle_map_city.yaml"
OUT = REPO / "gazebo/config/trailer_route_city.yaml"
# Trailer is 5x5m (gazebo/models/moving_platform_aruco/model.sdf -- a 4x4
# attempt on 2026-07-13 tipped repeatedly under real driving even on flat
# ground, reverted), so 2.5m (half-width) is the hard physical floor -- below
# that the inflated box no longer even covers the trailer's own footprint.
# 3.0 is the preferred search/verification margin (a bit of real buffer);
# individual A* legs may back off toward the floor for a tight street gap
# rather than failing outright (see the retry loop in main()), but the FINAL
# written route is always re-checked against the hard floor with fine
# sampling before writing.
SAFETY_MARGIN = 3.0
MIN_SAFETY_MARGIN = 2.5
GRID_RES = 1.0
FINAL_CHECK_STEP = 0.1   # m, line-of-sight sampling for the final safety re-check
# A* string-pulling (planner.simplify_path) can leave two consecutive
# waypoints very close together (a minor direction nudge on the searched
# path) -- fine for the drone's own MPC, but for the trailer's closed-loop
# P-servo (cross_arrive_radius=1.0m) a leg shorter than this gives it no room
# to actually decelerate into the arrival and re-accelerate into the next
# leg before both overlap. Found empirically (2026-07-13, real driving test):
# a 1.4m-spaced waypoint pair tipped the trailer right at that transition,
# even though BOTH individual segments were independently obstacle-free --
# the tip was a control/dynamics problem, not a safety-margin one.
MIN_LEG_LENGTH = 4.0
# city_map.world's terrain heightmap is 500x500m centered at world origin
# (checked in the world file), so E/N in [-250, 250] is the actual playable
# map; stay a bit inside that or find_open_spot's clearance search "escapes"
# off the edge of the terrain (unmapped space beyond the buildings has no
# obstacles at all and looks maximally open) -- see generate_patrol_route.py,
# which hit exactly this bug before WORLD_BOUNDS was added there too.
WORLD_BOUNDS = planner.Box(-230.0, 230.0, -230.0, 230.0)

# Rough compass-direction targets spread across the city (bounds ~
# E[-236,232] N[-230,239], see gazebo/tools/extract_obstacle_map.py output)
# -- loops through all four quadrants plus the known-clear drone-spawn area,
# "구석구석" (corner to corner) rather than a small local patch. Each is just
# an AIM POINT for find_open_spot() below, not used directly.
ROUGH_TARGETS = [
    (-120.0, 115.0),   # start: existing drone_spawn_pad area (already verified clear)
    (150.0, 180.0),    # NE
    (190.0, -30.0),    # E
    (90.0, -190.0),    # SE
    (-70.0, -200.0),   # S
    (-210.0, -110.0),  # SW
    (-175.0, 140.0),   # W (scanned for real open space -- (-190,90) was in a tight pocket)
    (0.0, 200.0),      # N
    (-120.0, 115.0),   # back to start (closes the loop)
]


def clearance(e, n, obstacles):
    """Distance from (e, n) to the nearest obstacle's real (uninflated) edge
    (0 if inside one)."""
    best = float("inf")
    for b in obstacles:
        de = max(b.e_min - e, 0.0, e - b.e_max)
        dn = max(b.n_min - n, 0.0, n - b.n_max)
        best = min(best, math.hypot(de, dn))
    return best


def find_open_spot(target, obstacles, search_radius=40.0, grid=5.0):
    """Best-clearance point within search_radius of `target` -- a genuinely
    open spot, not just the nearest point that happens to clear the safety
    margin by a hair (which the first version of this script did, and it
    picked corners sitting in tight pockets that then needed the margin
    fallback on every connecting leg)."""
    if not WORLD_BOUNDS.contains(*target):
        raise ValueError(f"rough target {target} is already outside WORLD_BOUNDS")
    best_pt, best_c = target, clearance(*target, obstacles)
    r = grid
    while r <= search_radius:
        n_pts = max(8, int(2 * math.pi * r / grid))
        for i in range(n_pts):
            ang = 2 * math.pi * i / n_pts
            cand = (target[0] + r * math.cos(ang), target[1] + r * math.sin(ang))
            if not WORLD_BOUNDS.contains(*cand):
                continue   # don't "escape" off the edge of the map (see WORLD_BOUNDS comment)
            c = clearance(*cand, obstacles)
            if c > best_c:
                best_pt, best_c = cand, c
        r += grid
    return best_pt, best_c


# Real local terrain hazard, NOT a building: a ~0.10m ridge/depression around
# (-101.7, 136.7) found 2026-07-13 by dropping a 5x5 grid of free-rolling
# sphere probes there (most rolled in from several metres away to the same
# spot; the city_map_mission trailer visibly tipped every time it drove
# through this area, at a physical Z jump too large for gradual tipping --
# a wide flat-plate probe bridged straight over this because it's a single
# rigid body, but a moving 5x5m trailer's own corners don't; a small rolling
# sphere isn't rigid across the gap the way the plate is, so it reveals the
# real slope). No building sits anywhere near it (checked against
# obstacle_map_city.yaml), so it's excluded here the same way a building
# would be: as an inflated no-go box for A* to route around, with generous
# margin since the exact edge of the hazard wasn't mapped precisely.
TERRAIN_HAZARDS = [
    planner.Box(-108.0, -98.0, 133.0, 142.0),
    # Second hazard found the same way, 2026-07-13: the trailer tipped again
    # (after the first hazard fix let it drive much further) right around
    # here; probes in this pocket showed up to 1.3m drift. Less dramatic
    # than the first one and noisier to pin down exactly (a full systematic
    # sweep of the whole 2km route was started to find every such spot
    # methodically but didn't finish in one session -- see
    # sweep_terrain_hazards.py; this box is deliberately generous to cover
    # the uncertainty instead of precisely localizing it).
    planner.Box(-85.0, -68.0, 175.0, 190.0),
]


def write_map_with_hazards(obstacles, hazards, out_path):
    """planner.plan()/load_obstacle_boxes always reload obstacles from a file
    path (there's no in-memory-list entry point), so to make A* actually
    route around TERRAIN_HAZARDS (not just find_open_spot's clearance search,
    which does take the in-memory list) this writes a temp copy of
    obstacle_map_city.yaml with the hazard boxes appended as ordinary
    obstacle entries."""
    with open(out_path, "w") as f:
        f.write("frame_id: map\nfootprint: [1.0, 1.0]\nheight: 10.0\nobstacles:\n")
        for i, b in enumerate(obstacles):
            e, n = (b.e_min + b.e_max) / 2, (b.n_min + b.n_max) / 2
            fw, fd = b.e_max - b.e_min, b.n_max - b.n_min
            f.write(f"  - {{name: building_{i}, e: {e:.3f}, n: {n:.3f}, "
                   f"footprint: [{fw:.3f}, {fd:.3f}], height: 10.0}}\n")
        for i, b in enumerate(hazards):
            e, n = (b.e_min + b.e_max) / 2, (b.n_min + b.n_max) / 2
            fw, fd = b.e_max - b.e_min, b.n_max - b.n_min
            f.write(f"  - {{name: terrain_hazard_{i}, e: {e:.3f}, n: {n:.3f}, "
                   f"footprint: [{fw:.3f}, {fd:.3f}], height: 10.0}}\n")


def main():
    obstacles = planner.load_obstacle_boxes(str(OBSTACLE_MAP))
    obstacles_with_hazards = obstacles + TERRAIN_HAZARDS
    hazard_map_path = REPO / "gazebo/config/.trailer_route_planning_map.yaml"
    write_map_with_hazards(obstacles, TERRAIN_HAZARDS, hazard_map_path)
    corners = []
    for i, target in enumerate(ROUGH_TARGETS):
        # Start/end corner is pinned EXACTLY to drone_spawn_pad's location
        # (city_map.world), not relocated by the clearance search: the
        # trailer's world-file pose (see the mission-world generator) is
        # computed from that pad's already-known ground height, and moving
        # even a little would silently invalidate that math. 20.5m clearance
        # there is already plenty (checked separately), so there's no benefit
        # to relocating it anyway.
        if i in (0, len(ROUGH_TARGETS) - 1):
            corners.append(target)
            print(f"corner: {target} (pinned to drone_spawn_pad, not searched)")
            continue
        pt, c = find_open_spot(target, obstacles_with_hazards)
        corners.append(pt)
        tag = "" if math_dist(pt, target) < 0.1 else f" (target was {target})"
        print(f"corner: {pt} clearance={c:.1f}m{tag}")

    es = [o.e_min for o in obstacles_with_hazards] + [o.e_max for o in obstacles_with_hazards] + [c[0] for c in corners]
    ns = [o.n_min for o in obstacles_with_hazards] + [o.n_max for o in obstacles_with_hazards] + [c[1] for c in corners]
    pad = 10.0
    bounds = planner.Box(min(es) - pad, max(es) + pad, min(ns) - pad, max(ns) + pad)

    route = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        margin = SAFETY_MARGIN
        simplified = None
        while margin >= MIN_SAFETY_MARGIN - 1e-9:
            raw, simplified, _corridor = planner.plan(
                a, b, str(hazard_map_path), bounds=bounds,
                grid_res=GRID_RES, safety_margin=margin,
            )
            if simplified is not None:
                break
            margin -= 0.2
        if simplified is None:
            raise RuntimeError(
                f"no obstacle-free path between {a} and {b} even at the "
                f"{MIN_SAFETY_MARGIN}m floor (trailer half-width) -- pick different corners")
        if margin < SAFETY_MARGIN:
            print(f"  (tight gap: {a} -> {b} needed safety_margin={margin:.1f} instead of {SAFETY_MARGIN})")
        # simplified[0] == a (already the last point of `route`), so skip it.
        route.extend(simplified[1:])
        print(f"{a} -> {b}: {len(simplified)} waypoints (from {len(raw)} raw grid cells)")

    # De-duplicate consecutive near-identical points (can happen at corner
    # targets shared between two legs).
    deduped = [route[0]]
    for p in route[1:]:
        if math_dist(p, deduped[-1]) > 0.5:
            deduped.append(p)
    route = deduped

    inflated_floor = [o.inflated(MIN_SAFETY_MARGIN) for o in obstacles_with_hazards]

    # Merge waypoints closer together than MIN_LEG_LENGTH (see that constant's
    # comment): drop the later point and bridge straight from its predecessor
    # to its successor, but ONLY if that bridge is still obstacle-free at the
    # hard floor -- never merge into an unsafe shortcut. Repeats until no more
    # merges happen, since removing one point can leave its former neighbours
    # too close together in turn. Never touches route[0]/route[-1] (the pinned
    # drone_spawn_pad start/end).
    merged_count = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(route) - 2:
            if (math_dist(route[i], route[i + 1]) < MIN_LEG_LENGTH and
                    not planner._segment_blocked(route[i], route[i + 2], inflated_floor, step=FINAL_CHECK_STEP)):
                del route[i + 1]
                merged_count += 1
                changed = True
                continue
            i += 1
    if merged_count:
        print(f"merged {merged_count} waypoint(s) closer than {MIN_LEG_LENGTH}m to their neighbour")
    leg_lengths = [math_dist(a, b) for a, b in zip(route[:-1], route[1:])]
    still_short = [l for l in leg_lengths if l < MIN_LEG_LENGTH]
    if still_short:
        print(f"WARNING: {len(still_short)} leg(s) still under {MIN_LEG_LENGTH}m after merging "
             f"(couldn't bridge safely, e.g. a tight alley) -- shortest is {min(still_short):.2f}m")
    else:
        print(f"shortest leg after merging: {min(leg_lengths):.2f}m (all >= {MIN_LEG_LENGTH}m)")

    # Final safety gate: re-verify EVERY stitched segment at fine sampling
    # against the hard floor, independent of whatever margin each leg's A*
    # search happened to use. Coarse 0.5m sampling (astar/simplify_path's
    # default) can step over a thin building corner right at the margin
    # boundary -- this was caught empirically (2 segments failed a 0.1m
    # recheck after the first version of this script used 0.5m throughout),
    # so don't trust the search-time check alone for the file we actually ship.
    bad = [(a, b) for a, b in zip(route[:-1], route[1:])
          if planner._segment_blocked(a, b, inflated_floor, step=FINAL_CHECK_STEP)]
    if bad:
        raise RuntimeError(
            f"{len(bad)} segment(s) fail the final {MIN_SAFETY_MARGIN}m hard-floor "
            f"recheck (fine sampling) -- NOT writing an unsafe route: {bad}")
    print(f"final check: all {len(route) - 1} segments clear at "
         f"{MIN_SAFETY_MARGIN}m hard floor, {FINAL_CHECK_STEP}m sampling")

    with open(OUT, "w") as f:
        f.write(
            "# Trailer driving route for the city map -- generated by\n"
            "# gazebo/tools/generate_trailer_route.py, NOT hand-authored. Every leg is a\n"
            "# real A*-planned path through gazebo/config/obstacle_map_city.yaml's 274\n"
            f"# buildings (safety_margin>={MIN_SAFETY_MARGIN}m hard floor for the 5x5m\n"
            "# trailer, fine-sampled final recheck), so it's obstacle-free by construction --\n"
            "# re-run the generator rather than hand-editing this file if the city map or\n"
            "# trailer size changes.\n"
            "#\n"
            "# Consumed by moving_marker_node's pattern='route' driver (route_file param) --\n"
            "# closed-loop P-servo + accel-capped, loops back to the first waypoint.\n\n"
            "waypoints:\n"
        )
        for e, n in route:
            f.write(f"  - {{e: {e:.3f}, n: {n:.3f}}}\n")
    print(f"wrote {len(route)} waypoints -> {OUT}")
    hazard_map_path.unlink(missing_ok=True)


def math_dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


if __name__ == "__main__":
    main()
