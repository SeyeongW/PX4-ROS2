#!/usr/bin/env python3
"""Generate gazebo/config/obstacle_map_{city,mountain}.yaml from the two
self-contained maps pulled from the jo branch (2026-07-12), following the
same Method A ("known map") approach as gazebo/config/obstacle_map.yaml --
see that file's header comment.

Unlike obstacle_field.sdf's 16 uniform boxes, city buildings and mountain
trees/walls vary wildly in footprint and height, so every obstacle here uses
the per-obstacle 'footprint'/'height' override (apf.py/planner.py load these
if present, falling back to the top-level default otherwise).

Sources (ground truth, not re-derived/eyeballed):
  city     -- gazebo/validation/city/building_height_scaling.csv, which
              already has each building's exact local_min/max_x/y footprint
              and new_above_ground_height_m (post height-pass). The
              buildings.dae model has pose "0 0 0 0 0 0" in city_map.world
              (checked in the world file), so local x/y == world E/N with no
              extra transform.
  mountain -- gazebo/models/mountain_forest/model.sdf's 288 baked tree-trunk
              <collision> cylinders -- the compiled, ground-truth geometry
              actually used by the physics engine, not the
              source/*.source.xml layout (which lacks trunk radius). The
              72-wall central maze this model originally also contained was
              removed (2026-07-13, a UGV-navigation structure that didn't fit
              this drone capstone's obstacle set) -- the box-collision branch
              below is now dead code kept only in case a future obstacle
              isn't circular.

Usage:
  python3 gazebo/tools/extract_obstacle_map.py city
  python3 gazebo/tools/extract_obstacle_map.py mountain
  python3 gazebo/tools/extract_obstacle_map.py all
"""
import csv
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CITY_CSV = REPO / "gazebo/validation/city/building_height_scaling.csv"
FOREST_SDF = REPO / "gazebo/models/mountain_forest/model.sdf"
OUT_CITY = REPO / "gazebo/config/obstacle_map_city.yaml"
OUT_MOUNTAIN = REPO / "gazebo/config/obstacle_map_mountain.yaml"

HEADER = """# Known obstacle map for gazebo/worlds/{world}.
#
# Method A ("known map", see gazebo/config/obstacle_map.yaml for the original
# obstacle_field.sdf version of this approach) -- extracted directly from
# {source} by gazebo/tools/extract_obstacle_map.py, NOT hand-authored. If the
# map geometry changes, re-run that script rather than hand-editing this file.
#
# Unlike obstacle_map.yaml's 16 uniform boxes, every obstacle here carries
# its OWN 'footprint'/'height' (apf.py/planner.py use these per-obstacle
# values when present, falling back to the top-level default below only for
# obstacles that omit them -- none do here).
#
# Frame: local ENU metres, origin = world origin (0, 0) -- {frame_note}

frame_id: map
footprint: [{default_fw:.2f}, {default_fd:.2f}]   # fallback only; every obstacle below overrides this
height: {default_h:.2f}                            # fallback only; every obstacle below overrides this

obstacles:
"""


def extract_city():
    rows = list(csv.DictReader(open(CITY_CSV)))
    obstacles = []
    for row in rows:
        e_min = float(row["local_min_x"])
        e_max = float(row["local_max_x"])
        n_min = float(row["local_min_y"])
        n_max = float(row["local_max_y"])
        e = (e_min + e_max) / 2.0
        n = (n_min + n_max) / 2.0
        fw = e_max - e_min
        fd = n_max - n_min
        height = float(row["new_above_ground_height_m"])
        name = f"building_{row['output_component_ordinal']}"
        obstacles.append((name, e, n, fw, fd, height))
    return obstacles


COLLISION_RE = re.compile(
    r'<collision name="([^"]+)">\s*'
    r'<pose>([\-\d.]+) ([\-\d.]+) ([\-\d.]+) [\-\d.]+ [\-\d.]+ ([\-\d.]+)</pose>\s*'
    r'<geometry>\s*'
    r'(?:<cylinder>\s*<radius>([\-\d.]+)</radius>\s*<length>([\-\d.]+)</length>\s*</cylinder>'
    r'|<box>\s*<size>([\-\d.]+) ([\-\d.]+) ([\-\d.]+)</size>\s*</box>)',
    re.MULTILINE,
)


def extract_mountain():
    text = FOREST_SDF.read_text()
    obstacles = []
    for m in COLLISION_RE.finditer(text):
        name, e, n, _z, yaw = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        radius, length, sx, sy, sz = m.group(6), m.group(7), m.group(8), m.group(9), m.group(10)
        if radius is not None:
            # Tree trunk: circular footprint, so orientation doesn't matter.
            r = float(radius)
            fw = fd = 2.0 * r
            height = float(length)
        else:
            # Dead branch since the maze was removed (see module docstring) --
            # no box collisions remain in mountain_forest/model.sdf. Kept for
            # any future non-circular obstacle: rotates its (sx, sy) footprint
            # by yaw into an axis-aligned E/N bounding box, exact for
            # multiples of 90 degrees and a safe over-approximation otherwise.
            sx, sy, sz = float(sx), float(sy), float(sz)
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            fw = sx * c + sy * s
            fd = sx * s + sy * c
            height = sz
        obstacles.append((name.replace("_collision", ""), e, n, fw, fd, height))
    return obstacles


def write_yaml(path, world, source, frame_note, obstacles):
    fws = [o[3] for o in obstacles]
    fds = [o[4] for o in obstacles]
    hs = [o[5] for o in obstacles]
    default_fw, default_fd, default_h = (
        sum(fws) / len(fws), sum(fds) / len(fds), sum(hs) / len(hs)
    )
    with open(path, "w") as f:
        f.write(HEADER.format(world=world, source=source, frame_note=frame_note,
                              default_fw=default_fw, default_fd=default_fd, default_h=default_h))
        for name, e, n, fw, fd, height in obstacles:
            f.write(
                f"  - {{name: {name}, e: {e:.3f}, n: {n:.3f}, "
                f"footprint: [{fw:.3f}, {fd:.3f}], height: {height:.3f}}}\n"
            )
    print(f"wrote {len(obstacles)} obstacles -> {path}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("city", "all"):
        write_yaml(
            OUT_CITY, "city_map/city_map.world",
            "gazebo/validation/city/building_height_scaling.csv",
            "matches city_map.world's buildings.dae model pose (0 0 0), so local mesh x/y == world E/N directly",
            extract_city(),
        )
    if which in ("mountain", "all"):
        write_yaml(
            OUT_MOUNTAIN, "mountain_map.world",
            "gazebo/models/mountain_forest/model.sdf (baked collisions)",
            "mountain terrain is NOT flat (0..40m) so 'height' here is each obstacle's own vertical "
            "extent (trunk collision length / wall box height), not a height above a flat ground plane",
            extract_mountain(),
        )


if __name__ == "__main__":
    main()
