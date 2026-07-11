#!/usr/bin/env python3
"""A*+corridor offline visual check -- renders precision_landing.planner's
output (raw grid path, simplified waypoints, Safe Flight Corridor box chain)
against gazebo/config/obstacle_map.yaml as a self-contained HTML/SVG page.

Purely a dev/verification tool (no ROS, no ament dependency) -- run it any
time obstacle_map.yaml, planner.py, or the safety_margin/grid_res/seed_spacing
tuning changes, to eyeball that the front-end (A*) and back-end (corridor)
are still geometrically sane BEFORE wiring the MPC/QP layer on top.

Usage:
    python3 gazebo/plan_demo.py [--start E N] [--goal E N] [--out FILE]
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'precision_landing'))
from precision_landing import planner  # noqa: E402

import yaml  # noqa: E402


def _dedupe_boxes(boxes, eps=0.3):
    """Corridor seeds every `seed_spacing` metres along a straight segment
    produce near-identical boxes -- collapse consecutive near-duplicates so
    the picture doesn't just show the same rectangle drawn a dozen times."""
    out = []
    for b in boxes:
        if out:
            p = out[-1]
            if (abs(b.e_min - p.e_min) < eps and abs(b.e_max - p.e_max) < eps and
                    abs(b.n_min - p.n_min) < eps and abs(b.n_max - p.n_max) < eps):
                continue
        out.append(b)
    return out


def build_html(start, goal, obstacle_map_path, grid_res=1.0, safety_margin=1.5,
               seed_spacing=3.0):
    obs_yaml = yaml.safe_load(open(obstacle_map_path))
    names = [o['name'] for o in obs_yaml['obstacles']]
    obs_boxes = planner.load_obstacle_boxes(obstacle_map_path)

    raw, simplified, corridor = planner.plan(
        start, goal, obstacle_map_path, grid_res=grid_res,
        safety_margin=safety_margin, seed_spacing=seed_spacing)
    if raw is None:
        raise SystemExit(f'no path found from {start} to {goal} '
                          f'(one of them may be inside the {safety_margin} m '
                          'obstacle safety margin -- try a different point)')
    corridor = _dedupe_boxes(corridor)

    # --- world -> SVG bounds (SVG y grows down, so N is flipped: y = -n) ---
    all_e = [b.e_min for b in obs_boxes] + [b.e_max for b in obs_boxes] + \
        [p[0] for p in raw] + [start[0], goal[0]]
    all_n = [b.n_min for b in obs_boxes] + [b.n_max for b in obs_boxes] + \
        [p[1] for p in raw] + [start[1], goal[1]]
    pad = 5.0
    e_min, e_max = min(all_e) - pad, max(all_e) + pad
    n_min, n_max = min(all_n) - pad, max(all_n) + pad
    vb_w, vb_h = e_max - e_min, n_max - n_min
    view_box = f'{e_min:.1f} {-n_max:.1f} {vb_w:.1f} {vb_h:.1f}'

    def pt(p):
        return f'{p[0]:.2f},{-p[1]:.2f}'

    # --- grid lines every 10 m ---
    grid_lines = []
    ge = 10 * (int(e_min) // 10)
    while ge <= e_max:
        grid_lines.append(f'<line class="grid" x1="{ge}" y1="{-n_min:.1f}" '
                          f'x2="{ge}" y2="{-n_max:.1f}"/>')
        ge += 10
    gn = 10 * (int(n_min) // 10)
    while gn <= n_max:
        grid_lines.append(f'<line class="grid" x1="{e_min:.1f}" y1="{-gn}" '
                          f'x2="{e_max:.1f}" y2="{-gn}"/>')
        gn += 10

    obstacle_svg = []
    for name, b in zip(names, obs_boxes):
        w, h = b.e_max - b.e_min, b.n_max - b.n_min
        cx, cy = (b.e_min + b.e_max) / 2, -(b.n_min + b.n_max) / 2
        short = name.replace('obstacle_', '')
        obstacle_svg.append(
            f'<g class="obstacle"><rect x="{b.e_min:.1f}" y="{-b.n_max:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}"/>'
            f'<text x="{cx:.1f}" y="{cy:.1f}">{short}</text></g>')

    corridor_svg = []
    for i, b in enumerate(corridor):
        w, h = b.e_max - b.e_min, b.n_max - b.n_min
        corridor_svg.append(
            f'<rect class="corridor" x="{b.e_min:.2f}" y="{-b.n_max:.2f}" '
            f'width="{w:.2f}" height="{h:.2f}" '
            f'data-info="corridor {i+1}/{len(corridor)}  '
            f'E [{b.e_min:.1f}, {b.e_max:.1f}]  N [{b.n_min:.1f}, {b.n_max:.1f}]  '
            f'({w:.1f} x {h:.1f} m)" tabindex="0"/>')

    raw_pts = ' '.join(pt(p) for p in raw)
    simp_pts = ' '.join(pt(p) for p in simplified)

    waypoint_rows = '\n'.join(
        f'<tr><td>{i}</td><td>{p[0]:.1f}</td><td>{p[1]:.1f}</td></tr>'
        for i, p in enumerate(simplified))
    corridor_rows = '\n'.join(
        f'<tr><td>{i+1}</td><td>[{b.e_min:.1f}, {b.e_max:.1f}]</td>'
        f'<td>[{b.n_min:.1f}, {b.n_max:.1f}]</td>'
        f'<td>{b.e_max-b.e_min:.1f} &times; {b.n_max-b.n_min:.1f}</td></tr>'
        for i, b in enumerate(corridor))

    return TEMPLATE.format(
        view_box=view_box,
        grid_lines='\n'.join(grid_lines),
        obstacles='\n'.join(obstacle_svg),
        corridor='\n'.join(corridor_svg),
        raw_pts=raw_pts,
        simp_pts=simp_pts,
        start_x=start[0], start_y=-start[1],
        goal_x=goal[0], goal_y=-goal[1],
        n_obstacles=len(obs_boxes),
        n_raw=len(raw),
        n_simplified=len(simplified),
        n_corridor=len(corridor),
        grid_res=grid_res, safety_margin=safety_margin, seed_spacing=seed_spacing,
        start_e=start[0], start_n=start[1], goal_e=goal[0], goal_n=goal[1],
        waypoint_rows=waypoint_rows, corridor_rows=corridor_rows,
    )


TEMPLATE = r"""<title>A* + Safe Flight Corridor -- obstacle_field</title>
<style>
:root {{
  --bg: #0f1417; --panel: #161d22; --panel-2: #1c242a; --line: #2a343b;
  --ink: #e8edf0; --muted: #7c8b93; --faint: #445059;
  --obstacle-fill: #3a4750; --obstacle-line: #5b6a73;
  --path: #f2a13c; --corridor: #4fb0c9; --corridor-hover: #7ecfe3;
  --start: #6fcf7a; --goal: #e8577c;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}
:root[data-theme="light"] {{
  --bg: #f3f5f6; --panel: #ffffff; --panel-2: #eef1f2; --line: #d7dde0;
  --ink: #1b232a; --muted: #5c6b73; --faint: #b6c0c5;
  --obstacle-fill: #d8dee1; --obstacle-line: #aab5ba;
  --path: #b5720a; --corridor: #1f8aa3; --corridor-hover: #14647a;
  --start: #2f9e46; --goal: #c23458;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg: #f3f5f6; --panel: #ffffff; --panel-2: #eef1f2; --line: #d7dde0;
    --ink: #1b232a; --muted: #5c6b73; --faint: #b6c0c5;
    --obstacle-fill: #d8dee1; --obstacle-line: #aab5ba;
    --path: #b5720a; --corridor: #1f8aa3; --corridor-hover: #14647a;
    --start: #2f9e46; --goal: #c23458;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--ink); font-family: var(--sans);
  margin: 0; padding: 2rem clamp(1rem, 4vw, 3rem) 4rem;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; }}
header {{ margin-bottom: 1.75rem; }}
.eyebrow {{
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--corridor); margin: 0 0 0.5rem;
}}
h1 {{
  font-size: clamp(1.5rem, 2.6vw, 2.1rem); margin: 0 0 0.4rem;
  font-weight: 650; letter-spacing: -0.01em; text-wrap: balance;
}}
.sub {{ color: var(--muted); font-size: 0.95rem; max-width: 62ch; line-height: 1.5; margin: 0; }}
.sub code {{ font-family: var(--mono); font-size: 0.88em; color: var(--ink); }}

.layout {{ display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(260px, 1fr); gap: 1.25rem; }}
@media (max-width: 860px) {{ .layout {{ grid-template-columns: 1fr; }} }}

.panel {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.2rem;
}}
.panel + .panel {{ margin-top: 1.25rem; }}

.canvas-panel {{ padding: 0.9rem; display: flex; flex-direction: column; gap: 0.7rem; }}
.canvas-head {{ display: flex; justify-content: space-between; align-items: baseline; padding: 0 0.3rem; }}
.canvas-head h2 {{ font-size: 0.95rem; font-weight: 600; margin: 0; }}
.compass {{ font-family: var(--mono); font-size: 0.72rem; color: var(--muted); }}

svg.field {{ width: 100%; height: auto; display: block; border-radius: 6px; background: var(--panel-2); }}
.grid {{ stroke: var(--line); stroke-width: 1; vector-effect: non-scaling-stroke; }}
.obstacle rect {{ fill: var(--obstacle-fill); stroke: var(--obstacle-line); stroke-width: 1.2; vector-effect: non-scaling-stroke; }}
.obstacle text {{
  font-family: var(--mono); font-size: 1.15px; fill: var(--muted);
  text-anchor: middle; dominant-baseline: middle;
}}
.corridor {{
  fill: color-mix(in srgb, var(--corridor) 14%, transparent);
  stroke: color-mix(in srgb, var(--corridor) 55%, transparent);
  stroke-width: 1; vector-effect: non-scaling-stroke;
  transition: fill 0.12s ease, stroke 0.12s ease;
}}
.corridor:hover, .corridor:focus {{
  fill: color-mix(in srgb, var(--corridor-hover) 26%, transparent);
  stroke: var(--corridor-hover); outline: none;
}}
.raw-path {{ fill: none; stroke: var(--faint); stroke-width: 1; stroke-dasharray: 0.4 1.2; vector-effect: non-scaling-stroke; }}
.simplified-path {{ fill: none; stroke: var(--path); stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }}
.waypoint {{ fill: var(--panel); stroke: var(--path); stroke-width: 1.6; vector-effect: non-scaling-stroke; }}
.marker-start {{ fill: var(--start); }}
.marker-goal {{ fill: var(--goal); }}
.marker-label {{
  font-family: var(--mono); font-size: 1.6px; font-weight: 700; fill: var(--ink);
  text-anchor: middle; dominant-baseline: middle;
}}

#inspector {{
  font-family: var(--mono); font-size: 0.78rem; color: var(--muted);
  padding: 0.5rem 0.7rem; border-top: 1px dashed var(--line); min-height: 1.4em;
}}
#inspector.active {{ color: var(--corridor-hover); }}

.legend {{ display: flex; flex-wrap: wrap; gap: 0.9rem 1.3rem; padding: 0 0.3rem; }}
.legend-item {{ display: flex; align-items: center; gap: 0.45rem; font-size: 0.78rem; color: var(--muted); }}
.swatch {{ width: 0.85rem; height: 0.85rem; border-radius: 2px; flex: none; }}
.swatch.line {{ height: 2px; border-radius: 0; align-self: center; }}

h2.side {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.8rem; font-weight: 650; }}

.stat-row {{ display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px solid var(--line); font-size: 0.85rem; }}
.stat-row:last-child {{ border-bottom: none; }}
.stat-row .k {{ color: var(--muted); }}
.stat-row .v {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}

table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
th {{
  text-align: left; font-family: var(--mono); font-weight: 600; font-size: 0.68rem;
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
  padding: 0.3rem 0.4rem; border-bottom: 1px solid var(--line);
}}
td {{
  font-family: var(--mono); font-variant-numeric: tabular-nums; padding: 0.3rem 0.4rem;
  border-bottom: 1px solid var(--line); color: var(--ink);
}}
tr:last-child td {{ border-bottom: none; }}
.scroll {{ max-height: 220px; overflow-y: auto; overflow-x: auto; }}

footer {{ margin-top: 1.75rem; color: var(--faint); font-size: 0.78rem; font-family: var(--mono); }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Dynamic path planning &mdash; front-end / back-end check</p>
    <h1>A* path + Safe Flight Corridor over <code>obstacle_field</code></h1>
    <p class="sub">
      Offline geometry check for the MPC upgrade: an 8-connected grid A*
      (<code>grid_res={grid_res:g} m</code>) finds a collision-free route, gets
      string-pulled down to a few straight segments, then each segment is
      inflated into the largest obstacle-free box it fits in
      (<code>safety_margin={safety_margin:g} m</code>). That box chain is what
      turns MPC's obstacle constraint from non-convex into a stack of linear
      inequalities &mdash; nothing here runs the solver yet, this just verifies
      the geometry is sane against <code>gazebo/config/obstacle_map.yaml</code>
      first.
    </p>
  </header>

  <div class="layout">
    <div>
      <div class="panel canvas-panel">
        <div class="canvas-head">
          <h2>Plan: ({start_e:g}, {start_n:g}) &rarr; ({goal_e:g}, {goal_n:g})</h2>
          <span class="compass">N &uarr;&nbsp;&nbsp;E &rarr;</span>
        </div>
        <svg class="field" viewBox="{view_box}" role="img" aria-label="Top-down obstacle field with planned path and safety corridor">
          {grid_lines}
          {corridor}
          {obstacles}
          <polyline class="raw-path" points="{raw_pts}"/>
          <polyline class="simplified-path" points="{simp_pts}"/>
          <circle class="waypoint" r="0.5" cx="{start_x}" cy="{start_y}" style="display:none"/>
          <circle class="marker-start" r="1.6" cx="{start_x}" cy="{start_y}"/>
          <text class="marker-label" x="{start_x}" y="{start_y}">S</text>
          <circle class="marker-goal" r="1.6" cx="{goal_x}" cy="{goal_y}"/>
          <text class="marker-label" x="{goal_x}" y="{goal_y}">G</text>
        </svg>
        <div class="legend">
          <span class="legend-item"><span class="swatch" style="background:var(--obstacle-fill);border:1px solid var(--obstacle-line)"></span>obstacle (real footprint)</span>
          <span class="legend-item"><span class="swatch" style="background:color-mix(in srgb, var(--corridor) 30%, transparent);border:1px solid var(--corridor)"></span>safe flight corridor</span>
          <span class="legend-item"><span class="swatch line" style="background:var(--faint)"></span>raw A* grid path</span>
          <span class="legend-item"><span class="swatch line" style="background:var(--path)"></span>simplified path</span>
          <span class="legend-item"><span class="swatch" style="background:var(--start);border-radius:50%"></span>start</span>
          <span class="legend-item"><span class="swatch" style="background:var(--goal);border-radius:50%"></span>goal</span>
        </div>
        <div id="inspector">Hover or tab into a corridor box to inspect its bounds.</div>
      </div>
    </div>

    <div>
      <div class="panel">
        <h2 class="side">Run stats</h2>
        <div class="stat-row"><span class="k">obstacles</span><span class="v">{n_obstacles}</span></div>
        <div class="stat-row"><span class="k">raw grid points</span><span class="v">{n_raw}</span></div>
        <div class="stat-row"><span class="k">simplified waypoints</span><span class="v">{n_simplified}</span></div>
        <div class="stat-row"><span class="k">corridor boxes</span><span class="v">{n_corridor}</span></div>
        <div class="stat-row"><span class="k">grid_res</span><span class="v">{grid_res:g} m</span></div>
        <div class="stat-row"><span class="k">safety_margin</span><span class="v">{safety_margin:g} m</span></div>
        <div class="stat-row"><span class="k">seed_spacing</span><span class="v">{seed_spacing:g} m</span></div>
      </div>

      <div class="panel">
        <h2 class="side">Simplified waypoints</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>#</th><th>E</th><th>N</th></tr></thead>
            <tbody>{waypoint_rows}</tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <h2 class="side">Corridor boxes</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>#</th><th>E range</th><th>N range</th><th>size (m)</th></tr></thead>
            <tbody>{corridor_rows}</tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <footer>precision_landing/precision_landing/planner.py &middot; gazebo/plan_demo.py &middot; source: gazebo/config/obstacle_map.yaml</footer>
</div>

<script>
  document.querySelectorAll('.corridor').forEach(function (el) {{
    var insp = document.getElementById('inspector');
    function show() {{ insp.textContent = el.getAttribute('data-info'); insp.classList.add('active'); }}
    function hide() {{ insp.textContent = 'Hover or tab into a corridor box to inspect its bounds.'; insp.classList.remove('active'); }}
    el.addEventListener('mouseenter', show);
    el.addEventListener('mouseleave', hide);
    el.addEventListener('focus', show);
    el.addEventListener('blur', hide);
  }});
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument('--goal', nargs=2, type=float, default=[-15.0, -20.0])
    ap.add_argument('--map', default=os.path.join(SCRIPT_DIR, 'config', 'obstacle_map.yaml'))
    ap.add_argument('--grid-res', type=float, default=1.0)
    ap.add_argument('--safety-margin', type=float, default=1.5)
    ap.add_argument('--seed-spacing', type=float, default=3.0)
    ap.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'plan_demo_out.html'))
    args = ap.parse_args()

    html = build_html(tuple(args.start), tuple(args.goal), args.map,
                      args.grid_res, args.safety_margin, args.seed_spacing)
    with open(args.out, 'w') as f:
        f.write(html)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
