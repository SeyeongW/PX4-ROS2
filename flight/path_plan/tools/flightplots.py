#!/usr/bin/env python3
"""Charts for one experiment-logger run.

    from flightplots import plot_run
    plot_run(FlightLog.newest('flight_logs_analysis'), 'run.png')

DESIGN NOTES, so a later edit does not undo them.

ONE SERIES PER PANEL, NOT A RAINBOW OF PHASES. Colouring the track by phase
would put seven categorical hues on one scatter, and seven cannot hold the
all-pairs colourblind separation floor at once. The track is therefore a single
blue line and the phases are named where they change, which is also what an
operator reads faster: the question is "where did it go wrong", not "which of
seven colours is HOVER".

NO DUAL AXES. A* plan time and MPC solve time differ by an order of magnitude,
so they get their own panels rather than a second y-scale on one.

RED MEANS ONE THING: outside the map's terrain box, where no route can be
spliced from the vehicle's true position. It is never a series colour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Validated against the light chart surface with the skill's checker:
# CVD dE 23.8, normal-vision dE 31.6, both marks >= 3:1 on the surface.
SURFACE = '#fcfcfb'
SERIES = '#2a78d6'        # the vehicle: one series, one hue
CRITICAL = '#d03b3b'      # reserved: outside the certified map extent
INK = '#0b0b0b'
INK_SOFT = '#52514e'
INK_MUTED = '#8a8983'
GRID = '#e6e5e1'
OBSTACLE = '#d8d7d2'      # scenery, deliberately recessive


def _style(axes) -> None:
    """Recessive grid and axes; the data is the only thing with weight."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, lw=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_SOFT, labelsize=9, length=3)
    axes.title.set_color(INK)
    axes.xaxis.label.set_color(INK_SOFT)
    axes.yaxis.label.set_color(INK_SOFT)


def _map_geometry(map_yaml):
    """Terrain box and obstacle rectangles, or None when unavailable."""
    if map_yaml is None or not Path(map_yaml).expanduser().exists():
        return None
    import yaml
    document = yaml.safe_load(Path(map_yaml).expanduser().read_text())
    terrain, mission = document['terrain'], document['mission']
    centre = np.asarray(terrain['center_m'], float)
    half = 0.5 * np.asarray(terrain['size_m'], float)
    clearance = float(mission['vehicle_clearance_xy_m'])
    boxes = [(np.asarray(o['center_m'][:2], float),
              np.asarray(o['size_m'][:2], float))
             for o in mission.get('obstacles', [])]
    return centre - half + clearance, centre + half - clearance, boxes, clearance


def plot_run(log, out_png, map_yaml=None):
    """Six panels: where it flew, how well, how high, and what the solvers cost."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pose = log.pose
    if pose.empty:
        raise ValueError(f'{log.run_id} has no pose samples to plot')
    t = pose['elapsed_s'].to_numpy(float)
    x = pose['x_m'].to_numpy(float)
    y = pose['y_m'].to_numpy(float)
    z = pose['z_m'].to_numpy(float)
    error = (pose['tracking_error_m'].astype(float).to_numpy()
             if 'tracking_error_m' in pose else np.full(len(t), np.nan))

    geometry = _map_geometry(map_yaml)
    outside = np.zeros(len(t), dtype=bool)
    if geometry is not None:
        low, high, _boxes, _clearance = geometry
        xy = np.column_stack((x, y))
        outside = np.any((xy < low) | (xy > high), axis=1)

    figure = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    # The track gets its own tall column so `aspect('equal')` can hold without
    # stretching the x range: on a wide panel equal aspect pads x to -200..250
    # for data that only spans 57 m, and the map disappears into the middle.
    grid = figure.add_gridspec(4, 2, width_ratios=[1.2, 1],
                               hspace=0.75, wspace=0.18)

    # --- 1. the track, with the map it was flown against -------------------
    track = figure.add_subplot(grid[:, 0])
    if geometry is not None:
        low, high, boxes, clearance = geometry
        for centre, size in boxes:
            track.add_patch(Rectangle(
                centre - size / 2 - clearance, *(size + 2 * clearance),
                facecolor=OBSTACLE, edgecolor='none', alpha=0.55, zorder=1))
            track.add_patch(Rectangle(
                centre - size / 2, *size,
                facecolor=INK_MUTED, edgecolor='none', zorder=2))
        track.add_patch(Rectangle(
            low, *(high - low), fill=False, edgecolor=INK_MUTED,
            lw=1.2, ls=(0, (6, 4)), zorder=1))
        track.annotate('map extent', (low[0], high[1]), xytext=(4, -12),
                       textcoords='offset points', color=INK_MUTED, fontsize=9)
    track.plot(x, y, color=SERIES, lw=2.0, zorder=4, solid_capstyle='round')
    if outside.any():
        broken = np.where(outside, y, np.nan)
        track.plot(x, broken, color=CRITICAL, lw=2.6, zorder=5,
                   solid_capstyle='round')
        track.annotate(
            f'outside the map — {100 * outside.mean():.0f}% of the run;\n'
            'no route can be spliced from out here',
            (x[outside][np.argmin(y[outside])], y[outside].min()),
            xytext=(12, -6), textcoords='offset points',
            color=CRITICAL, fontsize=10, fontweight='bold')
    track.plot(x[0], y[0], 'o', ms=9, color=SURFACE,
               markeredgecolor=INK, markeredgewidth=2, zorder=6)
    track.annotate('launch', (x[0], y[0]), xytext=(10, 8),
                   textcoords='offset points', color=INK, fontsize=10)

    # Phase changes named where they happen: identity by label, not by hue.
    phase = pose['phase'].to_numpy()
    changes = np.flatnonzero(phase[1:] != phase[:-1]) + 1
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    placed = [(x[0], y[0])]                      # 'launch' already occupies this
    side = 1
    for index in changes:
        track.plot(x[index], y[index], 'o', ms=6, color=SURFACE,
                   markeredgecolor=INK_SOFT, markeredgewidth=1.6, zorder=6)
        # SELECTIVE LABELS, NOT ONE PER EVENT. Several phases can change within
        # a metre of each other — MISSION -> HOVER -> RETURN_PLAN all happen at
        # the goal — and stacking their labels makes an unreadable smear. Keep
        # the marker for every change, the text only where it can be read.
        if any(np.hypot(x[index] - px, y[index] - py) < 0.06 * span
               for px, py in placed):
            continue
        placed.append((x[index], y[index]))
        track.annotate(f'{phase[index]}  {t[index]:.0f}s',
                       (x[index], y[index]), xytext=(9, 7 * side),
                       textcoords='offset points', color=INK_SOFT, fontsize=8.5)
        side *= -1
    track.set_title(f'{log.run_id} — flown track, local ENU',
                    fontsize=13, fontweight='bold', loc='left', pad=12)
    track.set_xlabel('x [m]'); track.set_ylabel('y [m]')
    # Equal aspect on a FIXED box: bound the view to the data and the map, then
    # let the box shrink to fit rather than the limits grow to fill.
    xs, ys = [x.min(), x.max()], [y.min(), y.max()]
    if geometry is not None:
        xs += [low[0], high[0]]
        ys += [low[1], high[1]]
    pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys))
    track.set_xlim(min(xs) - pad, max(xs) + pad)
    track.set_ylim(min(ys) - pad, max(ys) + pad)
    track.set_aspect('equal', adjustable='box')
    _style(track)

    # --- 2/3. how well it tracked, and how high ----------------------------
    for column, (values, label, unit) in enumerate((
            (error, 'tracking error', 'm'), (z, 'altitude', 'm'))):
        axes = figure.add_subplot(grid[column, 1])
        axes.plot(t, values, color=SERIES, lw=2.0, solid_capstyle='round')
        if outside.any():
            axes.plot(t, np.where(outside, values, np.nan),
                      color=CRITICAL, lw=2.4, solid_capstyle='round')
        finite = values[np.isfinite(values)]
        if len(finite):
            axes.annotate(f'max {finite.max():.1f} {unit}',
                          (1.0, 1.02), xycoords='axes fraction',
                          ha='right', color=INK_SOFT, fontsize=9)
        axes.set_title(label, fontsize=11, fontweight='bold', loc='left')
        axes.set_xlabel('t [s]'); axes.set_ylabel(f'[{unit}]')
        _style(axes)

    # --- 4/5. what the solvers cost ---------------------------------------
    import pandas as pd
    for column, (frame, key, label) in enumerate((
            (log.astar, 'astar_plan_time_ms', 'A* plan time'),
            (log.mpc, 'mpc_solve_time_ms', 'MPC solve time'))):
        axes = figure.add_subplot(grid[2 + column, 1])
        # Time and value must come from the SAME rows: dropping NaN from each
        # column independently silently pairs a solve time with another solve's
        # timestamp as soon as one row is incomplete.
        if key in frame:
            paired = frame[['elapsed_s', key]].apply(
                pd.to_numeric, errors='coerce').dropna()
        else:
            paired = pd.DataFrame(columns=['elapsed_s', key])
        when = paired['elapsed_s'].to_numpy(float)
        values = paired[key].to_numpy(float)
        if len(values):
            axes.vlines(when, 0.0, values, color=SERIES, lw=2.0)
            axes.plot(when, values, 'o', ms=4, color=SERIES)
            axes.annotate(
                f'n {len(values)}   median {np.median(values):.0f} ms   '
                f'max {values.max():.0f} ms',
                (1.0, 1.02), xycoords='axes fraction', ha='right',
                color=INK_SOFT, fontsize=9)
        else:
            axes.annotate('not logged in this run', (0.5, 0.5),
                          xycoords='axes fraction', ha='center',
                          color=INK_MUTED, fontsize=10)
        axes.set_title(label, fontsize=11, fontweight='bold', loc='left')
        axes.set_xlabel('t [s]'); axes.set_ylabel('[ms]')
        _style(axes)

    figure.savefig(out_png, dpi=110, facecolor=SURFACE,
                   bbox_inches='tight', pad_inches=0.35)
    plt.close(figure)
    return Path(out_png)
