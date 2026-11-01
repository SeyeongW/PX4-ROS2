#!/usr/bin/env python3
"""Plot recorded mission paths; never run a simulator or fabricate a flight.

Inputs are stl_mission plan.json / telemetry.jsonl. Optional Gazebo evaluation
rows have {"stamp": simulation_seconds, "position": [world_x, world_y, world_z]}.
All positions must already use the same world ENU frame and metres. Tracking
error uses the recorded EKF feedback and its time-matched mission reference,
not nearest-path distance or an interpolated Gazebo position. The elevation
panel projects horizontal position onto the home-to-target direction.

Example:
  python3 tools/plot_flight_path.py --mission /path/to/mission \
    --truth /path/to/truth.jsonl --truth-target 0 -40 10 \
    --output /explicit/output/flight_path
"""
import argparse
import json
from pathlib import Path

import numpy as np


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name}: expected three finite world-frame coordinates')
    return result


def json_rows(path):
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{number}: expected an object')
                rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no recorded samples')
    return rows


def trace(rows, name):
    times, points = [], []
    for index, row in enumerate(rows):
        stamp = row.get('own_stamp', row.get('stamp'))
        if stamp is None or not np.isfinite(float(stamp)):
            raise ValueError(f'{name}[{index}]: missing or invalid timestamp')
        times.append(float(stamp))
        # Explicit missing samples stay as gaps; invalid coordinates are errors.
        points.append([np.nan] * 3 if row.get('position') is None
                      else vector(row['position'], f'{name}[{index}].position'))
    times, points = np.asarray(times), np.asarray(points)
    if np.any(np.diff(times) < 0):
        raise ValueError(f'{name}: timestamps run backwards; split clock-reset runs')
    if not np.isfinite(points).any():
        raise ValueError(f'{name}: no measured position samples')
    return times, points


def break_gaps(times, values, max_gap):
    """Insert NaNs instead of connecting intervals without received telemetry."""
    indices = np.flatnonzero(np.diff(times) > max_gap) + 1
    return (np.insert(times, indices, np.nan),
            np.insert(values, indices, np.nan, axis=0))


def tracking_errors(rows, positions):
    errors = []
    for row, position in zip(rows, positions):
        desired = row.get('desired_position_at_feedback_stamp')
        if desired is not None and np.isfinite(position).all():
            error = float(np.linalg.norm(position - vector(desired, 'feedback reference')))
            recorded = row.get('tracking_error_m')
            if recorded is not None and not np.isclose(error, float(recorded), atol=1e-6):
                raise ValueError('tracking_error_m disagrees with recorded feedback reference')
        else:
            error = row.get('tracking_error_m', np.nan)
            error = np.nan if error is None else float(error)
        if np.isinf(error) or (np.isfinite(error) and error < 0):
            raise ValueError('tracking_error_m must be nonnegative or missing')
        errors.append(error)
    return np.asarray(errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mission', type=Path, required=True,
                        help='directory containing plan.json and telemetry.jsonl')
    parser.add_argument('--truth', type=Path, help='optional Gazebo evaluation JSONL')
    parser.add_argument('--truth-target', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='optional marker truth in world ENU metres; evaluation only')
    parser.add_argument('--output', type=Path, required=True,
                        help='explicit output stem; writes .png and .svg')
    parser.add_argument('--title', help='optional figure title / trial identifier')
    parser.add_argument('--gap-seconds', type=float, default=1.,
                        help='show a line break when telemetry gap exceeds this (default: 1 s)')
    parser.add_argument('--force', action='store_true', help='replace existing output images')
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.gap_seconds) or args.gap_seconds <= 0:
            raise ValueError('--gap-seconds must be positive and finite')
        outputs = [args.output.with_suffix(suffix) for suffix in ('.png', '.svg')]
        if not args.force and any(path.exists() for path in outputs):
            raise ValueError('output exists; use another stem or --force')
        plan = json.loads((args.mission / 'plan.json').read_text())
        planned = np.asarray([vector(row['position'], 'plan sample') for row in plan['samples']])
        if len(planned) < 2:
            raise ValueError('plan.json must contain at least two path samples')
        home, target, goal = [vector(plan[key], key) for key in ('home', 'target_world', 'goal')]
        rows = json_rows(args.mission / 'telemetry.jsonl')
        times, measured = trace(rows, 'telemetry')
        errors = tracking_errors(rows, measured)
        truth = None
        if args.truth:
            truth_times, truth_points = trace(json_rows(args.truth), 'truth')
            # Preserve only the same observed mission interval, without resampling.
            keep = (truth_times >= times[0]) & (truth_times <= times[-1])
            if not keep.any():
                raise ValueError('truth and mission time intervals do not overlap')
            truth = break_gaps(truth_times[keep], truth_points[keep], args.gap_seconds)[1]
        true_target = (vector(args.truth_target, 'truth target')
                       if args.truth_target is not None else None)
        draw_times, draw_measured = break_gaps(times - times[0], measured, args.gap_seconds)
        _, draw_errors = break_gaps(times - times[0], errors, args.gap_seconds)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator

    fonts = {item.name for item in font_manager.fontManager.ttflist}
    korean_font = next((name for name in ('Noto Sans CJK KR', 'Noto Sans CJK JP',
                                         'NanumGothic') if name in fonts), None)
    ko = bool(korean_font)
    def label(korean, english):
        return korean if ko else english

    horizontal = target[:2] - home[:2]
    direction = horizontal / np.linalg.norm(horizontal) if np.linalg.norm(horizontal) else np.array([1., 0.])
    series = [(planned, '#333333', '--', label('생성 경로', 'Planned path')),
              (draw_measured, '#0072B2', '-', label('PX4 EKF 추정 궤적', 'PX4 EKF trajectory'))]
    if truth is not None:
        series.append((truth, '#009E73', ':', label('Gazebo 실제 궤적 (평가용)', 'Gazebo truth (evaluation)')))
    landmarks = [(home, 'o', '#333333', label('출발점', 'Home')),
                 (target, '*', '#D55E00', label('락한 풍선 지점', 'Locked balloon target')),
                 (goal, 's', '#882255', label('통과 후 호버 지점' if plan.get('gate') else '호버 지점',
                                            'Post-pass hover goal' if plan.get('gate') else 'Hover goal'))]
    if plan.get('geometry', {}).get('lift') is not None:
        landmarks.append((vector(plan['geometry']['lift'], 'lift'), '^', '#666666',
                          label('수직 이륙 +3m 연결점', 'Vertical takeoff +3m join')))
    if true_target is not None:
        landmarks.append((true_target, 'x', '#CC3311', label('풍선 정답 (평가용)', 'Marker truth (evaluation)')))

    style = {'font.family': korean_font or 'DejaVu Sans', 'font.size': 10,
             'axes.titlesize': 12, 'axes.unicode_minus': False, 'axes.spines.top': False,
             'axes.spines.right': False, 'svg.fonttype': 'path', 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .20])
        spatial = fig.add_subplot(grid[0, 0], projection='3d')
        overhead = fig.add_subplot(grid[0, 1])
        elevation = fig.add_subplot(grid[1, 0])
        error_axis = fig.add_subplot(grid[1, 1])
        legend_axis = fig.add_subplot(grid[2, :])
        legend_axis.axis('off')
        for points, color, linestyle, name in series:
            spatial.plot(*points.T, color=color, ls=linestyle, lw=1.7, label=name)
            overhead.plot(points[:, 0], points[:, 1], color=color, ls=linestyle, lw=1.7)
            distance = (points[:, :2] - home[:2]) @ direction
            elevation.plot(distance, points[:, 2], color=color, ls=linestyle, lw=1.7)
        for point, marker, color, name in landmarks:
            spatial.scatter(*point, marker=marker, color=color, s=65, label=name, depthshade=False)
            overhead.scatter(*point[:2], marker=marker, color=color, s=65, zorder=4)
            elevation.scatter((point[:2] - home[:2]) @ direction, point[2],
                              marker=marker, color=color, s=65, zorder=4)
        spatial.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                    title=label('3차원 경로 · world ENU', '3D path · world ENU'))
        all_points = np.vstack([points for points, *_ in series] + [p[None, :] for p, *_ in landmarks])
        low, high = np.nanmin(all_points, axis=0), np.nanmax(all_points, axis=0)
        span = np.maximum(high - low, .5)
        # Widen the display box, not the trajectory, to keep nearly planar
        # flights legible. Box aspect still preserves equal physical units.
        extent = np.maximum(span * 1.06, span.max() * .28)
        for setter, center, width in zip((spatial.set_xlim, spatial.set_ylim, spatial.set_zlim),
                                         (low + high) / 2, extent):
            setter(center - width / 2, center + width / 2)
        spatial.set_box_aspect(extent)
        for axis in (spatial.xaxis, spatial.yaxis, spatial.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        spatial.view_init(elev=23, azim=38)
        overhead.set(xlabel='X (m)', ylabel='Y (m)',
                     title=label('수평 궤적 · X 범위 확대', 'Horizontal path · expanded X scale'))
        elevation.set(xlabel=label('출발점 기준 목표 방향 수평거리 (m)', 'Horizontal distance along approach (m)'),
                      ylabel=label('고도 Z (m)', 'Altitude Z (m)'),
                      title=label('수직 이륙 → 곡선 → 통과 후 대기', 'Takeoff → spline → pass → hover'))
        error_axis.plot(draw_times, draw_errors, color='#0072B2', lw=1.4)
        error_axis.set(xlabel=label('첫 비행 표본 이후 시간 (s)', 'Time since first flight sample (s)'),
                       ylabel=label('위치 추종 오차 (m)', 'Position tracking error (m)'), ylim=(0, None),
                       title=label('EKF 위치 − 같은 피드백 시각의 경로 참조',
                                   'EKF position − reference at feedback timestamp'))
        finite_errors = errors[np.isfinite(errors)]
        if not len(finite_errors):
            error_axis.text(.5, .5, label('동시각 참조·추종오차 기록 없음', 'Time-matched tracking error unavailable'),
                            ha='center', va='center', transform=error_axis.transAxes)
        for axis in (overhead, elevation, error_axis):
            axis.grid(True, color='#D9DDE3', lw=.6)
        handles, names = spatial.get_legend_handles_labels()
        legend_axis.legend(handles, names, loc='upper center', ncol=4, frameon=False, fontsize=9)
        note = label('기록된 표본 그대로 표시 · 평활화/재표본화 없음 · '
                     f'{args.gap_seconds:g}s 초과 수신 공백은 선을 끊어 표시',
                     f'Recorded samples; no smoothing/resampling; gaps > {args.gap_seconds:g}s break lines')
        legend_axis.text(.5, .05, note, ha='center', va='bottom', fontsize=9, color='#444444')
        fig.suptitle(args.title or label('계획 경로와 기록된 드론 궤적 · 시뮬레이션',
                                         'Planned path and recorded drone trajectory · simulation'), fontsize=15)
        source = f'plan/telemetry: {args.mission.resolve()}; truth: {args.truth}; {note}'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            fig.savefig(path, dpi=180, metadata={'Description': source})
        plt.close(fig)
    print(json.dumps(dict(outputs=[str(path) for path in outputs],
                          telemetry_samples=len(rows), truth_included=truth is not None,
                          tracking_error_max_m=float(finite_errors.max()) if len(finite_errors) else None,
                          tracking_error_rmse_m=float(np.sqrt(np.mean(finite_errors ** 2))) if len(finite_errors) else None,
                          locked_target_world=target.tolist(), hover_goal_world=goal.tolist(),
                          target_to_hover_distance_m=float(np.linalg.norm(goal-target))), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
