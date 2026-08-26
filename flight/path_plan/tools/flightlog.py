#!/usr/bin/env python3
"""Load and analyse one experiment-logger run.

    python3 flightlog.py                      newest run under ~/px4_experiment_logs
    python3 flightlog.py 20260826T0717        that run (a run_id prefix is enough)
    python3 flightlog.py path/to/dir          newest run in that directory
    python3 flightlog.py <run> --plot out.png track, error and altitude

Import it to do your own work — the report below is only one view of it:

    from flightlog import FlightLog
    log = FlightLog.newest('flight_logs_analysis')
    log.pose          # DataFrame: elapsed_s, x_m, y_m, z_m, phase, ...
    log.mpc           # DataFrame: mpc_solve_time_ms, ...
    log.astar         # DataFrame: astar_plan_time_ms, path_length_m, ...
    log.summary       # dict of the one-row summary CSV

THE TIMESERIES IS AN EVENT LOG, NOT A TABLE. Each row carries the `event` that
produced it and fills only that event's columns, so a naive `df.mean()` over a
column averages it against thousands of blanks from other event types. Splitting
by event is the whole reason this module exists rather than a `read_csv` call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns the logger leaves as a sentinel rather than a measurement. Fixing
#: that belongs in experiment_logger.py; refusing to plot it belongs here.
_IMPLAUSIBLE = {'min_clearance_m': 1.0e4}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """A run's column as floats, or an empty Series if it has none.

    A COLUMN CAN BE ABSENT, not merely empty: `events()` drops the ones this
    run never filled, and a run that never armed has no tracking error at all.
    `pd.to_numeric(None)` answers a bare nan rather than a Series, so every
    caller doing `.dropna()` on the result breaks on exactly those runs — three
    of sixteen here, all of them the short ones worth skimming quickly.
    """
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors='coerce').dropna()


class FlightLog:
    """One run: its summary row and its timeseries, split by event type."""

    def __init__(self, timeseries_csv: str | Path):
        self.path = Path(timeseries_csv)
        self.run_id = self.path.name.replace('_timeseries.csv', '')
        self.frame = pd.read_csv(self.path)
        summary_path = self.path.with_name(f'{self.run_id}_summary.csv')
        self.summary = {}
        if summary_path.exists():
            rows = pd.read_csv(summary_path).to_dict('records')
            self.summary = rows[0] if rows else {}

    # ------------------------------------------------------------- selection
    def events(self, name: str) -> pd.DataFrame:
        """Rows this event produced, with its all-blank columns dropped."""
        rows = self.frame[self.frame['event'] == name]
        return rows.dropna(axis=1, how='all').reset_index(drop=True)

    @property
    def pose(self) -> pd.DataFrame:
        return self.events('pose')

    @property
    def mpc(self) -> pd.DataFrame:
        return self.events('mpc')

    @property
    def astar(self) -> pd.DataFrame:
        return self.events('astar')

    @classmethod
    def newest(cls, directory: str | Path = '~/px4_experiment_logs') -> 'FlightLog':
        found = sorted(Path(directory).expanduser().glob('*_timeseries.csv'))
        if not found:
            raise FileNotFoundError(f'no *_timeseries.csv under {directory}')
        return cls(found[-1])          # run_id sorts chronologically by design

    @classmethod
    def find(cls, token: str,
             directory: str | Path = '~/px4_experiment_logs') -> 'FlightLog':
        root = Path(token).expanduser()
        if root.is_dir():
            return cls.newest(root)
        if root.is_file():
            return cls(root)
        found = sorted(Path(directory).expanduser().glob(
            f'{token}*_timeseries.csv'))
        if not found:
            raise FileNotFoundError(f'no run matching {token!r} in {directory}')
        return cls(found[-1])

    # --------------------------------------------------------------- derived
    def phase_timeline(self) -> pd.DataFrame:
        """One row per phase the run entered, with how it went while there."""
        pose = self.pose
        if pose.empty:
            return pd.DataFrame()
        block = (pose['phase'] != pose['phase'].shift()).cumsum()
        out = []
        for _, rows in pose.groupby(block, sort=True):
            error = _numeric(rows, 'tracking_error_m')
            out.append({
                'phase': rows['phase'].iloc[0],
                'from_s': float(rows['elapsed_s'].iloc[0]),
                'to_s': float(rows['elapsed_s'].iloc[-1]),
                'x_m': float(rows['x_m'].iloc[-1]),
                'y_m': float(rows['y_m'].iloc[-1]),
                'err_mean_m': float(error.mean()) if len(error) else np.nan,
                'err_max_m': float(error.max()) if len(error) else np.nan,
            })
        return pd.DataFrame(out)

    def outside_map(self, map_yaml: str | Path) -> pd.DataFrame:
        """Pose samples the route map could never plan from.

        The terrain box is not an obstacle, but `segment_is_free` fails on it
        exactly like one, so a vehicle out here cannot have a route spliced
        from its true position — every plan is rejected at the join. This is
        the first thing to check on a run that replanned and never moved.
        """
        import yaml
        # Read the terrain box from the document: route_map_info reports the
        # mission, not the extent, and the extent is the thing being tested.
        document = yaml.safe_load(Path(map_yaml).expanduser().read_text())
        centre = np.asarray(document['terrain']['center_m'], float)
        half = 0.5 * np.asarray(document['terrain']['size_m'], float)
        clearance = float(document['mission']['vehicle_clearance_xy_m'])
        low, high = centre - half + clearance, centre + half - clearance
        pose = self.pose
        xy = pose[['x_m', 'y_m']].to_numpy(float)
        out = np.any((xy < low) | (xy > high), axis=1)
        frame = pose.loc[out, ['elapsed_s', 'phase', 'x_m', 'y_m']]
        # The box is read from the map as it stands NOW, which is not
        # necessarily the box the run flew under. Carry the numbers so a report
        # can say which extent it judged, instead of an unqualified verdict.
        frame.attrs['bounds'] = (low, high)
        return frame

    # ---------------------------------------------------------------- report
    def report(self, map_yaml: str | Path | None = None) -> str:
        lines = [f'=== {self.run_id}']
        for key in ('started_at_utc', 'ended_at_utc', 'end_reason'):
            if self.summary.get(key) is not None:
                lines.append(f'  {key:24} {self.summary[key]}')

        pose, mpc, astar = self.pose, self.mpc, self.astar
        if not pose.empty:
            error = _numeric(pose, 'tracking_error_m')
            lines += [
                f'  {"duration_s":24} {pose["elapsed_s"].iloc[-1]:.1f}',
                f'  {"pose samples":24} {len(pose)}',
            ]
            if len(error):
                lines.append(
                    f'  {"tracking error m":24} '
                    f'mean {error.mean():.2f}  rmse '
                    f'{np.sqrt((error ** 2).mean()):.2f}  max {error.max():.2f}')
        for frame, column, label in ((astar, 'astar_plan_time_ms', 'A* plan ms'),
                                     (mpc, 'mpc_solve_time_ms', 'MPC solve ms')):
            values = _numeric(frame, column)
            if len(values):
                lines.append(
                    f'  {label:24} n {len(values):<5d} med '
                    f'{values.median():7.1f}  max {values.max():7.1f}')

        for column, ceiling in _IMPLAUSIBLE.items():
            values = _numeric(self.frame, column)
            if len(values) and values.max() > ceiling:
                lines.append(
                    f'  {column:24} NOT USABLE (max {values.max():.3g} — the '
                    f'logger emits a sentinel, not a measurement)')

        # THE FIRST THING TO CHECK on a run that replanned and never moved.
        # Two runs 18 minutes apart on the same code and map came out at rmse
        # 1.59 m and 23.38 m; the whole difference is that the second spent
        # 48% of its samples outside the terrain box, where no plan can be
        # spliced from the vehicle's true position.
        if map_yaml is not None:
            try:
                outside = self.outside_map(map_yaml)
            except (KeyError, OSError, ValueError, ImportError) as exc:
                lines.append(f'  {"map bounds":24} not checked ({exc})')
            else:
                total = max(len(self.pose), 1)
                low, high = outside.attrs.get('bounds', (None, None))
                extent = ('' if low is None else
                          f' [{low[0]:.0f},{high[0]:.0f}]x'
                          f'[{low[1]:.0f},{high[1]:.0f}]')
                if outside.empty:
                    lines.append(
                        f'  {"map bounds":24} inside for the whole run'
                        f'{extent} (map as it stands now, not necessarily the '
                        f'one flown)')
                else:
                    lines.append(
                        f'  {"map bounds":24} OUTSIDE{extent} for '
                        f'{len(outside)}/{total}'
                        f' samples ({100.0 * len(outside) / total:.1f}%), from '
                        f'{outside["elapsed_s"].iloc[0]:.1f}s in '
                        f'{outside["phase"].iloc[0]} — no route can be spliced '
                        f'from out here')

        timeline = self.phase_timeline()
        if not timeline.empty:
            lines.append('')
            lines.append('  phase          from     to        end x,y        '
                         'err mean / max')
            for row in timeline.itertuples():
                lines.append(
                    f'  {row.phase:<13} {row.from_s:6.1f} {row.to_s:6.1f}  '
                    f'({row.x_m:7.1f},{row.y_m:7.1f})  '
                    f'{row.err_mean_m:6.2f} /{row.err_max_m:7.2f}')
        return '\n'.join(lines)

    def plot(self, out_png: str | Path) -> Path:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pose = self.pose
        # matplotlib 3.5 will not index a Series positionally the way newer
        # pandas returns it; hand it plain arrays and the version stops mattering.
        t = pose['elapsed_s'].to_numpy(float)
        x = pose['x_m'].to_numpy(float)
        y = pose['y_m'].to_numpy(float)
        z = pose['z_m'].to_numpy(float)
        error = (pose['tracking_error_m'].astype(float).to_numpy()
                 if 'tracking_error_m' in pose else np.full(len(t), np.nan))
        figure, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].plot(x, y, lw=1.2)
        axes[0].plot(x[0], y[0], 'go', label='start')
        axes[0].plot(x[-1], y[-1], 'rx', label='end')
        axes[0].set_title('track (local ENU)')
        axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
        axes[0].axis('equal'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(t, error, lw=1.0)
        axes[1].set_title('tracking error')
        axes[1].set_xlabel('t [s]'); axes[1].set_ylabel('[m]')
        axes[1].grid(alpha=0.3)
        axes[2].plot(t, z, lw=1.0)
        axes[2].set_title('altitude (z, ENU up)')
        axes[2].set_xlabel('t [s]'); axes[2].set_ylabel('[m]')
        axes[2].grid(alpha=0.3)
        figure.suptitle(self.run_id)
        figure.tight_layout()
        out_png = Path(out_png)
        figure.savefig(out_png, dpi=110)
        plt.close(figure)
        return out_png


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith('--')]
    log = FlightLog.find(args[0]) if args else FlightLog.newest()
    # The map the mission flies by default, so the bounds check needs no flag.
    default_map = (Path(__file__).resolve().parents[1]
                   / 'config' / 'drone_field_route.yaml')
    print(log.report(default_map if default_map.exists() else None))
    if '--plot' in argv:
        index = argv.index('--plot')
        target = argv[index + 1] if index + 1 < len(argv) else 'flightlog.png'
        print(f'\nwrote {log.plot(target)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
