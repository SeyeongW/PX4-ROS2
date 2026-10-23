#!/usr/bin/env python3
"""Batch re-generate all PNG charts in flight_logs_analysis using updated flightplots."""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parents[2]
LOGS_DIR = REPO_DIR / 'flight_logs_analysis'
MAP_YAML = REPO_DIR / 'flight/path_plan/config/drone_field_route.yaml'

sys.path.insert(0, str(TOOLS_DIR))
from flightlog import FlightLog
from flightplots import plot_run, plot_tracking_error

def main():
    if not LOGS_DIR.exists():
        print(f'Directory {LOGS_DIR} does not exist.')
        return 1

    runs = sorted(LOGS_DIR.glob('*_timeseries.csv'))
    print(f'Found {len(runs)} timeseries runs in {LOGS_DIR}')

    map_path = MAP_YAML if MAP_YAML.exists() else None

    count_run = 0
    count_err = 0

    for csv_path in runs:
        run_id = csv_path.name.replace('_timeseries.csv', '')
        summary_path = LOGS_DIR / f'{run_id}_summary.csv'
        out_run = LOGS_DIR / f'{run_id}.png'
        out_err = LOGS_DIR / f'{run_id}_error.png'

        try:
            log = FlightLog(csv_path)
        except Exception as exc:
            print(f'[{run_id}] Failed to load: {exc}')
            continue

        # Plot 6-panel overview chart if pose exists
        if not log.pose.empty and summary_path.exists():
            try:
                plot_run(log, out_run, map_path)
                count_run += 1
                print(f'  [✓] {run_id}.png')
            except Exception as exc:
                print(f'  [x] {run_id}.png failed: {exc}')

        # Plot tracking error chart
        try:
            plot_tracking_error(log, out_err, map_path)
            count_err += 1
            print(f'  [✓] {run_id}_error.png')
        except Exception as exc:
            print(f'  [-] {run_id}_error.png skipped: {exc}')

    print(f'\nFinished re-charting! Generated {count_run} overview charts, {count_err} error charts.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
