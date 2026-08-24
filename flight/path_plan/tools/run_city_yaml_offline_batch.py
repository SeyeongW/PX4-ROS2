#!/usr/bin/env python3
"""Repeat the city-YAML dynamic-pursuit rollout and preserve every dataset."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_dynamic_planning_report as report
import pursuit_sim as pursuit


DATASET_TYPE = "city YAML offline kinematic dynamic-pursuit rollout"


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state = Path.home() / ".local" / "state"
    return state / "px4-ros2-jo" / "batches" / f"city-yaml_{stamp}"


def _validate_scenario(scenario: dict) -> None:
    expected = {
        "drone.cruise_speed_m_s": (
            scenario["drone"]["cruise_speed_m_s"], 12.0),
        "drone.max_speed_m_s": (
            scenario["drone"]["max_speed_m_s"], 12.0),
        "pursuit.mpc_reference_speed_m_s": (
            scenario["pursuit"]["mpc_reference_speed_m_s"], 12.0),
        "trailer.speed_m_s": (scenario["trailer"]["speed_m_s"], 10.0),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if not math.isclose(float(actual), wanted, abs_tol=1.0e-9)
    ]
    if mismatches:
        raise ValueError("city-yaml speed contract failed: " + "; ".join(mismatches))


def _validate_rollout(log, captured, splines, horizons, plan_stats) -> None:
    accepted = sum(int(row["accepted"]) for row in plan_stats)
    problems = []
    if not captured:
        problems.append("capture flag is false")
    if not log or int(log[-1]["captured"]) != 1 or log[-1]["phase"] != "capture":
        problems.append("final timeseries row is not capture")
    if not splines or accepted != len(splines):
        problems.append(
            f"accepted replan mismatch ({accepted} stats, {len(splines)} paths)")
    if not horizons:
        problems.append("no MPC horizons")
    if not plan_stats:
        problems.append("no planning attempts")
    if problems:
        raise RuntimeError("invalid offline rollout: " + "; ".join(problems))


def _write_run(run_dir: Path, scenario_path: Path, scenario: dict,
               *, verbose: bool) -> dict:
    figures = run_dir / "figures"
    tables = run_dir / "tables"
    data = run_dir / "data"
    for directory in (figures, tables, data):
        directory.mkdir(parents=True, exist_ok=True)

    (log, captured, world, trailer, splines, horizons,
     plan_stats, mpc_times) = pursuit.run_sim(scenario, verbose=verbose)
    _validate_rollout(log, captured, splines, horizons, plan_stats)

    arrays = report._offline_arrays(log)
    replans = report._replan_rows(splines)
    sfc_boxes = report._sfc_box_rows(splines)
    summary = report._offline_summary(
        log, arrays, replans, plan_stats, scenario, sfc_boxes, world)

    pursuit.save_csv(log, data / "offline_timeseries_10hz.csv")
    report._write_csv(
        data / "path_points.csv",
        report._path_point_rows(log, splines, horizons, mpc_times))
    report._write_csv(tables / "offline_plan_attempts.csv", plan_stats)
    report._write_csv(tables / "offline_replan_metrics.csv", replans)
    report._write_csv(tables / "offline_sfc_boxes.csv", sfc_boxes)
    report._write_csv(
        tables / "summary_metrics.csv", report._summary_rows(summary, {}))
    (tables / "summary_metrics.yaml").write_text(
        yaml.safe_dump({"offline_yaml": summary}, sort_keys=False,
                       allow_unicode=True),
        encoding="utf-8")

    report._figure_pipeline_panels(
        figures, arrays, world, trailer, splines, horizons)
    report._figure_overlay(
        figures, arrays, world, trailer, splines, horizons)

    shutil.copy2(scenario_path, data / "offline_scenario.yaml")
    shutil.copy2(
        pursuit.REPO / scenario["base_map"], data / "offline_base_map.yaml")
    (run_dir / "manifest.tsv").write_text(
        "\n".join((
            f"dataset_type\t{DATASET_TYPE}",
            "result\tcaptured",
            "telemetry_scope\toffline YAML; no Gazebo/PX4/ROS telemetry",
            "drone_speed_limit_m_s\t12.0",
            "trailer_speed_m_s\t10.0",
            f"timeseries_rows\t{len(log)}",
            f"planning_attempts\t{len(plan_stats)}",
            f"accepted_replans\t{len(splines)}",
            f"mpc_horizons\t{len(horizons)}",
            f"sfc_boxes\t{len(sfc_boxes)}",
        )) + "\n",
        encoding="utf-8")
    report._manifest(run_dir)
    return summary


def _statistics(rows: list[dict]) -> list[dict]:
    statistics = []
    if not rows:
        return statistics
    for key in rows[0]:
        if key in {"run_index", "status", "error"}:
            continue
        values = [row.get(key) for row in rows]
        if any(isinstance(value, bool) for value in values):
            continue
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        finite = np.asarray([float(value) for value in values], float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            continue
        statistics.append({
            "metric": key,
            "valid_runs": len(finite),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        })
    return statistics


def _write_batch_readme(output: Path, runs: int) -> None:
    (output / "README.md").write_text(
        f"""# City YAML offline batch

- Runs requested: {runs}
- Drone speed limit/reference: 12 m/s
- Trailer speed: 10 m/s
- Scope: deterministic YAML kinematic rollout; no Gazebo, PX4, ROS, ArUco or landing telemetry

Each `run_XX/` contains the two requested path figures plus raw timeseries,
path points, planning attempts, accepted-replan geometry, raw optimizer SFC
boxes and long-form/wide summary tables. `batch_summary.csv` has one row per
successful run and `batch_statistics.csv` reports mean/std/min/max across runs.

The same scenario has no random seed or injected noise, so path geometry is
expected to repeat. Wall-clock A*/B-spline/SFC/MPC solve times may vary.
`bspline_solve_ms` includes SFC generation; do not add the separate SFC time
again when reporting total planning latency.
`sfc_violation_count` is N/A: these are optimizer control-point boxes, not the
active-polyline vehicle-containment certificate used by the ROS logger. Raw
zero-width fallback boxes remain visible and are counted separately from the
non-degenerate optimizer-box width statistics.
""",
        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--scenario", type=Path, default=pursuit.DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be a positive integer")
    scenario_path = args.scenario.expanduser().resolve()
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    _validate_scenario(scenario)
    output = (args.output.expanduser().resolve()
              if args.output is not None else _default_output())

    if args.dry_run:
        print(f"dataset: {DATASET_TYPE}")
        print(f"scenario: {scenario_path}")
        print("speeds: drone=12.0 m/s, trailer=10.0 m/s")
        print(f"runs: {args.runs}")
        print(f"output: {output}")
        return 0
    if output.exists():
        raise FileExistsError(f"batch output already exists: {output}")
    output.mkdir(parents=True)

    run_rows = []
    summaries = []
    failures = 0
    for run_index in range(1, args.runs + 1):
        run_dir = output / f"run_{run_index:02d}"
        print(f"[{run_index}/{args.runs}] city-yaml -> {run_dir}", flush=True)
        try:
            summary = _write_run(
                run_dir, scenario_path, scenario, verbose=not args.quiet)
            record = {"run_index": run_index, "status": "captured", **summary}
            summaries.append(record)
            run_rows.append({
                "run_index": run_index,
                "status": "captured",
                "error": "",
                "run_directory": str(run_dir),
                "timeseries_csv": str(
                    run_dir / "data" / "offline_timeseries_10hz.csv"),
                "summary_csv": str(
                    run_dir / "tables" / "summary_metrics.csv"),
            })
        except Exception as error:  # keep independent repetitions independent
            failures += 1
            run_rows.append({
                "run_index": run_index,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "run_directory": str(run_dir),
                "timeseries_csv": "",
                "summary_csv": "",
            })
            print(f"run {run_index} failed: {error}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break

    report._write_csv(output / "batch_runs.csv", run_rows)
    summary_fields = list(summaries[0]) if summaries else [
        "run_index", "status", "error"]
    report._write_csv(
        output / "batch_summary.csv", summaries, fieldnames=summary_fields)
    report._write_csv(
        output / "batch_statistics.csv", _statistics(summaries),
        fieldnames=("metric", "valid_runs", "mean", "std", "min", "max"))
    shutil.copy2(scenario_path, output / "scenario.yaml")
    _write_batch_readme(output, args.runs)
    report._manifest(output)
    print(f"batch artifacts: {output} ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
