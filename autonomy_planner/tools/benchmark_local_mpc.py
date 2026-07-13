#!/usr/bin/env python3
"""Benchmark the configured jerk-MPC horizon candidates without ROS.

The benchmark deliberately uses the same deterministic straight reference for
every candidate.  It is a solver screening test, not a Gazebo collision or
tracking-success claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from autonomy_planner.mission_core import JerkInputMPC


def benchmark(horizon: int, dt_s: float, trials: int) -> dict[str, object]:
    solver = JerkInputMPC(
        dt_s=dt_s,
        horizon_steps=horizon,
        deadline_s=10.0,
        maximum_iterations=15,
    )
    reference = np.column_stack(
        (
            np.linspace(0.0, 4.0, horizon),
            np.zeros(horizon),
            np.full(horizon, -10.0),
        )
    )
    elapsed_ms: list[float] = []
    iterations: list[int] = []
    success = 0
    for _ in range(trials):
        started = time.monotonic()
        result = solver.solve(
            np.array([0.0, 0.0, -10.0]),
            np.zeros(3),
            np.zeros(3),
            reference,
            np.zeros_like(reference),
        )
        elapsed_ms.append(1000.0 * (time.monotonic() - started))
        iterations.append(result.iterations)
        success += int(result.success)
    return {
        "horizon_steps": horizon,
        "dt_s": dt_s,
        "prediction_horizon_s": horizon * dt_s,
        "trials": trials,
        "successes": success,
        "median_solve_ms": float(np.median(elapsed_ms)),
        "p95_solve_ms": float(np.percentile(elapsed_ms, 95.0)),
        "maximum_solve_ms": float(max(elapsed_ms)),
        "median_iterations": float(np.median(iterations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.trials < 1:
        parser.error("--trials must be positive")
    document = {
        "scope": "deterministic offline solver screening; not a flight claim",
        "deadline_ms": 40.0,
        "candidates": [
            benchmark(horizon, dt_s, arguments.trials)
            for horizon in (15, 20, 25)
            for dt_s in (0.10, 0.15)
        ],
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
