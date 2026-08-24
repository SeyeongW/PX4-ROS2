from pathlib import Path
import sys

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import paper_dynamic_planning_report as report
import run_city_yaml_offline_batch as batch


def test_optimizer_sfc_rows_keep_degenerate_box_visible():
    corridor = type("Corridor", (), {
        "__len__": lambda self: len(self.boxes_min),
    })()
    corridor.boxes_min = np.array([
        [0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    corridor.boxes_max = np.array([
        [0.0, 0.0, 0.0], [4.0, 8.0, 10.0]])
    spline = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    astar = spline.copy()

    boxes = report._sfc_box_rows([(1.0, spline, corridor, astar)])
    replans = report._replan_rows([(1.0, spline, corridor, astar)])

    assert [row["is_degenerate"] for row in boxes] == [1, 0]
    assert replans[0]["sfc_min_width_m"] == 0.0
    assert replans[0]["sfc_degenerate_box_count"] == 1
    assert replans[0]["sfc_non_degenerate_min_width_m"] == 2.0


def test_batch_statistics_ignore_unavailable_nan_metrics():
    rows = [
        {"run_index": 1, "status": "captured", "metric": 1.0,
         "unavailable": float("nan")},
        {"run_index": 2, "status": "captured", "metric": 3.0,
         "unavailable": float("nan")},
    ]

    statistics = {row["metric"]: row for row in batch._statistics(rows)}

    assert statistics["metric"]["mean"] == 2.0
    assert statistics["metric"]["valid_runs"] == 2
    assert "unavailable" not in statistics
