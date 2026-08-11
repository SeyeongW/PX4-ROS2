from types import SimpleNamespace

import numpy as np
import pytest

import path_plan.bspline_node as node_module
import path_plan.bspline_optimizer as optimizer_module
from path_plan.bspline_node import BSplineNode
from path_plan.bspline_optimizer import BsplineOptimizer
from path_plan.world_model import WorldModel


def _open_world(boxes_min=(), boxes_max=()):
    return WorldModel.from_boxes(
        boxes_min, boxes_max,
        [-10.0, -10.0, -10.0], [110.0, 10.0, 10.0])


def test_optimizer_exactly_rejects_thin_obstacle_between_samples():
    world = _open_world(
        [[49.95, -0.10, -1.0]], [[50.05, 0.10, 1.0]])
    result = BsplineOptimizer(
        world, cruise_speed_m_s=4.0, ctrl_spacing_m=100.0,
        max_rebound=2,
    ).optimize(np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))

    assert result.solver_success
    assert not result.collision_free
    assert result.free_fraction < 1.0
    assert not result.accepted


def test_control_spacing_is_an_upper_bound():
    length = 10.1
    spacing = 2.0
    result = BsplineOptimizer(
        _open_world(), cruise_speed_m_s=None, ctrl_spacing_m=spacing,
    ).optimize(np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]))

    interpolation_intervals = len(result.spline.q) - 3
    assert result.accepted
    assert length / interpolation_intervals <= spacing


@pytest.mark.parametrize("solver_success, nonfinite", [
    (False, False),
    (True, True),
])
def test_optimizer_preserves_and_rejects_bad_solver_result(
        monkeypatch, solver_success, nonfinite):
    def fake_minimize(_fun, x0, **_kwargs):
        value = np.full_like(x0, np.nan) if nonfinite else x0.copy()
        return SimpleNamespace(
            success=solver_success,
            status=2,
            message="synthetic solver result",
            x=value,
            fun=np.nan if nonfinite else 0.0,
        )

    monkeypatch.setattr(optimizer_module, "minimize", fake_minimize)
    result = BsplineOptimizer(
        _open_world(), ctrl_spacing_m=2.0,
    ).optimize(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))

    assert result.solver_success is solver_success
    assert result.solution_finite is (not nonfinite)
    assert result.solver_status == 2
    assert result.solver_message == "synthetic solver result"
    assert not result.accepted


def test_bspline_node_does_not_publish_rejected_result(monkeypatch):
    published = []
    errors = []
    rejected = SimpleNamespace(
        accepted=False,
        solver_success=False,
        solver_status=2,
        solver_message="synthetic failure",
        solution_finite=True,
        collision_free=False,
        free_fraction=0.5,
    )
    state = SimpleNamespace(
        optimizer=SimpleNamespace(optimize=lambda _wp: rejected),
        traj_pub=SimpleNamespace(publish=published.append),
        path_pub=SimpleNamespace(publish=published.append),
        marker_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(error=errors.append),
    )
    monkeypatch.setattr(
        node_module, "path_to_positions",
        lambda _msg: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

    BSplineNode._on_path(state, object())

    assert published == []
    assert len(errors) == 1 and "trajectory rejected" in errors[0]


def test_bspline_node_exactly_checks_the_published_chords(monkeypatch):
    published = []
    errors = []
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    accepted = SimpleNamespace(
        accepted=True,
        spline=SimpleNamespace(sample=lambda _count: (
            np.array([0.0, 1.0]), positions, np.zeros_like(positions),
            np.zeros_like(positions))),
    )
    state = SimpleNamespace(
        optimizer=SimpleNamespace(
            optimize=lambda _wp: accepted,
            world=WorldModel.from_boxes(
                [[0.45, -0.05, -0.05]], [[0.55, 0.05, 0.05]],
                [-1.0, -1.0, -1.0], [2.0, 1.0, 1.0])),
        samples=2,
        traj_pub=SimpleNamespace(publish=published.append),
        path_pub=SimpleNamespace(publish=published.append),
        marker_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(error=errors.append),
    )
    monkeypatch.setattr(
        node_module, "path_to_positions", lambda _msg: positions)

    BSplineNode._on_path(state, object())

    assert published == []
    assert errors == [
        "trajectory rejected: published samples are not exact-safe"]
