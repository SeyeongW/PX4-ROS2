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
    world = _open_world([[49.95, -0.10, -1.0]], [[50.05, 0.10, 1.0]])
    result = BsplineOptimizer(
        world, cruise_speed_m_s=4.0, ctrl_spacing_m=100.0,
        max_rebound=2, strict_validation=True).optimize(
            np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))
    assert result.solver_success
    assert not result.collision_free
    assert result.free_fraction < 1.0
    assert not result.accepted


def test_control_spacing_is_an_upper_bound():
    length, spacing = 10.1, 2.0
    waypoints = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    legacy = BsplineOptimizer(
        _open_world(), cruise_speed_m_s=None,
        ctrl_spacing_m=spacing).optimize(waypoints)
    strict = BsplineOptimizer(
        _open_world(), cruise_speed_m_s=None, ctrl_spacing_m=spacing,
        strict_validation=True).optimize(waypoints)
    legacy_intervals = len(legacy.spline.q) - 3
    strict_intervals = len(strict.spline.q) - 3
    assert length / legacy_intervals > spacing
    assert strict.accepted
    assert length / strict_intervals <= spacing


@pytest.mark.parametrize('solver_success, nonfinite', [
    (False, False), (True, True),
])
def test_optimizer_preserves_and_rejects_bad_solver_result(
        monkeypatch, solver_success, nonfinite):
    def fake_minimize(_fun, x0, **_kwargs):
        value = np.full_like(x0, np.nan) if nonfinite else x0.copy()
        return SimpleNamespace(
            success=solver_success, status=2,
            message='synthetic solver result', x=value,
            fun=np.nan if nonfinite else 0.0)

    monkeypatch.setattr(optimizer_module, 'minimize', fake_minimize)
    result = BsplineOptimizer(
        _open_world(), ctrl_spacing_m=2.0,
        strict_validation=True).optimize(
            np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    assert result.solver_success is solver_success
    assert result.solution_finite is (not nonfinite)
    assert result.solver_status == 2
    assert result.solver_message == 'synthetic solver result'
    assert not result.accepted


def test_bspline_node_preserves_legacy_publish_contract(monkeypatch):
    published, observations, infos = [], [], []
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    legacy_result = SimpleNamespace(
        accepted=False, solver_success=False, solver_status=2,
        solver_message='synthetic failure', solution_finite=True,
        collision_free=True, free_fraction=1.0, rebound_iters=1,
        sfc_generation_time_s=0.002,
        corridor=SimpleNamespace(
            boxes_min=np.zeros((1, 3)), boxes_max=np.ones((1, 3))),
        spline=SimpleNamespace(
            sample=lambda _count: (
                np.array([0.0, 1.0]), positions, np.zeros_like(positions),
                np.zeros_like(positions)),
            duration=lambda: 1.0))
    state = SimpleNamespace(
        optimizer=SimpleNamespace(optimize=lambda _wp: legacy_result),
        samples=2,
        traj_pub=SimpleNamespace(publish=published.append),
        path_pub=SimpleNamespace(publish=published.append),
        marker_pub=SimpleNamespace(publish=published.append),
        sfc_pub=SimpleNamespace(publish=observations.append),
        sfc_stats_pub=SimpleNamespace(publish=observations.append),
        _to_markers=lambda _lo, _hi: 'markers',
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: object())),
        get_logger=lambda: SimpleNamespace(info=infos.append))
    monkeypatch.setattr(
        node_module, 'path_to_positions',
        lambda _msg: positions)
    monkeypatch.setattr(node_module, 'trajectory_to_msg', lambda *_args: 'traj')
    monkeypatch.setattr(node_module, 'positions_to_path', lambda *_args: 'path')
    BSplineNode._on_path(state, object())
    assert published == ['traj', 'path', 'markers']
    assert len(observations) == 2
    assert list(observations[1].data) == [2.0, 1.0, 1.0, 1.0]
    assert len(infos) == 1
