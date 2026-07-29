"""Pin deprecated mission inputs before any later removal."""

import ast
from pathlib import Path


MISSION_PATH = Path(__file__).parents[1].joinpath(
    'landing_mpc', 'mission_manager_node.py')
MISSION_SOURCE = MISSION_PATH.read_text(encoding='utf-8')
MISSION_TREE = ast.parse(MISSION_SOURCE)


def _self_attribute(node, *names):
    current = node
    for name in reversed(names):
        if not isinstance(current, ast.Attribute) or current.attr != name:
            return False
        current = current.value
    return isinstance(current, ast.Name) and current.id == 'self'


def _function(name):
    return next(
        node for node in ast.walk(MISSION_TREE)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_contact_timeout_is_declared_but_not_read_by_control():
    assert "'contact_timeout_s'" in MISSION_SOURCE
    accesses = [
        node for node in ast.walk(MISSION_TREE)
        if isinstance(node, ast.Attribute)
        if node.attr == '_deprecated_contact_timeout_s'
    ]
    assert sum(isinstance(node.ctx, ast.Store) for node in accesses) == 1
    assert not any(isinstance(node.ctx, ast.Load) for node in accesses)
    assert 'contact_timeout_s is deprecated and has no effect' in MISSION_SOURCE


def test_legacy_v_max_is_only_a_seed_and_live_solve_uses_los_cap():
    assert "'v_max'" in MISSION_SOURCE
    constructors = [
        node for node in ast.walk(MISSION_TREE)
        if isinstance(node, ast.Call)
        if isinstance(node.func, ast.Name)
        if node.func.id == 'LandingMPC'
    ]
    assert len(constructors) == 1
    seed = next(
        keyword.value for keyword in constructors[0].keywords
        if keyword.arg == 'v_max'
    )
    assert _self_attribute(seed, '_deprecated_v_max')

    tick = _function('_tick')
    live_assignments = [
        node for node in ast.walk(tick)
        if isinstance(node, ast.Assign)
        if any(_self_attribute(target, 'mpc', 'v_max')
               for target in node.targets)
    ]
    assert len(live_assignments) == 1
    live_value = live_assignments[0].value
    assert isinstance(live_value, ast.Call)
    assert isinstance(live_value.func, ast.Attribute)
    assert _self_attribute(live_value.func, '_los_speed_cap')
    solves = [
        node for node in ast.walk(tick)
        if isinstance(node, ast.Call)
        if isinstance(node.func, ast.Attribute)
        if _self_attribute(node.func, 'mpc', 'solve')
    ]
    assert solves
    assert live_assignments[0].lineno < min(node.lineno for node in solves)
    assert 'v_max is deprecated in the current DESCEND path' in MISSION_SOURCE


def test_marker_velocity_consumer_is_retained_but_explicitly_deprecated():
    assert "'/marker/velocity'" in MISSION_SOURCE
    assert 'self._on_vis_v' in MISSION_SOURCE
    assert 'select_control_velocity(' in MISSION_SOURCE
    assert '/marker/velocity is a deprecated mission input' in MISSION_SOURCE
