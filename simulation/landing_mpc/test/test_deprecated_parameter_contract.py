"""Keep retired companion-side flight controls out of the live mission."""

from pathlib import Path


MISSION_SOURCE = Path(__file__).parents[1].joinpath(
    'landing_mpc', 'mission_manager_node.py').read_text(encoding='utf-8')


def test_custom_terminal_control_parameters_are_removed():
    for name in (
            'contact_timeout_s', 'touchdown_height_m', 'z_floor_margin_m',
            'vision_gate_height_m', 'vision_align_depression_deg',
            'approach_alt', 'acquire_xy_m', 'v_max'):
        assert f"'{name}'" not in MISSION_SOURCE


def test_retired_touchdown_and_forced_disarm_are_not_restored():
    assert 'LandingMPC(' in MISSION_SOURCE
    assert "'LANDING_DESCEND'" in MISSION_SOURCE
    assert '_descent_blocked' not in MISSION_SOURCE
    assert '_send_capped' not in MISSION_SOURCE
    assert 'VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 21196' not in MISSION_SOURCE
    assert "'TOUCHDOWN'" not in MISSION_SOURCE
    assert "'PRECLAND'" in MISSION_SOURCE


def test_deprecated_marker_velocity_control_input_is_removed():
    assert "'/marker/velocity'" not in MISSION_SOURCE
    assert '_on_vis_v' not in MISSION_SOURCE
    assert 'select_control_velocity(' not in MISSION_SOURCE
    assert "'/marker/position'" in MISSION_SOURCE
    assert "'/marker/valid'" in MISSION_SOURCE
    assert "'/marker/entry_valid'" in MISSION_SOURCE
