"""PX4, not the companion, owns the active descent and landing verdict."""

from landing_mpc.mission_manager_node import MissionManagerNode


def test_companion_has_no_descent_or_touchdown_gate():
    assert not hasattr(MissionManagerNode, '_descent_blocked')
    assert not hasattr(MissionManagerNode, '_descent_cone_k')
    assert not hasattr(MissionManagerNode, '_touchdown_geometry')


def test_active_terminal_interface_is_px4_precland():
    assert hasattr(MissionManagerNode, '_publish_landing_target')
    assert hasattr(MissionManagerNode, '_enter_precland')
