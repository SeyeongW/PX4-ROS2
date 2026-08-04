from types import SimpleNamespace

import numpy as np

from landing_mpc.mission_manager_node import MissionManagerNode


def _gate(*, height=3.0, last_seen=9.8, now=10.0, bias_n=1,
          settling=False, terminal_commit=False):
    state = SimpleNamespace(
        p_d=np.array([0.0, 0.0, height]),
        deck_z=0.0,
        vision_gate_h=3.0,
        _terminal_commit=terminal_commit,
        _t_aruco_seen=last_seen,
        vis_fresh=0.5,
        _bias_n=bias_n,
        _now=lambda: now,
        _bias_settling=lambda: settling,
    )
    return MissionManagerNode._descent_blocked(state)


def test_vision_gate_does_not_block_above_three_metres():
    assert _gate(height=3.01, last_seen=None, bias_n=0) == ''


def test_vision_gate_blocks_without_a_fresh_raw_detection():
    assert _gate(last_seen=None) == 'no fresh ArUco detection'
    assert _gate(last_seen=9.49) == 'no fresh ArUco detection'


def test_vision_gate_still_requires_a_marker_fix_and_settled_correction():
    assert _gate(bias_n=0) == 'no marker fix yet'
    assert _gate(settling=True) == 'correction still settling'


def test_vision_gate_allows_descent_with_fresh_converged_vision():
    assert _gate() == ''


def test_terminal_commit_does_not_abort_in_expected_near_field_blind_zone():
    assert _gate(height=0.6, last_seen=None, terminal_commit=True) == ''
