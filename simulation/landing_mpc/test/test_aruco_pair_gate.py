import numpy as np

from landing_mpc.aruco_detector_node import (
    maximum_pair_disagreement,
    minimum_marker_span_px,
)


def test_pair_gate_is_inclusive_at_one_metre():
    disagreement, pair = maximum_pair_disagreement({
        0: np.array([0.0, 0.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
    })

    assert disagreement == 1.0
    assert pair == (0, 2)
    assert not disagreement > 1.0


def test_pair_gate_rejects_gross_conflict():
    disagreement, pair = maximum_pair_disagreement({
        0: np.array([0.0, 0.0, 0.0]),
        2: np.array([4.311, 0.0, 0.0]),
    })

    assert disagreement > 1.0
    assert pair == (0, 2)


def test_single_marker_has_no_pair_conflict():
    disagreement, pair = maximum_pair_disagreement({
        1: np.array([0.2, -0.1, 3.0]),
    })

    assert disagreement == 0.0
    assert pair is None


def test_nonfinite_pair_fails_closed():
    disagreement, pair = maximum_pair_disagreement({
        0: np.array([0.0, 0.0, 0.0]),
        2: np.array([float('nan'), 0.0, 0.0]),
    })

    assert disagreement == float('inf')
    assert pair == (0, 2)


def test_marker_span_rejects_edge_on_quad_despite_long_edges():
    square = np.array([
        [-30.0, -30.0],
        [30.0, -30.0],
        [30.0, 30.0],
        [-30.0, 30.0],
    ])
    angle = np.deg2rad(45.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    rotated = square @ rotation.T
    edge_on = rotated * np.array([1.0, 0.05])

    shortest_edge = min(np.linalg.norm(edge_on[(i + 1) % 4] - edge_on[i])
                        for i in range(4))
    assert np.isclose(minimum_marker_span_px(rotated), 60.0)
    assert max(np.ptp(edge_on, axis=0)) > 30.0
    assert shortest_edge > 30.0
    assert minimum_marker_span_px(edge_on) < 30.0
