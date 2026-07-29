import numpy as np

from landing_mpc.aruco_detector_node import maximum_pair_disagreement


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
