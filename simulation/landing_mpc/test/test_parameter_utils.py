import math

import numpy as np
import pytest

from landing_mpc.mpc import LandingMPC
from landing_mpc.parameter_utils import (
    DEFAULT_DECK_Z_M,
    DEFAULT_MARKER_IDS,
    DEFAULT_MARKER_OFFSETS_M,
    DEFAULT_MARKER_SIZES_M,
    derive_control_timing,
    require_between,
    require_leq,
    require_nonnegative,
    require_positive,
    select_control_velocity,
    validate_marker_ladder,
)
from landing_mpc.predictor import predict_const_vel
from landing_mpc.reference import HorizonReference


def test_current_timing_defaults_are_unchanged():
    control_dt, mpc_dt, solve_every = derive_control_timing(50.0, 10.0)
    assert control_dt == pytest.approx(0.02)
    assert mpc_dt == pytest.approx(0.1)
    assert solve_every == 5


def test_mpc_reference_and_predictor_share_the_derived_mpc_period():
    _, mpc_dt, _ = derive_control_timing(50.0, 10.0)
    mpc = LandingMPC(dt_s=mpc_dt, horizon=3)
    reference = HorizonReference(lead_s=mpc_dt)
    predicted_position, _, _ = predict_const_vel(
        np.zeros(3), np.array([2.0, 0.0, 0.0]), mpc_dt, 3)

    assert mpc.dt == pytest.approx(mpc_dt)
    assert reference.lead == pytest.approx(mpc_dt)
    assert predicted_position[0, 0] == pytest.approx(2.0 * mpc_dt)


@pytest.mark.parametrize(
    ('control_rate', 'mpc_rate'),
    [(0.0, 10.0), (50.0, 0.0), (10.0, 50.0), (50.0, 7.0)],
)
def test_invalid_timing_is_rejected(control_rate, mpc_rate):
    with pytest.raises(ValueError):
        derive_control_timing(control_rate, mpc_rate)


def test_current_marker_ladder_defaults_are_valid():
    validate_marker_ladder(
        DEFAULT_MARKER_IDS,
        DEFAULT_MARKER_SIZES_M,
        DEFAULT_MARKER_OFFSETS_M,
    )
    assert math.isfinite(DEFAULT_DECK_Z_M)


@pytest.mark.parametrize(
    ('ids', 'sizes', 'offsets'),
    [
        ([], [], []),
        ([0, 0], [1.3, 1.3], [1.1, 0.0, -1.1, 0.0]),
        ([0], [0.0], [0.0, 0.0]),
        ([0], [1.3], [0.0]),
        ([0, 1], [1.3], [0.0, 0.0, 0.0, 0.0]),
        ([0], [1.3], [float('nan'), 0.0]),
    ],
)
def test_invalid_marker_ladders_are_rejected(ids, sizes, offsets):
    with pytest.raises(ValueError):
        validate_marker_ladder(ids, sizes, offsets)


def test_scalar_validation_contract():
    assert require_positive('rate', 1.0) == 1.0
    assert require_nonnegative('margin', 0.0) == 0.0
    assert require_between('quality', 50, 1, 100) == 50.0
    require_leq('settle', 0.5, 'handoff', 1.0)

    for value in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            require_positive('rate', value)
    with pytest.raises(ValueError):
        require_nonnegative('margin', -0.1)
    with pytest.raises(ValueError):
        require_between('quality', 101, 1, 100)
    with pytest.raises(ValueError):
        require_leq('settle', 1.1, 'handoff', 1.0)


def test_deprecated_marker_velocity_does_not_change_control_feedforward():
    cue_velocity = np.array([3.0, -0.2, 0.0])
    for marker_velocity in (
            np.array([0.0, 0.0, 0.0]),
            np.array([99.0, -99.0, 50.0]),
            None):
        selected = select_control_velocity(cue_velocity, marker_velocity)
        assert selected is cue_velocity
