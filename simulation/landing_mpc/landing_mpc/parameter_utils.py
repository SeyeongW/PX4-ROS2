"""Shared defaults and fail-fast validation for ``landing_mpc`` parameters.

Keep this module ROS-free so the parameter contract can be unit-tested without
starting nodes.  These checks intentionally reject ambiguous configurations at
startup; they do not change any valid default-flight behaviour.
"""

from __future__ import annotations

import math
from typing import Sequence, TypeVar


DEFAULT_DECK_Z_M = 1.811
DEFAULT_MARKER_IDS = (0, 2, 1)
DEFAULT_MARKER_SIZES_M = (1.3, 1.3, 0.30)
DEFAULT_MARKER_OFFSETS_M = (1.1, 0.0, -1.1, 0.0, 0.0, 0.0)
DEFAULT_MAX_PAIR_DISAGREEMENT_M = 1.0

VelocityT = TypeVar('VelocityT')


def select_control_velocity(
        cue_velocity: VelocityT,
        _deprecated_marker_velocity: object = None) -> VelocityT:
    """Return the continuous cue velocity used by control.

    ``/marker/velocity`` remains a deprecated mission input during the
    compatibility window, but its intermittent estimate must not replace the
    trailer cue feed-forward.
    """
    return cue_velocity


def require_finite(name: str, value: float) -> float:
    """Return ``value`` as float, or reject NaN/inf."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return result


def require_positive(name: str, value: float) -> float:
    """Return a finite value strictly greater than zero."""
    result = require_finite(name, value)
    if result <= 0.0:
        raise ValueError(f'{name} must be > 0, got {result}')
    return result


def require_nonnegative(name: str, value: float) -> float:
    """Return a finite value greater than or equal to zero."""
    result = require_finite(name, value)
    if result < 0.0:
        raise ValueError(f'{name} must be >= 0, got {result}')
    return result


def require_between(
        name: str, value: float, lower: float, upper: float,
        *, upper_inclusive: bool = True) -> float:
    """Return a finite bounded value."""
    result = require_finite(name, value)
    upper_ok = result <= upper if upper_inclusive else result < upper
    if result < lower or not upper_ok:
        right = ']' if upper_inclusive else ')'
        raise ValueError(
            f'{name} must be in [{lower}, {upper}{right}, got {result}')
    return result


def require_nonempty(name: str, value: str) -> str:
    """Reject blank required topic, frame, world or model names."""
    result = str(value).strip()
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result


def require_leq(
        lower_name: str, lower: float, upper_name: str, upper: float,
        *, strict: bool = False) -> None:
    """Validate an ordered pair after checking both values are finite."""
    low = require_finite(lower_name, lower)
    high = require_finite(upper_name, upper)
    valid = low < high if strict else low <= high
    if not valid:
        op = '<' if strict else '<='
        raise ValueError(
            f'{lower_name} must be {op} {upper_name}, got {low} and {high}')


def derive_control_timing(
        control_rate_hz: float,
        mpc_rate_hz: float) -> tuple[float, float, int]:
    """Derive both periods and the integer re-plan stride from two rates."""
    control_rate = require_positive('control_rate_hz', control_rate_hz)
    mpc_rate = require_positive('mpc_rate_hz', mpc_rate_hz)
    ratio = control_rate / mpc_rate
    solve_every = int(round(ratio))
    if solve_every < 1 or not math.isclose(
            ratio, solve_every, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            'control_rate_hz must be an integer multiple of mpc_rate_hz, '
            f'got {control_rate:g} and {mpc_rate:g}')
    return 1.0 / control_rate, 1.0 / mpc_rate, solve_every


def validate_marker_ladder(
        marker_ids: Sequence[int],
        marker_sizes_m: Sequence[float],
        marker_offsets_m: Sequence[float]) -> None:
    """Validate the parallel marker-ladder arrays used by the detector."""
    ids = [int(value) for value in marker_ids]
    sizes = [require_positive('marker_sizes_m item', value)
             for value in marker_sizes_m]
    offsets = [require_finite('marker_offsets_m item', value)
               for value in marker_offsets_m]
    if not ids:
        raise ValueError('marker_ids must contain at least one marker')
    if len(set(ids)) != len(ids):
        raise ValueError(f'marker_ids must be unique, got {ids}')
    if len(offsets) % 2:
        raise ValueError(
            'marker_offsets_m must contain x/y pairs, '
            f'got {len(offsets)} values')
    if not (len(ids) == len(sizes) == len(offsets) // 2):
        raise ValueError(
            f'marker_ids({len(ids)}), marker_sizes_m({len(sizes)}) and '
            f'marker_offsets_m({len(offsets)}/2) must describe the same '
            'number of markers')
