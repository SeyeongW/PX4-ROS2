import math

import numpy as np
import pytest

from path_plan.world_model import (
    WorldModel,
    _clip_segment_to_z_span,
    _point_segment_distance_squared_xy,
    _segment_aabb_distance_squared_xy,
)


def _world(clearance=0.0):
    return WorldModel.from_boxes(
        [[0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0]],
        [-10.0, -10.0, -10.0],
        [10.0, 10.0, 10.0],
        xy_clearance_m=clearance,
    )


def _numpy_point_segment_distance_squared_xy(point, a, b):
    direction = b - a
    length_squared = float(direction @ direction)
    if length_squared == 0.0:
        delta = point - a
        return float(delta @ delta)
    fraction = float(np.clip(
        (point - a) @ direction / length_squared, 0.0, 1.0))
    delta = point - (a + fraction * direction)
    return float(delta @ delta)


def _segment_is_free_without_broad_phase(world, a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    direction = b - a
    for low, high in zip(world.boxes_min, world.boxes_max):
        clipped = _clip_segment_to_z_span(a, direction, low[2], high[2])
        if clipped is None:
            continue
        start_xy = a[:2] + clipped[0] * direction[:2]
        end_xy = a[:2] + clipped[1] * direction[:2]
        if (_segment_aabb_distance_squared_xy(
                start_xy, end_xy, low[:2], high[:2])
                <= world.xy_clearance_m ** 2):
            return False
    return True


def test_zero_clearance_preserves_aabb_boundary_contract():
    world = _world()

    assert not world.is_free([[0.0, 0.5, 0.5]])[0]
    assert world.is_free([[-0.01, 0.5, 0.5]])[0]
    assert not world.segment_is_free([-1.0, 0.5, 0.5], [2.0, 0.5, 0.5])
    assert world.segment_is_free([-1.0, 1.01, 0.5], [2.0, 1.01, 0.5])
    assert not world.box_is_free([1.0, 0.2, 0.2], [2.0, 0.8, 0.8])
    assert world.clearance([2.0, 2.0, 2.0]) == pytest.approx(math.sqrt(3.0))


def test_xy_clearance_is_round_at_aabb_corners_and_respects_z():
    world = _world(2.0)

    free = world.is_free([
        [3.0, 0.5, 0.5],       # exactly 2 m from a side
        [2.4, 2.4, 0.5],       # 1.98 m from the upper-right corner
        [2.5, 2.5, 0.5],       # 2.12 m: free despite square inflation
        [0.5, 0.5, 1.01],      # horizontal dilation does not grow Z
    ])
    assert free.tolist() == [False, False, True, True]


def test_segment_uses_exact_continuous_distance_to_rounded_aabb():
    world = _world(2.0)

    # Both endpoints are clear, but the segment crosses the rounded corner.
    assert world.is_free([[-2.5, 2.5, 0.5], [2.5, -2.5, 0.5]]).all()
    assert not world.segment_is_free(
        [-2.5, 2.5, 0.5], [2.5, -2.5, 0.5], step_m=100.0)

    # This segment lies inside the old square inflation but over 2 m from the
    # physical upper-right corner throughout.
    assert world.segment_is_free([2.5, 3.0, 0.5], [3.0, 2.5, 0.5])
    assert world.segment_is_free([-2.5, 2.5, 1.01], [2.5, -2.5, 1.01])


def test_segment_broad_phase_keeps_tangent_zero_length_and_z_contracts():
    world = _world(2.0)

    # The broad phase must not discard exact XY tangency, including a point.
    assert not world.segment_is_free([3.0, -2.0, 0.5], [3.0, 3.0, 0.5])
    assert not world.segment_is_free([3.0, 0.5, 0.5], [3.0, 0.5, 0.5])
    assert world.segment_is_free([3.001, 0.5, 0.5], [3.001, 0.5, 0.5])

    # XY clearance does not extend Z, whose physical boundary is inclusive.
    assert world.segment_is_free([3.0, 0.5, 2.0], [3.0, 0.5, 1.001])
    assert not world.segment_is_free([3.0, 0.5, 2.0], [3.0, 0.5, 1.0])


def test_optimized_segment_math_matches_numpy_and_exhaustive_references():
    rng = np.random.default_rng(20260818)
    for index in range(2000):
        point = rng.uniform(-20.0, 20.0, size=2)
        start = rng.uniform(-20.0, 20.0, size=2)
        end = start if index % 20 == 0 else rng.uniform(-20.0, 20.0, size=2)
        expected = _numpy_point_segment_distance_squared_xy(point, start, end)
        actual = _point_segment_distance_squared_xy(point, start, end)
        assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)

    world = WorldModel.from_boxes(
        [[0.0, 0.0, 0.0], [-8.0, 4.0, -2.0], [5.0, -7.0, 2.0]],
        [[1.0, 1.0, 1.0], [-6.0, 7.0, 3.0], [9.0, -4.0, 6.0]],
        [-20.0, -20.0, -10.0],
        [20.0, 20.0, 10.0],
        xy_clearance_m=1.5,
    )
    for index in range(2000):
        start = rng.uniform(world.bounds_min, world.bounds_max)
        end = start if index % 20 == 0 else rng.uniform(
            world.bounds_min, world.bounds_max)
        assert world.segment_is_free(start, end) == \
            _segment_is_free_without_broad_phase(world, start, end)


def test_box_and_clearance_use_the_same_rounded_xy_geometry():
    world = _world(2.0)

    assert not world.box_is_free([2.4, 2.4, 0.2], [2.5, 2.5, 0.8])
    assert world.box_is_free([2.5, 2.5, 0.2], [2.6, 2.6, 0.8])
    assert world.box_is_free([0.2, 0.2, 1.01], [0.8, 0.8, 2.0])

    corner_residual = math.sqrt(1.5 ** 2 + 1.5 ** 2) - 2.0
    assert world.clearance([2.5, 2.5, 0.5]) == pytest.approx(corner_residual)
    assert world.clearance([2.5, 2.5, 2.0]) == pytest.approx(
        math.hypot(corner_residual, 1.0))
    assert world.clearance([0.5, 0.5, 2.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan, [1.0]])
def test_xy_clearance_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="xy_clearance_m"):
        _world(value)
