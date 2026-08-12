import math

import pytest

from path_plan.world_model import WorldModel


def _world(clearance=0.0):
    return WorldModel.from_boxes(
        [[0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0]],
        [-10.0, -10.0, -10.0],
        [10.0, 10.0, 10.0],
        xy_clearance_m=clearance,
    )


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
