import numpy as np
import pytest

from path_plan.sfc import SafeFlightCorridor
from path_plan.world_model import WorldModel


def _covered(corridor, a, b):
    return bool(np.any(np.all(
        (a >= corridor.boxes_min - 1.0e-9)
        & (a <= corridor.boxes_max + 1.0e-9)
        & (b >= corridor.boxes_min - 1.0e-9)
        & (b <= corridor.boxes_max + 1.0e-9), axis=1)))


def test_active_polyline_sfc_splits_conflict_and_covers_every_chord():
    world = WorldModel.from_boxes(
        [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]],
        [-10.0, -10.0, 0.0], [10.0, 10.0, 1.0],
        xy_clearance_m=2.0)
    path = np.array([
        [2.5, 2.5, 0.5], [3.0, 2.0, 0.5], [5.0, 2.0, 0.5],
    ])
    assert world.segment_is_free(path[0], path[1])
    assert not world.box_is_free(
        np.minimum(path[0], path[1]), np.maximum(path[0], path[1]))

    refined, corridor = SafeFlightCorridor(world).cover_polyline(path)
    assert len(refined) > len(path)
    assert all(world.box_is_free(lo, hi) for lo, hi in zip(
        corridor.boxes_min, corridor.boxes_max))
    assert all(_covered(corridor, a, b)
               for a, b in zip(refined[:-1], refined[1:]))
    assert all(np.all(np.minimum(right_hi, left_hi)
                      >= np.maximum(right_lo, left_lo))
               for left_lo, left_hi, right_lo, right_hi in zip(
                   corridor.boxes_min[:-1], corridor.boxes_max[:-1],
                   corridor.boxes_min[1:], corridor.boxes_max[1:]))


def test_active_polyline_sfc_fails_closed_on_a_colliding_chord():
    world = WorldModel.from_boxes(
        [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]],
        [-10.0, -10.0, 0.0], [10.0, 10.0, 1.0],
        xy_clearance_m=1.0)
    path = np.array([[-2.0, 0.5, 0.5], [3.0, 0.5, 0.5]])
    with pytest.raises(ValueError, match='colliding segment'):
        SafeFlightCorridor(world).cover_polyline(path)
