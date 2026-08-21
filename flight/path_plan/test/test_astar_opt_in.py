from path_plan.astar import AStarPlanner3D
from path_plan.world_model import WorldModel


def test_exact_edges_and_zero_clearance_cost_are_opt_in(monkeypatch):
    world = WorldModel.from_boxes(
        [[0.45, 0.45, -1.0]], [[0.55, 0.55, 1.0]],
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0])

    def unexpected_clearance(_self, _point):
        raise AssertionError('zero clearance weight must skip clearance()')

    monkeypatch.setattr(WorldModel, 'clearance', unexpected_clearance)
    common = dict(
        resolution_m=1.0, clearance_weight=0.0,
        altitude_weight=0.0, climb_weight=0.0)
    legacy = AStarPlanner3D(world, **common).plan(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    strict = AStarPlanner3D(world, exact_edges=True, **common).plan(
        [0.0, 0.0, 0.0], [1.0, 1.0, 0.0])

    assert legacy.success and len(legacy.waypoints_m) == 2
    assert not world.segment_is_free_exact(
        legacy.waypoints_m[0], legacy.waypoints_m[-1])
    assert strict.success and len(strict.waypoints_m) > 2
    assert all(world.segment_is_free_exact(a, b)
               for a, b in zip(
                   strict.waypoints_m[:-1], strict.waypoints_m[1:]))
