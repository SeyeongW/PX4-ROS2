"""Planner envelope for the opt-in STL high-speed research profile (not flight validation)."""
import numpy as np
import pytest

from jol_perception.observation_path import sample_plan
from jol_perception.static_flight_core import build_static_plan

HOME, TARGET = [-2., 0., .2045], [0., -40., 10.]
FAST = dict(path_kind='spline', takeoff_height=3., vertical_handle=4., max_speed=22.2, max_acceleration=27.,
            max_jerk=81., continuous_takeoff=True, takeoff_join_speed=.35, target_is_virtual_waypoint=True,
            standoff=0., target_is_pass_through_gate=True, gate_exit_distance=20., body_radius=.3,
            takeoff_min_duration=6., controller_profile='stl_fast_research')


def test_fast_research_plan_passes_target_near_20mps_with_positive_thrust():
    plan = build_static_plan(HOME, TARGET, **FAST)
    assert plan['controller_profile'] == 'stl_fast_research'
    times = np.arange(0., plan['duration_s'], .01)
    rows = [sample_plan(plan, t) for t in times]
    v = np.array([r['velocity'] for r in rows]); a = np.array([r['acceleration'] for r in rows])
    p = np.array([r['position'] for r in rows])
    at_target = np.argmin(np.linalg.norm(p-TARGET, axis=1))
    assert 18. < np.linalg.norm(v[at_target]) <= 20.
    thrust = a+[0., 0., 9.80665]
    assert thrust[:, 2].min() > 0  # never asks for downward rotor thrust
    tilt = np.degrees(np.arccos(thrust[:, 2]/np.linalg.norm(thrust, axis=1)))
    assert tilt.max() < 80.


@pytest.mark.parametrize('profile,limits', [
    ('stl_fast_research', dict(max_speed=26.)),
    ('stl_fast_research', dict(max_acceleration=31.)),
    ('legacy', dict(max_speed=20., max_acceleration=.6, max_jerk=1.2)),
])
def test_envelopes_still_reject_out_of_range_limits(profile, limits):
    with pytest.raises(ValueError, match='limits_exceed'):
        build_static_plan(HOME, TARGET, **dict(FAST, controller_profile=profile, **limits))
