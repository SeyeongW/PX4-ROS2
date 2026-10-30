"""Launch/model consistency and the actual 40m/10m planner contract."""
import inspect
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from launch import LaunchContext


LAUNCH = Path(__file__).parents[1] / 'launch/gazebo.launch.py'


@pytest.mark.parametrize('key,value', [
    ('scenario', 'multi'), ('flight_backend', 'native'), ('aircraft_type', 'iris_proxy'),
])
def test_incompatible_model_or_controller_fails_before_launch(key, value):
    context = LaunchContext()
    context.launch_configurations.update(scenario='static_flight', flight_backend='mavros',
                                        aircraft_type='stl_pacer_quad')
    context.launch_configurations[key] = value
    with pytest.raises(ValueError, match='balloon_gazebo_requires_' + key):
        run_path(str(LAUNCH))['setup'](context)


def test_default_path_keeps_3m_c2_takeoff_and_horizontal_goal_approach():
    from jol_perception.static_flight_core import build_static_plan, validate_tailsitter_plan_dynamics
    from jol_perception.observation_path import sample_plan

    description = run_path(str(LAUNCH))['generate_launch_description']()
    context = LaunchContext()
    defaults = {entity.name: ''.join(s.perform(context) for s in entity.default_value)
                for entity in description.entities if hasattr(entity, 'default_value')}
    profile_path=Path(defaults['physics_profile_path'])
    profile = json.loads(profile_path.read_text())
    options = {k: v for k, v in profile['flight_parameters'].items()
               if k in inspect.signature(build_static_plan).parameters}
    plan = build_static_plan(profile['home_world'], [0., -40., 10.],
                             controller_profile='legacy',
                             vertical_handle=float(defaults['vertical_handle']), **options)
    takeoff, curve = plan['segments']
    join = takeoff['duration_s']
    left, right = sample_plan(plan, join - 1e-6), sample_plan(plan, join + 1e-6)
    for field in ('position', 'velocity', 'acceleration'):
        np.testing.assert_allclose(left[field], right[field], atol=2e-5)
    assert left['velocity'][2] == pytest.approx(profile['flight_parameters']['takeoff_join_speed'],abs=2e-5)
    np.testing.assert_allclose(np.array(curve['control_points'])[0] - plan['home'], [0., 0., 3.])
    # A remaining upward tangent used to trigger late climb/back-transition errors.
    assert abs(curve['control_points'][-1][2] - curve['control_points'][-2][2]) < 1e-10
    assert plan['motion_validation']['passed']
