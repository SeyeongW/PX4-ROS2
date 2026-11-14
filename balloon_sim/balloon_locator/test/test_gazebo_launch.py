"""Launch/model consistency and the actual 40m/10m planner contract."""
import inspect
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from launch import LaunchContext


LAUNCH = Path(__file__).parents[1] / 'launch/gazebo.launch.py'


@pytest.mark.parametrize('transparency', ['0.0', '0.84'])
def test_exterior_view_ignores_old_cutaway_launch_override(tmp_path, monkeypatch, transparency):
    launch = run_path(str(LAUNCH))
    context = LaunchContext()
    for entity in launch['generate_launch_description']().entities:
        if hasattr(entity, 'default_value'):
            context.launch_configurations[entity.name] = ''.join(s.perform(context) for s in entity.default_value)
    context.launch_configurations.update(static_output=str(tmp_path), airframe_transparency=transparency)
    # Generate the real world without preparing or starting a flight controller.
    monkeypatch.setitem(launch['setup'].__globals__, 'start_stl_mavros', lambda *args, **kwargs: [])
    launch['setup'](context)
    model = ET.parse(tmp_path / 'balloon.world').getroot().find("world/model[@name='stl_pacer_quad']")
    assert model is not None
    shells = [v for v in model.iter('visual') if v.find('transparency') is not None]
    assert len(shells) == 8
    assert all(float(v.findtext('transparency')) == 0.0 for v in shells)


def test_onboard_mount_tf_uses_model_pose_and_ekf_body_frame():
    from launch.utilities import perform_substitutions
    from scipy.spatial.transform import Rotation
    context=LaunchContext()
    report={'onboard_camera':{'optical_position_body_m':[0.,0.,.1997],
                              'frame_name':'jol/onboard_camera_optical'}}
    node=run_path(str(LAUNCH))['onboard_camera_transform'](report)
    command=[]
    for arg in node.cmd[1:]:
        item=perform_substitutions(context,arg)
        if item=='--ros-args':break
        command.append(item)
    def value(flag):return command[command.index(flag)+1]
    assert value('--frame-id')=='jol/observer_body'
    assert value('--child-frame-id')=='jol/onboard_camera_optical'
    assert float(value('--z'))==pytest.approx(.1997)
    rotation=Rotation.from_euler('xyz',[float(value('--roll')),float(value('--pitch')),float(value('--yaw'))])
    np.testing.assert_allclose(rotation.as_matrix(),np.eye(3),atol=1e-12)


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


def test_default_path_keeps_3m_c2_takeoff_and_crosses_sensor_gate():
    from jol_perception.static_flight_core import build_static_plan
    from jol_perception.observation_path import sample_plan

    description = run_path(str(LAUNCH))['generate_launch_description']()
    context = LaunchContext()
    defaults = {entity.name: ''.join(s.perform(context) for s in entity.default_value)
                for entity in description.entities if hasattr(entity, 'default_value')}
    assert defaults['mode'] == 'mission' and defaults['experimental_tailsitter'] == 'false'
    assert [float(defaults[k]) for k in ('target_range','target_height','target_azimuth')] == [40.,10.,-90.]
    profile_path=Path(defaults['physics_profile_path'])
    profile = json.loads(profile_path.read_text())
    options = {k: v for k, v in profile['flight_parameters'].items()
               if k in inspect.signature(build_static_plan).parameters}
    plan = build_static_plan(profile['home_world'], [0., -40., 10.],
                             vertical_handle=float(defaults['vertical_handle']), **options)
    takeoff, curve = plan['segments']
    join = takeoff['duration_s']
    left, right = sample_plan(plan, join - 1e-6), sample_plan(plan, join + 1e-6)
    for field in ('position', 'velocity', 'acceleration'):
        np.testing.assert_allclose(left[field], right[field], atol=2e-5)
    assert left['velocity'][2] == pytest.approx(profile['flight_parameters']['takeoff_join_speed'],abs=2e-5)
    np.testing.assert_allclose(np.array(curve['control_points'])[0] - plan['home'], [0., 0., 3.])
    points = np.array(curve['control_points'])
    u = plan['gate']['cubic_parameter_at_center']
    crossing = (1-u)**3*points[0]+3*(1-u)**2*u*points[1]+3*(1-u)*u*u*points[2]+u**3*points[3]
    np.testing.assert_allclose(crossing, [0.,-40.,10.], atol=1e-9)
    assert np.linalg.norm(np.array(plan['goal'])-[0.,-40.,10.]) == pytest.approx(5.)
    assert plan['gate']['crossing_policy'] == 'forward_measured_segment_plane_intersection_with_body_margin'
    assert plan['motion_validation']['passed']


@pytest.mark.parametrize('mode,experimental', [('pymav','false'),('mission','true'),('vision_center','false')])
def test_controller_and_physics_mismatch_rejected(mode, experimental):
    launch = run_path(str(LAUNCH))
    context = LaunchContext()
    context.launch_configurations.update(scenario='static_flight',flight_backend='mavros',
        aircraft_type='stl_pacer_quad',mode=mode,experimental_tailsitter=experimental)
    with pytest.raises(ValueError):
        launch['setup'](context)
