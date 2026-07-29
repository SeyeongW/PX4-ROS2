import importlib.util
from pathlib import Path
from typing import List

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.utilities import evaluate_parameters


def _marker_sizes_value(text):
    context = LaunchContext()
    context.launch_configurations['marker_sizes_m'] = text
    return ParameterValue(
        LaunchConfiguration('marker_sizes_m'),
        value_type=List[float],
    ).evaluate(context)


def test_marker_sizes_launch_argument_is_a_double_array():
    assert _marker_sizes_value('[1.3, 1.3, 0.30]') == [1.3, 1.3, 0.3]


@pytest.mark.parametrize(
    'invalid',
    ['1.3,1.3,0.30', '[1, 1.3, 0.30]', '[1.3, bad, 0.30]'],
)
def test_invalid_marker_sizes_launch_argument_fails_fast(invalid):
    with pytest.raises(ValueError):
        _marker_sizes_value(invalid)


def test_gimbal_launch_wires_marker_sizes_and_deck_z_consistently():
    launch_path = (
        Path(__file__).parents[1] / 'launch' / 'gimbal_perception.launch.py')
    spec = importlib.util.spec_from_file_location(
        'landing_mpc_gimbal_launch_test', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    context = LaunchContext()
    context.launch_configurations.update({
        'marker_sizes_m': '[1.3, 1.3, 0.30]',
        'max_pair_disagreement_m': '0.75',
        'marker_size_m': '9.9',
        'deck_z': '2.25',
        'model_name': 'test_vehicle',
        'world': 'test_world',
        'use_deck_z': 'true',
    })
    nodes = {
        action._Node__node_executable: action
        for action in module.generate_launch_description().entities
        if isinstance(action, Node)
    }

    def parameters(executable):
        result = {}
        for item in evaluate_parameters(
                context, nodes[executable]._Node__parameters):
            result.update(item)
        return result

    detector = parameters('aruco_detector_node')
    assert 'marker_size_m' not in detector
    assert detector['marker_sizes_m'] == [1.3, 1.3, 0.3]
    assert detector['max_pair_disagreement_m'] == pytest.approx(0.75)
    declared_arguments = {
        action.name
        for action in module.generate_launch_description().entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert 'marker_size_m' in declared_arguments
    assert 'max_pair_disagreement_m' in declared_arguments

    for executable in (
            'gimbal_control_node', 'marker_tf_node', 'marker_kf_node'):
        assert parameters(executable)['deck_z'] == pytest.approx(2.25)
