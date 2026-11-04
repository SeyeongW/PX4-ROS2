"""SIYI sensor station only. Never starts the MAVLink bridge or aircraft scripts."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from balloon_locator.hardware_check import SENSOR_NODES, inspect_configuration, load_parameters


def setup(context):
    config = LaunchConfiguration('config').perform(context)
    parameters = load_parameters(config)
    report = inspect_configuration(parameters)
    if report['errors']:
        raise ValueError('Invalid hardware configuration: ' + '; '.join(report['errors']))
    actions = [LogInfo(msg='SIYI sensor station only; no flight controller or aircraft commands.')]
    if report['blockers']:
        actions.append(LogInfo(msg='Coordinate readiness: ' + '; '.join(report['blockers'])))
    for name in SENSOR_NODES:
        node = Node(package='balloon_locator', executable=name, name=name,
                    parameters=[parameters[name], {'use_sim_time': False}], output='screen')
        def stopped(event, launch_context, name=name):
            if launch_context.is_shutdown:
                return []
            return [EmitEvent(event=Shutdown(reason=f'Required sensor node exited: {name} ({event.returncode})'))]
        actions.extend([RegisterEventHandler(OnProcessExit(target_action=node, on_exit=stopped)), node])
    return actions


def generate_launch_description():
    config = Path(get_package_share_directory('balloon_locator')) / 'config/balloon_locator.yaml'
    return LaunchDescription([DeclareLaunchArgument('config', default_value=str(config)),
                              OpaqueFunction(function=setup)])
