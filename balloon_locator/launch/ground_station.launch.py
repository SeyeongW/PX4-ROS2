import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory('balloon_locator'), 'config', 'balloon_locator.yaml')
    cfg = LaunchConfiguration('config')
    names = ['siyi_gimbal_node', 'siyi_camera_node', 'balloon_detector_node',
             'gimbal_tracker_node', 'balloon_locator_node']
    return LaunchDescription([DeclareLaunchArgument('config', default_value=default_cfg)] + [
        Node(package='balloon_locator', executable=n, name=n, parameters=[cfg], output='screen') for n in names
    ])
