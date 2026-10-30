"""별도 자기 위치를 받아 관찰 경로만 제안한다. 기본 추적 실험에는 자동 추가하지 않는다."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    scene = Path(get_package_share_directory('jol_perception'))/'config/observation_scene.json'
    return LaunchDescription([
        DeclareLaunchArgument('scene_file', default_value=str(scene)),
        DeclareLaunchArgument('standoff_m', default_value='10.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='jol_perception', executable='observation', output='screen', parameters=[{
            'scene_file': LaunchConfiguration('scene_file'),
            'standoff_m': ParameterValue(LaunchConfiguration('standoff_m'), value_type=float),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}])])
