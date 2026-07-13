#!/usr/bin/env python3
"""Start the map-aware autonomous city mission node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("autonomy_planner"))
    default_config = str(share / "config/city_uav_1m_mission.yaml")
    default_map = str(share / "maps/city_coordinates_uav.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("map_yaml", default_value=default_map),
        Node(
            package="autonomy_planner",
            executable="city_autonomy",
            name="city_autonomy",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {"map_yaml": LaunchConfiguration("map_yaml")},
            ],
        ),
    ])
