#!/usr/bin/env python3
"""Launch the exact mission-contract moving landing trailer driver."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("autonomy_planner"),
        "config",
        "trailer_motion.yaml",
    )
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        Node(
            package="autonomy_planner",
            executable="trailer_motion",
            name="trailer_motion",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
    ])
