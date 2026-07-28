#!/usr/bin/env python3
"""Start the single moving-trailer Relative-OSQP landing stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    precision_share = get_package_share_directory("precision_landing")
    camera_share = get_package_share_directory("aruco_landing")
    config_default = os.path.join(
        precision_share, "config", "mpc_precision_landing.yaml"
    )
    aruco_launch = os.path.join(
        camera_share, "launch", "aruco_trailer.launch.py"
    )
    return LaunchDescription([
        DeclareLaunchArgument("baseline_run_id", default_value=""),
        DeclareLaunchArgument("vehicle_authority_id", default_value="vehicle_1"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(aruco_launch),
            launch_arguments={"enable_front": "false"}.items(),
        ),
        Node(
            package="precision_landing",
            executable="mpc_precision_landing_node",
            name="mpc_precision_landing",
            output="screen",
            parameters=[config_default, {
                "baseline_run_id": LaunchConfiguration("baseline_run_id"),
                "vehicle_authority_id": LaunchConfiguration(
                    "vehicle_authority_id"
                ),
            }],
        ),
    ])
