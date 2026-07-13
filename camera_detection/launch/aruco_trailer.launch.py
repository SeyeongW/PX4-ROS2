#!/usr/bin/env python3
"""Run independent front-acquisition and down-terminal ArUco estimators."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("camera_detection")
    front_default = os.path.join(share, "config", "aruco_front.yaml")
    down_default = os.path.join(share, "config", "aruco_down.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("front_config", default_value=front_default),
        DeclareLaunchArgument("down_config", default_value=down_default),
        Node(
            package="camera_detection",
            executable="aruco_pose_node",
            name="front_aruco_pose",
            output="screen",
            parameters=[LaunchConfiguration("front_config")],
        ),
        Node(
            package="camera_detection",
            executable="aruco_pose_node",
            name="down_aruco_pose",
            output="screen",
            parameters=[LaunchConfiguration("down_config")],
        ),
    ])
