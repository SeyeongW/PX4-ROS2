#!/usr/bin/env python3
"""Open a live window of the platform's forward camera WITH the line-processing
overlay (track_follower_node's debug image).

The overlay shows how the lane is perceived and how the platform drives:
detected line pixels, both line edges, the per-band lane centres, the
least-squares fit curve, the near / look-ahead targets, a steering arrow, a
binary-mask inset, and a text HUD (cross-track / look-ahead error, yaw rate,
speed, fit degree, TRACKING/LOST).

Prerequisites: Gazebo running (run_sim.sh), the camera bridge up
(camera_bridge.launch.py — feeds /front_camera/image to track_follower), and
track_follower_node running (run_sim.sh starts it when FOLLOW_TRACK=1). Then:

    ros2 launch ./launch/track_view.launch.py          # processed overlay
    ros2 launch ./launch/track_view.launch.py topic:=/front_camera/image   # raw cam

rqt_image_view defaults to the RAW transport of /perception/track_debug, which
track_follower publishes directly, so the window shows the overlay immediately.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    topic = LaunchConfiguration("topic")
    return LaunchDescription([
        DeclareLaunchArgument(
            "topic", default_value="/perception/track_debug",
            description="Image topic to view (default: line-processing overlay)."),
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="track_view",
            output="screen",
            arguments=[topic],
        ),
    ])
