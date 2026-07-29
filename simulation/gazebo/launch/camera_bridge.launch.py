#!/usr/bin/env python3
"""Bridge the PX4 payload cameras from Gazebo (gz-sim) to ROS 2.

Starts a ros_gz_bridge that maps the gz transport topics published by the
``down_camera`` sensor to ROS 2 topics:

    /down_camera/image        sensor_msgs/msg/Image
    /down_camera/camera_info  sensor_msgs/msg/CameraInfo

Run after a PX4 Gazebo model advertising these camera topics is already up.

    ros2 launch <this file>

or directly:

    python3 camera_bridge.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = "/down_camera/image"
    camera_info_topic = "/down_camera/camera_info"
    # Platform forward camera used by track_follower_node for line following.
    front_image_topic = "/front_camera/image"
    front_info_topic = "/front_camera/camera_info"

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="camera_bridge",
        output="screen",
        arguments=[
            # ROS <- GZ image stream (drone down camera)
            f"{image_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            # ROS <- GZ camera intrinsics
            f"{camera_info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # ROS <- GZ image stream (platform forward camera)
            f"{front_image_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{front_info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )

    return LaunchDescription([bridge])
