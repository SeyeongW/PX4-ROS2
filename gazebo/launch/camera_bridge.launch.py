#!/usr/bin/env python3
"""Bridge the iris downward camera from Gazebo (gz-sim) to ROS 2.

Starts a ros_gz_bridge that maps the gz transport topics published by the
``down_camera`` sensor to ROS 2 topics:

    /down_camera/image        sensor_msgs/msg/Image
    /down_camera/camera_info  sensor_msgs/msg/CameraInfo

Run *after* Gazebo (with the iris_with_down_camera model) is already up.

    ros2 launch <this file>

or directly:

    python3 camera_bridge.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = "/down_camera/image"
    camera_info_topic = "/down_camera/camera_info"

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="down_camera_bridge",
        output="screen",
        arguments=[
            # ROS <- GZ image stream
            f"{image_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            # ROS <- GZ camera intrinsics
            f"{camera_info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )

    return LaunchDescription([bridge])
