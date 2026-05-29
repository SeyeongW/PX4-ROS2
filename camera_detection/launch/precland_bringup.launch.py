#!/usr/bin/env python3
"""ArUco 정밀착륙 '서버' 일괄 실행.

Gazebo(run_sim.sh)와 ArduPilot SITL(sim_vehicle.py)만 따로 켜 두면,
이 런치 하나가 나머지 ROS 쪽을 전부 띄우고 SITL에 연결합니다:

    MAVROS (FCU 연결)  +  카메라 브리지  +  ArUco 검출  +  정밀착륙 제어

사용:
    source ~/ros_gz_ws/install/setup.bash    # ros_gz_bridge 경로
    ros2 launch camera_detection precland_bringup.launch.py

옵션:
    ros2 launch camera_detection precland_bringup.launch.py \
        fcu_url:=udp://127.0.0.1:14550@14555 \
        image_topic:=/down_camera/image
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fcu_url = LaunchConfiguration("fcu_url")
    image_topic = LaunchConfiguration("image_topic")

    mavros_launch = os.path.join(
        get_package_share_directory("mavros"), "launch", "apm.launch")

    # MAVROS — ROS 2 ↔ ArduPilot(SITL) MAVLink 브리지
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_launch),
        launch_arguments={"fcu_url": fcu_url}.items(),
    )

    # Gazebo 카메라 → ROS 2 토픽 브리지 (ros_gz_bridge 필요: 위 source)
    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="down_camera_bridge",
        output="screen",
        arguments=[
            "/down_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/down_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
    )

    # ArUco 마커 검출
    aruco_detector = Node(
        package="camera_detection",
        executable="aruco_detector_node",
        name="aruco_detector_node",
        output="screen",
        parameters=[{"image_topic": image_topic}],
    )

    # 정밀착륙 제어 (armed + GUIDED + 마커 감지 시 자동 인계)
    precision_landing = Node(
        package="offboard",
        executable="precision_landing",
        name="precision_landing_node",
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("fcu_url", default_value="udp://127.0.0.1:14550@14555"),
        DeclareLaunchArgument("image_topic", default_value="/down_camera/image"),
        mavros,
        camera_bridge,
        aruco_detector,
        precision_landing,
    ])
