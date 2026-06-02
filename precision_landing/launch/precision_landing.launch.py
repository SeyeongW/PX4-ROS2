#!/usr/bin/env python3
"""ArUco 정밀착륙 풀 bringup (Python 제어기).

Gazebo(run_sim.sh)와 ArduPilot SITL(sim_vehicle.py)만 따로 켜 두면,
이 런치 하나가 나머지 ROS 쪽을 전부 띄우고 SITL에 연결합니다:

    MAVROS (FCU 연결)  +  카메라 브리지  +  ArUco 검출  +  정밀착륙 제어(Python)

사용:
    source ~/ros2_ws/PX4-ROS2/install/setup.bash
    source ~/ros_gz_ws/install/setup.bash    # ros_gz_bridge 경로
    ros2 launch precision_landing precision_landing.launch.py

옵션:
    ros2 launch precision_landing precision_landing.launch.py \
        fcu_url:=udp://:14550@ \
        image_topic:=/down_camera/image \
        flight_alt:=5.0 auto_takeoff:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    image_topic = LaunchConfiguration('image_topic')
    flight_alt = LaunchConfiguration('flight_alt')
    auto_takeoff = LaunchConfiguration('auto_takeoff')
    # 횡방향(이미지->기체) 매핑 튜닝. 드론이 마커를 못 잡고 빙글빙글 돌면
    # 축이 90° 돌아간 것 -> lat_swap 으로 해결. 한쪽으로 직선 발산하면 부호를 뒤집는다.
    lat_swap = LaunchConfiguration('lat_swap')
    lat_sign_fwd = LaunchConfiguration('lat_sign_fwd')
    lat_sign_left = LaunchConfiguration('lat_sign_left')
    # 수평 속도 서보 게인/상한. 진동(overshoot)하면 낮추고, 너무 굼뜨면 올린다.
    vel_gain = LaunchConfiguration('vel_gain')
    vel_max = LaunchConfiguration('vel_max')

    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'apm.launch')

    # MAVROS — ROS 2 <-> ArduPilot(SITL) MAVLink 브리지
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_launch),
        launch_arguments={'fcu_url': fcu_url}.items(),
    )

    # Gazebo 카메라 -> ROS 2 토픽 브리지 (ros_gz_bridge 필요: 위 source)
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='down_camera_bridge',
        output='screen',
        arguments=[
            '/down_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/down_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
    )

    # ArUco 마커 검출 (perception)
    aruco_detector = Node(
        package='camera_detection',
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen',
        parameters=[{'image_topic': image_topic}],
    )

    # 정밀착륙 제어 (Python) — auto_takeoff=true 면 스스로 GUIDED/시동/이륙
    precision_landing = Node(
        package='precision_landing',
        executable='precision_landing_node',
        name='precision_landing_node',
        output='screen',
        parameters=[{
            'flight_alt': flight_alt,
            'auto_takeoff': auto_takeoff,
            'lat_swap': ParameterValue(lat_swap, value_type=bool),
            'lat_sign_fwd': ParameterValue(lat_sign_fwd, value_type=float),
            'lat_sign_left': ParameterValue(lat_sign_left, value_type=float),
            'vel_gain': ParameterValue(vel_gain, value_type=float),
            'vel_max': ParameterValue(vel_max, value_type=float),
        }],
    )

    return LaunchDescription([
        # MAVProxy(sim_vehicle.py)는 127.0.0.1:14550 으로만 송신하므로 14550 에 bind.
        # remote(@뒤)를 비우면 들어온 패킷에서 상대 주소를 자동 학습 → 양방향 통신.
        DeclareLaunchArgument('fcu_url', default_value='udp://:14550@'),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image'),
        DeclareLaunchArgument('flight_alt', default_value='5.0'),
        DeclareLaunchArgument('auto_takeoff', default_value='true'),
        # 비행 로그로 확정한 매핑: 가로축(left) +1, 전방축(fwd) +1, swap 없음.
        #   sign_left=-1 -> off.x 발산(가로 부호 반대), +1 이 맞음.
        #   sign_fwd=-1  -> 하강 중 off.y 발산(전방 부호 반대), +1 이 맞음.
        #   ("빙글빙글"의 원인은 회전 오류가 아니라 높은 게인의 X축 진동이었음.)
        DeclareLaunchArgument('lat_swap', default_value='false'),
        DeclareLaunchArgument('lat_sign_fwd', default_value='1.0'),
        DeclareLaunchArgument('lat_sign_left', default_value='1.0'),
        # 진동 억제: 게인/속도를 1.0 -> 0.4/0.5 로 낮춰 한계 진동을 줄인다.
        DeclareLaunchArgument('vel_gain', default_value='0.4'),
        DeclareLaunchArgument('vel_max', default_value='0.5'),
        mavros,
        camera_bridge,
        aruco_detector,
        precision_landing,
    ])
