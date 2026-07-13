#!/usr/bin/env python3
"""Follow-target → 비전 정밀착륙 bringup (PX4).

흐름: OFFBOARD 이륙(5m) → AUTO.FOLLOW_TARGET 로 노트북 추종 → 아루코 마커
획득 시 OFFBOARD 재진입 + KF 비전 서보로 이동 마커 위 정밀착륙 → 강제 disarm.

전제:
  * PX4 SITL(또는 실기)가 떠 있고 mavros(px4.launch)로 연결됨.
  * 노트북(GCS)이 자기 GPS를 MAVLink FOLLOW_TARGET(#144) 으로 송출 중이어야
    PX4 AUTO.FOLLOW_TARGET 이 추종함 (GLOBAL_POSITION_INT 아님!).
  * Gazebo 다운카메라 + aruco_detector 로 마커 검출.

사용:
    source ~/ros2_ws/PX4-ROS2/install/setup.bash
    source ~/ros_gz_ws/install/setup.bash    # ros_gz_bridge
    ros2 launch precision_landing follow_precland.launch.py
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
    gcs_url = LaunchConfiguration('gcs_url')
    tgt_system = LaunchConfiguration('tgt_system')
    image_topic = LaunchConfiguration('image_topic')
    flight_alt = LaunchConfiguration('flight_alt')
    platform_height = LaunchConfiguration('platform_height')
    vel_gain = LaunchConfiguration('vel_gain')
    vel_max = LaunchConfiguration('vel_max')
    lat_swap = LaunchConfiguration('lat_swap')
    lat_sign_fwd = LaunchConfiguration('lat_sign_fwd')
    lat_sign_left = LaunchConfiguration('lat_sign_left')
    follow_set_height = LaunchConfiguration('follow_set_height')
    follow_height_param = LaunchConfiguration('follow_height_param')
    follow_auto_mode = LaunchConfiguration('follow_auto_mode')
    disable_gcs_failsafe = LaunchConfiguration('disable_gcs_failsafe')

    # PX4 mavros — 이 런치가 mavros 연결까지 직접 올린다(px4_config.yaml +
    # px4_pluginlists.yaml 자동 로드). fcu_url = mavros<->PX4 링크,
    # gcs_url 비우면 됨(노트북 GCS는 PX4 의 14550 GCS 포트로 따로 붙어
    # FOLLOW_TARGET 를 송출; mavros 가 중계할 필요 없음).
    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'px4.launch')
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_launch),
        launch_arguments={
            'fcu_url': fcu_url,
            'gcs_url': gcs_url,
            'tgt_system': tgt_system,
        }.items(),
    )

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

    aruco_detector = Node(
        package='camera_detection',
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen',
        parameters=[{'image_topic': image_topic}],
    )

    follow_precland = Node(
        package='precision_landing',
        executable='follow_precland_node',
        name='follow_precland_node',
        output='screen',
        parameters=[{
            'flight_alt': ParameterValue(flight_alt, value_type=float),
            'platform_height': ParameterValue(platform_height, value_type=float),
            'vel_gain': ParameterValue(vel_gain, value_type=float),
            'vel_max': ParameterValue(vel_max, value_type=float),
            'lat_swap': ParameterValue(lat_swap, value_type=bool),
            'lat_sign_fwd': ParameterValue(lat_sign_fwd, value_type=float),
            'lat_sign_left': ParameterValue(lat_sign_left, value_type=float),
            'follow_set_height': ParameterValue(follow_set_height, value_type=bool),
            'follow_height_param': ParameterValue(follow_height_param, value_type=str),
            'follow_auto_mode': ParameterValue(follow_auto_mode, value_type=bool),
            'disable_gcs_failsafe': ParameterValue(disable_gcs_failsafe, value_type=bool),
        }],
    )

    return LaunchDescription([
        # Jetson <-> PX4 는 USB(Pixhawk USB-CDC). USB 는 보드레이트 무시하지만
        # 형식상 명시. 포트 확인: `ls /dev/ttyACM*` (보통 ttyACM0). 권한: dialout.
        # 노트북 GCS 는 별도 텔레메트리 링크로 PX4 의 TELEM 포트에 직접 붙어
        # FOLLOW_TARGET 을 송출하므로 이 USB 링크와 경합 없음(gcs_url 브릿지 불요).
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:57600'),
        # 빈값. 노트북 GCS 는 PX4 의 TELEM 포트(텔레메트리)에 직접 붙으므로 mavros
        # GCS 브릿지가 필요 없다. (mavros 를 통해 GCS 를 중계하고 싶을 때만 설정.)
        DeclareLaunchArgument('gcs_url', default_value=''),
        DeclareLaunchArgument('tgt_system', default_value='1'),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image'),
        DeclareLaunchArgument('flight_alt', default_value='5.0'),
        # 착륙 대상이 높이 있는 플랫폼이면 그 높이, 평면 지면 마커면 0.
        DeclareLaunchArgument('platform_height', default_value='0.0'),
        # 정밀 서보 게인/상한 (이동 마커 추종: 위치오차+속도FF 보정).
        DeclareLaunchArgument('vel_gain', default_value='0.4'),
        DeclareLaunchArgument('vel_max', default_value='5.0'),
        # 이미지->기체 매핑 (드론이 마커 반대로 가면 부호/축 뒤집기).
        DeclareLaunchArgument('lat_swap', default_value='false'),
        DeclareLaunchArgument('lat_sign_fwd', default_value='1.0'),
        DeclareLaunchArgument('lat_sign_left', default_value='1.0'),
        # AUTO.FOLLOW_TARGET 추종 고도 파라미터 세팅 (펌웨어 버전별 이름 상이:
        # v1.13+ FLW_TGT_HT, 구버전 NAV_FT_HT). 기본 off = PX4 자체 설정 사용.
        DeclareLaunchArgument('follow_set_height', default_value='false'),
        DeclareLaunchArgument('follow_height_param', default_value='FLW_TGT_HT'),
        # true = 노드가 mavros 로 AUTO.FOLLOW_TARGET 모드 전환을 명령(노드가 follow
        # 를 몬다). false = 폰 QGC Follow Me 버튼 등 외부 GCS 가 모드를 넣을 때
        # (노드는 이륙 후 hover 대기하다 모드 바뀌면 마커 감시로 넘어감).
        DeclareLaunchArgument('follow_auto_mode', default_value='true'),
        # 실 노트북 GCS 없이 bare SITL 로 arm 테스트할 때만 true (NAV_DLL_ACT=0).
        DeclareLaunchArgument('disable_gcs_failsafe', default_value='false'),
        mavros,
        camera_bridge,
        aruco_detector,
        follow_precland,
    ])
