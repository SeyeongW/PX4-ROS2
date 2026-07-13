#!/usr/bin/env python3
"""Follow-target → 비전 정밀착륙 실기체 bringup (PX4 / Jetson + USB 카메라).

SITL용 follow_precland.launch.py 의 실기체 버전. Gazebo 카메라 브리지 대신
실제 USB 카메라 드라이버를 띄우고, 나머지(mavros + aruco_detector + 미션 제어)는
동일하게 올립니다.

흐름: OFFBOARD 이륙(flight_alt) → 노드가 mavros 로 AUTO.FOLLOW_TARGET 명령
(PX4 가 폰/GCS 의 FOLLOW_TARGET 을 추종) → 아루코 마커 획득 시 OFFBOARD 재진입
+ KF 비전 서보로 이동 마커 위 정밀착륙 → 강제 disarm.

전제:
  * Jetson ↔ PX4 : USB (fcu_url=/dev/ttyACM0:57600). mavros 는 이 런치가 띄움.
  * USB 다운카메라 : /dev/video0 (camera_driver 로 usb_cam/v4l2/gscam 선택).
  * 폰(QGC) : 자기 GPS 를 MAVLink FOLLOW_TARGET(#144) 로 송출 (별도 텔레메트리).

사용 (USB MJPG 카메라, 기본):
    ros2 launch precision_landing follow_precland_hw.launch.py

주의 — cam_hfov / cam_aspect 를 실제 USB 카메라 값으로 맞추세요. 비전 서보의
픽셀→미터 변환이 여기에 달려 있습니다 (SITL 기본값 1.74 는 시뮬 카메라용).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    gcs_url = LaunchConfiguration('gcs_url')
    tgt_system = LaunchConfiguration('tgt_system')
    camera_driver = LaunchConfiguration('camera_driver')   # usb_cam | v4l2 | gscam
    video_device = LaunchConfiguration('video_device')
    gscam_config = LaunchConfiguration('gscam_config')
    image_topic = LaunchConfiguration('image_topic')
    cam_hfov = LaunchConfiguration('cam_hfov')
    cam_aspect = LaunchConfiguration('cam_aspect')
    cam_offset_fwd = LaunchConfiguration('cam_offset_fwd')
    cam_offset_left = LaunchConfiguration('cam_offset_left')
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
    gimbal_port = LaunchConfiguration('gimbal_port')
    pitch_down_deg = LaunchConfiguration('pitch_down_deg')

    # PX4 mavros — 실기체는 USB 시리얼. 이 런치가 mavros 연결까지 올린다.
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

    # 카메라 드라이버 (A) MJPG USB via usb_cam (mjpeg2rgb 디코딩).
    usb_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='down_camera',
        output='screen',
        condition=LaunchConfigurationEquals('camera_driver', 'usb_cam'),
        parameters=[{
            'video_device': video_device,
            'pixel_format': 'mjpeg2rgb',
            'image_width': 1280,
            'image_height': 720,
            'framerate': 30.0,
            'frame_id': 'down_camera_optical',
        }],
        remappings=[('image_raw', image_topic)],
    )

    # 카메라 드라이버 (B) USB raw via v4l2_camera.
    v4l2 = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='down_camera',
        output='screen',
        condition=LaunchConfigurationEquals('camera_driver', 'v4l2'),
        parameters=[{
            'video_device': video_device,
            'camera_frame_id': 'down_camera_optical',
        }],
        remappings=[('image_raw', image_topic)],
    )

    # 카메라 드라이버 (C) Jetson CSI via gscam (GStreamer).
    gscam = Node(
        package='gscam',
        executable='gscam_node',
        name='down_camera',
        output='screen',
        condition=LaunchConfigurationEquals('camera_driver', 'gscam'),
        parameters=[{
            'gscam_config': gscam_config,
            'camera_name': 'down_camera',
            'frame_id': 'down_camera_optical',
        }],
        remappings=[('camera/image_raw', image_topic)],
    )

    # ArUco 검출 (정규화 오프셋). follow_precland_node 가 구독하는
    # /perception/aruco_offset · /perception/aruco_detected 를 발행한다.
    aruco_detector = Node(
        package='camera_detection',
        executable='aruco_detector_node',
        name='aruco_detector_node',
        output='screen',
        parameters=[{'image_topic': image_topic}],
    )

    # SIYI 짐벌: 시동과 동시에 하방(-90°) 고정. 짐벌 없으면 포트 열기 실패만 로그.
    gimbal_down = Node(
        package='precision_landing',
        executable='gimbal_down_node',
        name='gimbal_down_node',
        output='screen',
        parameters=[{
            'gimbal_port': ParameterValue(gimbal_port, value_type=str),
            'pitch_down_deg': ParameterValue(pitch_down_deg, value_type=float),
        }],
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
            'cam_hfov': ParameterValue(cam_hfov, value_type=float),
            'cam_aspect': ParameterValue(cam_aspect, value_type=float),
            'cam_offset_fwd': ParameterValue(cam_offset_fwd, value_type=float),
            'cam_offset_left': ParameterValue(cam_offset_left, value_type=float),
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
        # Jetson <-> PX4 USB. 포트 확인 `ls /dev/ttyACM*`, 권한 dialout.
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:57600'),
        DeclareLaunchArgument('gcs_url', default_value=''),
        DeclareLaunchArgument('tgt_system', default_value='1'),
        # 카메라: usb_cam(MJPG USB) | v4l2(USB raw) | gscam(Jetson CSI)
        DeclareLaunchArgument('camera_driver', default_value='usb_cam'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument(
            'gscam_config',
            default_value=('nvarguscamerasrc sensor-id=0 ! '
                           'video/x-raw(memory:NVMM),width=1280,height=720,'
                           'framerate=30/1 ! nvvidconv ! '
                           'video/x-raw,format=BGRx ! videoconvert ! '
                           'video/x-raw,format=BGR')),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image_raw'),
        # ★ 실제 USB 카메라 값으로 맞출 것 (비전 픽셀→미터 스케일). 예: 1280x720
        #    (16:9)=1.777, 640x480(4:3)=1.333. cam_hfov 은 카메라 데이터시트 기준.
        DeclareLaunchArgument('cam_hfov', default_value='1.20'),
        DeclareLaunchArgument('cam_aspect', default_value='1.7777'),
        # 카메라 렌즈-암: 드론 중심 기준 전방 0.15 m (좌측 0). 드론 '중심'이
        # 마커에 착륙하도록 마커 월드 좌표에 반영됨. 장착 위치 바뀌면 여기 수정.
        DeclareLaunchArgument('cam_offset_fwd', default_value='0.15'),
        DeclareLaunchArgument('cam_offset_left', default_value='0.0'),
        DeclareLaunchArgument('flight_alt', default_value='5.0'),
        DeclareLaunchArgument('platform_height', default_value='0.0'),
        DeclareLaunchArgument('vel_gain', default_value='0.4'),
        DeclareLaunchArgument('vel_max', default_value='5.0'),
        DeclareLaunchArgument('lat_swap', default_value='false'),
        DeclareLaunchArgument('lat_sign_fwd', default_value='1.0'),
        # 실기 비행 확인: 좌우가 반대였음 → 기체 좌측(+Y) 부호 뒤집음(-1).
        # (SITL 은 +1; 실기는 카메라 장착 방향이 달라 반대.)
        DeclareLaunchArgument('lat_sign_left', default_value='-1.0'),
        DeclareLaunchArgument('follow_set_height', default_value='false'),
        DeclareLaunchArgument('follow_height_param', default_value='FLW_TGT_HT'),
        DeclareLaunchArgument('follow_auto_mode', default_value='true'),
        DeclareLaunchArgument('disable_gcs_failsafe', default_value='false'),
        # SIYI 짐벌 시리얼. Jetson 40핀 헤더 UART = /dev/ttyTHS1 (USB-TTL 이면
        # /dev/ttyUSB0). 하방 각도(A8 mini 는 -90~+25°).
        # ※ ttyTHS1 은 기본적으로 시리얼 콘솔(nvgetty)이 점유 → 아래 명령으로 해제:
        #     sudo systemctl disable --now nvgetty
        #   (또는 serial-getty@ttyTHS1.service). 권한: ls -l /dev/ttyTHS1 확인.
        DeclareLaunchArgument('gimbal_port', default_value='/dev/ttyTHS1'),
        DeclareLaunchArgument('pitch_down_deg', default_value='-90.0'),
        mavros,
        usb_cam,
        v4l2,
        gscam,
        aruco_detector,
        gimbal_down,
        follow_precland,
    ])
