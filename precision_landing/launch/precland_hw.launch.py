#!/usr/bin/env python3
"""ArUco 정밀착륙 실기체 bringup (Jetson + CSI/USB 카메라).

한 런치로 실기체 착륙에 필요한 ROS 스택을 전부 띄웁니다:

    MAVROS(FCU 연결) + 카메라 드라이버 + ArUco pose 검출 + 정밀착륙 제어

이 런치는 **무엇을 띄울지 + 어느 하드웨어에 붙일지**(배포 설정)만 담습니다.
정밀착륙 동작·게인·고도·라이다·카메라마운트 같은 **튜닝 파라미터는 노드 코드**
(precland_hw_node.py 상단 declare_parameter)에서 직접 조정하세요. 런치에 중복해
두지 않아, 값을 바꾸는 곳이 한 군데(노드 파일)뿐입니다.

사용 (USB MJPG 카메라, 기본):
    ros2 launch precision_landing precland_hw.launch.py \
        fcu_url:=/dev/ttyACM0:115200

사용 (Jetson CSI 카메라, GStreamer):
    ros2 launch precision_landing precland_hw.launch.py \
        camera_driver:=gscam fcu_url:=/dev/ttyTHS1:921600

배포 인자(런치에 남긴 것):
    fcu_url        FCU 연결 문자열(USB=/dev/ttyACM0:115200, UART=/dev/ttyTHS1:921600)
    camera_driver  usb_cam(MJPG) | v4l2(USB raw) | gscam(CSI) | none
    video_device   USB 카메라 장치 노드
    calib_file     카메라 캘리브레이션(camera_info yaml) 경로
    marker_size    인쇄한 마커 한 변 실측(m) — pose 정확도의 핵심
    aruco_dict / marker_id  마커 사전 / 특정 ID(-1=아무거나)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    camera_driver = LaunchConfiguration('camera_driver')   # usb_cam | v4l2 | gscam | none
    video_device = LaunchConfiguration('video_device')
    gscam_config = LaunchConfiguration('gscam_config')
    image_topic = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    calib_file = LaunchConfiguration('calib_file')
    calib_url = PythonExpression(["'file://' + '", calib_file, "'"])
    aruco_dict = LaunchConfiguration('aruco_dict')
    marker_size = LaunchConfiguration('marker_size')
    marker_id = LaunchConfiguration('marker_id')

    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'apm.launch')

    # MAVROS — ROS 2 ↔ ArduPilot(실기체 FCU) MAVLink 브리지.
    # 실기체는 보통 시리얼: fcu_url:=/dev/ttyACM0:115200 (USB) 등.
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_launch),
        launch_arguments={'fcu_url': fcu_url}.items(),
    )

    # 카메라 드라이버 (A) Jetson CSI/USB via GStreamer(gscam2).
    #   nvarguscamerasrc = Jetson CSI 카메라. USB 면 v4l2src 파이프라인으로 교체.
    gscam = Node(
        package='gscam',
        executable='gscam_node',
        name='down_camera',
        output='screen',
        condition=LaunchConfigurationEquals('camera_driver', 'gscam'),
        parameters=[{
            'gscam_config': gscam_config,
            'camera_name': 'down_camera',
            'camera_info_url': calib_url,
            'frame_id': 'down_camera_optical',
        }],
        remappings=[
            ('camera/image_raw', image_topic),
            ('camera/camera_info', camera_info_topic),
        ],
    )

    # 카메라 드라이버 (B) USB 카메라 via v4l2_camera.
    v4l2 = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='down_camera',
        output='screen',
        condition=LaunchConfigurationEquals('camera_driver', 'v4l2'),
        parameters=[{
            'video_device': video_device,
            'camera_info_url': calib_url,
            'camera_frame_id': 'down_camera_optical',
        }],
        remappings=[
            ('image_raw', image_topic),
            ('camera_info', camera_info_topic),
        ],
    )

    # 카메라 드라이버 (C) MJPG USB 카메라 via usb_cam (mjpeg2rgb 디코딩).
    #   v4l2_camera 가 MJPG 디코딩을 못 하는 카메라(예: icSpring 720p=MJPG 전용)용.
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
            'camera_info_url': calib_url,
            'frame_id': 'down_camera_optical',
        }],
        remappings=[
            ('image_raw', image_topic),
            ('camera_info', camera_info_topic),
        ],
    )

    # ArUco pose 검출 (캘리브레이션 기반). 마커/카메라 배포 설정만 넘김.
    aruco_pose = Node(
        package='camera_detection',
        executable='aruco_pose_node',
        name='aruco_pose_node',
        output='screen',
        parameters=[{
            'image_topic': image_topic,
            'camera_info_topic': camera_info_topic,
            'aruco_dict': aruco_dict,
            'marker_size': ParameterValue(marker_size, value_type=float),
            'marker_id': ParameterValue(marker_id, value_type=int),
            'calib_file': calib_file,
        }],
    )

    # 정밀착륙 제어 (실기체). 동작·게인·고도·라이다·마운트 파라미터는 모두
    # precland_hw_node.py 의 declare_parameter 기본값으로 관리 — 여기서 안 넘김.
    # (튜닝은 노드 파일 한 곳에서. .py 는 심볼릭 링크라 저장하면 바로 반영.)
    precland = Node(
        package='precision_landing',
        executable='precland_hw_node',
        name='precland_hw_node',
        output='screen',
    )

    return LaunchDescription([
        # --- 배포/하드웨어 설정 -------------------------------------------
        # 실기체 FCU 연결. USB=/dev/ttyACM0:115200, Jetson UART=/dev/ttyTHS1:921600,
        # SITL 테스트=udp://:14550@.
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:115200'),
        # usb_cam(MJPG USB) | v4l2(USB raw) | gscam(Jetson CSI) | none
        DeclareLaunchArgument('camera_driver', default_value='usb_cam'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        # Jetson CSI 파이프라인(예: IMX219). 카메라/해상도에 맞게 조정.
        DeclareLaunchArgument(
            'gscam_config',
            default_value=('nvarguscamerasrc sensor-id=0 ! '
                           'video/x-raw(memory:NVMM),width=1280,height=720,'
                           'framerate=30/1 ! nvvidconv ! '
                           'video/x-raw,format=BGRx ! videoconvert ! '
                           'video/x-raw,format=BGR')),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/down_camera/camera_info'),
        # 캘리브레이션 파일(ROS camera_info yaml) — camera_calibration 으로 생성.
        DeclareLaunchArgument(
            'calib_file',
            default_value=os.path.join(
                get_package_share_directory('camera_detection'),
                'config', 'down_camera.yaml')),
        # 마커 배포 설정. marker_size 는 인쇄한 마커 한 변 실측(m) — 자로 정확히.
        DeclareLaunchArgument('aruco_dict', default_value='DICT_4X4_50'),
        DeclareLaunchArgument('marker_size', default_value='0.25'),
        DeclareLaunchArgument('marker_id', default_value='-1'),  # -1 = 아무 마커

        mavros,
        gscam,
        v4l2,
        usb_cam,
        aruco_pose,
        precland,
    ])
