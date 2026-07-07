#!/usr/bin/env python3
"""ArUco 정밀착륙 실기체 bringup (Jetson + CSI/USB 카메라).

한 런치로 실기체 착륙에 필요한 ROS 스택을 전부 띄웁니다:

    MAVROS(FCU 연결) + 카메라 드라이버 + ArUco pose 검출 + 정밀착륙 제어

사용 (Jetson CSI 카메라, GStreamer):
    ros2 launch precision_landing precland_hw.launch.py \
        camera_driver:=gscam fcu_url:=/dev/ttyTHS1:921600 \
        marker_size:=0.20 calib_file:=<경로>/down_camera.yaml

사용 (USB 카메라):
    ros2 launch precision_landing precland_hw.launch.py \
        camera_driver:=v4l2 video_device:=/dev/video0 \
        marker_size:=0.20 calib_file:=<경로>/down_camera.yaml

핵심 안전 기본값:
    auto_takeoff:=false   조종사가 이륙·GUIDED 전환 후 노드가 인계
    require_guided:=true  GUIDED 아니면 셋포인트 미송출(조종사가 모드로 언제든 회수)

카메라 튜닝(마운트) — 정렬이 엉뚱하게 가면:
    lat_swap:=true            마커를 못 잡고 빙글빙글(축 90° 회전)
    lat_sign_fwd:=-1.0        하강 중 전방으로 발산
    lat_sign_left:=-1.0       가로로 직선 발산
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
    camera_driver = LaunchConfiguration('camera_driver')   # gscam | v4l2 | none
    video_device = LaunchConfiguration('video_device')
    gscam_config = LaunchConfiguration('gscam_config')
    image_topic = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    calib_file = LaunchConfiguration('calib_file')
    calib_url = PythonExpression(["'file://' + '", calib_file, "'"])
    aruco_dict = LaunchConfiguration('aruco_dict')
    marker_size = LaunchConfiguration('marker_size')
    marker_id = LaunchConfiguration('marker_id')

    flight_alt = LaunchConfiguration('flight_alt')
    auto_takeoff = LaunchConfiguration('auto_takeoff')
    require_guided = LaunchConfiguration('require_guided')
    land_switch_alt = LaunchConfiguration('land_switch_alt')
    land_align_radius = LaunchConfiguration('land_align_radius')
    vel_gain = LaunchConfiguration('vel_gain')
    vel_max = LaunchConfiguration('vel_max')
    descend_rate = LaunchConfiguration('descend_rate')
    use_lidar_height = LaunchConfiguration('use_lidar_height')
    lidar_topic = LaunchConfiguration('lidar_topic')
    lidar_min = LaunchConfiguration('lidar_min')
    lidar_max = LaunchConfiguration('lidar_max')
    lidar_offset = LaunchConfiguration('lidar_offset')
    lat_swap = LaunchConfiguration('lat_swap')
    lat_sign_fwd = LaunchConfiguration('lat_sign_fwd')
    lat_sign_left = LaunchConfiguration('lat_sign_left')

    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'apm.launch')

    # MAVROS — ROS 2 ↔ ArduPilot(실기체 FCU) MAVLink 브리지.
    # 실기체는 보통 시리얼: fcu_url:=/dev/ttyTHS1:921600 (Jetson UART) 등.
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

    # ArUco pose 검출 (캘리브레이션 기반)
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

    # 정밀착륙 제어 (실기체)
    precland = Node(
        package='precision_landing',
        executable='precland_hw_node',
        name='precland_hw_node',
        output='screen',
        parameters=[{
            'flight_alt': ParameterValue(flight_alt, value_type=float),
            'auto_takeoff': ParameterValue(auto_takeoff, value_type=bool),
            'require_guided': ParameterValue(require_guided, value_type=bool),
            'land_switch_alt': ParameterValue(land_switch_alt, value_type=float),
            'land_align_radius': ParameterValue(land_align_radius, value_type=float),
            'vel_gain': ParameterValue(vel_gain, value_type=float),
            'vel_max': ParameterValue(vel_max, value_type=float),
            'descend_rate': ParameterValue(descend_rate, value_type=float),
            'use_lidar_height': ParameterValue(use_lidar_height, value_type=bool),
            'lidar_topic': lidar_topic,
            'lidar_min': ParameterValue(lidar_min, value_type=float),
            'lidar_max': ParameterValue(lidar_max, value_type=float),
            'lidar_offset': ParameterValue(lidar_offset, value_type=float),
            'lat_swap': ParameterValue(lat_swap, value_type=bool),
            'lat_sign_fwd': ParameterValue(lat_sign_fwd, value_type=float),
            'lat_sign_left': ParameterValue(lat_sign_left, value_type=float),
        }],
    )

    return LaunchDescription([
        # 실기체 FCU 연결. Jetson UART 예: /dev/ttyTHS1:921600. USB 텔레메트리면
        # /dev/ttyUSB0:57600. SITL 테스트면 udp://:14550@.
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyTHS1:921600'),
        # gscam(Jetson CSI) | v4l2(USB raw) | usb_cam(MJPG USB, mjpeg2rgb) | none
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
        # 반드시 실제 카메라로 캘리브레이션한 파일 경로를 넣으세요.
        DeclareLaunchArgument(
            'calib_file',
            default_value=os.path.join(
                get_package_share_directory('camera_detection'),
                'config', 'down_camera.yaml')),
        DeclareLaunchArgument('aruco_dict', default_value='DICT_4X4_50'),
        # 인쇄한 마커 한 변 실측(m). pose 정확도의 핵심 — 자로 재서 정확히.
        DeclareLaunchArgument('marker_size', default_value='0.25'),
        DeclareLaunchArgument('marker_id', default_value='-1'),  # -1 = 아무 마커

        DeclareLaunchArgument('flight_alt', default_value='4.0'),
        # 실기체 안전 기본값: 조종사가 이륙·GUIDED 후 인계, GUIDED 아니면 미송출.
        DeclareLaunchArgument('auto_takeoff', default_value='false'),
        DeclareLaunchArgument('require_guided', default_value='true'),
        # 이 높이(마커 위 m) 아래 + 중심 정렬되면 LAND 모드로 인계.
        DeclareLaunchArgument('land_switch_alt', default_value='1.0'),
        DeclareLaunchArgument('land_align_radius', default_value='0.15'),
        # 실기체는 보수적으로 느리게. 진동하면 vel_gain↓, 굼뜨면 ↑.
        DeclareLaunchArgument('vel_gain', default_value='0.6'),
        DeclareLaunchArgument('vel_max', default_value='0.8'),
        DeclareLaunchArgument('descend_rate', default_value='0.25'),
        # 하방 라이다(FC RNGFND → MAVROS Range)로 높이 측정. 마커 검출과 무관하게
        # 연속·정밀 → 저고도 하강·LAND 인계가 안정적. 없으면 use_lidar_height:=false.
        DeclareLaunchArgument('use_lidar_height', default_value='false'),
        DeclareLaunchArgument('lidar_topic',
                              default_value='/mavros/rangefinder/rangefinder'),
        DeclareLaunchArgument('lidar_min', default_value='0.1'),
        DeclareLaunchArgument('lidar_max', default_value='40.0'),
        DeclareLaunchArgument('lidar_offset', default_value='0.0'),
        # 카메라 마운트 매핑(첫 비행에서 지상 테스트로 확정).
        DeclareLaunchArgument('lat_swap', default_value='false'),
        DeclareLaunchArgument('lat_sign_fwd', default_value='1.0'),
        DeclareLaunchArgument('lat_sign_left', default_value='1.0'),

        mavros,
        gscam,
        v4l2,
        usb_cam,
        aruco_pose,
        precland,
    ])
