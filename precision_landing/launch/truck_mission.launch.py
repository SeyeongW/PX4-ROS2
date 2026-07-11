#!/usr/bin/env python3
"""트럭 발진 풀 미션 bringup — 장애물 회피는 APF 베이스라인.

Gazebo(run_sim.sh, 트럭 마운트 월드)와 ArduPilot SITL(sim_vehicle.py)을 따로 켜 두면,
이 런치가 ROS 쪽 전부를 띄웁니다:

    MAVROS  +  카메라 브리지  +  ArUco 검출  +  정밀착륙 제어(대기 모드)  +  미션 매니저

공통 브링업(위 4개 중 미션 매니저를 뺀 나머지)은 common_bringup.launch.py에 있음 —
A*+정적회랑+MPC 버전(truck_mission_mpc.launch.py)과 여기가 똑같이 그걸 불러 쓰므로,
정밀착륙 파라미터를 튜닝할 땐 그 파일 하나만 고치면 됨. 이 파일이 직접 띄우는 건
mission_manager_node뿐 — apf_enable/mpc_enable로 장애물 회피 방식이 갈리는 노드라
두 최상위 런치파일이 각자 다르게 띄운다(로드맵의 APF vs MPC 벤치마크 비교용).

precision_landing_node 는 require_enable:=true 로 **DORMANT** 상태로 시작해 셋포인트를
쏘지 않고, mission_manager 가 복귀-착륙 시점에 /mission/land_enable 을 래치하면 그때
권한을 넘겨받아 이동 트럭 위에 착륙합니다(단일 setpoint 권한). 기존 정밀착륙 단독 데모는
precision_landing.launch.py 가 그대로 유지합니다.

사용:
    source ~/ros2_ws/PX4-ROS2/install/setup.bash
    source ~/ros_gz_ws/install/setup.bash
    ros2 launch precision_landing truck_mission.launch.py

흐름 관찰:
    ros2 topic echo /mission/phase
    ros2 topic echo /mission/battery_s
링크 손실(수동 트리거):
    ros2 service call /truck/drop_link std_srvs/srv/Trigger
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    flight_alt = LaunchConfiguration('flight_alt')
    # 미션 파라미터
    mission_area_e = LaunchConfiguration('mission_area_e')
    mission_area_n = LaunchConfiguration('mission_area_n')
    trigger_dist = LaunchConfiguration('trigger_dist')
    loiter_s = LaunchConfiguration('loiter_s')
    patrol_route = LaunchConfiguration('patrol_route')
    battery_capacity_s = LaunchConfiguration('battery_capacity_s')
    reserve_margin_s = LaunchConfiguration('reserve_margin_s')
    effective_speed = LaunchConfiguration('effective_speed')
    # 장애물 회피(APF 베이스라인). obstacle_field 월드에서만 의미 있음 — 기본 꺼짐.
    apf_enable = LaunchConfiguration('apf_enable')
    obstacle_map = LaunchConfiguration('obstacle_map')
    apf_influence_radius = LaunchConfiguration('apf_influence_radius')
    apf_gain = LaunchConfiguration('apf_gain')
    apf_vel_cap = LaunchConfiguration('apf_vel_cap')

    common_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('precision_landing'),
            'launch', 'common_bringup.launch.py')),
        launch_arguments={
            'fcu_url': LaunchConfiguration('fcu_url'),
            'image_topic': LaunchConfiguration('image_topic'),
            'flight_alt': flight_alt,
            'platform_height': LaunchConfiguration('platform_height'),
            'land_clearance': LaunchConfiguration('land_clearance'),
            'apf_enable': apf_enable,
            'obstacle_map': obstacle_map,
            'apf_influence_radius': apf_influence_radius,
            'apf_gain': apf_gain,
            'apf_vel_cap': apf_vel_cap,
        }.items(),
    )

    # 미션 매니저 — 고수준 FSM(MOUNTED→TAKEOFF→MISSION→LANDING + LINK_LOST→SAFE_LAND).
    # MISSION 구간 장애물 회피는 APF(apf_enable) — mpc_enable은 기본 false로 안 건드림.
    mission_manager = Node(
        package='precision_landing',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[{
            'flight_alt': flight_alt,
            'mission_area_e': ParameterValue(mission_area_e, value_type=float),
            'mission_area_n': ParameterValue(mission_area_n, value_type=float),
            'trigger_dist': ParameterValue(trigger_dist, value_type=float),
            'loiter_s': ParameterValue(loiter_s, value_type=float),
            'patrol_route': patrol_route,
            'battery_capacity_s': ParameterValue(battery_capacity_s, value_type=float),
            'reserve_margin_s': ParameterValue(reserve_margin_s, value_type=float),
            'effective_speed': ParameterValue(effective_speed, value_type=float),
            'apf_enable': ParameterValue(apf_enable, value_type=bool),
            'obstacle_map': obstacle_map,
            'apf_influence_radius': ParameterValue(apf_influence_radius, value_type=float),
            'apf_gain': ParameterValue(apf_gain, value_type=float),
            'apf_vel_cap': ParameterValue(apf_vel_cap, value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://:14540@'),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image'),
        DeclareLaunchArgument('flight_alt', default_value='5.0'),
        # 임무지역: 트럭 경로(E축, N=0)에서 벗어난 지점으로 잡아 드론이 트럭을 떠났다
        # 돌아오게. 트럭이 여기서 trigger_dist 이내로 접근하면 발진.
        DeclareLaunchArgument('mission_area_e', default_value='120.0'),
        DeclareLaunchArgument('mission_area_n', default_value='40.0'),
        DeclareLaunchArgument('trigger_dist', default_value='50.0'),
        DeclareLaunchArgument('loiter_s', default_value='15.0'),
        # 정찰 웨이포인트 리스트(비우면 mission_area 단일점, 기존 동작 그대로).
        # obstacle_field 월드에서 검증하려면 gazebo/config/patrol_route.yaml 지정.
        DeclareLaunchArgument('patrol_route', default_value=''),
        # 배터리 예산. 짧게(예: 40) 주면 임무 도중 배터리 조건으로 복귀가 트리거됨.
        DeclareLaunchArgument('battery_capacity_s', default_value='120.0'),
        DeclareLaunchArgument('reserve_margin_s', default_value='10.0'),
        DeclareLaunchArgument('effective_speed', default_value='8.0'),
        # 플랫폼(트럭) 위 착륙: 월드 aruco_platform 높이(1.0 m)와 맞춤.
        DeclareLaunchArgument('platform_height', default_value='1.0'),
        DeclareLaunchArgument('land_clearance', default_value='0.2'),
        # 장애물 회피(APF). obstacle_field 월드 + gazebo/config/obstacle_map.yaml 용.
        DeclareLaunchArgument('apf_enable', default_value='false'),
        DeclareLaunchArgument('obstacle_map', default_value=''),
        DeclareLaunchArgument('apf_influence_radius', default_value='15.0'),
        DeclareLaunchArgument('apf_gain', default_value='6.0'),
        DeclareLaunchArgument('apf_vel_cap', default_value='6.0'),
        common_bringup,
        mission_manager,
    ])
