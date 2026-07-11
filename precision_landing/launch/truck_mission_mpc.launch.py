#!/usr/bin/env python3
"""트럭 발진 풀 미션 bringup — 장애물 회피는 A*(front-end) + 정적회랑(back-end) +
MPC(QP-ish, roadmap step 2).

truck_mission.launch.py(APF 베이스라인)와 노드 구성은 동일하고(공통 브링업은
common_bringup.launch.py 공유), MISSION 구간 장애물 회피 알고리즘만 다름 — 벤치마크
비교용으로 두 파일을 나란히 씀. `mission_manager_node`의 `mpc_enable:=true`가 이
차이를 만드는 전부: MISSION 단계 서보 호출이 `_servo_to`(APF) 대신
`_mpc_servo_to`(A*+회랑+MPC)로 바뀐다. 자세한 알고리즘/안전장치는
precision_landing/precision_landing/{planner,mpc}.py, mission_manager_node.py의
_mpc_servo_to/_mpc_maybe_replan 참고.

obstacle_map은 필수(없으면 회랑 없이 무제약으로 돎). apf_enable은 이 파일에서도
노출은 하지만(precision_landing_node의 RETURN/착륙 구간용, common_bringup 공유
파라미터) 기본은 꺼짐 — MISSION 구간과 별개 스위치임에 주의.

사용:
    source ~/ros2_ws/PX4-ROS2/install/setup.bash
    source ~/ros_gz_ws/install/setup.bash
    ros2 launch precision_landing truck_mission_mpc.launch.py \
        mpc_enable:=true obstacle_map:=$(pwd)/gazebo/config/obstacle_map.yaml \
        mission_area_e:=20.0 mission_area_n:=16.0

흐름 관찰:
    ros2 topic echo /mission/phase
    ros2 topic echo /mission/battery_s
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
    mission_area_e = LaunchConfiguration('mission_area_e')
    mission_area_n = LaunchConfiguration('mission_area_n')
    trigger_dist = LaunchConfiguration('trigger_dist')
    loiter_s = LaunchConfiguration('loiter_s')
    patrol_route = LaunchConfiguration('patrol_route')
    battery_capacity_s = LaunchConfiguration('battery_capacity_s')
    reserve_margin_s = LaunchConfiguration('reserve_margin_s')
    effective_speed = LaunchConfiguration('effective_speed')
    apf_enable = LaunchConfiguration('apf_enable')
    obstacle_map = LaunchConfiguration('obstacle_map')
    apf_influence_radius = LaunchConfiguration('apf_influence_radius')
    apf_gain = LaunchConfiguration('apf_gain')
    apf_vel_cap = LaunchConfiguration('apf_vel_cap')
    # A*+정적회랑+MPC (MISSION 구간, mission_manager_node 전용)
    mpc_enable = LaunchConfiguration('mpc_enable')
    mpc_horizon = LaunchConfiguration('mpc_horizon')
    mpc_dt = LaunchConfiguration('mpc_dt')
    mpc_vmax = LaunchConfiguration('mpc_vmax')
    mpc_replan_period_s = LaunchConfiguration('mpc_replan_period_s')
    mpc_solve_period_s = LaunchConfiguration('mpc_solve_period_s')
    mpc_corridor_margin = LaunchConfiguration('mpc_corridor_margin')
    mpc_q_track = LaunchConfiguration('mpc_q_track')
    mpc_r_smooth = LaunchConfiguration('mpc_r_smooth')
    mpc_r_effort = LaunchConfiguration('mpc_r_effort')
    mpc_corridor_weight = LaunchConfiguration('mpc_corridor_weight')
    mpc_iters = LaunchConfiguration('mpc_iters')

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
            'mpc_enable': ParameterValue(mpc_enable, value_type=bool),
            'mpc_horizon': ParameterValue(mpc_horizon, value_type=int),
            'mpc_dt': ParameterValue(mpc_dt, value_type=float),
            'mpc_vmax': ParameterValue(mpc_vmax, value_type=float),
            'mpc_replan_period_s': ParameterValue(mpc_replan_period_s, value_type=float),
            'mpc_solve_period_s': ParameterValue(mpc_solve_period_s, value_type=float),
            'mpc_corridor_margin': ParameterValue(mpc_corridor_margin, value_type=float),
            'mpc_q_track': ParameterValue(mpc_q_track, value_type=float),
            'mpc_r_smooth': ParameterValue(mpc_r_smooth, value_type=float),
            'mpc_r_effort': ParameterValue(mpc_r_effort, value_type=float),
            'mpc_corridor_weight': ParameterValue(mpc_corridor_weight, value_type=float),
            'mpc_iters': ParameterValue(mpc_iters, value_type=int),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://:14550@'),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image'),
        DeclareLaunchArgument('flight_alt', default_value='5.0'),
        DeclareLaunchArgument('mission_area_e', default_value='120.0'),
        DeclareLaunchArgument('mission_area_n', default_value='40.0'),
        DeclareLaunchArgument('trigger_dist', default_value='50.0'),
        DeclareLaunchArgument('loiter_s', default_value='15.0'),
        DeclareLaunchArgument('patrol_route', default_value=''),
        DeclareLaunchArgument('battery_capacity_s', default_value='120.0'),
        DeclareLaunchArgument('reserve_margin_s', default_value='10.0'),
        DeclareLaunchArgument('effective_speed', default_value='8.0'),
        DeclareLaunchArgument('platform_height', default_value='1.0'),
        DeclareLaunchArgument('land_clearance', default_value='0.2'),
        DeclareLaunchArgument('apf_enable', default_value='false'),
        DeclareLaunchArgument('obstacle_map', default_value=''),
        DeclareLaunchArgument('apf_influence_radius', default_value='15.0'),
        DeclareLaunchArgument('apf_gain', default_value='6.0'),
        DeclareLaunchArgument('apf_vel_cap', default_value='6.0'),
        # A*+정적회랑+MPC. mpc_enable:=false로 두면 이 launch도 사실상
        # truck_mission.launch.py의 apf_enable 스위치만 남는 셈이라, 벤치마크
        # 비교 시에는 반드시 mpc_enable:=true를 같이 넘길 것.
        DeclareLaunchArgument('mpc_enable', default_value='false'),
        DeclareLaunchArgument('mpc_horizon', default_value='10'),
        DeclareLaunchArgument('mpc_dt', default_value='0.3'),
        DeclareLaunchArgument('mpc_vmax', default_value='0.0'),   # 0 -> vel_max 그대로 사용
        DeclareLaunchArgument('mpc_replan_period_s', default_value='2.0'),
        DeclareLaunchArgument('mpc_solve_period_s', default_value='0.15'),
        DeclareLaunchArgument('mpc_corridor_margin', default_value='1.5'),
        DeclareLaunchArgument('mpc_q_track', default_value='1.0'),
        DeclareLaunchArgument('mpc_r_smooth', default_value='0.08'),
        DeclareLaunchArgument('mpc_r_effort', default_value='0.01'),
        DeclareLaunchArgument('mpc_corridor_weight', default_value='40.0'),
        DeclareLaunchArgument('mpc_iters', default_value='60'),
        common_bringup,
        mission_manager,
    ])
