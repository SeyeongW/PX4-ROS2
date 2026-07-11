#!/usr/bin/env python3
"""트럭 미션 공통 브링업 — mission_manager를 뺀 나머지 4개 노드.

MAVROS + 다운카메라 브리지 + ArUco 검출 + precision_landing_node(통합/대기 모드).
장애물 회피 알고리즘이 바뀌는 건 mission_manager_node뿐이라(APF든 MPC든), 그 노드는
여기 안 넣고 최상위 런치파일(truck_mission.launch.py — APF, truck_mission_mpc.launch.py
— A*+정적회랑+MPC)이 각자 필요한 파라미터로 직접 띄운다. 이 파일을 직접
`ros2 launch`하지 않음 — 항상 IncludeLaunchDescription으로만 불러 쓴다.

이렇게 분리한 이유: 두 최상위 launch가 이 4개 노드 정의를 복붙해서 들고 있으면
한쪽만 고치고 다른 쪽을 깜빡하는 사고가 나기 쉬움(예: precision_landing_node
파라미터 하나 튜닝했는데 MPC 쪽 런치는 안 고쳐서 다른 값으로 계속 도는 식). 여기
하나만 고치면 둘 다 자동으로 반영됨.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    image_topic = LaunchConfiguration('image_topic')
    flight_alt = LaunchConfiguration('flight_alt')
    platform_height = LaunchConfiguration('platform_height')
    land_clearance = LaunchConfiguration('land_clearance')
    # 장애물 회피(APF). obstacle_field 월드 + gazebo/config/obstacle_map.yaml 용 —
    # precision_landing_node 쪽은 착륙 핸드오프 전 회귀(RETURN) 비행 구간에서만 씀,
    # mission_manager 쪽(MISSION 구간)의 apf_enable/mpc_enable과는 별개 스위치.
    apf_enable = LaunchConfiguration('apf_enable')
    obstacle_map = LaunchConfiguration('obstacle_map')
    apf_influence_radius = LaunchConfiguration('apf_influence_radius')
    apf_gain = LaunchConfiguration('apf_gain')
    apf_vel_cap = LaunchConfiguration('apf_vel_cap')

    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'apm.launch')

    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_launch),
        launch_arguments={'fcu_url': fcu_url}.items(),
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

    # 정밀착륙 제어 — 통합(대기) 모드: require_enable=true, auto_takeoff=false.
    # mission_manager 의 land_enable 게이트가 열릴 때까지 DORMANT.
    precision_landing = Node(
        package='precision_landing',
        executable='precision_landing_node',
        name='precision_landing_node',
        output='screen',
        parameters=[{
            'require_enable': True,
            'auto_takeoff': False,
            'use_cue': True,
            'flight_alt': flight_alt,
            'platform_height': ParameterValue(platform_height, value_type=float),
            'land_clearance': ParameterValue(land_clearance, value_type=float),
            # 비행 로그로 확정한 매핑/게인(precision_landing.launch.py 와 동일)
            'lat_swap': False,
            'lat_sign_fwd': 1.0,
            'lat_sign_left': 1.0,
            'vel_gain': 0.4,
            'vel_max': 5.0,
            'approach_vel_max': 10.0,
            'approach_decel_s': 5.0,
            'yaw_track': True,
            'yaw_track_min_speed': 0.5,
            # 트랙(이동+곡선) 위 착륙 안정화. 곡선 주행 트럭을 등속 KF가 못 따라가
            # 마커가 좁아진 다운카메라 프레임 밖으로 빠지면서 DESCEND->ALIGN 으로
            # 되돌아가는 limit cycle을 줄인다.
            #   kf_accel_std  : 0.1->2.0->3.5  곡선 기동(횡가속 ~2.5 m/s²)을 KF가 추적.
            #     2.0으로도 "+"자 순찰 트레일러의 급선회(순간 가속 2.0 초과) 직후 KF
            #     속도추정이 새 방향을 수 초간 못 믿어오는 지연이 남아있어 3.5로 추가
            #     상향. 1차 픽스는 _track()이 이제 KF 속도 대신 cue_vel(트레일러가 매
            #     틱 쏘는 정확한 속도)을 우선 피드포워드로 쓰는 것 — 이 파라미터는 cue
            #     가 stale일 때의 폴백 및 KF 위치추정 자체의 반응성을 위한 보조 상향.
            #     너무 올리면 비전 노이즈를 실제 기동으로 오인해 흔들림(jitter)이
            #     커지니, 재테스트 후 필요하면 더 조정할 것.
            #   final_descent_h: 0.6->1.4  더 높은 데서 정확한 큐 개루프 착지로 전환
            #   coast_ticks   : 30->60    비전 끊김 허용 1.5->3.0 s (큐로 코스트)
            #   descend_rate    : 0.3->0.8 m/s  기본 하강속도 (노드 기본값이 너무 느려서
            #     "찔끔찔끔" 내려가는 느낌 — 착륙까지 체감상 오래 걸림)
            #   descend_min_scale: 0.3->0.5    펀넬 벗어났을 때/FINAL 단계에서 쓰는 최소
            #     배율. FINAL(최종 접근, final_descent_h 아래)은 항상
            #     descend_rate*descend_min_scale 로만 내려가므로 이게 특히 체감 속도를
            #     좌우함 — 기존 0.3*0.3=0.09 m/s(9cm/s)에서 0.8*0.5=0.4 m/s로 4배 이상
            #     빨라짐. 너무 올리면 착지 직전 오정렬 상태로 빠르게 내려가 플랫폼을
            #     벗어날 위험이 커지니, 실제 테스트에서 여전히 정렬 유지되는지 확인할 것.
            'kf_accel_std': 3.5,
            'final_descent_h': 1.4,
            'coast_ticks': 60,
            'descend_rate': 0.8,
            'descend_min_scale': 0.5,
            'apf_enable': ParameterValue(apf_enable, value_type=bool),
            'obstacle_map': obstacle_map,
            'apf_influence_radius': ParameterValue(apf_influence_radius, value_type=float),
            'apf_gain': ParameterValue(apf_gain, value_type=float),
            'apf_vel_cap': ParameterValue(apf_vel_cap, value_type=float),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://:14550@'),
        DeclareLaunchArgument('image_topic', default_value='/down_camera/image'),
        DeclareLaunchArgument('flight_alt', default_value='5.0'),
        # 플랫폼(트럭) 위 착륙: 월드 aruco_platform 높이(1.0 m)와 맞춤.
        DeclareLaunchArgument('platform_height', default_value='1.0'),
        DeclareLaunchArgument('land_clearance', default_value='0.2'),
        DeclareLaunchArgument('apf_enable', default_value='false'),
        DeclareLaunchArgument('obstacle_map', default_value=''),
        DeclareLaunchArgument('apf_influence_radius', default_value='15.0'),
        DeclareLaunchArgument('apf_gain', default_value='6.0'),
        DeclareLaunchArgument('apf_vel_cap', default_value='6.0'),
        mavros,
        camera_bridge,
        aruco_detector,
        precision_landing,
    ])
