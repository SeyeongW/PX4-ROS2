# 정밀착륙 스택 실행 환경.
#
# .bashrc 는 모든 터미널을 FastDDS Discovery Server(127.0.0.1:11811)에 붙이는데,
# 그 모드에서는 노드끼리는 통신하지만 `ros2 topic list` / `service list` 가
# 그래프를 못 봅니다. 즉 미션 상태를 못 읽고 approve 도 못 부릅니다.
# 스택은 전부 젯슨 한 대 안에서 도니 Discovery Server 가 필요 없습니다.
# 이 파일을 source 한 터미널은 평범한 멀티캐스트로 돌아갑니다.
#
#   source ~/flight_env.sh      # 스택을 띄우는 터미널과 승인하는 터미널 모두에서
unset ROS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset ROS_SUPER_CLIENT
unset ROS_LOCALHOST_ONLY
source /opt/ros/humble/setup.bash
source ~/ros2_ws/PX4-ROS2/install/local_setup.bash
