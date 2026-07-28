# 착륙 스택 실행 환경 (젯슨).
#
# 보통은 이걸 부를 일이 없습니다 — .bashrc 가 이미 같은 일을 합니다. cron이나
# systemd처럼 .bashrc 를 읽지 않는 곳에서 스택을 띄울 때만 쓰세요.
#
#   source ~/ros2_ws/PX4-ROS2/flight/mpc_landing/flight_env.sh
#
# 예전에 이 파일은 Discovery Server 를 unset 했습니다. 그건 젯슨의 DDS 프로파일에
# interfaceWhiteList 가 없어서 PC 가 그래프를 못 보던 걸 우회한 것이었고, 원인을
# config/jetson_dds_fastdds.sh 에서 고쳤으므로 더는 필요 없습니다. 오히려 지금
# unset 하면 PC 에서 영상과 미션 상태를 못 봅니다.
source /opt/ros/humble/setup.bash
source ~/ros2_ws/PX4-ROS2/install/local_setup.bash
source ~/dds_fastdds.sh
