#!/usr/bin/env bash
# 실기체 지상 관측 시스템(Ground Control Station) 통합 실행 스크립트
# 구성:
#   1) SIYI A8 mini RTSP 카메라 드라이버 (siyi_camera_node)
#   2) SIYI A8 mini UDP 짐벌 드라이버 (siyi_gimbal_node)
#   3) HSV + 타원 피팅 풍선 검출기 (balloon_detector_node)
#   4) 상공 자동 탐색(Sky Search) 및 비주얼 서보잉 락온 제어기 (gimbal_tracker_node)
#   5) 6-DOF UKF 3차원 위치/속도 추정기 (balloon_locator_node)
#   6) 실시간 관측 뷰어 및 발진 승인 트리거 UI (detection_viewer_node)
#   7) [--with-bridge 옵션] SiK 텔레메트리 라디오(/dev/ttyUSB0) 10Hz MAVLink 송신 브리지 (mavlink_bridge_node)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/balloon_locator/config/balloon_locator.yaml}"
WITH_BRIDGE=0

for arg in "$@"; do
  case "$arg" in
    --with-bridge)
      WITH_BRIDGE=1
      ;;
    --config=*)
      CONFIG_FILE="${arg#*=}"
      ;;
  esac
done

rm -f /tmp/balloon_launch_approved

if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi
if [ -f "${WS_DIR}/install/setup.bash" ]; then
  set +u
  source "${WS_DIR}/install/setup.bash"
  set -u
fi
if [ -f "${ROOT_DIR}/install/setup.bash" ]; then
  set +u
  source "${ROOT_DIR}/install/setup.bash"
  set -u
fi

echo "========================================================================"
echo " [1/3] 오프라인 하드웨어 설정 점검 (hardware_check)"
echo "       설정 파일: ${CONFIG_FILE}"
echo "========================================================================"
if [ "${WITH_BRIDGE}" -eq 1 ]; then
  ros2 run balloon_locator hardware_check --config "${CONFIG_FILE}" --bridge || true
else
  ros2 run balloon_locator hardware_check --config "${CONFIG_FILE}" || true
fi

PIDS=()
cleanup() {
  echo ""
  echo "[run_hardware_gcs] 지상국 프로세스를 안전하게 종료합니다..."
  for pid in "${PIDS[@]}"; do
    kill -INT "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "========================================================================"
echo " [2/3] 지상 관측 뷰어 + 발진 승인 트리거 UI 기동 (SPACE / ENTER 키 승인)"
echo "========================================================================"
ros2 run balloon_locator detection_viewer_node --ros-args \
  -p ground_image_topic:=/camera/image_raw \
  -p ground_detection_topic:=/balloon/detections \
  -p ukf_status_topic:=/balloon/status &
PIDS+=($!)

if [ "${WITH_BRIDGE}" -eq 1 ]; then
  echo "========================================================================"
  echo " [2.5/3] SiK 텔레메트리 라디오 브리지(mavlink_bridge_node, 10Hz) 기동"
  echo "========================================================================"
  ros2 run balloon_locator mavlink_bridge_node --ros-args --params-file "${CONFIG_FILE}" &
  PIDS+=($!)
fi

echo "========================================================================"
echo " [3/3] SIYI A8 mini 센서 스테이션 & 6-DOF UKF 기동"
echo "========================================================================"
ros2 launch balloon_locator ground_station.launch.py config:="${CONFIG_FILE}" &
PIDS+=($!)

wait "${PIDS[-1]}"
