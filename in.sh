#!/bin/bash

# px4_ros2_offboard 컨테이너 내부로 접속하는 단축 스크립트

# 컨테이너 이름 확인 (만약 다르다면 아래를 수정하세요)
CONTAINER_NAME="px4_ros2_offboard"

echo ">>> [${CONTAINER_NAME}] 컨테이너 내부로 접속합니다..."
docker exec -it $CONTAINER_NAME bash
