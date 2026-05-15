#!/bin/bash

# --- 설정 ---
DOCKER_ID="seyeongw"  # 본인의 Docker Hub ID로 변경하세요
IMAGE_NAME="px4-ros2-offboard"
TAG="latest"

echo ">>> Docker 이미지 배포를 시작합니다: $DOCKER_ID/$IMAGE_NAME:$TAG"

# 1. 로컬 이미지 이름 확인 (docker-compose에서 생성된 이름)
LOCAL_IMAGE="px4-ros2_offboard_dev"

# 2. 이미지 태그 생성
echo ">>> 태그 생성 중..."
docker tag $LOCAL_IMAGE $DOCKER_ID/$IMAGE_NAME:$TAG

# 3. 도커 허브로 푸시
echo ">>> 도커 허브로 푸시 중 (시간이 다소 소요될 수 있습니다)..."
docker push $DOCKER_ID/$IMAGE_NAME:$TAG

if [ $? -eq 0 ]; then
    echo ">>> 배포 성공! 팀원들은 이제 이 이미지를 사용할 수 있습니다."
    echo ">>> 이미지 주소: $DOCKER_ID/$IMAGE_NAME:$TAG"
else
    echo ">>> 배포 실패. 'docker login' 상태를 확인하세요."
fi
