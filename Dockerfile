FROM ros:humble-ros-base

# 필수 패키지 설치 및 업데이트
RUN apt-get update && apt-get install -y \
    ros-humble-mavros \
    ros-humble-mavros-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# GeographicLib 데이터셋 설치 (빌드 시 미리 포함)
RUN wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh && \
    chmod +x install_geographiclib_datasets.sh && \
    ./install_geographiclib_datasets.sh && \
    rm install_geographiclib_datasets.sh

# 워크스페이스 디렉토리 생성
WORKDIR /ros2_ws

# .bashrc에 환경변수 추가
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 컨테이너 실행 시 기본 명령
CMD ["bash"]
