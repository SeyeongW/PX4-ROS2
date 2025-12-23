cmake_minimum_required(VERSION 3.8)
project(opencv_webcam_sub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 의존성 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 실행 파일 생성 (노드)
add_executable(opencv_webcam_sub src/opencv_webcam_sub.cpp)

# 의존성 링크
ament_target_dependencies(opencv_webcam_sub 
  rclcpp 
  sensor_msgs 
  image_transport 
  cv_bridge 
  OpenCV
)

# 설치 설정
install(TARGETS
  opencv_webcam_sub
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()