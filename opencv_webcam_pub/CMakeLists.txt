cmake_minimum_required(VERSION 3.8)
project(opencv_webcam_pub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(opencv_webcam_pub src/opencv_webcam_pub.cpp)
ament_target_dependencies(opencv_webcam_pub rclcpp sensor_msgs cv_bridge image_transport OpenCV)

install(TARGETS
  opencv_webcam_pub
  DESTINATION lib/${PROJECT_NAME}
)

#install(DIRECTORY launch
#  DESTINATION share/${PROJECT_NAME}
#)

ament_package()
