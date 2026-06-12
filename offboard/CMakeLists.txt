cmake_minimum_required(VERSION 3.8)
project(offboard)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(offboard_control src/offboard_control.cpp)
ament_target_dependencies(offboard_control rclcpp mavros_msgs std_msgs geometry_msgs)

add_executable(offboard_sim_waypoints src/offboard_sim_waypoints.cpp)
ament_target_dependencies(offboard_sim_waypoints rclcpp mavros_msgs geometry_msgs)

add_executable(offboard_tracking_control src/offboard_tracking_control.cpp)
ament_target_dependencies(offboard_tracking_control rclcpp mavros_msgs geometry_msgs)

add_executable(precision_landing src/precision_landing.cpp)
ament_target_dependencies(precision_landing rclcpp mavros_msgs geometry_msgs std_msgs)

add_executable(drone_core_controller src/drone_core_controller.cpp)
ament_target_dependencies(drone_core_controller rclcpp mavros_msgs)

install(TARGETS
  offboard_control
  offboard_sim_waypoints
  offboard_tracking_control
  precision_landing
  drone_core_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
