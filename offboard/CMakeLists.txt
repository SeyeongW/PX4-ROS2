cmake_minimum_required(VERSION 3.8)
project(offboard)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(offboard_control src/offboard_control.cpp)
ament_target_dependencies(offboard_control rclcpp px4_msgs std_msgs)

add_executable(offboard_sim_waypoints src/offboard_sim_waypoints.cpp)
ament_target_dependencies(offboard_sim_waypoints rclcpp px4_msgs)

install(TARGETS
  offboard_control
  offboard_sim_waypoints
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
