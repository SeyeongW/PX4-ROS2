# Standalone build for the Gazebo Classic joint controller.
#
# Plain CMake on purpose, NOT colcon: the gz version of this folder needs no
# build at all, and requiring a ROS workspace just to get one .so would be the
# only thing stopping `gimbal_camera` from being copied anywhere and run.
#
#     source /opt/ros/humble/setup.bash
#     cmake -S plugins -B plugins/build && cmake --build plugins/build
#
# The .so lands in plugins/build/, which the Classic launch file puts on
# GAZEBO_PLUGIN_PATH.
cmake_minimum_required(VERSION 3.10)
project(gimbal_camera_plugins CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gazebo Classic 11 pulls in ignition-cmake2 -> FindJSONCPP -> the system
# jsoncppConfig.cmake, which still declares cmake_minimum_required(VERSION 3.0).
# CMake 4 removed compatibility with < 3.5, so find_package(gazebo) dies inside
# a file we do not own.  Keep treating those as 3.5.  Remove once the distro
# ships a jsoncpp config with a modern minimum.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

find_package(gazebo REQUIRED)     # libgazebo-dev
find_package(rclcpp REQUIRED)     # source /opt/ros/humble/setup.bash first
find_package(std_msgs REQUIRED)

add_library(gimbal_joint_position_controller SHARED
  gimbal_joint_position_controller.cpp)

target_include_directories(gimbal_joint_position_controller
  PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_directories(gimbal_joint_position_controller
  PRIVATE ${GAZEBO_LIBRARY_DIRS})
target_link_libraries(gimbal_joint_position_controller
  ${GAZEBO_LIBRARIES}
  rclcpp::rclcpp
  ${std_msgs_TARGETS})

# Also installable, so a colcon build of the repo puts the .so somewhere the
# launch file looks.  Harmless for the standalone `cmake -S plugins` route,
# which simply never runs the install step.
install(TARGETS gimbal_joint_position_controller LIBRARY DESTINATION lib)
