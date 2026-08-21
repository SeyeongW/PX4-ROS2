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

# Gazebo Classic is OPTIONAL here, and that is deliberate. This package is
# simulator-only, but it sits in a workspace that also gets built on the flight
# computer — where there is no simulator and no reason to install one. A hard
# REQUIRED turns "the Jetson has no Gazebo" into "the whole workspace fails to
# build", which is a bad trade for a plugin that machine will never load.
find_package(gazebo QUIET)
find_package(rclcpp REQUIRED)     # source /opt/ros/humble/setup.bash first
find_package(std_msgs REQUIRED)

if(NOT gazebo_FOUND)
  message(STATUS
    "gimbal_camera_plugins: Gazebo Classic not found — skipping the gimbal "
    "joint controller. This is expected on a flight computer; install "
    "libgazebo-dev only where you intend to RUN the Classic simulation.")
  return()
endif()

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
