"""Attach the path_plan pipeline to a already-running PX4 + MAVROS sim.

Bring the simulator up FIRST with the repo's own launcher (it starts Gazebo, PX4
SITL, MAVROS and the ros_gz sensor bridge):

    ./gazebo/run_px4_map.sh city

Then, in a second terminal, run this to start the planner + the MAVROS OFFBOARD
bridge (this file NEVER starts Gazebo/PX4/MAVROS itself):

    ros2 launch path_plan px4_mavros.launch.py

It includes the standard pipeline launch (A* -> SFC -> B-spline -> MPC, already
wired to /path_plan/odometry and /path_plan/cmd_vel) and adds the
mavros_static_path bridge that feeds PX4 state in and OFFBOARD setpoints out.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_MAP = os.path.expanduser(
    "~/ros2_ws/PX4-ROS2/gazebo/maps/city_coordinates_uav.yaml")


def generate_launch_description():
    pkg = get_package_share_directory("path_plan")
    map_yaml = LaunchConfiguration("map_yaml")
    auto_arm = LaunchConfiguration("auto_arm")
    takeoff_alt_m = LaunchConfiguration("takeoff_alt_m")
    speed_from_fcu = LaunchConfiguration("speed_from_fcu")
    speed_scale = LaunchConfiguration("speed_scale")

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, "launch", "path_plan.launch.py")),
        launch_arguments={"map_yaml": map_yaml}.items(),
    )

    bridge = Node(
        package="path_plan", executable="mavros_static_path",
        name="mavros_static_path", output="screen",
        # map_offset_enu_m = PX4 EKF local origin = vehicle spawn (city map).
        parameters=[{"map_offset_enu_m": [587.0, 580.0, 0.0],
                     "rate_hz": 20.0,
                     "auto_arm": auto_arm,
                     "takeoff_alt_m": takeoff_alt_m,
                     "speed_from_fcu": speed_from_fcu,
                     "speed_scale": speed_scale}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=_DEFAULT_MAP),
        # auto_arm:=false streams odom + setpoints WITHOUT arming (safe dry-run to
        # verify the A*->B-spline->MPC->cmd_vel chain before a real flight).
        DeclareLaunchArgument("auto_arm", default_value="true"),
        DeclareLaunchArgument("takeoff_alt_m", default_value="20.0"),
        # Cruise at PX4's real MPC_XY_VEL_MAX (no hardcoded speed); scale trims it.
        DeclareLaunchArgument("speed_from_fcu", default_value="true"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        pipeline,
        bridge,
    ])
