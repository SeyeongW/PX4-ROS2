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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_MAP = os.path.expanduser(
    "~/ros2_ws/PX4-ROS2/gazebo/maps/city_coordinates_uav.yaml")
# RViz config lives in the source tree (loaded by absolute path so no rebuild is
# needed to tweak it), matching how the default map is referenced above.
_RVIZ_CONFIG = os.path.expanduser(
    "~/ros2_ws/PX4-ROS2/path_plan/rviz/path_plan.rviz")


def generate_launch_description():
    pkg = get_package_share_directory("path_plan")
    map_yaml = LaunchConfiguration("map_yaml")
    auto_arm = LaunchConfiguration("auto_arm")
    takeoff_alt_m = LaunchConfiguration("takeoff_alt_m")
    land_radius_m = LaunchConfiguration("land_radius_m")
    speed_from_fcu = LaunchConfiguration("speed_from_fcu")
    speed_scale = LaunchConfiguration("speed_scale")
    rviz = LaunchConfiguration("rviz")

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
                     "land_radius_m": land_radius_m,
                     "speed_from_fcu": speed_from_fcu,
                     "speed_scale": speed_scale}],
    )

    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="screen",
        arguments=["-d", _RVIZ_CONFIG],
        condition=IfCondition(rviz),
    )

    # Publish the city buildings as real 3D prisms for RViz (only when RViz is on).
    buildings_node = Node(
        package="path_plan", executable="building_markers",
        name="building_markers", output="screen",
        parameters=[{"map_yaml": map_yaml}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=_DEFAULT_MAP),
        # auto_arm:=false streams odom + setpoints WITHOUT arming (safe dry-run to
        # verify the A*->B-spline->MPC->cmd_vel chain before a real flight).
        DeclareLaunchArgument("auto_arm", default_value="true"),
        # Climb PURELY vertically to the cruise altitude (mid of the 30-40 m band)
        # before any lateral motion.  City buildings are 20-50 m tall, so a lower
        # takeoff (e.g. 20 m) would (a) make the A* start cell sit below the 30 m
        # planning floor -> "start cell blocked", and (b) drive the vehicle sideways
        # through the building band while still climbing -> wall strikes.  Keep this
        # >= cruise_floor_m so the plan starts, and inside the band so it never
        # overshoots the ceiling.
        DeclareLaunchArgument("takeoff_alt_m", default_value="35.0"),
        # Auto-land when the vehicle gets within this xy distance of the goal.
        DeclareLaunchArgument("land_radius_m", default_value="12.0"),
        # Fixed cruise speed from config (v_ref_m_s=10). speed_from_fcu:=true would
        # instead pull PX4's MPC_XY_VEL_MAX and override the config speed.
        DeclareLaunchArgument("speed_from_fcu", default_value="false"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        # Open RViz showing the A* path, MPC preview and the vehicle (rviz:=false
        # to skip it).
        DeclareLaunchArgument("rviz", default_value="true"),
        pipeline,
        bridge,
        rviz_node,
        buildings_node,
        Node(
            package="path_plan", executable="flight_logger",
            name="flight_logger", output="screen",
            parameters=[{"map_yaml": map_yaml}],
        ),
    ])
