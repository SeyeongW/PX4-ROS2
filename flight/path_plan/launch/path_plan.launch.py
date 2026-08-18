"""Bring up the full path_plan pipeline: A* -> SFC -> B-spline -> MPC.

Topic wiring (private ~/ topics remapped to shared names):

    astar  --global_path-->  sfc, bspline
    sfc    --corridor----->  bspline
    bspline --trajectory-->  mpc
    (external) --odometry, depth--> mpc --cmd_vel--> (PX4 OFFBOARD bridge)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Default map lives in the repo's gazebo tree (not installed in the package).
_DEFAULT_MAP = os.path.expanduser(
    "~/ros2_ws/PX4-ROS2/gazebo/maps/city_coordinates_uav.yaml")

_GLOBAL_PATH = "/path_plan/global_path"
_CORRIDOR = "/path_plan/corridor"
_TRAJECTORY = "/path_plan/trajectory"


def generate_launch_description():
    config = os.path.join(get_package_share_directory("path_plan"),
                          "config", "city_uav.yaml")
    map_yaml = LaunchConfiguration("map_yaml")
    map_param = {"map_yaml": map_yaml}

    def node(pkg_exec, name, remaps):
        return Node(package="path_plan", executable=pkg_exec, name=name,
                    output="screen", parameters=[config, map_param],
                    remappings=remaps)

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=_DEFAULT_MAP),
        node("astar_node", "astar_planner",
             [("~/global_path", _GLOBAL_PATH)]),
        node("sfc_node", "sfc_builder",
             [("~/global_path", _GLOBAL_PATH), ("~/corridor", _CORRIDOR)]),
        # The optimizer owns corridor generation + rebound, so it needs only the
        # A* path (plus the map, passed as a parameter to every node).
        node("bspline_node", "bspline_optimizer",
             [("~/global_path", _GLOBAL_PATH), ("~/trajectory", _TRAJECTORY)]),
        node("mpc_node", "tracking_mpc",
             [("~/trajectory", _TRAJECTORY),
              ("~/odometry", "/path_plan/odometry"),
              ("~/depth", "/path_plan/depth"),
              ("~/cmd_vel", "/path_plan/cmd_vel")]),
    ])
