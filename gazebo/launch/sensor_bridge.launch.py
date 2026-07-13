#!/usr/bin/env python3
"""Bridge the x500_depth_lidar sensors from Gazebo (gz-sim) to ROS 2.

Starts a ros_gz_bridge/parameter_bridge that maps the gz transport topics
published by the forward depth camera, forward RGB camera and downward lidar
onto ROS 2 topics. Run *after* Gazebo + PX4 (with the x500_depth_lidar model)
are already up -- run_px4_map.sh starts this automatically.

    ros2 launch gazebo/launch/sensor_bridge.launch.py world:=ugv_drone_mountain_map
    ros2 launch gazebo/launch/sensor_bridge.launch.py world:=applepark_city

ROS 2 topics produced:

    /clock                     rosgraph_msgs/msg/Clock
    /depth_camera              sensor_msgs/msg/Image        (forward depth image)
    /depth_camera/points       sensor_msgs/msg/PointCloud2  (forward depth cloud)
    /camera_info               sensor_msgs/msg/CameraInfo   (forward depth intrinsics)
    /front_camera/image        sensor_msgs/msg/Image        (forward RGB image)
    /front_camera/camera_info  sensor_msgs/msg/CameraInfo   (forward RGB intrinsics)
    /down_lidar                sensor_msgs/msg/LaserScan    (downward lidar)
    /down_lidar/points         sensor_msgs/msg/PointCloud2  (downward lidar cloud)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration("world").perform(context)
    model = LaunchConfiguration("model").perform(context)

    # The forward RGB camera (OAK-D Lite IMX214) keeps the auto-generated gz
    # topic, which embeds the world and model names.
    rgb_base = f"/world/{world}/model/{model}/link/camera_link/sensor/IMX214"
    rgb_image = f"{rgb_base}/image"
    rgb_info = f"{rgb_base}/camera_info"

    # The downward lidar keeps the default gz topic path so PX4's gz bridge can
    # also consume it as a distance_sensor. Bridge it under the friendlier
    # /down_lidar name for ROS 2 consumers.
    lidar_base = f"/world/{world}/model/{model}/link/lidar_sensor_link/sensor/lidar/scan"
    lidar_points = f"{lidar_base}/points"

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="sensor_bridge",
        output="screen",
        arguments=[
            # simulation clock (gz -> ros)
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # forward depth camera (short gz topics from OAK-D Lite)
            "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image",
            "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # forward RGB camera (world-qualified gz topics, remapped below)
            f"{rgb_image}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{rgb_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # downward lidar (default gz path, shared with PX4 distance_sensor)
            f"{lidar_base}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            f"{lidar_points}@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            (rgb_image, "/front_camera/image"),
            (rgb_info, "/front_camera/camera_info"),
            (lidar_base, "/down_lidar"),
            (lidar_points, "/down_lidar/points"),
        ],
    )

    return [bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value="ugv_drone_mountain_map",
            description="gz world name (ugv_drone_mountain_map or applepark_city).",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="x500_depth_lidar_0",
            description="PX4-spawned gz model entity name.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
