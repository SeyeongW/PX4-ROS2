#!/usr/bin/env python3
"""Bridge the PX4 RGB-D / lidar payload and publish its static TF tree.

The Gazebo-native names are launch arguments.  ROS consumers always receive
the stable contract below, independent of the selected world or PX4 entity:

* ``/front_camera/image`` and ``/front_camera/camera_info``
* ``/front_depth/image`` (mono8 display), ``/front_depth/image_raw`` (32FC1 metric), points and info
* ``/down_camera/image`` and ``/down_camera/camera_info``
* ``/down_depth/image`` (mono8 display), ``/down_depth/image_raw`` (32FC1 metric), points and info
* ``/down_lidar`` and ``/down_lidar/points``
* ``/gimbal_camera/image``, ``/gimbal_camera/camera_info`` and
  ``/gimbal_camera/imu`` (gimbal vehicle only; silent on every other vehicle)

Camera link frames use Gazebo's +X-forward convention.  Their optical children
use REP-103 camera axes (+X right, +Y down, +Z forward).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import importlib.util
from pathlib import Path
import sys


def _static_tf(name, parent, child, xyz, rpy):
    """Create one Humble-compatible static_transform_publisher action."""

    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="log",
        arguments=[
            "--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
            "--roll", str(rpy[0]), "--pitch", str(rpy[1]), "--yaw", str(rpy[2]),
            "--frame-id", parent, "--child-frame-id", child,
        ],
    )


def _launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration("world").perform(context)
    model = LaunchConfiguration("model").perform(context)
    base_frame = LaunchConfiguration("base_frame").perform(context)
    backend = LaunchConfiguration("bridge_backend").perform(context)
    # Normalise to a bool LITERAL: the native bridge declares gimbal_only as a
    # BOOL parameter, so "-p gimbal_only:=1" is parsed as INTEGER and crashes
    # the node (which takes /clock and the mission down with it).  Accept the
    # usual truthy spellings from any caller and always emit true/false.
    gimbal_only = (
        "true"
        if LaunchConfiguration("gimbal_only").perform(context).strip().lower()
        in ("1", "true", "yes", "on")
        else "false"
    )
    clock_bridge_rate_hz = LaunchConfiguration(
        "clock_bridge_rate_hz"
    ).perform(context)
    image_transport_max_rate_hz = LaunchConfiguration(
        "image_transport_max_rate_hz"
    ).perform(context)
    camera_info_transport_max_rate_hz = LaunchConfiguration(
        "camera_info_transport_max_rate_hz"
    ).perform(context)
    cloud_transport_max_rate_hz = LaunchConfiguration(
        "cloud_transport_max_rate_hz"
    ).perform(context)
    lidar_transport_max_rate_hz = LaunchConfiguration(
        "lidar_transport_max_rate_hz"
    ).perform(context)

    front_rgb_image = LaunchConfiguration("front_rgb_image_gz_topic").perform(context)
    front_rgb_info = LaunchConfiguration("front_rgb_info_gz_topic").perform(context)
    front_depth = LaunchConfiguration("front_depth_gz_topic").perform(context)
    front_depth_points = f"{front_depth}/points"
    front_depth_info = LaunchConfiguration("front_depth_info_gz_topic").perform(context)
    down_rgb_image = LaunchConfiguration("down_rgb_image_gz_topic").perform(context)
    down_rgb_info = LaunchConfiguration("down_rgb_info_gz_topic").perform(context)
    down_depth = LaunchConfiguration("down_depth_gz_topic").perform(context)
    down_depth_points = f"{down_depth}/points"
    down_depth_info = LaunchConfiguration("down_depth_info_gz_topic").perform(context)
    gimbal_rgb_image = LaunchConfiguration("gimbal_rgb_image_gz_topic").perform(context)
    gimbal_rgb_info = LaunchConfiguration("gimbal_rgb_info_gz_topic").perform(context)
    gimbal_imu = (
        f"/world/{world}/model/{model}/link/camera_link/sensor/camera_imu/imu"
    )

    lidar_base = (
        f"/world/{world}/model/{model}/link/lidar_sensor_link/"
        "sensor/lidar/scan"
    )
    lidar_points = f"{lidar_base}/points"

    bridge_arguments = [
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"{front_rgb_image}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{front_rgb_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"{front_depth}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{front_depth_points}@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            f"{front_depth_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"{down_rgb_image}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{down_rgb_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"{down_depth}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{down_depth_points}@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            f"{down_depth_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"{gimbal_rgb_image}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{gimbal_rgb_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            f"{gimbal_imu}@sensor_msgs/msg/Imu[gz.msgs.IMU",
            f"{lidar_base}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            f"{lidar_points}@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
    ]
    bridge_remappings = [
            (front_rgb_image, "/front_camera/image"),
            (front_rgb_info, "/front_camera/camera_info"),
            (front_depth, "/front_depth/image_raw"),
            (front_depth_points, "/front_depth/points"),
            (front_depth_info, "/front_depth/camera_info"),
            (down_rgb_image, "/down_camera/image"),
            (down_rgb_info, "/down_camera/camera_info"),
            (down_depth, "/down_depth/image_raw"),
            (down_depth_points, "/down_depth/points"),
            (down_depth_info, "/down_depth/camera_info"),
            (gimbal_rgb_image, "/gimbal_camera/image"),
            (gimbal_rgb_info, "/gimbal_camera/camera_info"),
            (gimbal_imu, "/gimbal_camera/imu"),
            (lidar_base, "/down_lidar"),
            (lidar_points, "/down_lidar/points"),
    ]
    if backend == "native":
        source_bridge = (
            Path(__file__).resolve().parents[2]
            / "gz_bridge/gz_bridge/native_gz_sensor_bridge.py"
        )
        try:
            installed = importlib.util.find_spec(
                "gz_bridge.native_gz_sensor_bridge"
            )
        except (ImportError, ModuleNotFoundError):
            installed = None
        native_bridge = (
            Path(installed.origin)
            if installed is not None and installed.origin is not None
            else source_bridge
        )
        if not native_bridge.is_file():
            raise RuntimeError(f"native sensor bridge not found: {native_bridge}")
        bridge = ExecuteProcess(
            cmd=[
                sys.executable, str(native_bridge), "--ros-args",
                "-p", f"world:={world}",
                "-p", f"model:={model}",
                "-p", f"clock_bridge_rate_hz:={clock_bridge_rate_hz}",
                "-p", (
                    "image_transport_max_rate_hz:="
                    f"{image_transport_max_rate_hz}"
                ),
                "-p", (
                    "camera_info_transport_max_rate_hz:="
                    f"{camera_info_transport_max_rate_hz}"
                ),
                "-p", (
                    "cloud_transport_max_rate_hz:="
                    f"{cloud_transport_max_rate_hz}"
                ),
                "-p", (
                    "lidar_transport_max_rate_hz:="
                    f"{lidar_transport_max_rate_hz}"
                ),
                "-p", f"front_rgb_image_gz_topic:={front_rgb_image}",
                "-p", f"front_rgb_info_gz_topic:={front_rgb_info}",
                "-p", f"front_depth_image_gz_topic:={front_depth}",
                "-p", f"front_depth_points_gz_topic:={front_depth_points}",
                "-p", f"front_depth_info_gz_topic:={front_depth_info}",
                "-p", f"down_rgb_image_gz_topic:={down_rgb_image}",
                "-p", f"down_rgb_info_gz_topic:={down_rgb_info}",
                "-p", f"down_depth_image_gz_topic:={down_depth}",
                "-p", f"down_depth_points_gz_topic:={down_depth_points}",
                "-p", f"down_depth_info_gz_topic:={down_depth_info}",
                "-p", f"gimbal_rgb_image_gz_topic:={gimbal_rgb_image}",
                "-p", f"gimbal_rgb_info_gz_topic:={gimbal_rgb_info}",
                "-p", f"gimbal_imu_gz_topic:={gimbal_imu}",
                "-p", f"lidar_gz_topic:={lidar_base}",
                "-p", f"gimbal_only:={gimbal_only}",
            ],
            name="native_gz_sensor_bridge",
            output="screen",
        )
    elif backend == "ros_gz":
        if gimbal_only.lower() in ("1", "true", "yes"):
            # keep only clock + the gimbal payload
            keep = ("/clock", gimbal_rgb_image, gimbal_rgb_info, gimbal_imu)
            bridge_arguments = [a for a in bridge_arguments
                                if a.split("@", 1)[0] in keep]
            bridge_remappings = [r for r in bridge_remappings if r[0] in keep]
        bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="sensor_bridge",
            output="screen",
            arguments=bridge_arguments,
            remappings=bridge_remappings,
        )
    else:
        raise RuntimeError("bridge_backend must be 'native' or 'ros_gz'")

    optical_rpy = (-1.57079632679, 0.0, -1.57079632679)
    transforms = [
        _static_tf("tf_front_rgb_link", base_frame, "front_rgb_link",
                   (0.12, 0.03, 0.002), (0.0, 0.0, 0.0)),
        _static_tf("tf_front_rgb_optical", "front_rgb_link",
                   "front_camera_optical_frame", (0.0, 0.0, 0.0), optical_rpy),
        _static_tf("tf_front_depth_link", base_frame, "front_depth_link",
                   (0.12, 0.0, 0.002), (0.0, 0.0, 0.0)),
        _static_tf("tf_front_depth_optical", "front_depth_link",
                   "front_depth_optical_frame", (0.0, 0.0, 0.0), optical_rpy),
        _static_tf("tf_down_rgb_link", base_frame, "down_rgb_link",
                   (0.0, 0.0, -0.05), (0.0, 1.57079632679, 0.0)),
        _static_tf("tf_down_rgb_optical", "down_rgb_link",
                   "down_camera_optical_frame", (0.0, 0.0, 0.0), optical_rpy),
        _static_tf("tf_down_depth_link", base_frame, "down_depth_link",
                   (0.0, -0.03, -0.05), (0.0, 1.57079632679, 0.0)),
        _static_tf("tf_down_depth_optical", "down_depth_link",
                   "down_depth_optical_frame", (0.0, 0.0, 0.0), optical_rpy),
        _static_tf("tf_down_lidar", base_frame, "lidar_sensor_link",
                   (0.08, 0.0, -0.20), (0.0, 1.57079632679, 0.0)),
    ]
    return [bridge, *transforms]


def _argument(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description():
    return LaunchDescription([
        _argument("world", "applepark_city", "Gazebo world name."),
        _argument("model", "x500_city_rgbd_lidar_0", "PX4-spawned Gazebo entity."),
        _argument("base_frame", "base_link", "ROS body frame for static sensor TF."),
        _argument(
            "bridge_backend", "native",
            "native for gz-sim8 transport13, ros_gz for ABI-compatible Garden.",
        ),
        _argument(
            "gimbal_only", "false",
            "Bridge only the gimbal camera (+clock); skip front/down RGB, "
            "depth, clouds and lidar. Cuts bridge load on the gimbal run.",
        ),
        _argument(
            "clock_bridge_rate_hz", "50.0",
            "Gazebo /clock callback limit and ROS /clock publication rate.",
        ),
        _argument(
            "image_transport_max_rate_hz", "50.0",
            "Maximum native Gazebo image callback rate per topic.",
        ),
        _argument(
            "camera_info_transport_max_rate_hz", "5.0",
            "Maximum native Gazebo camera-info callback rate per topic.",
        ),
        _argument(
            "cloud_transport_max_rate_hz", "5.0",
            "Maximum native Gazebo point-cloud callback rate per topic.",
        ),
        _argument(
            "lidar_transport_max_rate_hz", "25.0",
            "Maximum native Gazebo lidar callback rate.",
        ),
        _argument(
            "front_rgb_image_gz_topic", "/front_camera/image",
            "Front RGB Gazebo image topic.",
        ),
        _argument(
            "front_rgb_info_gz_topic", "/front_camera/camera_info",
            "Front RGB Gazebo CameraInfo topic.",
        ),
        _argument(
            "front_depth_gz_topic", "/front_depth/image",
            "Front depth Gazebo image topic.",
        ),
        _argument(
            "front_depth_info_gz_topic", "/front_depth/camera_info",
            "Front depth Gazebo CameraInfo topic.",
        ),
        _argument("down_rgb_image_gz_topic", "/down_camera/image", "Down RGB Gazebo image topic."),
        _argument(
            "down_rgb_info_gz_topic", "/down_camera/camera_info",
            "Down RGB Gazebo CameraInfo topic.",
        ),
        _argument(
            "down_depth_gz_topic", "/down_depth/image",
            "Down depth Gazebo image topic.",
        ),
        _argument(
            "down_depth_info_gz_topic", "/down_depth/camera_info",
            "Down depth Gazebo CameraInfo topic.",
        ),
        _argument(
            "gimbal_rgb_image_gz_topic", "/gimbal_camera/image",
            "Gimbal RGB Gazebo image topic (gimbal vehicle only).",
        ),
        _argument(
            "gimbal_rgb_info_gz_topic", "/gimbal_camera/camera_info",
            "Gimbal RGB Gazebo CameraInfo topic (gimbal vehicle only).",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
