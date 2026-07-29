#!/usr/bin/env python3
"""Launch stock PX4 MAVROS configuration in the Gazebo SITL clock domain."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _argument(name: str, default: str, description: str):
    return DeclareLaunchArgument(
        name,
        default_value=default,
        description=description,
    )


def generate_launch_description() -> LaunchDescription:
    mavros_share = Path(get_package_share_directory("mavros"))
    sitl_clock = (
        Path(__file__).resolve().parents[1]
        / "config/mavros_px4_sitl.yaml"
    )
    if not sitl_clock.is_file():
        raise RuntimeError(
            f"MAVROS SITL clock configuration is missing: {sitl_clock}"
        )

    mavros = Node(
        package="mavros",
        executable="mavros_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            {
                "fcu_url": LaunchConfiguration("fcu_url"),
                "gcs_url": LaunchConfiguration("gcs_url"),
                "tgt_system": ParameterValue(
                    LaunchConfiguration("tgt_system"),
                    value_type=int,
                ),
                "tgt_component": ParameterValue(
                    LaunchConfiguration("tgt_component"),
                    value_type=int,
                ),
                "fcu_protocol": LaunchConfiguration("fcu_protocol"),
            },
            str(mavros_share / "launch/px4_pluginlists.yaml"),
            str(mavros_share / "launch/px4_config.yaml"),
            # Keep this last so the repository-owned simulation clock contract
            # cannot be overridden by a stock wildcard parameter file.
            str(sitl_clock),
        ],
    )

    return LaunchDescription(
        [
            _argument(
                "fcu_url",
                "udp://:14540@127.0.0.1:14580",
                "MAVLink FCU endpoint.",
            ),
            _argument("gcs_url", "", "Optional MAVLink GCS endpoint."),
            _argument("tgt_system", "1", "MAVLink target system ID."),
            _argument("tgt_component", "1", "MAVLink target component ID."),
            _argument("fcu_protocol", "v2.0", "MAVLink protocol version."),
            _argument("namespace", "mavros", "ROS namespace."),
            mavros,
        ]
    )
