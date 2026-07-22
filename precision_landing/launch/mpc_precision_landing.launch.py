#!/usr/bin/env python3
"""Start only the single Relative-OSQP flight controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _controller_for_profile(
    context: object,
    *,
    base_config: str,
    real_override: str,
) -> list[Node]:
    """Create one controller with a small profile-specific parameter layer."""
    profile = LaunchConfiguration("profile").perform(context).strip().lower()
    if profile not in {"sim", "real"}:
        raise ValueError("profile must be 'sim' or 'real'")

    parameter_files = [base_config]
    if profile == "real":
        parameter_files.append(real_override)
    parameter_files.append({
        "baseline_run_id": LaunchConfiguration("baseline_run_id"),
        "validation_mode": ParameterValue(
            LaunchConfiguration("validation_mode"), value_type=bool
        ),
    })
    return [Node(
        package="precision_landing",
        executable="mpc_precision_landing_node",
        name="mpc_precision_landing",
        output="screen",
        parameters=parameter_files,
    )]


def generate_launch_description() -> LaunchDescription:
    precision_share = get_package_share_directory("precision_landing")
    config_default = os.path.join(
        precision_share, "config", "mpc_precision_landing.yaml"
    )
    real_override = os.path.join(
        precision_share, "config", "mpc_precision_landing_real.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "profile",
            default_value="sim",
            description=(
                "Runtime boundary: 'sim' keeps the SITL base configuration; "
                "'real' layers fail-closed hardware clock settings."
            ),
            choices=["sim", "real"],
        ),
        DeclareLaunchArgument("baseline_run_id", default_value=""),
        DeclareLaunchArgument(
            "validation_mode",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Explicitly authorize an unqualified SITL validation run."
            ),
        ),
        OpaqueFunction(
            function=_controller_for_profile,
            kwargs={
                "base_config": config_default,
                "real_override": real_override,
            },
        ),
    ])
