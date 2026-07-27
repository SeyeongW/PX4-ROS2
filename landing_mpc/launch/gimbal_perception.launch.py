"""Perception + pointing chain for the gimbal vehicle (x500_gimbal_rgbd_lidar).

Same nodes as the body-fixed chain, repointed at the gimbal payload, plus the
`gimbal_control_node` that aims it:

    /gimbal_camera/image -> aruco_detector_node -> /aruco/pose_cam
    /aruco/pose_cam + /gimbal_camera/imu -> marker_tf_node -> /marker/measured
    /marker/measured -> marker_kf_node -> /marker/position + /marker/velocity
    /marker/{cue,position} + airframe attitude -> gimbal_control_node -> joints

Two settings are load-bearing and are set here rather than left to defaults:

* ``camera_frame=gimbal`` — the lens no longer shares the airframe's attitude,
  so `marker_tf_node` must fold in the gimbal joint angles.  Leaving it at
  ``body`` reintroduces the exact tilt error the gimbal removes.
* ``world`` — `gimbal_control_node` reads the joint encoders straight off
  gz-transport, whose topic is world-scoped.  A wrong world means no
  `/gimbal/joint_state`, and `marker_tf_node` then drops every detection
  (both nodes say so loudly rather than going quiet).
* ``use_sim_time`` — image stamps are sim time, and the whole chain
  interpolates vehicle state to them.

Run the vehicle with ``GIMBAL=1 ./gazebo/run_px4_map.sh precision_landing_200m_moving``.
The mission itself (trailer_cue_node, mission_manager_node, px4_node) is
unchanged and launched separately: the gimbal is a sensor-pointing concern, and
keeping it out of the mission state machine is what makes a body-fixed vs
gimbal comparison a controlled experiment.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_partition():
    """Match what run_px4_map.sh exports, so the two halves find each other.

    `gimbal_control_node` talks to Gazebo over gz-transport, which silently
    sees NOTHING when the partition differs — no error, just no encoders and a
    gimbal that never moves.  The launcher uses px4_ros2_$USER, so default to
    the same thing rather than making every caller remember to export it.
    An already-exported GZ_PARTITION still wins.
    """
    return os.environ.get('GZ_PARTITION',
                          f"px4_ros2_{os.environ.get('USER', 'user')}")


def generate_launch_description():
    model = LaunchConfiguration('model_name')
    world = LaunchConfiguration('world')
    marker_size = LaunchConfiguration('marker_size_m')
    deck_z = LaunchConfiguration('deck_z')
    sim_time = {'use_sim_time': True}

    return LaunchDescription([
        DeclareLaunchArgument(
            'gz_partition', default_value=_default_partition(),
            description='Gazebo transport partition; must match the launcher.'),
        SetEnvironmentVariable('GZ_PARTITION',
                               LaunchConfiguration('gz_partition')),
        DeclareLaunchArgument(
            'model_name', default_value='x500_gimbal_rgbd_lidar_0',
            description='gz entity name — PX4 suffixes the SDF model with _0.'),
        DeclareLaunchArgument(
            'world', default_value='precision_landing_200m_moving',
            description='gz world name; the joint_state topic is scoped to it.'),
        DeclareLaunchArgument(
            'marker_size_m', default_value='1.5',
            description='Black-code side length; the 1.95 m plane includes the '
                        'quiet zone, so 1.5 is what solvePnP needs.'),
        # 1.811 = 2.051 (marker_surface_height_m in the map yaml, and what the
        # model composes to: platform_link sits at model z 2.0 with the marker
        # 0.051 above it) minus the 0.24 m by which x500_base offsets its
        # merged frame.  Confirmed in flight: with use_deck_z:=false the chain
        # reported 1.686 m, which is 1.811 less the 0.13 m camera lever arm
        # noted below.
        DeclareLaunchArgument(
            'deck_z', default_value='1.811',
            description='Marker surface height in PX4 local ENU.'),
        DeclareLaunchArgument(
            'use_deck_z', default_value='true',
            description='Pin the marker z to deck_z instead of trusting the '
                        'solvePnP range. false is useful for diagnostics.'),
        Node(
            package='landing_mpc', executable='gimbal_control_node',
            name='gimbal_control_node', output='screen',
            parameters=[sim_time, {'model_name': model, 'world': world,
                                   'deck_z': deck_z}],
        ),
        Node(
            package='landing_mpc', executable='aruco_detector_node',
            name='aruco_detector_node', output='screen',
            parameters=[sim_time, {
                'image_topic': '/gimbal_camera/image',
                'camera_info_topic': '/gimbal_camera/camera_info',
                'optical_frame_id': 'gimbal_camera_optical_frame',
                'marker_size_m': marker_size,
            }],
        ),
        Node(
            package='landing_mpc', executable='marker_tf_node',
            name='marker_tf_node', output='screen',
            parameters=[sim_time, {
                'camera_frame': 'gimbal',
                'joint_state_topic': '/gimbal/joint_state',
                'deck_z': deck_z,
                'use_deck_z': LaunchConfiguration('use_deck_z'),
            }],
        ),
        Node(
            package='landing_mpc', executable='marker_kf_node',
            name='marker_kf_node', output='screen',
            parameters=[sim_time],
        ),
    ])
