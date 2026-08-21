#!/usr/bin/env python3
"""Launch the PX4 CGO3 gimbal camera in Gazebo Sim (Harmonic) -- standalone.

Self-contained: paths are resolved relative to this file, so the whole
`gimbal_camera` folder can live anywhere (e.g. the home directory) and be run
with:

    ros2 launch ~/gimbal_camera/launch/gimbal_camera.launch.py

Only ROS 2 + the ros_gz packages (ros_gz_sim / ros_gz_bridge / ros_gz_image)
need to be installed and sourced -- no colcon build required.

The gimbal joints are driven by gz's JointPositionController (PID inside gz),
so control is by *target angle*, not torque -- this is what keeps it steady.

Bridged topics:
  ROS -> gz : /gimbal/yaw_cmd   (std_msgs/Float64)  target yaw   angle [rad]
              /gimbal/pitch_cmd (std_msgs/Float64)  target pitch angle [rad]
              /gimbal/roll_cmd  (std_msgs/Float64)  target roll  angle [rad]
  gz -> ROS : /gimbal/joint_states (sensor_msgs/JointState)
              /gimbal/camera        (sensor_msgs/Image)
              /gimbal/camera_info   (sensor_msgs/CameraInfo)

Drive it with the keyboard teleop script:
    ~/gimbal_camera/scripts/gimbal_keyboard_control.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Root of the gimbal_camera folder (parent of this launch/ directory).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(ROOT, 'worlds', 'gimbal.sdf')
    models_dir = os.path.join(ROOT, 'models')

    # Let the world's <include> of model://gimbal (and its meshes) resolve.
    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', models_dir)

    # gz_version=8 forces Harmonic (`gz sim`); ros_gz_sim otherwise defaults to
    # 6 (Fortress / `ign gazebo`), which cannot read this SDF 1.9/1.10 world.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {world}',
            'gz_version': '8',
        }.items(),
    )

    # Bridge control/feedback topics.  Direction tokens:
    #   ]  ROS -> gz      [  gz -> ROS      @  bidirectional
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/gimbal/command/gimbal_yaw@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/gimbal/command/gimbal_pitch'
            '@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/gimbal/command/gimbal_roll'
            '@std_msgs/msg/Float64]gz.msgs.Double',
            '/gimbal/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/gimbal/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/model/gimbal/command/gimbal_yaw', '/gimbal/yaw_cmd'),
            ('/model/gimbal/command/gimbal_pitch', '/gimbal/pitch_cmd'),
            ('/model/gimbal/command/gimbal_roll', '/gimbal/roll_cmd'),
            ('/gimbal/joint_state', '/gimbal/joint_states'),
        ],
        output='screen',
    )

    # Dedicated image bridge (handles gz.msgs.Image -> sensor_msgs/Image).
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/gimbal/camera'],
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action(set_resource_path)
    ld.add_action(gz_sim)
    ld.add_action(bridge)
    ld.add_action(image_bridge)
    return ld
