#!/usr/bin/env python3
"""Launch the PX4 CGO3 gimbal camera in Gazebo CLASSIC (11) -- standalone.

The Classic counterpart of gimbal_camera.launch.py.  Same folder layout, same
ROS topics, same keyboard script:

    ros2 launch ~/gimbal_camera/launch/gimbal_camera_classic.launch.py

What is NOT here is the interesting part.  The gz version needs three nodes to
talk to ROS -- `ros_gz_sim` to host the simulator, `ros_gz_bridge` to translate
and rename five topics, and `ros_gz_image` for the camera -- because gz-sim
speaks gz-transport and nothing else.  Classic's gazebo_ros plugins publish and
subscribe ROS directly, so the bridges have no counterpart: the model's own
plugins ARE the interface.  That is why this file only starts a simulator.

Topics (identical to the gz version, which is the point):
  ROS -> sim : /gimbal/yaw_cmd   (std_msgs/Float64)  target yaw   angle [rad]
               /gimbal/pitch_cmd (std_msgs/Float64)  target pitch angle [rad]
               /gimbal/roll_cmd  (std_msgs/Float64)  target roll  angle [rad]
  sim -> ROS : /gimbal/joint_states (sensor_msgs/JointState)
               /gimbal/camera       (sensor_msgs/Image)
               /gimbal/camera_info  (sensor_msgs/CameraInfo)
               /gimbal/imu          (sensor_msgs/Imu)

Requires, beyond the gz version:
    sudo apt install gazebo11 ros-humble-gazebo-ros-pkgs
    cmake -S ~/gimbal_camera/plugins -B ~/gimbal_camera/plugins/build \
      && cmake --build ~/gimbal_camera/plugins/build

Drive it with the same keyboard teleop script:
    ~/gimbal_camera/scripts/gimbal_keyboard_control.py
"""

import os

from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            ExecuteProcess)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

# Root of the gimbal_camera folder (parent of this launch/ directory).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description():
    world = os.path.join(ROOT, 'worlds', 'gimbal_classic.world')
    models_dir = os.path.join(ROOT, 'models')
    # The controller .so can come from either build route, so both are offered:
    # a standalone `cmake -S plugins -B plugins/build` (this folder copied
    # anywhere, as the README describes), or a colcon build of the repo, which
    # discovers plugins/CMakeLists.txt as the `gimbal_camera_plugins` package
    # and installs it under install/.  Listing both means neither route needs
    # the other to have been used.
    plugins_dir = os.pathsep.join([
        os.path.join(ROOT, 'plugins', 'build'),
        os.path.abspath(os.path.join(ROOT, '..', '..', 'install',
                                     'gimbal_camera_plugins', 'lib')),
    ])

    gui = LaunchConfiguration('gui')

    # model:// URIs in the world and the meshes inside the model both resolve
    # from here.  The Classic model deliberately points its meshes at
    # model://gimbal/meshes/..., i.e. the gz model's copy, so the 2.7 MB of STL
    # is not duplicated -- which only works because both live under models/.
    set_model_path = AppendEnvironmentVariable('GAZEBO_MODEL_PATH', models_dir)

    # Where libgimbal_joint_position_controller.so lands.  Classic resolves a
    # <plugin filename=...> against this path; the gazebo_ros sensor plugins
    # come from the system install and need no help.
    set_plugin_path = AppendEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH', plugins_dir)

    # gzserver/gzclient are started directly rather than through
    # gazebo_ros/gazebo.launch.py so this file keeps working whether or not that
    # package's launch API changes, and so `gui:=false` is a plain condition
    # rather than a nested include argument.
    gzserver = ExecuteProcess(
        cmd=['gzserver', '--verbose', world],
        output='screen',
    )
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition(gui),
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument(
        'gui', default_value='true',
        description='Run gzclient too; set false for a headless render-only run'))
    ld.add_action(set_model_path)
    ld.add_action(set_plugin_path)
    ld.add_action(gzserver)
    ld.add_action(gzclient)
    return ld
