import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # MicroXRCEAgent: UDP4 port 8888
    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )

    # PX4 SITL with Gazebo (X500 Mono Cam Down)
    # Note: Make sure your PX4 Autopilot path is correct
    px4_sitl = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500_mono_cam_down'],
        cwd=os.path.expanduser('~/PX4-Autopilot'), # Update path if different
        output='screen'
    )

    # YOLO Processor Node
    yolo_processor = Node(
        package='camera_detection',
        executable='yolo_processor_sim_node',
        output='screen'
    )

    return LaunchDescription([
        xrce_agent,
        px4_sitl,
        yolo_processor,
    ])
