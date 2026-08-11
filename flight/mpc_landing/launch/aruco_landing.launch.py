"""Everything needed to land on an ArUco marker, in one launch.

    ros2 launch mpc_landing aruco_landing.launch.py

The perception half is identical to `fixed_marker_landing.launch.py` — the same
proven camera + detector + frame chain — only the mission node is swapped: this
runs `aruco_landing_node` (plain proportional centre-and-descend) instead of the
MPC. It is the intermediate rung: prove the marker can be SEEN and FLOWN TO with
simple control before trusting the MPC with the descent.

Starts, in this order:

    gst_camera_node    down camera on the Jetson's NVJPG block -> /down_camera/image
    aruco_pose_node    -> /perception/down/marker_pose + /perception/down/aruco_detected
    siyi_gimbal_node   points the camera down the moment the vehicle arms
    landing_tf_node    map -> base_link -> gimbal -> camera, so the detector can
                       republish the marker in `map` (WITHOUT it the detector
                       publishes nothing — see fixed_marker_landing.launch.py)
    aruco_landing_node the gated mission: takeoff -> search -> centre-and-descend

MAVROS is NOT started here — its fcu_url depends on the companion wiring:

    ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600

APPROVING THE ONE GATE
----------------------
`ros2 launch` does not forward stdin, so the node's ENTER prompt is inert under
launch. Approve from a second terminal:

    ros2 topic echo /aruco_landing_node/state   # what it is waiting for
    ros2 run mpc_landing approve aruco_landing_node
    ros2 run mpc_landing abort   aruco_landing_node   # land now, from any phase

To press ENTER instead, run the node by itself (it detects the terminal):

    ros2 run mpc_landing aruco_landing_node

PARAMETER-FREE, like every launch under flight/: each node's values live in its
own `_declare()`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mission = LaunchConfiguration('mission')
    return LaunchDescription([
        # Set mission:=false to start ONLY the perception stack, and drive the
        # mission node separately with `ros2 run` so its ENTER-to-approve prompt
        # works (ros2 launch does not forward stdin to a child). run_px4 does
        # exactly this.
        DeclareLaunchArgument('mission', default_value='true'),
        # --- camera: hardware JPEG decode on the Jetson's NVJPG block --------
        Node(
            package='aruco_landing',
            executable='gst_camera_node',
            name='down_camera',
            output='screen',
        ),

        # --- perception ----------------------------------------------------
        Node(
            package='aruco_landing',
            executable='aruco_pose_node',
            name='down_aruco_pose',
            output='screen',
        ),

        # --- gimbal: nadir on arm ------------------------------------------
        Node(
            package='siyi_gimbal',
            executable='siyi_gimbal_node',
            name='siyi_gimbal_node',
            output='screen',
        ),

        # --- the frames the marker is measured through ----------------------
        Node(
            package='aruco_landing',
            executable='landing_tf_node',
            name='landing_tf_node',
            output='screen',
        ),

        # --- the gated ArUco landing mission (mission:=false omits it) ------
        Node(
            package='mpc_landing',
            executable='aruco_landing_node',
            name='aruco_landing_node',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(mission),
        ),
    ])
