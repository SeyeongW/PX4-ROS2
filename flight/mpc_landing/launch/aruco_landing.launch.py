"""Everything needed to land on an ArUco marker, in one launch.

    ros2 launch mpc_landing aruco_landing.launch.py

The perception half is identical to `fixed_marker_landing.launch.py` — the same
proven camera + detector + frame chain — only the mission node is swapped.
`run_px4 trailer` adds A* -> SFC -> B-spline and TrackingMPC cruise, then retains
the proven proportional ArUco descent in the same MAVROS authority.

Starts, in this order:

    gst_camera_node    down camera on the Jetson's NVJPG block -> /down_camera/image
    aruco_pose_node    -> /perception/down/marker_pose + /perception/down/aruco_detected
    siyi_gimbal_node   points the camera down the moment the vehicle arms
    landing_tf_node    map -> base_link -> gimbal -> camera, so the detector can
                       republish the marker in `map` (WITHOUT it the detector
                       publishes nothing — see fixed_marker_landing.launch.py)
    aruco_landing_node the gated mission: takeoff -> search -> centre-and-descend

GOING TO A TRAILER FIRST
------------------------
    ros2 launch mpc_landing aruco_landing.launch.py trailer:=true

adds the two-node trailer link and turns the mission's CRUISE phase on, so it
flies to the trailer's radioed coordinate before it starts looking:

    trailer_gps_node    900 MHz MAVLink radio -> /trailer/fix
    trailer_target_node /trailer/fix + the vehicle's own fix -> /trailer/target_local

`cruise_to_trailer` is set HERE rather than defaulted in the node, so the two can
never disagree: the phase is on exactly when the nodes that feed it are running.
Without the argument this is the same marker-only mission it has always been.

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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mission = LaunchConfiguration('mission')
    trailer = LaunchConfiguration('trailer')
    radio = LaunchConfiguration('trailer_radio')
    trailer_sync = LaunchConfiguration('trailer_input_sync_s')
    return LaunchDescription([
        # Set mission:=false to start ONLY the perception stack, and drive the
        # mission node separately with `ros2 run` so its ENTER-to-approve prompt
        # works (ros2 launch does not forward stdin to a child). run_px4 does
        # exactly this.
        DeclareLaunchArgument('mission', default_value='true'),
        # Set trailer:=true to fly to the trailer's radioed coordinate before
        # searching. Off by default: no radio, no cruise.
        DeclareLaunchArgument('trailer', default_value='false'),
        # The radio's serial port on the Jetson, and its baud. These are the two
        # things that change between vehicles, so they are arguments; everything
        # else about the trailer nodes lives in their own `_declare`.
        DeclareLaunchArgument('trailer_device', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('trailer_baud', default_value='57600'),
        # Set trailer_radio:=false to keep the coordinate pipeline but NOT start
        # the radio reader, for when trailer_gps_node is already running in its
        # own terminal. Two processes cannot share one serial port, and the one
        # that loses is silent about it in a log file nobody is watching — so
        # this exists to make "I am driving the radio myself" a supported setup
        # rather than a race.
        DeclareLaunchArgument('trailer_radio', default_value='true'),
        # Standalone launch keeps zero for compatibility. run_px4 trailer passes
        # the same non-zero local/global pairing tolerance as its mission node.
        DeclareLaunchArgument('trailer_input_sync_s', default_value='0.0'),
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

        # --- the trailer's coordinate (trailer:=true) -----------------------
        # The radio end: proves the link and republishes the trailer's
        # GLOBAL_POSITION_INT as /trailer/fix.
        Node(
            package='trailer_link',
            executable='trailer_gps_node',
            name='trailer_gps_node',
            output='screen',
            parameters=[{
                'serial_device': LaunchConfiguration('trailer_device'),
                'baud': ParameterValue(LaunchConfiguration('trailer_baud'),
                                       value_type=int),
            }],
            # Both must be true: the mission wants the trailer, AND nobody
            # else is already holding the radio.
            condition=IfCondition(PythonExpression(
                ["'", trailer, "' == 'true' and '", radio, "' == 'true'"])),
        ),
        # ...and the geodesy end: /trailer/fix -> a point in the vehicle's own
        # local ENU frame. Publishes ONLY while the target is fully valid.
        Node(
            package='trailer_link',
            executable='trailer_target_node',
            name='trailer_target_node',
            output='screen',
            parameters=[{
                'input_sync_tolerance_s': ParameterValue(
                    trailer_sync, value_type=float),
            }],
            condition=IfCondition(trailer),
        ),

        # --- the gated ArUco landing mission (mission:=false omits it) ------
        Node(
            package='mpc_landing',
            executable='aruco_landing_node',
            name='aruco_landing_node',
            output='screen',
            emulate_tty=True,
            # value_type=bool is not optional: a launch substitution arrives as
            # the STRING 'true', and the node declared this parameter as a bool,
            # so without the cast the node dies on a type mismatch.
            parameters=[{'cruise_to_trailer': ParameterValue(trailer,
                                                             value_type=bool)}],
            condition=IfCondition(mission),
        ),
    ])
