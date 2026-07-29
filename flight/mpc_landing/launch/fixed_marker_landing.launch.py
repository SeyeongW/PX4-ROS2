"""Everything needed to land on a FIXED marker, in one launch.

    ros2 launch mpc_landing fixed_marker_landing.launch.py

Starts, in this order:

    usb_cam            /dev/video0 -> /down_camera/image + camera_info
    aruco_pose_node    -> /perception/down/marker_pose
    siyi_gimbal_node   points the camera down the moment the vehicle arms
    landing_tf_node    map -> base_link -> gimbal_mount -> camera
    mpc_landing_node   the gated mission

WITHOUT landing_tf_node THE DETECTOR PUBLISHES NOTHING
------------------------------------------------------
aruco_pose_node solves the marker in the camera's optical frame and then asks
tf2 for `map`, because that is the frame mpc_landing_node differences against
the vehicle's local position. Until this launch grew landing_tf_node, nobody on
the aircraft published any part of that chain: MAVROS had the vehicle pose and
siyi_gimbal had the gimbal attitude, but nothing joined them. Every detection
was discarded with `tf_lookup` while the debug view kept drawing a green square
around a marker it was seeing perfectly — which reads, from outside, as a
camera that cannot find the marker.

landing_tf_node needs MAVROS running for the vehicle pose. Its
`gimbal_attitude_reference` is the one setting that cannot be decided from a
desk; `ros2 run siyi_gimbal gimbal_monitor` decides it on the aircraft.

WATCHING THIS FROM THE GROUND STATION PC
----------------------------------------
Nothing to do on this Jetson — .bashrc sources `~/dds_fastdds.sh`, which puts
every terminal on the local FastDDS Discovery Server with the right transport.
On the PC, `source ~/dds_fastdds.sh` and the whole graph appears.

The one thing that makes or breaks the remote view is the DDS profile, in
config/jetson_dds_fastdds.sh: the interface whitelist (so the Jetson advertises
its tailscale address rather than its wifi LAN address, which the PC cannot
reach) and maxMessageSize 1200 (tailscale's MTU is 1280; at the default 65500 a
compressed frame is IP-fragmented ~50 ways and one lost fragment discards the
whole frame). Measured with both in place: the PC sees 18.4 Hz on
/down_camera/image/compressed, which is exactly the rate the Jetson itself
sees — no loss over the link.

MAVROS is NOT started here. Its fcu_url depends on how the companion is wired,
and guessing it would be a parameter in a launch file by another name:

    ros2 launch mavros px4.launch fcu_url:=/dev/ttyACM0:921600

APPROVING STEPS
---------------
`ros2 launch` does not forward stdin to its children — measured, not assumed:
under launch a node sees `sys.stdin.isatty() == False`. So the mission node's
ENTER prompt cannot work here. Approve from a second terminal instead:

    ros2 topic echo /mpc_landing_node/state     # what it is waiting for
    ros2 run mpc_landing approve                # release the gate
    ros2 run mpc_landing abort                  # stop, land, disarm

If you would rather press ENTER, start the mission node by itself instead of
through this launch — it detects the terminal and prompts:

    ros2 run mpc_landing mpc_landing_node

THE CAMERA IS gst_camera_node, NOT usb_cam
------------------------------------------
usb_cam decoded MJPEG on the ARM cores at 21 fps while the camera pushes ~58, so
the V4L2 queue stayed permanently full and every frame handed to the detector was
already 618 ms old. gst_camera_node moves the decode to the Jetson's NVJPG block
and keeps up at 58 fps — see its module docstring for the measurements.

It also removes two topics that had no business being on a flight link.
image_transport loads every plugin it can find, so usb_cam advertised
/down_camera/image/compressedDepth (a DEPTH codec on a colour camera) and
/theora, which nothing in this stack speaks. `disable_pub_plugins` was supposed
to suppress them and did not — the parameter is looked up under the resolved
topic name, not the pre-remap `image_raw` this launch was setting. Rather than
chase that, gst_camera_node creates its two publishers directly and never loads
image_transport at all, so the extra transports cannot come back:

    /down_camera/image             raw BGR, for the detector
    /down_camera/image/compressed  the camera's own MJPEG bytes, for the PC

ABOUT THE PARAMETERS BELOW
--------------------------
There are none. Every node here is ours, so its values live in its own
`_declare()` with the reasoning beside them — the same rule as the rest of
flight/. The usb_cam exception that used to justify a block of camera settings
in this file went away with usb_cam; device path, resolution and calibration are
now in gst_camera_node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # --- camera: hardware JPEG decode on the Jetson's NVJPG block -------
        Node(
            package='aruco_landing',
            executable='gst_camera_node',
            name='down_camera',
            output='screen',
        ),

        # --- perception ---------------------------------------------------
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

        # --- the gated mission ---------------------------------------------
        Node(
            package='mpc_landing',
            executable='mpc_landing_node',
            name='mpc_landing_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
