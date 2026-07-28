"""Everything needed to land on a FIXED marker, in one launch.

    ros2 launch mpc_landing fixed_marker_landing.launch.py

Starts, in this order:

    usb_cam            /dev/video0 -> /down_camera/image + camera_info
    aruco_pose_node    -> /perception/down/marker_pose
    siyi_gimbal_node   points the camera down the moment the vehicle arms
    mpc_landing_node   the gated mission

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

ABOUT THE PARAMETERS BELOW
--------------------------
Our own nodes are started with NO parameters, as everywhere in flight/ — their
values live in each node's `_declare()` with the reasoning beside them.

`usb_cam` is the exception, and deliberately so: it is a third-party driver we
do not own, so there is no `_declare()` to put its device path, pixel format or
calibration file in. For it, this launch IS the source of truth. The three that
matter:

  video_device     which camera
  camera_info_url  the CALIBRATION. Without it solvePnP has no focal length and
                   every range is wrong — see aruco_landing/config.
  image_width/height  MUST match the resolution the calibration was taken at
                   (1280x720). Change the resolution and fx/fy/cx/cy all scale,
                   so the calibration silently stops applying.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    calib = os.path.join(
        get_package_share_directory('aruco_landing'), 'config', 'down_camera.yaml')

    return LaunchDescription([
        # --- camera -------------------------------------------------------
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='down_camera',
            output='screen',
            parameters=[{
                'video_device': '/dev/video0',
                # MJPG because YUYV at this resolution drops to 5 fps on this
                # camera, and the marker has to be seen while descending.
                'pixel_format': 'mjpeg2rgb',
                'image_width': 1280,
                'image_height': 720,
                'framerate': 30.0,
                'camera_name': 'down_camera',
                'camera_info_url': f'file://{calib}',
                'frame_id': 'down_camera_optical_frame',
                # JPEG quality for /down_camera/image/compressed. 30 is enough
                # to see framing and exposure over a field link while costing a
                # fraction of the raw 2.7 MB/frame — this is the picture you
                # watch, not the one the detector measures from.
                'image_raw.compressed.jpeg_quality': 30,
            }],
            remappings=[
                ('image_raw', '/down_camera/image'),
                ('camera_info', '/down_camera/camera_info'),
                # image_transport's sub-topics do NOT follow the base remap —
                # they are separate publishers created under the ORIGINAL name.
                # Without these three the compressed stream lands on
                # /image_raw/compressed, which is both confusing and impossible
                # to find when you are outside looking for the camera view.
                ('image_raw/compressed', '/down_camera/image/compressed'),
                ('image_raw/compressedDepth',
                 '/down_camera/image/compressedDepth'),
                ('image_raw/theora', '/down_camera/image/theora'),
            ],
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

        # --- the gated mission ---------------------------------------------
        Node(
            package='mpc_landing',
            executable='mpc_landing_node',
            name='mpc_landing_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
