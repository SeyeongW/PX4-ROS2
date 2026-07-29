"""Start the SIYI RTSP -> ROS image bridge.

PARAMETER-FREE, like every launch file in this repo. The stream URL, the jitter
buffer, the decoder override and the calibration path are all declared in
`rtsp_bridge_node._declare()` beside the comment explaining each one.

    ros2 launch rtsp_bridge rtsp_bridge.launch.py

Check it before trusting it:

    ros2 topic hz /gimbal_camera/image          # should be the camera's frame rate
    ros2 topic echo /rtsp_bridge_node/status    # resolution, decoder, calibration
    ros2 run rqt_image_view rqt_image_view /gimbal_camera/image

If nothing arrives, prove the stream exists independently of ROS first:

    gst-launch-1.0 rtspsrc location=rtsp://192.168.144.25:8554/video1 ! fakesink
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rtsp_bridge',
            executable='rtsp_bridge_node',
            name='rtsp_bridge_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
