"""Real-flight bringup: the ROS side of a gated precision landing.

Starts the two things in this repo that are ready to fly:

    siyi_gimbal_node   points the camera down as soon as the vehicle arms
    precland_hw_node   the gated mission: takeoff -> search -> MPC descent

PARAMETER-FREE, like every launch file under flight/. Each node declares its own
values in `_declare()`, so the source is the answer to what the vehicle is
flying and nothing here can quietly disagree with it.

    ros2 launch precland_hw flight_bringup.launch.py

TWO THINGS THIS DOES NOT START, ON PURPOSE
------------------------------------------
**MAVROS.** It needs an `fcu_url` that depends on how the companion is wired to
the autopilot, and guessing it here would be a parameter in a launch file by
another name. Start it yourself first:

    ros2 launch mavros apm.launch fcu_url:=/dev/ttyTHS1:921600

**The camera and the marker detector.** There is no camera driver in this repo
yet — nothing converts the A8 mini's RTSP stream (rtsp://192.168.144.25:8554/
video1) into a ROS image topic, and `flight/aruco_landing`'s `aruco_pose_node`
still defaults to the SIMULATOR's `/down_camera/image`. Until that gap is
closed, `precland_hw_node` will pass every preflight check except the marker
pipeline one, which is exactly the behaviour you want: it refuses to arm rather
than taking off with no way to see the marker.

So the full real-flight chain is currently:

    [ MAVROS ]  +  [ camera driver -> aruco_pose_node ]  +  this launch
       ready              MISSING                            ready
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Started first so the camera is already looking down by the time the
        # mission node's gates are released.
        Node(
            package='siyi_gimbal',
            executable='siyi_gimbal_node',
            name='siyi_gimbal_node',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='precland_hw',
            executable='precland_hw_node',
            name='precland_hw_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
