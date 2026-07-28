"""Start the SIYI A8 mini nadir-on-arm node.

PARAMETER-FREE, deliberately — the same rule precland_hw follows. Every tunable
(gimbal IP and port, the nadir angles, the re-assert period) is declared in
`siyi_gimbal_node._declare()` next to the comment explaining it, so the source
is the answer to "what is it actually commanding" and no launch override can
quietly disagree with it.

Override for a one-off without editing anything:

    ros2 run siyi_gimbal siyi_gimbal_node --ros-args -p gimbal_host:=192.168.1.25

This does not start MAVROS; it expects /mavros/state to already exist, which
keeps it usable on the bench against a fake publisher.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='siyi_gimbal',
            executable='siyi_gimbal_node',
            name='siyi_gimbal_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
