"""Start the gated hardware precision landing node.

DELIBERATELY PARAMETER-FREE. There is no `parameters=[...]`, no YAML, and no
launch argument that reaches the node — every tunable lives in
`precland_hw_node._declare()` with its value beside the comment explaining it.

That is a rule, not an oversight. Split tuning between a node and a launch file
and the source stops being the answer to "what is this actually flying with":
the file you read says 5 m, the vehicle flies 3 m, and nothing is wrong with
either file. One place to look, one place to change.

For a one-off experiment, override on the command line without editing anything:

    ros2 run precland_hw precland_hw_node --ros-args -p takeoff_alt_m:=3.0

This launch does not start MAVROS or the marker detector; it expects both to be
running already, which keeps it usable against a bench setup as well as a
vehicle.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='precland_hw',
            executable='precland_hw_node',
            name='precland_hw_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
