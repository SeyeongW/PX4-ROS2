"""approve / abort — one short command to release a mission gate.

`ros2 launch` does not forward stdin to the nodes it starts (measured:
`sys.stdin.isatty()` is False under launch), so the mission node's own
ENTER-to-approve prompt only works when it is run directly with `ros2 run`.
When the whole stack is brought up by one launch file, approvals have to arrive
over the service instead — and

    ros2 service call /mpc_landing_node/approve std_srvs/srv/Trigger

is a lot to type correctly while standing next to an armed vehicle. So:

    ros2 run mpc_landing approve      # release the current gate
    ros2 run mpc_landing abort        # stop, land, disarm

Both target `mpc_landing_node` by default. Pass a node name to talk to another
node that offers the same `~/approve` / `~/abort` services (e.g. the naive
takeoff/land node):

    ros2 run mpc_landing approve naive_flight_node
    ros2 run mpc_landing abort   naive_flight_node

Both print what the node said back, so you see WHICH step you just authorised
rather than just "success: True". `approve` also refuses politely when no gate
is pending, which is the node's answer, not this tool's.
"""

from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def _target_node() -> str:
    """The node to talk to: the first non-ROS CLI argument, or the default.

    `ros2 run mpc_landing approve naive_flight_node` passes the name through as
    argv; anything starting with '-' (e.g. an injected --ros-args) is skipped so
    a bare `approve` still hits mpc_landing_node.
    """
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            return arg
    return 'mpc_landing_node'


def _call(service: str, timeout: float = 5.0) -> int:
    node_name = _target_node()
    rclpy.init()
    node = Node(f'{node_name}_{service}_cli')
    client = node.create_client(Trigger, f'/{node_name}/{service}')
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            print(f'  no /{node_name}/{service} — is the node running?',
                  file=sys.stderr)
            return 2
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        result = future.result()
        if result is None:
            print('  no reply from the mission node', file=sys.stderr)
            return 3
        mark = 'OK' if result.success else 'REFUSED'
        print(f'  {mark}: {result.message}')
        return 0 if result.success else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def approve() -> None:
    sys.exit(_call('approve'))


def abort() -> None:
    sys.exit(_call('abort'))


if __name__ == '__main__':
    approve()
