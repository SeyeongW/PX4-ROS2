"""Terminal client for typed autonomy mission commands."""

from __future__ import annotations

import argparse
from typing import Sequence

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


SERVICE_BY_COMMAND = {
    "land": "/autonomy/land_on_trailer",
    "emergency-land": "/autonomy/emergency_land",
}


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Command the PX4 city autonomy coordinator."
    )
    parser.add_argument(
        "command",
        choices=tuple(SERVICE_BY_COMMAND),
        help="land returns to the moving trailer; emergency-land lands in place",
    )
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parsed, ros_args = parser.parse_known_args(args)
    if parsed.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")

    rclpy.init(args=ros_args)
    node = Node("mission_cli")
    service_name = SERVICE_BY_COMMAND[parsed.command]
    client = node.create_client(Trigger, service_name)
    try:
        if not client.wait_for_service(timeout_sec=parsed.timeout_s):
            raise SystemExit(f"ERROR: service unavailable: {service_name}")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=parsed.timeout_s)
        response = future.result()
        if response is None:
            raise SystemExit(f"ERROR: command timed out: {parsed.command}")
        print(response.message)
        if not response.success:
            raise SystemExit(2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
