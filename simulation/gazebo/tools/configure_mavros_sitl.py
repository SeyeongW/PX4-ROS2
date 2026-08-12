#!/usr/bin/env python3
"""Verify and configure MAVROS without depending on the ROS 2 CLI daemon."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter


CLOCK_NODES = (
    "/mavros/mavros",
    "/mavros/time",
    "/mavros/local_position",
    "/mavros/imu",
    "/mavros/sys",
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=_positive_float, default=10.0)
    parser.add_argument("--heartbeat-type", default="GCS")
    parser.add_argument("--heartbeat-rate", type=_positive_float)
    return parser.parse_args()


def _wait_for_service(client, deadline: float, service_name: str) -> None:
    while time.monotonic() < deadline:
        if client.wait_for_service(timeout_sec=0.2):
            return
    raise RuntimeError(f"parameter service unavailable: {service_name}")


def _call(node: Node, client, request, deadline: float):
    while time.monotonic() < deadline:
        future = client.call_async(request)
        remaining = max(0.0, deadline - time.monotonic())
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=min(2.0, remaining))
        if future.done() and future.result() is not None:
            return future.result()
        future.cancel()
    raise RuntimeError(f"parameter request timed out: {client.srv_name}")


def _require_sim_time(node: Node, remote_node: str, deadline: float) -> None:
    service = f"{remote_node}/get_parameters"
    client = node.create_client(GetParameters, service)
    _wait_for_service(client, deadline, service)
    response = _call(
        node,
        client,
        GetParameters.Request(names=["use_sim_time"]),
        deadline,
    )
    if len(response.values) != 1:
        raise RuntimeError(f"invalid use_sim_time response: {remote_node}")
    value = response.values[0]
    if value.type != ParameterType.PARAMETER_BOOL or not value.bool_value:
        raise RuntimeError(
            f"MAVROS component is outside Gazebo time: {remote_node}"
        )


def _set_heartbeat(
    node: Node,
    heartbeat_type: str,
    heartbeat_rate: float | None,
    deadline: float,
) -> None:
    service = "/mavros/sys/set_parameters"
    client = node.create_client(SetParameters, service)
    _wait_for_service(client, deadline, service)
    parameters = [
        Parameter(
            "heartbeat_mav_type",
            Parameter.Type.STRING,
            heartbeat_type,
        ).to_parameter_msg()
    ]
    if heartbeat_rate is not None:
        parameters.append(
            Parameter(
                "heartbeat_rate",
                Parameter.Type.DOUBLE,
                heartbeat_rate,
            ).to_parameter_msg()
        )
    response = _call(
        node,
        client,
        SetParameters.Request(parameters=parameters),
        deadline,
    )
    if len(response.results) != len(parameters):
        raise RuntimeError("invalid MAVROS heartbeat parameter response")
    failures = [
        result.reason or "rejected"
        for result in response.results
        if not result.successful
    ]
    if failures:
        raise RuntimeError(
            "MAVROS heartbeat parameter rejected: " + "; ".join(failures)
        )


def main() -> int:
    arguments = _arguments()
    heartbeat_type = str(arguments.heartbeat_type).strip()
    if not heartbeat_type:
        raise SystemExit("heartbeat type must not be empty")
    deadline = time.monotonic() + float(arguments.timeout)
    rclpy.init(args=[])
    node = rclpy.create_node("mavros_sitl_contract_probe")
    try:
        for remote_node in CLOCK_NODES:
            _require_sim_time(node, remote_node, deadline)
        _set_heartbeat(
            node,
            heartbeat_type,
            arguments.heartbeat_rate,
            deadline,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    rate = (
        "default"
        if arguments.heartbeat_rate is None
        else f"{arguments.heartbeat_rate:g}"
    )
    print("MAVROS clock     : Gazebo /clock on all plugin components")
    print(f"MAVROS heartbeat : {heartbeat_type} @ {rate} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
