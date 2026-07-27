#!/usr/bin/env python3
"""Start and observe one fixed Relative-OSQP moving-trailer SITL mission."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Sequence

import rclpy
from mavros_msgs.msg import ExtendedState
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


EXPECTED_IDENTITY = {
    "control_stack": "px4_mavros_relative_osqp",
    "landing_controller_type": "relative_landing_mpc",
    "landing_mpc_solver_effective": "osqp",
    "validation_mode": False,
    "production_qualified": True,
    "activation_qualified": True,
    "uses_sim_ground_truth_control_input": True,
}

DISPLAY_PHASES = {
    "waiting_for_px4_and_sensors": "WAITING",
    "ready_inactive": "READY",
    "offboard_setpoint_prestream": "TAKEOFF",
    "arming_and_entering_offboard": "TAKEOFF",
    "vertical_takeoff": "TAKEOFF",
    "stationary_trailer_approach": "TRAILER_APPROACH",
    "marker_track_down": "MARKER_SEARCH",
    "precision_align": "TRACK_AND_LAND",
    "precision_descent": "TRACK_AND_LAND",
    "final_approach": "TRACK_AND_LAND",
    "touchdown_confirm": "TOUCHDOWN_CONFIRM",
    "landed": "COMPLETE",
    "abort_climb": "ABORT",
    "failsafe_hold": "ABORT",
}


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the fixed Relative-OSQP SITL stack, issue exactly one "
            "mission start request, and print compact mission status."
        )
    )
    parser.add_argument("--ready-timeout", type=_positive_float, default=60.0)
    parser.add_argument(
        "--mission-timeout", type=_positive_float, default=240.0
    )
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def identity_errors(
    payload: dict[str, Any], expected_run_id: str
) -> list[str]:
    """Return fixed production-stack identity mismatches."""
    errors: list[str] = []
    for name, expected in EXPECTED_IDENTITY.items():
        actual = payload.get(name)
        if isinstance(expected, bool):
            matches = type(actual) is bool and actual is expected
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            errors.append(f"{name}={actual!r}")
    if payload.get("baseline_run_id") != expected_run_id:
        errors.append(f"baseline_run_id={payload.get('baseline_run_id')!r}")
    return errors


def readiness_errors(
    payload: dict[str, Any], expected_run_id: str = ""
) -> list[str]:
    """Return exact reasons a status sample is not safe to start."""
    errors = identity_errors(payload, expected_run_id)
    if payload.get("phase") != "ready_inactive":
        errors.append(f"phase={payload.get('phase')!r}")
    if payload.get("connected") is not True:
        errors.append("connected!=true")
    if payload.get("armed") is not False:
        errors.append("armed!=false")
    return errors


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compact_status(payload: dict[str, Any]) -> str:
    """Return the intentionally small operator-facing diagnostic line."""
    phase = str(payload.get("phase", "unknown"))
    display_phase = DISPLAY_PHASES.get(phase, phase.upper())
    target_valid = payload.get("target_estimate_valid") is True
    xy_error = _finite_or_none(payload.get("landing_horizontal_error_m"))
    relative_speed = _finite_or_none(
        payload.get("landing_relative_horizontal_speed_m_s")
    )
    mpc_status = str(payload.get("landing_mpc", "not_started"))
    reason = payload.get("last_safety_failure_reason")
    xy_text = "-" if xy_error is None else f"{xy_error:.3f}m"
    speed_text = (
        "-" if relative_speed is None else f"{relative_speed:.3f}m/s"
    )
    return (
        f"[{display_phase}] target={str(target_valid).lower()} "
        f"xy={xy_text} relative_speed={speed_text} "
        f"mpc={mpc_status} reason={reason or '-'}"
    )


class MissionOnce(Node):
    """Observe status and own the one permitted start service request."""

    def __init__(self, expected_run_id: str) -> None:
        super().__init__("precision_landing_once")
        self.expected_run_id = expected_run_id
        self.latest_status: dict[str, Any] | None = None
        self.latest_extended_state: ExtendedState | None = None
        self.status_sequence = 0
        self.status_parse_errors = 0
        self.create_subscription(
            String, "/autonomy/status", self._on_status, 10
        )
        self.create_subscription(
            ExtendedState,
            "/mavros/extended_state",
            self._on_extended_state,
            10,
        )
        self.start_client = self.create_client(
            Trigger, "/autonomy/start_precision_landing"
        )
        self.hold_client = self.create_client(
            Trigger, "/autonomy/hold_precision_landing"
        )

    def _on_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            self.status_parse_errors += 1
            return
        if not isinstance(payload, dict):
            self.status_parse_errors += 1
            return
        self.latest_status = payload
        self.status_sequence += 1

    def _on_extended_state(self, message: ExtendedState) -> None:
        self.latest_extended_state = message

    def _on_ground(self) -> bool:
        state = self.latest_extended_state
        return bool(
            state is not None
            and state.landed_state
            == ExtendedState.LANDED_STATE_ON_GROUND
        )

    def wait_until_ready(self, timeout_s: float) -> dict[str, Any]:
        """Wait for the exact unarmed fixed-profile READY contract."""
        deadline = time.monotonic() + timeout_s
        last_errors = ["no status received"]
        last_sequence = -1
        consecutive_ready = 0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_status is None:
                continue
            last_errors = readiness_errors(
                self.latest_status, self.expected_run_id
            )
            if not self._on_ground():
                last_errors.append("PX4 is not ON_GROUND")
            if self.status_sequence == last_sequence:
                continue
            last_sequence = self.status_sequence
            if last_errors:
                consecutive_ready = 0
            else:
                consecutive_ready += 1
            if consecutive_ready >= 2:
                return self.latest_status
        raise TimeoutError(
            "controller did not become safely ready: " + ", ".join(last_errors)
        )

    def request_start(self, timeout_s: float = 10.0) -> str:
        """Call Trigger once and require its application-level success flag."""
        if not self.start_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError("start service did not become available")
        future = self.start_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done():
            raise TimeoutError("start service response timed out")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"start service failed: {exception}")
        response = future.result()
        if response is None or not response.success:
            message = (
                "empty response" if response is None else response.message
            )
            raise RuntimeError(f"start request rejected: {message}")
        return str(response.message)

    def request_hold(self, timeout_s: float = 5.0) -> str:
        """Request and verify the controller's fail-closed position hold."""
        if not self.hold_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError("hold service did not become available")
        future = self.hold_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done():
            raise TimeoutError("hold service response timed out")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"hold service failed: {exception}")
        response = future.result()
        if response is None or not response.success:
            message = (
                "empty response" if response is None else response.message
            )
            raise RuntimeError(f"hold request rejected: {message}")
        return str(response.message)

    def _hold_after_failure(self, reason: str) -> None:
        try:
            message = self.request_hold()
            print(f"SAFE HOLD REQUESTED: {message} ({reason})", flush=True)
        except (RuntimeError, TimeoutError) as exc:
            print(f"SAFE HOLD UNCONFIRMED: {exc} ({reason})", flush=True)

    def monitor(self, timeout_s: float) -> int:
        """Print compact status until confirmed completion or safe abort."""
        deadline = time.monotonic() + timeout_s
        last_phase = ""
        next_periodic = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            status = self.latest_status
            if status is None:
                continue
            phase = str(status.get("phase", ""))
            mismatches = identity_errors(status, self.expected_run_id)
            if mismatches or status.get("connected") is not True:
                reason = ", ".join(mismatches or ["PX4 disconnected"])
                self._hold_after_failure(reason)
                print(
                    f"MISSION ABORTED: controller identity ({reason})",
                    flush=True,
                )
                return 2
            now = time.monotonic()
            if phase != last_phase or now >= next_periodic:
                print(compact_status(status), flush=True)
                last_phase = phase
                next_periodic = now + 2.0
            if (
                phase == "landed"
                and status.get("armed") is False
                and status.get("px4_landed") is True
            ):
                print("MISSION COMPLETE: landed and disarmed", flush=True)
                return 0
            if phase == "failsafe_hold":
                reason = status.get("last_safety_failure_reason") or "unknown"
                print(f"MISSION ABORTED: safe hold ({reason})", flush=True)
                return 2
        self._hold_after_failure("mission timeout")
        print("MISSION TIMEOUT: controller remains in safe hold", flush=True)
        return 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    rclpy.init(args=[])
    node = MissionOnce(args.run_id)
    try:
        node.wait_until_ready(args.ready_timeout)
        print("READY: PX4, sensors and Relative OSQP controller", flush=True)
        message = node.request_start()
        print(f"STARTED: {message}", flush=True)
        return node.monitor(args.mission_timeout)
    except (RuntimeError, TimeoutError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
