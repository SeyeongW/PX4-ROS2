#!/usr/bin/env python3
"""Drive the seo-branch moving platform through map-defined ENU waypoints.

This talks directly to Gazebo Transport, so it needs neither MAVROS nor a ROS
workspace build.  The platform is kinematic rather than a wheeled vehicle.
The mountain route is therefore kept on the exact-flat central corridor; a
heightmap safeguard raises the horizontal body before any off-route terrain
can penetrate its 5x5 m footprint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
from PIL import Image
import yaml

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


GAZEBO = Path(__file__).resolve().parent
REPO = GAZEBO.parent
STOP = threading.Event()


def quaternion_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny, cosy)


class PoseReceiver:
    def __init__(self, entity_name: str):
        self.entity_name = entity_name
        self.lock = threading.Lock()
        self.value: tuple[float, float, float, float, float, float] | None = None

    def callback(self, message: Pose_V) -> None:
        for pose in message.pose:
            if pose.name != self.entity_name:
                continue
            orientation = pose.orientation
            roll, pitch, yaw = quaternion_rpy(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            with self.lock:
                self.value = (
                    pose.position.x, pose.position.y, pose.position.z,
                    roll, pitch, yaw,
                )
            return

    def get(self) -> tuple[float, float, float, float, float, float] | None:
        with self.lock:
            return self.value


class Terrain:
    def __init__(self, document: dict[str, object]):
        terrain = document["terrain"]
        assert isinstance(terrain, dict)
        self.flat = document["map"]["name"] == "city"  # type: ignore[index]
        self.pixels: np.ndarray | None = None
        if not self.flat:
            image_path = REPO / str(terrain["image"])
            self.pixels = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
            self.half_size = 150.0
            self.height_scale = 40.0

    def height(self, x: float, y: float) -> float:
        if self.flat:
            return 0.0
        assert self.pixels is not None
        last = self.pixels.shape[0] - 1
        column = min(max((x + self.half_size) / (2.0 * self.half_size) * last, 0.0), last)
        row = min(max((self.half_size - y) / (2.0 * self.half_size) * last, 0.0), last)
        c0, r0 = int(math.floor(column)), int(math.floor(row))
        c1, r1 = min(c0 + 1, last), min(r0 + 1, last)
        tx, ty = column - c0, row - r0
        value = (
            (1.0 - tx) * (1.0 - ty) * self.pixels[r0, c0]
            + tx * (1.0 - ty) * self.pixels[r0, c1]
            + (1.0 - tx) * ty * self.pixels[r1, c0]
            + tx * ty * self.pixels[r1, c1]
        )
        return float(value / 255.0 * self.height_scale)

    def safe_root_z(self, x: float, y: float) -> float:
        # The body bottom is model-root + 0.05 m.  Sampling a 7x7, 6x6 m
        # envelope gives 0.5 m look-ahead around the physical 5x5 footprint.
        highest = max(
            self.height(x + dx, y + dy)
            for dx in np.linspace(-3.0, 3.0, 7)
            for dy in np.linspace(-3.0, 3.0, 7)
        )
        return max(0.0, highest + 0.08 - 0.05)


def publish_velocity(publisher: object, vx: float, vy: float, vz: float) -> None:
    message = Twist()
    message.linear.x = vx
    message.linear.y = vy
    message.linear.z = vz
    message.angular.x = 0.0
    message.angular.y = 0.0
    message.angular.z = 0.0
    publisher.publish(message)  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("map", choices=("city", "mountain"))
    parser.add_argument("--loops", type=int, default=0,
                        help="route repetitions; 0 repeats until interrupted")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="wall-clock timeout in seconds; 0 disables")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.0,
                        help="override the YAML cruise speed; 0 uses YAML")
    parser.add_argument("--start-index", type=int, default=0,
                        help="resume a validation route at this waypoint index")
    parser.add_argument("--route", choices=("flat", "slope"), default="flat",
                        help="flat operational route or mountain slope safeguard test")
    parser.add_argument("--no-terrain-follow", action="store_true")
    args = parser.parse_args()
    if args.loops < 0:
        parser.error("--loops must be non-negative")
    if args.timeout < 0.0:
        parser.error("--timeout must be non-negative")
    if args.rate <= 0.0:
        parser.error("--rate must be positive")

    coordinate_path = GAZEBO / "maps" / f"{args.map}_coordinates.yaml"
    document = yaml.safe_load(coordinate_path.read_text(encoding="utf-8"))
    trailer = document["trailer"]
    entity = str(trailer["entity_name"])
    topic = str(trailer["command_topic"])
    pose_topic = str(trailer["pose_topic"])
    route_key = (
        "slope_validation_waypoints_enu_m"
        if args.route == "slope" else "waypoints_enu_m"
    )
    if route_key not in trailer:
        parser.error(f"the {args.map} map does not define a {args.route} route")
    waypoints = [(float(point[0]), float(point[1])) for point in trailer[route_key]]
    if not 0 <= args.start_index < len(waypoints):
        parser.error(f"--start-index must be in 0..{len(waypoints) - 1}")
    speed = args.speed if args.speed > 0.0 else float(trailer["cruise_speed_m_s"])
    tolerance = float(trailer["waypoint_tolerance_m"])
    terrain = Terrain(document)

    node = Node()
    publisher = node.advertise(topic, Twist)
    receiver = PoseReceiver(entity)
    node.subscribe(Pose_V, pose_topic, receiver.callback)
    for handled in (signal.SIGINT, signal.SIGTERM):
        signal.signal(handled, lambda *_: STOP.set())

    print(f"trailer_map={args.map} route={args.route} entity={entity} waypoints={len(waypoints)}")
    print(f"command_topic={topic} pose_topic={pose_topic}")
    start = time.monotonic()
    while receiver.get() is None and not STOP.is_set():
        if time.monotonic() - start > 15.0:
            print(f"ERROR: no pose received for {entity}", file=sys.stderr)
            return 2
        time.sleep(0.05)
    if receiver.get() is None:
        print("TRAILER WAYPOINT VALIDATION: INTERRUPTED", file=sys.stderr)
        return 130

    index = args.start_index
    completed_loops = 0
    reached = 0
    previous_velocity = np.zeros(3, dtype=np.float64)
    previous_time = time.monotonic()
    max_abs_z_error = max_abs_roll = max_abs_pitch = 0.0
    tracking_error_samples = 0
    excessive_tilt_samples = 0
    period = 1.0 / args.rate

    try:
        while not STOP.is_set():
            now = time.monotonic()
            if args.timeout > 0.0 and now - start >= args.timeout:
                print("ERROR: trailer route timed out", file=sys.stderr)
                return 3
            pose = receiver.get()
            if pose is None:
                time.sleep(period)
                continue
            x, y, z, roll, pitch, _yaw = pose
            target_x, target_y = waypoints[index]
            dx, dy = target_x - x, target_y - y
            distance = math.hypot(dx, dy)
            if distance <= tolerance:
                reached += 1
                print(f"waypoint_reached={index} x={x:.3f} y={y:.3f} z={z:.3f}")
                index += 1
                if index == len(waypoints):
                    completed_loops += 1
                    if args.loops > 0 and completed_loops >= args.loops:
                        break
                    index = 0
                target_x, target_y = waypoints[index]
                dx, dy = target_x - x, target_y - y
                distance = math.hypot(dx, dy)

            desired_xy_speed = min(speed, 1.25 * distance)
            if distance > 1e-9:
                desired_vx = desired_xy_speed * dx / distance
                desired_vy = desired_xy_speed * dy / distance
            else:
                desired_vx = desired_vy = 0.0
            expected_z = (
                terrain.safe_root_z(x, y)
                if not args.no_terrain_follow else float(trailer["spawn_pose_enu"]["z"])
            )
            desired_vz = max(-2.5, min(2.5, 5.0 * (expected_z - z)))

            dt = max(now - previous_time, 1e-3)
            desired = np.array([desired_vx, desired_vy, desired_vz], dtype=np.float64)
            max_delta = np.array([1.0, 1.0, 3.0], dtype=np.float64) * dt
            delta = np.clip(desired - previous_velocity, -max_delta, max_delta)
            command = previous_velocity + delta
            publish_velocity(publisher, float(command[0]), float(command[1]), float(command[2]))
            previous_velocity = command
            previous_time = now

            z_error = abs(z - expected_z)
            max_abs_z_error = max(max_abs_z_error, z_error)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            if z_error > 0.10:
                tracking_error_samples += 1
            if abs(roll) > math.radians(3.0) or abs(pitch) > math.radians(3.0):
                excessive_tilt_samples += 1
            time.sleep(max(0.0, period - (time.monotonic() - now)))
    finally:
        for _ in range(5):
            publish_velocity(publisher, 0.0, 0.0, 0.0)
            time.sleep(0.02)

    route_complete = args.loops > 0 and completed_loops >= args.loops
    if STOP.is_set() and not route_complete:
        print("TRAILER WAYPOINT VALIDATION: INTERRUPTED", file=sys.stderr)
        print(f"completed_loops={completed_loops} reached_waypoints={reached}")
        return 130

    z_limit = 0.40 if args.route == "slope" else 0.10
    stable = (
        max_abs_z_error <= z_limit
        and max_abs_roll <= math.radians(3.0)
        and max_abs_pitch <= math.radians(3.0)
        and excessive_tilt_samples == 0
    )
    print(f"TRAILER WAYPOINT VALIDATION: {'PASS' if stable else 'FAIL'}")
    print(f"completed_loops={completed_loops} reached_waypoints={reached}")
    print(f"max_abs_z_error_m={max_abs_z_error:.6f}")
    print(f"max_abs_roll_deg={math.degrees(max_abs_roll):.6f}")
    print(f"max_abs_pitch_deg={math.degrees(max_abs_pitch):.6f}")
    print(f"terrain_tracking_error_samples={tracking_error_samples}")
    print(f"excessive_tilt_samples={excessive_tilt_samples}")
    print(f"impulse_launch_detected={'NO' if stable else 'YES'}")
    return 0 if stable else 4


if __name__ == "__main__":
    raise SystemExit(main())
