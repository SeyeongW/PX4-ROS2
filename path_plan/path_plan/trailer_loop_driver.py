#!/usr/bin/env python3
"""Drive the city trailer around a perimeter square loop and broadcast its pose.

This is the live-sim counterpart of ``tools/pursuit_sim.py``'s analytic ``Trailer``:
instead of evaluating the loop position offline, it drives the REAL Gazebo trailer
(``trailer_aruco``, spawned as ``trailer`` in the city world) around the same square
loop and publishes the trailer's ACTUAL pose so the pursuit bridge and the landing
node can chase it.

It talks to Gazebo Transport directly (no MAVROS / extra build): it streams a world
-frame velocity command to the trailer's ``VelocityControl`` plugin and reads the
trailer's actual pose back from the world ``dynamic_pose/info`` feed. The trailer has
no yaw, so a world-ENU velocity maps straight onto the (body-frame) Twist — the same
assumption ``precision_landing.moving_marker_node`` makes for its flat platform.

The velocity command is the loop's own tangent velocity (feed-forward, from the
analytic loop derivative) plus a proportional pull back onto the loop, so the trailer
tracks the square cleanly, including the corners, and recovers from its spawn offset.

Loop geometry (half-size, speed, start corner, direction) comes from the SAME scenario
file the offline sim uses — ``gazebo/maps/city_uav_trailer_loop.yaml`` — so both stay
in sync. Individual fields can still be overridden by ROS parameters.

Topic / service contract:
  out (ROS)  /trailer/position   geometry_msgs/PointStamped   ENU cue (x=E, y=N)
  out (ROS)  /trailer/velocity   geometry_msgs/Vector3Stamped ENU velocity (vE, vN)
  pub (gz)   /model/trailer/cmd_vel   gz.msgs.Twist            world-ENU drive command
  sub (gz)   /world/<world>/dynamic_pose/info   gz.msgs.Pose_V  actual trailer pose
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node

# gz-transport is optional: without it we still publish the analytic cue, we just
# can't physically drive the Gazebo trailer (publish-only / external-driver mode).
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.twist_pb2 import Twist as GzTwist
    from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV
    _GZ_OK = True
except Exception as _e:        # pragma: no cover - depends on host gz install
    _GZ_OK = False
    _GZ_ERR = _e

# x-half-size units of arc length to each corner of the CCW square (matches
# tools/pursuit_sim.py so the live loop is identical to the offline one).
_CORNER_S = {"SE": 0.0, "NE": 2.0, "NW": 4.0, "SW": 6.0}

# Default scenario file (same one tools/pursuit_sim.py reads). Resolved relative to
# the repo so it works from the source tree; a ROS param can override the path.
_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_SCENARIO = _REPO / "gazebo/maps/city_uav_trailer_loop.yaml"


def square_loop_pos(s: float, half: float) -> np.ndarray:
    """Position at perimeter arc-length ``s`` on a CCW square of half-size ``half``.

    Identical to ``tools/pursuit_sim.py.square_loop_pos`` so the driven loop matches
    the offline pursuit sim exactly.
    """
    p = s % (8.0 * half)
    if p < 2 * half:
        return np.array([half, -half + p])
    if p < 4 * half:
        return np.array([half - (p - 2 * half), half])
    if p < 6 * half:
        return np.array([-half, half - (p - 4 * half)])
    return np.array([-half + (p - 6 * half), -half])


class TrailerLoopDriver(Node):
    def __init__(self):
        super().__init__("trailer_loop_driver")
        p = self.declare_parameter

        self.world = p("world", "applepark_city_uav").value
        self.model = p("model", "trailer").value
        self.cmd_vel_topic = p("cmd_vel_topic", "/model/trailer/cmd_vel").value
        self.pos_topic = p("position_topic", "/trailer/position").value
        self.vel_topic = p("velocity_topic", "/trailer/velocity").value
        self.frame_id = p("frame_id", "map").value
        self.rate_hz = float(p("rate_hz", 20.0).value)
        # Proportional pull back onto the analytic loop point (1/s). The command is
        # loop_velocity_ff + track_kp * (loop_point - actual), so a spawn offset or
        # drift is corrected without overshoot.
        self.track_kp = float(p("track_kp", 0.6).value)
        # Cap so the correction never dwarfs the cruise speed.
        self.vel_max = float(p("vel_max", 12.0).value)

        # Loop geometry: defaults from the scenario YAML, overridable per field.
        scenario_path = p("scenario", str(_DEFAULT_SCENARIO)).value
        tr = self._load_scenario(scenario_path)
        self.half = float(p("half_size_m", tr.get("half_size_m", 600.0)).value)
        self.speed = float(p("speed_m_s", tr.get("speed_m_s", 6.0)).value)
        direction = str(p("direction", tr.get("direction", "ccw")).value).lower()
        self.sign = 1.0 if direction == "ccw" else -1.0
        start_corner = str(p("start_corner", tr.get("start_corner", "SW")).value).upper()
        self.s0 = _CORNER_S.get(start_corner, 6.0) * self.half

        # --- State ----------------------------------------------------------
        self._actual = None                 # latest actual (E, N) from gz
        self._prev_target = None            # (E, N) analytic target last tick (for v_ff)
        self.t0 = self.get_clock().now()

        # --- ROS I/O --------------------------------------------------------
        self.pos_pub = self.create_publisher(PointStamped, self.pos_topic, 10)
        self.vel_pub = self.create_publisher(Vector3Stamped, self.vel_topic, 10)

        # --- gz transport ---------------------------------------------------
        self.gz = None
        self.cmd_vel_pub = None
        if _GZ_OK:
            self.gz = GzNode()
            self.cmd_vel_pub = self.gz.advertise(self.cmd_vel_topic, GzTwist)
            self.gz.subscribe(GzPoseV, f"/world/{self.world}/dynamic_pose/info",
                              self._gz_pose_cb)
            self.get_logger().info(
                f'driving "{self.model}" via {self.cmd_vel_topic}; '
                f'pose from /world/{self.world}/dynamic_pose/info')
        else:
            self.get_logger().warn(
                f"gz Python bindings unavailable ({_GZ_ERR}); publishing the analytic "
                "cue only (trailer will NOT move).")

        self.get_logger().info(
            f"trailer loop: half={self.half:.0f} m speed={self.speed:.1f} m/s "
            f"start={start_corner} dir={direction} -> {self.pos_topic}")
        self.create_timer(1.0 / self.rate_hz, self.tick)

    # -----------------------------------------------------------------------
    def _load_scenario(self, path: str) -> dict:
        try:
            doc = yaml.safe_load(Path(path).read_text())
            return dict(doc.get("trailer", {}) or {})
        except Exception as exc:             # noqa: BLE001
            self.get_logger().warn(
                f"scenario '{path}' unreadable ({exc}); using built-in loop defaults")
            return {}

    def _loop_point(self, t: float) -> np.ndarray:
        """Analytic loop target position at time ``t`` (map ENU)."""
        return square_loop_pos(self.s0 + self.sign * self.speed * t, self.half)

    def tick(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9
        stamp = now.to_msg()

        target = self._loop_point(t)
        dt = 1.0 / self.rate_hz
        # Feed-forward loop velocity from the analytic derivative (robust across the
        # 90 deg corners, where a plain tangent would be undefined).
        if self._prev_target is not None:
            v_ff = (target - self._prev_target) / dt
        else:
            v_ff = np.zeros(2)
        self._prev_target = target

        # Command = loop feed-forward + proportional pull onto the loop point.
        actual = np.asarray(self._actual, float) if self._actual is not None else target
        cmd = v_ff + self.track_kp * (target - actual)
        speed = float(np.hypot(cmd[0], cmd[1]))
        if speed > self.vel_max:
            cmd = cmd * (self.vel_max / speed)

        if self.cmd_vel_pub is not None:
            twist = GzTwist()
            twist.linear.x = float(cmd[0])   # world ENU == body (trailer has no yaw)
            twist.linear.y = float(cmd[1])
            twist.linear.z = 0.0
            self.cmd_vel_pub.publish(twist)

        # Publish the trailer's ACTUAL pose (falls back to the analytic target until
        # the first gz pose arrives) so the pursuit chases where it really is.
        cue = actual
        pos = PointStamped()
        pos.header.stamp = stamp
        pos.header.frame_id = self.frame_id
        pos.point.x = float(cue[0])          # ENU East  (== Gazebo world x)
        pos.point.y = float(cue[1])          # ENU North (== Gazebo world y)
        pos.point.z = 0.0
        self.pos_pub.publish(pos)

        # Velocity cue = the commanded loop velocity (smooth, what the trailer tracks).
        vel = Vector3Stamped()
        vel.header.stamp = stamp
        vel.header.frame_id = self.frame_id
        vel.vector.x = float(v_ff[0])
        vel.vector.y = float(v_ff[1])
        self.vel_pub.publish(vel)

    def _gz_pose_cb(self, msg):
        for pose in msg.pose:
            if pose.name == self.model:
                self._actual = (pose.position.x, pose.position.y)
                return


def main(args=None):
    rclpy.init(args=args)
    node = TrailerLoopDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
