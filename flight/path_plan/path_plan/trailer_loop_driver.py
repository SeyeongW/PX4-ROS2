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

import getpass
import os
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from visualization_msgs.msg import Marker

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

        # Preferred motion: a building-CLEAR waypoint circuit (the analytic square
        # crosses buildings and physically jams the Gazebo trailer). Fall back to the
        # square only if no waypoints are provided (e.g. the offline pursuit_sim).
        self._wp = None
        wps = tr.get("loop_waypoints_enu")
        if wps and len(wps) >= 3:
            self._build_waypoint_loop(np.asarray(wps, float))
            self.get_logger().info(
                f"following building-clear waypoint loop: {len(self._wp)} pts, "
                f"perimeter {self._perim:.0f} m")
        else:
            self.get_logger().info("no loop_waypoints_enu -> analytic square loop")

        # --- State ----------------------------------------------------------
        self._actual = None                 # latest actual (E, N) from gz
        self._prev_target = None            # (E, N) analytic target last tick (for v_ff)
        self.t0 = self.get_clock().now()

        # --- ROS I/O --------------------------------------------------------
        self.pos_pub = self.create_publisher(PointStamped, self.pos_topic, 10)
        self.vel_pub = self.create_publisher(Vector3Stamped, self.vel_topic, 10)
        # RViz marker for the live trailer position. Sized to be visible on the
        # ~1300 m city map (a raw point is invisible at that zoom).
        self.marker_pub = self.create_publisher(Marker, "/trailer/marker", 10)
        self.marker_size_m = float(p("marker_size_m", 25.0).value)

        # --- gz transport ---------------------------------------------------
        # run_px4_map.sh starts gz sim inside GZ_PARTITION=px4_ros2_<user>. This
        # node runs in a separate `ros2 launch` shell that normally has no such
        # env var, so its gz-transport Node would land on the DEFAULT partition
        # and silently fail to reach the trailer's VelocityControl plugin (the
        # trailer never moves). Match run_px4_map.sh's partition before creating
        # the gz Node -- honour an explicitly-set GZ_PARTITION, else rebuild the
        # same default. Overridable with the `gz_partition` ROS param.
        default_partition = os.environ.get(
            "GZ_PARTITION", f"px4_ros2_{getpass.getuser()}")
        partition = str(p("gz_partition", default_partition).value)
        if partition:
            os.environ["GZ_PARTITION"] = partition
            self.get_logger().info(f"gz transport partition: {partition}")
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

    def _build_waypoint_loop(self, wp: np.ndarray):
        """Precompute the closed piecewise-linear circuit for arc-length lookup."""
        self._wp = wp
        closed = np.vstack([wp, wp[:1]])                 # wrap back to start
        seglen = np.hypot(*(np.diff(closed, axis=0).T))
        self._cum = np.concatenate([[0.0], np.cumsum(seglen)])   # (N+1,)
        self._perim = float(self._cum[-1])

    def _loop_point(self, t: float) -> np.ndarray:
        """Loop target position at time ``t`` (map ENU), constant ground speed."""
        if self._wp is None:
            return square_loop_pos(self.s0 + self.sign * self.speed * t, self.half)
        # Arc-length distance travelled along the closed circuit (spawn == wp[0]).
        d = (self.sign * self.speed * t) % self._perim
        i = int(np.searchsorted(self._cum, d, side="right") - 1)
        i = min(max(i, 0), len(self._wp) - 1)
        seg_d = d - self._cum[i]
        a = self._wp[i]
        b = self._wp[(i + 1) % len(self._wp)]
        seg_len = self._cum[i + 1] - self._cum[i]
        frac = seg_d / seg_len if seg_len > 1e-6 else 0.0
        return a + (b - a) * frac

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

        # Bright marker at the live trailer position (visible on the big map).
        mk = Marker()
        mk.header.stamp = stamp
        mk.header.frame_id = self.frame_id
        mk.ns = "trailer"
        mk.id = 0
        mk.type = Marker.CUBE
        mk.action = Marker.ADD
        mk.pose.position.x = float(cue[0])
        mk.pose.position.y = float(cue[1])
        mk.pose.position.z = 0.5 * self.marker_size_m
        mk.pose.orientation.w = 1.0
        mk.scale.x = self.marker_size_m
        mk.scale.y = self.marker_size_m
        mk.scale.z = self.marker_size_m
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.55, 0.0, 0.9
        self.marker_pub.publish(mk)

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

    def stop_trailer(self):
        """Zero the trailer velocity. The gz VelocityControl plugin HOLDS the last
        command, so if this driver stops/crashes without this the trailer keeps
        coasting at its last velocity straight off the map. Send a few zeros to be
        sure the plugin latches the stop."""
        if self.cmd_vel_pub is None:
            return
        try:
            for _ in range(5):
                self.cmd_vel_pub.publish(GzTwist())   # all-zero twist
                time.sleep(0.02)
        except Exception:            # never let cleanup raise
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TrailerLoopDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_trailer()          # don't leave the trailer coasting off the map
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
