#!/usr/bin/env python3
"""Moving ArUco marker — Gazebo mover + ENU coordinate broadcaster.

Drives the ground ArUco marker along a trajectory inside Gazebo and, on the
SAME tick, publishes the marker's position as a local-ENU PointStamped. This
is the "cue" the precision_landing controller flies toward BEFORE the down
camera can see the marker; once vision acquires it, the controller hands off
to its visual servo and lands.

Geometry note: the drone spawns at the Gazebo world origin and MAVROS sets its
local-position (EKF) origin there too, so for this world the local ENU frame
coincides with the Gazebo world XY plane (E = world x, N = world y). The marker
position is therefore published unchanged as the ENU cue.

Moving the model uses gz-transport's world `set_pose` service (teleport — no
physics needed for a static ground decal). If the gz Python bindings are not
importable the node degrades to publish-only (useful for the "static marker,
virtual cue" case, or replaying a real external publisher).

Topic / service contract:
  out (ROS)  <marker_topic>                 geometry_msgs/PointStamped  ENU cue
  call (gz)  /world/<world>/set_pose        gz.msgs.Pose -> gz.msgs.Boolean

Trajectory patterns (param `pattern`):
  static : stay at (center_e, center_n)
  line   : e = center_e + amplitude·sin(w t),  n = center_n
  circle : e = center_e + amplitude·cos(w t),  n = center_n + amplitude·sin(w t)
  where w = speed / amplitude (rad/s) so `speed` is the path speed in m/s.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, Vector3Stamped

# gz-transport is optional: without it we still publish the ENU cue, we just
# can't physically move the Gazebo model (publish-only / external-mover mode).
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_pb2 import Pose as GzPose
    from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
    _GZ_OK = True
except Exception as _e:        # pragma: no cover - depends on host gz install
    _GZ_OK = False
    _GZ_ERR = _e


class MovingMarkerNode(Node):
    def __init__(self):
        super().__init__('moving_marker_node')

        # --- Parameters -----------------------------------------------------
        self.world = self.declare_parameter('world', 'iris_down_camera_runway').value
        self.model = self.declare_parameter('model', 'aruco_marker_0').value
        self.marker_topic = self.declare_parameter('marker_topic', '/marker/position').value
        self.vel_topic = self.declare_parameter('vel_topic', '/marker/velocity').value
        self.rate = self.declare_parameter('rate', 50.0).value          # Hz (higher
        #   keeps the teleport step small at high speed: step = speed / rate)
        self.pattern = self.declare_parameter('pattern', 'line').value  # static|line|circle
        self.center_e = self.declare_parameter('center_e', 1.0).value   # m, ENU East
        self.center_n = self.declare_parameter('center_n', 0.0).value   # m, ENU North
        self.amplitude = self.declare_parameter('amplitude', 1.5).value  # m
        self.speed = self.declare_parameter('speed', 0.3).value         # m/s path speed
        self.z = self.declare_parameter('z', 0.002).value               # m, ground height
        self.frame_id = self.declare_parameter('frame_id', 'map').value
        # move_model=false → publish the cue only, leave the Gazebo model alone.
        self.move_model = self.declare_parameter('move_model', True).value

        # angular rate so |velocity| == speed along the path
        self.w = (self.speed / self.amplitude) if self.amplitude > 1e-6 else 0.0

        self.pub = self.create_publisher(PointStamped, self.marker_topic, 10)
        # Publish the marker's VELOCITY explicitly (ENU vE, vN) so the controller
        # feeds it forward directly instead of finite-differencing the position
        # (no lag/noise → tighter velocity match → the drone stays level and
        # lands cleanly on the moving platform). Derived from the node's own
        # successive positions, so it is exact for any pattern and never hard-codes
        # a speed.
        self.vel_pub = self.create_publisher(Vector3Stamped, self.vel_topic, 10)
        self._prev = None        # (e, n, t) for velocity differencing

        # --- gz mover -------------------------------------------------------
        self.gz = None
        self.set_pose_srv = f'/world/{self.world}/set_pose'
        if self.move_model:
            if _GZ_OK:
                self.gz = GzNode()
                self.get_logger().info(
                    f'moving model "{self.model}" via {self.set_pose_srv}')
            else:
                self.get_logger().warn(
                    f'gz Python bindings unavailable ({_GZ_ERR}); '
                    'publishing cue only (model will NOT move).')

        self.get_logger().info(
            f'pattern={self.pattern} center=({self.center_e:.2f},{self.center_n:.2f}) '
            f'amp={self.amplitude:.2f} speed={self.speed:.2f} -> {self.marker_topic}')

        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / self.rate, self.tick)

    # -----------------------------------------------------------------------
    def _position(self):
        """Marker (E, N) at the current time for the configured pattern."""
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        if self.pattern == 'static':
            return self.center_e, self.center_n
        if self.pattern == 'circle':
            return (self.center_e + self.amplitude * math.cos(self.w * t),
                    self.center_n + self.amplitude * math.sin(self.w * t))
        # default: line — CONSTANT-SPEED straight runs along East (a triangle wave,
        # not a sin: the velocity is constant in magnitude so the drone's velocity
        # feed-forward matches it exactly and it lands cleanly). The platform runs
        # straight out to +amplitude, reverses, runs straight to −amplitude and
        # back — one continuous straight leg is 2·amplitude metres long.
        if self.amplitude < 1e-6:
            return self.center_e, self.center_n
        A = self.amplitude
        phase = (self.speed * t) % (4.0 * A)     # distance into the back-and-forth
        if phase < A:
            d = phase                            # centre → +A
        elif phase < 3.0 * A:
            d = 2.0 * A - phase                  # +A → −A  (the long 2A straight leg)
        else:
            d = phase - 4.0 * A                  # −A → centre
        return self.center_e + d, self.center_n

    def tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()
        e, n = self._position()

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.point.x = e        # ENU East  (== Gazebo world x)
        msg.point.y = n        # ENU North (== Gazebo world y)
        msg.point.z = 0.0
        self.pub.publish(msg)

        # Velocity = backward difference of our own positions (exact, low noise at
        # this rate). Held at the last value across the brief reversal tick.
        vel = Vector3Stamped()
        vel.header.stamp = stamp
        vel.header.frame_id = self.frame_id
        if self._prev is not None:
            pe, pn, pt = self._prev
            dt = (now - pt).nanoseconds * 1e-9
            if dt > 1e-4:
                vel.vector.x = (e - pe) / dt
                vel.vector.y = (n - pn) / dt
        self.vel_pub.publish(vel)
        self._prev = (e, n, now)

        if self.gz is not None:
            self._set_model_pose(e, n)

    def _set_model_pose(self, e, n):
        req = GzPose()
        req.name = self.model
        req.position.x = float(e)
        req.position.y = float(n)
        req.position.z = float(self.z)
        req.orientation.w = 1.0
        # Blocking request with a short timeout; set_pose returns quickly.
        ok, _rep = self.gz.request(self.set_pose_srv, req, GzPose, GzBoolean, 100)
        if not ok:
            # Throttle: don't spam if the world/service isn't up yet.
            self.get_logger().warn(
                f'set_pose to {self.set_pose_srv} failed (is Gazebo running?)',
                throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = MovingMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
