"""trailer_cue_node — ONE job: publish the target's own reported position.

This is the LONG-RANGE target source.  A camera cannot find a 1.3 m marker from
90 m away (it subtends ~1 px per ArUco cell; the resolution limit is ~30 m), so
the approach must be flown to coordinates the target itself reports.  On a real
vehicle that is the truck's GPS/telemetry link; in SITL we stand in for it by
reading the trailer pose out of Gazebo.

Subscribes
    (gz) /world/<world>/dynamic_pose/info
Publishes
    /marker/cue           geometry_msgs/PointStamped    target position, local ENU
    /marker/cue_velocity  geometry_msgs/Vector3Stamped  target velocity, local ENU

Frame: gz reports world ENU. Local x/y subtract the drone spawn (the PX4 local
XY origin); z is the configured deck height in PX4 local ENU. Velocity is a
least-squares slope over a short window, never a two-point difference.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V

from .frame import LOCAL_ENU_FRAME_ID
from .parameter_utils import (
    DEFAULT_DECK_Z_M,
    require_finite,
    require_nonempty,
    require_positive,
)


class TrailerCueNode(Node):
    def __init__(self):
        super().__init__('trailer_cue_node')
        p = self.declare_parameter
        self.world = require_nonempty(
            'world', p('world', 'mpc_landing_200m_moving').value)
        self.entity = require_nonempty(
            'trailer_entity', p('trailer_entity', 'trailer').value)
        self.spawn = np.array(p('spawn_enu', [30.0, 30.0]).value, float)
        if self.spawn.shape != (2,) or not np.all(np.isfinite(self.spawn)):
            raise ValueError(
                f'spawn_enu must contain two finite values, got {self.spawn}')
        self.deck_z = require_finite(
            'deck_z', p('deck_z', DEFAULT_DECK_Z_M).value)
        self.rate = require_positive('rate_hz', p('rate_hz', 10.0).value)
        self.vel_window = require_positive(
            'vel_window_s', p('vel_window_s', 0.3).value)

        self._lock = threading.Lock()
        self._hist = deque()
        self._pos = None
        self._vel = np.zeros(2)
        self._updated = False
        self._source_ns = None
        self._last_report_t = -float('inf')

        self.pos_pub = self.create_publisher(PointStamped, '/marker/cue', 10)
        self.vel_pub = self.create_publisher(Vector3Stamped, '/marker/cue_velocity', 10)
        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f'/world/{self.world}/dynamic_pose/info', self._on_gz)
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(
            f'trailer_cue_node: "{self.entity}" in {self.world} -> /marker/cue '
            f'(spawn offset {self.spawn.tolist()})')

    def _on_gz(self, msg):
        for pose in msg.pose:
            if pose.name != self.entity:
                continue
            loc = np.array([pose.position.x - self.spawn[0],
                            pose.position.y - self.spawn[1]])
            source_ns = (int(msg.header.stamp.sec) * 1_000_000_000
                         + int(msg.header.stamp.nsec))
            if source_ns <= 0:
                source_ns = self.get_clock().now().nanoseconds
            t = source_ns * 1e-9
            with self._lock:
                self._hist.append((t, loc))
                while len(self._hist) > 2 and t - self._hist[0][0] > self.vel_window:
                    self._hist.popleft()
                if len(self._hist) >= 3:
                    ts = np.array([s[0] for s in self._hist])
                    ps = np.array([s[1] for s in self._hist])
                    dt_ = ts - ts.mean()
                    den = float(dt_ @ dt_)
                    if den > 1e-6:
                        self._vel = (dt_ @ (ps - ps.mean(axis=0))) / den
                self._pos = loc
                self._source_ns = source_ns
                self._updated = True
            return

    def _tick(self):
        with self._lock:
            if not self._updated:
                return
            pos = None if self._pos is None else self._pos.copy()
            vel = self._vel.copy()
            source_ns = self._source_ns
            self._updated = False
        if pos is None or source_ns is None:
            return
        stamp_s, stamp_ns = divmod(int(source_ns), 1_000_000_000)
        pm = PointStamped()
        pm.header.stamp.sec = stamp_s
        pm.header.stamp.nanosec = stamp_ns
        pm.header.frame_id = LOCAL_ENU_FRAME_ID
        pm.point.x, pm.point.y, pm.point.z = float(pos[0]), float(pos[1]), self.deck_z
        self.pos_pub.publish(pm)
        vm = Vector3Stamped()
        vm.header.stamp.sec = stamp_s
        vm.header.stamp.nanosec = stamp_ns
        vm.header.frame_id = LOCAL_ENU_FRAME_ID
        vm.vector.x, vm.vector.y, vm.vector.z = float(vel[0]), float(vel[1]), 0.0
        self.vel_pub.publish(vm)
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self._last_report_t >= 1.0:
            self._last_report_t = now_s
            self.get_logger().info(
                f'cue frame={LOCAL_ENU_FRAME_ID} '
                f'p=({pos[0]:.3f},{pos[1]:.3f},{self.deck_z:.3f}) m '
                f'v=({vel[0]:.3f},{vel[1]:.3f},0.000) m/s '
                f'publish_delay={max(0.0, now_s - source_ns * 1e-9):.3f} s')


def main(args=None):
    rclpy.init(args=args)
    node = TrailerCueNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
