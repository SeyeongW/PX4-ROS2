"""Does the gimbal actually put the marker in frame, and is the un-projection right?

A bounded flight test, deliberately much smaller than a mission: take off, hover
over the spawn, aim the gimbal at the STATIONARY trailer 10 m away, and check
two independent things at once.

    1. POINTING — with a body-fixed nadir camera the trailer sits at the very
       edge of the cone (r(h) = 0.772 h, so 10 m lateral needs h >= 13 m and
       leaves nothing for tilt).  If the gimbal aims, detection should be high.
    2. UN-PROJECTION — `/marker/measured` comes from the image + joint encoders
       + PX4 attitude.  Truth comes from Gazebo's own pose feed.  Those are two
       completely independent paths, so agreement is a real check.

The cue fed to the gimbal is derived from Gazebo truth, and must be rotated
into PX4's local frame first: the two are offset in yaw because the world's
magnetic_field encodes a declination that PX4's EKF then applies again
(measured 27.3 deg).  The offset is measured live here rather than hard-coded,
since it is an EKF estimate and drifts.

    GZ_PARTITION=... python3 scratchpad/gimbal_hover_test.py
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool

from gz.msgs10.imu_pb2 import IMU as GzIMU
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleCommand, VehicleLocalPosition)

WORLD = 'precision_landing_200m_moving'
MODEL = 'x500_gimbal_rgbd_lidar_0'
TRAILER = 'trailer'
DECK_Z = 1.811   # marker surface (world 2.051) less the x500 frame offset
# Altitude is a command-line knob so the range bias can be probed at two
# heights: a SCALE error (marker size / intrinsics) grows with slant range,
# a fixed offset does not.
HOVER_ALT = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
HOLD_S = 45.0
# 'over' flies to the marker and hovers on top of it; the default holds the
# spawn so the marker is seen obliquely.  Comparing the two separates a
# foreshortening bias from an always-present one.
OVER_MARKER = 'over' in sys.argv


def _qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class HoverTest(Node):
    def __init__(self):
        super().__init__('gimbal_hover_test')
        self.p_d = None
        self.engaged = False
        self.k = 0
        self.t_hold = None
        self.det = 0
        self.frames = 0
        self.errors = []
        self.range_err = []
        self.bearing_err = []
        self.zs = []
        self.alt_seen = float('nan')

        self.gz = GzNode()
        self.truth = {}
        self.q_base = None
        self.gz.subscribe(Pose_V, f'/world/{WORLD}/pose/info', self._on_gz_pose)
        self.gz.subscribe(
            GzIMU,
            f'/world/{WORLD}/model/{MODEL}/link/base_link/sensor/imu_sensor/imu',
            self._on_gz_imu)

        self.sp = self.create_publisher(TrajectorySetpoint,
                                        '/fmu/in/trajectory_setpoint', 10)
        self.ocm = self.create_publisher(OffboardControlMode,
                                         '/fmu/in/offboard_control_mode', 10)
        self.vc = self.create_publisher(VehicleCommand,
                                        '/fmu/in/vehicle_command', 10)
        self.cue = self.create_publisher(PointStamped, '/marker/cue', 10)

        self.create_subscription(VehicleLocalPosition,
                                 '/fmu/out/vehicle_local_position_v1',
                                 self._on_pos, _qos())
        self.create_subscription(PointStamped, '/marker/measured',
                                 self._on_meas, 10)
        self.create_subscription(Bool, '/aruco/detected', self._on_det, 10)
        self.create_timer(0.05, self._tick)

    # ------------------------------------------------------------- truth feed
    def _on_gz_pose(self, msg):
        for p in msg.pose:
            if p.name in (TRAILER, MODEL):
                self.truth[p.name] = np.array(
                    [p.position.x, p.position.y, p.position.z])

    def _on_gz_imu(self, msg):
        self.q_base = (msg.orientation.w, msg.orientation.x,
                       msg.orientation.y, msg.orientation.z)

    def _on_pos(self, m):
        self.p_d = np.array([m.y, m.x, -m.z])
        self.heading = m.heading

    def _on_det(self, m):
        self.frames += 1
        self.det += 1 if m.data else 0

    def _on_meas(self, m):
        """Score a measurement, split into a part that depends on the frame
        offset estimate and a part that does not.

        The RANGE of the drone->marker offset is invariant to any yaw error in
        `delta`, so it isolates the perception chain.  The BEARING carries both
        the perception error and whatever `delta` got wrong, so a large bearing
        residual with a clean range says the frame estimate is the suspect, not
        the camera maths.
        """
        expected = self._marker_in_px4_frame()
        if expected is None or self.p_d is None:
            return
        got = np.array([m.point.x, m.point.y, m.point.z])
        self.errors.append(float(np.linalg.norm((got - expected)[:2])))
        self.zs.append(float(got[2]))

        got_off = (got - self.p_d)[:2]
        exp_off = (expected - self.p_d)[:2]
        self.range_err.append(float(np.linalg.norm(got_off)
                                    - np.linalg.norm(exp_off)))
        self.bearing_err.append(math.degrees(
            math.atan2(got_off[1], got_off[0]) - math.atan2(exp_off[1], exp_off[0])))

    # ------------------------------------------------------- frame conversion
    def _declination_offset(self):
        """PX4 heading minus Gazebo's true heading, measured live."""
        if self.q_base is None or self.p_d is None:
            return None
        gz_from_north = math.pi / 2 - yaw_of(self.q_base)
        return self.heading - gz_from_north

    def _marker_in_px4_frame(self):
        """Trailer marker position expressed in PX4's local ENU."""
        delta = self._declination_offset()
        if delta is None or TRAILER not in self.truth or MODEL not in self.truth:
            return None
        offset_gz = self.truth[TRAILER] - self.truth[MODEL]
        # azimuth grows by delta going gz -> PX4, i.e. rotate the vector by -delta
        c, s = math.cos(-delta), math.sin(-delta)
        offset_px4 = np.array([c * offset_gz[0] - s * offset_gz[1],
                               s * offset_gz[0] + c * offset_gz[1],
                               0.0])
        marker = self.p_d + offset_px4
        marker[2] = DECK_Z
        return marker

    # ------------------------------------------------------------------ flight
    def _cmd(self, command, p1=0.0, p2=0.0):
        c = VehicleCommand()
        c.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        c.command = command
        c.param1, c.param2 = float(p1), float(p2)
        c.target_system = c.target_component = 1
        c.source_system = c.source_component = 1
        c.from_external = True
        self.vc.publish(c)

    def _tick(self):
        m = OffboardControlMode()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.position = True
        self.ocm.publish(m)

        self.k += 1
        if self.p_d is None:
            return
        if not self.engaged and self.k * 0.05 >= 1.5:
            self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.engaged = True
            self.get_logger().info('ARM + OFFBOARD commanded')

        marker = self._marker_in_px4_frame()
        s = TrajectorySetpoint()
        s.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        hold = np.array([0.0, 0.0, HOVER_ALT])
        if OVER_MARKER and marker is not None:
            hold = np.array([marker[0], marker[1], HOVER_ALT])
        s.position = [float(hold[1]), float(hold[0]), -HOVER_ALT]   # ENU -> NED
        s.yaw = float('nan')
        s.yawspeed = float('nan')
        self.sp.publish(s)

        if marker is not None:
            c = PointStamped()
            c.header.stamp = self.get_clock().now().to_msg()
            c.header.frame_id = 'map'
            c.point.x, c.point.y, c.point.z = (float(marker[0]),
                                               float(marker[1]),
                                               float(marker[2]))
            self.cue.publish(c)

        in_place = (marker is None or not OVER_MARKER
                    or float(np.linalg.norm((self.p_d - marker)[:2])) < 1.0)
        if self.p_d[2] >= HOVER_ALT - 0.5 and in_place and self.t_hold is None:
            self.t_hold = time.time()
            self.det = self.frames = 0
            self.errors.clear()
            self.range_err.clear()
            self.bearing_err.clear()
            self.zs.clear()
            self.alt_seen = float(self.p_d[2])
            delta = math.degrees(self._declination_offset())
            self.get_logger().info(
                f'at {self.p_d[2]:.1f} m — measuring. PX4/gz yaw offset = '
                f'{delta:+.2f} deg')


def main():
    rclpy.init()
    node = HoverTest()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.t_hold and time.time() - node.t_hold > HOLD_S:
                break
    except KeyboardInterrupt:
        pass
    rate = 100.0 * node.det / max(node.frames, 1)
    print('\n================ gimbal hover test ================')
    print(f'hover altitude    : {HOVER_ALT:.1f} m commanded, '
          f'{node.alt_seen:.1f} m actual')
    print(f'frames            : {node.frames}')
    print(f'detection rate    : {rate:.1f}%  ({node.det}/{node.frames})')
    if node.errors:
        e = np.array(node.errors)
        print(f'marker_tf vs truth: mean {e.mean():.3f} m  median '
              f'{np.median(e):.3f} m  p95 {np.percentile(e, 95):.3f} m  '
              f'n={len(e)}')
        r = np.array(node.range_err); b = np.array(node.bearing_err)
        print(f'  range residual  : mean {r.mean():+.3f} m  sd {r.std():.3f} m'
              '   (independent of the frame offset)')
        print(f'  bearing residual: mean {b.mean():+.3f} deg  sd {b.std():.3f} '
              'deg  (perception + delta error)')
    else:
        print('marker_tf vs truth: NO MEASUREMENTS')
    if node.zs:
        z = np.array(node.zs)
        print(f'  reported marker z: mean {z.mean():+.3f} m  sd {z.std():.3f} m'
              '   (raw, if use_deck_z=false)')
    print(f'  geometry          : {"OVERHEAD" if OVER_MARKER else "OBLIQUE"}')
    print('===================================================')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
