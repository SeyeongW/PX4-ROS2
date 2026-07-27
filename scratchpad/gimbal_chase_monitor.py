"""Score a gimbal chase run against the body-fixed baseline in docs/nodes.md 2.6.

That section measured the failure with three numbers, so this measures the same
three and nothing else:

    * detection rate while chasing (body-fixed: 0.3% at 3 m/s, 26% overall)
    * airframe tilt (body-fixed: >10 deg for 41 s, peak 109 deg)
    * mission state churn (body-fixed: 15-16 DESCEND<->APPROACH hunts)

Plus one thing 2.6 could not know about: the PX4/Gazebo yaw offset.  The cue
comes from `trailer_cue_node`, which publishes gz world coordinates as if they
were PX4-local, so that offset lands directly on the cue the gimbal aims at.
It starts near 27 deg on the ground and GPS-aided yaw pulls it toward zero in
flight, so its value DURING the chase decides whether the run means anything.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String

from gz.msgs10.imu_pb2 import IMU as GzIMU
from gz.transport13 import Node as GzNode
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

WORLD = 'precision_landing_200m_moving'
MODEL = 'x500_gimbal_rgbd_lidar_0'
RUN_S = 180.0


def _qos():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class ChaseMonitor(Node):
    def __init__(self):
        super().__init__('gimbal_chase_monitor')
        self.det = self.frames = 0
        self.tilts = []
        self.aim = []
        self.deltas = []
        self.states = []
        self.hunts = 0
        self.alt = []
        self.pitch = None
        self.bins = {}
        self.t0 = time.time()

        self.gz = GzNode()
        self.q_base = None
        self.gz.subscribe(
            GzIMU,
            f'/world/{WORLD}/model/{MODEL}/link/base_link/sensor/imu_sensor/imu',
            self._on_gz_imu)

        self.create_subscription(Bool, '/aruco/detected', self._on_det, 10)
        self.create_subscription(JointState, '/gimbal/joint_state',
                                 self._on_joints, 10)
        self.create_subscription(Float32, '/gimbal/aim_error_deg',
                                 lambda m: self.aim.append(float(m.data)), 10)
        self.create_subscription(String, '/mission/state', self._on_state, 10)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                 self._on_att, _qos())
        self.create_subscription(VehicleLocalPosition,
                                 '/fmu/out/vehicle_local_position_v1',
                                 self._on_pos, _qos())

    def _on_gz_imu(self, m):
        self.q_base = (m.orientation.w, m.orientation.x, m.orientation.y,
                       m.orientation.z)

    def _on_joints(self, m):
        if len(m.position) >= 3:
            self.pitch = math.degrees(float(m.position[2]))

    def _on_det(self, m):
        self.frames += 1
        self.det += 1 if m.data else 0
        # Bin by how steeply the lens looks down.  A marker lying flat on the
        # ground foreshortens as cos(incidence), so if detection collapses at
        # shallow pitch the limit is the TARGET's geometry, not the pointing.
        if self.pitch is not None:
            key = int(max(-90.0, min(0.0, self.pitch)) // 15) * 15
            hit, tot = self.bins.get(key, (0, 0))
            self.bins[key] = (hit + (1 if m.data else 0), tot + 1)

    def _on_state(self, m):
        s = str(m.data).split()[0]
        if not self.states or self.states[-1] != s:
            # a hunt is specifically falling BACK from DESCEND to APPROACH
            if self.states and self.states[-1].startswith('DESCEND') \
                    and s.startswith(('APPROACH', 'ABORT')):
                self.hunts += 1
            self.states.append(s)
            self.get_logger().info(f'state -> {s}  (t={time.time()-self.t0:.0f}s)')

    def _on_att(self, m):
        w, x, y, z = (float(v) for v in m.q)
        # angle between the body-down axis and world-down = total tilt
        self.tilts.append(math.degrees(
            math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))))

    def _on_pos(self, m):
        self.alt.append(float(-m.z))
        if self.q_base is None:
            return
        w, x, y, z = self.q_base
        gz_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.deltas.append(math.degrees(m.heading - (math.pi / 2 - gz_yaw)))


def main():
    rclpy.init()
    node = ChaseMonitor()
    try:
        while rclpy.ok() and time.time() - node.t0 < RUN_S:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    tilts = np.array(node.tilts) if node.tilts else np.zeros(1)
    dt = 1.0 / max(len(tilts) / max(time.time() - node.t0, 1e-9), 1e-9)
    print('\n============== gimbal chase run ==============')
    print(f'duration          : {time.time() - node.t0:.0f} s')
    print(f'detection rate    : {100.0 * node.det / max(node.frames, 1):.1f}%  '
          f'({node.det}/{node.frames})     [body-fixed baseline: 0.3%]')
    print(f'tilt              : mean {tilts.mean():.1f} deg  peak '
          f'{tilts.max():.1f} deg   >10 deg for {float((tilts > 10).sum()) * dt:.0f} s'
          '     [baseline: peak 109 deg, 41 s]')
    if node.aim:
        a = np.array(node.aim)
        print(f'gimbal aim error  : mean {a.mean():.2f} deg  p95 '
              f'{np.percentile(a, 95):.2f} deg  max {a.max():.2f} deg')
    if node.deltas:
        d = np.array(node.deltas)
        print(f'PX4/gz yaw offset : mean {d.mean():+.2f} deg  '
              f'last {d[-1]:+.2f} deg  (cue error = this x range)')
    print(f'altitude          : max {max(node.alt or [0]):.1f} m')
    print(f'DESCEND exits     : {node.hunts}            [baseline: 15-16]')
    if node.bins:
        print('detection vs gimbal pitch (nadir = -90):')
        for key in sorted(node.bins):
            hit, tot = node.bins[key]
            bar = '#' * int(40.0 * hit / max(tot, 1))
            print(f'  {key:>4}..{key+15:>4} deg : {100.0*hit/max(tot,1):5.1f}%  '
                  f'(n={tot:4d}) {bar}')
    print(f'states            : {" -> ".join(node.states) or "(none)"}')
    print('==============================================')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
