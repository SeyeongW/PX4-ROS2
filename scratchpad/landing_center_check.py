"""Score a headless landing run against GROUND TRUTH: did it land on the deck CENTRE?

Everything the mission itself reports is measured against its own target — the
cue plus a learned marker correction — so it will happily report perfect
alignment while sitting a metre off the real deck.  This script never reads the
mission's opinion of where the deck is.  It reads gz `dynamic_pose/info` for
both the trailer and the vehicle, which is truth by construction, and asks one
question: at touchdown, how far is the vehicle from the deck centre (0,0 of
`moving_platform_aruco_velocity`, which is where all three markers say the
landing point is)?

    deck top   = trailer model z + 2.05   (link pose z=2 + half of the 0.1 box)
    xy error   = |vehicle_xy - trailer_xy|
    height     = vehicle z - deck top

Reports the error at minimum height and at the end, the trajectory of the last
few metres, and — because a run that never gets there is the common outcome —
the phase timeline with the height and error at every transition.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode

WORLD = 'precision_landing_200m_moving'
DECK_TOP_IN_MODEL = 2.05        # link pose z=2 + 0.1 m platform box / 2
RUN_S = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0



class LandingCheck(Node):
    def __init__(self, model):
        super().__init__('landing_center_check')
        self.model = model
        self.trailer = None
        self.drone = None
        self.samples = []          # (t, h, xy_err, state)
        self.state = '(none)'
        self.timeline = []         # (t, state, h, xy_err)
        self.det = self.frames = 0

        self.gz = GzNode()
        self.gz.subscribe(Pose_V, f'/world/{WORLD}/dynamic_pose/info', self._on_pose)
        self.create_subscription(String, '/mission/state', self._on_state, 10)
        self.create_subscription(Bool, '/aruco/detected', self._on_det, 10)
        self.t0 = time.time()
        self.create_timer(0.1, self._tick)

    def _on_det(self, m):
        self.frames += 1
        self.det += 1 if m.data else 0

    def _on_pose(self, msg):
        for p in msg.pose:
            if p.name == 'trailer':
                self.trailer = np.array([p.position.x, p.position.y, p.position.z])
            elif p.name == self.model:
                self.drone = np.array([p.position.x, p.position.y, p.position.z])

    def _on_state(self, m):
        s = str(m.data)
        if s != self.state:
            self.state = s
            h, e = self._geom()
            self.timeline.append((time.time() - self.t0, s, h, e))
            self.get_logger().info(f't={time.time()-self.t0:6.1f}s  -> {s:9s}'
                                   f'  h={h:6.2f} m  xy_err={e:5.2f} m')

    def _geom(self):
        if self.trailer is None or self.drone is None:
            return float('nan'), float('nan')
        h = self.drone[2] - (self.trailer[2] + DECK_TOP_IN_MODEL)
        e = float(math.hypot(self.drone[0] - self.trailer[0],
                             self.drone[1] - self.trailer[1]))
        return h, e

    def _tick(self):
        h, e = self._geom()
        if not math.isnan(h):
            self.samples.append((time.time() - self.t0, h, e, self.state))


def main():
    rclpy.init()
    # PX4 names the vehicle after the airframe; try the gimbal one first.
    node = LandingCheck('x500_gimbal_rgbd_lidar_0')
    deadline = time.time() + RUN_S
    try:
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.drone is None and time.time() - node.t0 > 20:
                node.model = 'x500_city_rgbd_lidar_0'      # baseline airframe
            if node.state == 'DONE' and time.time() - node.t0 > 30:
                break
    except KeyboardInterrupt:
        pass

    s = node.samples
    print('\n============ landing centre check (ground truth) ============')
    if not s:
        print('NO DATA — never saw both the trailer and the vehicle in '
              f'/world/{WORLD}/dynamic_pose/info')
    else:
        arr = np.array([(t, h, e) for t, h, e, _ in s])
        i_min = int(np.argmin(arr[:, 1]))
        print(f'duration        : {arr[-1,0]:.0f} s   samples {len(s)}')
        print(f'detection rate  : {100.0*node.det/max(node.frames,1):.1f}%  '
              f'({node.det}/{node.frames})')
        print(f'min height      : {arr[i_min,1]:+.3f} m above the deck '
              f'at t={arr[i_min,0]:.0f}s')
        print(f'  xy error there: {arr[i_min,2]:.3f} m from the deck CENTRE')
        print(f'final           : h={arr[-1,1]:+.3f} m  xy_err={arr[-1,2]:.3f} m'
              f'  state={s[-1][3]}')
        # The error the mission actually committed to, and where it ended up
        # after riding the moving deck for a few more seconds.
        t_td = next((t for t, st, _, _ in node.timeline
                     if st.startswith('TOUCHDOWN')), None)
        if t_td is not None:
            j = int(np.argmin(np.abs(arr[:, 0] - t_td)))
            print(f'AT TOUCHDOWN    : t={t_td:.0f}s  h={arr[j,1]:+.3f} m'
                  f'  xy_err={arr[j,2]:.3f} m')
        else:
            print('AT TOUCHDOWN    : never reached')
        print('  (arm state is NOT read here: this PX4 publishes a '
              'VehicleStatus one field wider than the installed px4_msgs, so '
              'every message is dropped. Check the sim log for '
              '"Disarmed by external command".)')
        # The endgame only — samples from the FINAL descent, one per half metre.
        # Scanning the whole run instead put the climb-out through the same
        # half-metre buckets first and the table then described the TAKEOFF.
        t_desc = next((t for t, st, _, _ in node.timeline
                       if st.startswith('DESCEND')), None)
        print('\n  h (m)   xy_err (m)   state        t (s)')
        seen = set()
        for t, h, e, st in s:
            if t_desc is not None and t < t_desc:
                continue
            key = round(h * 2) / 2
            if key in seen:
                continue
            seen.add(key)
            print(f'  {h:6.2f}   {e:8.3f}     {st:12s} {t:6.1f}')
    print('\n  timeline:')
    for t, st, h, e in node.timeline:
        print(f'    {t:6.1f}s  {st:12s} h={h:6.2f}  xy_err={e:5.2f}')
    print('============================================================')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
