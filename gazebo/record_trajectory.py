#!/usr/bin/env python3
"""Record a REAL flight's (t, e, n, alt) to CSV during an actual sim run, for
gazebo/plot_3d_trajectory.py --log to render as a paper figure (a real
recorded flight, not the reconstructed/illustrative fallback that script uses
when no --log is given).

Run this in its OWN terminal (or as a background job) at the same time as
the mission launch -- same idea as the old "position log for post-mortem"
command in commands.md, just written straight to CSV columns
plot_3d_trajectory.py expects.

Usage (inside the sim container, after `source /opt/ros/humble/setup.bash`;
plain script, not colcon-installed, so run it with python3 directly):
    python3 gazebo/record_trajectory.py --out /tmp/traj.csv
Stop with Ctrl+C once the flight (or the portion you want plotted) is done.
"""
import argparse
import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped


class TrajectoryRecorder(Node):
    def __init__(self, out_path):
        super().__init__('trajectory_recorder')
        self.t0 = None
        self.f = open(out_path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['t', 'e', 'n', 'alt'])
        self.n_rows = 0
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._cb,
            qos_profile_sensor_data)
        self.get_logger().info(f'recording /mavros/local_position/pose -> {out_path}')

    def _cb(self, msg):
        now = self.get_clock().now()
        if self.t0 is None:
            self.t0 = now
        t = (now - self.t0).nanoseconds * 1e-9
        p = msg.pose.position
        self.writer.writerow([f'{t:.3f}', f'{p.x:.3f}', f'{p.y:.3f}', f'{p.z:.3f}'])
        self.n_rows += 1
        if self.n_rows % 100 == 0:
            self.f.flush()
            self.get_logger().info(f'{self.n_rows} rows written (t={t:.1f}s)', throttle_duration_sec=5.0)

    def destroy_node(self):
        self.f.close()
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='/tmp/trajectory.csv')
    args = ap.parse_args()

    rclpy.init()
    node = TrajectoryRecorder(args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'wrote {node.n_rows} rows to {args.out}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
