#!/usr/bin/env python3
"""ROS /clock 10 Hz versus controller timer 20 Hz regression; no Gazebo required.

ROS_DOMAIN_ID=73 python3 src/jol_perception/test/check_pointing_ros.py
Use an isolated ROS domain with the Humble environment and package sourced.
This fixture explicitly uses a turret-mounted camera; it does not validate an arm camera mount.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from jol_perception.pointing_core import arm_fk, pose, ray_error, turret_fk
from jol_perception.pointing_node import NAMES, Pointing, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-p', 'camera_mount:=turret'])
    controller = Pointing()
    controller.set_parameters([Parameter('use_sim_time', value=True)])
    fixture = Node('jol_pointing_coarse_clock_fixture')
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    executor.add_node(fixture)
    clock_pub = fixture.create_publisher(Clock, '/clock', 10)
    received = {key: [] for key in NAMES}
    statuses = []
    subscriptions = [fixture.create_subscription(JointState, '/jol/'+key+'/command',
                     lambda msg, key=key: received[key].append(msg), 10) for key in NAMES]
    subscriptions.append(fixture.create_subscription(String, '/jol/pointing/status',
                         lambda msg: statuses.append(json.loads(msg.data)), 10))
    joints = {'arm': np.zeros(4), 'turret': np.zeros(2)}

    def spin(duration):
        until = time.monotonic()+duration
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=.001)

    def transform(child, matrix, stamp):
        message = TransformStamped()
        message.header.frame_id, message.child_frame_id, message.header.stamp = 'world', child, stamp
        message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = matrix[:3, 3]
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        message.transform.rotation.x, message.transform.rotation.y, message.transform.rotation.z, message.transform.rotation.w = q
        controller.tf.set_transform(message, 'coarse_clock_fixture')

    try:
        spin(.1)
        for step in range(1, 21):
            now = step*.1
            stamp = Time(seconds=now).to_msg()
            for key in NAMES:
                if received[key]:
                    delta = np.asarray(received[key][-1].position)-joints[key]
                    rate = np.array([1.2, .8]) if key == 'turret' else 1.5
                    joints[key] += np.clip(delta, -rate*.1, rate*.1)
                state = JointState()
                state.header.stamp, state.name, state.position = stamp, NAMES[key], joints[key].tolist()
                controller.receive_joints(key, state)
            transform('jol/base', np.eye(4), stamp)
            arm_base = pose((0., 0., .7), yaw=joints['turret'][0]) @ pose((-.8, .9, 1.))
            transform('jol/arm_base', arm_base, stamp)
            origin, tip, axis = arm_fk(joints['arm'], arm_base)
            orientation = arm_base @ pose(yaw=joints['arm'][0], pitch=sum(joints['arm'][1:]))
            orientation[:3, 3] = origin
            transform('jol/rod_base', orientation, stamp)
            orientation[:3, 3] = tip
            transform('jol/rod_tip', orientation, stamp)
            origin, axis = turret_fk(joints['turret'])
            camera = pose(yaw=joints['turret'][0], pitch=joints['turret'][1])
            camera[:3, :3] = camera[:3, :3] @ Rotation.from_euler('xyz', [-np.pi/2, 0., -np.pi/2]).as_matrix()
            camera[:3, 3] = origin
            transform('jol/camera_optical', camera, stamp)
            track = dict(id=1, position=[50., 20., 10.], velocity=[0., 0., 0.],
                         state_covariance=np.eye(6).tolist(), position_covariance=np.eye(3).tolist(),
                         prediction={'acceleration_std': .7}, confirmed=True, hits=3,
                         observed=True, last_seen=now)
            controller.packet = dict(stamp=now, frame_id='world', tracks=[track])
            controller.background_ready = True
            clock_pub.publish(Clock(clock=stamp))
            spin(.035)
        command_stamps = {key: [seconds(msg.header.stamp) for msg in messages]
                          for key, messages in received.items()}
        arm_origin, _, arm_axis = arm_fk(joints['arm'], arm_base)
        camera_origin, camera_axis = turret_fk(joints['turret'])
        arm_error = ray_error(arm_origin, arm_axis, track['position'])['angle_error_deg']
        camera_error = ray_error(camera_origin, camera_axis, track['position'])['angle_error_deg']
        passed = (all(len(v) >= 10 and len(v) == len(set(v)) for v in command_stamps.values())
                  and abs(joints['arm'][0]) < .1 and joints['arm'][1] < -.5
                  and joints['turret'][0] > .2 and joints['turret'][1] < -.1
                  and arm_error < 1. and camera_error < 1.)
        result = dict(passed=bool(passed), scope='Turret-camera ROS DDS clock/timer/command regression; synthetic joints and tracks',
                      clock_hz=10, requested_timer_hz=20,
                      command_stamps=command_stamps, final_joints={key: v.tolist() for key, v in joints.items()},
                      arm_base_world_transform=arm_base.tolist(),
                      arm_world_angle_error_deg=arm_error, camera_world_angle_error_deg=camera_error,
                      status_states=sorted({s['state'] for s in statuses}))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if not passed:
            raise AssertionError('Coarse-clock callbacks must not emit duplicate/reversing slew commands')
    finally:
        executor.shutdown()
        controller.destroy_node()
        fixture.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
