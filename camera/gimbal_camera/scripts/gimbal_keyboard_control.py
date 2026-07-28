#!/usr/bin/env python3
"""Keyboard teleop for the PX4 CGO3 gimbal camera (Gazebo Sim / Harmonic).

The gimbal joints are driven by gz's JointPositionController (the PID runs
inside gz), so this node only publishes *target angles*; gz does the rest.
That is why the camera stays steady with no tremor.

Publishes (bridged to gz by gimbal_camera.launch.py):
  /gimbal/yaw_cmd    std_msgs/Float64   target yaw   angle [rad]
  /gimbal/pitch_cmd  std_msgs/Float64   target pitch angle [rad]
  /gimbal/roll_cmd   std_msgs/Float64   target roll  angle [rad]

Keys
  a / d : yaw   left / right       w / s : pitch up / down
  z / c : roll  left / right       space : reset all to 0
  q     : quit
"""

import select
import sys
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

HELP = """
==================== gimbal keyboard control ====================
  a / d : yaw   left / right       w / s : pitch up / down
  z / c : roll  left / right       space : reset all to 0
  q     : quit
================================================================
"""


def clamp(value, low, high):
    return max(low, min(high, value))


class GimbalKeyboardTeleop(Node):

    def __init__(self):
        super().__init__('gimbal_keyboard_teleop')

        # angle change per keypress [rad]
        self.declare_parameter('step', 0.05)
        # publish rate for the (held) targets [Hz]
        self.declare_parameter('publish_rate', 30.0)
        # joint limits (match the gimbal model's SDF)
        self.declare_parameter('yaw_min', -3.14159)
        self.declare_parameter('yaw_max', 3.14159)
        self.declare_parameter('pitch_min', -2.35619)   # -135 deg
        self.declare_parameter('pitch_max', 0.7854)     #  +45 deg
        self.declare_parameter('roll_min', -0.7854)     #  -45 deg
        self.declare_parameter('roll_max', 0.7854)      #  +45 deg

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.yaw_pub = self.create_publisher(Float64, '/gimbal/yaw_cmd', 10)
        self.pitch_pub = self.create_publisher(Float64, '/gimbal/pitch_cmd', 10)
        self.roll_pub = self.create_publisher(Float64, '/gimbal/roll_cmd', 10)

        rate = self.get_parameter('publish_rate').value
        self.create_timer(1.0 / rate, self.publish_targets)

        self.get_logger().info('gimbal keyboard teleop started')
        sys.stdout.write(HELP)
        sys.stdout.flush()

    def publish_targets(self):
        self.yaw_pub.publish(Float64(data=self.yaw))
        self.pitch_pub.publish(Float64(data=self.pitch))
        self.roll_pub.publish(Float64(data=self.roll))

    # ---- keyboard-driven target updates ----
    def nudge_yaw(self, s):
        self.yaw = clamp(self.yaw + s * self._step(),
                         self._p('yaw_min'), self._p('yaw_max'))
        self._show()

    def nudge_pitch(self, s):
        self.pitch = clamp(self.pitch + s * self._step(),
                           self._p('pitch_min'), self._p('pitch_max'))
        self._show()

    def nudge_roll(self, s):
        self.roll = clamp(self.roll + s * self._step(),
                          self._p('roll_min'), self._p('roll_max'))
        self._show()

    def reset(self):
        self.yaw = self.pitch = self.roll = 0.0
        self._show()

    def _step(self):
        return self.get_parameter('step').value

    def _p(self, name):
        return self.get_parameter(name).value

    def _show(self):
        sys.stdout.write(
            f'\rtarget  yaw={self.yaw:+.3f}  pitch={self.pitch:+.3f}  '
            f'roll={self.roll:+.3f} rad     ')
        sys.stdout.flush()


def keyboard_loop(node, stop_event):
    if not sys.stdin.isatty():
        node.get_logger().warn(
            'stdin is not a TTY: keyboard disabled, holding targets at 0. '
            'Run this node directly in a terminal to drive the gimbal.')
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch in ('q', '\x03'):
                    stop_event.set()
                    break
                elif ch == 'a':
                    node.nudge_yaw(+1)
                elif ch == 'd':
                    node.nudge_yaw(-1)
                elif ch == 'w':
                    node.nudge_pitch(+1)
                elif ch == 's':
                    node.nudge_pitch(-1)
                elif ch == 'z':
                    node.nudge_roll(+1)
                elif ch == 'c':
                    node.nudge_roll(-1)
                elif ch == ' ':
                    node.reset()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = GimbalKeyboardTeleop()

    stop_event = threading.Event()
    kb = threading.Thread(
        target=keyboard_loop, args=(node, stop_event), daemon=True)
    kb.start()

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.stdout.write('\ngimbal teleop stopped\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
