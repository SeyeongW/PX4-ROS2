#!/usr/bin/env python3
"""Manual keyboard teleop for the holonomic city-map trailer (flat_platform),
driven the same way moving_marker_node's pattern='route' drives it: a raw
gz.transport Twist on the model's VelocityControl cmd_vel topic (linear.x =
East, linear.y = North, no yaw -- see that node's _set_model_velocity). No
ROS2 node needed since VelocityControl reads gz-transport directly, same
reasoning as moving_marker_node's gz-side path.

Key layout mirrors ugv_ws/keyboard_ctrl.py's numpad diagram, but each key is
a direct ENU direction (holonomic: no heading, so no separate "turn"):

   u    i    o      NW   N   NE
   j    k    l  =    W  stop  E
   m    ,    .      SW   S   SE

q/z : increase/decrease speed by 10%
k or space : stop
CTRL-C : quit (publishes zero velocity first)

Usage:
  python3 gazebo/tools/trailer_keyboard_ctrl.py [--topic /model/flat_platform/cmd_vel] [--speed 2.0]
"""
import argparse
import math
import sys
import select
import termios
import tty

from gz.transport13 import Node as GzNode
from gz.msgs10.twist_pb2 import Twist as GzTwist

HELP = """
Trailer keyboard teleop (holonomic ENU)
----------------------------------------
   u    i    o      NW   N   NE
   j    k    l  =    W  stop  E
   m    ,    .      SW   S   SE

q/z : speed up/down 10%%
k / space : stop
CTRL-C : quit

speed=%.2f m/s
"""

# key -> (dE, dN) unit direction. Same physical keys as ugv_ws's
# teleop_twist_keyboard-style layout, reinterpreted as direct ENU strafe
# since the trailer has no heading to turn.
MOVE_BINDINGS = {
    'i': (0, 1), 'I': (0, 1),
    ',': (0, -1),
    'j': (-1, 0), 'J': (-1, 0),
    'l': (1, 0), 'L': (1, 0),
    'u': (-1, 1), 'U': (-1, 1),
    'o': (1, 1), 'O': (1, 1),
    'm': (-1, -1), 'M': (-1, -1),
    '.': (1, -1),
}
SPEED_BINDINGS = {'q': 1.1, 'Q': 1.1, 'z': 0.9, 'Z': 0.9}
STOP_KEYS = ('k', 'K', ' ')
# after this many consecutive idle polls (~0.1s each) with no recognized
# key, decay to zero -- same safety behavior as ugv_ws's keyboard_ctrl.py
# (a stuck/disconnected terminal shouldn't leave the trailer creeping).
IDLE_STOP_POLLS = 4


def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/model/flat_platform/cmd_vel')
    ap.add_argument('--speed', type=float, default=2.0, help='m/s, adjustable live with q/z')
    args = ap.parse_args()

    settings = termios.tcgetattr(sys.stdin)
    gz = GzNode()
    pub = gz.advertise(args.topic, GzTwist)

    speed = args.speed
    de = dn = 0
    idle_polls = 0
    print(HELP % speed)
    try:
        while True:
            key = get_key(settings)
            if key in MOVE_BINDINGS:
                de, dn = MOVE_BINDINGS[key]
                idle_polls = 0
            elif key in SPEED_BINDINGS:
                speed *= SPEED_BINDINGS[key]
                idle_polls = 0
                print(HELP % speed)
            elif key in STOP_KEYS:
                de, dn = 0, 0
                idle_polls = 0
            elif key == '\x03':  # Ctrl-C
                break
            else:
                idle_polls += 1
                if idle_polls > IDLE_STOP_POLLS:
                    de, dn = 0, 0

            norm = math.hypot(de, dn) or 1.0
            twist = GzTwist()
            twist.linear.x = speed * de / norm
            twist.linear.y = speed * dn / norm
            twist.linear.z = 0.0
            pub.publish(twist)
    finally:
        pub.publish(GzTwist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == '__main__':
    main()
