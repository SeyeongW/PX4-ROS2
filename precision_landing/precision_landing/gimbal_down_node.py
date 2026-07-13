#!/usr/bin/env python3
"""SIYI gimbal → point straight down (nadir) and hold while armed (PX4 / MAVROS).

The instant the vehicle ARMS, this points a SIYI gimbal camera (A8 mini / ZR10 /
ZR30 …) straight down and keeps re-asserting the angle so the down-facing view
stays fixed for the vision precision landing. Adapted from the user's SIYI
speed-control snippet -- same 0x55 0x66 framing and CRC-16 -- but it uses the
SIYI *set-angle* command (0x0E, absolute angle) instead of speed (0x07): a speed
command only nudges the gimbal, it can't HOLD an orientation, whereas the
stabilised set-angle command drives to -90° pitch (nadir) and keeps it there
against vehicle motion.

  in  /mavros/state   mavros_msgs/State   (armed flag drives the down command)

SIYI 0x0E DATA = yaw int16, pitch int16, unit 0.1° LE (pitch −900 = −90° = down).
A8 mini pitch range is −90°…+25°; adjust pitch_down_deg per your model.
Serial: SIYI UART/USB-TTL, default /dev/ttyUSB0 @ 115200.
"""

import struct

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State

try:
    import serial
except ImportError:
    serial = None


class SIYIGimbal:
    """Minimal SIYI serial SDK: set-angle (0x0E) + speed (0x07)."""

    def __init__(self, port, baud):
        self.seq = 0
        self.err = None
        if serial is None:
            self.ser = None
            self.err = 'pyserial not installed (pip install pyserial)'
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
        except Exception as e:  # noqa: BLE001
            self.ser = None
            self.err = str(e)

    def _send(self, cmd_id, payload):
        if self.ser is None:
            return
        packet = bytearray([0x55, 0x66, 0x01])        # STX + CTRL(need_ack)
        packet += struct.pack('<H', len(payload))     # data len
        packet += struct.pack('<H', self.seq)         # seq
        packet.append(cmd_id)
        packet += payload
        packet += struct.pack('<H', self.calc_crc16(packet))
        self.ser.write(packet)
        self.seq = (self.seq + 1) & 0xFFFF

    def set_angle(self, yaw_deg, pitch_deg):
        # 0x0E: absolute control angle, 0.1° units. Stabilised -> holds nadir.
        self._send(0x0E, struct.pack('<hh', int(yaw_deg * 10), int(pitch_deg * 10)))

    def send_speed(self, yaw_speed, pitch_speed):
        y = max(-100, min(100, int(yaw_speed)))
        p = max(-100, min(100, int(pitch_speed)))
        self._send(0x07, struct.pack('<bb', y, p))

    @staticmethod
    def calc_crc16(data):
        # CRC-16/CCITT-FALSE, matching the SIYI SDK.
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
                crc &= 0xFFFF
        return crc


class GimbalDownNode(Node):
    def __init__(self):
        super().__init__('gimbal_down_node')

        self.port = self.declare_parameter('gimbal_port', '/dev/ttyUSB0').value
        self.baud = self.declare_parameter('gimbal_baud', 115200).value
        # −90° pitch = straight down. yaw held at 0 so the frame stays aligned.
        self.pitch_down = self.declare_parameter('pitch_down_deg', -90.0).value
        self.yaw_fixed = self.declare_parameter('yaw_deg', 0.0).value
        # Re-assert the angle at this period so a bump / mode change can't leave
        # the gimbal off-nadir mid-landing (set-angle is idempotent).
        self.reassert_s = self.declare_parameter('reassert_period_s', 1.0).value
        # always_down (default TRUE): point straight down the moment the node
        # starts and hold it, regardless of arm state. Set false to instead wait
        # for ARM before pointing down.
        self.always = self.declare_parameter('always_down', True).value
        # Optionally recentre the gimbal (yaw0/pitch0) when the vehicle disarms.
        self.center_on_disarm = self.declare_parameter('center_on_disarm', False).value

        self.gimbal = SIYIGimbal(self.port, self.baud)
        if self.gimbal.ser is None:
            self.get_logger().error(f'gimbal open failed on {self.port}: {self.gimbal.err}')
        else:
            self.get_logger().info(
                f'gimbal on {self.port} @ {self.baud} '
                f'(down={self.pitch_down:.0f}°, always={self.always})')

        self.armed = False
        self.was_armed = False
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.create_timer(self.reassert_s, self._tick)

        if self.always:
            self.gimbal.set_angle(self.yaw_fixed, self.pitch_down)

    def _state_cb(self, msg):
        self.armed = msg.armed

    def _tick(self):
        if self.always or self.armed:
            if self.armed and not self.was_armed:
                self.get_logger().info(
                    f'armed -> gimbal to nadir (yaw {self.yaw_fixed:.0f}, '
                    f'pitch {self.pitch_down:.0f})')
            self.gimbal.set_angle(self.yaw_fixed, self.pitch_down)
        elif self.was_armed and not self.armed and self.center_on_disarm:
            self.get_logger().info('disarmed -> recentre gimbal')
            self.gimbal.set_angle(0.0, 0.0)
        self.was_armed = self.armed


def main(args=None):
    rclpy.init(args=args)
    node = GimbalDownNode()
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
