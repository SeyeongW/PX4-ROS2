#!/usr/bin/env python3
"""SIYI A8 mini 짐벌 제어/상태 노드 (UDP SDK).

발행: /gimbal/attitude (Vector3Stamped, x=yaw y=pitch z=roll [deg], 짐벌 보고값 그대로)
      /gimbal/zoom (Float32)
구독: /gimbal/cmd_rate (Vector3, x=yaw y=pitch 속도 -100..100), /gimbal/cmd_zoom (Float32, 절대 줌 배율)
서비스: /gimbal/center (Trigger)
"""
import math
import socket
import time

import rclpy
from geometry_msgs.msg import Vector3, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from . import siyi_protocol as sp


class SiyiGimbalNode(Node):
    def __init__(self):
        super().__init__('siyi_gimbal_node')
        self.declare_parameter('ip', '192.168.144.25')
        self.declare_parameter('port', 37260)
        self.declare_parameter('poll_hz', 20.0)
        self.declare_parameter('cmd_timeout_sec', 0.3)   # 이 시간 동안 속도 명령이 없으면 정지
        self.declare_parameter('attitude_latency_sec', 0.0)  # 보고값이 실제보다 늦다면 보정

        self.addr = (self.get_parameter('ip').value, int(self.get_parameter('port').value))
        self.latency = float(self.get_parameter('attitude_latency_sec').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout_sec').value)
        hz = float(self.get_parameter('poll_hz').value)
        if (not all(math.isfinite(v) for v in (self.latency, self.cmd_timeout, hz))
                or self.latency < 0 or self.cmd_timeout <= 0 or hz <= 0):
            raise ValueError('invalid_gimbal_timing_configuration')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.seq = 0
        self.rate_cmd = (0.0, 0.0)
        self.rate_cmd_time = 0.0
        self.rate_active = False
        self.stop_pending = False

        self.pub_att = self.create_publisher(Vector3Stamped, '/gimbal/attitude', 10)
        self.pub_zoom = self.create_publisher(Float32, '/gimbal/zoom', 10)
        self.create_subscription(Vector3, '/gimbal/cmd_rate', self._on_rate, 10)
        self.create_subscription(Float32, '/gimbal/cmd_zoom', self._on_zoom, 10)
        self.create_service(Trigger, '/gimbal/center', self._on_center)

        self.create_timer(1.0 / hz, self._poll)
        self.create_timer(0.5, lambda: self._send(sp.CMD_CURRENT_ZOOM))
        self.get_logger().info(f'SIYI gimbal -> {self.addr[0]}:{self.addr[1]}')

    def _send(self, cmd, data=b''):
        self.seq = (self.seq + 1) & 0xFFFF
        try:
            frame = sp.build_frame(cmd, data, self.seq)
            if self.sock.sendto(frame, self.addr) != len(frame):
                raise OSError('incomplete UDP datagram send')
            return True  # Local UDP send only; not acknowledgement of device motion.
        except OSError as e:
            self.get_logger().warn(f'send 실패: {e}', throttle_duration_sec=2.0)
            return False

    def _on_rate(self, msg: Vector3):
        if not math.isfinite(msg.x) or not math.isfinite(msg.y):
            return
        if self.stop_pending:
            return  # Retry the failed stop at the bounded poll rate first.
        self.rate_cmd = (msg.x, msg.y)
        self.rate_cmd_time = time.monotonic()
        sent = self._send(sp.CMD_ROTATE, sp.encode_rotate(msg.x, msg.y))
        self.rate_active = not sent or msg.x != 0. or msg.y != 0.
        self.stop_pending = not sent

    def _on_zoom(self, msg: Float32):
        if not math.isfinite(msg.data) or not 1. <= msg.data <= 255.9:
            return
        self._send(sp.CMD_ABSOLUTE_ZOOM, sp.encode_absolute_zoom(msg.data))

    def _on_center(self, _req, resp):
        resp.success = (not getattr(self, 'stop_pending', False)
                        and self._send(sp.CMD_CENTER, b'\x01'))
        resp.message = ('Center datagram sent; physical position unconfirmed' if resp.success
                        else 'Center datagram not sent; physical position unconfirmed')
        return resp

    def _poll(self):
        if (getattr(self, 'stop_pending', False)
                or self.rate_active and time.monotonic() - self.rate_cmd_time > self.cmd_timeout):
            sent = self._send(sp.CMD_ROTATE, sp.encode_rotate(0, 0))
            self.rate_active = self.stop_pending = not sent
            if sent:
                self.rate_cmd = (0., 0.)
            else:
                self.get_logger().warn(
                    '정지 송신 실패; 다음 poll에서 재시도. 실제 짐벌 정지 미확인.',
                    throttle_duration_sec=2.0)
        self._send(sp.CMD_ATTITUDE)
        while True:
            try:
                buf, peer = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            except OSError as e:
                self.get_logger().warn(f'receive 실패: {e}', throttle_duration_sec=2.0)
                break
            if peer != self.addr:
                continue
            try:
                fr = sp.parse_frame(buf)
                if fr.cmd_id == sp.CMD_ATTITUDE:
                    yaw, pitch, roll = sp.decode_attitude(fr.data)
                    m = Vector3Stamped()
                    t = self.get_clock().now().nanoseconds - int(self.latency * 1e9)
                    if t <= 0:
                        continue
                    m.header.stamp = rclpy.time.Time(nanoseconds=t).to_msg()
                    m.header.frame_id = 'gimbal'
                    m.vector.x, m.vector.y, m.vector.z = yaw, pitch, roll
                    self.pub_att.publish(m)
                elif fr.cmd_id == sp.CMD_CURRENT_ZOOM:
                    zoom = sp.decode_zoom(fr.data)
                    if zoom >= 1.:
                        self.pub_zoom.publish(Float32(data=zoom))
            except sp.ProtocolError as e:
                self.get_logger().warn(f'프레임 오류: {e}', throttle_duration_sec=2.0)
                continue


def main(args=None):
    rclpy.init(args=args)
    node = SiyiGimbalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send(sp.CMD_ROTATE, sp.encode_rotate(0, 0))
        node.sock.close()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__ == '__main__':
    main()
