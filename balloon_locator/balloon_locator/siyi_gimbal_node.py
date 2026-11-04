#!/usr/bin/env python3
"""SIYI A8 mini 짐벌 제어/상태 노드 (UDP SDK).

발행: /gimbal/attitude (Vector3Stamped, x=yaw y=pitch z=roll [deg], 짐벌 보고값 그대로)
      /gimbal/zoom (Float32)
구독: /gimbal/cmd_rate (Vector3, x=yaw y=pitch 속도 -100..100), /gimbal/cmd_zoom (Float32, 절대 줌 배율)
서비스: /gimbal/center (Trigger)
"""
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

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.seq = 0
        self.rate_cmd = (0.0, 0.0)
        self.rate_cmd_time = 0.0
        self.rate_active = False

        self.pub_att = self.create_publisher(Vector3Stamped, '/gimbal/attitude', 10)
        self.pub_zoom = self.create_publisher(Float32, '/gimbal/zoom', 10)
        self.create_subscription(Vector3, '/gimbal/cmd_rate', self._on_rate, 10)
        self.create_subscription(Float32, '/gimbal/cmd_zoom', self._on_zoom, 10)
        self.create_service(Trigger, '/gimbal/center', self._on_center)

        hz = float(self.get_parameter('poll_hz').value)
        self.create_timer(1.0 / hz, self._poll)
        self.create_timer(0.5, lambda: self._send(sp.CMD_CURRENT_ZOOM))
        self.get_logger().info(f'SIYI gimbal -> {self.addr[0]}:{self.addr[1]}')

    def _send(self, cmd, data=b''):
        self.seq = (self.seq + 1) & 0xFFFF
        try:
            self.sock.sendto(sp.build_frame(cmd, data, self.seq), self.addr)
        except OSError as e:
            self.get_logger().warn(f'send 실패: {e}', throttle_duration_sec=2.0)

    def _on_rate(self, msg: Vector3):
        self.rate_cmd = (msg.x, msg.y)
        self.rate_cmd_time = time.monotonic()
        self.rate_active = True
        self._send(sp.CMD_ROTATE, sp.encode_rotate(msg.x, msg.y))

    def _on_zoom(self, msg: Float32):
        self._send(sp.CMD_ABSOLUTE_ZOOM, sp.encode_absolute_zoom(msg.data))

    def _on_center(self, _req, resp):
        self._send(sp.CMD_CENTER, b'\x01')
        resp.success = True
        return resp

    def _poll(self):
        if self.rate_active and time.monotonic() - self.rate_cmd_time > self.cmd_timeout:
            self._send(sp.CMD_ROTATE, sp.encode_rotate(0, 0))
            self.rate_active = False
        self._send(sp.CMD_ATTITUDE)
        while True:
            try:
                buf, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            try:
                fr = sp.parse_frame(buf)
            except sp.ProtocolError as e:
                self.get_logger().warn(f'프레임 오류: {e}', throttle_duration_sec=2.0)
                continue
            if fr.cmd_id == sp.CMD_ATTITUDE:
                yaw, pitch, roll = sp.decode_attitude(fr.data)
                m = Vector3Stamped()
                t = self.get_clock().now().nanoseconds - int(self.latency * 1e9)
                m.header.stamp = rclpy.time.Time(nanoseconds=t).to_msg()
                m.header.frame_id = 'gimbal'
                m.vector.x, m.vector.y, m.vector.z = yaw, pitch, roll
                self.pub_att.publish(m)
            elif fr.cmd_id == sp.CMD_CURRENT_ZOOM:
                self.pub_zoom.publish(Float32(data=sp.decode_zoom(fr.data)))


def main(args=None):
    rclpy.init(args=args)
    node = SiyiGimbalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send(sp.CMD_ROTATE, sp.encode_rotate(0, 0))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
