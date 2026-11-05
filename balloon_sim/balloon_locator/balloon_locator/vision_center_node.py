#!/usr/bin/env python3
"""Fly to a target waypoint and track it with the onboard camera.

No path planning — PX4's position controller handles the flight trajectory.
Position setpoints go through the existing MAVROS adapter (/jol/sim_observer/setpoint).

Phases: idle → takeoff → cruise → center → hold

Camera mount (stl_airframe.py):
    optical Z → body (+X+Z)/√2   (45° upward from forward)
    optical X → body -Y
    optical Y → body (+X-Z)/√2
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Transform, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import (MultiDOFJointTrajectory,
                                 MultiDOFJointTrajectoryPoint)
from vision_msgs.msg import Detection2DArray

S = 1.0 / math.sqrt(2.0)
R_BODY_OPTICAL = np.array([[0., S, S], [-1., 0., 0.], [0., -S, S]])


class VisionCenterNode(Node):
    def __init__(self):
        super().__init__('vision_center_node')
        if not self.get_parameter('use_sim_time').value:
            raise ValueError('simulation_time_required')

        self.declare_parameter('target_world', [50.0, 0.0, 15.0])
        self.declare_parameter('hover_altitude_m', 5.0)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 360)
        self.declare_parameter('hfov_deg', 82.9)
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('max_center_speed', 1.0)
        self.declare_parameter('center_range_m', 15.0)
        self.declare_parameter('lost_timeout_sec', 2.0)

        w = self.get_parameter('image_width').value
        h = self.get_parameter('image_height').value
        hfov = math.radians(self.get_parameter('hfov_deg').value)
        self.cx = (w - 1) / 2.0
        self.cy = (h - 1) / 2.0
        self.fx = w / (2.0 * math.tan(hfov / 2.0))
        self.fy = self.fx

        self.own_position = None
        self.own_velocity = None
        self.own_yaw = None
        self.origin = None
        self.north_yaw = None
        self.controller_ready = False
        self.armed = False
        self.adapter_phase = None

        self.phase = 'idle'
        self.hold_ned = None
        self.detection = None
        self.last_detection_time = None
        self.detection_count = 0
        self.last_tick = None

        self.setpoint_pub = self.create_publisher(
            MultiDOFJointTrajectory, '/jol/sim_observer/setpoint', 1)
        self.status_pub = self.create_publisher(
            String, '/jol/vision_center/status', 10)

        self.create_subscription(
            Detection2DArray, '/balloon/onboard/detections',
            self._on_detection, 10)
        self.create_subscription(
            Odometry, '/jol/observer/odometry',
            self._on_odometry, qos_profile_sensor_data)
        self.create_subscription(
            String, '/jol/sim_observer/state', self._on_state, 10)

        self.start_client = self.create_client(Trigger, '/jol/mavros/start')
        self.stop_client = self.create_client(Trigger, '/jol/mavros/stop')
        self.create_service(
            Trigger, '/jol/vision_center/activate', self._activate)
        self.create_service(
            Trigger, '/jol/vision_center/deactivate', self._deactivate)

        self.create_timer(0.05, self._tick)

    # ── helpers ──────────────────────────────────────────────────────

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _world_to_ned(self, pos):
        if self.origin is None or self.north_yaw is None:
            return None
        c, s = math.cos(self.north_yaw), math.sin(self.north_yaw)
        R = np.array([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])
        return R @ (np.asarray(pos, float) - np.asarray(self.origin, float))

    def _vel_world_to_ned(self, vel):
        c, s = math.cos(self.north_yaw), math.sin(self.north_yaw)
        R = np.array([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])
        return R @ np.asarray(vel, float)

    def _publish_setpoint(self, ned_pos, ned_vel=None):
        if ned_vel is None:
            ned_vel = np.zeros(3)
        msg = MultiDOFJointTrajectory()
        msg.header.frame_id = 'jol/flight_ned'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['observer_body']
        pt = MultiDOFJointTrajectoryPoint()
        t = Transform()
        t.translation.x, t.translation.y, t.translation.z = map(float, ned_pos)
        t.rotation.w = 1.0
        v = Twist()
        v.linear.x, v.linear.y, v.linear.z = map(float, ned_vel)
        pt.transforms = [t]
        pt.velocities = [v]
        pt.accelerations = [Twist()]
        msg.points = [pt]
        self.setpoint_pub.publish(msg)

    # ── callbacks ────────────────────────────────────────────────────

    def _on_odometry(self, msg):
        if msg.header.frame_id != 'world':
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vb = msg.twist.twist.linear
        self.own_position = np.array([p.x, p.y, p.z])
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.own_yaw = math.atan2(siny, cosy)
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self.own_velocity = rot.apply([vb.x, vb.y, vb.z])

    def _on_state(self, msg):
        try:
            s = json.loads(msg.data)
            self.controller_ready = s.get('controller_ready', False)
            self.armed = s.get('armed', False)
            self.adapter_phase = s.get('adapter_phase')
            if 'origin_world' in s:
                self.origin = s['origin_world']
            if 'north_yaw_world_rad' in s:
                self.north_yaw = s['north_yaw_world_rad']
        except (ValueError, TypeError):
            pass

    def _on_detection(self, msg):
        if not msg.detections:
            return
        now = self._now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not 0 < stamp <= now + 0.1 or now - stamp > 1.0:
            return
        det = msg.detections[0]
        u = det.bbox.center.position.x
        v = det.bbox.center.position.y
        diam = (det.bbox.size_x + det.bbox.size_y) / 2.0
        if not (math.isfinite(u) and math.isfinite(v) and diam > 3.0):
            return
        self.detection = (u, v, diam, stamp)
        self.last_detection_time = now
        self.detection_count += 1
        if self.detection_count <= 5 or self.detection_count % 50 == 0:
            ex, ey = u - self.cx, v - self.cy
            self.get_logger().info(
                f'DETECT #{self.detection_count}: px=({u:.0f},{v:.0f}) '
                f'err=({ex:.0f},{ey:.0f}) diam={diam:.1f}px')

    # ── services ─────────────────────────────────────────────────────

    def _activate(self, request, response):
        if self.phase != 'idle':
            response.success = True
            response.message = f'already_{self.phase}'
            return response
        if self.own_position is None or self.origin is None:
            response.success = False
            response.message = 'odometry_or_reference_not_ready'
            return response

        hover = self.get_parameter('hover_altitude_m').value
        ned = self._world_to_ned(self.own_position)
        self.hold_ned = ned.copy()
        self.hold_ned[2] = -(hover - self.origin[2])

        self.detection_count = 0
        self.phase = 'takeoff'

        if self.adapter_phase == 'standby' and self.start_client.service_is_ready():
            self.start_client.call_async(Trigger.Request())
            self.get_logger().info('Requesting OFFBOARD arm')

        target = self.get_parameter('target_world').value
        self.get_logger().info(
            f'ACTIVATED  target_world={target}  hover={hover}m  '
            f'hold_ned={self.hold_ned.tolist()}')
        response.success = True
        response.message = 'vision_center_activated'
        return response

    def _deactivate(self, request, response):
        if self.stop_client.service_is_ready():
            self.stop_client.call_async(Trigger.Request())
        self.phase = 'idle'
        response.success = True
        response.message = 'deactivated'
        return response

    # ── vision centering ─────────────────────────────────────────────

    def _compute_center_velocity_ned(self):
        if (self.detection is None or self.own_yaw is None
                or self.north_yaw is None):
            return np.zeros(3)
        u, v, _d, _s = self.detection
        ex, ey = u - self.cx, v - self.cy
        ray_opt = np.array([ex / self.fx, ey / self.fy, 0.0])
        ray_body = R_BODY_OPTICAL @ ray_opt
        kp = self.get_parameter('kp').value
        vx_b = kp * ray_body[0]
        vy_b = kp * ray_body[1]
        lim = self.get_parameter('max_center_speed').value
        spd = math.hypot(vx_b, vy_b)
        if spd > lim:
            vx_b *= lim / spd
            vy_b *= lim / spd
        yaw = self.own_yaw
        vx_w = math.cos(yaw) * vx_b - math.sin(yaw) * vy_b
        vy_w = math.sin(yaw) * vx_b + math.cos(yaw) * vy_b
        return self._vel_world_to_ned([vx_w, vy_w, 0.0])

    # ── main loop ────────────────────────────────────────────────────

    def _try_auto_activate(self):
        if not self.get_parameter('auto_start').value:
            return
        if self.phase != 'idle':
            return
        if (self.own_position is None or self.origin is None
                or not self.controller_ready
                or self.adapter_phase != 'standby'):
            return
        req = Trigger.Request()
        resp = Trigger.Response()
        self._activate(req, resp)
        if resp.success:
            self.get_logger().info(f'AUTO-ACTIVATE: {resp.message}')

    def _tick(self):
        now = self._now()
        if self.phase == 'idle':
            self._try_auto_activate()
            self.last_tick = now
            return
        if self.own_position is None or self.origin is None:
            self.last_tick = now
            return

        dt = min(0.05, max(0.0, now - (self.last_tick or now)))
        self.last_tick = now

        lost_t = self.get_parameter('lost_timeout_sec').value
        det_fresh = (self.last_detection_time is not None
                     and now - self.last_detection_time < lost_t)

        target_world = np.array(self.get_parameter('target_world').value, float)
        hover = self.get_parameter('hover_altitude_m').value
        goal_world = target_world.copy()
        goal_world[2] = hover
        goal_ned = self._world_to_ned(goal_world)
        if goal_ned is None:
            return

        current_ned = self._world_to_ned(self.own_position)
        if current_ned is None:
            return

        # ── state transitions ──
        if self.phase == 'takeoff':
            alt_err = abs(current_ned[2] - self.hold_ned[2])
            if self.armed and alt_err < 0.5:
                self.phase = 'cruise'
                self.get_logger().info(
                    f'CRUISE  → goal_ned={goal_ned.tolist()}')

        elif self.phase == 'cruise':
            horiz = float(np.linalg.norm(current_ned[:2] - goal_ned[:2]))
            center_r = self.get_parameter('center_range_m').value
            if horiz < center_r and det_fresh:
                self.phase = 'center'
                self.hold_ned = current_ned.copy()
                self.hold_ned[2] = goal_ned[2]
                self.get_logger().info(
                    f'CENTER  detected at {horiz:.1f}m, switching to vision')
            elif horiz < 1.0:
                self.phase = 'hold'
                self.hold_ned = goal_ned.copy()
                self.get_logger().info('HOLD  arrived at target')

        elif self.phase == 'center':
            if not det_fresh:
                self.phase = 'cruise'
                self.get_logger().info('CRUISE  detection lost, resuming')

        # ── setpoint computation ──
        vel_ned = np.zeros(3)
        if self.phase == 'takeoff':
            pos_ned = self.hold_ned.copy()

        elif self.phase == 'cruise':
            pos_ned = goal_ned.copy()
            delta = goal_ned[:2] - current_ned[:2]
            dist = float(np.linalg.norm(delta))
            if dist > 1.0:
                vel_ned[:2] = delta / dist * min(4.0, dist)

        elif self.phase == 'center':
            vel_ned = self._compute_center_velocity_ned()
            if dt > 0:
                self.hold_ned[:2] += vel_ned[:2] * dt
            pos_ned = self.hold_ned.copy()

        elif self.phase == 'hold':
            pos_ned = self.hold_ned.copy()

        else:
            return

        self._publish_setpoint(pos_ned, vel_ned)

        # ── status ──
        px_err = None
        if self.detection:
            px_err = [round(self.detection[0] - self.cx, 1),
                      round(self.detection[1] - self.cy, 1)]
        self.status_pub.publish(String(data=json.dumps(dict(
            stamp=now, phase=self.phase,
            armed=self.armed, controller_ready=self.controller_ready,
            detection_fresh=det_fresh, detection_count=self.detection_count,
            pixel_error=px_err,
            detection_diam=(round(self.detection[2], 1)
                            if self.detection else None),
            position_world=(self.own_position.tolist()
                            if self.own_position is not None else None),
            hold_ned=(self.hold_ned.tolist()
                      if self.hold_ned is not None else None),
            velocity_ned=vel_ned.tolist(),
        ))))


def main(args=None):
    rclpy.init(args=args)
    node = VisionCenterNode()
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
