"""LiDAR 점군을 취득 시각의 TF로 world에 정렬하고 추적 결과를 ROS 토픽으로 발행한다.

tracker_node는 통신·시간·좌표계를, tracking_core는 추적 계산을 담당한다."""
from collections import deque
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String, Header
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from .tracking_core import Config, Tracker, pose_matches
from .motion_compensation import register_cloud, self_filter_points, observer_filter_points
from .tf_guard import guarded_tf


class TrackerNode(Node):
    def __init__(self):
        super().__init__('jol_tracker')
        defaults = Config()
        settings = {f.name: self.declare_parameter(f.name, getattr(defaults, f.name)).value
                    for f in fields(Config)}
        self.tracker = Tracker(Config(**settings))
        self.capture_seconds = self.declare_parameter('background_capture_seconds', 12.0).value
        self.background_file = self.declare_parameter(
            'background_file', str(Path.home() / '.ros/jol/background.npz')).value
        self.frame = self.declare_parameter('lidar_frame', 'jol/lidar').value
        self.fixed_frame = self.declare_parameter('fixed_frame', 'world').value
        self.stabilize = self.declare_parameter('stabilize_frame', True).value
        self.output_frame = self.fixed_frame if self.stabilize else self.frame
        self.map_id = self.declare_parameter('map_id', 'jol_monitor').value
        self.observer_own = None
        if self.declare_parameter('observer_self_filter', False).value:
            if not self.stabilize:
                raise ValueError('Observer self filter requires world stabilization')
            self.create_subscription(Odometry, '/jol/observer/odometry', self.observer_odometry, 10)
        self.pose_position_tolerance = self.declare_parameter('pose_position_tolerance', 0.02).value
        self.pose_angle_tolerance = self.declare_parameter('pose_angle_tolerance', 0.005).value
        self.self_filter = self.declare_parameter('self_filter_enabled', True).value
        self.self_min = np.array(self.declare_parameter('self_filter_min', [-2.0, -2.0, -0.1]).value)
        self.self_max = np.array(self.declare_parameter('self_filter_max', [2.0, 2.0, 3.0]).value)
        self_filter_points(np.empty((0, 3)), self.self_min, self.self_max)
        self.time_field = self.declare_parameter('point_time_field', '').value
        self.time_scale = self.declare_parameter('point_time_scale', 1.0).value
        self.time_reference = self.declare_parameter('point_time_reference', 'relative').value
        self.max_scan_seconds = self.declare_parameter('max_scan_seconds', 0.2).value
        self.deskew_bin_seconds = self.declare_parameter('deskew_bin_seconds', 0.01).value
        self.tf_wait_seconds = self.declare_parameter('tf_wait_seconds', 1.0).value
        self.tf_queue_size = self.declare_parameter('tf_queue_size', 8).value
        if self.tf_wait_seconds <= 0 or self.tf_queue_size < 1:
            raise ValueError('TF retry wait and queue size must be positive')
        self.pending = deque()
        self.drops = {'tf_unavailable_drops': 0, 'tf_queue_overflow_drops': 0, 'invalid_cloud_drops': 0}
        self.capture = True
        self.load_pending = self.declare_parameter('load_background_on_start', False).value
        self.capture_start = None
        self.capture_clouds = []
        self.sensor_pose = None
        self.invalid_pose = False
        self.ready_refresh_pending = False
        self.clock_resets = 0
        self.last_capture_stamp = None
        self.tf_buffer, self.tf_listener = guarded_tf(self)
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        lidar_reliable = self.declare_parameter('lidar_reliable', False).value
        lidar_qos = QoSProfile(depth=1, reliability=(ReliabilityPolicy.RELIABLE
                              if lidar_reliable else ReliabilityPolicy.BEST_EFFORT))
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.background_pub = self.create_publisher(PointCloud2, '/jol/background', latched)
        self.ready_pub = self.create_publisher(Bool, '/jol/background_ready', latched)
        self.registered_pub = self.create_publisher(PointCloud2, '/jol/registered_points', sensor_qos)
        self.foreground_pub = self.create_publisher(PointCloud2, '/jol/foreground', sensor_qos)
        self.tracks_pub = self.create_publisher(String, '/jol/tracks', 10)
        self.detections_pub = self.create_publisher(String, '/jol/detections', 10)
        self.metrics_pub = self.create_publisher(String, '/jol/metrics', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/jol/markers', 10)
        self.subscription = self.create_subscription(
            PointCloud2, '/jol/lidar/points', self.on_cloud, lidar_qos)
        self.retry_timer = self.create_timer(0.01, self._drain_pending,
                                            clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.capture_service = self.create_service(Trigger, '/jol/capture_background', self.recapture)
        self.load_service = self.create_service(Trigger, '/jol/load_background', self.request_load)
        self.save_service = self.create_service(Trigger, '/jol/save_background', self.save)
        self.ready_pub.publish(Bool(data=False))
        self.clock_jump_handle = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            post_callback=self.reset_clock)
        self.get_logger().info(f'Waiting for exact-time TF; empty-scene background in {self.output_frame}')

    def observer_odometry(self, msg):
        if msg.header.frame_id != 'world' or msg.child_frame_id != 'jol/observer_body': return
        p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        values = np.array([p.x,p.y,p.z,q.x,q.y,q.z,q.w,v.x,v.y,v.z])
        if not np.isfinite(values).all() or np.linalg.norm(values[3:7]) < 1e-6: return
        self.observer_own = dict(stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                                position=values[:3], velocity=Rotation.from_quat(values[3:7]).apply(values[7:]))

    def reset_clock(self, jump):
        self.observer_own = None
        # Discard old-epoch messages and partial captures. A completed fixed
        # world map remains valid; announce it again only after new-epoch TF
        # and a cloud have passed the normal registration/pose checks.
        self.pending.clear()
        self.tracker.reset_tracks()
        self.capture_start = self.last_capture_stamp = None
        self.capture_clouds = []
        if self.capture:
            self.sensor_pose = None
        self.ready_refresh_pending = True
        self.clock_resets += 1
        self.ready_pub.publish(Bool(data=False))

    def recapture(self, request, response):
        if self.tracker.tracks and not self.invalid_pose:
            response.success = False
            response.message = 'Move all tracked objects out of view and wait for track expiry before recapture'
            return response
        self.capture = True
        self.load_pending = False
        self.capture_start = self.last_capture_stamp = None
        self.capture_clouds = []
        self.sensor_pose = None
        self.invalid_pose = False
        self.pending.clear()
        self.tracker.reset_tracks()
        self.ready_pub.publish(Bool(data=False))
        response.success = True
        response.message = f'Capturing {self.capture_seconds:.1f} sensor seconds; scene must remain empty while turret sweeps'
        return response

    def request_load(self, request, response):
        response.success = Path(self.background_file).expanduser().is_file()
        if response.success:
            self.load_pending = True
            response.message = 'Load queued; fixed frame, map identity and format will be validated on next cloud'
        else:
            response.message = f'Background does not exist: {self.background_file}'
        return response

    def save(self, request, response):
        if self.tracker.background_tree is None or self.sensor_pose is None or self.invalid_pose or self.capture:
            response.success = False
            response.message = 'No valid, complete background to save'
            return response
        try:
            self.tracker.save_background(self.background_file, self.output_frame, self.sensor_pose,
                                         self.map_id, self.stabilize)
            response.success, response.message = True, self.background_file
        except (OSError, ValueError) as error:
            response.success, response.message = False, str(error)
        return response

    def _pose_at(self, seconds, msg, stamp):
        # Preserve the exact header nanoseconds for the common whole-cloud path.
        when = Time.from_msg(msg.header.stamp) if seconds == stamp else Time(nanoseconds=round(seconds * 1e9))
        transform = self.tf_buffer.lookup_transform(self.fixed_frame, self.frame, when).transform
        p, q = transform.translation, transform.rotation
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def _publish_ready(self, header):
        self.capture = False
        self.invalid_pose = False
        self.ready_refresh_pending = False
        self.capture_clouds = []
        self.background_pub.publish(point_cloud2.create_cloud_xyz32(header, self.tracker.background))
        self.ready_pub.publish(Bool(data=True))
        self.get_logger().info(f'Background ready: {len(self.tracker.background)} returns in {self.output_frame}; frozen until explicit recapture')

    def _drop(self, msg, reason, error):
        self.drops[reason] += 1
        self.metrics_pub.publish(String(data=json.dumps(dict(
            self.drops, stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            status=reason, reason=str(error), tf_pending=len(self.pending), stabilized=self.stabilize))))
        self.get_logger().warning(f'Cloud rejected ({reason}): {error}', throttle_duration_sec=5.0)

    def on_cloud(self, msg):
        if msg.header.frame_id != self.frame:
            self._drop(msg, 'invalid_cloud_drops', f'Expected frame {self.frame}, received {msg.header.frame_id}')
            return
        if len(self.pending) >= self.tf_queue_size:
            old, _ = self.pending.popleft()
            self._drop(old, 'tf_queue_overflow_drops', 'Exact-time TF queue full')
        self.pending.append((msg, time.monotonic()))
        self._drain_pending()

    def _drain_pending(self):
        # TF가 늦으면 잠시 보관한다. 기다리는 동안 ROS 콜백을 막지 않는다.
        if not self.pending:
            return
        msg, received = self.pending[0]
        try:
            self._process_cloud(msg, received)
        except TransformException as error:
            if time.monotonic() - received < self.tf_wait_seconds:
                return  # Yield to TF subscriptions; never block the single-thread executor.
            self._drop(msg, 'tf_unavailable_drops', error)
        except (ValueError, AssertionError, KeyError, IndexError) as error:
            self._drop(msg, 'invalid_cloud_drops', error)
        self.pending.popleft()

    def _process_cloud(self, msg, received):
        # 점별 취득 시각으로 world에 정렬한 뒤 같은 좌표계의 배경과 비교한다.
        started = time.perf_counter()
        queue_wait_ms = 1000 * (time.monotonic() - received)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0 or not np.isfinite(stamp):
            raise ValueError('Cloud timestamp must be positive; zero would request latest TF')
        acquisition_pose = self._pose_at(stamp, msg, stamp)
        acquisition_support = {frame: dict(support) for frame, support in self.tf_buffer.last_support.items()}
        point_times = None
        if self.time_field:
            data = point_cloud2.read_points(msg, field_names=['x', 'y', 'z', self.time_field], skip_nans=False)
            points = np.column_stack([data[name].reshape(-1) for name in ('x', 'y', 'z')]).astype(float)
            point_times = np.asarray(data[self.time_field], dtype=float).reshape(-1)
        else:
            points = np.asarray(point_cloud2.read_points_numpy(
                msg, field_names=['x', 'y', 'z'], skip_nans=False), dtype=float).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if point_times is not None:
            point_times = point_times[finite]
        fixed_points, pose, deskewed = register_cloud(
            points, stamp, lambda seconds: acquisition_pose if seconds == stamp else self._pose_at(seconds, msg, stamp),
            point_times if self.stabilize else None, self.time_scale, self.time_reference,
            self.max_scan_seconds, self.deskew_bin_seconds)
        if self.stabilize:
            points = fixed_points
            sensor_origin = pose[:3]
            if self.self_filter:
                points = self_filter_points(points, self.self_min, self.self_max)
            points = observer_filter_points(points, stamp, self.observer_own, self.max_scan_seconds)
        else:
            sensor_origin = np.zeros(3)
        header = Header(stamp=msg.header.stamp, frame_id=self.output_frame)
        if self.load_pending:
            self.load_pending = False
            try:
                self.tracker.load_background(self.background_file, self.output_frame, pose,
                                             self.pose_position_tolerance, self.pose_angle_tolerance,
                                             self.map_id, self.stabilize)
                self.sensor_pose = pose
                self._publish_ready(header)
            except (OSError, ValueError, KeyError) as error:
                self.get_logger().error(f'Background load refused: {error}')
        if self.sensor_pose is None or self.stabilize:
            self.sensor_pose = pose
        if not self.stabilize and not pose_matches(self.sensor_pose, pose, self.pose_position_tolerance, self.pose_angle_tolerance):
            if not self.invalid_pose:
                self.get_logger().error('Sensor moved in legacy mode: recapture or enable stabilize_frame.')
                self.invalid_pose = True
                self.tracker.reset_tracks()
                self.ready_pub.publish(Bool(data=False))
                self.markers_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
                self.tracks_pub.publish(String(data=json.dumps(
                    {'stamp': stamp, 'frame_id': self.output_frame, 'tracks': [], 'status': 'sensor_pose_changed'})))
            return
        if self.invalid_pose:
            return
        if self.ready_refresh_pending and not self.capture and self.tracker.background_tree is not None:
            self._publish_ready(header)
        self.registered_pub.publish(point_cloud2.create_cloud_xyz32(header, points))
        if self.capture:
            if self.last_capture_stamp is not None and stamp == self.last_capture_stamp:
                return
            if self.last_capture_stamp is not None and stamp < self.last_capture_stamp:
                self.capture_start, self.capture_clouds = None, []
            self.last_capture_stamp = stamp
            if self.capture_start is None:
                self.capture_start = stamp
            self.capture_clouds.append(self.tracker.clean(points, sensor_origin))
            elapsed = stamp - self.capture_start
            self.metrics_pub.publish(String(data=json.dumps(dict(
                self.drops, stamp=stamp, status='capturing_background', capture_seconds=elapsed,
                frame_id=self.output_frame, stabilized=self.stabilize, deskewed=deskewed,
                sensor_pose=acquisition_pose.tolist(), sensor_pose_stamp=stamp,
                sensor_pose_target_frame=self.fixed_frame, sensor_pose_reference='cloud_header',
                tf_sample_support=acquisition_support, tf_guard_rejections=self.tf_buffer.guard_rejections))))
            if elapsed < self.capture_seconds:
                return
            try:
                self.tracker.set_background(np.concatenate(self.capture_clouds), already_cleaned=True)
            except ValueError as error:
                self.get_logger().error(str(error))
                self.capture_start, self.capture_clouds = None, []
                return
            try:
                self.tracker.save_background(self.background_file, self.output_frame, self.sensor_pose,
                                             self.map_id, self.stabilize)
            except OSError as error:
                self.get_logger().error(f'Background active but could not be saved: {error}')
            self._publish_ready(header)
        tracks, foreground = self.tracker.process(points, stamp, sensor_origin)
        if tracks is None:
            return
        self.detections_pub.publish(String(data=json.dumps(
            {'stamp': stamp, 'frame_id': self.output_frame,
             'sensor_origin': sensor_origin.tolist(),
             'detections': self.tracker.last_detections}, allow_nan=False)))
        self.foreground_pub.publish(point_cloud2.create_cloud_xyz32(header, foreground))
        self.markers_pub.publish(self.make_markers(header, tracks, sensor_origin))
        metrics = dict(self.tracker.last_metrics)
        metrics.update(self.drops)
        metrics.update(clock_resets=self.clock_resets,
                       tf_future_message_rejections=self.tf_buffer.future_rejections)
        metrics.update(stamp=stamp, node_processing_ms=1000 * (time.perf_counter() - started),
                       message_age_sim_ms=max(0.0, self.get_clock().now().nanoseconds / 1e6 - stamp * 1000),
                       tf_queue_wait_ms=queue_wait_ms,
                       tf_pending=len(self.pending) - 1, frame_id=self.output_frame,
                       stabilized=self.stabilize, deskewed=deskewed,
                       acquisition_model='per_point_time' if deskewed else 'instantaneous_scan',
                       tf_sample_support=acquisition_support,
                       tf_guard_rejections=self.tf_buffer.guard_rejections)
        self.tracks_pub.publish(String(data=json.dumps(
            {'stamp': stamp, 'frame_id': self.output_frame, 'sensor_frame': self.frame,
             'source_frame': self.frame, 'sensor_origin': sensor_origin.tolist(),
             'sensor_pose': acquisition_pose.tolist(), 'sensor_pose_stamp': stamp,
             'sensor_pose_target_frame': self.fixed_frame, 'sensor_pose_reference': 'cloud_header',
             'stabilized': self.stabilize, 'deskewed': deskewed,
             'tracks': tracks, 'metrics': metrics}, allow_nan=False)))
        self.metrics_pub.publish(String(data=json.dumps(metrics, allow_nan=False)))

    def make_markers(self, header, tracks, sensor_origin):
        markers = [Marker(action=Marker.DELETEALL)]
        for track in tracks:
            object_id = track['id']
            observed = track['observed']
            position = track['position']
            color = ((object_id * 0.61803398875) % 1, 0.9, 0.35)

            def marker(kind, namespace):
                output = Marker(header=header, ns=namespace, id=object_id, type=kind, action=Marker.ADD)
                output.pose.orientation.w = 1.0
                output.color.r, output.color.g, output.color.b = color
                output.color.a = 0.95 if observed else 0.35
                return output

            box = marker(Marker.CUBE, 'boxes')
            box.pose.position = Point(x=position[0], y=position[1], z=position[2])
            box.scale.x, box.scale.y, box.scale.z = track['size']
            box.color.a = 0.22 if observed else 0.08
            markers.append(box)
            label = marker(Marker.TEXT_VIEW_FACING, 'labels')
            label.pose.position = Point(x=position[0], y=position[1], z=position[2] + track['size'][2] / 2 + 1.5)
            label.scale.z = 2.0
            speed = np.linalg.norm(track['velocity'])
            distance = track['nearest_distance']
            distance_text = f'{distance:.2f} m surface' if distance is not None else 'occluded / predicted'
            label.text = f'ID {object_id} | {speed:.1f} m/s\n{distance_text}'
            markers.append(label)
            trail = marker(Marker.LINE_STRIP, 'trails')
            trail.scale.x = 0.07
            trail.points = [Point(x=p[0], y=p[1], z=p[2]) for p in track['trail']]
            if len(trail.points) >= 2:
                markers.append(trail)
            prediction = track.get('prediction')
            if prediction and prediction['horizon_s'] > 0:
                future = prediction['position']
                # Arrow displacement is v * horizon, so it ends at the forecast point.
                if track.get('direction') is not None:
                    arrow = marker(Marker.ARROW, 'velocity_prediction')
                    arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.10, 0.25, 0.40
                    arrow.color.r, arrow.color.g, arrow.color.b = 0.1, 0.85, 1.0
                    arrow.points = [Point(x=position[0], y=position[1], z=position[2]),
                                    Point(x=future[0], y=future[1], z=future[2])]
                    markers.append(arrow)
                forecast = marker(Marker.SPHERE, 'predicted_position')
                forecast.pose.position = Point(x=future[0], y=future[1], z=future[2])
                forecast.scale.x = forecast.scale.y = forecast.scale.z = 0.45
                forecast.color.r, forecast.color.g, forecast.color.b = 0.1, 0.85, 1.0
                forecast.color.a = 0.6 if observed else 0.25
                markers.append(forecast)
                forecast_label = marker(Marker.TEXT_VIEW_FACING, 'prediction_labels')
                forecast_label.pose.position = Point(x=future[0], y=future[1], z=future[2] + 0.8)
                forecast_label.scale.z = 0.8
                forecast_label.color.r, forecast_label.color.g, forecast_label.color.b = 0.1, 0.85, 1.0
                forecast_label.text = f"ID {object_id} +{prediction['horizon_s']:.2f}s predicted"
                markers.append(forecast_label)
            if observed and track['nearest_point'] is not None:
                nearest = track['nearest_point']
                ray = marker(Marker.LINE_LIST, 'surface_distance')
                ray.scale.x = 0.025
                ray.color.a = 0.55
                ray.points = [Point(x=float(sensor_origin[0]), y=float(sensor_origin[1]), z=float(sensor_origin[2])),
                              Point(x=nearest[0], y=nearest[1], z=nearest[2])]
                markers.append(ray)
        return MarkerArray(markers=markers)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
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
