"""Single static target estimation and independent camera gimbal centering.

No ground-truth subscription exists. The fixed LiDAR is registered at point
acquisition times and projected through the camera TF at image exposure time.
"""
from array import array
from collections import Counter, deque
import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from .fusion_core import image_bgr, transform_matrix
from .motion_compensation import register_cloud
from .static_estimator_core import (associate_surface, detect_orange_marker,
    gimbal_center_command, detection_center, lidar_acquisition_candidate, lidar_acquisition_command,
    validate_detection, display_clouds, SENSOR_QUALITY_DEFAULTS,
    sensor_quality_config, calibration_revision, pose_motion, add_sensor_uncertainty)
from .tf_guard import guarded_tf


def stamp_seconds(stamp):
    return (stamp.sec*1000000000+stamp.nanosec)*1e-9


def corrected_stamp_seconds(stamp, offset):
    return (stamp.sec*1000000000+stamp.nanosec+round(offset*1e9))*1e-9


class StaticEstimator(Node):
    def __init__(self, *, camera_only=False):
        super().__init__('jol_static_estimator')
        self.detector_mode = self.declare_parameter('detector_mode', 'orange_marker').value
        if self.detector_mode not in ('orange_marker', 'yolo'):
            raise ValueError('detector_mode must be orange_marker or yolo')
        self.declare_parameter('calibration_valid', False)
        self.declare_parameter('calibration_source', 'unspecified')
        # Offsets correct acquisition stamps before TF lookup, never the TF
        # itself. Restart to change them, so queued clouds cannot mix epochs.
        self.sensor_quality = sensor_quality_config(**{key: float(self.declare_parameter(
            key, value, ParameterDescriptor(read_only=True)).value)
            for key, value in SENSOR_QUALITY_DEFAULTS.items()})
        self.max_age = float(self.declare_parameter('max_age', .8).value)
        self.lidar_window = float(self.declare_parameter('lidar_window', .6).value)
        self.min_points = int(self.declare_parameter('min_points', 3).value)
        self.gimbal_enabled = bool(self.declare_parameter('gimbal_enabled', True).value)
        self.acquisition_requires_start = bool(self.declare_parameter('acquisition_requires_start', True).value)
        self.center_tolerance_deg = float(self.declare_parameter('center_tolerance_deg', 1.).value)
        self.marker_minimum_pixels = int(self.declare_parameter('marker_minimum_pixels', 8).value)
        self.fixed_frame = self.declare_parameter('fixed_frame', 'world').value
        if (not .1 <= self.max_age <= 3 or not .1 <= self.lidar_window <= 1
                or not 3 <= self.min_points <= 100 or not .1 <= self.center_tolerance_deg <= 5 or not self.fixed_frame):
            raise ValueError('Invalid static association age, window, point count or frame')
        self.tf, self.listener = guarded_tf(self, cache_time=Duration(seconds=10.))
        self.info = self.pending_detection = self.gimbal = None
        self.last_image = self.last_result = None
        self.last_image_stamp = self.last_clock = self.last_processed = None
        self.last_annotated = self.last_command = None
        self.last_search = self.last_visual_stamp = None
        self.search_proposal = None
        self.acquisition_started = not self.acquisition_requires_start
        self.search_stage = 'camera_search' if camera_only else 'lidar_search'
        self.acquisition_stage = self.search_stage if self.acquisition_started else 'waiting_start'
        self.acquisition_started_stamp = self.acquisition_stage_stamp = None
        self.acquisition_events = deque(maxlen=16)
        self.acquisition_initial_position = self.acquisition_motion = None
        self.restore_initial_gimbal = False
        self.initial_restore_commanded = False
        self.coarse_gimbal_target = None
        self.coarse_acquisition_complete = False
        self.last_reason = 'waiting_for_camera'
        self.calibration_key = None
        self.calibration_revision = None
        self.lidar_frame = None
        self.pending_clouds = deque(maxlen=12)
        self.clouds = deque(maxlen=30)
        self.counters = Counter()
        self.estimate_pub = self.create_publisher(String, '/jol/static_target/estimate', 10)
        self.status_pub = self.create_publisher(String, '/jol/static_target/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/jol/static_target/markers', 10)
        if not camera_only:
            self.registered_pub = self.create_publisher(PointCloud2, '/jol/registered_points', qos_profile_sensor_data)
            self.context_pub = self.create_publisher(PointCloud2, '/jol/display/context_points', qos_profile_sensor_data)
            self.target_points_pub = self.create_publisher(PointCloud2, '/jol/display/target_points', qos_profile_sensor_data)
        self.command_pub = self.create_publisher(JointState, '/jol/gimbal/command', 10)
        self.start_acquisition_service = self.create_service(
            Trigger, '/jol/static_target/start_acquisition', self.start_acquisition)
        self.reset_acquisition_service = self.create_service(
            Trigger, '/jol/static_target/reset_acquisition', self.reset_acquisition)
        reliable = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        if not camera_only:
            self.create_subscription(PointCloud2, '/jol/lidar/points', self.receive_cloud, reliable)
        self.create_subscription(CameraInfo, '/jol/camera/camera_info', self.receive_info, qos_profile_sensor_data)
        self.create_subscription(JointState, '/jol/gimbal/joint_states', self.receive_gimbal, 10)
        if self.detector_mode == 'orange_marker':
            self.detection_pub = self.create_publisher(String, '/jol/camera/detections', 10)
            self.annotated_pub = self.create_publisher(Image, '/jol/camera/annotated', qos_profile_sensor_data)
            self.create_subscription(Image, '/jol/camera/image_raw', self.receive_image, reliable)
        else:
            self.create_subscription(String, '/jol/camera/detections', self.receive_external_detection, 10)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.02, self.tick, clock=self.steady_clock)
        self.create_timer(.2, self.publish_status, clock=self.steady_clock)
        self.get_logger().info('Static sensor estimator: '+self.detector_mode+'; no ground-truth inputs')

    def now(self):
        current = self.get_clock().now().nanoseconds*1e-9
        if self.last_clock is not None and current < self.last_clock-.001:
            self.clouds.clear()
            self.pending_clouds.clear()
            self.pending_detection = self.last_result = self.gimbal = self.last_image = None
            self.last_image_stamp = self.last_processed = self.last_annotated = self.last_command = None
            self.last_search = self.last_visual_stamp = self.search_proposal = None
            self.reset_acquisition_state(current)
            self.last_reason = 'clock_reset'
        self.last_clock = current
        return current

    def set_acquisition_stage(self, stage, stamp, **details):
        if stage != self.acquisition_stage or not self.acquisition_events:
            self.acquisition_stage, self.acquisition_stage_stamp = stage, stamp
            self.acquisition_events.append(dict(stage=stage, stamp=stamp, **details))

    def reset_acquisition_state(self, current):
        self.acquisition_started = not self.acquisition_requires_start
        self.acquisition_started_stamp = None
        self.pending_detection = self.last_result = None
        self.last_command = self.last_search = self.last_visual_stamp = self.search_proposal = None
        self.acquisition_motion = self.coarse_gimbal_target = None
        self.coarse_acquisition_complete = False
        self.acquisition_events.clear()
        self.restore_initial_gimbal = self.acquisition_initial_position is not None
        self.initial_restore_commanded = False
        self.set_acquisition_stage('resetting' if self.restore_initial_gimbal else
            self.search_stage if self.acquisition_started else 'waiting_start', current)

    def reset_acquisition(self, request, response):
        current = self.now()
        self.reset_acquisition_state(current)
        self.invalidate('acquisition_reset', current)
        response.success, response.message = True, 'acquisition_reset_pending' if self.restore_initial_gimbal else 'acquisition_reset'
        return response

    def drive_initial_gimbal(self, current):
        if not self.restore_initial_gimbal or self.gimbal is None or not 0 <= current-self.gimbal['stamp'] <= .3:
            return
        yaw, pitch = self.acquisition_initial_position
        error = max(abs(math.remainder(yaw-self.gimbal['yaw'], 2*math.pi)), abs(pitch-self.gimbal['pitch']))
        if error <= math.radians(1.):
            self.restore_initial_gimbal = False
            self.set_acquisition_stage('waiting_start', current, measured_position=[self.gimbal['yaw'], self.gimbal['pitch']])
        elif not self.initial_restore_commanded:
            command = JointState()
            command.header.stamp = self.get_clock().now().to_msg()
            command.name = ['gimbal_yaw_joint', 'gimbal_pitch_joint']
            command.position = self.acquisition_initial_position
            self.command_pub.publish(command)
            self.initial_restore_commanded = True

    def start_acquisition(self, request, response):
        """Start camera acquisition only; this service cannot arm or command flight."""
        current = self.now()
        if self.acquisition_started:
            response.success, response.message = True, 'acquisition_already_started'
            return response
        self.drive_initial_gimbal(current)
        if self.restore_initial_gimbal:
            response.success, response.message = False, 'returning_to_initial'
            return response
        if (not bool(self.get_parameter('calibration_valid').value)
                or self.get_parameter('calibration_source').value in ('', 'unspecified')):
            response.success, response.message = False, 'calibration_not_confirmed'
            return response
        self.acquisition_started = True
        self.acquisition_started_stamp = current
        self.pending_detection = self.last_result = None
        self.last_visual_stamp = self.last_search = self.search_proposal = None
        self.acquisition_motion = None
        self.coarse_gimbal_target = None
        self.coarse_acquisition_complete = False
        self.restore_initial_gimbal = False
        if self.acquisition_initial_position is None and self.gimbal is not None and 0 <= current-self.gimbal['stamp'] <= .3:
            self.acquisition_initial_position = [self.gimbal['yaw'], self.gimbal['pitch']]
        self.set_acquisition_stage(self.search_stage, current)
        self.last_reason = 'searching_lidar_candidate'
        response.success, response.message = True, 'acquisition_started'
        return response

    def receive_info(self, msg):
        self.info = dict(k=list(msg.k), d=list(msg.d), width=msg.width, height=msg.height,
                         frame_id=msg.header.frame_id, distortion_model=msg.distortion_model,
                         binning_x=msg.binning_x, binning_y=msg.binning_y,
                         roi_offset=[msg.roi.x_offset, msg.roi.y_offset])

    def receive_gimbal(self, msg):
        if len(msg.name) != len(msg.position):
            return
        state = dict(zip(msg.name, msg.position))
        try:
            yaw, pitch = state['gimbal_yaw_joint'], state['gimbal_pitch_joint']
            stamp = stamp_seconds(msg.header.stamp)
            if np.isfinite([yaw, pitch, stamp]).all() and stamp > 0:
                self.gimbal = dict(yaw=yaw, pitch=pitch, stamp=stamp)
                current = self.now()
                if (self.acquisition_started and self.search_proposal is not None
                        and self.acquisition_motion is None and 0 <= current-stamp <= .3):
                    start = self.search_proposal['gimbal_origin_position']
                    delta = [math.remainder(yaw-start[0], 2*math.pi), pitch-start[1]]
                    if max(map(abs, delta)) > math.radians(.2):
                        self.acquisition_motion = dict(stamp=stamp, measured_position=[yaw, pitch],
                                                       measured_change_rad=delta)
                        if not self.acquisition_requires_start:
                            self.set_acquisition_stage('slewing', stamp, measured_position=[yaw, pitch])
        except KeyError:
            pass

    def receive_cloud(self, msg):
        current = self.now()
        stamp = corrected_stamp_seconds(msg.header.stamp, self.sensor_quality['lidar_time_offset_s'])
        if not msg.header.frame_id or not 0 < stamp <= current+.1:
            self.counters['invalid_cloud'] += 1
            return
        if len(self.pending_clouds) == self.pending_clouds.maxlen:
            self.counters['cloud_queue_overflow'] += 1
        self.lidar_frame = msg.header.frame_id
        self.pending_clouds.append((msg, time.monotonic()))

    def receive_image(self, msg):
        current = self.now()
        stamp = stamp_seconds(msg.header.stamp)
        if not 0 < stamp <= current+.1 or (self.last_image_stamp is not None and stamp <= self.last_image_stamp):
            return
        self.last_image_stamp = stamp
        try:
            image = image_bgr(msg)
            detections = detect_orange_marker(image, self.marker_minimum_pixels)
            packet = dict(stamp=stamp, frame_id=msg.header.frame_id, width=msg.width, height=msg.height,
                          detections=detections, model='controlled_orange_marker_not_neural',
                          result_age_s=current-stamp, source='simulation_color_marker')
            self.last_image = (msg, image)
            self.detection_pub.publish(String(data=json.dumps(packet, allow_nan=False)))
            self.pending_detection = (self.correct_detection_time(packet), time.monotonic())
            self.counters['images_processed'] += 1
            self.annotate(packet)
        except (ValueError, TypeError, cv2.error) as error:
            self.invalidate('invalid_image: '+str(error), current)

    def receive_external_detection(self, msg):
        try:
            packet = json.loads(msg.data)
            if not isinstance(packet, dict):
                raise ValueError('Detection must be an object')
            self.pending_detection = (self.correct_detection_time(packet), time.monotonic())
        except (ValueError, TypeError, KeyError) as error:
            self.invalidate('invalid_detection_packet: '+str(error), self.now())

    def correct_detection_time(self, packet):
        source_stamp = float(packet['stamp'])
        if source_stamp <= 0 or not math.isfinite(source_stamp*1e9):
            raise ValueError('stale_or_invalid_camera_stamp')
        return dict(packet, source_stamp=source_stamp,
                    stamp=(round(source_stamp*1e9)+round(self.sensor_quality['camera_time_offset_s']*1e9))*1e-9)

    def lookup_matrix(self, target, source, stamp):
        transform = self.tf.lookup_transform(target, source, Time(nanoseconds=round(stamp*1e9)))
        p, q = transform.transform.translation, transform.transform.rotation
        return transform_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])

    def lookup_pose(self, source, stamp):
        transform = self.tf.lookup_transform(self.fixed_frame, source, Time(nanoseconds=round(stamp*1e9)))
        p, q = transform.transform.translation, transform.transform.rotation
        return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    def static_calibration_transforms(self, stamp):
        """Fingerprint actual fixed mounting edges, excluding gimbal motion."""
        edges = {}
        for frame in (self.info.get('frame_id') if self.info else None, self.lidar_frame):
            visited = set()
            while frame in self.tf.parents and frame != self.fixed_frame and frame not in visited:
                visited.add(frame)
                parent = self.tf.parents[frame]
                if frame in self.tf.static_frames:
                    transform = self.tf.lookup_transform(parent, frame, Time(nanoseconds=max(1, round(stamp*1e9))))
                    p, q = transform.transform.translation, transform.transform.rotation
                    edges[frame] = dict(parent=parent, pose=[p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                frame = parent
        return edges

    def drain_clouds(self, current):
        for _ in range(len(self.pending_clouds)):
            msg, received = self.pending_clouds.popleft()
            stamp = corrected_stamp_seconds(msg.header.stamp, self.sensor_quality['lidar_time_offset_s'])
            if stamp-current > .5e-9 and time.monotonic()-received <= 1.:
                self.pending_clouds.append((msg, received))
                continue
            if not -.5e-9 <= current-stamp <= self.max_age or time.monotonic()-received > 1.:
                self.counters['stale_cloud'] += 1
                continue
            try:
                fields = {field.name for field in msg.fields}
                if not {'x', 'y', 'z', 'time'} <= fields:
                    raise ValueError('MID360 cloud requires XYZ and header-relative time')
                data = point_cloud2.read_points(msg, field_names=('x', 'y', 'z', 'time'), skip_nans=True)
                values = np.column_stack([data[name].reshape(-1) for name in ('x', 'y', 'z', 'time')]).astype(float)
                # This simulator's MID360 header marks scan end. Do not use
                # points acquired after that end or after the current clock.
                if len(values) and (values[:, 3].max() > 1e-6 or stamp+values[:, 3].min() <= 0):
                    raise ValueError('invalid_scan_end_relative_point_time')
                points, pose, deskewed = register_cloud(values[:, :3], stamp,
                    lambda seconds: self.lookup_pose(msg.header.frame_id, seconds),
                    point_times=values[:, 3], max_scan_seconds=.2, bin_seconds=.01)
                # Registration is general; new fixed LiDAR TF makes deskew exact.
                motion = pose_motion(self.lookup_matrix(self.fixed_frame, msg.header.frame_id, stamp-.02),
                                     self.lookup_matrix(self.fixed_frame, msg.header.frame_id, stamp), .02)
                self.clouds.append(dict(stamp=stamp, source_stamp=stamp_seconds(msg.header.stamp),
                    points=points, origin=pose[:3], deskewed=deskewed, motion=motion))
                header = Header(stamp=Time(nanoseconds=round(stamp*1e9)).to_msg(), frame_id=self.fixed_frame)
                self.registered_pub.publish(point_cloud2.create_cloud_xyz32(header, points))
                target=(self.last_result['position'] if self.last_result and
                        0<=current-self.last_result['stamp']<=self.max_age else None)
                context,neighborhood=display_clouds(points,pose[:3],target)
                self.context_pub.publish(point_cloud2.create_cloud_xyz32(header,context))
                self.target_points_pub.publish(point_cloud2.create_cloud_xyz32(header,neighborhood))
                self.counters['clouds_registered'] += 1
            except (ValueError, TypeError) as error:
                self.counters['invalid_cloud'] += 1
                self.get_logger().warning(str(error), throttle_duration_sec=5)
            except TransformException:
                # TF and cloud DDS callbacks can arrive out of order. Retry this
                # acquisition timestamp, never Time(0) or an unmeasured pose.
                self.pending_clouds.append((msg, received))

    def tick(self):
        current = self.now()
        self.drain_clouds(current)
        fixed_mounts = self.static_calibration_transforms(current)
        key = (bool(self.get_parameter('calibration_valid').value),
               self.get_parameter('calibration_source').value, json.dumps(self.info, sort_keys=True),
               json.dumps(fixed_mounts, sort_keys=True))
        if key != self.calibration_key:
            if self.calibration_key is not None and self.last_result is not None:
                self.invalidate('calibration_changed', current)
            self.calibration_key = key
            try:
                self.calibration_revision = calibration_revision(self.info, key[1], self.sensor_quality, fixed_mounts)
            except ValueError:
                self.calibration_revision = None
            self.last_result = None
        if not self.acquisition_started:
            self.pending_detection = None
            self.last_result = None
            self.drive_initial_gimbal(current)
            self.last_reason = 'returning_to_initial' if self.restore_initial_gimbal else 'waiting_for_acquisition_start'
            self.set_acquisition_stage('resetting' if self.restore_initial_gimbal else 'waiting_start', current)
            return
        if self.acquisition_requires_start and not self.coarse_acquisition_complete:
            # RGB can enter the image during yaw. It must not skip the measured
            # yaw-then-elevation sequence or publish a premature flight point.
            if self.coarse_gimbal_target is None:
                self.search_gimbal(current)
            else:
                self.advance_coarse_gimbal(current)
            self.pending_detection = None
            self.last_result = None
            self.last_reason = 'acquisition_'+self.acquisition_stage
            return
        if self.pending_detection is None:
            self.search_gimbal(current)
            if self.last_result is not None and current-self.last_result['stamp'] > self.max_age:
                self.invalidate('camera_estimate_stale', current)
            return
        packet, received = self.pending_detection
        try:
            if not key[0] or not key[1] or key[1] == 'unspecified':
                raise ValueError('calibration_not_confirmed')
            if self.info is None:
                raise ValueError('waiting_for_camera_info')
            if time.monotonic()-received > self.max_age:
                raise ValueError('camera_result_wall_stale')
            # RGB and /clock arrive on independent DDS queues. Retain this
            # exposure until its acquisition time becomes causal; do not
            # discard a still-fresh coordinate because /clock trails by 1ms.
            if .5e-9 < float(packet['stamp'])-current <= .1+.5e-9:
                self.counters['camera_clock_wait_ticks'] += 1
                if self.last_result is not None and current-self.last_result['stamp'] > self.max_age:
                    self.invalidate('camera_estimate_stale', current)
                elif self.last_result is None:
                    self.last_reason = 'waiting_for_camera_clock'
                return
            detection = validate_detection(packet, self.info, current, self.max_age)
            self.last_visual_stamp = packet['stamp']
            self.search_proposal = None
            if self.acquisition_stage != 'ready':
                self.set_acquisition_stage('visual_centering', current)
            self.center_gimbal(packet, detection, current)
            usable = [cloud for cloud in self.clouds
                      if -.5e-9 <= packet['stamp']-cloud['stamp'] <= self.lidar_window
                      and -.5e-9 <= current-cloud['stamp'] <= self.max_age]
            if not usable:
                raise ValueError('waiting_for_lidar')
            world_camera = self.lookup_matrix(self.fixed_frame, packet['frame_id'], packet['stamp'])
            camera_tf_support = self.tf.last_support.copy()
            result = associate_surface(np.vstack([cloud['points'] for cloud in usable]),
                                       usable[-1]['origin'], world_camera, detection, self.info, self.min_points)
            camera_motion = pose_motion(self.lookup_matrix(self.fixed_frame, packet['frame_id'], packet['stamp']-.02),
                                        world_camera, .02)
            # Every accumulated cloud contributes registration clock error;
            # retain the largest measured motion rather than the newest only.
            lidar_motion = max((c['motion'] for c in usable), key=lambda m: np.linalg.norm(m[0])+
                               np.linalg.norm(m[1])*np.linalg.norm(np.asarray(result['position'])-usable[-1]['origin']))
            add_sensor_uncertainty(result, world_camera, usable[-1]['origin'], self.sensor_quality,
                                   camera_motion, lidar_motion)
            bearing=np.asarray(result['camera_bearing'],float)
            alignment_error_deg=math.degrees(math.atan2(float(np.linalg.norm(bearing[:2])),float(bearing[2])))
            gimbal_tracking=bool(self.gimbal_enabled and self.gimbal is not None
                and 0 <= current-self.gimbal['stamp'] <= .3)
            evidence = dict(points_count=result.pop('points_count'), candidates_count=result.pop('candidates_count'),
                            surface_model=result['surface_model'], surface_fit=result.pop('surface_fit'),
                            cloud_stamps=[cloud['stamp'] for cloud in usable],
                            cloud_source_stamps=[cloud['source_stamp'] for cloud in usable],
                            image_source_stamp=packet.get('source_stamp', packet['stamp']),
                            exposure_stamp=packet['stamp'], camera_tf_stamp=packet['stamp'],
                            association_max_cloud_age_s=max(packet['stamp']-c['stamp'] for c in usable),
                            sensor_quality=result.pop('sensor_quality'),
                            calibration_fixed_transforms=fixed_mounts,
                            camera_linear_velocity_world_m_s=np.asarray(camera_motion[0]).tolist(),
                            camera_angular_velocity_world_rad_s=np.asarray(camera_motion[1]).tolist(),
                            reprojection_consistency=result.pop('reprojection_consistency'),
                            image_size=[packet['width'],packet['height']],
                            camera_alignment_error_deg=alignment_error_deg,
                            gimbal_centered=bool(gimbal_tracking and alignment_error_deg <= self.center_tolerance_deg),
                            gimbal_tracking=gimbal_tracking, center_tolerance_deg=self.center_tolerance_deg,
                            tf_support=camera_tf_support, point_times_applied=all(c['deskewed'] for c in usable),
                            camera_confidence=detection['confidence'], detector_source=detection.get('source', 'yolo'),
                            camera_origin_world=world_camera[:3, 3].tolist(),
                            camera_ray_world=(world_camera[:3, :3]@(
                                np.asarray(result['camera_bearing'])/np.linalg.norm(result['camera_bearing']))).tolist(),
                            target_is_static=True, accumulation_window_s=self.lidar_window)
            self.last_result = dict(valid=True, reason='ready', stamp=packet['stamp'], publication_stamp=current,
                                    frame_id=self.fixed_frame, target_id=1, source='camera_bbox_lidar',
                                    calibration_source=key[1], evidence=evidence, **result)
            self.last_result['calibration_revision'] = self.calibration_revision
            self.last_processed = packet['stamp']
            self.last_reason = 'ready'
            if gimbal_tracking and alignment_error_deg <= self.center_tolerance_deg:
                self.set_acquisition_stage('ready', current)
            else:
                self.set_acquisition_stage('visual_centering', current)
            self.counters['coordinate_updates'] += 1
            self.estimate_pub.publish(String(data=json.dumps(self.last_result, allow_nan=False)))
            self.publish_marker(self.last_result)
            self.pending_detection = None
        except (ValueError, KeyError, TypeError, cv2.error) as error:
            self.invalidate(str(error), current, packet.get('stamp'))
            if str(error) == 'no_detection':
                self.search_gimbal(current)
            # Wait for enough measured returns; retries still obey source age.
            if str(error) not in ('waiting_for_lidar', 'insufficient_associated_lidar_points'):
                self.pending_detection = None
        except TransformException as error:
            self.last_reason = 'exposure_tf_unavailable'
            if time.monotonic()-received > .5:
                self.invalidate('exposure_tf_unavailable', current, packet.get('stamp'))
                self.pending_detection = None
                self.get_logger().warning('Exposure TF: '+str(error), throttle_duration_sec=5)

    def search_gimbal(self, current):
        """LiDAR may steer a search; only confirmed camera association supplies coordinates."""
        if (not self.acquisition_started or not self.gimbal_enabled or self.gimbal is None or self.info is None
                or not bool(self.get_parameter('calibration_valid').value)
                or not 0 <= current-self.gimbal['stamp'] <= .3
                or self.last_search is not None and current-self.last_search < .2
                or self.last_visual_stamp is not None and current-self.last_visual_stamp <= self.max_age):
            return
        if self.acquisition_requires_start and self.coarse_gimbal_target is not None:
            return
        self.last_search = current
        self.set_acquisition_stage('slewing' if self.acquisition_motion is not None else 'lidar_search', current)
        usable = [cloud for cloud in self.clouds if 0 <= current-cloud['stamp'] <= self.lidar_window]
        if not usable:
            return
        candidate = lidar_acquisition_candidate(np.vstack([cloud['points'] for cloud in usable]), usable[-1]['origin'])
        if candidate is None:
            return
        try:
            stamp = self.gimbal['stamp']
            world_base = self.lookup_matrix(self.fixed_frame, 'jol/base', stamp)
            pivot = self.lookup_matrix(self.fixed_frame, 'jol/gimbal_yaw', stamp)[:3, 3]
            position = lidar_acquisition_command(candidate, world_base, pivot)
        except TransformException:
            return
        if position is None:
            return
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = ['gimbal_yaw_joint', 'gimbal_pitch_joint']
        if self.acquisition_requires_start:
            self.coarse_gimbal_target = position
            command.position = [position[0], self.gimbal['pitch']]
            self.set_acquisition_stage('yaw_slew', current,
                measured_start_position=[self.gimbal['yaw'], self.gimbal['pitch']])
        else:
            command.position = position
        self.command_pub.publish(command)
        if self.acquisition_initial_position is None:
            self.acquisition_initial_position = [self.gimbal['yaw'], self.gimbal['pitch']]
        self.search_proposal = dict(stamp=usable[-1]['stamp'], position=candidate.tolist(),
            source='measured_lidar_compact_airborne_cluster', camera_confirmed=False, control_coordinate=False,
            gimbal_command_position=position, gimbal_origin_position=[self.gimbal['yaw'], self.gimbal['pitch']])
        self.counters['lidar_acquisition_commands'] += 1

    def advance_coarse_gimbal(self, current):
        """Advance only on fresh actual joint angles, never command timestamps."""
        if (self.gimbal is None or not 0 <= current-self.gimbal['stamp'] <= .3
                or not bool(self.get_parameter('calibration_valid').value)
                or not any(0 <= current-cloud['stamp'] <= self.lidar_window for cloud in self.clouds)):
            return
        yaw, pitch = self.coarse_gimbal_target
        yaw_error = abs(math.remainder(yaw-self.gimbal['yaw'], 2*math.pi))
        pitch_error = abs(pitch-self.gimbal['pitch'])
        if self.acquisition_stage == 'yaw_slew' and yaw_error <= math.radians(1.):
            command = JointState()
            command.header.stamp = self.get_clock().now().to_msg()
            command.name = ['gimbal_yaw_joint', 'gimbal_pitch_joint']
            command.position = self.coarse_gimbal_target
            self.command_pub.publish(command)
            self.set_acquisition_stage('pitch_slew', current,
                measured_position=[self.gimbal['yaw'], self.gimbal['pitch']], measured_yaw_error_rad=yaw_error)
        elif self.acquisition_stage == 'pitch_slew' and max(yaw_error, pitch_error) <= math.radians(1.):
            self.coarse_acquisition_complete = True
            self.set_acquisition_stage('visual_centering', current,
                measured_position=[self.gimbal['yaw'], self.gimbal['pitch']], measured_pitch_error_rad=pitch_error)

    def center_gimbal(self, packet, detection, current):
        if (not self.acquisition_started or not self.gimbal_enabled or self.gimbal is None or not 0 <= current-self.gimbal['stamp'] <= .3
                or self.last_command == packet['stamp']):
            return
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = ['gimbal_yaw_joint', 'gimbal_pitch_joint']
        command.position = gimbal_center_command(detection['bbox'], self.info,
                                                 self.gimbal['yaw'], self.gimbal['pitch'],
                                                 center=detection_center(detection))
        self.command_pub.publish(command)
        self.last_command = packet['stamp']
        self.counters['gimbal_commands'] += 1

    def invalidate(self, reason, current, source_stamp=None):
        self.last_result = None
        self.last_reason = reason
        self.estimate_pub.publish(String(data=json.dumps(dict(valid=False, reason=reason, stamp=source_stamp or current,
            publication_stamp=current, frame_id=self.fixed_frame, target_id=1,
            position=None, covariance=None, source='camera_bbox_lidar'), allow_nan=False)))

    def annotate(self, packet):
        if self.last_image is None:
            return
        original, image = self.last_image
        image = image.copy()
        for detection in packet['detections']:
            x, y, w, h = map(round, detection['bbox'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (60, 235, 80), 2)
            cv2.putText(image, 'DETECTED', (max(0, x), max(16, y-7)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 235, 80), 1)
        height, width = image.shape[:2]
        cv2.drawMarker(image, (width//2, height//2), (200, 200, 200), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(image, self.last_reason, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
        annotated = Image()
        annotated.header = original.header
        annotated.height, annotated.width = height, width
        annotated.encoding, annotated.step = 'bgr8', width*3
        annotated.data = array('B', np.ascontiguousarray(image).tobytes())
        self.annotated_pub.publish(annotated)

    def publish_marker(self, result):
        marker = Marker()
        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = Time(nanoseconds=round(result['stamp']*1e9)).to_msg()
        marker.ns, marker.id, marker.type, marker.action = 'static_surface_estimate', 1, Marker.SPHERE, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = result['position']
        marker.pose.orientation.w = 1.
        marker.scale.x = marker.scale.y = marker.scale.z = .35
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = .1, .9, 1., 1.
        marker.lifetime = Duration(seconds=self.max_age).to_msg()
        self.marker_pub.publish(MarkerArray(markers=[marker]))

    def publish_status(self):
        now = self.now()
        ready = self.last_result is not None and 0 <= now-self.last_result['stamp'] <= self.max_age
        if not ready and self.last_result is not None:
            self.invalidate('camera_estimate_stale', now)
        self.status_pub.publish(String(data=json.dumps(dict(stamp=now, ready=ready, reason=self.last_reason,
            detector_mode=self.detector_mode, detector_validation='controlled_simulation_marker' if self.detector_mode == 'orange_marker'
            else 'drone_model_semantics_unvalidated_for_marker', calibration_valid=bool(self.get_parameter('calibration_valid').value),
            calibration_source=self.get_parameter('calibration_source').value, counters=dict(self.counters),
            calibration_revision=self.calibration_revision, sensor_quality=self.sensor_quality,
            estimate_age_s=now-self.last_result['stamp'] if self.last_result else None,
            gimbal_center_alignment=True, search_proposal=self.search_proposal,
            acquisition=dict(started=self.acquisition_started, requires_start=self.acquisition_requires_start,
                stage=self.acquisition_stage, stage_started_stamp=self.acquisition_stage_stamp,
                started_stamp=self.acquisition_started_stamp, events=list(self.acquisition_events),
                gimbal_motion_observed=self.acquisition_motion,
                returning_to_initial=self.restore_initial_gimbal,
                coarse_gimbal_target=self.coarse_gimbal_target,
                coarse_acquisition_complete=self.coarse_acquisition_complete),
            ground_truth_used=False), allow_nan=False)))


def main():
    cv2.setNumThreads(1)
    rclpy.init()
    node = StaticEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
