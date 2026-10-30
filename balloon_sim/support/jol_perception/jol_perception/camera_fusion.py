"""영상의 ONNX 검출 결과를 발행하고, 보정·노출 시각·TF가 유효할 때만 LiDAR와 융합한다.

실물 카메라도 독립 2D 검출은 가능하다. 보정값이나 LiDAR가 없으면 3D 융합하지 않는다."""
from array import array
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from .tf_guard import guarded_tf
from visualization_msgs.msg import Marker, MarkerArray

from .fusion_core import (BearingConfig, DroneNetwork, calibration_error, crop_regions, fuse,
                          image_bgr, project_tracks, reacquisition_regions, transform_matrix)
from .fusion_history import replay_camera_track, reanchor_camera_seed


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class PipelineAudit:
    """Bounded callback evidence; these counters cannot measure DDS non-receipt."""
    def __init__(self, timing_point):
        self.timing_point = timing_point
        self.sequence = 0
        self.events = deque(maxlen=64)
        self.counters = Counter({name: 0 for name in (
            'image_received', 'input_accepted', 'input_rejected', 'queue_dropped',
            'inference_submitted', 'inference_completed', 'inference_failed',
            'result_discarded', 'publication_failed', 'detections_published', 'result_published',
            'tf_wait_started', 'tf_wait_recovered', 'tf_wait_timeout')})

    def record(self, event, stamp, epoch, reason='', **detail):
        self.sequence += 1
        self.counters[event] += 1
        if reason and event in ('input_rejected', 'queue_dropped', 'result_discarded'):
            self.counters[event+'_'+reason] += 1
        self.events.append(dict(sequence=self.sequence, event=event, stamp=stamp, epoch=epoch,
                                reason=reason, **self.timing_point(), **detail))


class CameraFusion(Node):
    def __init__(self):
        super().__init__('jol_camera_fusion')
        model_path = self.declare_parameter('model_path', str(Path.home() / 'ros2_nub/models/drone_yolo.onnx')).value
        threshold = self.declare_parameter('confidence_threshold', .35).value
        threads = self.declare_parameter('inference_threads', 2).value
        self.requested_provider = self.declare_parameter('onnx_provider', 'CPUExecutionProvider').value
        # Generic sensor compatibility stays best effort. A known reliable
        # publisher (such as the local Gazebo camera) can explicitly opt in.
        self.camera_reliable = bool(self.declare_parameter('camera_reliable', False).value)
        self.declare_parameter('calibration_valid', False)
        self.declare_parameter('calibration_source', 'unspecified')
        self.tolerance = float(self.declare_parameter('sync_tolerance', .15).value)
        self.max_age = float(self.declare_parameter('max_result_age', .8).value)
        self.tf_wait_seconds = float(self.declare_parameter('camera_tf_wait_seconds', .1).value)
        self.roi_enabled = bool(self.declare_parameter('lidar_roi_enabled', True).value)
        # Recorded coast-frame A/B found extra CPU cost without more detections
        # for the present weights; retain as an explicit experimental option.
        self.reacquisition_enabled = bool(self.declare_parameter('kalman_reacquisition_enabled', False).value)
        self.coordinate_fusion = bool(self.declare_parameter('coordinate_fusion_enabled', True).value)
        self.bearing_config = BearingConfig(
            pixel_std=float(self.declare_parameter('camera_pixel_std', 2.).value),
            box_fraction=float(self.declare_parameter('camera_box_fraction', .1).value),
            calibration_std=float(self.declare_parameter('camera_calibration_std', 1.).value),
            innovation_gate=float(self.declare_parameter('camera_innovation_gate', 9.21).value))
        if not 0 < self.tolerance <= .5 or not .05 <= self.max_age <= 5:
            raise ValueError('Invalid camera synchronization/result age limits')
        if not 0 <= self.tf_wait_seconds <= .5:
            raise ValueError('Camera TF wait must be between zero and 0.5 wall seconds')
        self.network, self.error = None, ''
        try:
            self.network = DroneNetwork(model_path, float(threshold), int(threads), self.requested_provider)
        except Exception as error:
            self.error = 'model_initialization: ' + str(error)
            self.get_logger().error(self.error)
        self.tf, self.listener = guarded_tf(self, cache_time=Duration(seconds=15.))
        self.info = self.latest = self.future = None
        self.latest_timing = self.fusion_seed = self.last_timing = None
        self.fusion_context = None
        self.last_replay_key = None
        self.replay_publications = 0
        self.last_clock = self.last_stamp = self.last_result_stamp = None
        self.last_image_wall_ns = self.last_result_wall_ns = None
        self.calibration_revision = 0
        self.calibration_key = self.current_calibration_key()
        self.epoch = self.processed = self.superseded = self.failures = 0
        self.pipeline = PipelineAudit(self.timing_point)
        self.last_inference_ms = None
        self.last_reason = 'waiting_for_camera'
        self.geometry_error, self.geometry_tf_bounds = '', {}
        self.geometry_wait_frames = self.geometry_wait_timeouts = 0
        self.tf_wait_budget_start = None
        self.history = deque(maxlen=60)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jol_onnx')
        self.image_pub = self.create_publisher(Image, '/jol/camera/annotated', qos_profile_sensor_data)
        self.detection_pub = self.create_publisher(String, '/jol/camera/detections', 10)
        self.fused_pub = self.create_publisher(String, '/jol/fused_tracks', 10)
        self.status_pub = self.create_publisher(String, '/jol/camera/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/jol/fusion_markers', 10)
        image_qos = QoSProfile(depth=5, reliability=(ReliabilityPolicy.RELIABLE if self.camera_reliable
                                                   else ReliabilityPolicy.BEST_EFFORT))
        self.image_subscription = self.create_subscription(Image, '/jol/camera/image_raw', self.receive_image, image_qos)
        self.create_subscription(CameraInfo, '/jol/camera/camera_info', self.receive_info, qos_profile_sensor_data)
        self.create_subscription(String, '/jol/tracks', self.receive_tracks, 10)
        # Completion must not wait for the next /clock message (10 Hz in this
        # simulation). Workers only wake the ROS executor; ROS state stays there.
        self.completion_guard = self.create_guard_condition(self.tick)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.01, self.tick, clock=self.steady_clock)
        self.create_timer(.5, self.publish_status, clock=self.steady_clock)

    def receive_info(self, msg):
        self.info = dict(k=list(msg.k), d=list(msg.d), width=msg.width, height=msg.height,
                         frame_id=msg.header.frame_id, distortion_model=msg.distortion_model,
                         binning_x=msg.binning_x, binning_y=msg.binning_y,
                         roi_offset=[msg.roi.x_offset, msg.roi.y_offset])
        self.check_calibration()

    def current_calibration_key(self):
        return (bool(self.get_parameter('calibration_valid').value),
                self.get_parameter('calibration_source').value,
                json.dumps(self.info, sort_keys=True))

    def check_calibration(self):
        key = self.current_calibration_key()
        if key != self.calibration_key:
            self.calibration_key = key
            self.calibration_revision += 1
            self.tf_wait_budget_start = None
            self.invalidate_fusion('calibration_changed')

    def invalidate_fusion(self, reason):
        seed = self.fusion_seed
        self.fusion_seed = self.fusion_context = self.last_replay_key = None
        self.last_reason = reason
        if seed is not None:
            # Revoke the prior branch in subscribers as well as this node. A
            # higher revision preserves the controller's monotonic input gate.
            now = self.get_clock().now().nanoseconds*1e-9
            invalid = dict(seed, tracks=[], fusion_reason=reason, publication_kind='invalidation',
                           fusion_revision=seed.get('fusion_revision', 0)+1,
                           current_state_stamp=max(seed['stamp'], now), coordinate_updates=0,
                           current_coordinate_updates=0, result_age_s=now-seed['stamp'],
                           source_exposure_age_s=now-seed['stamp'], state_age_s=0.,
                           calibration_revision=self.calibration_revision)
            self.fused_pub.publish(String(data=json.dumps(invalid, allow_nan=False)))
            self.publish_markers(self.get_clock().now().to_msg(), seed['frame_id'], [])

    def health(self, now):
        wall = time.monotonic_ns()
        image_age = ((wall-self.last_image_wall_ns)*1e-9
                     if self.last_image_wall_ns is not None else None)
        result_wall_age = ((wall-self.last_result_wall_ns)*1e-9
                           if self.last_result_wall_ns is not None else None)
        age = now-self.last_result_stamp if self.last_result_stamp is not None else None
        if self.network is None:
            reason = 'model_unavailable'
        elif self.error:
            reason = 'inference_or_fusion_error'
        elif image_age is None:
            reason = 'waiting_for_camera'
        elif image_age > self.max_age:
            reason = 'camera_stream_stale'
        elif result_wall_age is None:
            reason = 'waiting_for_inference'
        elif result_wall_age > self.max_age:
            reason = 'inference_result_stale'
        elif age is None or not -.05 <= age <= self.max_age:
            reason = 'camera_exposure_stale'
        else:
            reason = 'ready'
        ready = reason == 'ready'
        return dict(ready=ready, inference_ready=ready,
                    fusion_ready=ready and self.last_reason == 'ready' and self.fusion_seed is not None,
                    health_reason=reason, result_age_s=age,
                    image_wall_age_s=image_age, result_wall_age_s=result_wall_age)

    def receive_tracks(self, msg):
        try:
            current = self.check_clock()
            packet = json.loads(msg.data)
            if (not isinstance(packet, dict) or not isinstance(packet.get('frame_id'), str)
                    or not packet['frame_id'] or not np.isfinite(float(packet['stamp']))
                    or not isinstance(packet['tracks'], list) or len(packet['tracks']) > 256):
                raise ValueError('Invalid track packet')
            if packet['stamp'] <= 0 or packet['stamp'] > current+self.tolerance:
                return
            if self.history and packet['stamp'] <= self.history[-1]['stamp']:
                return
            identities = set()
            for track in packet['tracks']:
                if (not isinstance(track, dict) or type(track.get('id')) is not int
                        or track['id'] < 0 or track['id'] in identities
                        or type(track.get('observed')) is not bool
                        or not isinstance(track.get('prediction', {}), dict)):
                    raise ValueError('Invalid track identity, observation flag, or prediction')
                identities.add(track['id'])
                for name in ('position', 'velocity', 'size'):
                    value = np.asarray(track[name], dtype=float)
                    if (value.shape != (3,) or not np.isfinite(value).all()
                            or (name == 'size' and (value < 0).any())):
                        raise ValueError('Invalid track '+name)
            self.history.append(packet)
            if self.fusion_seed is not None:
                self.publish_replay(current)
        except (ValueError, KeyError, TypeError) as error:
            self.get_logger().warning('Rejected track input: ' + str(error), throttle_duration_sec=5)

    def check_clock(self):
        current = self.get_clock().now().nanoseconds * 1e-9
        if self.last_clock is not None and current < self.last_clock - .001:
            if self.latest is not None:
                self.pipeline.record('queue_dropped', self.last_stamp, self.epoch, 'clock_reset')
            self.epoch += 1
            self.history.clear()
            self.latest = self.last_stamp = self.last_result_stamp = None
            self.latest_timing = self.fusion_seed = self.last_timing = None
            self.last_image_wall_ns = self.last_result_wall_ns = None
            self.fusion_context = None
            self.last_replay_key = None
            self.last_reason = 'waiting_for_camera'
            self.tf_wait_budget_start = None
        self.last_clock = current
        return current

    def timing_point(self):
        return dict(monotonic_ns=time.monotonic_ns(), ros_s=self.get_clock().now().nanoseconds*1e-9)

    def receive_image(self, msg):
        current = self.check_clock()
        self.check_calibration()
        stamp = stamp_seconds(msg.header.stamp)
        self.pipeline.record('image_received', stamp, self.epoch, frame_id=msg.header.frame_id)
        rejection = ('missing_frame' if not msg.header.frame_id else
                     'nonpositive_stamp' if stamp <= 0 else
                     'future_stamp' if stamp > current+self.tolerance else
                     'nonincreasing_stamp' if self.last_stamp is not None and stamp <= self.last_stamp else '')
        if rejection:
            self.pipeline.record('input_rejected', stamp, self.epoch, rejection)
            return
        self.pipeline.record('input_accepted', stamp, self.epoch)
        if self.latest is not None:
            reason = ('worker_inflight' if self.future is not None else
                      'tf_wait' if 'tf_wait_start' in self.latest_timing else
                      'clock_wait' if 'clock_wait_start' in self.latest_timing else 'callback_pending')
            self.pipeline.record('queue_dropped', self.last_stamp, self.epoch, reason,
                                 replaced_by_stamp=stamp)
        self.superseded += int(self.latest is not None)
        self.latest, self.last_stamp = msg, stamp
        self.latest_timing = dict(image_receive=dict(self.timing_point(),
                                 calibration_revision=self.calibration_revision))
        self.last_image_wall_ns = self.latest_timing['image_receive']['monotonic_ns']
        if self.network is None:
            self.pipeline.record('queue_dropped', stamp, self.epoch, 'model_unavailable')
            self.latest = self.latest_timing = None
            self.image_pub.publish(msg)
        else:
            self.completion_guard.trigger()

    def geometry(self, msg):
        self.geometry_error = ''
        if not self.get_parameter('calibration_valid').value:
            return [], None, None, 'uncalibrated'
        if self.info is None:
            return [], None, None, 'waiting_for_camera_info'
        reason = calibration_error(self.info, msg.width, msg.height, msg.header.frame_id)
        if reason:
            return [], None, None, reason
        if not self.history:
            return [], None, None, 'waiting_for_lidar'
        stamp = stamp_seconds(msg.header.stamp)
        # tf2 reserves zero for the latest transform, which is unsafe on a
        # rotating rig. A real exposure time is required for 3-D fusion.
        if not np.isfinite(stamp) or stamp <= 0:
            return [], None, None, 'invalid_image_stamp'
        # A future LiDAR posterior already contains measurements after exposure.
        # Rewinding it then replaying those measurements would count them twice.
        causal = [packet for packet in self.history if packet['stamp'] <= stamp+1e-9]
        if not causal:
            return [], None, None, 'no_causal_lidar_prior'
        packet = max(causal, key=lambda packet: packet['stamp'])
        if stamp-packet['stamp'] > self.tolerance:
            return [], None, packet, 'stale_lidar'
        try:
            # This exact-time request may interpolate buffered dynamic TF, but
            # must fail on unavailable/extrapolated time; never retry Time(0).
            transform = self.tf.lookup_transform(msg.header.frame_id, packet['frame_id'], Time.from_msg(msg.header.stamp))
            t, q = transform.transform.translation, transform.transform.rotation
            matrix = transform_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            projections, reason = project_tracks(packet, stamp, matrix, self.info, self.tolerance)
            return projections, matrix, packet, reason
        except Exception as error:
            self.geometry_error = str(error)
            self.geometry_tf_bounds = {child: dict(oldest_ns=values[0], latest_ns=values[-1])
                                       for child, values in getattr(self.tf, 'dynamic_times', {}).copy().items()
                                       if values}
            self.get_logger().warning('Fusion TF/geometry unavailable: ' + str(error), throttle_duration_sec=5)
            return [], None, packet, 'tf_or_geometry_unavailable'

    def infer(self, msg, regions, timing):
        timing['worker_start'] = self.timing_point()
        start = time.perf_counter()
        image = image_bgr(msg)
        timing['preprocess_end'] = self.timing_point()
        detections = self.network.infer(image, regions)
        timing['worker_end'] = self.timing_point()
        return image, detections, 1000 * (time.perf_counter() - start)

    def tick(self):
        # 추론은 작업 스레드에서 수행하고 ROS 시간·TF·결과 발행은 여기서 처리한다.
        current = self.check_clock()
        self.check_calibration()
        if self.future is not None:
            future, context = self.future
            if not future.done():
                if self.fusion_seed is not None:
                    self.publish_replay(current)
                return
            self.future = None
            if context['epoch'] == self.epoch:
                stage = 'inference'
                stamp = stamp_seconds(context['message'].header.stamp)
                try:
                    context['timing']['completion_callback'] = self.timing_point()
                    image, detections, elapsed = future.result()
                    self.pipeline.record('inference_completed', stamp, self.epoch,
                                         worker_end=context['timing'].get('worker_end'))
                    stage = 'publication'
                    self.error = ''
                    self.publish_result(context, image, detections, elapsed)
                    self.pipeline.record('result_published', stamp, self.epoch, self.last_reason)
                except Exception as error:
                    self.pipeline.record('inference_failed' if stage == 'inference' else 'publication_failed',
                                         stamp, self.epoch, error=str(error))
                    self.error = 'inference_or_fusion: ' + str(error)
                    self.failures += 1
                    self.invalidate_fusion('inference_or_fusion_error')
                    self.get_logger().error(self.error, throttle_duration_sec=5)
                    self.image_pub.publish(context['message'])
            else:
                self.pipeline.record('result_discarded', stamp_seconds(context['message'].header.stamp),
                                     context['epoch'], 'clock_reset')
        if self.fusion_seed is not None:
            self.publish_replay(current)
        if self.network is None or self.latest is None:
            return
        msg = self.latest
        timing = self.latest_timing
        stamp = stamp_seconds(msg.header.stamp)
        if current < stamp_seconds(msg.header.stamp):
            if 'clock_wait_start' not in timing:
                timing['clock_wait_start'] = self.timing_point()
                self.pipeline.record('clock_wait_started', stamp, self.epoch)
            return  # Preserve a valid exposure ahead of the most recent /clock.
        if 'clock_wait_start' in timing and 'clock_wait_end' not in timing:
            timing['clock_wait_end'] = self.timing_point()
            self.pipeline.record('clock_wait_recovered', stamp, self.epoch)
        # A paused simulator supplies no new exposure. A queued old frame must
        # not start an inference after a long simulated-time backlog either.
        stale_ros = not 0 <= current-stamp <= self.max_age
        stale_wall = (time.monotonic_ns()-timing['image_receive']['monotonic_ns'])*1e-9 > self.max_age
        if stale_ros or stale_wall:
            self.pipeline.record('queue_dropped', stamp, self.epoch,
                                 'stale_ros' if stale_ros else 'stale_wall')
            self.latest = self.latest_timing = None
            self.superseded += 1
            return
        timing['geometry_start'] = self.timing_point()
        projections, matrix, packet, reason = self.geometry(msg)
        timing['geometry_end'] = self.timing_point()
        if reason == 'tf_or_geometry_unavailable' and self.tf_wait_seconds > 0:
            # Images and TF arrive through different DDS subscriptions. Keep
            # this exposure briefly while yielding to TF callbacks; never block
            # the executor, widen interpolation gaps, or request latest TF.
            if 'tf_wait_start' not in timing:
                timing['tf_wait_start'] = self.timing_point()
                self.geometry_wait_frames += 1
                self.pipeline.record('tf_wait_started', stamp, self.epoch)
            # A new exposure must not renew this budget: at 15Hz replacements
            # would otherwise prevent a 100ms timeout forever. After exhaustion,
            # keep full-frame NN running while exact-time fusion remains gated.
            if self.tf_wait_budget_start is None:
                self.tf_wait_budget_start = self.timing_point()
            timing['tf_wait_budget_start'] = self.tf_wait_budget_start
            waited = (time.monotonic_ns()-self.tf_wait_budget_start['monotonic_ns'])*1e-9
            if waited < self.tf_wait_seconds:
                return
            self.geometry_wait_timeouts += 1
            self.pipeline.record('tf_wait_timeout', stamp, self.epoch)
        else:
            self.tf_wait_budget_start = None
        if 'tf_wait_start' in timing:
            timing['tf_wait_end'] = self.timing_point()
            if reason != 'tf_or_geometry_unavailable':
                self.pipeline.record('tf_wait_recovered' if reason == 'ready' else 'tf_wait_aborted',
                                     stamp, self.epoch, reason)
        self.latest = self.latest_timing = None
        regions = crop_regions(projections, msg.width, msg.height) if self.roi_enabled else []
        if self.roi_enabled and self.reacquisition_enabled and reason == 'ready':
            regions = (regions + reacquisition_regions(packet, stamp_seconds(msg.header.stamp),
                                                       matrix, self.info, self.tolerance))[:4]
        context = dict(message=msg, projections=projections, matrix=matrix, packet=packet,
                       reason=reason, info=self.info, epoch=self.epoch, regions=regions, timing=timing,
                       calibration_revision=timing['image_receive']['calibration_revision'])
        timing['worker_submit'] = self.timing_point()
        future = self.worker.submit(self.infer, msg, regions, timing)
        self.future = future, context
        self.pipeline.record('inference_submitted', stamp, self.epoch, reason,
                             input_receive=timing['image_receive'])
        future.add_done_callback(lambda _: self.completion_guard.trigger())

    def publish_result(self, context, image, detections, elapsed):
        # 추론 중에도 LiDAR가 도착하므로 원래 영상 시각을 기준으로 다시 연결한다.
        msg = context['message']
        stamp = stamp_seconds(msg.header.stamp)
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        # Tracking may finish while ONNX is running. Match the original exposure
        # again against the now-available history; never widen the time gate.
        projections, matrix, packet, reason = self.geometry(msg)
        if reason == 'ready':
            self.tf_wait_budget_start = None
        context.update(projections=projections, matrix=matrix, packet=packet, info=self.info)
        if context.get('calibration_revision', 0) != getattr(self, 'calibration_revision', 0):
            reason = 'calibration_changed'
        if not self.get_parameter('calibration_valid').value:
            reason = 'uncalibrated'
        image_receive_ns = context.get('timing', {}).get('image_receive', {}).get('monotonic_ns')
        if (age < -.05 or age > self.max_age or (image_receive_ns is not None
                and (time.monotonic_ns()-image_receive_ns)*1e-9 > self.max_age)):
            reason = 'stale_camera_result'
        tracks = (fuse(context['projections'], detections, context['matrix'], context['info'],
                       bearing_config=getattr(self, 'bearing_config', None),
                       position_fusion_enabled=getattr(self, 'coordinate_fusion', True))
                  if reason == 'ready' else [])
        packet = context['packet']
        timing = context.get('timing', {})
        timing['fusion_end'] = dict(monotonic_ns=time.monotonic_ns(),
                                   ros_s=self.get_clock().now().nanoseconds*1e-9)
        detection_result = dict(stamp=stamp, frame_id=msg.header.frame_id, width=msg.width, height=msg.height,
                                model=self.network.model, inference_ms=elapsed, result_age_s=age,
                                detections=detections, lidar_roi_count=sum(not isinstance(r, dict) for r in context['regions']),
                                reacquisition_rois=[r for r in context['regions'] if isinstance(r, dict)], timing=timing,
                                active_providers=getattr(self.network, 'providers', []),
                                provider_options=getattr(self.network, 'provider_options', {}))
        detection_result['subtype_validation'] = 'unvalidated_for_current_scene'
        detection_result['tf_wait_ms'] = ((timing['worker_submit']['monotonic_ns']-
                                          timing['tf_wait_start']['monotonic_ns'])*1e-6
                                         if 'tf_wait_start' in timing else 0.)
        fused_result = dict(stamp=stamp, frame_id=packet['frame_id'] if packet else '', tracks=tracks,
                            camera_frame=msg.header.frame_id, fusion_reason=reason,
                            calibration_valid=bool(self.get_parameter('calibration_valid').value),
                            calibration_source=self.get_parameter('calibration_source').value,
                            lidar_stamp=packet['stamp'] if packet else None,
                            sensor_frame=packet.get('sensor_frame', packet.get('source_frame', packet['frame_id'])) if packet else None,
                            sensor_origin=packet.get('sensor_origin') if packet else None,
                            stabilized=packet.get('stabilized', False) if packet else False,
                            deskewed=packet.get('deskewed', False) if packet else False,
                            camera_transform_stamp=stamp if context['matrix'] is not None else None,
                            time_difference_s=stamp-packet['stamp'] if packet else None,
                            inference_ms=elapsed, result_age_s=age, timing=timing,
                            source_exposure_stamp=stamp, publication_kind='exposure_result', fusion_revision=0)
        fused_result['subtype_validation'] = 'unvalidated_for_current_scene'
        fused_result['calibration_revision'] = getattr(self, 'calibration_revision', 0)
        fused_result['image_calibration_revision'] = context.get('calibration_revision', 0)
        fused_result['coordinate_fusion_enabled'] = getattr(self, 'coordinate_fusion', True)
        fused_result['coordinate_updates'] = sum(t['fusion_update']['accepted'] for t in tracks)
        # Preserve exposure-time XYZ for sensor accuracy evaluation. Replay lives
        # in each track.current_state and never overwrites that original state.
        self.fusion_seed = fused_result if reason == 'ready' else None
        self.fusion_context = dict(matrix=matrix, info=context['info'], detections=detections) if reason == 'ready' else None
        fused_result = CameraFusion.replay_result(self, fused_result, stamp+age, 'exposure_result')
        self.last_replay_key = (stamp, 0, max((packet['stamp'] for packet in getattr(self, 'history', [])
                                           if packet['stamp'] <= stamp+age+1e-9), default=0.))
        timing['publish'] = dict(monotonic_ns=time.monotonic_ns(),
                                ros_s=self.get_clock().now().nanoseconds*1e-9)
        fused_result['timing'] = dict(timing)
        self.detection_pub.publish(String(data=json.dumps(detection_result, allow_nan=False)))
        self.pipeline.record('detections_published', stamp, self.epoch, reason)
        self.fused_pub.publish(String(data=json.dumps(fused_result, allow_nan=False)))
        timing['annotation_start'] = CameraFusion.timing_point(self)
        for projection in context['projections']:
            x, y, w, h = map(round, projection['bbox'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 190, 0), 1)
            cv2.putText(image, 'L' + str(projection['id']), (x, max(14, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 190, 0), 1)
        for region in detection_result['reacquisition_rois']:
            x, y, w, h = region['region']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 150, 255), 1)
            cv2.putText(image, 'KF search '+str(region['track_id']), (x, max(14, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 150, 255), 1)
        for detection in detections:
            x, y, w, h = map(round, detection['bbox'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (40, 230, 60), 2)
            label = f"NN {detection['class_name']} {detection['confidence']:.2f} [unverified]"
            if detection.get('subtype_ambiguous'):
                label += ' subtype?'
            cv2.putText(image, label, (max(0, x), min(msg.height-8, y+h+15)), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 230, 60), 1)
        cv2.putText(image, f'{reason} | NN {elapsed:.0f} ms | XYZ updates {fused_result["coordinate_updates"]}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
        annotated = Image()
        annotated.header = msg.header
        annotated.height, annotated.width = image.shape[:2]
        annotated.encoding, annotated.step = 'bgr8', image.shape[1]*3
        # Humble's bytes setter validates every byte twice in Python. The typed
        # array fast path preserves identical pixels/CDR with bounded C copies.
        annotated.data = array('B', np.ascontiguousarray(image).tobytes())
        timing['annotation_ready'] = CameraFusion.timing_point(self)
        self.image_pub.publish(annotated)
        timing['image_publish_complete'] = CameraFusion.timing_point(self)
        self.publish_markers(msg.header.stamp, fused_result['frame_id'], tracks)
        timing['publish_complete'] = CameraFusion.timing_point(self)
        self.processed += 1
        self.last_inference_ms, self.last_result_stamp, self.last_reason = elapsed, stamp, reason
        self.last_timing = dict(timing)
        self.last_result_wall_ns = time.monotonic_ns()

    def replay_result(self, seed, now, kind):
        result = dict(seed)
        tracks = []
        for track in seed['tracks']:
            if track['fusion_update']['accepted']:
                try:
                    track = replay_camera_track(track, [packet for packet in getattr(self, 'history', [])
                                                        if packet['stamp'] <= now+1e-9])
                except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                    track = dict(track, temporal_fusion=dict(valid=False, reason=str(error)))
            tracks.append(track)
        valid = [track for track in tracks if track.get('temporal_fusion', {}).get('valid') is True]
        state_stamp = max((track['current_state_stamp'] for track in valid), default=seed['stamp'])
        result.update(tracks=tracks, publication_kind=kind, current_state_stamp=state_stamp,
                      last_measurement_stamp=min((track['last_measurement_stamp'] for track in valid), default=None),
                      source_exposure_age_s=now-seed['stamp'], state_age_s=now-state_stamp,
                      result_age_s=now-seed['stamp'], current_coordinate_updates=len(valid))
        return result

    def publish_replay(self, now):
        self.check_calibration()
        if self.fusion_seed is None:
            return
        health = self.health(now)
        if not health['ready']:
            self.invalidate_fusion(health['health_reason'])
            return
        if not 0 <= now-self.fusion_seed['stamp'] <= self.max_age:
            self.invalidate_fusion('stale_camera_result')
            return
        context = self.fusion_context
        if context is not None:
            try:
                self.fusion_seed = reanchor_camera_seed(
                    self.fusion_seed, [packet for packet in self.history if packet['stamp'] <= now+1e-9],
                    context['matrix'], context['info'], context['detections'], self.tolerance,
                    self.bearing_config, self.coordinate_fusion)
            except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                self.error = 'reanchor: '+str(error)
                self.invalidate_fusion('reanchor_error')
                self.failures += 1
                self.get_logger().warning(self.error, throttle_duration_sec=5)
                return
        key = (self.fusion_seed['stamp'], self.fusion_seed.get('fusion_revision', 0),
               max((packet['stamp'] for packet in self.history
                                              if packet['stamp'] <= now+1e-9), default=0.))
        if key == self.last_replay_key:
            return
        self.last_replay_key = key
        result = self.replay_result(self.fusion_seed, now, 'lidar_replay')
        result['timing'] = dict(result.get('timing', {}), replay_publish=self.timing_point())
        self.fused_pub.publish(String(data=json.dumps(result, allow_nan=False)))
        self.replay_publications += 1

    def publish_markers(self, stamp, frame, tracks):
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers = [clear]
        for track in tracks:
            marker = Marker()
            marker.header.stamp, marker.header.frame_id = stamp, frame
            marker.ns, marker.id, marker.type = 'fused_identity', int(track['id']), Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = track['position']
            marker.pose.position.z += max(7.0, float(track['size'][2]))
            marker.pose.orientation.w = 1.
            marker.scale.z = 1.5
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = .2, 1., .95, 1.
            x, y, z = track['position']
            ambiguity = ' [subtype ambiguous]' if track.get('subtype_ambiguous') else ' [subtype unverified]'
            source = 'FUSED' if track['fusion_update']['accepted'] else 'LIDAR'
            marker.text = f"ID {track['id']} {track['class_name']} {track['confidence']:.0%}{ambiguity}\n{source} XYZ {x:.2f} {y:.2f} {z:.2f} m"
            marker.lifetime = Duration(seconds=.7).to_msg()
            markers.append(marker)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_status(self):
        now = self.check_clock()
        self.check_calibration()
        health = self.health(now)
        if not health['ready'] and self.fusion_seed is not None:
            self.invalidate_fusion(health['health_reason'])
        self.status_pub.publish(String(data=json.dumps(dict(stamp=now, **health,
            error=self.error, fusion_reason=self.last_reason, processed_frames=self.processed,
            superseded_frames=self.superseded, failures=self.failures, inference_ms=self.last_inference_ms,
            calibrated=bool(self.get_parameter('calibration_valid').value),
            calibration_revision=self.calibration_revision,
            subtype_validation='unvalidated_for_current_scene',
            model=self.network.model if self.network else None, epoch=self.epoch,
            requested_provider=self.requested_provider,
            active_providers=getattr(self.network, 'providers', []),
            provider_options=getattr(self.network, 'provider_options', {}),
            replay_publications=self.replay_publications, timing=self.last_timing,
            kalman_reacquisition_enabled=self.reacquisition_enabled,
            geometry_error=self.geometry_error, geometry_tf_bounds=self.geometry_tf_bounds,
            geometry_wait_frames=self.geometry_wait_frames, geometry_wait_timeouts=self.geometry_wait_timeouts,
            pipeline=self.pipeline_status(),
            completion_dispatch='executor_guard_condition', camera_evidence_max_age_s=self.max_age), allow_nan=False)))

    def pipeline_status(self):
        pending, inflight = self.latest is not None, self.future is not None
        def frame(message, timing):
            return dict(stamp=stamp_seconds(message.header.stamp),
                        image_receive=timing.get('image_receive'),
                        worker_start=timing.get('worker_start'), worker_end=timing.get('worker_end'),
                        tf_wait_start=timing.get('tf_wait_start'), clock_wait_start=timing.get('clock_wait_start'))
        context = self.future[1] if inflight else None
        return dict(schema_version=1, counter_scope='node_lifetime',
                    counters=dict(self.pipeline.counters), event_sequence=self.pipeline.sequence,
                    recent_events=list(self.pipeline.events), pending=pending, inflight=inflight,
                    pending_frame=frame(self.latest, self.latest_timing) if pending else None,
                    inflight_frame=frame(context['message'], context['timing']) if inflight else None,
                    input_qos=dict(reliability='reliable' if self.camera_reliable else 'best_effort',
                                   history='keep_last', depth=5),
                    accounting_scope='callback receipts only; recorder images do not prove this subscriber received them')


def main():
    cv2.setNumThreads(1)
    rclpy.init()
    node = CameraFusion()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.worker.shutdown(wait=True, cancel_futures=True)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
