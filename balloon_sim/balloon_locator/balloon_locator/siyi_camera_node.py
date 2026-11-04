#!/usr/bin/env python3
"""Bounded RTSP capture with one latest-frame slot and optional measured calibration.

OpenCV does not expose this RTSP stream's exposure clock. Stamps are local decode
completion minus measured latency_sec, not verified hardware exposure timestamps.
"""
import math
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32

from .camera_calibration import CameraCalibration
from .ros_utils import bgr_to_image


class SiyiCameraNode(Node):
    def __init__(self):
        super().__init__('siyi_camera_node')
        defaults = dict(rtsp_url='rtsp://192.168.144.25:8554/main.264',
            image_topic='/camera/image_raw', camera_info_topic='/camera/camera_info',
            frame_id='camera', latency_sec=0., calibration_file='', calibration_zoom=1.,
            zoom_max_age_sec=2., open_timeout_sec=3., read_timeout_sec=1., reconnect_interval_sec=1.)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        self.url, self.frame_id = str(p('rtsp_url')), str(p('frame_id'))
        self.latency = float(p('latency_sec'))
        self.open_timeout, self.read_timeout = float(p('open_timeout_sec')), float(p('read_timeout_sec'))
        self.reconnect_interval = float(p('reconnect_interval_sec'))
        self.calibration_zoom, self.zoom_max_age = float(p('calibration_zoom')), float(p('zoom_max_age_sec'))
        timing = (self.open_timeout, self.read_timeout, self.reconnect_interval, self.zoom_max_age)
        if (not all(math.isfinite(v) and v > 0 for v in timing)
                or not math.isfinite(self.latency) or self.latency < 0
                or not math.isfinite(self.calibration_zoom) or self.calibration_zoom < 1.
                or not self.frame_id or not self.url):
            raise ValueError('invalid_camera_configuration')
        self.calibration = CameraCalibration(p('calibration_file')) if p('calibration_file') else None
        self.pub = self.create_publisher(Image, str(p('image_topic')), qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, str(p('camera_info_topic')), qos_profile_sensor_data)
        self.create_subscription(Float32, '/gimbal/zoom', self._on_zoom, 10)
        self.zoom = None
        self.zoom_received = 0.
        self._zoom_generation = 0
        self._latest = None
        self._frame_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_stamp = None
        self.create_timer(.02, self._publish_latest)
        self._thread = threading.Thread(target=self._loop, name='siyi_capture', daemon=True)
        self._thread.start()

    def _on_zoom(self, message):
        value = float(message.data) if math.isfinite(message.data) and message.data >= 1. else None
        now = time.monotonic()
        with self._publish_lock, self._frame_lock:
            if value != self.zoom or now - self.zoom_received > self.zoom_max_age:
                self._zoom_generation += 1
                self._latest = None
            self.zoom = value
            self.zoom_received = now

    def _zoom_matches(self):
        return (self.zoom is not None and abs(self.zoom - self.calibration_zoom) <= .05
                and 0 <= time.monotonic() - self.zoom_received <= self.zoom_max_age)

    def _loop(self):
        while not self._stop_event.is_set() and rclpy.ok():
            capture = None
            try:
                # Open-only parameters supported by OpenCV 4.5.4's FFmpeg backend.
                # Do not silently retry without timeouts if a backend rejects them.
                capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, max(1, round(self.open_timeout * 1000)),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, max(1, round(self.read_timeout * 1000))])
                if not capture.isOpened():
                    raise RuntimeError('rtsp_open_failed')
                while not self._stop_event.is_set() and rclpy.ok():
                    with self._frame_lock:
                        generation, zoom_ok = self._zoom_generation, self._zoom_matches()
                    ok, frame = capture.read()
                    if not ok or self._stop_event.is_set():
                        break
                    received = self.get_clock().now().nanoseconds
                    with self._frame_lock:
                        if self.calibration is not None and (not zoom_ok
                                or generation != self._zoom_generation or not self._zoom_matches()):
                            continue
                        self._latest = (frame, received, time.monotonic(), generation)
            except (cv2.error, RuntimeError) as error:
                self.get_logger().warn(f'RTSP capture failed: {type(error).__name__}', throttle_duration_sec=3.)
            finally:
                if capture is not None:
                    capture.release()
            if self._stop_event.wait(self.reconnect_interval):
                break

    def _publish_latest(self):
        if self._stop_event.is_set():
            return
        with self._frame_lock:
            latest, self._latest = self._latest, None
        if latest is None:
            return
        frame, received, wall_received, generation = latest
        now = self.get_clock().now().nanoseconds
        stamp_ns = received - round(self.latency * 1e9)
        if (stamp_ns <= 0 or stamp_ns > now or time.monotonic() - wall_received > .5
                or self._last_stamp is not None and stamp_ns <= self._last_stamp):
            return
        try:
            if self.calibration is not None:
                with self._frame_lock:
                    if generation != self._zoom_generation or not self._zoom_matches():
                        return
                frame = self.calibration.rectify(frame)
            stamp = Time(nanoseconds=stamp_ns).to_msg()
            image = bgr_to_image(frame, stamp, self.frame_id)
        except (ValueError, cv2.error) as error:
            self.get_logger().warn(f'Camera frame rejected: {error}', throttle_duration_sec=3.)
            return
        with self._publish_lock:
            if self._stop_event.is_set():
                return
            if self.calibration is not None:
                with self._frame_lock:
                    if generation != self._zoom_generation or not self._zoom_matches():
                        return
                self.info_pub.publish(self.calibration.message(stamp, self.frame_id))
            self.pub.publish(image)
            self._last_stamp = stamp_ns

    def stop(self):
        with self._publish_lock:
            self._stop_event.set()
        self._thread.join(timeout=self.open_timeout + self.read_timeout + .5)
        if self._thread.is_alive():
            raise RuntimeError('camera_capture_did_not_stop_within_configured_timeouts')

    def destroy_node(self):
        self.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SiyiCameraNode()
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
