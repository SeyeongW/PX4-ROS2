"""rtsp_bridge_node — SIYI A8 mini RTSP stream -> ROS 2 Image + CameraInfo.

The missing link in the real-flight chain. The gimbal camera publishes H.264
over RTSP; `aruco_landing`'s solvePnP needs `sensor_msgs/Image` **and** a
`CameraInfo` carrying real intrinsics. This node does the three jobs in between:

    rtspsrc ──► depay ──► parse ──► decode ──► convert ──► appsink
                                                             │
                                         BGR frame ──► /gimbal_camera/image
                                         intrinsics ──► /gimbal_camera/camera_info

WHY NATIVE GStreamer AND NOT cv2.VideoCapture(CAP_GSTREAMER)
------------------------------------------------------------
Because it does not work here: the installed OpenCV reports `GStreamer: NO`, so
the CAP_GSTREAMER backend silently falls back and the capture never opens. The
`gi` bindings are present, so the pipeline is built directly. That is also what
lets the decoder be chosen at runtime — see below.

DECODER SELECTION
-----------------
On a Jetson the H.264 decode belongs on the hardware block (`nvv4l2decoder`);
burning a CPU core on it starves the detector that is the point of the flight.
On a desktop that element does not exist and the pipeline must fall back to
`avdec_h264`. The node probes the registry rather than making the operator
choose, and says which one it picked.

CALIBRATION IS NOT OPTIONAL
---------------------------
A `CameraInfo` full of zeros makes solvePnP return garbage that LOOKS like a
pose, which is worse than no pose at all. If no calibration file is given, this
node publishes a clearly-marked PLACEHOLDER derived from the image size and
warns on a loop. Calibrate the real A8 mini before trusting a range.

ALL PARAMETERS ARE DECLARED HERE, IN `_declare`. The launch file passes none.

Interfaces
----------
Publishes  <camera_name>/image        sensor_msgs/Image      (bgr8)
           <camera_name>/camera_info  sensor_msgs/CameraInfo
           ~/status                   std_msgs/String
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst  # noqa: E402  (must follow require_version)


class RtspBridgeNode(Node):
    def __init__(self):
        super().__init__('rtsp_bridge_node')
        self._declare()
        g = self.get_parameter
        self.url = str(g('rtsp_url').value)
        self.camera_name = str(g('camera_name').value)
        self.frame_id = str(g('frame_id').value)
        self.latency_ms = int(g('latency_ms').value)
        self.protocol = str(g('rtsp_protocol').value)
        self.reconnect_s = float(g('reconnect_period_s').value)
        self.calib_file = str(g('camera_info_file').value)
        self.force_decoder = str(g('force_decoder').value)

        self.image_pub = self.create_publisher(Image, f'{self.camera_name}/image', 10)
        self.info_pub = self.create_publisher(
            CameraInfo, f'{self.camera_name}/camera_info', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)

        self._frames = 0
        self._last_frame_t = 0.0
        self._width = self._height = 0
        self._info: CameraInfo | None = None
        self._calibrated = False
        self._load_calibration()

        Gst.init(None)
        self.decoder = self.force_decoder or self._pick_decoder()
        self.pipeline = None
        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()
        self._start_pipeline()

        self.create_timer(self.reconnect_s, self._watchdog)
        self.create_timer(2.0, self._publish_status)

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these may be set. The launch file passes none."""
        p = self.declare_parameter
        # SIYI's factory stream. video1 is RGB, video2 is thermal on units that
        # have it. Same IP as the control link, different port and protocol.
        p('rtsp_url', 'rtsp://192.168.144.25:8554/video1')
        # Topic prefix. Deliberately the SAME name the SITL gimbal camera uses,
        # so the detector and the whole perception chain do not care which one
        # is feeding them.
        p('camera_name', '/gimbal_camera')
        p('frame_id', 'gimbal_camera_optical_frame')
        # Jitter buffer. Lower = fresher frames but more dropouts; every ms here
        # is added latency between the marker moving and the MPC seeing it.
        p('latency_ms', 100)
        # TCP survives lossy wifi; UDP is lower latency on a clean wired link.
        p('rtsp_protocol', 'tcp')
        # How often to check the stream is alive and rebuild it if not. RTSP
        # links drop; treating that as fatal would ground the vehicle.
        p('reconnect_period_s', 3.0)
        # Intrinsics from `ros2 run camera_calibration cameracalibrator`.
        # Empty = publish a placeholder and warn (see the module docstring).
        p('camera_info_file', '')
        # Override the automatic Jetson/desktop decoder choice.
        p('force_decoder', '')

    # ---------------------------------------------------------------- helpers
    def _pick_decoder(self) -> str:
        """Hardware decode where it exists, software where it does not."""
        registry = Gst.Registry.get()
        for element in ('nvv4l2decoder', 'avdec_h264'):
            if registry.find_feature(element, Gst.ElementFactory.__gtype__):
                self.get_logger().info(
                    f'H.264 decoder: {element}'
                    + (' (hardware)' if element.startswith('nv') else ' (software)'))
                return element
        self.get_logger().error(
            'no H.264 decoder found — install gstreamer1.0-libav '
            '(avdec_h264) or run on a Jetson (nvv4l2decoder)')
        return 'avdec_h264'

    def _load_calibration(self) -> None:
        if not self.calib_file:
            return
        try:
            import yaml
            with open(self.calib_file) as handle:
                data = yaml.safe_load(handle)
            info = CameraInfo()
            info.width = int(data['image_width'])
            info.height = int(data['image_height'])
            info.distortion_model = data.get('distortion_model', 'plumb_bob')
            info.d = list(data['distortion_coefficients']['data'])
            info.k = list(data['camera_matrix']['data'])
            info.r = list(data.get('rectification_matrix', {}).get(
                'data', [1, 0, 0, 0, 1, 0, 0, 0, 1]))
            info.p = list(data.get('projection_matrix', {}).get(
                'data', info.k[:3] + [0.0] + info.k[3:6] + [0.0] + info.k[6:9] + [0.0]))
            self._info = info
            self._calibrated = True
            self.get_logger().info(f'intrinsics loaded from {self.calib_file}: '
                                   f'fx={info.k[0]:.1f} fy={info.k[4]:.1f}')
        except Exception as exc:
            self.get_logger().error(
                f'could not read {self.calib_file} ({exc}) — falling back to a '
                f'PLACEHOLDER calibration, which will give WRONG ranges')

    def _placeholder_info(self, width: int, height: int) -> CameraInfo:
        """A guess, marked as one. Better than zeros, far worse than a calibration.

        Assumes a 60 deg horizontal FOV and no distortion. solvePnP will return
        a plausible-looking pose whose SCALE is wrong, which is exactly the
        failure mode that is hard to spot in flight — hence the repeated warning.
        """
        fx = fy = width / (2.0 * 0.5774)          # tan(30 deg)
        cx, cy = width / 2.0, height / 2.0
        info = CameraInfo()
        info.width, info.height = width, height
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    # ---------------------------------------------------------------- pipeline
    def _pipeline_description(self) -> str:
        # `drop=true max-buffers=1` on the sink is what keeps this REAL-TIME: a
        # backed-up queue would hand the detector frames from seconds ago, and a
        # landing controller acting on stale vision is worse than one with none.
        return (
            f'rtspsrc location={self.url} protocols={self.protocol} '
            f'latency={self.latency_ms} ! '
            'rtph264depay ! h264parse ! '
            f'{self.decoder} ! videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink name=sink emit-signals=true sync=false '
            'drop=true max-buffers=1')

    def _start_pipeline(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        description = self._pipeline_description()
        self.get_logger().info(f'opening {self.url}')
        try:
            self.pipeline = Gst.parse_launch(description)
        except GLib.Error as exc:
            self.get_logger().error(f'could not build the pipeline: {exc}')
            self.pipeline = None
            return
        sink = self.pipeline.get_by_name('sink')
        sink.connect('new-sample', self._on_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_sample(self, sink) -> int:
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value('width'), caps.get_value('height')
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            stamp = self.get_clock().now().to_msg()
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height, msg.width = height, width
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = width * 3
            msg.data = bytes(info.data)
            self.image_pub.publish(msg)

            if self._info is None or (width, height) != (self._width, self._height):
                self._width, self._height = width, height
                if not self._calibrated:
                    self._info = self._placeholder_info(width, height)
            cam = self._info
            cam.header.stamp = stamp
            cam.header.frame_id = self.frame_id
            self.info_pub.publish(cam)

            self._frames += 1
            self._last_frame_t = self.get_clock().now().nanoseconds * 1e-9
        finally:
            buf.unmap(info)
        return Gst.FlowReturn.OK

    # --------------------------------------------------------------- watchdog
    def _watchdog(self) -> None:
        """Rebuild the pipeline if frames stopped. RTSP links drop; that is normal."""
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.pipeline is None:
            self._start_pipeline()
            return
        if self._last_frame_t and (now - self._last_frame_t) > self.reconnect_s:
            self.get_logger().warn(
                f'no frame for {now - self._last_frame_t:.1f} s — reopening the stream')
            self._start_pipeline()

    def _publish_status(self) -> None:
        calib = 'calibrated' if self._calibrated else 'PLACEHOLDER intrinsics'
        self.status_pub.publish(String(
            data=f'{self.url} | {self._width}x{self._height} | '
                 f'{self._frames} frames | {self.decoder} | {calib}'))
        if not self._calibrated:
            self.get_logger().warn(
                'publishing PLACEHOLDER intrinsics — solvePnP ranges will be '
                'WRONG. Calibrate the camera and set camera_info_file.',
                throttle_duration_sec=30.0)
        if self._frames == 0:
            self.get_logger().warn(
                f'no frames yet from {self.url} — check the vehicle network, '
                f'and that the camera is streaming (try: gst-launch-1.0 '
                f'rtspsrc location={self.url} ! fakesink)',
                throttle_duration_sec=10.0)

    def destroy_node(self):
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        self._loop.quit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RtspBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
