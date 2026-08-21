"""Down camera, decoded on the Jetson's hardware JPEG engine.

Replaces usb_cam, which was the single largest source of delay in the landing
stack. Measured on this Orin Nano, 1280x720 MJPEG:

    v4l2 straight to /dev/null      58.3 fps      what the camera can do
    gstreamer + nvv4l2decoder       58.2 fps      this node's pipeline
    usb_cam (mjpeg2rgb)             21   fps      and 618 ms stale

The 618 ms was not network and not the detector — a local subscriber on the
Jetson saw it too, and it did not move when the requested framerate was changed
from 30 to 20 to 10, nor when the compressed stream was switched off. usb_cam
decodes MJPEG on the ARM cores at 21 fps while the camera pushes ~58, so the
V4L2 queue sits permanently full and every frame dequeued is already old.

Two things fix it here:

  nvv4l2decoder     JPEG decode moves to the NVJPG hardware block, so the
                    pipeline keeps up with the camera and nothing queues. It
                    also hands the ~81% CPU that usb_cam was using back to the
                    ArUco detector, which needs it.

  drop=true         the appsinks hold ONE buffer and throw away anything older.
  max-buffers=1     if we ever do fall behind, we lose frames instead of
                    accumulating delay. For landing, a fresh frame matters far
                    more than a complete sequence.

WHY THE COMPRESSED STREAM COSTS NOTHING
---------------------------------------
The camera's native output is already MJPEG. usb_cam decoded it to RGB and then
image_transport re-encoded it back to JPEG for the ground station — paying for
the same picture twice. Here a `tee` takes the camera's JPEG bytes before the
decoder and publishes them as-is. No encode, no quality loss, and it is
literally the original frame.

The cost is that jpeg_quality no longer applies; the size is whatever the
camera produced (~40-90 KB). `compressed_every_n` throttles the frame rate
instead, which is the honest knob for a link budget — dropping to every 3rd
frame is a 3x saving with no loss of picture quality on the frames that do
arrive.

NEVER ASSIGN bytes TO A uint8[] FIELD
-------------------------------------
`msg.data` must be handed an `array.array('B', ...)`, and this is not style.
The setter rosidl generates for a `uint8[]` field takes the value unchanged if
it is already an `array.array`; anything else falls through to a debug-mode
validation that runs

    all(isinstance(v, int) and 0 <= v < 256 for v in value)

over every element. For a 1280x720 BGR frame that is 2,764,800 elements, and it
was MEASURED here at 457 ms per frame — which capped this node at 2.1 fps and
looked exactly like a slow camera. Through array.array the same publish costs
4.7 ms and the node runs at the camera's full 30 fps. Same trap applies to the
compressed stream, and to anyone else in this repo filling a uint8[].

TIMESTAMPS
----------
Frames are stamped when this node pulls them, not from the V4L2 buffer clock.
With a one-deep dropping queue the pull happens within a frame period of
capture, so the stamp is good to ~30 ms — but it is an upper bound on freshness,
not a measurement of the sensor's own pipeline. Do not read a latency number off
this node and call it glass-to-ROS.
"""

from __future__ import annotations

import array
import os

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
# GstApp looks unused and is not. `Gst.parse_launch` hands back an object that
# already IS a GstAppSink, but PyGObject only attaches that class's methods once
# its typelib has been imported — without this line `try_pull_sample` simply
# does not exist on the sink and the first timer tick dies with AttributeError.
# The typelib ships with the JetPack gstreamer plugins, so this adds no package.
from gi.repository import Gst, GstApp  # noqa: E402,F401

import rclpy  # noqa: E402
import yaml  # noqa: E402
from ament_index_python.packages import get_package_share_directory  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,  # noqa: E402
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image  # noqa: E402


def _sensor_qos(depth: int = 1) -> QoSProfile:
    """BEST_EFFORT, one deep — the newest frame or nothing.

    A RELIABLE image publisher retransmits what the subscriber missed, which on
    a busy or lossy link turns dropped frames into growing delay. For a camera
    the old frame has no value once a new one exists.
    """
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


class GstCamera(Node):
    def __init__(self):
        super().__init__('down_camera')
        p = self.declare_parameter

        # Every parameter lives here, not in the launch file — the same rule the
        # rest of flight/ and camera/ follow. usb_cam had to be configured from
        # the launch file only because it was third-party; this node is ours, so
        # that exception is gone.
        p('device', '/dev/video0')
        # Must match the calibration below. Change the resolution and fx, fy,
        # cx, cy all scale, which silently invalidates every range solvePnP
        # produces — recalibrate rather than editing one of these.
        p('width', 1280)
        p('height', 720)
        # The camera offers 60 at this resolution and the hardware decoder keeps
        # up, but 30 is plenty for a descent and leaves the ArUco detector, which
        # runs at ~11 Hz, more of the machine.
        p('framerate', 30)
        p('frame_id', 'down_camera_optical_frame')
        # Ground-station view only. Every 3rd frame of a 30 fps stream is 10 Hz,
        # which is smooth enough to judge framing and exposure while costing a
        # third of the link. 1 sends every frame.
        p('compressed_every_n', 3)
        # Publish the RAW frame every Nth pull. This throttles the detector at
        # the source, which is the only place it can be done: rclpy builds the
        # whole Python message before any subscriber callback runs, so a
        # detector that drops the frame on arrival has already paid for it.
        # MEASURED: at 27 fps the detector sat at 111% of a core and could not
        # drain its own /tf subscription, leaving its transform buffer 100-950
        # ms behind and every marker rejected as a lookup into the future.
        # Every 2nd frame of 27 is ~13 Hz, which is what the landing loop
        # consumes (mpc_landing_node runs at 20 Hz, marker timeout 1.5 s).
        # 1 sends every frame.
        p('image_every_n', 2)
        # Empty means "the calibration shipped with this package". An absolute
        # path overrides it, for a second airframe or a spare camera.
        p('calibration_file', '')
        p('image_topic', '/down_camera/image')
        p('compressed_topic', '/down_camera/image/compressed')
        p('camera_info_topic', '/down_camera/camera_info')

        g = self.get_parameter
        self.width = int(g('width').value)
        self.height = int(g('height').value)
        self.frame_id = str(g('frame_id').value)
        self.every_n = max(1, int(g('compressed_every_n').value))
        self.image_every_n = max(1, int(g('image_every_n').value))
        self._n = 0
        self._raw_n = 0

        self.image_pub = self.create_publisher(
            Image, str(g('image_topic').value), _sensor_qos())
        self.comp_pub = self.create_publisher(
            CompressedImage, str(g('compressed_topic').value), _sensor_qos())
        self.info_pub = self.create_publisher(
            CameraInfo, str(g('camera_info_topic').value), _sensor_qos())

        self.camera_info = self._load_calibration(str(g('calibration_file').value))

        Gst.init(None)
        self.pipeline = self._build_pipeline(str(g('device').value),
                                             int(g('framerate').value))
        self.bgr_sink = self.pipeline.get_by_name('bgrsink')
        self.jpeg_sink = self.pipeline.get_by_name('jpegsink')
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.get_logger().fatal(
                'gstreamer pipeline refused to start — is the camera already '
                'open? `fuser /dev/video0` names whoever has it.')
            raise SystemExit(1)

        # Poll at twice the frame rate so a frame never waits a full period for
        # a tick. Pulling is non-blocking, so an empty tick costs nothing.
        self.create_timer(1.0 / (2.0 * int(g('framerate').value)), self._pump)
        self.create_timer(5.0, self._report)
        self._frames = 0
        self._pulled = 0
        self._last_report = self.get_clock().now()
        self.get_logger().info(
            f'down camera up: {self.width}x{self.height} @'
            f'{int(g("framerate").value)} fps, hardware JPEG decode, '
            f'compressed every {self.every_n} frame(s)')

    # ---------------------------------------------------------------- pipeline
    def _build_pipeline(self, device: str, fps: int) -> Gst.Pipeline:
        """Camera JPEG -> tee -> {published as-is, hardware-decoded to BGR}.

        `leaky=downstream` on both queues matters as much as drop=true on the
        sinks: without it a stalled branch would back-pressure the tee and
        stall the OTHER branch with it, so a slow ground-station link could
        hold up the detector.
        """
        desc = (
            f'v4l2src device={device} io-mode=2 ! '
            f'image/jpeg,width={self.width},height={self.height},'
            f'framerate={fps}/1 ! tee name=t '
            't. ! queue max-size-buffers=1 leaky=downstream ! '
            'appsink name=jpegsink max-buffers=1 drop=true sync=false '
            't. ! queue max-size-buffers=1 leaky=downstream ! '
            'nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! '
            'videoconvert ! video/x-raw,format=BGR ! '
            'appsink name=bgrsink max-buffers=1 drop=true sync=false'
        )
        try:
            return Gst.parse_launch(desc)
        except Exception as exc:  # GLib.Error
            self.get_logger().fatal(
                f'could not build the pipeline ({exc}).\n'
                '  nvv4l2decoder comes from the JetPack gstreamer plugins — '
                'check with `gst-inspect-1.0 nvv4l2decoder`.')
            raise SystemExit(1) from None

    # ------------------------------------------------------------------- clock
    def _pump(self) -> None:
        stamp = self.get_clock().now().to_msg()

        sample = self.jpeg_sink.try_pull_sample(0)
        if sample is not None and self._n % self.every_n == 0:
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.format = 'jpeg'
                msg.data = array.array('B', info.data)
                self.comp_pub.publish(msg)
                buf.unmap(info)
        if sample is not None:
            self._n += 1

        sample = self.bgr_sink.try_pull_sample(0)
        if sample is None:
            return
        # Pull it either way — the appsink is one deep with drop=true, so a
        # frame left unpulled is a frame the pipeline has to discard, and the
        # next one is no fresher for it. Only the publish is skipped.
        self._raw_n += 1
        self._pulled += 1
        if self._raw_n % self.image_every_n != 0:
            return
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return
        try:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height = self.height
            msg.width = self.width
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = self.width * 3
            msg.data = array.array('B', info.data)
            self.image_pub.publish(msg)
        finally:
            buf.unmap(info)

        # camera_info rides with every frame and carries the same stamp, so a
        # subscriber that pairs them by timestamp always finds a match.
        self.camera_info.header.stamp = stamp
        self.camera_info.header.frame_id = self.frame_id
        self.info_pub.publish(self.camera_info)
        self._frames += 1

    def _report(self) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_report).nanoseconds * 1e-9
        if dt > 0:
            self.get_logger().info(
                f'{self._pulled / dt:.1f} fps from the camera, '
                f'{self._frames / dt:.1f} fps published to the detector')
        self._frames = 0
        self._pulled = 0
        self._last_report = now

    # ------------------------------------------------------------ calibration
    def _load_calibration(self, path: str) -> CameraInfo:
        """Read a ROS camera_info yaml into a CameraInfo message.

        A missing or mismatched calibration is not survivable: solvePnP has no
        focal length without it and every range it returns is wrong, so this
        refuses to start rather than fly blind on defaults.
        """
        if not path:
            path = os.path.join(
                get_package_share_directory('aruco_landing'), 'config',
                'down_camera.yaml')
        try:
            with open(path) as handle:
                calib = yaml.safe_load(handle)
        except OSError as exc:
            self.get_logger().fatal(f'no calibration at {path} ({exc})')
            raise SystemExit(1) from None

        if (calib.get('image_width') != self.width
                or calib.get('image_height') != self.height):
            self.get_logger().fatal(
                f'calibration is for {calib.get("image_width")}x'
                f'{calib.get("image_height")} but this node is capturing '
                f'{self.width}x{self.height}. fx, fy, cx and cy all scale with '
                'resolution, so using it anyway would put a silent bias on '
                'every range. Recalibrate, or capture at the calibrated size.')
            raise SystemExit(1)

        info = CameraInfo()
        info.width = self.width
        info.height = self.height
        info.distortion_model = calib.get('distortion_model', 'plumb_bob')
        info.d = [float(v) for v in calib['distortion_coefficients']['data']]
        info.k = [float(v) for v in calib['camera_matrix']['data']]
        info.r = [float(v) for v in calib['rectification_matrix']['data']]
        info.p = [float(v) for v in calib['projection_matrix']['data']]
        self.get_logger().info(
            f'calibration {os.path.basename(path)}: fx={info.k[0]:.1f} '
            f'fy={info.k[4]:.1f} cx={info.k[2]:.1f} cy={info.k[5]:.1f}')
        return info

    def shutdown(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


def main(args=None):
    rclpy.init(args=args)
    node = GstCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
