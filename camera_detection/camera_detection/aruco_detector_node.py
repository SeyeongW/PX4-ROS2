import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        self.declare_parameter('image_topic', '/down_camera/image')
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('publish_debug', True)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        dict_name   = self.get_parameter('aruco_dict').get_parameter_value().string_value
        self.publish_debug = self.get_parameter('publish_debug').get_parameter_value().bool_value

        dict_map = {
            'DICT_4X4_50':  aruco.DICT_4X4_50,
            'DICT_5X5_100': aruco.DICT_5X5_100,
            'DICT_6X6_250': aruco.DICT_6X6_250,
        }
        adict  = aruco.getPredefinedDictionary(dict_map.get(dict_name, aruco.DICT_4X4_50))
        params = aruco.DetectorParameters()

        # Support OpenCV >= 4.7 (ArucoDetector) and older API
        try:
            self._detector = aruco.ArucoDetector(adict, params)
            self._detect = lambda g: self._detector.detectMarkers(g)
        except AttributeError:
            self._detect = lambda g: aruco.detectMarkers(g, adict, parameters=params)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.offset_pub   = self.create_publisher(Point,           '/perception/aruco_offset',   10)
        self.detected_pub = self.create_publisher(Bool,            '/perception/aruco_detected',  10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(CompressedImage,
                                                   '/perception/aruco_debug/compressed', 10)

        self.create_subscription(Image, image_topic, self._cb, qos)
        self.get_logger().info(f'ArUco detector ready — dict={dict_name}, topic={image_topic}')

    @staticmethod
    def _imgmsg_to_bgr(msg: Image):
        # Manual sensor_msgs/Image -> BGR ndarray. Avoids cv_bridge, whose
        # compiled boost module breaks under NumPy 2.x (_ARRAY_API not found).
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        h, w = msg.height, msg.width
        enc = msg.encoding
        if enc in ('rgb8', 'bgr8'):
            img = buf.reshape(h, w, 3)
            if enc == 'rgb8':
                img = img[:, :, ::-1]              # RGB -> BGR
            return np.ascontiguousarray(img)
        if enc in ('mono8', '8UC1'):
            return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
        # fallback: assume 3-channel packed
        return np.ascontiguousarray(buf.reshape(h, w, 3))

    @staticmethod
    def _quad_area(pts):
        # Shoelace formula on the 4 detected corners — apparent pixel area,
        # used only to rank which of several co-located markers is most
        # confidently resolved (bigger = less corner-jitter as a % of size).
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _cb(self, msg: Image):
        frame = self._imgmsg_to_bgr(msg)
        h, w  = frame.shape[:2]
        cx, cy = w // 2, h // 2

        gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        detected        = ids is not None and len(ids) > 0

        offset = Point()
        sel = -1
        if detected:
            # The platform carries THREE concentric ArUco markers of different
            # physical sizes sharing one centre (aruco_platform/model.sdf), so
            # whichever ones are resolvable at the current altitude all report
            # ~the same centroid. Pick the one with the largest apparent pixel
            # area (most confidently/robustly detected) rather than always the
            # first ID in the dictionary's detection order.
            sel = int(np.argmax([self._quad_area(c[0]) for c in corners]))
            pts = corners[sel][0]
            mx  = float(np.mean(pts[:, 0]))
            my  = float(np.mean(pts[:, 1]))
            # Normalised offset: -1 (left/top) to +1 (right/bottom)
            offset.x = (mx - cx) / (w / 2.0)
            offset.y = (my - cy) / (h / 2.0)

        if self.publish_debug:
            self._draw(frame, corners, ids, sel, cx, cy, offset, detected, msg.header)

        self.offset_pub.publish(offset)

        det_msg      = Bool()
        det_msg.data = detected
        self.detected_pub.publish(det_msg)

    def _draw(self, frame, corners, ids, sel, cx, cy, offset, detected, header):
        if detected:
            aruco.drawDetectedMarkers(frame, corners, ids)
            mx = int(cx + offset.x * (frame.shape[1] / 2))
            my = int(cy + offset.y * (frame.shape[0] / 2))
            cv2.circle(frame, (mx, my), 6, (0, 255, 0), -1)
            cv2.line(frame, (cx, cy), (mx, my), (0, 220, 255), 2)
            label = (f'ID:{np.ravel(ids)[sel]} (of {len(ids)} seen)  '
                     f'dx={offset.x:+.2f} dy={offset.y:+.2f}')
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'SEARCHING...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 60, 255), 2)

        # Centre crosshair (always visible)
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)

        ok, enc = cv2.imencode('.jpg', frame)
        if ok:
            dbg = CompressedImage()
            dbg.header = header
            dbg.format = 'jpeg'
            dbg.data = enc.tobytes()
            self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
