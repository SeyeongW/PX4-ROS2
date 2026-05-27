import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from cv_bridge import CvBridge
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

        self.bridge = CvBridge()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.offset_pub   = self.create_publisher(Point,           '/perception/aruco_offset',   10)
        self.detected_pub = self.create_publisher(Bool,            '/perception/aruco_detected',  10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(CompressedImage,
                                                   '/perception/aruco_debug/compressed', 10)

        self.create_subscription(Image, image_topic, self._cb, qos)
        self.get_logger().info(f'ArUco detector ready — dict={dict_name}, topic={image_topic}')

    def _cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w  = frame.shape[:2]
        cx, cy = w // 2, h // 2

        gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        detected        = ids is not None and len(ids) > 0

        offset = Point()
        if detected:
            # Use the first detected marker
            pts = corners[0][0]
            mx  = float(np.mean(pts[:, 0]))
            my  = float(np.mean(pts[:, 1]))
            # Normalised offset: -1 (left/top) to +1 (right/bottom)
            offset.x = (mx - cx) / (w / 2.0)
            offset.y = (my - cy) / (h / 2.0)

        if self.publish_debug:
            self._draw(frame, corners, ids, cx, cy, offset, detected, msg.header)

        self.offset_pub.publish(offset)

        det_msg      = Bool()
        det_msg.data = detected
        self.detected_pub.publish(det_msg)

    def _draw(self, frame, corners, ids, cx, cy, offset, detected, header):
        if detected:
            aruco.drawDetectedMarkers(frame, corners, ids)
            mx = int(cx + offset.x * (frame.shape[1] / 2))
            my = int(cy + offset.y * (frame.shape[0] / 2))
            cv2.circle(frame, (mx, my), 6, (0, 255, 0), -1)
            cv2.line(frame, (cx, cy), (mx, my), (0, 220, 255), 2)
            label = f'ID:{ids[0][0]}  dx={offset.x:+.2f} dy={offset.y:+.2f}'
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'SEARCHING...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 60, 255), 2)

        # Centre crosshair (always visible)
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)

        dbg            = self.bridge.cv2_to_compressed_imgmsg(frame)
        dbg.header     = header
        self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
