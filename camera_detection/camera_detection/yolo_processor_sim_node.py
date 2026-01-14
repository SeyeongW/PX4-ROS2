import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from ultralytics import YOLO


class YoloProcessorSimNode(Node):
    def __init__(self):
        super().__init__("yolo_processor_sim_node")

        self.declare_parameter("image_topic", "/camera/image_raw/compressed")
        self.declare_parameter("debug_image_topic", "/perception/debug_image/compressed")
        self.declare_parameter("model_path", "yolo11n.engine")
        self.declare_parameter("conf_thres", 0.3)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("publish_debug", True)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.debug_image_topic = (
            self.get_parameter("debug_image_topic").get_parameter_value().string_value
        )
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.publish_debug = self.get_parameter("publish_debug").get_parameter_value().bool_value

        if not os.path.isabs(self.model_path):
            self.model_path = os.path.abspath(self.model_path)

        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Missing model: {self.model_path}")

        self.model = YOLO(self.model_path, task="detect")
        self.bridge = CvBridge()

        qos_profile = QoSProfile(depth=1)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            qos_profile,
        )

        self.target_center_pub = self.create_publisher(Point, "/perception/target_center", 10)
        self.target_found_pub = self.create_publisher(Bool, "/perception/target_found", 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, self.debug_image_topic, 10)

        self.get_logger().info("YOLO Sim Node Started.")

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            self.get_logger().error(f"Image decode failed: {exc}")
            return

        if frame is None:
            self.get_logger().warn("Empty image frame.")
            return

        h, w, _ = frame.shape
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        best_box = None
        best_conf = -1.0

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])
            best_box = coords[best_idx]

        if best_box is None:
            self.publish_target(np.nan, np.nan, np.nan)
            self.target_found_pub.publish(Bool(data=False))
        else:
            cx = (best_box[0] + best_box[2]) * 0.5
            cy = (best_box[1] + best_box[3]) * 0.5
            area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])

            offset_x = (cx - (w * 0.5)) / (w * 0.5)
            offset_y = (cy - (h * 0.5)) / (h * 0.5)
            area_norm = area / float(w * h)

            self.publish_target(offset_x, offset_y, area_norm)
            self.target_found_pub.publish(Bool(data=True))

            if self.publish_debug:
                frame = self.draw_debug(frame, best_box, offset_x, offset_y, best_conf)

        if self.publish_debug:
            self.publish_debug_image(frame, msg)

    def publish_target(self, x: float, y: float, z: float):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        self.target_center_pub.publish(point)

    def publish_debug_image(self, frame, msg: CompressedImage):
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not success:
            return
        debug_msg = CompressedImage()
        debug_msg.header = msg.header
        debug_msg.format = "jpeg"
        debug_msg.data = encoded.tobytes()
        self.debug_image_pub.publish(debug_msg)

    def draw_debug(self, frame, box, offset_x, offset_y, conf):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"off=({offset_x:.2f},{offset_y:.2f}) conf={conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame


def main(args=None):
    rclpy.init(args=args)
    node = YoloProcessorSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
