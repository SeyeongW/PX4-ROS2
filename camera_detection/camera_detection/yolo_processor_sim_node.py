import os
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from ultralytics import YOLO


class YoloProcessorSimNode(Node):
    def __init__(self):
        super().__init__("yolo_processor_sim_node")

        self.declare_parameter("image_topic", "/image_raw/compressed")
        self.declare_parameter("debug_image_topic", "/perception/debug_image/compressed")
        self.declare_parameter("command_topic", "/perception/set_target_id")
        self.declare_parameter("model_path", "yolo11s.pt")
        self.declare_parameter("track_history_len", 30)
        self.declare_parameter("conf_thres", 0.3)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("publish_debug", True)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.debug_image_topic = (
            self.get_parameter("debug_image_topic").get_parameter_value().string_value
        )
        command_topic = self.get_parameter("command_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.track_history_len = (
            self.get_parameter("track_history_len").get_parameter_value().integer_value
        )
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.publish_debug = self.get_parameter("publish_debug").get_parameter_value().bool_value

        if not os.path.isabs(self.model_path):
            self.model_path = os.path.abspath(self.model_path)

        if not os.path.exists(self.model_path):
            self.get_logger().warn(f"Model file not found: {self.model_path}")

        self.model = YOLO(self.model_path, task="detect")
        self.bridge = CvBridge()
        self.track_history = defaultdict(list)

        self.initial_locked_id = None
        self.locked_id = None
        self.last_known_box = None
        self.vx = 0.0
        self.vy = 0.0
        self.lost_count = 0

        self.fps = 30.0
        self.wait_seconds = 2.0
        self.lost_threshold = self.fps * self.wait_seconds

        qos_profile = QoSProfile(depth=1)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            qos_profile,
        )

        self.command_subscription = self.create_subscription(
            Int32,
            command_topic,
            self.command_callback,
            10,
        )

        self.target_center_pub = self.create_publisher(Point, "/perception/target_center", 10)
        self.target_found_pub = self.create_publisher(Bool, "/perception/target_found", 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, self.debug_image_topic, 10)

        self.get_logger().info(f"YOLO Sim Node Started. Subscribing to {image_topic}")

    def command_callback(self, msg: Int32):
        self.get_logger().info(f"Command Received: {msg.data}")
        if msg.data <= -1:
            self.locked_id = None
            self.initial_locked_id = None
            self.lost_count = 0
            self.last_known_box = None
            self.vx = 0.0
            self.vy = 0.0
            self.get_logger().info("Target Unlock.")
            return

        if msg.data <= 0:
            self.locked_id = None
            self.initial_locked_id = None
            self.lost_count = 0
            self.last_known_box = None
            self.vx = 0.0
            self.vy = 0.0
            self.get_logger().info("Target Unlock.")
            return

        self.locked_id = msg.data
        self.initial_locked_id = msg.data
        self.lost_count = 0
        self.last_known_box = None
        self.vx = 0.0
        self.vy = 0.0
        self.get_logger().info(f"Target Locked: {self.locked_id}")

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
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            tracker="botsort.yaml",
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        current_ids = []
        current_boxes = []
        current_confs = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            current_ids = boxes.id.cpu().numpy().astype(int)
            current_boxes = boxes.xyxy.cpu().numpy()
            current_confs = boxes.conf.cpu().numpy()

        for i, track_id in enumerate(current_ids):
            box = current_boxes[i]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

            self.track_history[track_id].append((float(cx), float(cy)))
            if len(self.track_history[track_id]) > self.track_history_len:
                self.track_history[track_id].pop(0)

            if track_id != self.locked_id and self.publish_debug:
                self.draw_styled_box(frame, box, track_id, current_confs[i], (255, 0, 0))

        target_box = None
        target_found = False
        target_conf = 0.0

        if self.locked_id is not None:
            if self.locked_id in current_ids:
                idx = int(np.where(current_ids == self.locked_id)[0][0])
                target_box = current_boxes[idx]
                target_conf = float(current_confs[idx])

                if self.last_known_box is not None:
                    curr_cx = (target_box[0] + target_box[2]) / 2
                    curr_cy = (target_box[1] + target_box[3]) / 2
                    prev_cx = (self.last_known_box[0] + self.last_known_box[2]) / 2
                    prev_cy = (self.last_known_box[1] + self.last_known_box[3]) / 2
                    self.vx = curr_cx - prev_cx
                    self.vy = curr_cy - prev_cy

                self.last_known_box = target_box
                self.lost_count = 0
                target_found = True

                if self.publish_debug:
                    self.draw_styled_box(
                        frame,
                        target_box,
                        self.initial_locked_id,
                        target_conf,
                        (0, 0, 255),
                        is_locked=True,
                        real_id=self.locked_id,
                    )
            else:
                self.lost_count += 1
                if self.lost_count <= self.lost_threshold and self.last_known_box is not None:
                    lx1, ly1, lx2, ly2 = self.last_known_box
                    self.vx *= 0.95
                    self.vy *= 0.95
                    lx1 += self.vx
                    lx2 += self.vx
                    ly1 += self.vy
                    ly2 += self.vy
                    self.last_known_box = np.array([lx1, ly1, lx2, ly2])

                    if self.publish_debug:
                        sx1, sy1, sx2, sy2 = map(int, self.last_known_box)
                        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
                        cv2.putText(
                            frame,
                            "LOST...",
                            (sx1, sy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

                    if len(current_boxes) > 0:
                        pred_area = (lx2 - lx1) * (ly2 - ly1)
                        best_score = 0.0
                        best_id = None
                        best_box = None
                        best_conf = 0.0

                        for i, box in enumerate(current_boxes):
                            bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                            if lx1 < bcx < lx2 and ly1 < bcy < ly2:
                                curr_area = (box[2] - box[0]) * (box[3] - box[1])
                                sim = min(pred_area, curr_area) / max(pred_area, curr_area)
                                if sim > best_score:
                                    best_score = sim
                                    best_id = int(current_ids[i])
                                    best_box = box
                                    best_conf = float(current_confs[i])

                        if best_score > 0.5 and best_id is not None:
                            self.get_logger().warn(
                                f"Re-locked: {self.locked_id} -> {best_id}"
                            )
                            self.locked_id = best_id
                            target_box = best_box
                            target_conf = best_conf
                            self.last_known_box = target_box
                            self.lost_count = 0
                            target_found = True

                            if self.publish_debug:
                                self.draw_styled_box(
                                    frame,
                                    target_box,
                                    self.initial_locked_id,
                                    target_conf,
                                    (0, 0, 255),
                                    is_locked=True,
                                    real_id=self.locked_id,
                                )
                elif self.lost_count > self.lost_threshold:
                    self.locked_id = None
                    self.initial_locked_id = None
                    self.last_known_box = None

        if target_found and target_box is not None:
            cx = (target_box[0] + target_box[2]) / 2.0
            cy = (target_box[1] + target_box[3]) / 2.0
            area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

            err_x = (cx - (w / 2.0)) / (w / 2.0)
            err_y = (cy - (h / 2.0)) / (h / 2.0)
            norm_area = area / (w * h)

            pt_msg = Point()
            pt_msg.x = float(err_x)
            pt_msg.y = float(err_y)
            pt_msg.z = float(norm_area)
            self.target_center_pub.publish(pt_msg)
            self.target_found_pub.publish(Bool(data=True))
        else:
            self.target_found_pub.publish(Bool(data=False))
            pt_msg = Point()
            pt_msg.x = float("nan")
            pt_msg.y = float("nan")
            pt_msg.z = float("nan")
            self.target_center_pub.publish(pt_msg)

        if self.publish_debug:
            self.publish_debug_image(frame, msg)

    def draw_styled_box(self, frame, box, display_id, conf, color, is_locked=False, real_id=None):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"id:{display_id} person {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y1 = y1 - 20 if y1 - 20 > 0 else y1
        label_y2 = y1 if y1 - 20 > 0 else y1 + 20
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, label_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        if is_locked:
            track_key = real_id if real_id is not None else display_id
            if track_key in self.track_history:
                pts = np.array(self.track_history[track_key], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

    def publish_debug_image(self, frame, msg: CompressedImage):
        success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not success:
            return
        debug_msg = CompressedImage()
        debug_msg.header = msg.header
        debug_msg.format = "jpeg"
        debug_msg.data = encoded.tobytes()
        self.debug_image_pub.publish(debug_msg)


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
