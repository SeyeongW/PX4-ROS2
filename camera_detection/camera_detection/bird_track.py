import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from ultralytics import YOLO


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")
        self._init_parameters()
        self._init_camera()
        self._init_model()
        self._init_ros_io()
        self._init_state()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info("YOLO detector node started. Trail visualization enabled.")

    # ---------- initialization ----------

    def _init_parameters(self):
        default_model = self._resolve_default_model()

        self.declare_parameter("device_id", 0)
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("confidence", 0.6)
        self.declare_parameter("target_class", 14)
        self.declare_parameter("tracker", "botsort.yaml")
        self.declare_parameter("trail_length", 30)
        self.declare_parameter("trail_timeout", 30)
        self.declare_parameter("trail_thickness", 2)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("timer_period", 0.033)

        gp = self.get_parameter
        self.device_id = gp("device_id").get_parameter_value().integer_value
        self.model_path = gp("model_path").get_parameter_value().string_value
        self.confidence = gp("confidence").get_parameter_value().double_value
        self.target_class = gp("target_class").get_parameter_value().integer_value
        self.tracker = gp("tracker").get_parameter_value().string_value
        self.trail_length = gp("trail_length").get_parameter_value().integer_value
        self.trail_timeout = gp("trail_timeout").get_parameter_value().integer_value
        self.trail_thickness = gp("trail_thickness").get_parameter_value().integer_value
        self.frame_width = gp("frame_width").get_parameter_value().integer_value
        self.frame_height = gp("frame_height").get_parameter_value().integer_value
        self.jpeg_quality = gp("jpeg_quality").get_parameter_value().integer_value
        self.timer_period = gp("timer_period").get_parameter_value().double_value

    def _resolve_default_model(self):
        # search standard workspace locations for the engine file
        paths_to_search = [
            os.path.expanduser("~/ros2_ws/PX4-ROS2/yolo11n.engine"),
            os.path.expanduser("~/ros2_ws/src/PX4-ROS2/yolo11n.engine"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolo11n.engine")),
        ]
        for path in paths_to_search:
            if os.path.exists(path):
                return path
        return "yolo11n.engine"

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    def _init_model(self):
        self.model = YOLO(self.model_path, task="detect")

    def _init_ros_io(self):
        self.bridge = CvBridge()
        self.target_found_pub = self.create_publisher(Bool, "/perception/target_found", 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, "/perception/yolo_result/compressed", 10)

    def _init_state(self):
        # per-track center history for trail drawing
        self.trails = defaultdict(lambda: deque(maxlen=self.trail_length))
        self.last_seen = {}
        self.frame_count = 0
        self.last_log_time = time.time()

    # ---------- main loop ----------

    def timer_callback(self):
        frame = self._capture_frame()
        if frame is None:
            return

        self.frame_count += 1
        ids, boxes, confs = self._run_inference(frame)

        self._draw_detections(frame, ids, boxes, confs)
        self._prune_trails()

        self.target_found_pub.publish(Bool(data=len(ids) > 0))
        self._log_status(ids)
        self._publish_debug_image(frame)

    def _capture_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def _run_inference(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[self.target_class],
            tracker=self.tracker,
            conf=self.confidence,
            verbose=False,
        )
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            return ids, xyxy, confs
        return [], [], []

    # ---------- visualization ----------

    def _draw_detections(self, frame, ids, boxes, confs):
        for i, track_id in enumerate(ids):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)

            self.trails[track_id].append((cx, cy))
            self.last_seen[track_id] = self.frame_count

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{track_id} {confs[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            self._draw_trail(frame, track_id)

    def _draw_trail(self, frame, track_id):
        pts = self.trails[track_id]
        for j in range(1, len(pts)):
            cv2.line(frame, pts[j - 1], pts[j], (0, 255, 0), self.trail_thickness)

    def _prune_trails(self):
        # drop trails for tracks not seen within the timeout window
        stale = [tid for tid, seen in self.last_seen.items()
                 if self.frame_count - seen > self.trail_timeout]
        for tid in stale:
            self.trails.pop(tid, None)
            self.last_seen.pop(tid, None)

    # ---------- output ----------

    def _log_status(self, ids):
        now = time.time()
        if now - self.last_log_time >= 1.0:
            self.get_logger().info(f"Tracking IDs: {list(ids)}")
            self.last_log_time = now

    def _publish_debug_image(self, frame):
        _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.debug_image_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(YoloDetectorNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()