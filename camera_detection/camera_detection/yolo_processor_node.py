import os
import time
import cv2
import numpy as np
import rclpy
from collections import defaultdict
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from ultralytics import YOLO

class YoloProcessorRealNode(Node):
    def __init__(self):
        super().__init__("yolo_processor_real_node")

        self.declare_parameter("device_id", 0)
        self.declare_parameter("model_path", "/home/sw/ros2_ws/yolo11s.engine")
        self.declare_parameter("deadzone", 0.08)

        self.device_id = self.get_parameter("device_id").get_parameter_value().integer_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.deadzone = self.get_parameter("deadzone").get_parameter_value().double_value

        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path, task="detect")

        self.target_center_pub = self.create_publisher(Point, "/perception/target_center", 10)
        self.target_found_pub = self.create_publisher(Bool, "/perception/target_found", 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, "/perception/yolo_result/compressed", 10)

        self.command_subscription = self.create_subscription(Int32, "/perception/set_target_id", self.command_callback, 10)

        self.locked_track_id = None
        self.pending_lock_id = None
        self.last_known_box = None
        self.vx, self.vy = 0.0, 0.0
        self.lost_count = 0
        self.last_log_time = time.time()

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("YOLO Real-Node Started. Distance control enabled via box height.")

    def command_callback(self, msg: Int32):
        if msg.data <= 0:
            self.clear_lock()
            return
        self.pending_lock_id = msg.data
        self.pending_start_time = time.time()
        self.locked_track_id = None

    def clear_lock(self):
        self.locked_track_id = None
        self.pending_lock_id = None
        self.lost_count = 0
        self.last_known_box = None

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        h, w, _ = frame.shape
        results = self.model.track(source=frame, persist=True, classes=[0], tracker="botsort.yaml", conf=0.3, verbose=False)

        current_ids, current_boxes, current_confs = [], [], []
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes
            current_ids = boxes.id.cpu().numpy().astype(int)
            current_boxes = boxes.xyxy.cpu().numpy()
            current_confs = boxes.conf.cpu().numpy()

        # 로그 출력
        now = time.time()
        if now - self.last_log_time >= 1.0:
            if self.locked_track_id: self.get_logger().info(f"LOCKED: ID {self.locked_track_id}")
            else: self.get_logger().info(f"Scanning... IDs: {list(current_ids)}")
            self.last_log_time = now

        # Lock 로직
        if self.pending_lock_id in current_ids:
            self.locked_track_id = int(self.pending_lock_id)
            self.pending_lock_id = None

        target_found = False
        target_box = None

        if self.locked_track_id in current_ids:
            idx = int(np.where(current_ids == self.locked_track_id)[0][0])
            target_box = current_boxes[idx]
            target_found = True

        # 시각화 및 발행
        for i, track_id in enumerate(current_ids):
            box = current_boxes[i]
            x1, y1, x2, y2 = map(int, box)
            is_locked = (track_id == self.locked_track_id)
            color = (0, 0, 255) if is_locked else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if not is_locked else 4)
            cv2.putText(frame, f"ID:{track_id} {current_confs[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if is_locked:
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                box_h = (box[3] - box[1]) / h 
                err_x = (cx - (w / 2.0)) / (w / 2.0)
                err_y = (cy - (h / 2.0)) / (h / 2.0)
                if abs(err_x) < self.deadzone: err_x = 0.0
                self.target_center_pub.publish(Point(x=float(err_x), y=float(err_y), z=float(box_h)))

        if not target_found:
            self.target_center_pub.publish(Point(x=float('nan'), y=float('nan'), z=0.0))
        self.target_found_pub.publish(Bool(data=target_found))

        _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        msg_img = CompressedImage()
        msg_img.format = "jpeg"
        msg_img.data = encoded.tobytes()
        self.debug_image_pub.publish(msg_img)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(YoloProcessorRealNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()