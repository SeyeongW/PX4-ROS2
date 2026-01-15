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

        # Parameters
        self.declare_parameter("device_id", 0)
        self.declare_parameter("model_path", "/home/sw/ros2_ws/yolo11s.engine")
        self.declare_parameter("track_history_len", 30)
        self.declare_parameter("deadzone", 0.08)

        self.device_id = self.get_parameter("device_id").get_parameter_value().integer_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.track_history_len = self.get_parameter("track_history_len").get_parameter_value().integer_value
        self.deadzone = self.get_parameter("deadzone").get_parameter_value().double_value

        # Camera Setup
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video device {self.device_id}")
            raise RuntimeError("Camera Open Failed")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.bridge = CvBridge()
        self.track_history = defaultdict(list)

        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Engine file not found: {self.model_path}")
            raise FileNotFoundError(f"Missing: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")

        self.target_center_pub = self.create_publisher(Point, "/perception/target_center", 10)
        self.target_found_pub = self.create_publisher(Bool, "/perception/target_found", 10)
        self.debug_image_pub = self.create_publisher(CompressedImage, "/image_raw/compressed", 10)

        # Subscriber
        self.command_subscription = self.create_subscription(
            Int32, "/perception/set_target_id", self.command_callback, 10
        )

        # States
        self.locked_track_id = None
        self.pending_lock_id = None
        self.pending_start_time = 0.0
        self.pending_timeout = 3.0

        self.last_known_box = None
        self.vx, self.vy = 0.0, 0.0
        self.lost_count = 0
        self.fps = 30.0
        self.lost_threshold = self.fps * 3.0 

        # Timer (30fps)
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.get_logger().info("YOLO Real-Time Node Started with Robust Locking.")

    def command_callback(self, msg: Int32):
        if msg.data <= 0:
            self.clear_lock()
            self.get_logger().info("Target Unlock.")
            return

        self.pending_lock_id = msg.data
        self.pending_start_time = time.time()
        self.locked_track_id = None
        self.lost_count = 0
        self.last_known_box = None
        self.get_logger().info(f"Searching for ID {self.pending_lock_id}...")

    def clear_lock(self):
        self.locked_track_id = None
        self.pending_lock_id = None
        self.lost_count = 0
        self.last_known_box = None
        self.vx, self.vy = 0.0, 0.0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        h, w, _ = frame.shape
        results = self.model.track(
            source=frame, persist=True, classes=[0], tracker="botsort.yaml",
            conf=0.3, iou=0.45, imgsz=640, verbose=False
        )

        current_ids = []
        current_boxes = []
        if results and results[0].boxes.id is not None:
            current_ids = results[0].boxes.id.cpu().numpy().astype(int)
            current_boxes = results[0].boxes.xyxy.cpu().numpy()

        # [1] Pending Lock Logic
        if self.pending_lock_id is not None:
            if self.pending_lock_id in current_ids:
                self.locked_track_id = int(self.pending_lock_id)
                self.pending_lock_id = None
                self.get_logger().info(f">>> LOCKED SUCCESS: ID {self.locked_track_id}")
            elif time.time() - self.pending_start_time > self.pending_timeout:
                self.get_logger().warn(f"Lock Timeout for ID {self.pending_lock_id}")
                self.pending_lock_id = None

        # [2] Tracking & Control Logic
        target_box = None
        target_found = False

        if self.locked_track_id is not None:
            if self.locked_track_id in current_ids:
                idx = int(np.where(current_ids == self.locked_track_id)[0][0])
                target_box = current_boxes[idx]
                
                if self.last_known_box is not None:
                    self.vx = ((target_box[0]+target_box[2])/2) - ((self.last_known_box[0]+self.last_known_box[2])/2)
                    self.vy = ((target_box[1]+target_box[3])/2) - ((self.last_known_box[1]+self.last_known_box[3])/2)

                self.last_known_box = target_box
                self.lost_count = 0
                target_found = True
            else:
                self.lost_count += 1
                if self.lost_count <= self.lost_threshold and self.last_known_box is not None:
                    self.last_known_box += [self.vx, self.vy, self.vx, self.vy]
                    self.vx *= 0.95; self.vy *= 0.95
                    
                    if len(current_boxes) > 0:
                        for i, box in enumerate(current_boxes):
                            bcx, bcy = (box[0]+box[2])/2, (box[1]+box[3])/2
                            if self.last_known_box[0] < bcx < self.last_known_box[2]:
                                self.locked_track_id = int(current_ids[i])
                                self.get_logger().warn(f"Re-locked to ID {self.locked_track_id}")
                                break
                else:
                    self.clear_lock()

        # [3] Publish Results
        if target_found and target_box is not None:
            cx, cy = (target_box[0]+target_box[2])/2.0, (target_box[1]+target_box[3])/2.0
            err_x = (cx - (w/2.0)) / (w/2.0)
            err_y = (cy - (h/2.0)) / (h/2.0)
            
            if abs(err_x) < self.deadzone: err_x = 0.0
            if abs(err_y) < self.deadzone: err_y = 0.0

            self.target_center_pub.publish(Point(x=float(err_x), y=float(err_y), z=0.0))
            self.target_found_pub.publish(Bool(data=True))
            
            cv2.rectangle(frame, (int(target_box[0]), int(target_box[1])), 
                          (int(target_box[2]), int(target_box[3])), (0, 0, 255), 3)
            cv2.putText(frame, f"LOCKED ID:{self.locked_track_id}", (int(target_box[0]), int(target_box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.target_found_pub.publish(Bool(data=False))
            self.target_center_pub.publish(Point(x=float('nan'), y=float('nan'), z=0.0))

        # [4] Debug Image (Compressed)
        if self.pending_lock_id is not None:
            cv2.putText(frame, f"SEARCHING ID {self.pending_lock_id}...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        msg_img = CompressedImage()
        msg_img.header.stamp = self.get_clock().now().to_msg()
        msg_img.format = "jpeg"
        msg_img.data = encoded.tobytes()
        self.debug_image_pub.publish(msg_img)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = YoloProcessorRealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()