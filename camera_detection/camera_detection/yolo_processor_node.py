import rclpy
from collections import defaultdict
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os

class YoloProcessorRealNode(Node):
    def __init__(self):
        super().__init__('yolo_processor_real_node')

        # Parameters
        self.declare_parameter('device_id', 0)
        self.declare_parameter('track_history_len', 30)
        
        self.device_id = self.get_parameter('device_id').get_parameter_value().integer_value
        self.track_history_len = self.get_parameter('track_history_len').get_parameter_value().integer_value
        
        # Model path
        self.model_path = "/home/sw/ros2_ws/yolo11s.engine"

        # Camera Setup
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video device {self.device_id}")
            raise RuntimeError("Camera Open Failed")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.bridge = CvBridge()
        self.track_history = defaultdict(list)

        # Publishers
        self.debug_image_pub = self.create_publisher(CompressedImage, '/image_raw/compressed', 10)
        self.person_detected_pub = self.create_publisher(Bool, '/perception/person_detected', 10)
        
        # Subscriber
        self.id_subscription = self.create_subscription(
            Int32, '/perception/set_target_id', self.target_id_callback, 10)

        # Load TensorRT Engine
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Engine file not found: {self.model_path}")
            raise FileNotFoundError(f"Missing: {self.model_path}")
            
        self.model = YOLO(self.model_path, task='detect')

        # Logic States
        self.locked_id = None  # Only tracks when this is not None
        self.frame_count = 0
        self.lost_count = 0
        self.fps = 30
        self.wait_seconds = 3.0
        self.lost_threshold = self.fps * self.wait_seconds
        self.last_known_box = None
        self.vx = 0
        self.vy = 0

        # Processing Timer
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("YOLO Real Node Started. Waiting for manual Target ID...")

    def target_id_callback(self, msg):
        # If ID is 0 or negative, release the lock
        if msg.data <= 0:
            self.locked_id = None
            self.get_logger().info("Target Released.")
        else:
            self.locked_id = msg.data
            self.get_logger().info(f"Target Locked: ID {self.locked_id}")

        self.lost_count = 0
        self.last_known_box = None
        self.vx = 0
        self.vy = 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_count += 1

        try:
            h, w, _ = frame.shape

            # Inference using TensorRT
            results = self.model.track(
                source=frame, classes=[0], verbose=False, persist=True,
                tracker="botsort.yaml", conf=0.30, iou=0.45, imgsz=640
            )

            frame_plot = results[0].plot()
            det = results[0]
            boxes = det.boxes
            current_ids = []
            current_boxes = []

            if boxes.id is not None:
                current_ids = boxes.id.cpu().numpy().astype(int)
                current_boxes = boxes.xyxy.cpu().numpy()
                for track_id, box in zip(current_ids, current_boxes):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    self.track_history[track_id].append((float(cx), float(cy)))
                    if len(self.track_history[track_id]) > self.track_history_len:
                        self.track_history[track_id].pop(0)

            self.person_detected_pub.publish(Bool(data=len(current_boxes) > 0))

            chosen_box = None

            # MANUAL TRACKING LOGIC (Only runs if locked_id is set)
            if self.locked_id is not None:
                if self.locked_id in current_ids:
                    idx = np.where(current_ids == self.locked_id)[0][0]
                    chosen_box = current_boxes[idx]

                    # Velocity calculation
                    if self.last_known_box is not None:
                        curr_cx = (chosen_box[0] + chosen_box[2]) / 2
                        curr_cy = (chosen_box[1] + chosen_box[3]) / 2
                        prev_cx = (self.last_known_box[0] + self.last_known_box[2]) / 2
                        prev_cy = (self.last_known_box[1] + self.last_known_box[3]) / 2
                        self.vx, self.vy = (curr_cx - prev_cx), (curr_cy - prev_cy)
                    
                    self.lost_count = 0
                    self.last_known_box = chosen_box
                else:
                    # Prediction when target is lost
                    if self.last_known_box is not None:
                        lx1, ly1, lx2, ly2 = self.last_known_box
                        self.vx *= 0.95
                        self.vy *= 0.95
                        lx1 += self.vx; lx2 += self.vx
                        ly1 += self.vy; ly2 += self.vy
                        self.last_known_box = np.array([lx1, ly1, lx2, ly2])

                        box_w, box_h = lx2 - lx1, ly2 - ly1
                        margin_x, margin_y = box_w * 4.0, box_h * 4.0
                        sx1, sy1 = max(0, lx1 - margin_x), max(0, ly1 - margin_y)
                        sx2, sy2 = min(w, lx2 + margin_x), min(h, ly2 + margin_y)

                        cv2.rectangle(frame_plot, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 255), 3)
                        cv2.putText(frame_plot, f"LOST ID:{self.locked_id} - PREDICTING", (int(sx1), int(sy1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Re-lock logic
                        if len(current_boxes) > 0:
                            best_score = 0
                            best_id = None
                            for i, box in enumerate(current_boxes):
                                bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                                if (sx1 < bcx < sx2) and (sy1 < bcy < sy2):
                                    curr_area = (box[2] - box[0]) * (box[3] - box[1])
                                    size_sim = min(box_w*box_h, curr_area) / max(box_w*box_h, curr_area)
                                    if size_sim > best_score:
                                        best_score = size_sim
                                        best_id = current_ids[i]
                                        chosen_box = box

                            if best_score > 0.5:
                                self.get_logger().warn(f"Re-locked: {self.locked_id} -> {best_id}")
                                self.locked_id = best_id
                                self.last_known_box = chosen_box
                                self.lost_count = 0

            # UI Overlays
            if chosen_box is not None:
                self.lost_count = 0
                x1, y1, x2, y2 = chosen_box
                # Path history
                if self.locked_id in self.track_history:
                    pts = np.array(self.track_history[self.locked_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_plot, [pts], False, (0, 255, 0), 2)
                
                cv2.putText(frame_plot, f"LOCKED ID:{self.locked_id}", (int(x1), int(y1)-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if self.locked_id is not None:
                    self.lost_count += 1
                    if self.lost_count > self.lost_threshold:
                        self.locked_id = None # Give up after timeout
                        self.last_known_box = None
                    else:
                        remain = self.wait_seconds - (self.lost_count / self.fps)
                        cv2.putText(frame_plot, f"SEARCHING ID:{self.locked_id}... {remain:.1f}s", 
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Debug Image Publish
            if self.frame_count % 2 == 0:
                _, encoded = cv2.imencode('.jpg', frame_plot, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                debug_msg = CompressedImage()
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.format = "jpeg"
                debug_msg.data = encoded.tobytes()
                self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
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

if __name__ == '__main__':
    main()