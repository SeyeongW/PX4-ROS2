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
        
        # [SET PATH] Path to your TensorRT engine file
        # Ensure 'yolo11s.engine' exists in this directory
        self.model_path = "/home/sw/ros2_ws/yolo11s.engine"
        
        self.device_id = self.get_parameter('device_id').get_parameter_value().integer_value
        self.track_history_len = self.get_parameter('track_history_len').get_parameter_value().integer_value

        # Camera Setup
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video device {self.device_id}")
            raise RuntimeError("Camera Open Failed")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.bridge = CvBridge()
        self.track_history = defaultdict(list)

<<<<<<< Updated upstream
        self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
=======
        # Publishers
        self.debug_image_pub = self.create_publisher(CompressedImage, '/image_raw/compressed', 10)
        self.person_detected_pub = self.create_publisher(Bool, '/perception/person_detected', 10)
>>>>>>> Stashed changes
        
        # Subscriber
        self.id_subscription = self.create_subscription(Int32, '/perception/set_target_id', self.target_id_callback, 10)

        # [ENGINE LOAD] Loading TensorRT Engine
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Engine file not found at: {self.model_path}")
            self.get_logger().info("Please export your model first: yolo export model=yolo11s.pt format=engine")
            raise FileNotFoundError(f"Model file missing: {self.model_path}")

        self.model = YOLO(self.model_path, task='detect')
        self.get_logger().info(f"YOLO TensorRT Engine loaded from {self.model_path}")

        # State Variables
        self.locked_id = None
        self.display_id = None
        self.last_known_box = None
        self.vx, self.vy = 0, 0
        self.lost_count = 0
        self.fps = 30
        self.frame_count = 0

        # Processing Timer
        self.timer = self.create_timer(0.033, self.timer_callback)

    def target_id_callback(self, msg):
        if msg.data == 0:
            self.locked_id = None
            self.display_id = None
        else:
            self.locked_id = msg.data
            self.display_id = msg.data
        self.last_known_box = None
        self.vx, self.vy = 0, 0
        self.get_logger().info(f"Target Set: ID {self.locked_id}")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        self.frame_count += 1
        
        try:
            # Inference using Engine
            results = self.model.track(
                source=frame, classes=[0], verbose=False, persist=True,
                tracker="botsort.yaml", conf=0.30, iou=0.45, imgsz=640
            )

            annotated_frame = results[0].plot()
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
            chosen_id = None

            if self.locked_id is None and len(current_boxes) > 0:
                areas = (current_boxes[:, 2] - current_boxes[:, 0]) * (current_boxes[:, 3] - current_boxes[:, 1])
                idx = int(np.argmax(areas))
                chosen_box = current_boxes[idx]
                chosen_id = current_ids[idx]
                self.display_id = chosen_id

            elif self.locked_id is not None:
                if self.locked_id in current_ids:
                    idx = np.where(current_ids == self.locked_id)[0][0]
                    chosen_box = current_boxes[idx]
                    chosen_id = self.locked_id
                    
                    if self.last_known_box is not None:
                        curr_cx = (chosen_box[0] + chosen_box[2]) / 2
                        curr_cy = (chosen_box[1] + chosen_box[3]) / 2
                        prev_cx = (self.last_known_box[0] + self.last_known_box[2]) / 2
                        prev_cy = (self.last_known_box[1] + self.last_known_box[3]) / 2
                        self.vx, self.vy = (curr_cx - prev_cx), (curr_cy - prev_cy)
                    
                    self.last_known_box = chosen_box
                else:
                    if self.last_known_box is not None:
                        self.vx *= 0.95
                        self.vy *= 0.95
                        self.last_known_box += np.array([self.vx, self.vy, self.vx, self.vy])
                        bx = self.last_known_box
                        cv2.rectangle(annotated_frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 255), 2)

            if chosen_box is not None:
                x1, y1, x2, y2 = chosen_box
                if chosen_id in self.track_history:
                    pts = np.array(self.track_history[chosen_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], False, (0, 255, 0), 2)
                
                label = f"{'LOCK' if self.locked_id else 'AUTO'} ID:{chosen_id}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Publish
            if self.frame_count % 2 == 0:
                _, encoded_img = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.format = "jpeg"
                msg.data = encoded_img.tobytes()
                self.debug_image_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

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