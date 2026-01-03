import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from ultralytics import YOLO
import cv2
import numpy as np
import os
import serial
import struct
import time

# ==========================================
# [Gimbal Control Class]
# ==========================================
class SIYIGimbal:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"Gimbal Connected on {port}")
        except Exception as e:
            print(f"Error connecting to Gimbal: {e}")
            self.ser = None
        
        self.seq = 0

    def send_speed(self, yaw_speed, pitch_speed):
        if self.ser is None: return

        cmd_id = 0x07
        data_len = 2
        
        # Speed Limit (-100 ~ 100)
        yaw_speed = max(-100, min(100, int(yaw_speed)))
        pitch_speed = max(-100, min(100, int(pitch_speed)))
        
        payload = struct.pack('<b b', yaw_speed, pitch_speed) 
        
        packet = bytearray()
        packet.append(0x55)
        packet.append(0x66)
        packet.append(0x01) 
        packet.extend(struct.pack('<H', data_len)) 
        packet.extend(struct.pack('<H', self.seq)) 
        packet.append(cmd_id)
        packet.extend(payload)
        
        crc = self.calc_crc16(packet)
        packet.extend(struct.pack('<H', crc))
        
        self.ser.write(packet)
        self.seq += 1

    def calc_crc16(self, data):
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if (crc & 0x8000):
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc = crc << 1
            crc &= 0xFFFF
        return crc

    def center(self):
        """Return Gimbal to Center (0,0)"""
        if self.ser is None: return

        cmd_id = 0x08
        data_len = 1
        payload = struct.pack('<B', 1)  # 1 = Center
        
        packet = bytearray()
        packet.append(0x55)
        packet.append(0x66)
        packet.append(0x01) 
        packet.extend(struct.pack('<H', data_len)) 
        packet.extend(struct.pack('<H', self.seq)) 
        packet.append(cmd_id)
        packet.extend(payload)
        
        crc = self.calc_crc16(packet)
        packet.extend(struct.pack('<H', crc))
        
        self.ser.write(packet)
        self.seq += 1

# ==========================================
# [YOLO Processor Node]
# ==========================================
class YoloProcessorNode(Node):
    def __init__(self):
        super().__init__('yolo_processor_node')

        self.gimbal = SIYIGimbal('/dev/ttyUSB0')

        self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.id_subscription = self.create_subscription(
            Int32, '/perception/set_target_id', self.target_id_callback, 10)

        self.debug_image_pub = self.create_publisher(
            CompressedImage, '/image_raw/compressed', 10)

        # YOLO Model Load
        base_path = "/home/sw/ros2_ws" 
        engine_path = os.path.join(base_path, "yolo11s.engine") 
        self.model = YOLO(engine_path, task='detect')

        # Tracking Variables
        self.locked_id = None    
        self.display_id = None   
        
        self.lock_timeout_frames = 120 # Memory duration (4 sec)
        self.lock_miss_count = 0
        
        self.kp_yaw = 1.0
        self.kp_pitch = 1.0
        
        self.frame_count = 0
        
        # Auto-Center & Re-locking Variables
        self.lost_count = 0
        self.fps = 30
        self.wait_seconds = 3.0
        self.lost_threshold = self.fps * self.wait_seconds
        self.last_known_box = None 
        
        # [Fix] Velocity initialization added to prevent errors
        self.vx = 0
        self.vy = 0

        self.timer = self.create_timer(0.033, self.timer_callback)

        self.get_logger().info("YOLO Node Started.")

    def target_id_callback(self, msg):
        self.locked_id = msg.data
        self.display_id = msg.data 
        
        self.lock_miss_count = 0
        self.lost_count = 0
        self.last_known_box = None
        self.vx = 0
        self.vy = 0
        self.get_logger().info(f"Target Set: ID {self.locked_id}")

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        self.frame_count += 1
        
        # try 블록 시작 (들여쓰기 주의)
        try:
            h, w, _ = frame.shape
            cx0, cy0 = w / 2.0, h / 2.0 

            # YOLO Tracking
            results = self.model.track(
                source=frame, classes=[0], verbose=False, persist=True,
                tracker="custom_botsort.yaml", conf=0.20, iou=0.45, imgsz=640
            )
            
            frame = results[0].plot()

            det = results[0]
            boxes = det.boxes
            current_ids = []
            current_boxes = []

            if boxes.id is not None:
                current_ids = boxes.id.cpu().numpy().astype(int)
                current_boxes = boxes.xyxy.cpu().numpy()

            chosen_box = None
            
            # =========================================================
            # [Logic] Motion Prediction & Dynamic Search Zone
            # =========================================================
            if self.locked_id is not None:
                
                # 1. Existing ID Visible?
                if self.locked_id in current_ids:
                    idx = np.where(current_ids == self.locked_id)[0][0]
                    chosen_box = current_boxes[idx]
                    
                    # [NEW] Calculate Velocity (Current - Previous)
                    if self.last_known_box is not None:
                        prev_cx = (self.last_known_box[0] + self.last_known_box[2]) / 2
                        prev_cy = (self.last_known_box[1] + self.last_known_box[3]) / 2
                        
                        curr_cx = (chosen_box[0] + chosen_box[2]) / 2
                        curr_cy = (chosen_box[1] + chosen_box[3]) / 2
                        
                        # Store velocity for prediction
                        self.vx = (curr_cx - prev_cx) 
                        self.vy = (curr_cy - prev_cy)
                    else:
                        self.vx = 0
                        self.vy = 0

                    # Reset counters
                    self.lock_miss_count = 0
                    self.lost_count = 0
                    self.last_known_box = chosen_box 
                
                else:
                    # 2. Target Lost -> Prediction & Re-locking
                    self.lock_miss_count += 1
                    
                    if self.last_known_box is not None:
                        # [NEW] Predict Next Position based on Velocity
                        lx1, ly1, lx2, ly2 = self.last_known_box
                        
                        # Apply velocity (simulate movement)
                        self.vx *= 0.95
                        self.vy *= 0.95
                        
                        lx1 += self.vx
                        lx2 += self.vx
                        ly1 += self.vy
                        ly2 += self.vy
                        
                        # Update last_known_box to this predicted position
                        self.last_known_box = np.array([lx1, ly1, lx2, ly2])

                        # [NEW] Expanded Search Zone (4.0x size)
                        box_w = lx2 - lx1
                        box_h = ly2 - ly1
                        
                        margin_x = box_w * 4.0  # Width 4x expansion
                        margin_y = box_h * 4.0  # Height 4x expansion
                        
                        sx1 = max(0, lx1 - margin_x)
                        sy1 = max(0, ly1 - margin_y)
                        sx2 = min(w, lx2 + margin_x)
                        sy2 = min(h, ly2 + margin_y)
                        
                        # Draw Moving Search Zone (Yellow)
                        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 255), 3)
                        cv2.putText(frame, "PREDICTING PATH...", (int(sx1), int(sy1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Re-lock Check
                        if len(current_boxes) > 0:
                            best_score = 0
                            best_id = None
                            best_new_box = None
                            last_area = box_w * box_h

                            for i, box in enumerate(current_boxes):
                                bx1, by1, bx2, by2 = box
                                bcx = (bx1 + bx2) / 2
                                bcy = (by1 + by2) / 2
                                
                                is_inside = (sx1 < bcx < sx2) and (sy1 < bcy < sy2)
                                
                                if is_inside:
                                    curr_area = (bx2 - bx1) * (by2 - by1)
                                    if last_area > 0 and curr_area > 0:
                                        size_sim = min(last_area, curr_area) / max(last_area, curr_area)
                                    else:
                                        size_sim = 0
                                    
                                    if size_sim > best_score:
                                        best_score = size_sim
                                        best_id = current_ids[i]
                                        best_new_box = box

                            if best_score > 0.5:
                                self.get_logger().warn(f"Re-locked! Internal: {self.locked_id} -> {best_id}")
                                self.locked_id = best_id 
                                chosen_box = best_new_box
                                
                                self.lock_miss_count = 0
                                self.lost_count = 0
                                self.last_known_box = chosen_box

            # =========================================================
            # [Gimbal Control]
            # =========================================================
            if chosen_box is not None:
                # Tracking
                self.lost_count = 0
                x1, y1, x2, y2 = chosen_box
                tx = (x1 + x2) / 2.0
                ty = (y1 + y2) / 2.0

                error_x = (tx - cx0) / w  
                error_y = (ty - cy0) / h  

                if abs(error_x) < 0.05: error_x = 0
                if abs(error_y) < 0.05: error_y = 0

                yaw_cmd = int(error_x * 100 * self.kp_yaw)       
                pitch_cmd = int(-error_y * 100 * self.kp_pitch) 
                yaw_cmd = max(-100, min(100, yaw_cmd))
                pitch_cmd = max(-100, min(100, pitch_cmd))

                self.gimbal.send_speed(yaw_cmd, pitch_cmd)
                
                display_text = f"LOCK ID:{self.display_id}" if self.display_id is not None else "LOCK"
                cv2.putText(frame, display_text, (int(x1), int(y1)-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                # Lost
                if self.locked_id is not None:
                    self.lost_count += 1
                    
                    if self.lost_count > self.lost_threshold:
                        self.gimbal.center()
                        self.last_known_box = None 
                    else:
                        self.gimbal.send_speed(0, 0)
                        
                        remain_time = self.wait_seconds - (self.lost_count / self.fps)
                        cv2.putText(frame, f"SEARCHING... {remain_time:.1f}s", (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if self.frame_count % 2 == 0:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                _, encoded_img = cv2.imencode('.jpg', frame, encode_param)
                
                debug_msg = CompressedImage()
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.header.frame_id = "camera_link"
                debug_msg.format = "jpeg"
                debug_msg.data = encoded_img.tobytes()
                
                self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Processing Error: {e}")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = YoloProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.gimbal:
            node.gimbal.send_speed(0, 0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()