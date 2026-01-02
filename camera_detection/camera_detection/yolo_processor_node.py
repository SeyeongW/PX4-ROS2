import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
from ultralytics import YOLO
import cv2
import numpy as np
import os
import serial
import struct
import time

# ==========================================
# [Gimbal Control Class] - C코드의 기능을 옮겨옴
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
        """
        yaw_speed, pitch_speed: -100 ~ 100
        """
        if self.ser is None: return

        # SIYI SDK Format: STX(2) + CTRL(1) + LEN(2) + SEQ(2) + CMD_ID(1) + DATA(N) + CRC(2)
        # Rotate Command ID is 0x07
        
        cmd_id = 0x07
        data_len = 2
        
        # Clamp speeds to -100 ~ 100
        yaw_speed = max(-100, min(100, int(yaw_speed)))
        pitch_speed = max(-100, min(100, int(pitch_speed)))
        
        # Prepare Header & Payload
        # Header: 0x55 0x66
        # Ctrl: 0x01 (Need ACK? usually 0 is fine for stream, but code used 1)
        payload = struct.pack('<b b', yaw_speed, pitch_speed) # signed char (1 byte each)
        
        packet = bytearray()
        packet.append(0x55)
        packet.append(0x66)
        packet.append(0x01) # Ctrl
        packet.extend(struct.pack('<H', data_len)) # Data Len (Little Endian)
        packet.extend(struct.pack('<H', self.seq)) # Seq
        packet.append(cmd_id)
        packet.extend(payload)
        
        # Calculate CRC16
        crc = self.calc_crc16(packet)
        packet.extend(struct.pack('<H', crc))
        
        self.ser.write(packet)
        self.seq += 1

    def calc_crc16(self, data):
        # CRC-16/CCITT-FALSE implementation matching SIYI SDK
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

# ==========================================
# [ROS 2 Node]
# ==========================================
class YoloTrackingNode(Node):
    def __init__(self):
        super().__init__('yolo_tracking_node')

        # 1. Initialize Gimbal
        self.gimbal = SIYIGimbal('/dev/ttyUSB0')

        # 2. Subscribers
        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        self.id_subscription = self.create_subscription(
            Int32, '/perception/set_target_id', self.target_id_callback, 10)

        # 3. Debug Publisher
        self.debug_image_pub = self.create_publisher(CompressedImage, '/perception/debug_image/compressed', 10)

        # 4. YOLO Model Setup (Check Path!)
        base_path = "/home/sw/ros2_ws/src/PX4-ROS2"
        engine_path = os.path.join(base_path, "yolo11n.engine")
        yaml_path = os.path.join(base_path, "bytetrack.yaml")

        if not os.path.exists(yaml_path):
            self.tracker_yaml = "bytetrack.yaml"
        else:
            self.tracker_yaml = yaml_path
        
        self.model = YOLO(engine_path, task='detect')

        # 5. Tracking Variables
        self.locked_id = None
        self.lock_timeout_frames = 60
        self.lock_miss_count = 0
        
        # [PID Control Variables]
        self.kp_yaw = 40.0   
        self.kp_pitch = 40.0 

        self.get_logger().info("SIYI Tracking Node Started. Waiting for ID...")

    def target_id_callback(self, msg):
        self.locked_id = msg.data
        self.lock_miss_count = 0
        self.get_logger().info(f"Target ID Set: {self.locked_id}")

    def image_callback(self, msg):
        try:
            # Decode Image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None: return

            h, w, _ = frame.shape
            cx0, cy0 = w / 2.0, h / 2.0 # Image Center

            # Run YOLO + ByteTrack
            results = self.model.track(
                source=frame, classes=[0], verbose=False, persist=True,
                tracker=self.tracker_yaml, conf=0.20, iou=0.45, imgsz=640
            )

            det = results[0]
            boxes = det.boxes
            
            chosen_box = None
            
            # Find the locked ID
            if self.locked_id is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                
                idxs = np.where(ids == self.locked_id)[0]
                if len(idxs) > 0:
                    i = idxs[0]
                    chosen_box = xyxy[i]
                    self.lock_miss_count = 0
                else:
                    self.lock_miss_count += 1
                    if self.lock_miss_count > self.lock_timeout_frames:
                        self.locked_id = None # Lost target
                        self.get_logger().warn("Target Lost.")

            # [Control Logic]
            yaw_cmd = 0
            pitch_cmd = 0

            if chosen_box is not None:
                # Calculate Target Center
                x1, y1, x2, y2 = chosen_box
                tx = (x1 + x2) / 2.0
                ty = (y1 + y2) / 2.0

                # Calculate Error (-0.5 ~ 0.5 range)
                error_x = (tx - cx0) / w  # Horizontal error
                error_y = (ty - cy0) / h  # Vertical error

                # P-Control: Speed = Error * Kp
                # Error X > 0 (Target is right) -> Turn Right (Positive Yaw Speed?)
                # Check your gimbal direction! If it goes opposite, change sign (-).
                yaw_cmd = int(error_x * 100 * self.kp_yaw)     
                pitch_cmd = int(-error_y * 100 * self.kp_pitch) # Pitch is usually inverted (Up is negative error)

                # Send Command to Gimbal
                self.gimbal.send_speed(yaw_cmd, pitch_cmd)
                
                # Draw Lock Indicator
                cv2.circle(frame, (int(tx), int(ty)), 10, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{self.locked_id}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                # No target -> Stop Gimbal
                self.gimbal.send_speed(0, 0)

            # Debug Image Publish
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            debug_msg = CompressedImage()
            debug_msg.header = msg.header
            debug_msg.format = "jpeg"
            debug_msg.data = encoded_img.tobytes()
            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloTrackingNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    
    finally:
        if node.gimbal:
            try:
                node.gimbal.send_speed(0, 0)
                print("Gimbal stop.")
            except:
                pass

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node exit.")

if __name__ == '__main__':
    main()
