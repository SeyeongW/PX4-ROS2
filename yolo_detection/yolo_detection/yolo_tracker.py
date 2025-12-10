import rclcpp
from rclcpp.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import socket
import struct
import numpy as np
from ultralytics import YOLO

class SiyiGimbalInterface:
    def __init__(self, ip='192.168.144.25', port=37260):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def _crc16(self, data):
        crc = 0x0000
        poly = 0x1021
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000: crc = (crc << 1) ^ poly
                else: crc <<= 1
            crc &= 0xFFFF
        return crc

    def send_speed(self, yaw, pitch):
        yaw = max(-100, min(100, int(yaw)))
        pitch = max(-100, min(100, int(pitch)))
        
        # Packet Structure: STX(2) CTRL(1) LEN(2) SEQ(2) CMD(1) DATA(2) CRC(2)
        payload = struct.pack('bb', yaw, pitch)
        header = struct.pack('<BBHHB', 0x55, 0x66, 1, len(payload), self.seq, 0x07)
        self.seq += 1
        
        msg = header + payload
        crc = self._crc16(msg)
        self.sock.sendto(msg + struct.pack('<H', crc), self.addr)

class YoloTrackerNode(Node):
    def __init__(self):
        super().__init__('yolo_tracker_node')
        self.model = YOLO("best.pt") # Ensure best.pt is in the execution directory
        self.gimbal = SiyiGimbalInterface()
        self.bridge = CvBridge()
        self.kp = 0.8
        
        self.sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.pub_result = self.create_publisher(CompressedImage, "/yolo/result/compressed", 10)
        self.get_logger().info("YOLO Tracker Started.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        
        # Inference
        results = self.model.predict(frame, conf=0.4, verbose=False, imgsz=640)
        
        target = None
        min_dist = float('inf')

        # Find target closest to center
        for r in results:
            for box in r.boxes:
                # Uncomment the next line to detect persons only (Class 0)
                # if int(box.cls[0]) == 0: 
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1+x2)/2, (y1+y2)/2
                dist = (cx - w/2)**2 + (cy - h/2)**2
                if dist < min_dist:
                    min_dist = dist
                    target = [x1, y1, x2, y2]

        if target:
            x1, y1, x2, y2 = target
            cx, cy = (x1+x2)/2, (y1+y2)/2
            
            # Error calculation (-0.5 ~ 0.5)
            err_x = (cx / w) - 0.5
            err_y = (cy / h) - 0.5
            
            # Deadzone
            if abs(err_x) < 0.05: err_x = 0
            if abs(err_y) < 0.05: err_y = 0
            
            # Control Command
            yaw_cmd = int(-err_x * 100 * self.kp)
            pitch_cmd = int(-err_y * 100 * self.kp)
            
            self.gimbal.send_speed(yaw_cmd, pitch_cmd)
            
            # Visualization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        else:
            self.gimbal.send_speed(0, 0)

        self.publish_compressed(frame)

    def publish_compressed(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        success, enc_data = cv2.imencode('.jpg', frame, encode_param)
        if success:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(enc_data).tobytes()
            self.pub_result.publish(msg)

def main(args=None):
    rclcpp.init(args=args)
    rclcpp.spin(YoloTrackerNode())
    rclcpp.shutdown()

if __name__ == '__main__':
    main()