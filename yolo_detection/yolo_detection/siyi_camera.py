import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import os

class SiyiCamera(Node):
    def __init__(self):
        super().__init__('siyi_camera')
        
        # Publisher
        self.pub_raw = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_compressed = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)
        
        self.bridge = CvBridge()
        
        # SIYI A8 mini RTSP Address
        self.rtsp_url = "rtsp://192.168.144.25:8554/main.264"
        
        # GStreamer Pipeline (Hardware Acceleration for Jetson)
        self.pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        
        try:
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        except Exception:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open RTSP stream: {self.rtsp_url}")
            # exit() # Don't exit, keep trying
        else:
            self.get_logger().info(f"Connected to SIYI Camera: {self.rtsp_url}")

        # Timer (30 FPS)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Publish Raw Image
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_raw.publish(msg)
            
            # Publish Compressed Image (Optional, for WiFi bandwidth)
            self.publish_compressed(frame)
        else:
            self.get_logger().warn("Frame drop or stream disconnected")
            # Reconnect logic could be added here
            
    def publish_compressed(self, frame):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        # JPEG Compression quality 50%
        msg.data = np.array(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).tobytes()
        self.pub_compressed.publish(msg)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    import numpy as np # Import locally
    rclpy.init(args=args)
    node = SiyiCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import numpy as np
    main()