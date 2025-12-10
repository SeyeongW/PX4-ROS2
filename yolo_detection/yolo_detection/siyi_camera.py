import rclcpp
from rclcpp.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class SiyiCameraNode(Node):
    def __init__(self):
        super().__init__('siyi_camera_node')
        
        # Raw Image Publisher
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

        # SIYI A8 + Jetson GStreamer Pipeline
        pipeline = (
            "rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        
        self.get_logger().info("Connecting to Camera...")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error("Camera Open Failed! Check RTSP Connection.")
            exit(1)

        # 30Hz Timer
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "siyi_camera"
            self.pub.publish(msg)

def main(args=None):
    rclcpp.init(args=args)
    rclcpp.spin(SiyiCameraNode())
    rclcpp.shutdown()

if __name__ == '__main__':
    main()