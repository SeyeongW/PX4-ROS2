import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YoloProcessorNode(Node):
    def __init__(self):
        super().__init__('yolo_processor_node')

        # [구독] C++ 노드에서 넘어오는 압축 영상 수신
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10)
        
        # [발행 1] 타겟(사람)의 정규화된 좌표 (x, y: -0.5 ~ 0.5)
        self.target_pub = self.create_publisher(Point, '/perception/target_pos', 10)
        
        # [발행 2] 디버깅용 결과 영상 (PC 전송용, 압축됨)
        self.debug_image_pub = self.create_publisher(CompressedImage, '/perception/debug_image/compressed', 10)

        self.bridge = CvBridge()
        
        self.model = YOLO("yolov8n.pt") 
        self.get_logger().info("YOLOv8 Processor Node Started. Waiting for images...")

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # 화면 크기 및 중앙점 계산
            height, width, _ = frame.shape
            center_x, center_y = width / 2, height / 2

            # 2. YOLO 추론 (사람(class=0)만 필터링)
            results = self.model(frame, classes=[0], verbose=False)
            
            # 탐지 결과 시각화 (바운딩 박스 그리기)
            annotated_frame = results[0].plot()

            # 3. 좌표 계산 및 발행 Logic
            # 탐지된 객체가 하나라도 있을 경우
            if len(results[0].boxes) > 0:
                # 가장 신뢰도(conf)가 높은 객체 하나만 타겟팅 (필요 시 로직 변경 가능)
                best_box = results[0].boxes[0]
                
                # 바운딩 박스 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(float, best_box.xyxy[0])
                
                # 타겟의 중심점 계산
                target_x = (x1 + x2) / 2
                target_y = (y1 + y2) / 2

                # [중요] 좌표 정규화 (-0.5 ~ 0.5 범위)
                # (타겟위치 - 화면중앙) / 화면크기
                norm_x = (target_x - center_x) / width
                norm_y = (target_y - center_y) / height

                # Point 메시지 생성 및 발행
                point_msg = Point()
                point_msg.x = norm_x
                point_msg.y = norm_y
                point_msg.z = 0.0 # 2D 좌표이므로 0
                
                self.target_pub.publish(point_msg)
                # self.get_logger().info(f"Target Detected: x={norm_x:.3f}, y={norm_y:.3f}")

            # 4. 디버깅 영상 압축 및 발행
            # 대역폭 절약을 위해 Quality 50으로 설정
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, encoded_img = cv2.imencode('.jpg', annotated_frame, encode_param)

            debug_msg = CompressedImage()
            debug_msg.header = msg.header # 원본 타임스탬프 유지
            debug_msg.format = "jpeg"
            debug_msg.data = encoded_img.tobytes()

            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()