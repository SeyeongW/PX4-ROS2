import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import defaultdict

class YoloProcessorSimNode(Node):
    def __init__(self):
        super().__init__('yolo_processor_sim_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.latest_frame = None

        self.image_subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.id_subscription = self.create_subscription(
            Int32, '/perception/set_target_id', self.target_id_callback, 10)

        self.debug_image_pub = self.create_publisher(
            CompressedImage, '/image_raw/compressed', 10)
        self.person_detected_pub = self.create_publisher(
            Bool, '/perception/person_detected', 10)
        
        self.target_info_pub = self.create_publisher(
            Point, '/perception/target_pos_info', 10)

        self.model = YOLO("yolo11s.pt") 

        # [추가] 이동 경로(Trail) 저장을 위한 딕셔너리
        self.track_history = defaultdict(lambda: [])

        self.locked_id = None
        self.display_id = None

        self.lock_miss_count = 0
        self.frame_count = 0

        self.lost_count = 0
        self.fps = 30
        self.wait_seconds = 3.0
        self.lost_threshold = self.fps * self.wait_seconds
        self.last_known_box = None

        self.vx = 0
        self.vy = 0

        self.timer_period_s = 0.033
        self.timer = self.create_timer(self.timer_period_s, self.timer_callback)

        self.get_logger().info("YOLO Sim Node Started (with Tracking Trails).")

    def target_id_callback(self, msg):
        self.locked_id = msg.data
        self.display_id = msg.data
        self.lock_miss_count = 0
        self.lost_count = 0
        self.last_known_box = None
        self.vx = 0
        self.vy = 0
        
        # 타겟 변경 시 기존 경로 기록 초기화 (선택 사항)
        self.track_history.clear()
        
        self.get_logger().info(f"Target Set: ID {self.locked_id}")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.latest_frame = None
            self.get_logger().error(f"Image conversion failed: {e}")

    def timer_callback(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        self.frame_count += 1

        try:
            h, w, _ = frame.shape
            cx0, cy0 = w / 2.0, h / 2.0 

            # [핵심] persist=True는 ID 유지를 위해 필수
            results = self.model.track(
                source=frame, classes=[0], verbose=False, persist=True,
                tracker="botsort.yaml", conf=0.10, iou=0.45, imgsz=640
            )

            # YOLO 기본 박스 그리기
            frame = results[0].plot()
            
            det = results[0]
            boxes = det.boxes
            
            current_ids = []
            current_boxes = []

            # ---------------------------------------------------------
            # [추가] 이동 경로(Trail) 시각화 로직
            # ---------------------------------------------------------
            if boxes.id is not None:
                current_ids = boxes.id.cpu().numpy().astype(int)
                current_boxes = boxes.xyxy.cpu().numpy()
                
                # 시각화: 과거 이동 경로 그리기
                for box, track_id in zip(current_boxes, current_ids):
                    x1, y1, x2, y2 = box
                    track = self.track_history[track_id]
                    
                    # 현재 중심점 계산
                    center_x = float((x1 + x2) / 2)
                    center_y = float((y1 + y2) / 2) 
                    
                    track.append((center_x, center_y))
                    
                    if len(track) > 30:
                        track.pop(0)

                    # 선 그리기
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    
                    # 타겟 ID면 빨간색 굵은 선, 아니면 얇은 초록선
                    if track_id == self.locked_id:
                        cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=3)
                    else:
                        cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)

            self.person_detected_pub.publish(Bool(data=len(current_boxes) > 0))

            chosen_box = None

            # ---------------------------------------------------------
            # 타겟 ID 잠금 및 추적
            # ---------------------------------------------------------
            if self.locked_id is not None:
                if self.locked_id in current_ids:
                    # 1. 타겟 찾음
                    idx = np.where(current_ids == self.locked_id)[0][0]
                    chosen_box = current_boxes[idx]

                    if self.last_known_box is not None:
                        prev_cx = (self.last_known_box[0] + self.last_known_box[2]) / 2
                        prev_cy = (self.last_known_box[1] + self.last_known_box[3]) / 2
                        curr_cx = (chosen_box[0] + chosen_box[2]) / 2
                        curr_cy = (chosen_box[1] + chosen_box[3]) / 2
                        self.vx = (curr_cx - prev_cx)
                        self.vy = (curr_cy - prev_cy)
                    else:
                        self.vx, self.vy = 0, 0

                    self.lock_miss_count = 0
                    self.lost_count = 0
                    self.last_known_box = chosen_box

                else:
                    self.lock_miss_count += 1
                    if self.last_known_box is not None:
                        lx1, ly1, lx2, ly2 = self.last_known_box
                        
                        # 관성 예측 적용
                        self.vx *= 0.95
                        self.vy *= 0.95
                        lx1 += self.vx; lx2 += self.vx
                        ly1 += self.vy; ly2 += self.vy
                        self.last_known_box = np.array([lx1, ly1, lx2, ly2])

                        # 탐색 영역(Search Zone) 표시
                        sx1 = max(0, lx1 - (lx2-lx1)*2)
                        sy1 = max(0, ly1 - (ly2-ly1)*2)
                        sx2 = min(w, lx2 + (lx2-lx1)*2)
                        sy2 = min(h, ly2 + (ly2-ly1)*2)
                        
                        # 예측 중일 때는 노란색 박스
                        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 255), 2)
                        cv2.putText(frame, "PREDICTING...", (int(sx1), int(sy1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # ---------------------------------------------------------
            # C++ 드론 제어 노드로 정보 전송
            # ---------------------------------------------------------
            target_msg = Point()
            
            if chosen_box is not None:
                x1, y1, x2, y2 = chosen_box
                box_cx = (x1 + x2) / 2.0
                box_cy = (y1 + y2) / 2.0
                box_area = (x2 - x1) * (y2 - y1)
                
                error_x = (box_cx - cx0) / (w / 2.0)
                error_y = (box_cy - cy0) / (h / 2.0)
                area_ratio = box_area / (w * h)

                target_msg.x = float(error_x)
                target_msg.y = float(error_y)
                target_msg.z = float(area_ratio)

                cv2.putText(frame, f"LOCK {self.locked_id}", (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                target_msg.x = 0.0
                target_msg.y = 0.0
                target_msg.z = 0.0

            self.target_info_pub.publish(target_msg)

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

def main(args=None):
    rclpy.init(args=args)
    node = YoloProcessorSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()