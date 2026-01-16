import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from ultralytics import YOLO
import pyrealsense2 as rs

class YoloCombinedDebugNode(Node):
    def __init__(self):
        super().__init__("yolo_combined_debug_node")

        # 1. RealSense SDK Setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 뎁스 데이터를 색상으로 변환해주는 도구
        self.colorizer = rs.colorizer()
        self.pipeline.start(config)

        # 2. YOLO Setup
        self.model = YOLO("/home/sw/ros2_ws/yolo11s.engine", task="detect")

        # 3. Publisher
        self.combined_pub = self.create_publisher(CompressedImage, "/perception/yolo_result/compressed", 10)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Combined Debug Node Started (RGB + Depth Map)")

    def timer_callback(self):
        # 프레임 가져오기
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # 데이터 변환
        frame_rgb = np.asanyarray(color_frame.get_data())
        
        # --- 1. 뎁스맵 생성 ---
        depth_color_frame = self.colorizer.colorize(depth_frame)
        frame_depth_map = np.asanyarray(depth_color_frame.get_data())

        # --- 2. YOLO 추적 및 컬러 영상 처리 ---
        results = self.model.track(source=frame_rgb, persist=True, classes=[0], verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for i, track_id in enumerate(ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 실제 거리 측정 (m 단위)
                dist_m = depth_frame.get_distance(cx, cy)
                dist_cm = round(dist_m * 100, 1)

                # 컬러 영상에 박스와 거리 표시
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"ID:{track_id} {dist_cm}cm", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 3. 영상 가로로 합치기 ---
        # frame_rgb와 frame_depth_map을 가로로 붙임 (결과 크기: 1280x480)
        combined_image = cv2.hconcat([frame_rgb, frame_depth_map])

        # 상단에 라벨 추가 (선택 사항)
        cv2.putText(combined_image, "RGB + YOLO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, "Real-time Depth Map", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 전송
        _, encoded = cv2.imencode('.jpg', combined_image)
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.combined_pub.publish(msg)

    def __del__(self):
        self.pipeline.stop()

def main():
    rclpy.init()
    node = YoloCombinedDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()