import rclpy
from collections import defaultdict
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os


class YoloProcessorSimNode(Node):
    def __init__(self):
        super().__init__('yolo_processor_sim_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('enable_reid', True)
        self.declare_parameter('track_history_len', 30)
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.enable_reid = self.get_parameter('enable_reid').get_parameter_value().bool_value
        self.track_history_len = self.get_parameter('track_history_len').get_parameter_value().integer_value

        self.bridge = CvBridge()
        self.latest_frame = None
        self.track_history = defaultdict(list)

        self.image_subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.id_subscription = self.create_subscription(
            Int32, '/perception/set_target_id', self.target_id_callback, 10)

        self.debug_image_pub = self.create_publisher(
            CompressedImage, '/image_raw/compressed', 10)
        self.person_detected_pub = self.create_publisher(
            Bool, '/perception/person_detected', 10)

        base_path = "/home/sw/ros2_ws"
        engine_path = os.path.join(base_path, "yolo11s.engine")
        self.model = YOLO(engine_path, task='detect')

        self.locked_id = None
        self.display_id = None

        self.lock_timeout_frames = 120
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

        self.get_logger().info("YOLO Sim Node Started.")

    def target_id_callback(self, msg):
        if msg.data == 0:
            self.locked_id = None
            self.display_id = None
        else:
            self.locked_id = msg.data
            self.display_id = msg.data

        self.lock_miss_count = 0
        self.lost_count = 0
        self.last_known_box = None
        self.vx = 0
        self.vy = 0
        self.get_logger().info(f"Target Set: ID {self.locked_id}")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.latest_frame = None
            self.get_logger().error(f"Image conversion failed: {e}")

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
        if self.latest_frame is None:
            self.person_detected_pub.publish(Bool(data=False))
            return

        frame = self.latest_frame.copy()
        self.frame_count += 1

        try:
            h, w, _ = frame.shape

            results = self.model.track(
                source=frame, classes=[0], verbose=False, persist=True,
                tracker="custom_botsort.yaml", conf=0.60, iou=0.45, imgsz=640,
                reid=self.enable_reid
            )

            frame = results[0].plot()

            det = results[0]
            boxes = det.boxes
            current_ids = []
            current_boxes = []

            if boxes.id is not None:
                current_ids = boxes.id.cpu().numpy().astype(int)
                current_boxes = boxes.xyxy.cpu().numpy()
                for track_id, box in zip(current_ids, current_boxes):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    history = self.track_history[track_id]
                    history.append((float(cx), float(cy)))
                    if len(history) > self.track_history_len:
                        history.pop(0)
            self.person_detected_pub.publish(Bool(data=len(current_boxes) > 0))

            chosen_box = None
            chosen_id = None
            auto_track = self.locked_id is None
            if auto_track and len(current_boxes) > 0:
                areas = (current_boxes[:, 2] - current_boxes[:, 0]) * (
                    current_boxes[:, 3] - current_boxes[:, 1]
                )
                idx = int(np.argmax(areas))
                chosen_box = current_boxes[idx]
                if len(current_ids) > idx:
                    chosen_id = current_ids[idx]
                    self.display_id = current_ids[idx]
                else:
                    self.display_id = None

            if self.locked_id is not None:
                if self.locked_id in current_ids:
                    idx = np.where(current_ids == self.locked_id)[0][0]
                    chosen_box = current_boxes[idx]
                    chosen_id = self.locked_id

                    if self.last_known_box is not None:
                        prev_cx = (self.last_known_box[0] + self.last_known_box[2]) / 2
                        prev_cy = (self.last_known_box[1] + self.last_known_box[3]) / 2

                        curr_cx = (chosen_box[0] + chosen_box[2]) / 2
                        curr_cy = (chosen_box[1] + chosen_box[3]) / 2

                        self.vx = (curr_cx - prev_cx)
                        self.vy = (curr_cy - prev_cy)
                    else:
                        self.vx = 0
                        self.vy = 0

                    self.lock_miss_count = 0
                    self.lost_count = 0
                    self.last_known_box = chosen_box

                else:
                    self.lock_miss_count += 1

                    if self.last_known_box is not None:
                        lx1, ly1, lx2, ly2 = self.last_known_box

                        self.vx *= 0.95
                        self.vy *= 0.95

                        lx1 += self.vx
                        lx2 += self.vx
                        ly1 += self.vy
                        ly2 += self.vy

                        self.last_known_box = np.array([lx1, ly1, lx2, ly2])

                        box_w = lx2 - lx1
                        box_h = ly2 - ly1

                        margin_x = box_w * 4.0
                        margin_y = box_h * 4.0

                        sx1 = max(0, lx1 - margin_x)
                        sy1 = max(0, ly1 - margin_y)
                        sx2 = min(w, lx2 + margin_x)
                        sy2 = min(h, ly2 + margin_y)

                        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 255), 3)
                        cv2.putText(frame, "PREDICTING PATH...", (int(sx1), int(sy1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

            if chosen_box is not None:
                self.lost_count = 0
                x1, y1, x2, y2 = chosen_box
                if chosen_id is not None and chosen_id in self.track_history:
                    points = self.track_history[chosen_id]
                    if len(points) > 1:
                        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
                if self.locked_id is None:
                    display_text = f"AUTO ID:{self.display_id}" if self.display_id is not None else "AUTO"
                else:
                    display_text = f"LOCK ID:{self.display_id}" if self.display_id is not None else "LOCK"
                cv2.putText(frame, display_text, (int(x1), int(y1)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if self.locked_id is not None:
                    self.lost_count += 1

                    if self.lost_count > self.lost_threshold:
                        self.last_known_box = None
                    else:
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
