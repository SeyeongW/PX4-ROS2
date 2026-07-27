"""조류 인식 + SIYI 짐벌 중앙추종 + 줌 클로즈업 + ID 락 노드.

동작 흐름 (상태 머신):

  SEARCH : 낮은 confidence 로 화면 전체에서 조류를 탐지한다.
           검출된 것 중 가장 정확도(conf)가 높은 대상을 후보로 잡고 CENTER 로.
  CENTER : 후보를 화면 중앙으로 가져오도록 짐벌 yaw/pitch 속도 제어.
           중앙 오차가 충분히 작아지면 ZOOM 으로.
  ZOOM   : 중앙추종을 유지한 채, 바운딩박스가 화면을 적당히 채울 때까지
           (target_fill_ratio, 완전 풀줌이 아닌 max_zoom 한계 내) 줌을 당긴다.
           클로즈업 상태에서 높은 confidence 가 confirm_frames 만큼 연속되면
           "조류 확정" -> 해당 track ID 를 락(LOCK)하고 LOCKED 로.
  LOCKED : 락된 ID 만 계속 중앙추종한다. 줌은 fill ratio 를 유지하도록 미세 조정.
           대상을 lost_timeout 이상 놓치면 줌 아웃 + 센터 복귀 후 SEARCH 로.

짐벌 프로토콜은 yolo_processor_node.SIYIGimbal / gimbal_check.py 와 동일한
SIYI 시리얼 프로토콜을 사용하고, 여기에 줌 제어(0x05 manual / 0x0F absolute)를
추가했다. A8 mini 는 최대 6배 디지털 줌이며, 완전 풀줌을 피하기 위해
max_zoom 기본값을 4.0 으로 둔다.
"""

import os
import struct
import time

import cv2
import numpy as np
import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from ultralytics import YOLO


# ==========================================================================
# SIYI Gimbal (속도 제어 + 센터 + 줌)
# ==========================================================================
class SIYIGimbal:
    def __init__(self, port="/dev/ttyTHS1", baud=115200):
        self.seq = 0
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"[Gimbal] connected on {port} @ {baud}")
        except Exception as e:  # noqa: BLE001
            print(f"[Gimbal] connect failed ({port}): {e}")
            self.ser = None

    # ---- low level framing ----
    @staticmethod
    def calc_crc16(data):
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
                crc &= 0xFFFF
        return crc

    def _send(self, cmd_id, payload=b""):
        if self.ser is None:
            return
        packet = bytearray()
        packet.append(0x55)
        packet.append(0x66)
        packet.append(0x01)  # control frame
        packet.extend(struct.pack("<H", len(payload)))
        packet.extend(struct.pack("<H", self.seq))
        packet.append(cmd_id)
        packet.extend(payload)
        packet.extend(struct.pack("<H", self.calc_crc16(packet)))
        self.ser.write(packet)
        self.seq = (self.seq + 1) & 0xFFFF

    # ---- commands ----
    def send_speed(self, yaw_speed, pitch_speed):
        yaw_speed = max(-100, min(100, int(yaw_speed)))
        pitch_speed = max(-100, min(100, int(pitch_speed)))
        self._send(0x07, struct.pack("<bb", yaw_speed, pitch_speed))

    def center(self):
        self._send(0x08, struct.pack("<B", 1))

    def manual_zoom(self, direction):
        """0x05: direction 1=zoom in, -1=zoom out, 0=stop (연속 줌)."""
        self._send(0x05, struct.pack("<b", max(-1, min(1, int(direction)))))

    def set_zoom(self, level):
        """0x0F: 절대 줌 배율 지정 (정수부, 소수부 1자리). A8 mini 펌웨어 의존."""
        level = max(1.0, level)
        int_part = int(level)
        dec_part = int(round((level - int_part) * 10)) % 10
        self._send(0x0F, struct.pack("<BB", int_part & 0xFF, dec_part & 0xFF))


# ==========================================================================
# Bird lock-tracking node
# ==========================================================================
class BirdLockTrackNode(Node):
    SEARCH, CENTER, ZOOM, LOCKED = "SEARCH", "CENTER", "ZOOM", "LOCKED"

    def __init__(self):
        super().__init__("bird_lock_track_node")
        self._init_parameters()
        self._init_gimbal()
        self._init_camera()
        self._init_model()
        self._init_ros_io()
        self._init_state()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info(
            f"Bird lock-track node started. model={self.model_path} state=SEARCH")

    # ---------------------- init ----------------------
    def _init_parameters(self):
        d = self.declare_parameter
        d("model_path", self._resolve_default_model())
        d("camera_index", 0)
        d("gimbal_port", "/dev/ttyTHS1")
        d("gimbal_baud", 115200)
        d("frame_width", 1280)
        d("frame_height", 720)
        d("camera_fps", 30)
        d("flip_horizontal", False)
        d("timer_period", 0.033)
        # detection / tracking
        d("target_class", 14)  # yolo11s(COCO): 14=bird, bird.engine: 0
        # BoT-SORT(+ReID, GMC): 짐벌이 움직여 배경이 흐를 때 ByteTrack 보다
        # 카메라 모션 보정/외형 재매칭이 강해 겹침 구간에서 ID 유지가 잘 된다.
        d("tracker", "custom_botsort.yaml")
        d("search_conf", 0.12)  # 낮춰서 놓침↓ (오탐은 트래커/확정단계가 거름)
        d("confirm_conf", 0.50)  # 클로즈업에서 조류 확정용 높은 confidence
        d("imgsz", 1280)  # ★ TRT 엔진은 빌드 크기와 일치해야 함. .pt 는 자유(CPU면 640~960 권장)
        # gimbal centering
        d("kp_yaw", 1.0)
        d("kp_pitch", 1.0)
        d("center_deadzone", 0.05)  # 이 안이면 오차 0 취급 (떨림 방지)
        d("center_threshold", 0.10)  # CENTER->ZOOM 전환 임계 (정규화 오차)
        # zoom close-up
        d("use_absolute_zoom", True)  # True=0x0F 절대줌, False=0x05 수동줌
        d("max_zoom", 4.0)  # 완전 풀줌(6x) 피하기
        d("min_zoom", 1.0)
        d("zoom_step", 0.3)  # 절대줌 1틱 증감
        d("target_fill_ratio", 0.6)  # bbox 긴 변 / 화면 긴 변 목표
        d("fill_tolerance", 0.12)
        d("zoom_interval_s", 0.20)  # 줌 명령 throttle
        # lock / lost
        d("confirm_frames", 5)  # 클로즈업 고conf 연속 프레임 -> 락
        d("lost_timeout_s", 3.0)
        # motion prediction & re-lock (배경에 잠깐 묻혀 ID 가 끊겨도 추종 유지)
        d("vel_decay", 0.92)  # 놓친 동안 추정 속도 감쇠율
        d("relock_search_scale", 3.0)  # 예측 박스 주변 탐색존 배율
        d("relock_size_sim", 0.4)  # 재락 시 크기 유사도 최소
        d("debug_scale", 1.0)
        d("jpeg_quality", 90)
        d("publish_every_n", 1)  # 디버그 영상 N프레임당 1장 (1=매 프레임)

        g = self.get_parameter
        gi = lambda n: g(n).get_parameter_value().integer_value
        gd = lambda n: g(n).get_parameter_value().double_value
        gs = lambda n: g(n).get_parameter_value().string_value
        gb = lambda n: g(n).get_parameter_value().bool_value

        self.model_path = gs("model_path")
        self.camera_index = gi("camera_index")
        self.gimbal_port = gs("gimbal_port")
        self.gimbal_baud = gi("gimbal_baud")
        self.frame_width = gi("frame_width")
        self.frame_height = gi("frame_height")
        self.camera_fps = gi("camera_fps")
        self.flip_horizontal = gb("flip_horizontal")
        self.timer_period = gd("timer_period")
        self.target_class = gi("target_class")
        self.tracker = self._resolve_tracker(gs("tracker"))
        self.search_conf = gd("search_conf")
        self.confirm_conf = gd("confirm_conf")
        self.imgsz = gi("imgsz")
        self.kp_yaw = gd("kp_yaw")
        self.kp_pitch = gd("kp_pitch")
        self.center_deadzone = gd("center_deadzone")
        self.center_threshold = gd("center_threshold")
        self.use_absolute_zoom = gb("use_absolute_zoom")
        self.max_zoom = gd("max_zoom")
        self.min_zoom = gd("min_zoom")
        self.zoom_step = gd("zoom_step")
        self.target_fill_ratio = gd("target_fill_ratio")
        self.fill_tolerance = gd("fill_tolerance")
        self.zoom_interval_s = gd("zoom_interval_s")
        self.confirm_frames = gi("confirm_frames")
        self.lost_timeout_s = gd("lost_timeout_s")
        self.vel_decay = gd("vel_decay")
        self.relock_search_scale = gd("relock_search_scale")
        self.relock_size_sim = gd("relock_size_sim")
        self.debug_scale = gd("debug_scale")
        self.jpeg_quality = gi("jpeg_quality")
        self.publish_every_n = max(1, gi("publish_every_n"))

    def _resolve_default_model(self):
        roots = [
            os.path.expanduser("~/ros2_ws/PX4-ROS2"),
            os.path.expanduser("~/ros2_ws"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        ]
        # PC(CUDA 없음)에서는 TRT .engine 을 못 쓰므로 .pt 를 우선 찾는다.
        # 정확도 우선: yolo11l > yolo11m > yolo11s. 그 다음 Jetson 엔진/커스텀 폴백.
        for name in ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt",
                     "yolo11s_1280.engine", "yolo11s.engine", "bird.pt"):
            for root in roots:
                path = os.path.join(root, name)
                if os.path.exists(path):
                    return path
        return "yolo11l.pt"

    def _resolve_tracker(self, tracker):
        if os.path.isabs(tracker) or os.path.dirname(tracker):
            return tracker
        for root in (os.path.expanduser("~/ros2_ws/PX4-ROS2"),
                     os.path.expanduser("~/ros2_ws"),
                     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))):
            path = os.path.join(root, tracker)
            if os.path.exists(path):
                return path
        return tracker

    def _init_gimbal(self):
        self.gimbal = SIYIGimbal(self.gimbal_port, self.gimbal_baud)

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _init_model(self):
        self.model = YOLO(self.model_path, task="detect")

    def _init_ros_io(self):
        self.locked_id_pub = self.create_publisher(Int32, "/perception/bird/locked_id", 10)
        self.locked_pub = self.create_publisher(Bool, "/perception/bird/locked", 10)
        self.debug_image_pub = self.create_publisher(
            CompressedImage, "/perception/bird_track/compressed", 10)

    def _init_state(self):
        self.state = self.SEARCH
        self.candidate_id = None  # CENTER/ZOOM 중 추적하는 후보 track id
        self.locked_id = None  # LOCKED 상태의 확정 id
        self.confirm_count = 0  # 클로즈업 고conf 연속 카운트
        # 모션 예측용: 마지막 박스와 픽셀 속도(EMA)
        self.last_box = None
        self.vx = 0.0
        self.vy = 0.0
        self.predicted_box = None  # HUD 표시용(놓친 동안 예측 위치)
        self.zoom_level = self.min_zoom
        self.last_zoom_cmd_t = 0.0
        self.last_seen_t = time.time()
        self.frame_count = 0
        self.last_log_t = time.time()
        self.fps = 0.0
        self.last_frame_t = time.time()
        # 시작 시 줌 초기화
        self._apply_zoom(self.min_zoom, force=True)
        self.gimbal.send_speed(0, 0)

    # ---------------------- main loop ----------------------
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        self.frame_count += 1
        self._update_fps()

        h, w = frame.shape[:2]
        self.predicted_box = None  # 이번 프레임 예측 위치(놓쳤을 때만 채워짐)
        ids, boxes, confs = self._run_inference(frame)

        # 현재 상태 처리
        if self.state == self.SEARCH:
            self._do_search(ids, boxes, confs, w, h)
        elif self.state == self.CENTER:
            self._do_center(ids, boxes, confs, w, h)
        elif self.state == self.ZOOM:
            self._do_zoom(ids, boxes, confs, w, h)
        elif self.state == self.LOCKED:
            self._do_locked(ids, boxes, confs, w, h)

        self._publish_status()
        self._draw_hud(frame, ids, boxes, confs, w, h)
        self._publish_debug_image(frame)
        self._log_status(ids)

    def _run_inference(self, frame):
        kwargs = dict(
            source=frame,
            persist=True,
            classes=[self.target_class],
            tracker=self.tracker,
            conf=self.search_conf,
            verbose=False,
        )
        # 정적 TRT 엔진은 imgsz 가 빌드 크기와 일치해야 한다(불일치 시 크래시).
        # imgsz<=0 이면 생략해 ultralytics 기본값을 쓰게 둔다.
        if self.imgsz and self.imgsz > 0:
            kwargs["imgsz"] = self.imgsz
        results = self.model.track(**kwargs)
        if results and results[0].boxes.id is not None:
            b = results[0].boxes
            ids = b.id.cpu().numpy().astype(int)
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            return ids, xyxy, confs
        return np.array([], dtype=int), np.zeros((0, 4)), np.array([])

    # ---------------------- helpers ----------------------
    @staticmethod
    def _box_center(box):
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    def _norm_error(self, box, w, h):
        tx, ty = self._box_center(box)
        ex = (tx - w / 2.0) / w
        ey = (ty - h / 2.0) / h
        return ex, ey

    def _fill_ratio(self, box, w, h):
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        return max(bw / w, bh / h)

    def _track_gimbal(self, box, w, h):
        """주어진 박스를 중앙으로 가져오는 속도 명령. 정규화 오차 크기 반환."""
        ex, ey = self._norm_error(box, w, h)
        cex = 0.0 if abs(ex) < self.center_deadzone else ex
        cey = 0.0 if abs(ey) < self.center_deadzone else ey
        yaw_cmd = int(np.clip(cex * 100 * self.kp_yaw, -100, 100))
        pitch_cmd = int(np.clip(-cey * 100 * self.kp_pitch, -100, 100))
        self.gimbal.send_speed(yaw_cmd, pitch_cmd)
        return max(abs(ex), abs(ey))

    def _find_box(self, ids, boxes, track_id):
        if len(ids) == 0:
            return None
        where = np.where(ids == track_id)[0]
        if len(where) == 0:
            return None
        return boxes[where[0]]

    def _best_detection(self, ids, boxes, confs):
        """가장 conf 높은 (id, box, conf) 반환. 없으면 None."""
        if len(ids) == 0:
            return None
        i = int(np.argmax(confs))
        return ids[i], boxes[i], confs[i]

    # ---------------------- motion prediction & re-lock ----------------------
    def _reset_motion(self):
        self.last_box = None
        self.vx = 0.0
        self.vy = 0.0

    def _remember_motion(self, box):
        """대상이 보일 때마다 호출: 픽셀 속도(EMA) 와 마지막 박스 갱신."""
        box = np.asarray(box, dtype=float)
        if self.last_box is not None:
            pcx, pcy = self._box_center(self.last_box)
            ccx, ccy = self._box_center(box)
            self.vx = 0.6 * self.vx + 0.4 * (ccx - pcx)
            self.vy = 0.6 * self.vy + 0.4 * (ccy - pcy)
        self.last_box = box

    def _predict_motion(self):
        """놓친 동안: 마지막 박스를 감쇠된 속도만큼 전진시켜 예측 위치 반환."""
        if self.last_box is None:
            return None
        self.vx *= self.vel_decay
        self.vy *= self.vel_decay
        self.last_box = self.last_box + np.array(
            [self.vx, self.vy, self.vx, self.vy], dtype=float)
        return self.last_box

    def _relock(self, ids, boxes, predicted):
        """예측 위치 주변 탐색존 안에서 크기 유사한 검출을 찾아 (id, box) 반환."""
        if predicted is None or len(ids) == 0:
            return None
        px1, py1, px2, py2 = predicted
        bw = max(1.0, px2 - px1)
        bh = max(1.0, py2 - py1)
        last_area = bw * bh
        mx, my = bw * self.relock_search_scale, bh * self.relock_search_scale
        sx1, sy1, sx2, sy2 = px1 - mx, py1 - my, px2 + mx, py2 + my
        best, best_score = None, self.relock_size_sim
        for i, b in enumerate(boxes):
            cx, cy = self._box_center(b)
            if not (sx1 < cx < sx2 and sy1 < cy < sy2):
                continue
            area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            size_sim = min(last_area, area) / max(last_area, area)
            if size_sim > best_score:
                best_score, best = size_sim, (int(ids[i]), b)
        return best

    def _apply_zoom(self, level, force=False):
        level = float(np.clip(level, self.min_zoom, self.max_zoom))
        now = time.time()
        if not force and (now - self.last_zoom_cmd_t) < self.zoom_interval_s:
            return
        self.zoom_level = level
        if self.use_absolute_zoom:
            self.gimbal.set_zoom(level)
        else:
            # 수동 줌: 목표와 현재 추정치 비교해 방향만 보냄
            self.gimbal.manual_zoom(0)
        self.last_zoom_cmd_t = now

    def _adjust_zoom_toward_fill(self, box, w, h):
        """fill ratio 가 목표에 맞도록 줌 미세 조정. 목표 도달 여부 반환."""
        fill = self._fill_ratio(box, w, h)
        if fill < self.target_fill_ratio - self.fill_tolerance:
            if self.use_absolute_zoom:
                self._apply_zoom(self.zoom_level + self.zoom_step)
            else:
                self._manual_zoom_tick(1)
            return False
        if fill > self.target_fill_ratio + self.fill_tolerance:
            if self.use_absolute_zoom:
                self._apply_zoom(self.zoom_level - self.zoom_step)
            else:
                self._manual_zoom_tick(-1)
            return False
        if not self.use_absolute_zoom:
            self._manual_zoom_tick(0)
        return True

    def _manual_zoom_tick(self, direction):
        now = time.time()
        if (now - self.last_zoom_cmd_t) < self.zoom_interval_s and direction != 0:
            return
        # max_zoom 추정 한계: 절대줌이 아니라 정확한 값은 모르나, 풀줌 방지를 위해
        # 추정 zoom_level 을 같이 갱신해 상한에서 더 줌인하지 않도록 한다.
        if direction > 0 and self.zoom_level >= self.max_zoom:
            self.gimbal.manual_zoom(0)
            return
        self.gimbal.manual_zoom(direction)
        self.zoom_level = float(np.clip(
            self.zoom_level + direction * self.zoom_step, self.min_zoom, self.max_zoom))
        self.last_zoom_cmd_t = now

    def _reset_to_search(self, reason=""):
        self.state = self.SEARCH
        self.candidate_id = None
        self.locked_id = None
        self.confirm_count = 0
        self._reset_motion()
        self.gimbal.send_speed(0, 0)
        self._apply_zoom(self.min_zoom, force=True)
        if reason:
            self.get_logger().info(f"-> SEARCH ({reason})")

    # ---------------------- state handlers ----------------------
    def _do_search(self, ids, boxes, confs, w, h):
        self.gimbal.send_speed(0, 0)
        best = self._best_detection(ids, boxes, confs)
        if best is None:
            return
        self.candidate_id, cand_box, conf = best
        self.last_seen_t = time.time()
        self._reset_motion()
        self._remember_motion(cand_box)
        self.state = self.CENTER
        self.get_logger().info(
            f"-> CENTER candidate id={self.candidate_id} conf={conf:.2f}")

    def _do_center(self, ids, boxes, confs, w, h):
        box = self._find_box(ids, boxes, self.candidate_id)
        if box is None:
            if time.time() - self.last_seen_t > self.lost_timeout_s:
                self._reset_to_search("candidate lost")
            else:
                self.gimbal.send_speed(0, 0)
            return
        self.last_seen_t = time.time()
        self._remember_motion(box)
        err = self._track_gimbal(box, w, h)
        if err < self.center_threshold:
            self.state = self.ZOOM
            self.confirm_count = 0
            self.get_logger().info(f"-> ZOOM (centered, err={err:.3f})")

    def _do_zoom(self, ids, boxes, confs, w, h):
        # 후보 추적 + conf 확인
        idx = np.where(ids == self.candidate_id)[0] if len(ids) else []
        if len(idx) == 0:
            # 잠깐 놓침: 예측 + 재락으로 후보를 이어간다.
            predicted = self._predict_motion()
            relock = self._relock(ids, boxes, predicted)
            if relock is not None:
                self.candidate_id, box = relock
                conf = 0.0
            elif time.time() - self.last_seen_t > self.lost_timeout_s:
                self._reset_to_search("candidate lost during zoom")
                return
            else:
                self.predicted_box = predicted
                if predicted is not None:
                    self._track_gimbal(predicted, w, h)
                else:
                    self.gimbal.send_speed(0, 0)
                return
        else:
            i = idx[0]
            box, conf = boxes[i], confs[i]
        self.last_seen_t = time.time()
        self._remember_motion(box)

        self._track_gimbal(box, w, h)
        at_target = self._adjust_zoom_toward_fill(box, w, h)

        # 클로즈업 상태에서 높은 conf 가 연속되면 조류 확정 -> 락
        if at_target and conf >= self.confirm_conf:
            self.confirm_count += 1
        else:
            self.confirm_count = max(0, self.confirm_count - 1)

        if self.confirm_count >= self.confirm_frames:
            self.locked_id = self.candidate_id
            self.state = self.LOCKED
            self.get_logger().info(
                f"=== LOCKED id={self.locked_id} (bird confirmed, conf={conf:.2f}) ===")

    def _do_locked(self, ids, boxes, confs, w, h):
        box = self._find_box(ids, boxes, self.locked_id)
        if box is not None:
            # 정상 추적
            self.last_seen_t = time.time()
            self._remember_motion(box)
            self._track_gimbal(box, w, h)
            self._adjust_zoom_toward_fill(box, w, h)
            return

        # 놓침: 모션 예측 -> 같은 자리 근처 검출로 재락 시도
        predicted = self._predict_motion()
        self.predicted_box = predicted
        relock = self._relock(ids, boxes, predicted)
        if relock is not None:
            new_id, new_box = relock
            if new_id != self.locked_id:
                self.get_logger().warn(f"Re-locked {self.locked_id} -> {new_id}")
            self.locked_id = new_id
            self.last_seen_t = time.time()
            self._remember_motion(new_box)
            self._track_gimbal(new_box, w, h)
            self._adjust_zoom_toward_fill(new_box, w, h)
            return

        # 재락 실패: 타임아웃 전까지는 예측 위치로 짐벌을 계속 보낸다(정지하지 않음)
        if time.time() - self.last_seen_t > self.lost_timeout_s:
            self._reset_to_search("locked target lost")
        elif predicted is not None:
            self._track_gimbal(predicted, w, h)
        else:
            self.gimbal.send_speed(0, 0)

    # ---------------------- output ----------------------
    def _publish_status(self):
        self.locked_pub.publish(Bool(data=(self.state == self.LOCKED)))
        if self.locked_id is not None:
            self.locked_id_pub.publish(Int32(data=int(self.locked_id)))

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_frame_t
        self.last_frame_t = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst

    def _log_status(self, ids):
        now = time.time()
        if now - self.last_log_t >= 1.0:
            self.get_logger().info(
                f"[{self.state}] cand={self.candidate_id} lock={self.locked_id} "
                f"zoom={self.zoom_level:.1f} ids={list(ids)} fps={self.fps:.1f}")
            self.last_log_t = now

    def _draw_hud(self, frame, ids, boxes, confs, w, h):
        color = {self.SEARCH: (0, 255, 255), self.CENTER: (0, 165, 255),
                 self.ZOOM: (255, 0, 0), self.LOCKED: (0, 0, 255)}[self.state]
        # crosshair
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 24, 1)
        # detections
        focus = self.locked_id if self.state == self.LOCKED else self.candidate_id
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = map(int, boxes[i])
            c = (0, 0, 255) if tid == focus else (0, 200, 0)
            th = 3 if tid == focus else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, th)
            cv2.putText(frame, f"ID:{tid} {confs[i]:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        # 놓친 동안 예측 위치(노란 점선 박스)
        if self.predicted_box is not None:
            px1, py1, px2, py2 = map(int, self.predicted_box)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            cv2.putText(frame, "PREDICT", (px1, py1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        text = (f"{self.state} zoom:{self.zoom_level:.1f}x "
                f"lock:{self.locked_id} conf#:{self.confirm_count} "
                f"fps:{self.fps:.1f}")
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)

    def _publish_debug_image(self, frame):
        if self.frame_count % self.publish_every_n != 0:
            return
        if self.debug_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.debug_scale, fy=self.debug_scale)
        _, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.debug_image_pub.publish(msg)

    def destroy_node(self):
        try:
            self.gimbal.send_speed(0, 0)
        except Exception:  # noqa: BLE001
            pass
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BirdLockTrackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
