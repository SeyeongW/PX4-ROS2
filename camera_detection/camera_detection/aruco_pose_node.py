#!/usr/bin/env python3
"""ArUco 마커 pose 추정 노드 (실기체 하드웨어용).

SITL 의 aruco_detector_node 는 정규화 오프셋(-1..1)만 뽑고, 고도로 픽셀→미터를
근사(cam_hfov)했습니다. 실기체에서는 렌즈 왜곡·화각 오차 때문에 그 근사가
부정확하므로, 이 노드는 **카메라 캘리브레이션(내부 파라미터 K, 왜곡 D)** 과
**마커 실측 크기**로 solvePnP 를 풀어 마커의 **실제 3D 위치(m)** 를 카메라
광학 프레임에서 바로 계산합니다.

카메라 광학 프레임 규약 (OpenCV):
    +x = 이미지 오른쪽, +y = 이미지 아래, +z = 렌즈가 보는 방향(하방 카메라면 지면 쪽)
따라서 tvec = (오른쪽 m, 아래 m, 마커까지 거리 m). 하방 카메라를 정하방으로 달면
tvec.z ≈ 마커 윗면까지의 높이가 됩니다.

토픽:
  in  <image_topic>            sensor_msgs/Image        (raw, sensor QoS)
  in  <camera_info_topic>      sensor_msgs/CameraInfo   (K, D — 없으면 calib_file 폴백)
  out /perception/aruco_detected   std_msgs/Bool
  out /perception/marker_pose      geometry_msgs/PoseStamped  (카메라 프레임, tvec+rvec)
  out /perception/aruco_offset     geometry_msgs/Point        (정규화 -1..1, 디버그/호환용)
  out /perception/aruco_debug/compressed  sensor_msgs/CompressedImage (publish_debug 시)

정밀착륙 제어 노드(precland_hw_node)는 /perception/marker_pose 의 position(tvec)을
그대로 소비합니다.
"""

import os

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool


class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose_node')

        # --- 파라미터 -------------------------------------------------------
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        # 마커 한 변의 실측 크기 (m). 인쇄한 마커 실물을 자로 재서 정확히 넣어야
        # pose(특히 거리 tvec.z)가 맞습니다.
        self.declare_parameter('marker_size', 0.15)
        # 특정 ID만 착륙 표적으로 쓰려면 지정(-1 = 처음 검출된 마커 아무거나).
        self.declare_parameter('marker_id', -1)
        # /camera_info 토픽이 없을 때 읽을 캘리브레이션 파일(ROS camera_info yaml).
        self.declare_parameter('calib_file', '')
        self.declare_parameter('publish_debug', True)

        image_topic = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        dict_name = self.get_parameter('aruco_dict').value
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.marker_id = int(self.get_parameter('marker_id').value)
        calib_file = self.get_parameter('calib_file').value
        self.publish_debug = bool(self.get_parameter('publish_debug').value)

        dict_map = {
            'DICT_4X4_50': aruco.DICT_4X4_50,
            'DICT_5X5_100': aruco.DICT_5X5_100,
            'DICT_6X6_250': aruco.DICT_6X6_250,
        }
        adict = aruco.getPredefinedDictionary(dict_map.get(dict_name, aruco.DICT_4X4_50))
        params = aruco.DetectorParameters()
        try:
            self._detector = aruco.ArucoDetector(adict, params)
            self._detect = lambda g: self._detector.detectMarkers(g)
        except AttributeError:
            self._detect = lambda g: aruco.detectMarkers(g, adict, parameters=params)

        # --- 카메라 내부 파라미터 ------------------------------------------
        self.K = None      # 3x3 카메라 행렬
        self.D = None      # 왜곡 계수
        if calib_file:
            self._load_calib_file(calib_file)

        # --- Pub/Sub --------------------------------------------------------
        self.detected_pub = self.create_publisher(Bool, '/perception/aruco_detected', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/perception/marker_pose', 10)
        self.offset_pub = self.create_publisher(Point, '/perception/aruco_offset', 10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(
                CompressedImage, '/perception/aruco_debug/compressed', 10)

        # camera_info: 보통 RELIABLE 로 발행되지만 드라이버마다 다를 수 있어
        # sensor QoS(BEST_EFFORT)와 호환되도록 depth 넉넉히.
        self.create_subscription(CameraInfo, info_topic, self._info_cb, 10)
        self.create_subscription(Image, image_topic, self._image_cb,
                                 qos_profile_sensor_data)

        self.get_logger().info(
            f'ArUco pose ready — dict={dict_name}, marker={self.marker_size} m, '
            f'image={image_topic}, camera_info={info_topic}')
        if self.K is None:
            self.get_logger().warn(
                'no camera intrinsics yet — waiting for camera_info (or set calib_file). '
                'pose 는 K/D 를 받은 뒤부터 발행됩니다.')

    # -----------------------------------------------------------------------
    def _load_calib_file(self, path):
        if not os.path.isfile(path):
            self.get_logger().error(f'calib_file not found: {path}')
            return
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        try:
            self.K = np.array(data['camera_matrix']['data'],
                              dtype=np.float64).reshape(3, 3)
            self.D = np.array(data['distortion_coefficients']['data'],
                              dtype=np.float64).reshape(1, -1)
            self.get_logger().info(f'loaded intrinsics from {path}')
        except (KeyError, TypeError) as e:
            self.get_logger().error(f'bad calib_file format ({path}): {e}')

    def _info_cb(self, msg: CameraInfo):
        # 토픽 camera_info 가 파일보다 우선(라이브 값). 한 번 받으면 계속 갱신.
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64).reshape(1, -1) \
            if len(msg.d) else np.zeros((1, 5))

    # -----------------------------------------------------------------------
    @staticmethod
    def _imgmsg_to_bgr(msg: Image):
        # cv_bridge 우회(NumPy 2.x 에서 컴파일 모듈이 깨짐). 수동 디코드.
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        h, w = msg.height, msg.width
        enc = msg.encoding
        if enc in ('rgb8', 'bgr8'):
            img = buf.reshape(h, w, 3)
            if enc == 'rgb8':
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        if enc in ('mono8', '8UC1'):
            return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(buf.reshape(h, w, 3))

    def _estimate_pose(self, marker_corners):
        # 한 변 s 인 정사각 마커를 원점 중심 평면에 놓고 solvePnP.
        # ArUco corner 순서: TL, TR, BR, BL (시계방향).
        s = self.marker_size
        objp = np.array([[-s / 2,  s / 2, 0.0],
                         [ s / 2,  s / 2, 0.0],
                         [ s / 2, -s / 2, 0.0],
                         [-s / 2, -s / 2, 0.0]], dtype=np.float32)
        imgp = marker_corners.reshape(4, 2).astype(np.float32)
        # 평면 정사각엔 IPPE_SQUARE 가 가장 안정적(가능하면 사용).
        flags = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, self.K, self.D, flags=flags)
        if not ok:
            return None, None
        return rvec.reshape(3), tvec.reshape(3)

    @staticmethod
    def _rvec_to_quat(rvec):
        R, _ = cv2.Rodrigues(rvec)
        t = R[0, 0] + R[1, 1] + R[2, 2]
        if t > 0:
            sq = np.sqrt(t + 1.0) * 2
            w = 0.25 * sq
            x = (R[2, 1] - R[1, 2]) / sq
            y = (R[0, 2] - R[2, 0]) / sq
            z = (R[1, 0] - R[0, 1]) / sq
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            sq = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / sq
            x = 0.25 * sq
            y = (R[0, 1] + R[1, 0]) / sq
            z = (R[0, 2] + R[2, 0]) / sq
        elif R[1, 1] > R[2, 2]:
            sq = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / sq
            x = (R[0, 1] + R[1, 0]) / sq
            y = 0.25 * sq
            z = (R[1, 2] + R[2, 1]) / sq
        else:
            sq = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / sq
            x = (R[0, 2] + R[2, 0]) / sq
            y = (R[1, 2] + R[2, 1]) / sq
            z = 0.25 * sq
        return x, y, z, w

    # -----------------------------------------------------------------------
    def _image_cb(self, msg: Image):
        frame = self._imgmsg_to_bgr(msg)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)

        detected = False
        offset = Point()
        tvec = rvec = None
        chosen = None

        if ids is not None and len(ids) > 0:
            idx = 0
            if self.marker_id >= 0:
                match = np.where(ids.flatten() == self.marker_id)[0]
                idx = int(match[0]) if len(match) else None
            if idx is not None:
                chosen = corners[idx]
                pts = chosen[0]
                mx = float(np.mean(pts[:, 0]))
                my = float(np.mean(pts[:, 1]))
                offset.x = (mx - cx) / (w / 2.0)   # 정규화(디버그/호환)
                offset.y = (my - cy) / (h / 2.0)
                # pose 는 캘리브레이션이 있어야만.
                if self.K is not None:
                    rvec, tvec = self._estimate_pose(chosen)
                    detected = tvec is not None
                else:
                    detected = True   # 검출은 됐으나 pose 없음

        # --- 발행 ----------------------------------------------------------
        self.offset_pub.publish(offset)
        self.detected_pub.publish(Bool(data=detected and tvec is not None))

        if tvec is not None:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(tvec[0])
            ps.pose.position.y = float(tvec[1])
            ps.pose.position.z = float(tvec[2])
            qx, qy, qz, qw = self._rvec_to_quat(rvec)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            self.pose_pub.publish(ps)

        if self.publish_debug:
            self._draw(frame, corners, ids, cx, cy, offset, chosen,
                       tvec, rvec, msg.header)

    def _draw(self, frame, corners, ids, cx, cy, offset, chosen,
              tvec, rvec, header):
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
        if tvec is not None:
            cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec,
                              self.marker_size * 0.5)
            label = (f'x={tvec[0]:+.2f} y={tvec[1]:+.2f} z={tvec[2]:.2f} m')
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        elif chosen is not None:
            cv2.putText(frame, 'NO CALIB (pose off)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 2)
        else:
            cv2.putText(frame, 'SEARCHING...', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
        ok, enc = cv2.imencode('.jpg', frame)
        if ok:
            dbg = CompressedImage()
            dbg.header = header
            dbg.format = 'jpeg'
            dbg.data = enc.tobytes()
            self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
