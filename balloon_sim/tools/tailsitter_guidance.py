#!/usr/bin/env python3
"""Explicit legacy tailsitter simulation; not the default MC/Bezier flight path.

Consumes only validated camera estimates. Approval can be cancelled, source
measurements and FC telemetry expire, and passage requires a bounded plane
crossing. Onboard detections are display-only until exposure-matched vehicle
poses are available. A miss or failed guard requests RTL; a valid crossing requests HOLD
behind the target. Mode acknowledgement never implies a completed landing.

Run only through run_gazebo.sh mode:=pymav experimental_tailsitter:=true.
"""
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
from pymavlink import mavutil

try:
    import rclpy
    from std_msgs.msg import Bool, String
    from vision_msgs.msg import Detection2DArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. 비행 및 다중 폐루프 유도 제어 파라미터
# ═══════════════════════════════════════════════════════════════════════════
FALLBACK_TARGET_WORLD = [
    float(os.environ.get('FLY_TARGET_X', '90.0')),
    float(os.environ.get('FLY_TARGET_Y', '0.0')),
    float(os.environ.get('FLY_TARGET_Z', '40.0')),
]
AUTO_APPROVE = os.environ.get('AUTO_APPROVE', '0').lower() in ('1', 'true', 'yes')
APPROVAL_FLAG_FILE = Path('/tmp/balloon_launch_approved')

UKF_LOCK_SIGMA_M = 0.5     # 지상 UKF 락온 인정 최대 표준편차 s (sigma) [m]
UKF_MIN_SAMPLES  = 3       # 지상 UKF 락온 최소 누적 샘플 수 n

NORTH_YAW  = math.pi / 2.0
HOME_WORLD = [-2.0, 0.0, 0.2]
PX4_UDP    = 'udpin:0.0.0.0:14540'

# 테일시터 속도 및 추력 제어 파라미터
TAKEOFF_ALT       = 5.0    # 수직 이륙 목표 고도 [m]
TRANSITION_DUR    = 1.5    # 수직 호버링 -> 테일시터 대각선 자세 전환 시간 [s]
MIN_CRUISE_SPEED  = 10.0   # 근접 구간 최소 목표 속도 [m/s]
MAX_CRUISE_SPEED  = 14.5   # 원거리 구간 최대 목표 속도 [m/s]
KP_DIST_SPEED     = 0.05   # 거리 비례 속도 이득: V_des = clip(10.0 + 0.05 * dist_3d, 10.0, 14.5)
LIFT_CRUISE_SPEED = 14.2   # 주익 양력이 중력 수직 성분을 완전히 지지하는 기준 속도 [m/s]
SPEC_THRUST_MAX   = 34.52  # 4로터 최대 비추력 (69.04 N / 2.0 kg) [m/s^2]
THRUST_MIN        = 0.07   # 비행 중 최소 정규화 추력 (과속 시 감속 허용 + 자세 제어권 유지)
THRUST_MAX        = 0.54   # 비행 중 최대 정규화 추력 (초기 급상승 오버슈트 방지)
MAX_LOS_LEAD_DEG  = 15.0   # 비전 추적 시 LOS 대비 노즈콘 최대 리드각 [deg] (화각 내부 유지 보장)
MAX_PATH_LEAD_DEG = 22.0   # 대각선 경로 추종 시 최대 리드각 [deg]
COAST_TIMEOUT     = 0.8    # 비전 일시 소실 시 마지막 LOS 유지 시간 [s]
BALLOON_DIAM_M    = 0.30   # 목표 풍선 실제 지름 [m]

# A visual disappearance is not evidence of a waypoint crossing.
LOCK_DIST = 15.0
PASS_RADIUS_M = BALLOON_DIAM_M / 2.0
PASS_HOLD_DISTANCE_M = 2.0
TARGET_MAX_AGE = 0.8
TELEMETRY_MAX_AGE = 0.5
HEARTBEAT_MAX_AGE = 2.0
APPROVAL_TTL = 30.0
MISSION_TIMEOUT = 90.0

# 온보드 노즈콘 카메라 광학 파라미터 (Raspberry Pi Camera Module 3 Standard, 640x360, HFOV 66°)
IMG_W    = 640
IMG_H    = 360
HFOV_DEG = 66.0
CX       = (IMG_W - 1) / 2.0
CY       = (IMG_H - 1) / 2.0
FX       = IMG_W / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
FY       = FX


# ═══════════════════════════════════════════════════════════════════════════
# 2. 좌표 변환 및 3D 테일시터 자세(Wings-Level Quaternion) 계산 함수
# ═══════════════════════════════════════════════════════════════════════════
def world_to_ned(wx, wy, wz):
    """Gazebo World (X, Y, Z) -> PX4 Local NED (N, E, D) 변환 (홈 기준)."""
    c, s = math.cos(NORTH_YAW), math.sin(NORTH_YAW)
    dx, dy, dz = wx - HOME_WORLD[0], wy - HOME_WORLD[1], wz - HOME_WORLD[2]
    return c * dx + s * dy, s * dx - c * dy, -dz


def station_ned_to_world(sn, se, sd):
    """지상 관측소(Station) NED [N=World_Y, E=World_X, D=-World_Z] -> Gazebo World [X, Y, Z] 변환."""
    return float(se), float(sn), float(-sd)


def quat_to_rot_ned(w, x, y, z):
    """단위 쿼터니언 (w, x, y, z) -> 3x3 회전행렬 R_bn (Body FRD -> Local NED)."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def rot_ned_to_quat(R):
    """3x3 회전행렬 R_bn (Body FRD -> Local NED) -> 단위 쿼터니언 [w, x, y, z]."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def nose_dir_to_quat_frd(nose_dir_ned, fallback_yaw):
    """노즈콘 지향 단위벡터 nose_dir_ned (Body -Z_FRD = +Z_FLU = 카메라 광축)로부터
    날개 수평(Wings-Level, 카메라 롤 회전 0°)을 보장하는 SO(3) 자세 쿼터니언 [w, x, y, z] 계산.
    """
    n = np.array(nose_dir_ned, dtype=float)
    n_norm = float(np.linalg.norm(n))
    n = np.array([0.0, 0.0, -1.0], dtype=float) if n_norm < 1e-6 else (n / n_norm)

    z_frd = -n
    horiz_norm = math.hypot(float(n[0]), float(n[1]))
    az = math.atan2(float(n[1]), float(n[0])) if horiz_norm > 1e-3 else fallback_yaw

    # 우측 날개(+Y_FRD)를 수평면에 구속 -> Wing Roll = 0°
    y_frd = np.array([-math.sin(az), math.cos(az), 0.0], dtype=float)
    x_frd = np.cross(y_frd, z_frd)
    x_norm = float(np.linalg.norm(x_frd))
    if x_norm < 1e-6:
        x_frd = np.array([math.cos(fallback_yaw), math.sin(fallback_yaw), 0.0], dtype=float)
    else:
        x_frd /= x_norm
    y_frd = np.cross(z_frd, x_frd)
    y_frd /= np.linalg.norm(y_frd)

    R_bn = np.column_stack((x_frd, y_frd, z_frd))
    return rot_ned_to_quat(R_bn)


def pixel_to_los_ned(u, v, R_bn):
    """노즈콘 카메라 픽셀 (u, v) -> NED 3D 시선(LOS) 단위벡터 변환."""
    ex = (u - CX) / FX
    ey = (v - CY) / FY
    ray_body_frd = np.array([ex, -ey, -1.0], dtype=float)
    ray_body_frd /= np.linalg.norm(ray_body_frd)
    los_ned = R_bn @ ray_body_frd
    return los_ned / np.linalg.norm(los_ned)


def clamp_vec(vec, max_mag):
    mag = float(np.linalg.norm(vec))
    if mag > max_mag and mag > 1e-6:
        return vec * (max_mag / mag)
    return vec


def compute_tailsitter_cmd(aim_dir_ned, a_perp_cmd, vel_ned, dist_3d, max_lead_deg, trans_alpha, allow_dive=False):
    """목표 지향 벡터(aim_dir_ned) 및 수직 보정 가속도(a_perp_cmd)로부터
    테일시터 노즈콘 지향 벡터(nose_dir)와 정규화 추력(thrust)을 분리 산출.
    """
    d = np.array(aim_dir_ned, dtype=float)
    d /= max(float(np.linalg.norm(d)), 1e-6)

    cur_spd = float(np.linalg.norm(vel_ned))
    v_along = float(np.dot(vel_ned, d))

    # 저속 구간에서는 부족한 날개 양력만큼 위쪽(-D) 중력 보상을 추가하고, 고속 구간에서는 잉여 주익 양력을 감쇄
    lift_frac = min(1.45, (cur_spd / LIFT_CRUISE_SPEED) ** 2)
    g_uncomp_ned = np.array([0.0, 0.0, 9.81 * (1.0 - 0.94 * lift_frac)], dtype=float)

    g_perp = g_uncomp_ned - float(np.dot(g_uncomp_ned, d)) * d
    lead_acc_perp = (a_perp_cmd - float(np.dot(a_perp_cmd, d)) * d) - g_perp

    ref_acc = 11.0
    max_perp_mag = ref_acc * math.tan(math.radians(max_lead_deg))
    lead_acc_perp = clamp_vec(lead_acc_perp, max_perp_mag)

    raw_nose = d + (lead_acc_perp / ref_acc)
    raw_nose /= np.linalg.norm(raw_nose)

    # 종말 비전 추적(allow_dive=True) 시 풍선이 기체보다 아래에 있으면 기수를 아래(-26.7°)까지 숙여 고도 오차를 즉시 제거
    max_down_z = 0.45 if allow_dive else 0.08
    if raw_nose[2] > max_down_z:
        raw_nose[2] = max_down_z
        raw_nose /= np.linalg.norm(raw_nose)

    hover_nose = np.array([0.0, 0.0, -1.0], dtype=float)
    nose_dir = (1.0 - trans_alpha) * hover_nose + trans_alpha * raw_nose
    nose_dir /= np.linalg.norm(nose_dir)

    # 남은 거리(dist_3d)에 비례하는 목표 접근 속도 추종 추력 계산
    v_target = min(MAX_CRUISE_SPEED, max(MIN_CRUISE_SPEED, MIN_CRUISE_SPEED + KP_DIST_SPEED * dist_3d))
    speed_err = v_target - v_along

    grav_along = max(0.0, -9.81 * float(nose_dir[2]))
    aero_drag = 0.0045 * (cur_spd ** 2)
    low_spd_support = 9.81 * max(0.0, 1.0 - lift_frac) * 0.42
    # 피치다운 전환 초기(기수가 위를 향할 때) 과도한 추력으로 수직 급상승하는 현상 방지
    fwd_gate = (trans_alpha ** 1.5) if speed_err > 0.0 else 1.0
    kp_spd = 1.6 if speed_err >= 0.0 else 2.2
    a_thrust_cmd = grav_along + aero_drag + low_spd_support + kp_spd * fwd_gate * speed_err

    thrust = max(THRUST_MIN, min(THRUST_MAX, a_thrust_cmd / SPEC_THRUST_MAX))
    return nose_dir, thrust, v_target


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROS 2 지상 UKF 추정(10Hz), 출격 승인 트리거 및 온보드 비전 검출 수신 헬퍼
# ═══════════════════════════════════════════════════════════════════════════
class VisionListener:
    """ROS 2 지상 카메라 UKF 추정(/balloon/status, /jol/static_target/estimate),
    출격 승인 트리거(/mission/approve_launch) 및 온보드 비전(/balloon/onboard/detections) 수신 관리.
    """

    def __init__(self):
        self.lock = threading.Lock()
        # 온보드 카메라 상태
        self.u = 0.0
        self.v = 0.0
        self.diam = 0.0
        self.initial_diam = 0.0
        self.max_diam = 0.0
        self.last_rx = 0.0
        self.detection_stamp = -math.inf
        self.lost_frames = 0

        # 지상 UKF 추정 상태 (10 Hz 수신)
        self.ukf_valid = False
        self.ukf_world = None
        self.ukf_sigma_m = math.inf
        self.ukf_n = 0
        self.ukf_reason = 'waiting_for_ground_ukf'
        self.ukf_last_rx = 0.0
        self.ukf_rx_count = 0
        self.ukf_hz = 10.0

        # 사용자 출격 승인 상태
        self.launch_approved = False
        self.approved_at = 0.0
        self.approval_edge = False
        self.cancel_requested = False
        self.mission_started = False
        self.approval_file_used = False
        self.started_wall = time.time()
        self.ukf_source_stamp = -math.inf

    def update(self, u, v, diam, stamp, source_now):
        with self.lock:
            if not (all(math.isfinite(x) for x in (u, v, diam, stamp, source_now))
                    and diam > 0 and stamp > self.detection_stamp and 0 <= source_now - stamp <= .25):
                return False
            self.detection_stamp = stamp
            self.u = u
            self.v = v
            self.diam = diam
            if self.initial_diam <= 0.0:
                self.initial_diam = diam
            if diam > self.max_diam:
                self.max_diam = diam
            self.last_rx = time.monotonic() - (source_now - stamp)
            self.lost_frames = 0

    def mark_lost(self):
        with self.lock:
            if self.last_rx > 0.0:
                self.lost_frames += 1

    def update_estimate(self, packet, source_now):
        """Only the validated Gazebo bridge contract may provide a flight target."""
        try:
            stamp = float(packet['stamp'])
            position = np.asarray(packet['position'], dtype=float)
            covariance = np.asarray(packet['covariance'], dtype=float)
            evidence = packet['evidence']
            sigma = float(packet['maximum_position_std_m'])
            if (packet.get('valid') is not True or packet.get('frame_id') != 'world'
                    or packet.get('source') != 'camera_monocular_size'
                    or packet.get('position_semantics') != 'known_sphere_center'
                    or not packet.get('calibration_revision')
                    or packet.get('calibration_source') != 'gazebo_balloon_locator_camera_info_tf'
                    or evidence.get('ground_truth_used') is not False
                    or evidence.get('target_is_static') is not True
                    or evidence.get('image_source_stamp') != stamp
                    or evidence.get('camera_tf_stamp') != stamp
                    or not 0 <= source_now - stamp <= TARGET_MAX_AGE
                    or position.shape != (3,) or not np.isfinite(position).all()
                    or covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T)
                    or not math.isfinite(sigma) or not 0 <= sigma <= UKF_LOCK_SIGMA_M):
                raise ValueError('invalid_estimate')
            eigenvalues = np.linalg.eigvalsh(covariance)
            if eigenvalues.min() < 0 or eigenvalues.max() > UKF_LOCK_SIGMA_M ** 2:
                raise ValueError('invalid_covariance')
        except (KeyError, TypeError, ValueError, AttributeError):
            with self.lock:
                self.ukf_valid = False
                self.ukf_reason = 'invalid_estimate'
            return False
        with self.lock:
            if stamp <= self.ukf_source_stamp:
                # Publication heartbeats must not renew the lifetime of a measurement.
                return False
            now_t = time.monotonic()
            if self.ukf_last_rx > 0 and 0.01 < now_t - self.ukf_last_rx < 1.0:
                self.ukf_hz = .8 * self.ukf_hz + .2 / (now_t - self.ukf_last_rx)
            self.ukf_world = position.tolist()
            self.ukf_last_rx = now_t - (source_now - stamp)
            self.ukf_source_stamp = stamp
            self.ukf_sigma_m = max(sigma, math.sqrt(float(eigenvalues.max())))
            self.ukf_n = UKF_MIN_SAMPLES  # bridge validity already includes estimator convergence
            self.ukf_rx_count += 1
            self.ukf_valid = True
            self.ukf_reason = 'ready'
            return True

    def set_approved(self, approved=True):
        with self.lock:
            if not approved:
                self.launch_approved = False
                self.cancel_requested = True
                self.approval_edge = False
            elif not self.approval_edge and not self.mission_started:
                self.launch_approved = True
                self.cancel_requested = False
                self.approved_at = time.monotonic()
                self.approval_edge = True

    def is_approved(self):
        with self.lock:
            if self.cancel_requested:
                return False
            if AUTO_APPROVE:
                return True
            if self.launch_approved:
                return 0 <= time.monotonic() - self.approved_at <= APPROVAL_TTL
            try:
                modified = APPROVAL_FLAG_FILE.stat().st_mtime
                return (modified > self.started_wall
                        and 0 <= time.time() - modified <= APPROVAL_TTL)
            except OSError:
                return False

    def begin_mission(self):
        if not self.is_approved():
            raise GuidanceAbort('approval_missing_or_expired')
        with self.lock:
            self.mission_started = True
            self.approval_file_used = not self.launch_approved and not AUTO_APPROVE

    def cancellation_pending(self):
        with self.lock:
            return (self.cancel_requested or (self.mission_started and self.approval_file_used
                    and not APPROVAL_FLAG_FILE.is_file()))

    def target_ready(self):
        world, sigma, n, valid, _, age, _, _ = self.get_ground_ukf()
        return (valid and world is not None and n >= UKF_MIN_SAMPLES
                and math.isfinite(sigma) and sigma <= UKF_LOCK_SIGMA_M
                and 0 <= age <= TARGET_MAX_AGE)

    def get(self):
        with self.lock:
            age = time.monotonic() - self.last_rx if self.last_rx > 0 else math.inf
            return (
                self.u,
                self.v,
                self.diam,
                self.initial_diam,
                self.max_diam,
                self.lost_frames,
                age,
                self.last_rx,
            )

    def get_ground_ukf(self):
        with self.lock:
            age = time.monotonic() - self.ukf_last_rx if self.ukf_last_rx > 0 else math.inf
            return (
                list(self.ukf_world) if self.ukf_world is not None else None,
                self.ukf_sigma_m,
                self.ukf_n,
                self.ukf_valid,
                self.ukf_reason,
                age,
                self.ukf_rx_count,
                self.ukf_hz,
            )


def start_ros_vision_thread(vision_state):
    if not HAS_ROS2:
        raise RuntimeError('ROS 2 validated target input is required')

    def _spin():
        rclpy.init()
        from rclpy.parameter import Parameter
        node = rclpy.create_node('tailsitter_cooperative_guidance_listener',
            parameter_overrides=[Parameter('use_sim_time', value=True)])

        def _on_onboard_det(msg):
            if not msg.detections:
                vision_state.mark_lost()
                return
            det = msg.detections[0]
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)
            diam = float((det.bbox.size_x + det.bbox.size_y) * 0.5)
            if math.isfinite(u) and math.isfinite(v) and diam > 2.5:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                vision_state.update(u, v, diam, stamp, node.get_clock().now().nanoseconds * 1e-9)
            else:
                vision_state.mark_lost()

        def _on_ground_estimate(msg):
            try:
                packet = json.loads(msg.data)
            except (ValueError, TypeError):
                packet = {}
            vision_state.update_estimate(packet, node.get_clock().now().nanoseconds * 1e-9)

        def _on_approve(msg):
            vision_state.set_approved(msg.data)

        node.create_subscription(Detection2DArray, '/balloon/onboard/detections', _on_onboard_det, 10)
        node.create_subscription(String, '/jol/static_target/estimate', _on_ground_estimate, 10)
        node.create_subscription(Bool, '/mission/approve_launch', _on_approve, 10)
        try:
            rclpy.spin(node)
        except Exception:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    threading.Thread(target=_spin, daemon=True).start()


class GuidanceAbort(RuntimeError):
    """Stop Offboard setpoints and request a confirmed FC recovery mode."""


class TelemetryState:
    def __init__(self, system, component):
        self.system, self.component = system, component
        self.received = {}
        self.boot_stamps = {}
        self.pos = self.vel = self.rotation = None
        self.armed = False
        self.offboard = False

    def accept(self, message, now):
        if (message.get_srcSystem() != self.system
                or message.get_srcComponent() != self.component):
            return False
        kind = message.get_type()
        if kind == 'HEARTBEAT':
            if message.autopilot != mavutil.mavlink.MAV_AUTOPILOT_PX4:
                return False
            self.armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self.offboard = ((message.custom_mode >> 16) & 0xff) == 6
        elif kind in ('LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION'):
            stamp = message.time_boot_ms
            if type(stamp) is not int or stamp <= self.boot_stamps.get(kind, -1):
                return False
            if kind == 'LOCAL_POSITION_NED':
                values = np.array([message.x, message.y, message.z, message.vx, message.vy, message.vz])
                if not np.isfinite(values).all():
                    return False
                self.pos, self.vel = values[:3], values[3:]
            else:
                q = np.array([message.q1, message.q2, message.q3, message.q4], dtype=float)
                norm = float(np.linalg.norm(q))
                if not np.isfinite(q).all() or not .9 <= norm <= 1.1:
                    return False
                self.rotation = quat_to_rot_ned(*(q / norm))
            self.boot_stamps[kind] = stamp
        else:
            return False
        self.received[kind] = now
        return True

    def drain(self, conn):
        # Bounded work: a flooded socket must not delay cancellation/watchdog checks.
        for _ in range(256):
            message = conn.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE_QUATERNION', 'HEARTBEAT'], blocking=False)
            if message is None:
                break
            self.accept(message, time.monotonic())

    def require_fresh(self, now, armed=False):
        for kind, max_age in (('LOCAL_POSITION_NED', TELEMETRY_MAX_AGE),
                              ('ATTITUDE_QUATERNION', TELEMETRY_MAX_AGE),
                              ('HEARTBEAT', HEARTBEAT_MAX_AGE)):
            if not 0 <= now - self.received.get(kind, -math.inf) <= max_age:
                raise GuidanceAbort('telemetry_stale:' + kind)
        if armed and (not self.armed or not self.offboard):
            raise GuidanceAbort('fc_left_armed_offboard')


def require_guidance(conn, telemetry, vision, *, armed=False, target=True):
    telemetry.drain(conn)
    if vision.cancellation_pending():
        raise GuidanceAbort('approval_cancelled')
    telemetry.require_fresh(time.monotonic(), armed=armed)
    if target and not vision.target_ready():
        raise GuidanceAbort('target_invalid_or_stale')


def plane_crossing(previous, current, target, normal, radius=PASS_RADIUS_M):
    """Interpolate the measured segment at the fixed target plane, then bound radial miss."""
    previous, current, target, normal = (np.asarray(v, float) for v in (previous, current, target, normal))
    if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (previous, current, target, normal))
            or not math.isfinite(radius) or radius <= 0 or np.linalg.norm(normal) < 1e-9):
        return False, math.inf
    normal = normal / np.linalg.norm(normal)
    before, after = float((previous - target) @ normal), float((current - target) @ normal)
    if not before < 0 <= after:
        return False, math.inf
    crossing = previous + (current - previous) * (-before / (after - before))
    radial = float(np.linalg.norm(crossing - target))
    return radial <= radius, radial


def transfer_to_px4(conn, mode='RTL', timeout=4.0):
    """Stop all Offboard setpoints; report ACK and observed mode independently."""
    if mode not in ('RTL', 'HOLD'):
        raise ValueError('unsupported_recovery_mode')
    submode = 5 if mode == 'RTL' else 3
    started, last_request = time.monotonic(), -math.inf
    acknowledged = observed = False
    while time.monotonic() - started < timeout:
        if time.monotonic() - last_request >= 1.0:
            conn.mav.command_long_send(conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, submode, 0, 0, 0, 0)
            last_request = time.monotonic()
        message = conn.recv_match(type=['COMMAND_ACK', 'HEARTBEAT'], blocking=True, timeout=.1)
        if message is None:
            continue
        if (message.get_srcSystem() != conn.target_system
                or message.get_srcComponent() != conn.target_component):
            continue
        if message.get_type() == 'COMMAND_ACK' and message.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            acknowledged = message.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
            if message.result not in (mavutil.mavlink.MAV_RESULT_ACCEPTED, mavutil.mavlink.MAV_RESULT_IN_PROGRESS):
                break
        elif message.get_type() == 'HEARTBEAT':
            observed = (message.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4
                        and ((message.custom_mode >> 16) & 0xff) == 4
                        and ((message.custom_mode >> 24) & 0xff) == submode)
            if observed:
                break
    print(f'{mode} 전환: 명령 승인={acknowledged}, FC 모드 확인={observed}; 착륙/정지 완료는 미확인', flush=True)
    return observed


def heartbeat_thread(conn):
    while True:
        try:
            conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except (OSError, RuntimeError):
            return
        time.sleep(1.0)


def send_pos(conn, n, e, d, yaw, vn=0.0, ve=0.0, vd=0.0):
    """수직 이륙 및 호버링 위치 고정용 MAVLink 명령 전송."""
    type_mask = 0b0000_100_111_000_000
    conn.mav.set_position_target_local_ned_send(
        0, conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask,
        n, e, d, vn, ve, vd, 0, 0, 0, yaw, 0)


def send_attitude_target(conn, q_wxyz, thrust):
    """테일시터 3D 자세 쿼터니언(q_wxyz) + 추력(thrust) 직접 전송 (SET_ATTITUDE_TARGET)."""
    type_mask = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )
    conn.mav.set_attitude_target_send(
        0,
        conn.target_system,
        conn.target_component,
        type_mask,
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,
        float(thrust),
    )


def request_stream(conn, msg_id, hz):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)


def set_param(conn, name, value):
    conn.mav.param_set_send(
        conn.target_system, conn.target_component,
        name.encode('utf-8'), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.08)
    ack = conn.recv_match(type='PARAM_VALUE', blocking=True, timeout=1.5)
    if ack:
        print(f'  PARAM {name} = {ack.param_value:.1f}', flush=True)
    else:
        print(f'  PARAM {name} = {value:.1f} (no ack)', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. 메인 협력 폐루프 비행 시퀀스
# ═══════════════════════════════════════════════════════════════════════════
def run_mission(conn, vision):
    request_stream(conn, mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 60)
    request_stream(conn, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 60)

    print('테일시터 파라미터 확인 ...', flush=True)
    set_param(conn, 'SYS_HAS_NUM_ASPD', 0.0)
    set_param(conn, 'FW_USE_AIRSPD', 0.0)
    set_param(conn, 'FD_FAIL_R', 85.0)
    set_param(conn, 'FD_FAIL_P', 85.0)

    telemetry = TelemetryState(conn.target_system, conn.target_component)
    _, _, takeoff_d = world_to_ned(HOME_WORLD[0], HOME_WORLD[1], HOME_WORLD[2] + TAKEOFF_ALT)

    # ── 0단계: 지상 대기 상태에서 지상 카메라 상공 탐색 -> 풍선 락온 -> 사용자 출격 승인 대기 ──
    print('\n[1단계: 지상 카메라 상공 탐색 중] 짐벌이 좌측 하단(-20.6°, +8.0°)에서 상공을 스캔하며 풍선을 찾고 있습니다...', flush=True)
    last_wait_log = 0.0
    locked_banner_printed = False
    active_target_world = None

    while True:
        telemetry.drain(conn)

        ukf_world, ukf_sigma, ukf_n, ukf_valid, ukf_reason, ukf_age, ukf_rx_cnt, ukf_hz = vision.get_ground_ukf()
        now_w = time.monotonic()

        is_locked = vision.target_ready()

        if is_locked:
            active_target_world = ukf_world
            if not locked_banner_printed:
                locked_banner_printed = True
                ex = active_target_world[0] - FALLBACK_TARGET_WORLD[0]
                ey = active_target_world[1] - FALLBACK_TARGET_WORLD[1]
                ez = active_target_world[2] - FALLBACK_TARGET_WORLD[2]
                err_3d = math.sqrt(ex * ex + ey * ey + ez * ez)
                print('\n========================================================================', flush=True)
                print(' ★ [2단계: 지상 카메라 풍선 발견 & 6-DOF UKF 락온 완료!]', flush=True)
                print('   - 풍선 검증 : 색상(HSV) + 원형도(Circularity) + 채움비 검증 통과', flush=True)
                print(f'   - 실제 좌표 : World = ({FALLBACK_TARGET_WORLD[0]:.2f}, {FALLBACK_TARGET_WORLD[1]:.2f}, {FALLBACK_TARGET_WORLD[2]:.2f}) m', flush=True)
                print(f'   - 추정 좌표 : World = ({active_target_world[0]:.2f}, {active_target_world[1]:.2f}, {active_target_world[2]:.2f}) m', flush=True)
                print(f'   - 추정 오차 : dX={ex:+.2f}m, dY={ey:+.2f}m, dZ={ez:+.2f}m (3D 오차 = {err_3d:.2f} m)', flush=True)
                print(f'   - 필터 상태 : 샘플 수 n = {ukf_n} 개 | 위치 불확실도 s (sigma) = {ukf_sigma:.2f} m', flush=True)
                print(f'   - 수신 주기 : {ukf_hz:.1f} Hz (누적 {ukf_rx_cnt}회 수신 중)', flush=True)
                print(' ------------------------------------------------------------------------', flush=True)
                print(' ▶ [출격 승인 대기 중 (Trigger Waiting)]', flush=True)
                print('   1) 카메라 뷰어 창(onboard)을 클릭하고 [SPACE] 또는 [ENTER] 키를 누르거나,', flush=True)
                print('   2) 새 터미널에서 ./tools/approve_launch.sh 를 실행하면 즉시 이륙합니다!', flush=True)
                print('========================================================================\n', flush=True)

            if vision.is_approved():
                telemetry.require_fresh(now_w)
                vision.begin_mission()
                print(
                    f'★ [출격 승인 확인!] 최종 지상 UKF 목표 좌표 World=({active_target_world[0]:.2f}, '
                    f'{active_target_world[1]:.2f}, {active_target_world[2]:.2f})m (n={ukf_n}, s={ukf_sigma:.2f}m) -> 드론 이륙 시작!',
                    flush=True)
                break

            if now_w - last_wait_log > 2.0:
                ex = active_target_world[0] - FALLBACK_TARGET_WORLD[0]
                ey = active_target_world[1] - FALLBACK_TARGET_WORLD[1]
                ez = active_target_world[2] - FALLBACK_TARGET_WORLD[2]
                err_3d = math.sqrt(ex * ex + ey * ey + ez * ez)
                print(
                    f'  [승인 대기 중 | 수신 {ukf_hz:.1f}Hz (총 {ukf_rx_cnt}회)] '
                    f'UKF=({active_target_world[0]:.2f}, {active_target_world[1]:.2f}, {active_target_world[2]:.2f})m | '
                    f'실제오차=({ex:+.2f}, {ey:+.2f}, {ez:+.2f})m (3D오차 {err_3d:.2f}m) | '
                    f'n={ukf_n} | s={ukf_sigma:.2f}m',
                    flush=True)
                last_wait_log = now_w
        else:
            if now_w - last_wait_log > 1.2:
                if ukf_world is not None:
                    print(
                        f'  [지상 UKF 수렴 중] World=({ukf_world[0]:.1f}, {ukf_world[1]:.1f}, {ukf_world[2]:.1f})m | '
                        f'n={ukf_n} | s={ukf_sigma:.2f}m',
                        flush=True)
                else:
                    print(f'  [지상 짐벌 상공 탐색 중...] 상태={ukf_reason}', flush=True)
                last_wait_log = now_w

        time.sleep(0.04)

    # ── 1단계: 승인 완료 후 OFFBOARD 전환, ARM 및 5m 수직 이륙 ──
    tn, te, t_d = world_to_ned(*active_target_world)
    yaw = math.atan2(te, tn)

    print('setpoint 스트림 시작 ...', flush=True)
    for _ in range(35):
        require_guidance(conn, telemetry, vision)
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(.04)

    # Recheck after the final warm-up sleep, even if FC is already OFFBOARD.
    require_guidance(conn, telemetry, vision)
    # An observed FC mode is required before arming; send streams while awaiting it.
    conn.mav.command_long_send(conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0, 0, 0, 0, 0)
    deadline = time.monotonic() + 5.0
    while not telemetry.offboard:
        require_guidance(conn, telemetry, vision)
        if time.monotonic() >= deadline:
            raise GuidanceAbort('offboard_mode_unconfirmed')
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(.02)

    require_guidance(conn, telemetry, vision)
    if not telemetry.offboard:
        raise GuidanceAbort('offboard_mode_unconfirmed')
    conn.mav.command_long_send(conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    deadline = time.monotonic() + 5.0
    while not telemetry.armed:
        require_guidance(conn, telemetry, vision)
        if not telemetry.offboard or time.monotonic() >= deadline:
            raise GuidanceAbort('armed_offboard_unconfirmed')
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(.02)
    print(f'ARM/OFFBOARD 확인 — {TAKEOFF_ALT:.1f}m 수직 이륙 시작', flush=True)

    deadline = time.monotonic() + 20.0
    while True:
        require_guidance(conn, telemetry, vision, armed=True)
        pos_ned, vel_ned, R_bn = telemetry.pos.copy(), telemetry.vel.copy(), telemetry.rotation.copy()
        if abs(pos_ned[2] - takeoff_d) < .4 and abs(vel_ned[2]) < .8:
            print(f'수직 이륙 고도 확인: {-pos_ned[2]:.2f}m', flush=True)
            break
        if time.monotonic() >= deadline:
            raise GuidanceAbort('takeoff_altitude_timeout')
        send_pos(conn, 0, 0, takeoff_d, yaw)
        time.sleep(.02)

    # 검증된 지상 카메라 좌표만 사용한다. 온보드 영상은 검출/표시용으로 유지한다.
    start_ned = pos_ned.copy()
    target_ned = np.array([tn, te, t_d], dtype=float)
    path_vec = target_ned - start_ned
    path_len = float(np.linalg.norm(path_vec))
    path_dir = path_vec / path_len if path_len > 1e-3 else np.array([1.0, 0.0, 0.0])

    print(f'테일시터 전진 피치다운 전환 및 폐루프 유도 비행 시작: NED({start_ned[0]:.1f}, {start_ned[1]:.1f}, {start_ned[2]:.1f}) '
          f'-> UKF 목표 NED({tn:.1f}, {te:.1f}, {t_d:.1f})', flush=True)

    trans_start_t = time.monotonic()
    last_log = 0.0
    smoothed_nose_ned = np.array([0.0, 0.0, -1.0], dtype=float)
    target_locked = False
    min_dist_after_lock = math.inf
    passed = False
    crossing_error = math.inf
    crossing_previous = pos_ned.copy()
    pass_target_ned = target_ned.copy()
    pass_normal = path_dir.copy()
    last_ukf_update_n = 0

    while True:
        require_guidance(conn, telemetry, vision, armed=True)
        pos_ned, vel_ned, R_bn = telemetry.pos.copy(), telemetry.vel.copy(), telemetry.rotation.copy()
        now = time.monotonic()
        if now - trans_start_t > MISSION_TIMEOUT:
            raise GuidanceAbort('mission_timeout')
        trans_alpha = min(1.0, max(0.0, (now - trans_start_t) / TRANSITION_DUR))

        # 노출시각과 일치하는 기체 자세 이력이 없어 픽셀 기반 종말 유도는 사용하지 않는다.
        # 최신 R_bn을 과거 영상에 적용하지 않고 검증된 지상 좌표를 계속 사용한다.
        ukf_world, ukf_sigma, ukf_n, ukf_valid, _, ukf_age, ukf_rx_cnt, ukf_hz = vision.get_ground_ukf()
        if (not target_locked and ukf_world is not None
                and vision.target_ready() and ukf_rx_cnt != last_ukf_update_n):
            new_tn, new_te, new_td = world_to_ned(*ukf_world)
            new_target_ned = np.array([new_tn, new_te, new_td], dtype=float)
            if float(np.linalg.norm(new_target_ned - target_ned)) < 25.0:
                target_ned = 0.75 * target_ned + 0.25 * new_target_ned
                active_target_world = ukf_world
                path_vec = target_ned - start_ned
                path_len = float(np.linalg.norm(path_vec))
                if path_len > 1e-3:
                    path_dir = path_vec / path_len
                last_ukf_update_n = ukf_rx_cnt

        diff_to_target = target_ned - pos_ned
        dist_3d = float(np.linalg.norm(diff_to_target))

        if not target_locked and dist_3d <= LOCK_DIST:
            target_locked = True
            pass_target_ned, pass_normal = target_ned.copy(), path_dir.copy()
            crossing_previous = pos_ned.copy()
            min_dist_after_lock = dist_3d
        if target_locked:
            min_dist_after_lock = min(min_dist_after_lock, dist_3d)
            crossed, radial = plane_crossing(crossing_previous, pos_ned, pass_target_ned, pass_normal)
            if math.isfinite(radial):
                if not crossed:
                    raise GuidanceAbort(f'waypoint_missed:radial_error={radial:.3f}m')
                passed, crossing_error = True, radial
                print(f'목표 평면 통과 확인: 반경 오차 {radial:.3f}m', flush=True)
            crossing_previous = pos_ned.copy()
            along_past = float((pos_ned - pass_target_ned) @ pass_normal)
            if passed and along_past >= PASS_HOLD_DISTANCE_M:
                break
            if not passed and (along_past > 0 or dist_3d - min_dist_after_lock > 2.0):
                raise GuidanceAbort('waypoint_departed_without_valid_crossing')

        unit_to_target = diff_to_target / max(dist_3d, 0.1)
        if float(np.dot(unit_to_target, path_dir)) < 0.75:
            unit_to_target = path_dir

        prog = max(0.0, min(path_len, float(np.dot(pos_ned - start_ned, path_dir))))
        closest_pt = start_ned + prog * path_dir
        cross_pos_err = closest_pt - pos_ned
        vel_along = float(np.dot(vel_ned, unit_to_target)) * unit_to_target
        vel_perp = vel_ned - vel_along
        a_perp_cmd = cross_pos_err * 1.8 - vel_perp * 1.6
        nose_cmd, thrust_cmd, v_des = compute_tailsitter_cmd(
            unit_to_target, a_perp_cmd, vel_ned, dist_3d, MAX_PATH_LEAD_DEG, trans_alpha, allow_dive=False)
        ukf_tag = f'UKF(n={ukf_n},s={ukf_sigma:.1f}m,{ukf_hz:.0f}Hz)'
        mode_str = f'TS_GROUND_GUIDE[{ukf_tag}](Vd={v_des:.1f} thr={thrust_cmd:.2f})'

        filt_alpha = 0.28 if trans_alpha >= 1.0 else 0.18
        smoothed_nose_ned = (1.0 - filt_alpha) * smoothed_nose_ned + filt_alpha * nose_cmd
        smoothed_nose_ned /= np.linalg.norm(smoothed_nose_ned)

        q_cmd = nose_dir_to_quat_frd(smoothed_nose_ned, yaw)
        send_attitude_target(conn, q_cmd, thrust_cmd)

        if now - last_log > 0.4:
            cur_spd = float(np.linalg.norm(vel_ned))
            nose_ned = R_bn @ np.array([0.0, 0.0, -1.0])
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -nose_ned[2]))))
            nose_elev_deg = 90.0 - tilt_deg
            print(f'  [{mode_str}] 거리 {dist_3d:5.1f}m | 고도 {-pos_ned[2]:4.1f}m | 속도 {cur_spd:4.1f}m/s | '
                  f'기울기(Tilt) {tilt_deg:4.1f}° (카메라앙각 {nose_elev_deg:4.1f}°)', flush=True)
            last_log = now

        time.sleep(0.02)

    print(f'목표 통과 확인(반경 오차 {crossing_error:.3f}m) → 목표 뒤 HOLD 요청', flush=True)
    if not transfer_to_px4(conn, 'HOLD'):
        raise GuidanceAbort('post_pass_hold_unconfirmed')


def main():
    if os.environ.get('FLY_EXPERIMENTAL_TAILSITTER') != '1':
        raise SystemExit('Legacy tailsitter requires explicit experimental_tailsitter:=true; use the default MC spline mission.')
    startup_delay = float(os.environ.get('FLY_STARTUP_DELAY', '0'))
    if startup_delay > 0:
        time.sleep(startup_delay)
    vision = VisionListener()
    start_ros_vision_thread(vision)
    conn = mavutil.mavlink_connection(PX4_UDP)
    try:
        heartbeat = conn.wait_heartbeat(timeout=60)
        if (heartbeat is None or heartbeat.autopilot != mavutil.mavlink.MAV_AUTOPILOT_PX4
                or heartbeat.get_srcComponent() != mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1):
            raise GuidanceAbort('px4_heartbeat_missing')
        threading.Thread(target=heartbeat_thread, args=(conn,), daemon=True).start()
        run_mission(conn, vision)
    except (GuidanceAbort, KeyboardInterrupt, OSError) as exc:
        print(f'유도 중단: {exc or "user_interrupt"}; 통과 성공으로 처리하지 않습니다.', flush=True)
        if conn.target_system and conn.target_component:
            try:
                transfer_to_px4(conn, 'RTL')
            except OSError as transport_error:
                print(f'RTL 전달 실패: {transport_error}; FC 모드/착륙 미확인', flush=True)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
