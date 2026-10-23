"""픽셀 -> 방향 -> NED 위치. ROS 의존성 없음.

좌표 규약 (이 패키지 전용, 레거시 pantilt_system의 부호 규약과 무관):
  - NED: x=북, y=동, z=하(아래가 +)
  - 카메라 FRD: x=전방, y=오른쪽, z=아래. 픽셀 (u,v)에서 u 오른쪽+, v 아래+
  - 방위각 az: 북 기준 시계방향 +[deg], 고각 el: 위쪽 +[deg]
  - 카메라->NED 회전 = Rz(az) @ Ry(el)
"""
import math

import numpy as np


def focal_px(width_px: float, hfov_deg_at_1x: float, zoom: float) -> float:
    """핀홀 모델. 줌 배율은 초점거리에 비례."""
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg_at_1x) / 2.0) * max(zoom, 1e-6)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ned_from_cam(az_deg: float, el_deg: float) -> np.ndarray:
    return rot_z(math.radians(az_deg)) @ rot_y(math.radians(el_deg))


def gimbal_to_azel(yaw_deg, pitch_deg, base_heading_deg=0.0,
                   yaw_sign=1.0, yaw_zero_deg=0.0, pitch_sign=1.0, pitch_zero_deg=0.0):
    """짐벌 보고값 -> (방위각, 고각). 부호/영점은 실기에서 확인해 파라미터로 맞춘다."""
    az = base_heading_deg + yaw_sign * (yaw_deg - yaw_zero_deg)
    el = pitch_sign * (pitch_deg - pitch_zero_deg)
    return az, el


def pixel_ray_ned(u, v, cx, cy, f_px, az_deg, el_deg) -> np.ndarray:
    ray_cam = np.array([1.0, (u - cx) / f_px, (v - cy) / f_px])
    ray_cam /= np.linalg.norm(ray_cam)
    return ned_from_cam(az_deg, el_deg) @ ray_cam


def range_from_diameter(diameter_px: float, diameter_m: float, f_px: float) -> float:
    return diameter_m * f_px / max(diameter_px, 1e-6)


def measurement_covariance(ray: np.ndarray, range_m: float, diameter_px: float, f_px: float,
                           pixel_sigma_px=1.0, diameter_rel_sigma=0.05, angle_sigma_deg=0.2) -> np.ndarray:
    """시선 방향(거리 오차)과 직교 방향(각도 오차)을 분리한 3x3 공분산 [m^2]."""
    rel_r = math.hypot(pixel_sigma_px / max(diameter_px, 1e-6), diameter_rel_sigma)
    sigma_r = range_m * rel_r
    sigma_lat = range_m * math.hypot(pixel_sigma_px / f_px, math.radians(angle_sigma_deg))
    outer = np.outer(ray, ray)
    return sigma_r ** 2 * outer + sigma_lat ** 2 * (np.eye(3) - outer)


def locate_balloon(u, v, diameter_px, *, width_px, height_px, hfov_deg_at_1x, zoom,
                   az_deg, el_deg, balloon_diameter_m, offset_ned=(0.0, 0.0, 0.0), **sigma_kw):
    """한 프레임의 검출 -> (위치 NED[3], 공분산[3x3], 거리[m]). offset_ned = 카메라 회전중심의 NED 위치."""
    f = focal_px(width_px, hfov_deg_at_1x, zoom)
    ray = pixel_ray_ned(u, v, width_px / 2.0, height_px / 2.0, f, az_deg, el_deg)
    r = range_from_diameter(diameter_px, balloon_diameter_m, f)
    cov = measurement_covariance(ray, r, diameter_px, f, **sigma_kw)
    return np.asarray(offset_ned, dtype=float) + r * ray, cov, r
