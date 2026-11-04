"""픽셀 -> 방향 -> NED 위치. ROS 의존성 없음.

좌표 규약 (이 패키지 전용, 레거시 pantilt_system의 부호 규약과 무관):
  - NED: x=북, y=동, z=하(아래가 +)
  - 카메라 FRD: x=전방, y=오른쪽, z=아래. 픽셀 (u,v)에서 u 오른쪽+, v 아래+
  - 방위각 az: 북 기준 시계방향 +[deg], 고각 el: 위쪽 +[deg]
  - 일반 Euler 기본: Rz(az) @ Ry(el) @ Rx(roll)
  - SIYI SDK 312: Rz(az) @ Rx(roll) @ Ry(el), rotation_order='zxy'
"""
import math

import numpy as np


def focal_px(width_px: float, hfov_deg_at_1x: float, zoom: float) -> float:
    """핀홀 모델. 줌 배율은 초점거리에 비례."""
    if (not np.isfinite([width_px, hfov_deg_at_1x, zoom]).all()
            or width_px <= 0 or not 0 < hfov_deg_at_1x < 180 or zoom <= 0):
        raise ValueError('invalid_focal_geometry')
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg_at_1x) / 2.0) * zoom


def calibrated_intrinsics(info):
    """Rectified CameraInfo dictionary -> (fx, fy, cx, cy).

    Nonzero distortion is rejected; the hardware input must first be rectified
    using its measured calibration. A nominal FOV is not a substitute for K.
    """
    try:
        k = np.asarray(info['k'], float).reshape(3, 3)
        d = np.asarray(info.get('d', []), float)
        width, height = float(info['width']), float(info['height'])
        if (not np.isfinite(k).all() or not np.isfinite(d).all()
                or not np.isfinite([width, height]).all() or min(width, height) <= 0
                or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0., 0., 1.])
                or not np.allclose([k[0, 1], k[1, 0]], 0.)
                or not 0 <= k[0, 2] < width or not 0 <= k[1, 2] < height
                or info.get('binning_x', 0) not in (0, 1)
                or info.get('binning_y', 0) not in (0, 1)
                or any(info.get('roi_offset', (0, 0)))):
            raise ValueError('invalid_calibrated_intrinsics')
        if (np.any(np.abs(d) > 1e-12)
                or info.get('distortion_model', 'plumb_bob') not in ('plumb_bob', 'rational_polynomial')):
            raise ValueError('rectified_camera_required')
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('invalid_calibrated_intrinsics') from error


def optical_ray(u, v, intrinsics):
    """Unit ray in optical XYZ (right, down, forward), using actual K."""
    fx, fy, cx, cy = map(float, intrinsics)
    if not np.isfinite([u, v, fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError('invalid_optical_ray')
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.])
    length = np.linalg.norm(ray)
    if not math.isfinite(float(length)):
        raise ValueError('invalid_optical_ray')
    return ray / length


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def ned_from_cam(az_deg: float, el_deg: float, roll_deg=0., rotation_order='zyx') -> np.ndarray:
    if not np.isfinite([az_deg,el_deg,roll_deg]).all() or rotation_order not in ('zyx','zxy'):
        raise ValueError('invalid_gimbal_orientation')
    yaw, pitch, roll = rot_z(math.radians(az_deg)), rot_y(math.radians(el_deg)), rot_x(math.radians(roll_deg))
    return yaw @ (pitch @ roll if rotation_order == 'zyx' else roll @ pitch)


def gimbal_to_azel(yaw_deg, pitch_deg, base_heading_deg=0.0,
                   yaw_sign=1.0, yaw_zero_deg=0.0, pitch_sign=1.0, pitch_zero_deg=0.0):
    """짐벌 보고값 -> (방위각, 고각). 부호/영점은 실기에서 확인해 파라미터로 맞춘다."""
    az = base_heading_deg + yaw_sign * (yaw_deg - yaw_zero_deg)
    el = pitch_sign * (pitch_deg - pitch_zero_deg)
    return az, el


def pixel_ray_ned(u, v, cx, cy, f_px, az_deg, el_deg, roll_deg=0., rotation_order='zyx') -> np.ndarray:
    optical = optical_ray(u, v, (f_px, f_px, cx, cy))
    ray_cam = optical[[2, 0, 1]]
    if not np.isfinite([az_deg, el_deg]).all():
        raise ValueError('invalid_gimbal_orientation')
    return ned_from_cam(az_deg, el_deg, roll_deg, rotation_order) @ ray_cam


def range_from_diameter(diameter_px: float, diameter_m: float, f_px: float) -> float:
    if not np.isfinite([diameter_px, diameter_m, f_px]).all() or min(diameter_px, diameter_m, f_px) <= 0:
        raise ValueError('invalid_range_geometry')
    result = diameter_m * f_px / diameter_px
    if not math.isfinite(result):
        raise ValueError('invalid_range_geometry')
    return result


def measurement_covariance(ray: np.ndarray, range_m: float, diameter_px: float, f_px: float,
                           pixel_sigma_px=1.0, diameter_rel_sigma=0.05, angle_sigma_deg=0.2) -> np.ndarray:
    """시선 방향(거리 오차)과 직교 방향(각도 오차)을 분리한 3x3 공분산 [m^2]."""
    ray = np.asarray(ray, float)
    values = [range_m, diameter_px, f_px, pixel_sigma_px, diameter_rel_sigma, angle_sigma_deg]
    if (ray.shape != (3,) or not np.isfinite(ray).all() or not np.isfinite(values).all()
            or np.linalg.norm(ray) <= 0 or min(range_m, diameter_px, f_px) <= 0
            or min(pixel_sigma_px, diameter_rel_sigma, angle_sigma_deg) < 0):
        raise ValueError('invalid_measurement_covariance')
    ray = ray / np.linalg.norm(ray)
    rel_r = math.hypot(pixel_sigma_px / diameter_px, diameter_rel_sigma)
    sigma_r = range_m * rel_r
    sigma_lat = range_m * math.hypot(pixel_sigma_px / f_px, math.radians(angle_sigma_deg))
    outer = np.outer(ray, ray)
    return sigma_r ** 2 * outer + sigma_lat ** 2 * (np.eye(3) - outer)


def locate_balloon(u, v, diameter_px, *, width_px, height_px, hfov_deg_at_1x, zoom,
                   az_deg, el_deg, balloon_diameter_m, offset_ned=(0.0, 0.0, 0.0),
                   intrinsics=None, roll_deg=0., rotation_order='zyx', **sigma_kw):
    """한 프레임의 검출 -> (위치 NED[3], 공분산[3x3], 거리[m]). offset_ned = 카메라 회전중심의 NED 위치."""
    if intrinsics is None:
        f = focal_px(width_px, hfov_deg_at_1x, zoom)
        intrinsics = (f, f, width_px / 2.0, height_px / 2.0)
    else:
        f = (intrinsics[0] + intrinsics[1]) / 2.
    ray = ned_from_cam(az_deg, el_deg, roll_deg, rotation_order) @ optical_ray(u, v, intrinsics)[[2, 0, 1]]
    offset = np.asarray(offset_ned, float)
    if (offset.shape != (3,) or not np.isfinite(offset).all()
            or not np.isfinite([az_deg, el_deg]).all()):
        raise ValueError('invalid_target_pose')
    r = range_from_diameter(diameter_px, balloon_diameter_m, f)
    cov = measurement_covariance(ray, r, diameter_px, f, **sigma_kw)
    return offset + r * ray, cov, r
