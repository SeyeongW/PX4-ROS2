"""Bearing-only pursuit guidance for high-speed balloon approach."""
import numpy as np


def quat_to_rot(w, x, y, z):
    """Unit quaternion → 3×3 rotation matrix (body → NED)."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def pixel_to_ray_ned(dx, dy, fx, fy, R_bn):
    """Pixel centre offset → NED unit ray. Camera FRD = Body FRD.
    detector sends dy with up-positive, so invert for FRD (down-positive)."""
    ray = np.array([1.0, dx / fx, -dy / fy])
    ray /= np.linalg.norm(ray)
    return R_bn @ ray


class BearingPN:
    """3-D bearing-only proportional navigation with coast-through on vision dropout.

    update()  — vision available: returns lateral PN + axial acceleration in NED.
    coast()   — vision lost: returns axial acceleration along last known LOS.
    """

    def __init__(self, N, fx, fy, a_max, forward_acc):
        self.N = N
        self.fx, self.fy = fx, fy
        self.a_max = a_max
        self.forward_acc = forward_acc
        self._prev_los = None
        self._prev_t = None
        self._last_los = None

    def reset(self):
        self._prev_los = None
        self._prev_t = None
        self._last_los = None

    @property
    def has_lock(self):
        return self._last_los is not None

    def update(self, dx, dy, R_bn, vel_ned, t):
        """Feed a vision measurement. Returns NED acceleration or None (first frame)."""
        los = pixel_to_ray_ned(dx, dy, self.fx, self.fy, R_bn)
        self._last_los = los
        if self._prev_los is None:
            self._prev_los, self._prev_t = los, t
            return None
        dt = t - self._prev_t
        if dt < 1e-4:
            return None
        # LOS angular rate via cross product (small-angle approx, valid at 60 Hz)
        los_rate = np.cross(self._prev_los, los) / dt
        self._prev_los, self._prev_t = los, t

        vc = float(-vel_ned @ los)
        a_lat = np.cross(los_rate, los) * (self.N * max(vc, 2.0))
        a_cmd = a_lat + self.forward_acc * los
        return self._clamp(a_cmd)

    def coast(self):
        """No vision: continue along last known LOS (forward accel only, no lateral correction)."""
        if self._last_los is None:
            return None
        return self._clamp(self.forward_acc * self._last_los)

    def _clamp(self, a):
        mag = np.linalg.norm(a)
        if mag > self.a_max:
            a = a / mag * self.a_max
        return a
