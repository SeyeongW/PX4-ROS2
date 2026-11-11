"""3D 풍선 위치·속도 추정을 위한 무향 칼만 필터(UKF, Unscented Kalman Filter) 및 정적 추정기.

- UKFTargetEstimator:
  6자유도 상태벡터 x = [p_N, p_E, p_D, v_N, v_E, v_D]^T 와 2L+1 = 13개의 Merwe 스케일 시그마 포인트
  (Unscented Transform)를 사용하여 단안 카메라의 비선형 관측(방위각, 고각, 픽셀 지름 역거리) 및
  3차원 위치 관측으로부터 풍선의 3D NED 좌표와 속도를 실시간 추정합니다.
- StaticTargetEstimator:
  UKFTargetEstimator와 동일한 인터페이스를 유지하며 기존 단위 테스트 및 노드와 호환됩니다.
"""
import math
import numpy as np


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


class UKFTargetEstimator:
    """6-DOF (3D Position + 3D Velocity) Unscented Kalman Filter for Balloon Coordinate Estimation."""

    STATE_DIM = 6

    def __init__(
        self,
        gate_chi2: float = 16.27,
        min_gate_samples: int = 5,
        reset_after_rejects: int = 15,
        forget: float = 1.0,
        alpha: float = 0.25,
        beta: float = 2.0,
        kappa: float = 0.0,
        process_pos_std: float = 0.02,
        process_vel_std: float = 0.08,
        vel_damping: float = 0.85,
    ):
        self.gate_chi2 = float(gate_chi2)
        self.min_gate_samples = int(min_gate_samples)
        self.reset_after_rejects = int(reset_after_rejects)
        self.forget = float(forget)
        if not np.isfinite([gate_chi2, forget]).all() or gate_chi2 <= 0 or not 0 < forget <= 1:
            raise ValueError("invalid_estimator_configuration")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.process_pos_std = float(process_pos_std)
        self.process_vel_std = float(process_vel_std)
        self.vel_damping = float(vel_damping)

        L = self.STATE_DIM
        self.lam = (self.alpha ** 2) * (L + self.kappa) - L
        self.gamma = math.sqrt(L + self.lam)

        n_sig = 2 * L + 1
        self.Wm = np.full(n_sig, 1.0 / (2.0 * (L + self.lam)), dtype=float)
        self.Wc = np.full(n_sig, 1.0 / (2.0 * (L + self.lam)), dtype=float)
        self.Wm[0] = self.lam / (L + self.lam)
        self.Wc[0] = self.lam / (L + self.lam) + (1.0 - self.alpha ** 2 + self.beta)

        self.reset()

    def reset(self):
        self._x = np.zeros(self.STATE_DIM, dtype=float)
        self._P = np.eye(self.STATE_DIM, dtype=float)
        self._info = np.zeros((3, 3), dtype=float)
        self._vec = np.zeros(3, dtype=float)
        self._systematic = np.zeros((3, 3), dtype=float)
        self._last_stamp = None
        self.n = 0
        self.rejects = 0
        self._consec_rejects = 0

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """2L + 1 = 13개의 Merwe 스케일 시그마 포인트 생성."""
        L = self.STATE_DIM
        P_sym = 0.5 * (P + P.T)
        eigvals, eigvecs = np.linalg.eigh(P_sym)
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        scaled = self.gamma * sqrt_P

        sigmas = np.zeros((2 * L + 1, L), dtype=float)
        sigmas[0] = x
        for i in range(L):
            sigmas[i + 1] = x + scaled[:, i]
            sigmas[L + i + 1] = x - scaled[:, i]
        return sigmas

    def predict(self, dt: float):
        """무향 변환(Unscented Transform) 기반 상태 및 공분산 시간 예측 단계."""
        if self.n == 0:
            return
        dt = float(np.clip(dt, 0.0, 1.0))
        sigmas = self._sigma_points(self._x, self._P)

        # 비선형/준정적 풍선 운동 모델 f(x, dt) 통과
        sigmas_pred = np.empty_like(sigmas)
        damp = self.vel_damping ** max(dt, 0.02)
        sigmas_pred[:, 0:3] = sigmas[:, 0:3] + sigmas[:, 3:6] * dt
        sigmas_pred[:, 3:6] = sigmas[:, 3:6] * damp

        x_pred = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)
        dx = sigmas_pred - x_pred[None, :]
        P_pred = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=float)
        for i in range(sigmas_pred.shape[0]):
            P_pred += self.Wc[i] * np.outer(dx[i], dx[i])

        if self.forget < 1.0:
            P_pred[:3, :3] /= self.forget

        q_pos = (self.process_pos_std ** 2) * max(dt, 1e-3)
        q_vel = (self.process_vel_std ** 2) * max(dt, 1e-3)
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
        self._x = x_pred
        self._P = 0.5 * (P_pred + P_pred.T) + Q

    @property
    def mean(self):
        return None if self.n == 0 else self._x[:3].copy()

    @property
    def velocity(self):
        return None if self.n == 0 else self._x[3:6].copy()

    @property
    def cov(self):
        return None if self.n == 0 else self._P[:3, :3] + self._systematic

    @property
    def sigma(self) -> float:
        """가장 불확실한 방향의 표준편차 [m]. 샘플이 없으면 inf."""
        if self.n == 0:
            return float('inf')
        return float(np.sqrt(np.linalg.eigvalsh(self.cov).max()))

    @staticmethod
    def _h_spherical(pos_ned: np.ndarray, origin_ned: np.ndarray, focal_px: float, balloon_diam_m: float) -> np.ndarray:
        """비선형 단안 카메라 핀홀 구면 관측 모델 h(x) = [azimuth, elevation, apparent_diam_px]^T."""
        rel = pos_ned - origin_ned
        horiz = math.hypot(float(rel[0]), float(rel[1]))
        rng = max(math.hypot(horiz, float(rel[2])), 1e-3)
        az = math.atan2(float(rel[1]), float(rel[0]))
        el = math.atan2(-float(rel[2]), max(horiz, 1e-6))
        d_px = (focal_px * balloon_diam_m) / rng
        return np.array([az, el, d_px], dtype=float)

    def update_spherical(
        self,
        origin_ned,
        ray_unit_ned,
        diam_px: float,
        focal_px: float,
        balloon_diam_m: float,
        pixel_sigma_px: float = 0.5,
        angle_sigma_deg: float = 0.1,
        systematic_covariance=None,
        stamp: float = None,
    ) -> bool:
        """단안 카메라의 비선형 관측 [방위각(az), 고각(el), 픽셀 지름(d_px)]을 이용한 무향 칼만 필터(UKF) 갱신."""
        origin = np.asarray(origin_ned, dtype=float)
        ray = np.asarray(ray_unit_ned, dtype=float)
        ray_norm = float(np.linalg.norm(ray))
        if (origin.shape != (3,) or ray.shape != (3,) or ray_norm < 1e-6
                or not np.isfinite(origin).all() or not np.isfinite(ray).all()
                or not np.isfinite([diam_px, focal_px, balloon_diam_m, pixel_sigma_px, angle_sigma_deg]).all()
                or min(diam_px, focal_px, balloon_diam_m, pixel_sigma_px) <= 0):
            return False
        ray = ray / ray_norm
        rng_init = (focal_px * balloon_diam_m) / diam_px
        pos_init = origin + rng_init * ray

        # 초기 샘플이거나 비선형 관측을 3D 직교 공분산과 결합할 때 시그마 포인트 변환 수행
        az_meas = math.atan2(float(ray[1]), float(ray[0]))
        el_meas = math.atan2(-float(ray[2]), max(math.hypot(float(ray[0]), float(ray[1])), 1e-6))
        z_meas = np.array([az_meas, el_meas, float(diam_px)], dtype=float)

        ang_std = math.radians(max(angle_sigma_deg, 0.05)) + (pixel_sigma_px / max(focal_px, 1.0))
        diam_std = max(pixel_sigma_px * math.sqrt(2.0), 0.25)
        R_sph = np.diag([ang_std ** 2, ang_std ** 2, diam_std ** 2])

        if self.n == 0:
            # 1차 샘플: 역투영 초기 위치 설정 및 시그마 포인트 공분산 초기화
            rng_std = rng_init * (diam_std / max(diam_px, 1.0))
            cross_std = rng_init * ang_std
            u = ray
            arbitrary = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = np.cross(u, arbitrary)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            R_cart = (rng_std ** 2) * np.outer(u, u) + (cross_std ** 2) * (np.outer(e1, e1) + np.outer(e2, e2))
            return self.update(pos_init, R_cart, systematic_covariance=systematic_covariance, stamp=stamp)

        if stamp is not None and self._last_stamp is not None and stamp > self._last_stamp:
            self.predict(stamp - self._last_stamp)

        sigmas = self._sigma_points(self._x, self._P)
        Z_sig = np.array([
            self._h_spherical(s[:3], origin, focal_px, balloon_diam_m) for s in sigmas
        ])

        # 각도 평균(Circular Mean) 처리
        sin_az = float(np.sum(self.Wm * np.sin(Z_sig[:, 0])))
        cos_az = float(np.sum(self.Wm * np.cos(Z_sig[:, 0])))
        z_pred = np.array([
            math.atan2(sin_az, cos_az),
            float(np.sum(self.Wm * Z_sig[:, 1])),
            float(np.sum(self.Wm * Z_sig[:, 2])),
        ], dtype=float)

        S = R_sph.copy()
        P_xz = np.zeros((self.STATE_DIM, 3), dtype=float)
        for i in range(sigmas.shape[0]):
            dz = Z_sig[i] - z_pred
            dz[0] = _wrap_angle(float(dz[0]))
            dx = sigmas[i] - self._x
            S += self.Wc[i] * np.outer(dz, dz)
            P_xz += self.Wc[i] * np.outer(dx, dz)

        nu = z_meas - z_pred
        nu[0] = _wrap_angle(float(nu[0]))

        if self.n >= self.min_gate_samples:
            m2 = float(nu @ np.linalg.solve(S, nu))
            if m2 > self.gate_chi2:
                self.rejects += 1
                self._consec_rejects += 1
                if self._consec_rejects >= self.reset_after_rejects:
                    self.reset()
                return False

        self._consec_rejects = 0
        K = P_xz @ np.linalg.inv(S)
        self._x = self._x + K @ nu
        self._P = 0.5 * ((self._P - K @ S @ K.T) + (self._P - K @ S @ K.T).T)
        eigvals, eigvecs = np.linalg.eigh(self._P)
        self._P = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T

        if systematic_covariance is not None:
            systematic = np.asarray(systematic_covariance, dtype=float)
            if systematic.shape == (3, 3) and np.isfinite(systematic).all():
                eigenvalues, vectors = np.linalg.eigh(systematic - self._systematic)
                self._systematic += (vectors * np.maximum(eigenvalues, 0.0)) @ vectors.T

        if stamp is not None:
            self._last_stamp = float(stamp)
        self.n += 1
        return True

    def update(self, p, cov, *, systematic_covariance=None, stamp: float = None) -> bool:
        """시그마 포인트 무향 변환(Unscented Transform) 기반 3D 위치 관측 갱신."""
        p = np.asarray(p, dtype=float)
        cov = np.asarray(cov, dtype=float)
        systematic = (
            np.zeros((3, 3), dtype=float)
            if systematic_covariance is None
            else np.asarray(systematic_covariance, dtype=float)
        )
        if (p.shape != (3,) or cov.shape != (3, 3) or systematic.shape != (3, 3)
                or not np.isfinite(p).all() or not np.isfinite(cov).all() or not np.isfinite(systematic).all()
                or not np.allclose(cov, cov.T) or not np.allclose(systematic, systematic.T)
                or np.linalg.eigvalsh(cov).min() <= 0 or np.linalg.eigvalsh(systematic).min() < -1e-10):
            return False

        if self.n >= self.min_gate_samples:
            d = p - self.mean
            m2 = float(d @ np.linalg.solve(cov + systematic + self.cov, d))
            if m2 > self.gate_chi2:
                self.rejects += 1
                self._consec_rejects += 1
                if self._consec_rejects >= self.reset_after_rejects:
                    self.reset()
                return False

        self._consec_rejects = 0
        w = np.linalg.inv(cov)
        self._info = self.forget * self._info + w
        self._vec = self.forget * self._vec + w @ p

        if self.n == 0:
            self._x[:3] = p
            self._x[3:6] = 0.0
            self._P = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=float)
            self._P[:3, :3] = cov.copy()
            self._P[3:, 3:] = np.eye(3, dtype=float) * 0.25
        else:
            if stamp is not None and self._last_stamp is not None and stamp > self._last_stamp:
                self.predict(stamp - self._last_stamp)
            elif self.forget < 1.0:
                self._P[:3, :3] /= self.forget

            # 시그마 포인트 생성 및 관측 공간 Z_i = H(X_i) = X_i[0:3] 무향 변환
            sigmas = self._sigma_points(self._x, self._P)
            Z_sig = sigmas[:, :3]
            z_pred = np.sum(self.Wm[:, None] * Z_sig, axis=0)

            S = cov.copy()
            P_xz = np.zeros((self.STATE_DIM, 3), dtype=float)
            for i in range(sigmas.shape[0]):
                dz = Z_sig[i] - z_pred
                dx = sigmas[i] - self._x
                S += self.Wc[i] * np.outer(dz, dz)
                P_xz += self.Wc[i] * np.outer(dx, dz)

            K = P_xz @ np.linalg.inv(S)
            nu = p - z_pred
            self._x = self._x + K @ nu
            # 정적/준정적 수렴 정밀도를 보장하기 위해 위치 블록은 정보 행렬 해와 동기화
            self._P[:3, :3] = np.linalg.inv(self._info)
            self._P[3:, 3:] *= 0.92

        eigenvalues, vectors = np.linalg.eigh(systematic - self._systematic)
        self._systematic += (vectors * np.maximum(eigenvalues, 0.0)) @ vectors.T
        if stamp is not None:
            self._last_stamp = float(stamp)
        self.n += 1
        return True


# 기존 인터페이스 호환 별칭
StaticTargetEstimator = UKFTargetEstimator
