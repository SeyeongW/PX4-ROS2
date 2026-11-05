"""
등속 UKF와 달리 속도 상태가 없다. 목표가 정지해 있다는 가정이므로 샘플이 쌓일수록 sigma가 줄고,
가정이 깨지면(연속 게이트 거부) 스스로 초기화한다. forget < 1 이면 오래된 샘플의 비중을 줄여
천천히 떠내려가는 목표를 따라간다.
"""
import numpy as np


class StaticTargetEstimator:
    def __init__(self, gate_chi2=16.27, min_gate_samples=5, reset_after_rejects=15, forget=1.0):
        self.gate_chi2 = gate_chi2            # 3자유도, 99.9%
        self.min_gate_samples = min_gate_samples
        self.reset_after_rejects = reset_after_rejects
        self.forget = forget
        if not np.isfinite([gate_chi2, forget]).all() or gate_chi2 <= 0 or not 0 < forget <= 1:
            raise ValueError("invalid_estimator_configuration")
        self.reset()

    def reset(self):
        self._info = np.zeros((3, 3))
        self._vec = np.zeros(3)
        self._systematic = np.zeros((3,3))
        self.n = 0
        self.rejects = 0          # 누적
        self._consec_rejects = 0

    @property
    def mean(self):
        return None if self.n == 0 else np.linalg.solve(self._info, self._vec)

    @property
    def cov(self):
        return None if self.n == 0 else np.linalg.inv(self._info) + self._systematic

    @property
    def sigma(self) -> float:
        """가장 불확실한 방향의 표준편차 [m]. 샘플이 없으면 inf."""
        if self.n == 0:
            return float('inf')
        return float(np.sqrt(np.linalg.eigvalsh(self.cov).max()))

    def update(self, p, cov, *, systematic_covariance=None) -> bool:
        """Average independent measurement noise, preserve correlated calibration error.

        The PSD envelope dominates every accepted systematic covariance; repeated
        frames cannot manufacture certainty about diameter, heading or mounting.
        """
        p = np.asarray(p, dtype=float)
        cov = np.asarray(cov, dtype=float)
        systematic = np.zeros((3,3)) if systematic_covariance is None else np.asarray(systematic_covariance, dtype=float)
        if (p.shape != (3,) or cov.shape != (3,3) or systematic.shape != (3,3)
                or not np.isfinite(p).all() or not np.isfinite(cov).all() or not np.isfinite(systematic).all()
                or not np.allclose(cov,cov.T) or not np.allclose(systematic,systematic.T)
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
        eigenvalues, vectors = np.linalg.eigh(systematic-self._systematic)
        self._systematic += (vectors*np.maximum(eigenvalues,0.))@vectors.T
        self.n += 1
        return True
