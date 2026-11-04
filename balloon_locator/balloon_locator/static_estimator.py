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
        self.reset()

    def reset(self):
        self._info = np.zeros((3, 3))
        self._vec = np.zeros(3)
        self.n = 0
        self.rejects = 0          # 누적
        self._consec_rejects = 0

    @property
    def mean(self):
        return None if self.n == 0 else np.linalg.solve(self._info, self._vec)

    @property
    def cov(self):
        return None if self.n == 0 else np.linalg.inv(self._info)

    @property
    def sigma(self) -> float:
        """가장 불확실한 방향의 표준편차 [m]. 샘플이 없으면 inf."""
        if self.n == 0:
            return float('inf')
        return float(np.sqrt(np.linalg.eigvalsh(self.cov).max()))

    def update(self, p, cov) -> bool:
        p = np.asarray(p, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(cov))):
            return False
        if self.n >= self.min_gate_samples:
            d = p - self.mean
            m2 = float(d @ np.linalg.solve(cov + self.cov, d))
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
        self.n += 1
        return True
