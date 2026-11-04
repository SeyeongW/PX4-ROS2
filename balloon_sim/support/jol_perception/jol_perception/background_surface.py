"""빈 장면에서 학습한 바닥 평면으로 관측 영역 안의 배경 구멍만 보완한다.

관측한 XY 경계 밖으로 확장하거나 실시간 표적을 배경으로 흡수하지 않는다."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass
class BackgroundSurface:
    plane: np.ndarray
    boundary: np.ndarray
    tolerance: float = .06

    @classmethod
    def fit(cls, background):
        points = np.asarray(background, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 200:
            return None
        random = np.random.default_rng(360)
        sample = points[random.choice(len(points), min(2048, len(points)), replace=False)]
        best, count = None, 0
        for _ in range(64):
            a, b, c = sample[random.choice(len(sample), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            length = np.linalg.norm(normal)
            if length < 1e-8:
                continue
            normal /= length
            # world +Z is gravity-up in this fixed-base installation.
            if abs(normal[2]) < np.cos(np.deg2rad(10)):
                continue
            offset = -normal @ a
            mask = np.abs(sample @ normal+offset) <= .04
            if mask.sum() > count:
                best, count = mask, int(mask.sum())
        if best is None or count < max(100, len(sample)*.5):
            return None
        inliers = sample[best]
        center = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers-center, full_matrices=False)
        normal = vh[-1]
        if abs(normal[2]) < np.cos(np.deg2rad(10)):
            return None
        offset = -normal @ center
        support = points[np.abs(points @ normal+offset) <= .06]
        try:
            hull = ConvexHull(support[:, :2])
        except QhullError:
            return None
        if hull.volume < 1.0:
            return None
        return cls(np.r_[normal, offset], hull.equations.copy())

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        mask = np.abs(points @ self.plane[:3]+self.plane[3]) <= self.tolerance
        indices = np.flatnonzero(mask)
        if len(indices):
            inside = np.all(points[indices, :2] @ self.boundary[:, :2].T
                            + self.boundary[:, 2] <= 1e-8, axis=1)
            mask[indices] = inside
        return mask
