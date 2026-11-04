import numpy as np

from balloon_locator.static_estimator import StaticTargetEstimator


def _feed(est, truth, n, rng, sigma=1.0):
    cov = np.eye(3) * sigma ** 2
    for _ in range(n):
        est.update(truth + rng.normal(0, sigma, 3), cov)


def test_converges_and_sigma_shrinks():
    rng = np.random.default_rng(0)
    truth = np.array([50.0, -20.0, -10.0])
    est = StaticTargetEstimator()
    _feed(est, truth, 10, rng)
    s10 = est.sigma
    _feed(est, truth, 190, rng)
    assert np.linalg.norm(est.mean - truth) < 0.5
    assert est.sigma < s10 / 3


def test_outlier_rejected():
    rng = np.random.default_rng(1)
    truth = np.array([10.0, 10.0, -5.0])
    est = StaticTargetEstimator()
    _feed(est, truth, 20, rng, sigma=0.5)
    before = est.mean.copy()
    assert est.update(truth + [30, 0, 0], np.eye(3) * 0.25) is False
    assert np.allclose(est.mean, before) and est.rejects == 1


def test_resets_when_target_really_moved():
    rng = np.random.default_rng(2)
    est = StaticTargetEstimator(reset_after_rejects=10)
    _feed(est, np.array([10.0, 0, 0]), 20, rng, sigma=0.3)
    moved = np.array([40.0, 0, 0])
    _feed(est, moved, 40, rng, sigma=0.3)
    assert np.linalg.norm(est.mean - moved) < 1.0


def test_rejects_nan():
    est = StaticTargetEstimator()
    assert est.update([np.nan, 0, 0], np.eye(3)) is False and est.n == 0
