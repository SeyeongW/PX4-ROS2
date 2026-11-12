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


def test_common_bias_covariance_does_not_average_away():
    est = StaticTargetEstimator()
    systematic = np.diag([4., .04, .04])
    for _ in range(500):
        assert est.update([42., 0., 0.], np.eye(3)*.01,
                          systematic_covariance=systematic)
    assert est.sigma >= 2.
    np.testing.assert_allclose(est.mean, [42., 0., 0.])


def test_bad_covariance_is_rejected_without_corrupting_state():
    est = StaticTargetEstimator()
    for covariance in (np.zeros((3,3)), np.diag([-1.,1.,1.]),
                       np.array([[1.,2.,0.],[0.,1.,0.],[0.,0.,1.]])):
        assert not est.update([1.,2.,3.], covariance)
    assert est.n == 0
    assert not est.update([1.,2.,3.], np.eye(3), systematic_covariance=np.diag([-1.,0.,0.]))
