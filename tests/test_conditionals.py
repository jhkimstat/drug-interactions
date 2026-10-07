import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, logit
from scipy.stats import norm

from factorregression.conditionals import (
    beta_parameters,
    draw_precision_normal,
    indicator_log_odds,
    loading_parameters,
)


def test_loading_conditional_matches_joint_density_ratio():
    rng = np.random.default_rng(57)
    h, g, kappa = rng.normal(size=(3, 12))
    omega = rng.uniform(0.1, 0.8, 12)
    q = 0.6
    mean, variance = loading_parameters(h, g, omega, kappa, q)

    def joint(v):
        eta = g + v * h
        return np.dot(kappa, eta) - 0.5 * np.dot(omega, eta**2) + norm.logpdf(v, scale=q**0.5)

    for a, b in [(-0.5, 1.2), (0.1, 0.3)]:
        expected = norm.logpdf(a, mean, variance**0.5) - norm.logpdf(b, mean, variance**0.5)
        assert_allclose(joint(a) - joint(b), expected, atol=1e-8, rtol=1e-6)
    m, v = loading_parameters(np.zeros(12), g, omega, kappa, q)
    assert m == 0
    assert v == q


def test_indicator_odds_matches_two_joint_probabilities():
    rng = np.random.default_rng(71)
    h, g, kappa = rng.normal(size=(3, 20))
    omega = rng.uniform(0.1, 1, 20)
    pi = 0.3
    logs = []
    for bit in (0, 1):
        eta = g + bit * h
        logs.append(
            np.dot(kappa, eta) - 0.5 * np.dot(omega, eta**2) + np.log(pi if bit else 1 - pi)
        )
    assert_allclose(
        indicator_log_odds(h, g, omega, kappa, pi), logs[1] - logs[0], atol=1e-8, rtol=1e-6
    )
    assert_allclose(expit(indicator_log_odds(h * 0, g, omega, kappa, pi)), pi)


def test_precision_gaussian_covariance_and_linear_solve(problem):
    data, _ = problem
    rng = np.random.default_rng(992)
    omega = rng.uniform(0.1, 1, len(data.y))
    r = rng.normal(size=len(data.y))
    mean, lower = beta_parameters(data.X_tilde, omega, data.kappa, r, 0.7)
    Q = data.X_tilde.T @ np.diag(omega) @ data.X_tilde + np.eye(6) / 0.7
    b = data.X_tilde.T @ (data.kappa - omega * r)
    expected = np.linalg.inv(Q)  # Independent test reference only.
    assert_allclose(Q @ mean, b, atol=1e-8, rtol=1e-6)
    assert_allclose(lower @ lower.T, Q, atol=1e-8, rtol=1e-6)
    N = 50_000
    samples = draw_precision_normal(mean, lower, rng, size=N)
    assert np.all(np.abs(samples.mean(axis=1) - mean) < 5 * np.sqrt(np.diag(expected) / N))
    covariance = np.cov(samples)
    se = np.sqrt((np.outer(np.diag(expected), np.diag(expected)) + expected**2) / N)
    assert np.all(np.abs(covariance - expected) < 5 * se)


@pytest.mark.parametrize("q", [0, -1, np.inf])
def test_invalid_conditional_variance(q):
    with pytest.raises(ValueError):
        loading_parameters(np.ones(2), np.ones(2), np.ones(2), np.ones(2), q)


def test_logit_boundary_not_silently_clipped():
    with pytest.raises(ValueError):
        indicator_log_odds(np.ones(1), np.zeros(1), np.ones(1), np.zeros(1), 0)
    assert np.isfinite(logit(0.5))
