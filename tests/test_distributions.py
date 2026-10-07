import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import invgamma, kstest

from factorregression.distributions import (
    inverse_gamma,
    polya_gamma,
    truncated_exponential,
    truncated_gamma,
)

N = 50_000


def pg_pdf_reference(x, c):
    """Independent Jacobi series, small/large x representations (PG paper)."""
    if x <= 0:
        return 0.0
    n = np.arange(16)
    odd = 2 * n + 1
    tilt = np.logaddexp(c / 2, -c / 2) - np.log(2) - c * c * x / 2
    if x <= 0.16:
        terms = odd * np.exp(tilt - odd**2 / (8 * x)) / np.sqrt(2 * np.pi * x**3)
    else:
        terms = 2 * np.pi * odd * np.exp(tilt - np.pi**2 * odd**2 * x / 2)
    return float(np.sum((-1.0) ** n * terms))


@pytest.mark.parametrize("c", [0, 1, -1, 5, -5, 20, -20, 100, -100])
def test_pg_analytic_moments_and_independent_cdf(c):
    samples = polya_gamma(np.full(N, c, dtype=float), np.random.default_rng(100 + abs(c)))
    assert np.isfinite(samples).all() and (samples > 0).all()
    mu = 0.25 if c == 0 else np.tanh(c / 2) / (2 * c)
    var = 1 / 24 if c == 0 else (np.sinh(c) - c) / (2 * c**3 * (np.cosh(c) + 1))
    assert abs(samples.mean() - mu) < 5 * np.sqrt(var / N)
    moment4 = (
        mu**4
        * quad(
            lambda t: pg_pdf_reference(mu * t, abs(c)) * mu * (t - 1) ** 4, 0, np.inf, epsabs=1e-10
        )[0]
    )
    assert abs(np.mean((samples - mu) ** 2) - var) < 5 * np.sqrt((moment4 - var**2) / N)
    for threshold in (0.5 * mu, mu, 2 * mu):
        prob = quad(lambda x: pg_pdf_reference(x, abs(c)), 0, threshold, epsabs=1e-10)[0]
        observed = np.mean(samples <= threshold)
        assert abs(observed - prob) <= 5 * np.sqrt(prob * (1 - prob) / N) + 1 / N


def test_inverse_gamma_parameterization():
    values = inverse_gamma(6, 2, np.random.default_rng(1), size=N)
    mean, variance = invgamma.stats(6, scale=2, moments="mv")
    assert abs(values.mean() - mean) < 5 * np.sqrt(variance / N)
    assert kstest(values, invgamma(6, scale=2).cdf).pvalue > 0.001 / 10


@pytest.mark.parametrize("rate,upper", [(0, 4), (1e-12, 4), (2, 4), (200, 0.5)])
def test_truncated_exponential_cdf(rate, upper):
    rng = np.random.default_rng(881)
    values = np.array([truncated_exponential(rate, upper, rng) for _ in range(N)])
    assert ((0 < values) & (values < upper)).all()
    if rate == 0:
        pit = values / upper
    else:
        pit = -np.expm1(-rate * values) / (-np.expm1(-rate * upper))
    assert kstest(pit, "uniform").pvalue > 0.001 / 10


@pytest.mark.parametrize(
    "shape,rate,upper", [(3, 0, 2), (3, 2, 2), (3, 1e-12, 2), (1000, 1, 1), (1000, 0.01, 1)]
)
def test_truncated_gamma_against_independent_integral(shape, rate, upper):
    rng = np.random.default_rng(9921)
    values = np.array([truncated_gamma(shape, rate, upper, rng) for _ in range(N)])
    assert ((0 < values) & (values < upper)).all()

    # Work on the unit interval with density scaled at its upper endpoint.
    def density(t):
        return 0 if t == 0 else np.exp((shape - 1) * np.log(t) + rate * upper * (1 - t))

    denominator = quad(density, 0, 1, epsabs=1e-12)[0]
    for q in (0.05, 0.5, 0.95):
        point = np.quantile(values, q) / upper
        expected = quad(density, 0, point, epsabs=1e-12)[0] / denominator
        assert abs(expected - q) < 5 * np.sqrt(q * (1 - q) / N)


def test_invalid_distribution_inputs():
    rng = np.random.default_rng(1)
    with pytest.raises(FloatingPointError):
        truncated_gamma(3, -1, 1, rng)
    with pytest.raises(FloatingPointError):
        truncated_exponential(1, 0, rng)
