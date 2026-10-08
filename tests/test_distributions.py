import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import kstest

from factorregression.distributions import (
    open_uniform,
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


@pytest.mark.parametrize("shape,rate,upper", [(3, 0, 2), (3, 2, 2), (3, 1e-12, 2)])
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


@pytest.mark.parametrize("rate", [1, 0.01])
def test_truncated_gamma_underflow_raises_instead_of_using_fallback(rate):
    with pytest.raises(
        FloatingPointError, match="truncated Gamma draw must be positive and finite"
    ):
        truncated_gamma(1000, rate, 1, np.random.default_rng(9921))


def test_invalid_distribution_inputs():
    rng = np.random.default_rng(1)
    with pytest.raises(FloatingPointError):
        truncated_exponential(1, 0, rng)
    with pytest.raises(FloatingPointError):
        truncated_gamma(3, -1, 2, rng)


def test_truncated_exponential_broadcasts_mixed_zero_and_positive_rates():
    rng = np.random.default_rng(94)
    rates = np.broadcast_to([0.0, 1e-12, 2.0], (N, 3))
    upper = np.array([4.0, 3.0, 2.0])
    values = truncated_exponential(rates, upper, rng)
    assert values.shape == rates.shape
    assert ((values > 0) & (values <= upper)).all()
    for j, rate in enumerate(rates[0]):
        pit = (
            values[:, j] / upper[j]
            if rate == 0
            else -np.expm1(-rate * values[:, j]) / (-np.expm1(-rate * upper[j]))
        )
        assert kstest(pit, "uniform").pvalue > 0.001 / 10


def test_open_uniform_scalar_and_array_reject_zero_endpoints():
    class EndpointRNG:
        def random(self, size=None):
            return 0.0 if size is None else np.full(size, 0.5)

    assert open_uniform(EndpointRNG()) == 0.5
    values = open_uniform(np.random.default_rng(74), size=(4, 3))
    assert values.shape == (4, 3) and ((values > 0) & (values < 1)).all()


def test_rounded_truncation_endpoint_does_not_cause_spurious_failure():
    class NearOneRNG:
        def power(self, shape):
            return 1.0  # Power draw can round to its upper endpoint.

    value = truncated_gamma(10, 0, 2, NearOneRNG())
    assert value == 2
