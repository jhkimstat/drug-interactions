from itertools import combinations, product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from scipy.stats import bernoulli

from factorregression.conditionals import loading_parameters
from factorregression.model import (
    advance_symmetric_prefix,
    coefficient,
    elementary_symmetric,
    loading_slope,
    log_likelihood,
    predictor,
    predictor_reference,
    suffix_symmetric_coefficients,
)
from factorregression.state import Data, ModelSpec, Prior


@pytest.mark.parametrize("p,D", [(1, 1), (5, 2), (5, 3), (5, 5), (7, 4)])
def test_predictor_matches_explicit_combinations(p, D):
    rng = np.random.default_rng(23)
    X = rng.binomial(1, 0.5, (40, p)).astype(float)
    V = {d: rng.normal(size=(p, 1 + d % 3)) for d in range(2, D + 1)}
    beta = rng.normal(size=p + 1)
    assert_allclose(predictor(X, beta, V), predictor_reference(X, beta, V), atol=1e-8, rtol=1e-6)
    for d, load in V.items():
        for alpha in combinations(range(p), d):
            expected = sum(np.prod([load[j, k] for j in alpha]) for k in range(load.shape[1]))
            assert_allclose(coefficient(V, alpha), expected, atol=1e-8, rtol=1e-6)


def test_each_coordinate_is_conditionally_linear(problem):
    data, spec = problem
    rng = np.random.default_rng(12)
    V = {d: rng.normal(size=(spec.p, r)) for d, r in spec.ranks.items()}
    beta = rng.normal(size=spec.p + 1)
    for d, load in V.items():
        for k in range(load.shape[1]):
            for j in range(spec.p):
                before = predictor_reference(data.X, beta, V)
                h = loading_slope(data.X, load[:, k], d, j)
                load[j, k] += 0.7
                after = predictor_reference(data.X, beta, V)
                assert_allclose(after - before, 0.7 * h, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("p,d", [(5, 2), (5, 3), (8, 4), (5, 5)])
def test_sequential_prefix_suffix_matches_explicit_interactions(p, d):
    rng = np.random.default_rng(91)
    X = rng.binomial(1, 0.5, (40, p)).astype(float)
    v = rng.normal(0, 0.5, p)
    for _ in range(3):
        suffix = suffix_symmetric_coefficients(X * v, d - 1)
        prefix = np.zeros((len(X), d))
        prefix[:, 0] = 1.0
        for j in range(p):
            other = [i for i in range(p) if i != j]
            expected = (
                sum(
                    (np.prod(X[:, a] * v[list(a)], axis=1) for a in combinations(other, d - 1)),
                    np.zeros(len(X)),
                )
                * X[:, j]
            )
            assert_allclose(
                loading_slope(X, v, d, j, prefix=prefix, suffix=suffix),
                expected,
                atol=1e-8,
                rtol=1e-6,
            )
            v[j] = rng.normal(0, 0.5)
            advance_symmetric_prefix(prefix, X[:, j] * v[j], j)
            for t in range(1, d):
                expected_coefficient = sum(
                    (np.prod(X[:, a] * v[list(a)], axis=1) for a in combinations(range(j + 1), t)),
                    np.zeros(len(X)),
                )
                assert_allclose(prefix[:, t], expected_coefficient, atol=1e-8, rtol=1e-6)


def test_prefix_suffix_error_is_negligible_for_loading_conditional():
    rng = np.random.default_rng(492)
    X = rng.binomial(1, 0.5, (64, 100)).astype(float)
    v = rng.normal(0, 0.2, 100)
    omega = rng.uniform(0.1, 0.7, len(X))
    kappa = rng.binomial(1, 0.5, len(X)) - 0.5
    suffix = suffix_symmetric_coefficients(X * v, 2)
    prefix = np.zeros((len(X), 3))
    prefix[:, 0] = 1.0
    eta = 0.4 + elementary_symmetric(X * v, 3)
    for j in range(len(v)):
        other = np.arange(len(v)) != j
        reference_h = X[:, j] * elementary_symmetric(X[:, other] * v[other], 2)
        reference_eta = 0.4 + elementary_symmetric(X * v, 3)
        h = loading_slope(X, v, 3, j, prefix=prefix, suffix=suffix)
        mean, variance = loading_parameters(h, eta - v[j] * h, omega, kappa, 0.7)
        reference_mean, reference_variance = loading_parameters(
            reference_h, reference_eta - v[j] * reference_h, omega, kappa, 0.7
        )
        # Assess the conditional distribution, not agreement in the last digits of h.
        assert abs(mean - reference_mean) / np.sqrt(reference_variance) < 1e-6
        assert abs(variance / reference_variance - 1) < 1e-6
        old = v[j]
        v[j] = rng.normal(reference_mean, np.sqrt(reference_variance))
        eta += (v[j] - old) * h
        advance_symmetric_prefix(prefix, X[:, j] * v[j], j)


@pytest.mark.parametrize("magnitude", [1e3, 1e4, 1e6, 1e8])
def test_large_excluded_loading_keeps_slope_and_eta_consistent(magnitude):
    X = np.array(list(product((0, 1), repeat=5)), dtype=float)
    v = np.array([magnitude] + [1 / magnitude] * 4)
    V = {3: v[:, None]}
    eta = predictor_reference(X, np.zeros(6), V)
    suffix = suffix_symmetric_coefficients(X * v, 2)
    prefix = np.zeros((len(X), 3))
    prefix[:, 0] = 1.0
    for j in range(5):
        expected_h = X[:, j] * sum(
            (
                np.prod(X[:, a] * v[list(a)], axis=1)
                for a in combinations([q for q in range(5) if q != j], 2)
            ),
            np.zeros(len(X)),
        )
        h = loading_slope(X, v, 3, j, prefix=prefix, suffix=suffix)
        assert_allclose(h, expected_h, atol=1e-8, rtol=1e-6)
        assert (h[expected_h == 0] == 0).all()  # Structural zeros must stay zero.
        old = v[j]
        v[j] *= 0.5
        eta += (v[j] - old) * h
        advance_symmetric_prefix(prefix, X[:, j] * v[j], j)
        assert_allclose(eta, predictor_reference(X, np.zeros(6), V), atol=1e-8, rtol=1e-6)


def test_zero_and_degree_boundaries():
    a = np.zeros((2, 5))
    assert_allclose(elementary_symmetric(a, 0), 1)
    assert_allclose(elementary_symmetric(a, 3), 0)
    assert_allclose(elementary_symmetric(a, 6), 0)
    X = np.array(list(product((0, 1), repeat=5)), dtype=float)
    assert_allclose(predictor(X, np.zeros(6), {3: np.zeros((5, 2))}), 0)


def test_log_likelihood_matches_library_bernoulli_for_regular_logits():
    rng = np.random.default_rng(56)
    eta = rng.normal(size=32)
    y = rng.binomial(1, 0.5, 32)
    assert_allclose(
        log_likelihood(y, eta), bernoulli.logpmf(y, expit(eta)).sum(), atol=1e-8, rtol=1e-6
    )


def test_log_likelihood_remains_finite_for_extreme_logits():
    eta = np.array([1000.0, -1000.0, 1000.0, -1000.0])
    assert log_likelihood(np.array([1, 0, 0, 1]), eta) == -2000.0


@pytest.mark.parametrize(
    "args", [(5, 6, {}), (5, 3, {2: 1}), (5, 2, {2: 0}), (True, 1, {}), (5, 2, {2: 1.2})]
)
def test_invalid_spec(args):
    with pytest.raises(ValueError):
        ModelSpec(*args)


@pytest.mark.parametrize(
    "X,y", [([[0, 2]], [0]), ([[0, np.nan]], [0]), ([[0, 1]], [0, 1]), ([], []), ([[0, 1]], [0.1])]
)
def test_invalid_binary_data(X, y):
    with pytest.raises(ValueError):
        Data(X, y)


def test_prior_validation_and_data_ownership():
    with pytest.raises(ValueError):
        Prior(a_v=0)
    X = np.zeros((3, 2))
    data = Data(X, np.zeros(3))
    X[:] = 1
    assert not data.X.any()
    with pytest.raises(ValueError):
        data.X[0, 0] = 1
