from itertools import combinations, product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from factorregression.model import (
    coefficient,
    elementary_symmetric,
    loading_slope,
    predictor,
    predictor_reference,
)
from factorregression.state import Data, ModelSpec, Prior


@pytest.mark.parametrize("p,D", [(1, 1), (5, 2), (5, 3), (5, 5), (7, 4)])
def test_predictor_matches_explicit_combinations(p, D):
    rng = np.random.default_rng(23)
    X = rng.binomial(1, 0.5, (40, p)).astype(float)
    V = {d: rng.normal(size=(p, 1 + d % 3)) for d in range(2, D + 1)}
    beta = rng.normal(size=p + 1)
    assert_allclose(predictor(X, beta, V), predictor_reference(X, beta, V), atol=1e-12, rtol=1e-10)
    for d, load in V.items():
        for alpha in combinations(range(p), d):
            expected = sum(np.prod([load[j, k] for j in alpha]) for k in range(load.shape[1]))
            assert_allclose(coefficient(V, alpha), expected, atol=1e-12, rtol=1e-10)


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
                assert_allclose(after - before, 0.7 * h, atol=1e-12, rtol=1e-10)


def test_zero_and_degree_boundaries():
    a = np.zeros((2, 5))
    assert_allclose(elementary_symmetric(a, 0), 1)
    assert_allclose(elementary_symmetric(a, 3), 0)
    assert_allclose(elementary_symmetric(a, 6), 0)
    X = np.array(list(product((0, 1), repeat=5)), dtype=float)
    assert_allclose(predictor(X, np.zeros(6), {3: np.zeros((5, 2))}), 0)


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
