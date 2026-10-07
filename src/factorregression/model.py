"""Predictors on the original binary exposure scale, with distinct indices only."""

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def elementary_symmetric(a: FloatArray, degree: int) -> FloatArray:
    """e_degree for each row; O(n * columns * degree) and O(n * degree) storage."""
    n, p = a.shape
    if degree > p:
        return np.zeros(n)
    e = np.zeros((n, degree + 1))
    e[:, 0] = 1.0
    for j in range(p):
        for t in range(min(degree, j + 1), 0, -1):
            e[:, t] += a[:, j] * e[:, t - 1]
    return e[:, degree]


def loading_slope(X: FloatArray, v: FloatArray, d: int, j: int) -> FloatArray:
    """Current h_v; recomputing the excluded polynomial avoids subtraction cancellation."""
    other = np.arange(X.shape[1]) != j
    return X[:, j] * elementary_symmetric(X[:, other] * v[other], d - 1)


def interactions(X: FloatArray, V: dict[int, FloatArray]) -> FloatArray:
    r = np.zeros(X.shape[0])
    for d, loadings in V.items():
        for k in range(loadings.shape[1]):
            r += elementary_symmetric(X * loadings[:, k], d)
    return r


def predictor(X: FloatArray, beta: FloatArray, V: dict[int, FloatArray]) -> FloatArray:
    return beta[0] + X @ beta[1:] + interactions(X, V)


def coefficient(V: dict[int, FloatArray], alpha: tuple[int, ...]) -> float:
    """Interaction tuple uses zero-based, distinct, ascending variable indices."""
    if not alpha or tuple(sorted(set(alpha))) != alpha or len(alpha) not in V:
        raise ValueError("interaction tuple must be distinct, ascending and of an included order")
    loadings = V[len(alpha)]
    if alpha[0] < 0 or alpha[-1] >= loadings.shape[0]:
        raise ValueError("interaction index outside the exposure dimension")
    return float(np.prod(loadings[list(alpha)], axis=0).sum())


def predictor_reference(X: FloatArray, beta: FloatArray, V: dict[int, FloatArray]) -> FloatArray:
    """Independent explicit sum for small problems. Never used in fitting sweeps."""
    eta = beta[0] + X @ beta[1:]
    for d, loadings in V.items():
        for alpha in combinations(range(X.shape[1]), d):
            theta = sum(np.prod([loadings[j, k] for j in alpha]) for k in range(loadings.shape[1]))
            eta = eta + theta * np.prod(X[:, alpha], axis=1)
    return eta


def log_likelihood(y: FloatArray, eta: FloatArray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))
