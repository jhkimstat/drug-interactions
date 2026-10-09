"""Predictors on the original binary exposure scale, with distinct indices only."""

from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_expit

FloatArray = NDArray[np.float64]


def elementary_symmetric_coefficients(a: FloatArray, degree: int) -> FloatArray:
    """Rows of (e_0,...,e_degree) from the ordinary forward DP."""
    n, p = a.shape
    e = np.zeros((n, degree + 1))
    e[:, 0] = 1.0
    for j in range(p):
        for t in range(min(degree, j + 1), 0, -1):
            e[:, t] += a[:, j] * e[:, t - 1]
    return e


def elementary_symmetric(a: FloatArray, degree: int) -> FloatArray:
    """e_degree for each row; O(n * columns * degree) and O(n * degree) storage."""
    if degree > a.shape[1]:
        return np.zeros(a.shape[0])
    return elementary_symmetric_coefficients(a, degree)[:, degree]


def suffix_symmetric_coefficients(a: FloatArray, degree: int) -> FloatArray:
    """suffix[j] holds e_0,...,e_degree for columns j,...,p-1; O(np*degree)."""
    n, p = a.shape
    suffix = np.zeros((p + 1, n, degree + 1))
    suffix[:, :, 0] = 1.0
    for j in range(p - 1, -1, -1):
        suffix[j] = suffix[j + 1]
        stop = min(degree, p - j)
        suffix[j, :, 1 : stop + 1] += a[:, j, None] * suffix[j + 1, :, :stop]
    return suffix


def advance_symmetric_prefix(prefix: FloatArray, new_a: FloatArray, j: int) -> None:
    """Include the newly updated coordinate j in a rolling prefix.

    Multiplication materializes the old coefficients before the overlapping +=.
    """
    stop = min(prefix.shape[1] - 1, j + 1)
    prefix[:, 1 : stop + 1] += new_a[:, None] * prefix[:, :stop]


def loading_slope(
    X: FloatArray,
    v: FloatArray,
    d: int,
    j: int,
    *,
    prefix: FloatArray | None = None,
    suffix: FloatArray | None = None,
) -> FloatArray:
    """Excluded e_(d-1) from updated prefix and untouched suffix, without subtraction.

    A sampler builds the suffix once per component and advances the prefix after
    each draw, so each slope costs O(nd). Standalone calls build the two tables
    from current v; cached calls must follow the ascending coordinate scan.
    """
    degree = d - 1
    if prefix is None:
        prefix = elementary_symmetric_coefficients(X[:, :j] * v[:j], degree)
    if suffix is None:
        suffix = suffix_symmetric_coefficients(X * v, degree)
    low = max(0, degree - (X.shape[1] - j - 1))
    high = min(degree, j)
    h = prefix[:, low] * suffix[j + 1, :, degree - low]
    for t in range(low + 1, high + 1):
        h += prefix[:, t] * suffix[j + 1, :, degree - t]
    return X[:, j] * h


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
    """Bernoulli log likelihood using SciPy's stable log-sigmoid on signed logits."""
    return float(log_expit(np.where(y, eta, -eta)).sum())
