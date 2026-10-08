"""The 13 note-based DGPs, with the user's updated signal strengths (1, 2, 3)."""

from itertools import combinations, product

import numpy as np
from scipy.special import expit

from .mcmc import rng_stream
from .state import positive_int

# Note indices 1,...,5 are stored as ascending zero-based tuples.
SUPPORTS = {
    "no-interaction": ((), ()),
    "sparse-2way": (((0, 1), (0, 2), (1, 2)), ()),
    "sparse-3way": (((0, 1), (0, 2), (1, 2)), ((0, 1, 2),)),
    "dense-2way": (((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3)), ()),
    "dense-3way": (
        ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3)),
        ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3)),
    ),
}
ALPHAS = tuple(alpha for d in (2, 3) for alpha in combinations(range(5), d))
PATTERNS = np.array(list(product((0, 1), repeat=5)), dtype=np.float64)
CASES = [("no-interaction", "no-interaction", 1)] + [
    (f"{structure}-s{signal}", structure, signal)
    for structure in SUPPORTS
    if structure != "no-interaction"
    for signal in (1, 2, 3)
]
LONG_RUN_CASES = ("no-interaction", "sparse-3way-s3", "dense-3way-s3")


def generate(case_index, n=200, seed=20261008):
    """Return observed arrays and a separate truth fixture; no factor truth is assumed."""
    positive_int(case_index, "case_index", 0)
    positive_int(n, "n")
    if case_index >= len(CASES):
        raise ValueError("case_index must be between 0 and 12")
    _, structure, signal = CASES[case_index]
    truth_rng = rng_stream(seed, purpose=10, dataset=case_index)
    data_rng = rng_stream(seed, purpose=11, dataset=case_index)

    def coefficients(size, norm):
        values = truth_rng.uniform(0.5, 1.5, size) * truth_rng.choice([-1, 1], size)
        return norm * values / np.linalg.norm(values)

    beta = np.empty(6)
    beta[0] = truth_rng.uniform(-1, 1)
    beta[1:] = coefficients(5, 1)
    theta = np.zeros(len(ALPHAS))
    for support in SUPPORTS[structure]:
        if support:
            indices = [ALPHAS.index(alpha) for alpha in support]
            theta[indices] = coefficients(len(indices), signal)

    def probability(X):
        eta = beta[0] + X @ beta[1:]
        for alpha, effect in zip(ALPHAS, theta, strict=True):
            eta += effect * X[:, alpha].prod(axis=1)
        return expit(eta)

    X = data_rng.binomial(1, 0.5, size=(n, 5)).astype(np.float64)
    y = data_rng.binomial(1, probability(X)).astype(np.float64)
    truth = {
        "beta": beta,
        "theta": theta,
        "prediction_probability": probability(PATTERNS),
        "prediction_patterns": PATTERNS,
        "realized_norms": np.array([1, np.linalg.norm(theta[:10]), np.linalg.norm(theta[10:])]),
        "signal_strengths": np.array([1, signal, signal]),
    }
    return X, y, truth
