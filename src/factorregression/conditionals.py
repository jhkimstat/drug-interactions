"""Parameters of the conditionals in the modeling notes."""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import logit


def loading_parameters(h, g, omega, kappa, prior_variance):
    if not np.isfinite(prior_variance) or prior_variance <= 0:
        raise ValueError("prior variance must be positive and finite")
    precision = np.dot(omega, h * h) + 1.0 / prior_variance
    linear = np.dot(h, kappa - omega * g)
    if not np.isfinite(precision + linear) or precision <= 0:
        raise FloatingPointError("invalid loading conditional")
    return linear / precision, 1.0 / precision


def indicator_log_odds(h, g, omega, kappa, pi):
    if not 0 < pi < 1:
        raise ValueError("inclusion probability must be strictly between 0 and 1")
    return float(logit(pi) + np.dot(h, kappa - omega * g) - 0.5 * np.dot(omega, h * h))


def variance_parameters(values, a, b):
    return a + values.size / 2, b + 0.5 * float(np.sum(values * values))


def beta_parameters(X_tilde, omega, kappa, r, sigma_beta2):
    precision = X_tilde.T @ (omega[:, None] * X_tilde)
    precision += np.eye(X_tilde.shape[1]) / sigma_beta2
    linear = X_tilde.T @ (kappa - omega * r)
    # Inputs are validated by the sampler; avoid repeated scans and a separate tril copy.
    lower = cholesky(precision, lower=True, check_finite=False)
    mean = cho_solve((lower, True), linear, check_finite=False)
    return mean, lower


def draw_precision_normal(mean, lower, rng, size=None):
    """If Q=L L^T, noise is L^{-T} z, not L^{-1} z."""
    shape = (len(mean),) if size is None else (len(mean), size)
    noise = solve_triangular(lower.T, rng.normal(size=shape), lower=False, check_finite=False)
    return mean + noise if size is None else mean[:, None] + noise
