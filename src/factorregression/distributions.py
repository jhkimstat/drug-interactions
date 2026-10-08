"""Library-backed distribution adapters and shared support checks."""

import numpy as np
from polyagamma import random_polyagamma
from scipy.special import gammainc, gammaincinv
from scipy.stats import truncexpon


def positive_finite(value, name):
    """Shared model-support check using NumPy predicates, with failure context."""
    if not np.isfinite(value).all() or not (np.asarray(value) > 0).all():
        raise FloatingPointError(f"{name} must be positive and finite")
    return value


def open_uniform(rng, size=None):
    """Endpoint policy for slice auxiliaries; NumPy only supplies half-open uniforms."""
    u = np.asarray(rng.random(size)) if size is not None else np.asarray(rng.random())
    invalid = (u <= 0) | (u >= 1)
    while np.any(invalid):
        u[invalid] = rng.random(np.count_nonzero(invalid))
        invalid = (u <= 0) | (u >= 1)
    return float(u) if u.ndim == 0 else u


def polya_gamma(eta, rng):
    """Policy adapter: PG(1,eta), explicit Generator and the library's Devroye method."""
    if not np.isfinite(eta).all():
        raise FloatingPointError("nonfinite PG tilt")
    return positive_finite(
        np.asarray(random_polyagamma(1.0, eta, method="devroye", random_state=rng)), "PG draw"
    )


def truncated_exponential(rate, upper, rng):
    """SciPy's truncated exponential in shape-rate units; scalar or broadcast arrays.

    A zero scaled rate uses NumPy's uniform limit. Array support lets independent
    Horseshoe local scales share a single library dispatch rather than p*R calls.
    """
    rate, upper = np.broadcast_arrays(np.asarray(rate, dtype=float), np.asarray(upper, dtype=float))
    positive_finite(upper, "truncation upper bound")
    if not np.isfinite(rate).all() or (rate < 0).any():
        raise FloatingPointError("exponential rate must be nonnegative and finite")
    c = rate * upper
    value = np.empty(rate.shape)
    uniform = c == 0
    if np.any(uniform):
        value[uniform] = rng.uniform(0, upper[uniform])
    positive = ~uniform
    if np.any(positive):
        value[positive] = truncexpon.rvs(c[positive], scale=1 / rate[positive], random_state=rng)
    positive_finite(value, "truncated exponential draw")
    if (value > upper).any():
        raise FloatingPointError("truncated exponential draw exceeded its upper bound")
    return float(value) if value.ndim == 0 else value


def truncated_gamma(shape, rate, upper, rng):
    """Gamma(shape, rate) restricted to (0, upper)."""
    positive_finite(np.array([shape, upper]), "Gamma shape/bound")
    if not np.isfinite(rate) or rate < 0:
        raise FloatingPointError("Gamma rate must be nonnegative and finite")
    if rate == 0:
        value = upper * rng.power(shape)
    else:
        c = rate * upper
        mass = gammainc(shape, c)
        q = open_uniform(rng) * mass
        value = gammaincinv(shape, q) / rate
    positive_finite(value, "truncated Gamma draw")
    if value > upper:
        raise FloatingPointError("truncated Gamma draw exceeded its upper bound")
    return float(value)
