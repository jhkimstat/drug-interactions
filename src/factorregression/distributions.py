"""Distribution adapters, including exact slice-kernel boundary cases."""

import numpy as np
from polyagamma import random_polyagamma
from scipy.special import gammainc, gammaincinv


def positive_finite(value, name):
    if not np.isfinite(value).all() or not (np.asarray(value) > 0).all():
        raise FloatingPointError(f"{name} must be positive and finite")
    return value


def open_uniform(rng):
    # RNG has [0,1) support; reject the measure-zero mathematical endpoint.
    while True:
        u = rng.random()
        if 0 < u < 1:
            return float(u)


def inverse_gamma(a, b, rng, size=None):
    positive_finite(np.array([a, b]), "IG parameters")
    return positive_finite(b / rng.gamma(a, size=size), "IG draw")


def polya_gamma(eta, rng):
    if not np.isfinite(eta).all():
        raise FloatingPointError("nonfinite PG tilt")
    return positive_finite(
        np.asarray(random_polyagamma(1.0, eta, method="devroye", random_state=rng)), "PG draw"
    )


def truncated_exponential(rate, upper, rng):
    """Exp(rate) restricted to (0,upper); rate=0 is the exact uniform limit."""
    positive_finite(upper, "truncation upper bound")
    if not np.isfinite(rate) or rate < 0:
        raise FloatingPointError("exponential rate must be nonnegative and finite")
    u = open_uniform(rng)
    c = rate * upper
    if c == 0:
        value = upper * u
    elif c < 1e-5:
        # Normalize to the unit interval. Division before scaling avoids underflow.
        value = upper * (-np.log1p(-u * -np.expm1(-c)) / c)
    else:
        value = -np.log1p(-u * -np.expm1(-c)) / rate
    positive_finite(value, "truncated exponential draw")
    if value >= upper:
        raise FloatingPointError("truncated exponential draw reached its upper bound")
    return float(value)


def truncated_gamma(shape, rate, upper, rng):
    """Gamma(shape,rate) on (0,upper), without replacing a zero rate or CDF floor.

    Inverse CDF is the usual path. If its truncation mass underflows, use exact
    rejection: a bounded power proposal, or the tangent envelope of the log-concave
    gamma density at the upper boundary. No tiny CDF is replaced by a constant.
    """
    positive_finite(np.array([shape, upper]), "Gamma shape/bound")
    if not np.isfinite(rate) or rate < 0:
        raise FloatingPointError("Gamma rate must be nonnegative and finite")
    if rate == 0:
        value = upper * np.exp(np.log(open_uniform(rng)) / shape)
    else:
        c = rate * upper
        mass = gammainc(shape, c)
        q = open_uniform(rng) * mass
        value = gammaincinv(shape, q) / rate if q > 0 else 0.0
        if not 0 < value < upper or not np.isfinite(value):
            for _ in range(100_000):
                if c <= 1:
                    t = np.exp(np.log(open_uniform(rng)) / shape)
                    log_accept = -c * t
                elif shape > 1 and c <= shape - 1:
                    tangent_rate = shape - 1 - c
                    t = 1 - truncated_exponential(tangent_rate, 1.0, rng)
                    log_accept = (shape - 1) * (np.log(t) - (t - 1))
                else:
                    # A numerically exceptional inverse CDF, not a tiny lower tail.
                    trial = rng.gamma(shape) / rate
                    if 0 < trial < upper:
                        value = trial
                        break
                    continue
                if np.log(open_uniform(rng)) <= log_accept:
                    value = upper * t
                    break
            else:
                raise FloatingPointError("truncated Gamma rejection limit exceeded")
    positive_finite(value, "truncated Gamma draw")
    if value >= upper:
        raise FloatingPointError("truncated Gamma draw reached its upper bound")
    return float(value)
