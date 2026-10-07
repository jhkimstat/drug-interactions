import numpy as np

from ..conditionals import beta_parameters, draw_precision_normal, variance_parameters
from ..distributions import inverse_gamma, polya_gamma


def update_common(data, spec, state, rng):
    state.context = "omega"
    state.omega = polya_gamma(state.eta, rng)
    r = state.eta - data.X_tilde @ state.beta
    state.context = "beta"
    mean, lower = beta_parameters(data.X_tilde, state.omega, data.kappa, r, state.sigma_beta2)
    state.beta = draw_precision_normal(mean, lower, rng)
    state.eta = data.X_tilde @ state.beta + r
    state.context = "sigma_beta2"
    a, b = variance_parameters(state.beta, spec.prior.a_beta, spec.prior.b_beta)
    state.sigma_beta2 = float(inverse_gamma(a, b, rng))


def update_loading(data, state, v, j, h, variance, rng):
    from ..conditionals import loading_parameters

    old = v[j]
    g = state.eta - old * h
    mean, var = loading_parameters(h, g, state.omega, data.kappa, variance)
    value = rng.normal(mean, np.sqrt(var))
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite loading draw")
    state.eta += (value - old) * h
    v[j] = value
