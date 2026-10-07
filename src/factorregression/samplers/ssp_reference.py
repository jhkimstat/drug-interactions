import numpy as np
from scipy.special import expit

from ..conditionals import indicator_log_odds, variance_parameters
from ..distributions import inverse_gamma
from ..model import (
    elementary_symmetric,
    elementary_symmetric_coefficients,
    loading_slope,
    update_symmetric_coefficients,
)
from .common import update_common, update_loading


def update_z(data, spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            state.context = f"z[{d},{k}]"
            v = state.gamma[d][:, k] * state.tilde_v[d][:, k]
            h = elementary_symmetric(data.X * v, d)
            old = int(state.z[d][k])
            g = state.eta - old * h
            odds = indicator_log_odds(h, g, state.omega, data.kappa, state.pi_z[d])
            if not np.isfinite(odds):
                raise FloatingPointError("nonfinite z odds")
            new = rng.random() < expit(odds)
            state.z[d][k] = new
            state.eta += (int(new) - old) * h


def update_gamma(data, spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            state.context = f"gamma[{d},:,{k}]"
            if not state.z[d][k]:
                state.gamma[d][:, k] = rng.random(spec.p) < state.pi_gamma[d]
                continue
            v = state.gamma[d][:, k] * state.tilde_v[d][:, k]
            polynomial = elementary_symmetric_coefficients(data.X * v, d - 1)
            for j in range(spec.p):
                state.context = f"gamma[{d},{j},{k}]"
                h = state.tilde_v[d][j, k] * loading_slope(data.X, v, d, j, polynomial)
                old = int(state.gamma[d][j, k])
                g = state.eta - old * h
                odds = indicator_log_odds(h, g, state.omega, data.kappa, state.pi_gamma[d])
                if not np.isfinite(odds):
                    raise FloatingPointError("nonfinite gamma odds")
                new = rng.random() < expit(odds)
                state.gamma[d][j, k] = new
                if new != old:
                    delta_v = (int(new) - old) * state.tilde_v[d][j, k]
                    state.eta += (int(new) - old) * h
                    update_symmetric_coefficients(
                        polynomial, data.X[:, j] * v[j], data.X[:, j] * delta_v
                    )
                    v[j] += delta_v


def update_slabs(data, spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            state.context = f"tilde_v[{d},:,{k}]"
            if not state.z[d][k] or not state.gamma[d][:, k].any():
                state.tilde_v[d][:, k] = rng.normal(0, np.sqrt(state.sigma_v2[d]), spec.p)
                continue
            v = state.gamma[d][:, k] * state.tilde_v[d][:, k]
            polynomial = elementary_symmetric_coefficients(data.X * v, d - 1)
            for j in range(spec.p):
                state.context = f"tilde_v[{d},{j},{k}]"
                if state.z[d][k] and state.gamma[d][j, k]:
                    h = loading_slope(data.X, v, d, j, polynomial)
                    update_loading(
                        data,
                        state,
                        state.tilde_v[d][:, k],
                        j,
                        h,
                        state.sigma_v2[d],
                        rng,
                        polynomial,
                    )
                    v[j] = state.tilde_v[d][j, k]
                else:
                    state.tilde_v[d][j, k] = rng.normal(0, np.sqrt(state.sigma_v2[d]))


def update_hyperparameters(spec, state, rng):
    prior = spec.prior
    for d, rank in spec.ranks.items():
        state.context = f"SSP hyperparameters[{d}]"
        total_gamma = int(state.gamma[d].sum())  # Includes z=0 components.
        total_z = int(state.z[d].sum())
        state.pi_gamma[d] = rng.beta(
            prior.a_gamma + total_gamma, prior.b_gamma + spec.p * rank - total_gamma
        )
        state.pi_z[d] = rng.beta(prior.a_z + total_z, prior.b_z + rank - total_z)
        if not (0 < state.pi_z[d] < 1 and 0 < state.pi_gamma[d] < 1):
            raise FloatingPointError("invalid SSP inclusion probability")
        a, b = variance_parameters(state.tilde_v[d], prior.a_v, prior.b_v)
        state.sigma_v2[d] = float(inverse_gamma(a, b, rng))  # Includes inactive slabs.


def sweep(data, spec, state, rng):
    update_common(data, spec, state, rng)
    update_z(data, spec, state, rng)
    update_gamma(data, spec, state, rng)
    update_slabs(data, spec, state, rng)
    update_hyperparameters(spec, state, rng)
