import numpy as np

from ..distributions import open_uniform, positive_finite, truncated_exponential, truncated_gamma
from ..model import loading_slope
from .common import update_common, update_loading


def update_loadings(data, spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            v = state.V[d][:, k]
            for j in range(spec.p):
                state.context = f"V[{d},{j},{k}]"
                h = loading_slope(data.X, v, d, j)
                variance = (state.lambda_[d][j, k] * state.tau[d][k]) ** 2
                update_loading(data, state, v, j, h, variance, rng)


def update_local_scales(spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            for j in range(spec.p):
                state.context = f"lambda[{d},{j},{k}]"
                xi = state.lambda_[d][j, k] ** -2
                u = open_uniform(rng) / (1 + xi)
                upper = (1 - u) / u
                rate = 0.5 * (state.V[d][j, k] / state.tau[d][k]) ** 2
                xi = truncated_exponential(rate, upper, rng)
                state.lambda_[d][j, k] = positive_finite(xi**-0.5, "lambda")


def update_component_scales(spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            state.context = f"tau[{d},{k}]"
            zeta = state.tau[d][k] ** -2
            u = open_uniform(rng) / (1 + zeta)
            upper = (1 - u) / u
            rate = 0.5 * np.sum((state.V[d][:, k] / state.lambda_[d][:, k]) ** 2)
            zeta = truncated_gamma((spec.p + 1) / 2, rate, upper, rng)
            state.tau[d][k] = positive_finite(zeta**-0.5, "tau")


def sweep(data, spec, state, rng):
    update_common(data, spec, state, rng)
    update_loadings(data, spec, state, rng)
    update_local_scales(spec, state, rng)
    update_component_scales(spec, state, rng)
