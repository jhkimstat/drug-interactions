import numpy as np
from scipy.stats import invgamma

from ..conditionals import variance_parameters
from ..distributions import positive_finite
from ..model import advance_symmetric_prefix, loading_slope, suffix_symmetric_coefficients
from .common import update_common, update_loading


def update_loadings(data, spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            v = state.V[d][:, k]
            suffix = suffix_symmetric_coefficients(data.X * v, d - 1)
            prefix = np.zeros((len(data.y), d))
            prefix[:, 0] = 1.0
            for j in range(spec.p):
                state.context = f"V[{d},{j},{k}]"
                h = loading_slope(data.X, v, d, j, prefix=prefix, suffix=suffix)
                update_loading(data, state, v, j, h, state.sigma_v2[d], rng)
                advance_symmetric_prefix(prefix, data.X[:, j] * v[j], j)


def update_variances(spec, state, rng):
    for d in spec.ranks:
        state.context = f"sigma_v2[{d}]"
        a, b = variance_parameters(state.V[d], spec.prior.a_v, spec.prior.b_v)
        state.sigma_v2[d] = float(
            positive_finite(invgamma.rvs(a, scale=b, random_state=rng), "sigma_v2")
        )


def sweep(data, spec, state, rng):
    update_common(data, spec, state, rng)
    update_loadings(data, spec, state, rng)
    update_variances(spec, state, rng)
