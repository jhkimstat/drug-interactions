from ..conditionals import variance_parameters
from ..distributions import inverse_gamma
from ..model import elementary_symmetric_coefficients, loading_slope
from .common import update_common, update_loading


def update_loadings(data, spec, state, rng):
    for d, rank in spec.ranks.items():
        for k in range(rank):
            v = state.V[d][:, k]
            polynomial = elementary_symmetric_coefficients(data.X * v, d - 1)
            for j in range(spec.p):
                state.context = f"V[{d},{j},{k}]"
                h = loading_slope(data.X, v, d, j, polynomial)
                update_loading(data, state, v, j, h, state.sigma_v2[d], rng, polynomial)


def update_variances(spec, state, rng):
    for d in spec.ranks:
        state.context = f"sigma_v2[{d}]"
        a, b = variance_parameters(state.V[d], spec.prior.a_v, spec.prior.b_v)
        state.sigma_v2[d] = float(inverse_gamma(a, b, rng))


def sweep(data, spec, state, rng):
    update_common(data, spec, state, rng)
    update_loadings(data, spec, state, rng)
    update_variances(spec, state, rng)
