"""Marginal posterior log density with the PG variables integrated out."""

import numpy as np
from scipy.special import xlog1py, xlogy
from scipy.stats import beta, halfcauchy, invgamma, norm

from .model import log_likelihood
from .state import HorseshoeState, SSPState


def log_prior(spec, state):
    """Full parameter prior, including inactive SSP variables."""
    p = spec.prior
    value = norm.logpdf(state.beta, scale=np.sqrt(state.sigma_beta2)).sum()
    value += invgamma.logpdf(state.sigma_beta2, p.a_beta, scale=p.b_beta)
    for d in spec.ranks:
        if isinstance(state, HorseshoeState):
            value += norm.logpdf(state.V[d], scale=state.lambda_[d] * state.tau[d]).sum()
            value += halfcauchy.logpdf(state.lambda_[d]).sum()
            value += halfcauchy.logpdf(state.tau[d]).sum()
        else:
            v = state.tilde_v[d] if isinstance(state, SSPState) else state.V[d]
            value += norm.logpdf(v, scale=np.sqrt(state.sigma_v2[d])).sum()
            value += invgamma.logpdf(state.sigma_v2[d], p.a_v, scale=p.b_v)
        if isinstance(state, SSPState):
            for bits, pi, a, b in [
                (state.gamma[d], state.pi_gamma[d], p.a_gamma, p.b_gamma),
                (state.z[d], state.pi_z[d], p.a_z, p.b_z),
            ]:
                value += (xlogy(bits, pi) + xlog1py(1 - bits.astype(int), -pi)).sum()
                value += beta.logpdf(pi, a, b)
    return float(value)


def log_posterior(data, spec, state):
    """Use the current maintained eta, rather than rebuilding it just for diagnostics."""
    return log_likelihood(data.y, state.eta) + log_prior(spec, state)
