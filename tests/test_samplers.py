import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from scipy.stats import halfcauchy, norm

from factorregression.model import predictor_reference
from factorregression.samplers import horseshoe, normal, ssp_reference
from factorregression.state import Data, ModelSpec, initialize, validate_state


@pytest.mark.parametrize(
    "method,kernel",
    [
        ("normal", normal.sweep),
        ("horseshoe", horseshoe.sweep),
        ("ssp_reference", ssp_reference.sweep),
    ],
)
def test_full_sweeps_preserve_current_predictor(problem, method, kernel):
    data, spec = problem
    rng = np.random.default_rng(883)
    state = initialize(data, spec, method, rng, 3)
    for _ in range(30):
        kernel(data, spec, state, rng)
        validate_state(state, data, spec)
        assert_allclose(
            state.eta,
            predictor_reference(data.X, state.beta, state.effective()),
            atol=1e-12,
            rtol=1e-10,
        )


def reference_indicators(data, spec, state, rng, name):
    """Conditional mass from two explicit augmented joint evaluations."""
    for d, rank in spec.ranks.items():
        for k in range(rank):
            indices = [(k,)] if name == "z" else [(j, k) for j in range(spec.p)]
            for index in indices:
                logs = []
                pi = state.pi_z[d] if name == "z" else state.pi_gamma[d]
                for bit in (0, 1):
                    getattr(state, name)[d][index] = bit
                    eta = predictor_reference(data.X, state.beta, state.effective())
                    logs.append(
                        np.dot(data.kappa, eta)
                        - 0.5 * np.dot(state.omega, eta**2)
                        + np.log(pi if bit else 1 - pi)
                    )
                getattr(state, name)[d][index] = rng.random() < expit(logs[1] - logs[0])
    state.eta = predictor_reference(data.X, state.beta, state.effective())


def test_ssp_indicator_scan_matches_independent_joint(problem):
    data, spec = problem
    state = initialize(data, spec, "ssp_reference", np.random.default_rng(89), 3)
    state.tilde_v = {d: v * 3 for d, v in state.tilde_v.items()}
    state.eta = predictor_reference(data.X, state.beta, state.effective())
    state.omega[:] = 0.3
    reference = copy.deepcopy(state)
    actual_rng, reference_rng = np.random.default_rng(93), np.random.default_rng(93)
    for name, update in [("z", ssp_reference.update_z), ("gamma", ssp_reference.update_gamma)]:
        update(data, spec, state, actual_rng)
        reference_indicators(data, spec, reference, reference_rng, name)
        for d in spec.ranks:
            assert np.array_equal(getattr(state, name)[d], getattr(reference, name)[d])
        assert_allclose(state.eta, reference.eta, atol=1e-12, rtol=1e-10)


class RecordingRNG:
    def __init__(self):
        self.betas, self.gammas = [], []

    def beta(self, a, b):
        self.betas.append((a, b))
        return 0.5

    def gamma(self, a, size=None):
        self.gammas.append(a)
        return 1.0


def test_ssp_hyperparameters_include_inactive_indicators_and_slabs(problem):
    data, spec = problem
    state = initialize(data, spec, "ssp_reference", np.random.default_rng(85), 0)
    for d, rank in spec.ranks.items():
        state.z[d][:] = False
        state.gamma[d][:] = np.arange(spec.p * rank).reshape(spec.p, rank) % 2
        state.tilde_v[d][:] = np.arange(spec.p * rank).reshape(spec.p, rank) + 1
    recorder = RecordingRNG()
    ssp_reference.update_hyperparameters(spec, state, recorder)
    for index, (d, rank) in enumerate(spec.ranks.items()):
        count = state.gamma[d].sum()
        assert recorder.betas[2 * index] == (1 + count, 1 + spec.p * rank - count)
        assert recorder.betas[2 * index + 1] == (1, 1 + rank)
        assert recorder.gammas[index] == 4 + spec.p * rank / 2
        assert state.sigma_v2[d] == 1 + 0.5 * np.sum(state.tilde_v[d] ** 2)


def test_normal_variance_uses_every_loading(problem):
    data, spec = problem
    state = initialize(data, spec, "normal", np.random.default_rng(21))
    recorder = RecordingRNG()
    normal.update_variances(spec, state, recorder)
    for index, d in enumerate(spec.ranks):
        assert recorder.gammas[index] == 4 + state.V[d].size / 2
        assert state.sigma_v2[d] == 1 + 0.5 * np.sum(state.V[d] ** 2)


def test_inactive_slabs_are_prior_draws_without_predictor_change():
    data, spec = Data(np.zeros((2, 2)), np.array([0, 1])), ModelSpec(2, 2, {2: 1})
    rng = np.random.default_rng(111)
    state = initialize(data, spec, "ssp_reference", rng, 0)
    before = state.eta.copy()
    N = 50_000
    values = np.empty(N)
    for i in range(N):
        ssp_reference.update_slabs(data, spec, state, rng)
        values[i] = state.tilde_v[2][0, 0]
    assert_allclose(state.eta, before, atol=0, rtol=0)
    variance = state.sigma_v2[2]
    assert abs(values.mean()) < 5 * np.sqrt(variance / N)
    assert abs(values.var() - variance) < 5 * variance * np.sqrt(2 / N)


def test_inactive_component_gamma_conditions_only_on_pi(problem):
    data, spec = problem
    state = initialize(data, spec, "ssp_reference", np.random.default_rng(45), 0)
    reference_rng = np.random.default_rng(22)
    expected = {
        d: reference_rng.random((r, spec.p)).T < state.pi_gamma[d] for d, r in spec.ranks.items()
    }
    ssp_reference.update_gamma(data, spec, state, np.random.default_rng(22))
    for d in spec.ranks:
        assert np.array_equal(state.gamma[d], expected[d])


def test_horseshoe_transformed_density_jacobians(problem):
    data, spec = problem
    state = initialize(data, spec, "horseshoe", np.random.default_rng(22))
    v, tau = 0.3, 0.7
    x1, x2 = 0.2, 3.0

    def local_joint(x):
        lam = x**-0.5
        return (
            norm.logpdf(v, scale=lam * tau) + halfcauchy.logpdf(lam) - np.log(2) - 1.5 * np.log(x)
        )

    rate = v * v / (2 * tau * tau)
    assert_allclose(
        local_joint(x1) - local_joint(x2),
        -rate * (x1 - x2) - np.log1p(x1) + np.log1p(x2),
        atol=1e-12,
        rtol=1e-10,
    )

    def component_joint(zeta):
        t = zeta**-0.5
        return (
            norm.logpdf(state.V[2][:, 0], scale=state.lambda_[2][:, 0] * t).sum()
            + halfcauchy.logpdf(t)
            - np.log(2)
            - 1.5 * np.log(zeta)
        )

    rate = 0.5 * np.sum((state.V[2][:, 0] / state.lambda_[2][:, 0]) ** 2)
    ratio = (spec.p - 1) / 2 * np.log(x1 / x2) - rate * (x1 - x2) - np.log1p(x1) + np.log1p(x2)
    assert_allclose(component_joint(x1) - component_joint(x2), ratio, atol=1e-12, rtol=1e-10)


def test_horseshoe_zero_rate_boundaries(problem):
    data, spec = problem
    state = initialize(data, spec, "horseshoe", np.random.default_rng(222))
    for v in state.V.values():
        v[:] = 0
    before = state.eta.copy()
    rng = np.random.default_rng(45)
    horseshoe.update_local_scales(spec, state, rng)
    horseshoe.update_component_scales(spec, state, rng)
    assert all(np.isfinite(a).all() and (a > 0).all() for a in state.lambda_.values())
    assert all(np.isfinite(a).all() and (a > 0).all() for a in state.tau.values())
    assert np.array_equal(state.eta, before)
