"""Independent low-dimensional posterior references, not simulation accuracy studies."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, roots_hermitenorm
from scipy.stats import t

from factorregression import ModelSpec, SamplerSettings, fit
from factorregression.diagnostics import scalar_diagnostics
from factorregression.distributions import polya_gamma
from factorregression.model import predictor_reference
from factorregression.samplers.normal import update_loadings
from factorregression.state import Data, initialize


@pytest.mark.slow
def test_main_only_posterior_against_marginal_student_t_quadrature():
    X = np.zeros((24, 1))
    y = np.array([1] * 15 + [0] * 9)
    spec = ModelSpec(1, 1, {})
    result = fit(
        X,
        y,
        method="normal",
        spec=spec,
        settings=SamplerSettings(burn_in=1000, draws=3000, seed=871, max_seconds=120),
    )
    assert result.metadata["status"] == "completed"
    samples = np.stack([c.draws["beta"][:, 0] for c in result.chains])
    # All unused beta_j and sigma_beta2 are analytically integrated from the joint prior.
    mle = np.log(15 / 9)
    maximum = 15 * mle - 24 * np.logaddexp(0, mle)

    def density(b):
        log_prior = t.logpdf(
            b, df=2 * spec.prior.a_beta, scale=np.sqrt(spec.prior.b_beta / spec.prior.a_beta)
        )
        return np.exp(log_prior + 15 * b - 24 * np.logaddexp(0, b) - maximum)

    denominator = quad(density, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-8)[0]
    expected_mean = (
        quad(lambda b: b * density(b), -np.inf, np.inf, epsabs=1e-10, epsrel=1e-8)[0] / denominator
    )
    stats = scalar_diagnostics(samples)
    assert stats["rhat"] < 1.01 and stats["bulk_ess"] >= 400 and stats["tail_ess"] >= 400
    assert abs(samples.mean() - expected_mean) < 5 * stats["mcse_mean"] + 1e-8
    for point in (0, 0.5, 1):
        reference = quad(density, -np.inf, point, epsabs=1e-10, epsrel=1e-8)[0] / denominator
        events = (samples <= point).astype(float)
        mcse = scalar_diagnostics(events)["mcse_mean"]
        assert abs(events.mean() - reference) < 5 * mcse + 1e-8


@pytest.mark.slow
def test_normal_interaction_block_against_gaussian_tensor_quadrature():
    # Fixed nuisance beta and sigma_v2: isolates the nonlinear loading/PG kernel.
    data, spec = Data(np.ones((20, 2)), np.array([1] * 12 + [0] * 8)), ModelSpec(2, 2, {2: 1})
    variance = 0.7

    def reference(order):
        nodes, weights = roots_hermitenorm(order)
        theta = variance * np.outer(nodes, nodes)
        eta = 0.2 + theta
        logs = 12 * eta - 20 * np.logaddexp(0, eta)
        posterior = np.outer(weights, weights) * np.exp(logs - logs.max())
        posterior /= posterior.sum()
        return np.array([(posterior * theta).sum(), (posterior * expit(eta)).sum()])

    expected = reference(320)
    assert np.max(np.abs(expected - reference(640))) < 1e-8
    theta, probability = np.empty((4, 5000)), np.empty((4, 5000))
    for chain in range(4):
        rng = np.random.default_rng(80 + chain)
        state = initialize(data, spec, "normal", rng, chain)
        state.beta[:] = [-0.3, 0.1, 0.4]
        state.sigma_v2[2] = variance
        state.eta = predictor_reference(data.X, state.beta, state.effective())
        for sweep in range(5500):
            state.omega = polya_gamma(state.eta, rng)
            update_loadings(data, spec, state, rng)
            if sweep >= 500:
                value = np.prod(state.V[2][:, 0])
                theta[chain, sweep - 500] = value
                probability[chain, sweep - 500] = expit(0.2 + value)
    for values, mean in zip((theta, probability), expected, strict=True):
        stats = scalar_diagnostics(values)
        assert stats["rhat"] < 1.01 and stats["bulk_ess"] >= 400 and stats["tail_ess"] >= 400
        assert abs(values.mean() - mean) < 5 * stats["mcse_mean"] + 1e-8
