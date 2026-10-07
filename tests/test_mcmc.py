import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from factorregression import ModelSpec, SamplerSettings, fit, mcmc
from factorregression.cli import load_configuration, main
from factorregression.diagnostics import scalar_diagnostics
from factorregression.mcmc import rng_stream
from factorregression.model import log_likelihood, predictor_reference
from factorregression.samplers import normal
from factorregression.state import initialize
from factorregression.target import log_prior


@pytest.mark.parametrize("method", ["normal", "horseshoe", "ssp_reference"])
def test_fit_reproducibility_storage_and_no_input_mutation(problem, tmp_path, method):
    data, spec = problem
    X, y = data.X.copy(), data.y.copy()
    settings = SamplerSettings(chains=2, burn_in=5, draws=8, thin=2, cache_every=1)
    result = fit(X, y, method=method, spec=spec, settings=settings, output=tmp_path / "run")
    again = fit(X, y, method=method, spec=spec, settings=settings)
    assert result.metadata["status"] == "completed"
    assert np.array_equal(X, data.X) and np.array_equal(y, data.y)
    for c, other in zip(result.chains, again.chains, strict=True):
        assert np.array_equal(c.draws["sweep"], np.arange(7, 22, 2))
        assert "omega" not in c.draws
        for key, values in c.draws.items():
            assert np.array_equal(values, other.draws[key])
        assert not np.array_equal(result.chains[0].draws["beta"], result.chains[1].draws["beta"])
    with np.load(tmp_path / "run/chain-1.npz", allow_pickle=False) as saved:
        assert_allclose(saved["theta"], result.chains[0].draws["theta"])
    with open(tmp_path / "run/metadata.json") as stream:
        assert json.load(stream)["status"] == "completed"
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "run")


def test_general_dimensions_and_explicit_output_queries():
    X = np.random.default_rng(9).binomial(1, 0.5, (12, 6))
    y = X[:, 0]
    settings = SamplerSettings(
        chains=1,
        burn_in=0,
        draws=3,
        save_omega=True,
        interaction_tuples=((0, 1, 2, 3),),
        cache_every=1,
    )
    result = fit(X, y, method="normal", spec=ModelSpec(6, 4, {2: 1, 3: 2, 4: 1}), settings=settings)
    assert result.metadata["status"] == "completed"
    assert result.chains[0].draws["theta"].shape == (3, 1)
    assert result.chains[0].draws["omega"].shape == (3, 12)


def test_budget_and_numerical_failure_are_not_reported_as_complete(problem, monkeypatch):
    data, spec = problem
    settings = SamplerSettings(chains=4, burn_in=2, draws=3, max_seconds=1e-12)
    result = fit(data.X, data.y, method="normal", spec=spec, settings=settings)
    assert result.metadata["status"] == "budget_exhausted"
    assert result.diagnostics["status"] == "incomplete"

    def broken(*args):
        raise FloatingPointError("test numerical failure")

    monkeypatch.setattr(normal, "sweep", broken)
    result = fit(
        data.X, data.y, method="normal", spec=spec, settings=replace(settings, max_seconds=60)
    )
    assert result.metadata["status"] == "numerical_failure"
    assert result.chains[0].metadata["retained_draws"] == 0
    assert "test numerical failure" in result.chains[0].metadata["failure"]["reason"]


@pytest.mark.parametrize("drift,expected", [(1e-10, "completed"), (1e-3, "numerical_failure")])
def test_cache_guard_accepts_rounding_but_rejects_material_drift(
    problem, monkeypatch, drift, expected
):
    data, spec = problem

    def drifting_kernel(data, spec, state, rng):
        state.eta += drift

    monkeypatch.setattr(normal, "sweep", drifting_kernel)
    result = fit(
        data.X,
        data.y,
        method="normal",
        spec=spec,
        settings=SamplerSettings(chains=1, burn_in=0, draws=2, cache_every=1),
    )
    assert result.metadata["status"] == expected
    if expected == "numerical_failure":
        assert "cache mismatch" in result.chains[0].metadata["failure"]["reason"]


def test_snapshot_reuses_observed_eta_and_evaluates_only_requested_patterns(problem, monkeypatch):
    data, spec = problem
    state = initialize(data, spec, "normal", np.random.default_rng(841))
    reference = log_likelihood(data.y, predictor_reference(data.X, state.beta, state.effective()))
    original = mcmc.predictor
    rows = []

    def record_rows(X, beta, V):
        rows.append(len(X))
        return original(X, beta, V)

    monkeypatch.setattr(mcmc, "predictor", record_rows)
    values = mcmc._snapshot(state, data, spec, SamplerSettings(), (), data.X[:2], 1)
    assert rows == [2]
    assert_allclose(values["log_likelihood"], reference, atol=1e-8, rtol=1e-6)
    assert_allclose(
        values["log_posterior"], reference + log_prior(spec, state), atol=1e-8, rtol=1e-6
    )


def test_memory_and_configuration_guards(problem, tmp_path):
    data, spec = problem
    with pytest.raises(ValueError, match="bytes"):
        fit(data.X, data.y, method="normal", spec=spec, settings=SamplerSettings(max_draw_bytes=1))
    with pytest.raises(ValueError):
        SamplerSettings(thin=0)
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {"model": {"p": 5, "D": 3, "ranks": {"2": 1, "3": 1}}, "sampling": {"unused_option": 1}}
        )
    )
    with pytest.raises(ValueError, match="unknown"):
        load_configuration(bad)


def test_rng_keys_are_independent_and_order_stable():
    expected = rng_stream(83, method=2, chain=3).normal(size=10)
    rng_stream(83, method=0, chain=0).normal(size=100)
    assert np.array_equal(expected, rng_stream(83, method=2, chain=3).normal(size=10))
    assert not np.array_equal(expected, rng_stream(83, method=2, chain=4).normal(size=10))


def test_diagnostics_flag_constant_and_distinct_chain_means():
    assert scalar_diagnostics(np.zeros((4, 200)))["rhat"] is None
    values = np.random.default_rng(19).normal(size=(4, 1000))
    values[0] += 5
    assert scalar_diagnostics(values)["rhat"] > 1.01


def test_boolean_diagnostics_and_four_chain_ssp(problem):
    values = np.random.default_rng(22).random((4, 1000)) < 0.4
    assert np.isfinite(scalar_diagnostics(values)["rhat"])
    data, spec = problem
    result = fit(
        data.X,
        data.y,
        method="ssp_reference",
        spec=spec,
        settings=SamplerSettings(burn_in=10, draws=20, cache_every=1),
    )
    assert result.metadata["status"] == "completed"
    assert "gamma_2[0]" in result.diagnostics["variables"]


def test_empty_derived_outputs_are_supported(problem):
    data, spec = problem
    result = fit(
        data.X,
        data.y,
        method="normal",
        spec=spec,
        settings=SamplerSettings(
            chains=1, burn_in=1, draws=2, interaction_tuples=(), prediction_patterns=()
        ),
    )
    assert result.chains[0].draws["theta"].shape == (2, 0)
    assert result.chains[0].draws["prediction_probability"].shape == (2, 0)


def test_cli_roundtrip(problem, tmp_path):
    data, _ = problem
    np.savez(tmp_path / "input.npz", X=data.X, y=data.y)
    assert (
        main(
            [
                "fit",
                "--data",
                str(tmp_path / "input.npz"),
                "--config",
                "configs/smoke.json",
                "--method",
                "normal",
                "--chains",
                "1",
                "--burn-in",
                "2",
                "--draws",
                "3",
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    with open(tmp_path / "output/configuration.json") as stream:
        config = json.load(stream)
    assert config["sampling"]["chains"] == 1
    assert config["sampling"]["draws"] == 3
    assert (
        main(
            [
                "fit",
                "--data",
                str(tmp_path / "output/observations.npz"),
                "--config",
                str(tmp_path / "output/configuration.json"),
                "--method",
                "normal",
                "--output",
                str(tmp_path / "rerun"),
            ]
        )
        == 0
    )
    with (
        np.load(tmp_path / "output/chain-1.npz") as first,
        np.load(tmp_path / "rerun/chain-1.npz") as second,
    ):
        assert np.array_equal(first["beta"], second["beta"])
