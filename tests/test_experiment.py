import csv
import json
from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from factorregression.evaluation import compare_runs, evaluate_fit, summarize
from factorregression.experiment import prepare, run_task
from factorregression.simulation import CASES, PATTERNS, generate


@pytest.mark.parametrize("index", range(13))
def test_note_dgps_support_norms_and_independent_probabilities(index):
    X, y, truth = generate(index, n=200, seed=371)
    assert X.shape == (200, 5) and y.shape == (200,)
    assert X.dtype == np.float64 and np.isin(X, (0, 1)).all() and np.isin(y, (0, 1)).all()
    name, structure, signal = CASES[index]
    assert signal in (1, 2, 3)
    expected_support = {
        "no-interaction": ([], []),
        "sparse-2way": ([12, 13, 23], []),
        "sparse-3way": ([12, 13, 23], [123]),
        "dense-2way": ([12, 13, 14, 15, 23, 24, 25, 34], []),
        "dense-3way": ([12, 13, 14, 15, 23, 24, 25, 34], [123, 124, 125, 134]),
    }[structure]
    alphas = [a for d in (2, 3) for a in combinations(range(1, 6), d)]
    selected = [
        int("".join(map(str, alpha)))
        for alpha, coefficient in zip(alphas, truth["theta"], strict=True)
        if coefficient != 0
    ]
    assert selected == expected_support[0] + expected_support[1]
    assert -1 <= truth["beta"][0] <= 1
    assert_allclose(np.linalg.norm(truth["beta"][1:]), 1)
    expected_norms = [1] + [signal if support else 0 for support in expected_support]
    assert_allclose(truth["realized_norms"], expected_norms)
    assert len(set(case[0] for case in CASES)) == 13 and name
    reference = []
    for pattern in PATTERNS:
        eta = truth["beta"][0] + sum(b * x for b, x in zip(truth["beta"][1:], pattern, strict=True))
        for alpha, coefficient in zip(alphas, truth["theta"], strict=True):
            eta += coefficient * np.prod([pattern[j - 1] for j in alpha])
        reference.append(expit(eta))
    assert_allclose(truth["prediction_probability"], reference, atol=1e-8, rtol=1e-6)
    replay = generate(index, n=200, seed=371)
    assert np.array_equal(X, replay[0]) and np.array_equal(y, replay[1])
    for key in truth:
        assert np.array_equal(truth[key], replay[2][key])


@pytest.fixture
def prepared_experiment(tmp_path):
    config = json.loads(open("configs/pilot.json").read())
    config["sampling"].update(burn_in=2, draws=20, max_seconds=120)
    config_path = tmp_path / "short.json"
    config_path.write_text(json.dumps(config))
    prepared = tmp_path / "prepared"
    manifest = prepare(prepared, config_path, config_path, n=32)
    return prepared, manifest


def test_preparation_freezes_configs_shares_data_and_separates_seeds(prepared_experiment):
    prepared, manifest = prepared_experiment
    pilot, long_run = (manifest["stages"][stage]["jobs"] for stage in ("pilot", "long-run"))
    assert len(pilot) == 39 and len(long_run) == 9
    assert {j["case"] for j in long_run} == {"no-interaction", "sparse-3way-s3", "dense-3way-s3"}
    assert {j["seed"] for j in pilot}.isdisjoint(j["seed"] for j in long_run)
    for job in pilot + long_run:
        with np.load(prepared / "datasets" / job["case"] / "observations.npz") as observed:
            assert set(observed.files) == {"X", "y"}
    with pytest.raises(FileExistsError):
        prepare(prepared, prepared / "pilot.json", prepared / "long-run.json")


def test_run_does_not_read_truth_and_reports_missing_fits(prepared_experiment, tmp_path):
    prepared, manifest = prepared_experiment
    truth_path = prepared / "datasets/no-interaction/truth.npz"
    truth_bytes = truth_path.read_bytes()
    truth_path.unlink()  # Sampling must succeed with no accessible truth fixture.
    output = tmp_path / "results"
    assert run_task(prepared, output, "pilot", 0)
    assert run_task(prepared, output, "long-run", 0)
    with pytest.raises(FileExistsError):
        run_task(prepared, output, "pilot", 0)
    truth_path.write_bytes(truth_bytes)
    row, functions, _ = evaluate_fit(
        prepared, output, "pilot", manifest["stages"]["pilot"]["jobs"][0]
    )
    assert row["completion"] == "completed" and len(functions) == 58
    assert row["theta_active_rmse"] is None  # No-interaction has no active coefficients.
    with np.load(output / "pilot/no-interaction/normal/chain-1.npz") as first:
        assert first["beta"].shape == (20, 6)
    p_functions = [f for f in functions if f["functional"].startswith("prediction_probability")]
    p = np.array([f["truth"] for f in p_functions])
    estimate = np.array([f["mean"] for f in p_functions])
    assert_allclose(row["probability_rmse"], np.sqrt(np.mean((p - estimate) ** 2)))
    assert_allclose(row["expected_brier"], np.mean(p * (1 - estimate) ** 2 + (1 - p) * estimate**2))
    report = tmp_path / "report"
    summarize(prepared, output, report)
    with (report / "fits.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48 and sum(r["completion"] == "missing" for r in rows) == 46
    with (report / "pilot-vs-long-run.csv").open() as stream:
        comparisons = list(csv.DictReader(stream))
    assert len(comparisons) == 58 + 8
    assert all(r["status"] != "reference_qualified" for r in comparisons)


@pytest.mark.parametrize(
    "reference_status,mcse,status",
    [
        ("diagnostics_ok", 0.02, "reference_qualified"),
        ("mixing_flagged", 0.02, "reference_not_converged"),
        ("diagnostics_ok", 0.05, "reference_mcse_too_large"),
    ],
)
def test_long_run_comparison_qualification_and_combined_mcse(reference_status, mcse, status):
    def result(seed, diagnostic, mean, error):
        return (
            {"case": "example", "method": "normal", "diagnostics": diagnostic},
            [
                {
                    "functional": "beta[0]",
                    "mean": mean,
                    "mcse_mean": error,
                    "q05": mean - 1,
                    "median": mean,
                    "q95": mean + 1,
                }
            ],
            {"model": {"p": 5}, "sampling": {"seed": seed}},
        )

    pilot = result(1, "diagnostics_ok", 0.3, 0.1)
    long_run = result(2, reference_status, 0.1, mcse)
    (row,) = compare_runs(pilot, long_run)
    assert row["status"] == status
    assert_allclose(row["standardized_mean_difference"], 0.2 / np.sqrt(0.1**2 + mcse**2))
    long_run[2]["model"]["p"] = 6
    with pytest.raises(ValueError, match="different models"):
        compare_runs(pilot, long_run)


def test_failed_fit_retains_failure_row_without_accuracy(
    prepared_experiment, tmp_path, monkeypatch
):
    from factorregression.samplers import normal

    def fail(data, spec, state, rng):
        state.context = "test failure"
        raise FloatingPointError("intentional numerical failure")

    monkeypatch.setattr(normal, "sweep", fail)
    prepared, manifest = prepared_experiment
    assert not run_task(prepared, tmp_path / "failed", "pilot", 0)
    row, functions, _ = evaluate_fit(
        prepared, tmp_path / "failed", "pilot", manifest["stages"]["pilot"]["jobs"][0]
    )
    assert row["completion"] == "numerical_failure" and not functions
    assert "intentional numerical failure" in row["failures"]
