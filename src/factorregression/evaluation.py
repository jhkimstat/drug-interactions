"""Simulation-only accuracy and efficiency tables from saved pilot/long-run fits."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import xlog1py, xlogy

from .simulation import ALPHAS, PATTERNS

QUANTITIES = ("beta", "theta", "prediction_probability")


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return value is not None and np.isfinite(value)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_fit(prepared, results, stage, job):
    """Incomplete/missing fits keep a table row; accuracy is evaluated only when complete."""
    directory = Path(results) / stage / job["case"] / job["method"]
    row = {"stage": stage, "case": job["case"], "method": job["method"], "task_id": job["task_id"]}
    if not (directory / "metadata.json").exists():
        row["completion"] = "unfinished" if directory.exists() else "missing"
        return row, [], None
    metadata = read_json(directory / "metadata.json")
    diagnostics = read_json(directory / "diagnostics.json")
    config = read_json(directory / "configuration.json")
    seconds = metadata["sampling_seconds"]
    sweeps = sum(chain["completed_sweeps"] for chain in metadata["chains"])
    row.update(
        completion=metadata["status"],
        diagnostics=diagnostics["status"],
        sampling_seconds=seconds,
        sampling_and_diagnostics_seconds=metadata["elapsed_seconds"],
        completed_sweeps=sweeps,
        seconds_per_sweep=seconds / sweeps if sweeps else None,
        diagnostic_flags=len(diagnostics["flags"]),
        failures=json.dumps([c["failure"] for c in metadata["chains"] if c["failure"]]),
    )
    if (directory / "execution.json").exists():
        row.update(read_json(directory / "execution.json"))
    if metadata["status"] != "completed":
        return row, [], config
    if metadata["method"] != job["method"] or config["sampling"]["seed"] != job["seed"]:
        raise ValueError(f"fit method/seed does not match prepared task: {directory}")
    sampling = config["sampling"]
    if (
        sampling["interaction_tuples"] != list(map(list, ALPHAS))
        or sampling["prediction_patterns"] != PATTERNS.tolist()
    ):
        raise ValueError(f"unexpected functional ordering: {directory}")
    dataset = Path(prepared) / "datasets" / job["case"]
    with (
        np.load(dataset / "observations.npz") as expected,
        np.load(directory / "observations.npz") as actual,
    ):
        if not all(np.array_equal(expected[key], actual[key]) for key in ("X", "y")):
            raise ValueError(f"observed data differ from the prepared dataset: {directory}")
    with np.load(dataset / "truth.npz") as stored:
        truth = {key: stored[key] for key in QUANTITIES}
    chain_values = {key: [] for key in QUANTITIES}
    for index in range(sampling["chains"]):
        with np.load(directory / f"chain-{index + 1}.npz") as draws:
            for key in QUANTITIES:
                values = draws[key]
                if values.shape != (sampling["draws"], len(truth[key])):
                    raise ValueError(f"incomplete or incompatible saved draws: {directory}")
                chain_values[key].append(values)
    functions, means = [], {}
    for key in QUANTITIES:
        values = np.stack(chain_values[key])
        means[key] = values.mean(axis=(0, 1))
        pooled = values.reshape(-1, values.shape[-1])
        quantiles = np.quantile(pooled, [0.05, 0.5, 0.95], axis=0)
        for index in range(values.shape[-1]):
            name = f"{key}[{index}]"
            stats = diagnostics["variables"].get(name, {})
            entry = {
                "stage": stage,
                "case": job["case"],
                "method": job["method"],
                "functional": name,
                "truth": float(truth[key][index]),
                "mean": float(means[key][index]),
                "q05": float(quantiles[0, index]),
                "median": float(quantiles[1, index]),
                "q95": float(quantiles[2, index]),
                "diagnostic_status": stats.get("status"),
            }
            for metric in ("rhat", "bulk_ess", "tail_ess", "mcse_mean", "relative_mean_mcse"):
                entry[metric] = stats.get(metric)
            for metric in ("bulk_ess", "tail_ess"):
                entry[f"{metric}_per_second"] = (
                    stats[metric] / seconds if finite(stats.get(metric)) and seconds > 0 else None
                )
            functions.append(entry)
    row["common_functionals"] = len(functions)
    row["undefined_common_diagnostics"] = sum(
        any(not finite(f[metric]) for metric in ("rhat", "bulk_ess", "tail_ess", "mcse_mean"))
        for f in functions
    )
    for prefix, metric, column in (
        ("prediction_probability", "mcse_mean", "probability_mcse_max"),
        ("beta", "relative_mean_mcse", "beta_relative_mcse_max"),
        ("theta", "relative_mean_mcse", "theta_relative_mcse_max"),
    ):
        valid = [
            f[metric] for f in functions if f["functional"].startswith(prefix) and finite(f[metric])
        ]
        row[column] = max(valid) if valid else None
    for metric, quantiles in (
        ("rhat", (("max", 1),)),
        ("bulk_ess_per_second", (("p10", 0.1), ("median", 0.5))),
        ("tail_ess_per_second", (("p10", 0.1), ("median", 0.5))),
    ):
        valid = [f[metric] for f in functions if finite(f[metric])]
        for label, q in quantiles:
            row[f"common_{metric}_{label}"] = float(np.quantile(valid, q)) if valid else None
    for key, mask, name in (
        ("beta", slice(None), "beta"),
        ("theta", slice(0, 10), "theta2"),
        ("theta", slice(10, 20), "theta3"),
        ("theta", truth["theta"] != 0, "theta_active"),
        ("theta", truth["theta"] == 0, "theta_inactive"),
        ("prediction_probability", slice(None), "probability"),
    ):
        error = means[key][mask] - truth[key][mask]
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2))) if error.size else None
    p, estimate = truth["prediction_probability"], means["prediction_probability"]
    row["expected_log_loss"] = float(-np.mean(xlogy(p, estimate) + xlog1py(1 - p, -estimate)))
    row["expected_brier"] = float(np.mean((estimate - p) ** 2 + p * (1 - p)))
    return row, functions, config


def compare_runs(pilot, long_run):
    """Same-data, same-target independent runs; a long run is not an exact oracle."""
    p_row, p_functions, p_config = pilot
    l_row, l_functions, l_config = long_run
    base = {"case": l_row["case"], "method": l_row["method"]}
    if not p_functions or not l_functions:
        return [{**base, "status": "missing_or_incomplete_pair"}]
    if p_config["model"] != l_config["model"]:
        raise ValueError("cannot compare runs with different models/priors/ranks")
    if p_config["sampling"]["seed"] == l_config["sampling"]["seed"]:
        raise ValueError("pilot and long-run must have independent seeds")
    rows = []
    for p, l in zip(p_functions, l_functions, strict=True):
        if p["functional"] != l["functional"]:
            raise ValueError("functional order differs between stages")
        p_mcse, l_mcse = p["mcse_mean"], l["mcse_mean"]
        valid = finite(p_mcse) and finite(l_mcse) and p_mcse > 0 and l_mcse > 0
        status = (
            "reference_not_converged"
            if l_row["diagnostics"] != "diagnostics_ok"
            else "undefined_mcse"
            if not valid
            else "reference_mcse_too_large"
            if l_mcse > p_mcse / 3
            else "reference_qualified"
        )
        rows.append(
            {
                **base,
                "functional": p["functional"],
                "status": status,
                "pilot_diagnostics": p_row["diagnostics"],
                "long_run_diagnostics": l_row["diagnostics"],
                "mean_difference": p["mean"] - l["mean"],
                "combined_mcse": float(np.hypot(p_mcse, l_mcse)) if valid else None,
                "standardized_mean_difference": (
                    (p["mean"] - l["mean"]) / np.hypot(p_mcse, l_mcse) if valid else None
                ),
                "q05_difference": p["q05"] - l["q05"],
                "median_difference": p["median"] - l["median"],
                "q95_difference": p["q95"] - l["q95"],
            }
        )
    return rows


def summarize(prepared, results, output):
    manifest = read_json(Path(prepared) / "manifest.json")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fits, functions, comparisons, evaluated = [], [], [], {}
    for stage, stage_spec in manifest["stages"].items():
        for job in stage_spec["jobs"]:
            result = evaluate_fit(prepared, results, stage, job)
            fits.append(result[0])
            functions.extend(result[1])
            evaluated[stage, job["case"], job["method"]] = result
    for job in manifest["stages"]["long-run"]["jobs"]:
        key = (job["case"], job["method"])
        comparisons.extend(compare_runs(evaluated["pilot", *key], evaluated["long-run", *key]))
    write_csv(output / "fits.csv", fits)
    write_csv(output / "functionals.csv", functions)
    write_csv(output / "pilot-vs-long-run.csv", comparisons)
    print(
        json.dumps(
            {
                "expected_fits": len(fits),
                "completed_fits": sum(row["completion"] == "completed" for row in fits),
                "output": str(output),
            }
        )
    )
