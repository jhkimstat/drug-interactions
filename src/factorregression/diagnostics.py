"""MCMC diagnostics; scientific PIP/effect reporting is deferred."""

import warnings

import numpy as np


def scalar_diagnostics(values):
    import arviz as az

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.ptp(values) == 0:
        return {
            "status": "constant_or_nonfinite",
            "rhat": None,
            "bulk_ess": None,
            "tail_ess": None,
            "mcse_mean": None,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = {
            "status": "estimated",
            "rhat": float(az.rhat(values, method="rank")),
            "bulk_ess": float(az.ess(values, method="bulk")),
            "tail_ess": float(az.ess(values, method="tail")),
            "mcse_mean": float(az.mcse(values, method="mean")),
        }
        spread = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        result["quantile_mcse_ratio"] = (
            max(float(az.mcse(values, method="quantile", prob=q)) for q in (0.05, 0.5, 0.95))
            / spread
            if spread > 0
            else None
        )
    sd = float(np.std(values, ddof=1))
    result["relative_mean_mcse"] = result["mcse_mean"] / sd if sd > 0 else None
    return result


def diagnose(chains, requested_chains, thresholds):
    if len(chains) != requested_chains or any(c.status != "completed" for c in chains):
        return {"status": "incomplete", "variables": {}, "flags": ["incomplete chains"]}
    n = len(chains[0].draws["beta"])
    if requested_chains < 4 or n < 20:
        return {
            "status": "insufficient_draws",
            "variables": {},
            "flags": ["need four completed chains and twenty draws per chain"],
        }
    variables, flags, movement, movement_notes = {}, [], {}, []
    for key in chains[0].draws:
        if key in ("omega", "sweep"):
            continue
        values = np.stack([c.draws[key] for c in chains])
        scale = key.startswith(("lambda_", "tau_"))
        primary = not key.startswith(("V_", "lambda_", "tau_", "tilde_v_", "gamma_", "z_"))
        if scale:
            values = np.log(values)
        flat = values.reshape(requested_chains, n, -1)
        for j in range(flat.shape[-1]):
            name = ("log_" if scale else "") + key
            if flat.shape[-1] > 1:
                name += f"[{j}]"
            stats = scalar_diagnostics(flat[:, :, j])
            stats["primary"] = primary
            variables[name] = stats
            if not primary:
                continue
            if stats["status"] != "estimated":
                flags.append(f"{name}: undefined diagnostics")
                continue
            for metric, limit, lower_bound in [
                ("rhat", thresholds["rhat"], False),
                ("bulk_ess", thresholds["ess"], True),
                ("tail_ess", thresholds["ess"], True),
            ]:
                value = stats[metric]
                if not np.isfinite(value) or (value < limit if lower_bound else value >= limit):
                    flags.append(f"{name}: {metric}")
            if key in ("beta", "theta"):
                ratio = stats["relative_mean_mcse"]
                if ratio is None or not np.isfinite(ratio) or ratio > thresholds["relative_mcse"]:
                    flags.append(f"{name}: relative MCSE")
            if (
                key == "prediction_probability"
                and stats["mcse_mean"] > thresholds["probability_mcse"]
            ):
                flags.append(f"{name}: probability MCSE")
            if key.startswith("sigma_"):
                ratio = stats["quantile_mcse_ratio"]
                if ratio is None or not np.isfinite(ratio) or ratio > thresholds["quantile_mcse"]:
                    flags.append(f"{name}: quantile MCSE")
        if key.startswith(("z_", "gamma_", "K_", "M_")):
            changes = np.count_nonzero(np.diff(values.astype(float), axis=1), axis=1)
            movement[key] = changes.tolist()
            if key.startswith(("K_", "M_")) and np.any(changes == 0):
                flags.append(f"{key}: no changes in at least one chain")
            if key.startswith(("K_", "M_")) and np.any((changes > 0) & (changes < 20)):
                movement_notes.append(f"{key}: fewer than 20 changes in at least one chain")
    return {
        "status": "diagnostics_ok" if not flags else "mixing_flagged",
        "variables": variables,
        "flags": flags,
        "movement": movement,
        "movement_notes": movement_notes,
        "thresholds": thresholds,
    }
