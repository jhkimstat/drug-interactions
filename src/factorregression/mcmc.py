"""Explicit sweep loops, independent RNG streams and portable result files."""

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.special import expit

from .diagnostics import diagnose
from .model import coefficient, log_likelihood, predictor
from .samplers import horseshoe, normal, ssp_reference
from .state import (
    METHODS,
    Data,
    HorseshoeState,
    ModelSpec,
    SSPState,
    State,
    initialize,
    positive_int,
    validate_state,
)
from .target import log_prior


@dataclass(frozen=True)
class SamplerSettings:
    chains: int = 4
    burn_in: int = 2000
    draws: int = 2000
    thin: int = 1
    seed: int = 20261006
    cache_every: int = 100
    cache_atol: float = 1e-8
    cache_rtol: float = 1e-6
    max_seconds: float | None = 2700.0
    max_draw_bytes: int = 512 * 1024**2
    save_omega: bool = False
    interaction_tuples: tuple[tuple[int, ...], ...] | None = None
    prediction_patterns: tuple[tuple[int, ...], ...] | None = None
    rhat_limit: float = 1.01
    ess_min: float = 400.0
    relative_mcse_limit: float = 0.05
    probability_mcse_limit: float = 0.01
    quantile_mcse_limit: float = 0.05

    def __post_init__(self):
        for name in ("chains", "draws", "thin", "cache_every", "max_draw_bytes"):
            positive_int(getattr(self, name), name)
        for name in ("burn_in", "seed"):
            positive_int(getattr(self, name), name, 0)
        for name in (
            "cache_atol",
            "cache_rtol",
            "rhat_limit",
            "ess_min",
            "relative_mcse_limit",
            "probability_mcse_limit",
            "quantile_mcse_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_seconds is not None and (
            isinstance(self.max_seconds, bool)
            or not np.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be positive/finite or null")
        if not isinstance(self.save_omega, bool):
            raise ValueError("save_omega must be a boolean")


@dataclass
class ChainResult:
    draws: dict[str, np.ndarray]
    status: str
    metadata: dict


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class FitResult:
    chains: list[ChainResult]
    configuration: dict
    diagnostics: dict
    metadata: dict
    data: Data = field(repr=False)

    def save(self, directory):
        root = Path(directory)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"output must be an empty/new directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "observations.npz", X=self.data.X, y=self.data.y)
        for i, chain in enumerate(self.chains):
            np.savez_compressed(root / f"chain-{i + 1}.npz", **chain.draws)
        for name, value in {
            "configuration": self.configuration,
            "diagnostics": self.diagnostics,
            "metadata": {**self.metadata, "chains": [c.metadata for c in self.chains]},
        }.items():
            (root / f"{name}.json").write_text(
                json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n"
            )


def rng_stream(seed, *, purpose=0, method=0, chain=0, dataset=0, replication=0):
    """Stable integer keys, independent of execution order and Python hash."""
    for name, value in locals().copy().items():
        positive_int(value, name, 0)
    return np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(purpose, dataset, replication, method, chain))
    )


def _outputs(data, spec, settings):
    if settings.interaction_tuples is None:
        alphas = (
            tuple(alpha for d in spec.ranks for alpha in combinations(range(spec.p), d))
            if spec.p == 5 and spec.D == 3
            else ()
        )
    else:
        alphas = tuple(tuple(a) for a in settings.interaction_tuples)
    for alpha in alphas:
        if (
            any(isinstance(j, bool) or not isinstance(j, int) for j in alpha)
            or len(alpha) not in spec.ranks
            or tuple(sorted(set(alpha))) != alpha
            or alpha[0] < 0
            or alpha[-1] >= spec.p
        ):
            raise ValueError("invalid zero-based interaction output tuple")
    if len(set(alphas)) != len(alphas):
        raise ValueError("duplicate interaction output tuples")
    if settings.prediction_patterns is None:
        patterns = (
            np.array(list(product((0.0, 1.0), repeat=5)))
            if spec.p == 5 and spec.D == 3
            else data.X[: min(16, len(data.y))].copy()
        )
    else:
        patterns = np.asarray(settings.prediction_patterns, dtype=float)
        if patterns.size == 0:
            patterns = np.empty((0, spec.p), dtype=float)
        if patterns.ndim != 2 or patterns.shape[1] != spec.p or not np.isin(patterns, [0, 1]).all():
            raise ValueError("prediction patterns must be a binary (rows,p) array")
    return alphas, patterns


def _snapshot(state, data, spec, settings, alphas, patterns, sweep):
    values = {
        "beta": state.beta.copy(),
        "sigma_beta2": np.asarray(state.sigma_beta2),
        "sweep": np.asarray(sweep, dtype=np.int64),
    }
    if settings.save_omega:
        values["omega"] = state.omega.copy()
    for d in spec.ranks:
        if isinstance(state, SSPState):
            for name in ("z", "gamma", "tilde_v", "pi_z", "pi_gamma", "sigma_v2"):
                values[f"{name}_{d}"] = np.array(getattr(state, name)[d], copy=True)
            values[f"K_{d}"] = np.asarray(state.z[d].sum())
            values[f"T_{d}"] = np.asarray(state.gamma[d].sum())
            values[f"M_{d}"] = np.asarray((state.gamma[d] * state.z[d]).sum())
        else:
            values[f"V_{d}"] = state.V[d].copy()
            if isinstance(state, HorseshoeState):
                values[f"lambda_{d}"] = state.lambda_[d].copy()
                values[f"tau_{d}"] = state.tau[d].copy()
            else:
                values[f"sigma_v2_{d}"] = np.asarray(state.sigma_v2[d])
    V = state.effective() if alphas or len(patterns) else {}
    values["theta"] = np.array([coefficient(V, a) for a in alphas])
    values["prediction_probability"] = (
        expit(predictor(patterns, state.beta, V)) if len(patterns) else np.empty(0)
    )
    values["log_likelihood"] = np.asarray(log_likelihood(data.y, state.eta))
    values["log_posterior"] = np.asarray(values["log_likelihood"] + log_prior(spec, state))
    if not all(np.isfinite(a).all() for a in values.values()):
        raise FloatingPointError("nonfinite retained quantity")
    return values


def _run_chain(data, spec, settings, method, rng, state, alphas, patterns, deadline, index):
    start = time.monotonic()
    first = _snapshot(state, data, spec, settings, alphas, patterns, 0)
    draws = {k: np.empty((settings.draws, *a.shape), dtype=a.dtype) for k, a in first.items()}
    retained, completed_sweeps, status, failure = 0, 0, "completed", None
    initial = {k: v.tolist() for k, v in first.items() if k != "omega"}
    kernel = {
        "normal": normal.sweep,
        "horseshoe": horseshoe.sweep,
        "ssp_reference": ssp_reference.sweep,
    }[method]
    total = settings.burn_in + settings.draws * settings.thin
    sweep = 0
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for sweep in range(1, total + 1):
                kernel(data, spec, state, rng)
                if sweep % settings.cache_every == 0 or sweep == total:
                    state.context = "eta cache check"
                    validate_state(state, data, spec)
                    eta = predictor(data.X, state.beta, state.effective())
                    if not np.allclose(
                        eta, state.eta, atol=settings.cache_atol, rtol=settings.cache_rtol
                    ):
                        raise FloatingPointError(
                            f"eta cache mismatch; max error={np.max(np.abs(eta - state.eta))}"
                        )
                    state.eta = eta
                if sweep > settings.burn_in and (sweep - settings.burn_in) % settings.thin == 0:
                    state.context = "retained snapshot"
                    for key, value in _snapshot(
                        state, data, spec, settings, alphas, patterns, sweep
                    ).items():
                        draws[key][retained] = value
                    retained += 1
                completed_sweeps = sweep
                if deadline is not None and time.monotonic() >= deadline and sweep < total:
                    status = "budget_exhausted"
                    break
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, OverflowError) as exc:
        status = "numerical_failure"
        failure = {"sweep": sweep, "update": state.context, "reason": str(exc)}
    draws = {k: a[:retained].copy() for k, a in draws.items()}
    metadata = {
        "chain": index,
        "status": status,
        "completed_sweeps": completed_sweeps,
        "retained_draws": retained,
        "elapsed_seconds": time.monotonic() - start,
        "initial": initial,
        "failure": failure,
    }
    return ChainResult(draws, status, metadata)


def fit(
    X,
    y,
    *,
    method,
    spec: ModelSpec,
    settings: SamplerSettings | None = None,
    initial_states: list[State] | None = None,
    output=None,
) -> FitResult:
    """Fit observed inputs only. Output requests never restrict interaction fitting."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    settings = settings or SamplerSettings()
    data = Data(X, y)
    if data.X.shape[1] != spec.p:
        raise ValueError("observed p differs from ModelSpec.p")
    if initial_states is not None and len(initial_states) != settings.chains:
        raise ValueError("one initial state per chain is required")
    alphas, patterns = _outputs(data, spec, settings)
    probe = initialize(data, spec, method, rng_stream(settings.seed), 0)
    schema = _snapshot(probe, data, spec, settings, alphas, patterns, 0)
    estimated_bytes = sum(a.nbytes for a in schema.values()) * settings.chains * settings.draws
    if estimated_bytes > settings.max_draw_bytes:
        raise ValueError(
            f"requested draws need about {estimated_bytes} bytes; limit={settings.max_draw_bytes}"
        )
    if output is not None:
        root = Path(output)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"output must be an empty/new directory: {root}")
    started = time.monotonic()
    deadline = started + settings.max_seconds if settings.max_seconds is not None else None
    chains, completion = [], "completed"
    for index in range(settings.chains):
        if deadline is not None and time.monotonic() >= deadline:
            completion = "budget_exhausted"
            break
        rng = rng_stream(settings.seed, purpose=1, method=METHODS.index(method), chain=index)
        state = (
            copy.deepcopy(initial_states[index])
            if initial_states is not None
            else initialize(data, spec, method, rng, index)
        )
        if not isinstance(state, type(probe)):
            raise ValueError("initial state type differs from requested method")
        validate_state(state, data, spec)
        if initial_states is not None:
            expected_eta = predictor(data.X, state.beta, state.effective())
            if not np.allclose(
                state.eta, expected_eta, atol=settings.cache_atol, rtol=settings.cache_rtol
            ):
                raise ValueError("initial eta differs from its parameter state")
        chain = _run_chain(
            data, spec, settings, method, rng, state, alphas, patterns, deadline, index
        )
        chains.append(chain)
        if chain.status != "completed":
            completion = chain.status
            break
    thresholds = {
        "rhat": settings.rhat_limit,
        "ess": settings.ess_min,
        "relative_mcse": settings.relative_mcse_limit,
        "probability_mcse": settings.probability_mcse_limit,
        "quantile_mcse": settings.quantile_mcse_limit,
    }
    resolved_sampling = asdict(settings)
    resolved_sampling["interaction_tuples"] = alphas
    resolved_sampling["prediction_patterns"] = patterns.tolist()
    configuration = {
        "model": {"p": spec.p, "D": spec.D, "ranks": dict(spec.ranks), "prior": asdict(spec.prior)},
        "sampling": resolved_sampling,
    }
    metadata = {
        "status": completion,
        "method": method,
        "interaction_indexing": "zero-based, distinct, ascending",
        "n": len(data.y),
        "elapsed_seconds": time.monotonic() - started,
        "estimated_draw_bytes": estimated_bytes,
        "versions": {
            name: version(name)
            for name in ("factorregression", "numpy", "scipy", "polyagamma", "arviz")
        },
    }
    diagnostics = diagnose(chains, settings.chains, thresholds)
    metadata["sampling_seconds"] = metadata["elapsed_seconds"]
    metadata["elapsed_seconds"] = time.monotonic() - started
    result = FitResult(chains, configuration, diagnostics, metadata, data)
    if output is not None:
        result.save(output)
    return result
