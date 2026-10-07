"""Explicit model specifications and the three sampler states."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np

from .model import FloatArray, predictor

Method = Literal["normal", "horseshoe", "ssp_reference"]
METHODS = ("normal", "horseshoe", "ssp_reference")


def positive_int(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Prior:
    a_beta: float = 3.0
    b_beta: float = 2.0
    a_v: float = 4.0
    b_v: float = 1.0
    a_gamma: float = 1.0
    b_gamma: float = 1.0
    a_z: float = 1.0
    b_z: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class ModelSpec:
    p: int
    D: int
    ranks: Mapping[int, int]
    prior: Prior = field(default_factory=Prior)

    def __post_init__(self) -> None:
        positive_int(self.p, "p")
        positive_int(self.D, "D")
        if self.D > self.p:
            raise ValueError("D must be <= p")
        ranks = dict(self.ranks)
        if set(ranks) != set(range(2, self.D + 1)):
            raise ValueError("ranks must specify exactly orders 2,...,D")
        for d, rank in ranks.items():
            positive_int(d, "order", 2)
            positive_int(rank, f"R_{d}")
        object.__setattr__(self, "ranks", MappingProxyType(dict(sorted(ranks.items()))))


@dataclass(frozen=True)
class Data:
    X: FloatArray
    y: FloatArray
    X_tilde: FloatArray = field(init=False, repr=False)
    kappa: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        X, y = (
            np.array(self.X, dtype=np.float64, copy=True),
            np.array(self.y, dtype=np.float64, copy=True),
        )
        if X.ndim != 2 or min(X.shape) < 1 or y.shape != (X.shape[0],):
            raise ValueError("X must be (n,p) and y must be (n,), with n,p >= 1")
        if not np.isin(X, (0, 1)).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("X and y must contain only binary 0/1 values")
        for name, array in {
            "X": X,
            "y": y,
            "X_tilde": np.column_stack((np.ones(len(y)), X)),
            "kappa": y - 0.5,
        }.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass
class BaseState:
    beta: FloatArray
    sigma_beta2: float
    omega: FloatArray
    eta: FloatArray
    context: str = field(default="initialization", init=False)


@dataclass
class NormalState(BaseState):
    V: dict[int, FloatArray]
    sigma_v2: dict[int, float]

    def effective(self) -> dict[int, FloatArray]:
        return self.V


@dataclass
class HorseshoeState(BaseState):
    V: dict[int, FloatArray]
    lambda_: dict[int, FloatArray]
    tau: dict[int, FloatArray]

    def effective(self) -> dict[int, FloatArray]:
        return self.V


@dataclass
class SSPState(BaseState):
    z: dict[int, np.ndarray]
    gamma: dict[int, np.ndarray]
    tilde_v: dict[int, FloatArray]
    pi_z: dict[int, float]
    pi_gamma: dict[int, float]
    sigma_v2: dict[int, float]

    def effective(self) -> dict[int, FloatArray]:
        return {d: self.tilde_v[d] * self.gamma[d] * self.z[d][None, :] for d in self.tilde_v}


State = NormalState | HorseshoeState | SSPState


def initialize(data: Data, spec: ModelSpec, method: Method, rng, chain: int = 0) -> State:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    sd = 0.1 * (0.5, 1.0, 2.0, 4.0)[chain % 4]
    beta = rng.normal(0, sd, spec.p + 1)
    beta[0] += np.log((data.y.sum() + 0.5) / (len(data.y) - data.y.sum() + 0.5))
    V = {d: rng.normal(0, sd, (spec.p, r)) for d, r in spec.ranks.items()}
    common = (
        beta,
        spec.prior.b_beta / (spec.prior.a_beta + 1),
        np.ones(len(data.y)),
        np.zeros(len(data.y)),
    )
    sigma_v2 = {d: spec.prior.b_v / (spec.prior.a_v + 1) for d in spec.ranks}
    if method == "normal":
        state = NormalState(*common, V, sigma_v2)
    elif method == "horseshoe":
        state = HorseshoeState(
            *common,
            V,
            {d: np.ones_like(v) for d, v in V.items()},
            {d: np.ones(r) for d, r in spec.ranks.items()},
        )
    else:
        z = {
            d: (
                np.zeros(r, dtype=bool)
                if chain % 4 == 0
                else rng.random(r) < 0.5
                if chain % 4 == 1
                else np.ones(r, dtype=bool)
            )
            for d, r in spec.ranks.items()
        }
        gamma = {
            d: (
                np.ones((spec.p, r), dtype=bool)
                if chain % 4 == 3
                else rng.random((spec.p, r)) < 0.5
            )
            for d, r in spec.ranks.items()
        }
        state = SSPState(
            *common,
            z,
            gamma,
            V,
            {d: spec.prior.a_z / (spec.prior.a_z + spec.prior.b_z) for d in V},
            {d: spec.prior.a_gamma / (spec.prior.a_gamma + spec.prior.b_gamma) for d in V},
            sigma_v2,
        )
    state.eta = predictor(data.X, state.beta, state.effective())
    return state


def validate_state(state: State, data: Data, spec: ModelSpec) -> None:
    if state.beta.shape != (spec.p + 1,) or state.eta.shape != data.y.shape:
        raise ValueError("invalid beta/eta state shape")
    if state.omega.shape != data.y.shape:
        raise ValueError("invalid omega shape")
    arrays = [state.beta, state.eta, state.omega]
    positive = [state.sigma_beta2, *state.omega]
    effective = state.effective()
    if set(effective) != set(spec.ranks):
        raise ValueError("state orders differ from model specification")
    for d, r in spec.ranks.items():
        if effective[d].shape != (spec.p, r):
            raise ValueError(f"invalid loading shape at order {d}")
        arrays.append(effective[d])
        if isinstance(state, HorseshoeState):
            if state.lambda_[d].shape != (spec.p, r) or state.tau[d].shape != (r,):
                raise ValueError("invalid horseshoe scale shape")
            arrays.extend([state.lambda_[d], state.tau[d]])
            positive.extend(state.lambda_[d].ravel())
            positive.extend(state.tau[d])
        else:
            positive.append(state.sigma_v2[d])
        if isinstance(state, SSPState):
            if state.z[d].shape != (r,) or state.gamma[d].shape != (spec.p, r):
                raise ValueError("invalid SSP indicator shape")
            if not np.isin(state.z[d], [0, 1]).all() or not np.isin(state.gamma[d], [0, 1]).all():
                raise ValueError("SSP indicators must be binary")
            if state.tilde_v[d].shape != (spec.p, r):
                raise ValueError("invalid slab shape")
            arrays.append(state.tilde_v[d])
            if not (0 < state.pi_z[d] < 1 and 0 < state.pi_gamma[d] < 1):
                raise ValueError("inclusion probabilities must be strictly between 0 and 1")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("nonfinite state")
    if not np.isfinite(positive).all() or not (np.array(positive) > 0).all():
        raise ValueError("variances, scales and omega must be positive and finite")
