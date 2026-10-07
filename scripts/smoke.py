"""Three small note-based fixtures. Execution checks, not a scientific pilot."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import expit

from factorregression import fit
from factorregression.cli import load_configuration
from factorregression.mcmc import rng_stream
from factorregression.state import METHODS


def observed_fixture(index, n=64):
    truth_rng = rng_stream(20261006, purpose=10, dataset=index)
    data_rng = rng_stream(20261006, purpose=11, dataset=index)
    beta = truth_rng.uniform(0.5, 1.5, 5) * truth_rng.choice([-1, 1], 5)
    beta /= np.linalg.norm(beta)
    X = data_rng.binomial(1, 0.5, (n, 5)).astype(float)
    eta = truth_rng.uniform(-1, 1) + X @ beta
    support = {
        0: {2: [], 3: []},
        1: {2: [(0, 1), (0, 2), (1, 2)], 3: [(0, 1, 2)]},
        2: {
            2: list(combinations(range(5), 2))[:8],
            3: [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3)],
        },
    }[index]
    for alphas in support.values():
        if alphas:
            theta = truth_rng.uniform(0.5, 1.5, len(alphas)) * truth_rng.choice(
                [-1, 1], len(alphas)
            )
            theta *= 2.0 / np.linalg.norm(theta)
            for alpha, effect in zip(alphas, theta, strict=True):
                eta += effect * np.prod(X[:, alpha], axis=1)
    return X, data_rng.binomial(1, expit(eta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec, settings = load_configuration(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    for index, label in enumerate(("no-interaction", "sparse-3way", "dense-3way")):
        X, y = observed_fixture(index)
        for method in METHODS:
            result = fit(
                X,
                y,
                method=method,
                spec=spec,
                settings=settings,
                output=Path(args.output) / label / method,
            )
            print(
                json.dumps(
                    {
                        "fixture": label,
                        "method": method,
                        "status": result.metadata["status"],
                        "seconds": result.metadata["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
            if result.metadata["status"] != "completed":
                raise RuntimeError(result.metadata)


if __name__ == "__main__":
    main()
