"""Small explicit CLI with strict configuration keys."""

import argparse
import json
from dataclasses import fields, replace

import numpy as np

from .mcmc import SamplerSettings, fit, json_safe
from .state import METHODS, ModelSpec, Prior


def load_configuration(path):
    with open(path) as stream:
        config = json.load(stream)
    if set(config) != {"model", "sampling"}:
        raise ValueError("configuration must contain exactly model and sampling")
    model = dict(config["model"])
    if set(model) - {"p", "D", "ranks", "prior"}:
        raise ValueError("unknown model configuration key")
    model["ranks"] = {int(d): rank for d, rank in model["ranks"].items()}
    model["prior"] = Prior(**model.get("prior", {}))
    sampling = dict(config["sampling"])
    if set(sampling) - {f.name for f in fields(SamplerSettings)}:
        raise ValueError("unknown sampling configuration key")
    for name in ("interaction_tuples", "prediction_patterns"):
        if sampling.get(name) is not None:
            sampling[name] = tuple(tuple(row) for row in sampling[name])
    return ModelSpec(**model), SamplerSettings(**sampling)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("fit", help="sample one observed binary dataset")
    run.add_argument("--data", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--output", required=True)
    for name in ("seed", "chains", "burn-in", "draws"):
        run.add_argument(f"--{name}", type=int)
    run.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        spec, settings = load_configuration(args.config)
        overrides = {
            name: getattr(args, name)
            for name in ("seed", "chains", "burn_in", "draws", "max_seconds")
            if getattr(args, name) is not None
        }
        settings = replace(settings, **overrides)
        with np.load(args.data, allow_pickle=False) as data:
            result = fit(
                data["X"],
                data["y"],
                method=args.method,
                spec=spec,
                settings=settings,
                output=args.output,
            )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            json_safe(
                {
                    "completion": result.metadata["status"],
                    "diagnostics": result.diagnostics["status"],
                    "elapsed_seconds": result.metadata["elapsed_seconds"],
                    "output": args.output,
                }
            ),
            indent=2,
        )
    )
    return 0 if result.metadata["status"] == "completed" else 1
