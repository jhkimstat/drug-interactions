"""Prepare, run one fit, or summarize the fixed pilot/long-run experiment."""

import argparse
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .cli import load_configuration
from .mcmc import fit, json_safe, rng_stream
from .simulation import ALPHAS, CASES, LONG_RUN_CASES, PATTERNS, generate
from .state import METHODS


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def prepare(output, pilot_config, long_run_config, *, n=200, seed=20261008):
    """Freeze both configs and generate the common data once, before any fitting."""
    configurations = {}
    for stage, path in (("pilot", pilot_config), ("long-run", long_run_config)):
        spec, settings = load_configuration(path)
        if (spec.p, spec.D) != (5, 3):
            raise ValueError("this experiment requires p=5,D=3")
        settings = replace(
            settings, interaction_tuples=ALPHAS, prediction_patterns=tuple(map(tuple, PATTERNS))
        )
        configurations[stage] = {
            "model": {"p": 5, "D": 3, "ranks": dict(spec.ranks), "prior": asdict(spec.prior)},
            "sampling": asdict(settings),
        }
    if configurations["pilot"]["model"] != configurations["long-run"]["model"]:
        raise ValueError("pilot and long-run must use the same model/prior/ranks")
    # Validate generation arguments before creating any output.
    first = generate(0, n, seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"n": n, "seed": seed, "replication": 0, "datasets": [], "stages": {}}
    for index, (name, structure, signal) in enumerate(CASES):
        X, y, truth = first if index == 0 else generate(index, n, seed)
        directory = root / "datasets" / name
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "observations.npz", X=X, y=y)
        np.savez_compressed(directory / "truth.npz", **truth)
        manifest["datasets"].append(
            {
                "index": index,
                "name": name,
                "structure": structure,
                "signal_strengths": [1, signal, signal],
                "realized_norms": truth["realized_norms"],
            }
        )
    for stage_index, (stage, config) in enumerate(configurations.items()):
        write_json(root / f"{stage}.json", config)
        jobs = []
        for index, (name, _, _) in enumerate(CASES):
            if stage == "long-run" and name not in LONG_RUN_CASES:
                continue
            # fit() further separates methods/chains. Stage streams are disjoint.
            fit_seed = int(
                rng_stream(
                    config["sampling"]["seed"], purpose=20 + stage_index, dataset=index
                ).integers(0, 2**63)
            )
            for method in METHODS:
                jobs.append(
                    {"task_id": len(jobs), "case": name, "method": method, "seed": fit_seed}
                )
        manifest["stages"][stage] = {"config": f"{stage}.json", "jobs": jobs}
    write_json(root / "manifest.json", manifest)
    return manifest


def run_task(prepared, output, stage, task_id, check_sweeps=None):
    """Run four sequential chains using observations/config only; never read truth."""
    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared / "manifest.json").read_text())
    stage_spec = manifest["stages"][stage]
    jobs = stage_spec["jobs"]
    if not 0 <= task_id < len(jobs):
        raise ValueError(f"task_id must be between 0 and {len(jobs) - 1}")
    job = jobs[task_id]
    spec, settings = load_configuration(prepared / stage_spec["config"])
    settings = replace(settings, seed=job["seed"])
    if check_sweeps is not None:
        settings = replace(settings, burn_in=check_sweeps, draws=check_sweeps)
    directory = output / (f"check-{stage}" if check_sweeps is not None else stage)
    directory = directory / job["case"] / job["method"]
    # Reserve exclusively before sampling; duplicate tasks cannot overwrite an earlier fit.
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(prepared / "datasets" / job["case"] / "observations.npz") as observed:
        start = time.monotonic()
        result = fit(
            observed["X"], observed["y"], method=job["method"], spec=spec, settings=settings
        )
    result.save(directory)
    write_json(
        directory / "execution.json",
        {
            "stage": stage,
            "task_id": task_id,
            "check_sweeps": check_sweeps,
            "host": platform.node(),
            "machine": platform.machine(),
            "wall_seconds_including_save": time.monotonic() - start,
        },
    )
    print(
        json.dumps(
            {
                "stage": stage,
                **job,
                "status": result.metadata["status"],
                "diagnostics": result.diagnostics["status"],
                "output": str(directory),
            }
        ),
        flush=True,
    )
    if result.metadata["status"] != "completed":
        for chain in result.chains:
            if chain.metadata["failure"]:
                print(json.dumps(chain.metadata["failure"]), flush=True)
    return result.metadata["status"] == "completed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("prepare")
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--n", type=int, default=200)
    generate_parser.add_argument("--seed", type=int, default=20261008)
    generate_parser.add_argument("--pilot-config", default="configs/pilot.json")
    generate_parser.add_argument("--long-run-config", default="configs/long-run.json")
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--stage", choices=("pilot", "long-run"), required=True)
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument(
        "--check-sweeps", type=int, help="short execution check in a separate output tree"
    )
    summary = sub.add_parser("summarize")
    summary.add_argument("--prepared", required=True)
    summary.add_argument("--results", required=True)
    summary.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            manifest = prepare(**args)
            print(
                json.dumps(
                    {stage: len(value["jobs"]) for stage, value in manifest["stages"].items()}
                )
            )
        elif command == "run":
            return 0 if run_task(**args) else 1
        else:
            from .evaluation import summarize

            summarize(**args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
