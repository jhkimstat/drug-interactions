# FactorRegression

CPU/float64 reference MCMC for Bayesian Bernoulli-logit factorization machines.
The current implementation covers Normal, Horseshoe and expanded-state SSP loading priors.
Inference reporting and Scalable SSP are deferred.

## Development

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run python scripts/smoke.py --output outputs/smoke
```

Python 3.13.15 is the development baseline, following BayesianCalibration's uv setup.
`uv.lock` records this project's own tested dependency resolution.
When changing dependencies, update and commit `pyproject.toml` and `uv.lock` together.
Install uv using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Sampling

Input NPZ files contain binary arrays `X` with shape `(n,p)` and `y` with shape `(n,)`.
The default configuration uses `p=5,D=3,R_2=R_3=5`; these are pilot values.

```sh
uv run factorregression fit --data data/observations.npz \
  --config configs/sampling-default.json --method normal --output outputs/normal
```

Methods are `normal`, `horseshoe` and `ssp_reference`. Explicit overrides are available for
the seed, chains, burn-in, retained draws and time limit; see `factorregression fit --help`.
Output contains observed inputs, resolved configuration, per-chain NPZ draws and JSON diagnostics.
Existing nonempty output directories are rejected to preserve earlier runs.

`configuration.json` can be passed back to `--config`, and `observations.npz` to `--data`,
with the method recorded in `metadata.json`. Interaction output tuples use zero-based
ascending NumPy indices. Empty interaction/prediction query lists disable those derived
outputs without changing fitting. Sampling completion and mixing diagnostics have separate
statuses; the short smoke run is not expected to meet the pilot ESS/MCSE thresholds.

General `p`, `1 <= D <= p` and a positive integer `R_d` per included order are supported.
No range of computational feasibility or convergence is guaranteed. True support is never
an input to fitting. Every loading update uses the latest predictor and loading values.
Pólya–Gamma sampling explicitly uses the Devroye method; IG uses shape and inverse-scale
parameters with density proportional to `x**(-a-1) * exp(-b/x)`.

## Repository

```text
src/factorregression/   model, conditionals, distributions, samplers, MCMC, diagnostics, CLI
tests/                 independent algebra, CDF/moment and posterior checks
configs/               editable pilot/smoke settings
scripts/smoke.py        short local sampling verification
docs/                  implementation plan and validation record
.github/workflows/     GitHub Actions checks
Archive/               existing local historical files, excluded from Git
```

Local environments, other data and generated results are excluded by `.gitignore`.
The prepared `data/pilot-long-run-20261008/` dataset is explicitly included for transfer
through GitHub to Unity. `Archive/` is preserved locally. The GitHub workflow installs
locked dependencies and runs lint and tests.
Set the remote URL to an existing GitHub repository when connecting this checkout; local
setup does not create or publish a remote repository.

The detailed specification and provisional defaults are in
[the implementation plan](docs/implementation-plan.md). Numerical and sampling validation
results are recorded in [the validation document](docs/validation.md).

## Unity pilot and long-run

The initial experiment uses signals `(1,1,1)`, `(1,2,2)`, `(1,3,3)`: 13 datasets at n=200,
39 pilot fits and 9 long-run fits on three shared representative datasets. Each fit runs
four sequential CPU chains. Prepared observations and truth are stored separately.

Commit the prepared `data/pilot-long-run-20261008/` files with the source and clone/pull
them on Unity. This directory already contains the datasets and frozen settings; the
server can use it directly. Use a new output directory when preparing another experiment.

See [the Unity run guide](docs/unity.md) for locked Linux setup, ASC `batch` submissions,
execution checks, and CSV accuracy/efficiency reports. `configs/pilot.json` and
`configs/long-run.json` are editable starting settings; preparation freezes copies for each
experiment. No remote jobs are submitted automatically.
