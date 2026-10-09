# Pilot and long-run on ASC Unity

Current experiment: p=5, n=200, D=3, R_2=R_3=5; Normal, Horseshoe and Reference SSP.
Signals are (1,1,1), (1,2,2), (1,3,3), as requested on 2026-10-08. Empty-support orders
have realized norm zero. There are 13 unique DGPs and one dataset per DGP.
External Obsidian notes remain unchanged; the updated signal values are encoded here.

| Stage | Data | Chains per fit | Burn-in / retained per chain | Fits / total sweeps |
| --- | --- | --- | --- | --- |
| Pilot | All 13 datasets | 4 | 2,000 / 2,000 | 39 / 624,000 |
| Long-run | no-interaction, sparse-3way-s3, dense-3way-s3 | 4 | 4,000 / 20,000 | 9 / 864,000 |

Both stages use the same prepared observations and model/prior/ranks, with distinct stage,
method and chain RNG streams. Each fit runs four chains sequentially on one CPU. Array
tasks parallelize independent fits. Thinning is 1. Repeated-dataset experiments, SBC,
rank/prior sensitivity and scientific inference reporting remain deferred.
These are provisional budgets; long runs must still pass diagnostics to qualify as references.

## 1. Install and prepare

Copy or clone this repository to storage visible from Unity compute nodes. Do not transfer
the macOS `.venv`; create a Linux environment from the committed `uv.lock`:

```bash
cd /your/path/FactorRegression
uv python install 3.13.15
uv sync --locked
```

`data/pilot-long-run-20261008/` is included in Git through an explicit ignore exception.
After committing and pushing these prepared files, cloning/pulling the repository on Unity
provides all 13 datasets, the manifest and frozen stage settings. Use this directory directly;
running `prepare` again at the same path will fail because it already exists.

Use the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
if needed. A compiler may be needed if polyagamma is built from source; use the compiler
module available on Unity (`module avail`) if installation reports this requirement.
Install once before starting the array; jobs use `.venv/bin/python` without downloading or
modifying dependencies. Python/library availability must be checked on Unity itself.

Preparation saves `manifest.json`, frozen `pilot.json`/`long-run.json`, and 13 dataset
directories. Each dataset has **observations.npz (X,y only)** and a separate **truth.npz**.
The manifest records task IDs and seeds. Fitting never opens truth.npz; evaluation does.
`--n`, `--seed`, `--pilot-config`, and `--long-run-config` can change preparation defaults.
Existing prepared directories are rejected. Configuration changes require a new preparation
directory; changing a source config after preparation does not affect the frozen experiment.

To prepare a new experiment, use a new directory:

```bash
uv run --locked python -m factorregression.experiment prepare \
  --output data/pilot-long-run-new
```

Additional data directories are ignored by default; add a specific ignore exception if
they also need to be transferred through GitHub. Set `prepared` below to the directory used
for that experiment. Do not regenerate a second dataset specifically for long-run.
Task IDs are dataset-major, method-minor in the order
normal, horseshoe, ssp_reference. The manifest is the definitive mapping.

## 2. Check on a compute node

ASC Unity uses Slurm; `batch` is its default shared partition. Run sampling on compute
nodes, as described in the [Unity walkthrough](https://osu.teamdynamix.com/TDClient/1929/ASC/KB/Article/61538/Unity-Walk-Through)
and [partition guide](https://osu.teamdynamix.com/TDClient/1929/ASC/KB/Article/130515/Unity-Partitions).
The commands below submit jobs when you run them. No remote jobs are submitted by preparation.

```bash
repo="$PWD"
prepared="$repo/data/pilot-long-run-20261008"
results="$repo/outputs/unity-20261008"
mkdir -p "$results/logs"

sbatch --chdir="$repo" --time=00:15:00 --array=0-2 \
  --output="$results/logs/check-%A_%a.out" --error="$results/logs/check-%A_%a.err" \
  scripts/unity_experiment.sh pilot "$prepared" "$results" check
```

The three checks run the no-interaction dataset through all methods with 20 burn-in and
20 retained sweeps per chain. Outputs go to `check-pilot/`, outside the production tables.
Check logs and saved completion status; mixing flags are expected with this short run.
All numerical failures use nonzero job exit status and retain available failure metadata.

## 3. Pilot and long-run

After the execution check, submit the pilot:

```bash
sbatch --chdir="$repo" --time=01:00:00 --array=0-38 \
  --output="$results/logs/pilot-%A_%a.out" --error="$results/logs/pilot-%A_%a.err" \
  scripts/unity_experiment.sh pilot "$prepared" "$results"
```

Inspect the pilot completion/diagnostic tables, then submit the fixed long-run subset:

```bash
sbatch --chdir="$repo" --time=04:00:00 --array=0-8 \
  --output="$results/logs/long-run-%A_%a.out" --error="$results/logs/long-run-%A_%a.err" \
  scripts/unity_experiment.sh long-run "$prepared" "$results"
```

Per user instruction (2026-10-08), submit arrays without a client-side concurrency cap:
do not append `%3`, `%30`, or another `%N` limit to `--array`. Let Slurm enforce available
resources and account/QOS limits. For an existing active array, use
`scontrol update JobId=JOB_ID ArrayTaskThrottle=0` to remove its cap.
This policy applies to pilot, long-run, and additional experiment arrays. Each job requests
1 CPU and 4 GiB memory; BLAS/OpenMP threads are fixed at one. The Python fit limits are
45 minutes for pilot and 3 hours for long-run, leaving time for diagnosis/output before
the Slurm walltime. These limits are per fit, not a promise of runtime or an aggregate
six-hour limit. No checkpoint/retry or automatic extension is used. A scheduler kill can
leave an unfinished output directory; partial draws are saved only if Python returns normally.

Use `squeue --me` and `sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,NodeList` to inspect
jobs. Failed/missing tasks stay in the reports. Re-runs use a new results directory, so
original results are preserved. For timing comparisons, use the same permitted node class
via Slurm `--constraint` where available, and inspect recorded host names before pooling
ESS/sec from different hardware. Account/constraint options can be supplied to `sbatch`.

## 4. Evaluation

After each stage, run the following in an interactive compute-node allocation (for example
`sinteractive -p batch -c 1 -M 4096 -t 01:00:00`), with paths set as above:

```bash
.venv/bin/python -m factorregression.experiment summarize \
  --prepared "$prepared" --results "$results" --output "$results/report-after-pilot"
# After long-run completes, use a new report directory:
.venv/bin/python -m factorregression.experiment summarize \
  --prepared "$prepared" --results "$results" --output "$results/report-after-long-run"
```

- `fits.csv`: all 48 expected fits, including missing, unfinished, budget-exhausted and
  numerical-failure cases; timing, diagnostic status, common-functional ESS/sec, coefficient
  RMSE, probability RMSE and expected log loss/Brier risk across all 32 equally weighted patterns.
- `functionals.csv`: beta (including intercept), 20 theta values and 32 prediction
  probabilities; truth, posterior mean/quantiles, R-hat, ESS, MCSE and ESS/sec. Undefined
  diagnostics remain blank and are counted. Fit-level ESS summaries use finite entries only.
- `pilot-vs-long-run.csv`: same-data/same-model mean and quantile differences, combined MCSE,
  and standardized mean differences. `reference_qualified` requires long-run `diagnostics_ok`
  and long-run mean MCSE <= pilot MCSE/3 for that functional. Pilot diagnostic status is
  reported separately; qualification is not a claim that the pilot or reference is exact.
  Standardized differences from flagged runs are descriptive, not calibrated accuracy tests.

ESS/sec uses `sampling_seconds`, including initialization, burn-in and draw collection,
excluding final diagnostics and file writing. End-to-end time including saving is also
recorded. Different priors imply different posterior targets; compare each pilot to its own
long-run. RMSE against one generated truth per DGP combines estimation, prior and possible
rank-representation effects and is not a sampler correctness test or a repeated-sample
coverage estimate. Existing independent conditional/quadrature tests remain the correctness
checks. No tolerance changes or automatic suppression of numerical failures are made.
