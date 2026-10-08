#!/usr/bin/env bash
#SBATCH --job-name=factorregression
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
# Array range and log paths are supplied to sbatch; see docs/unity.md.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ( $1 != pilot && $1 != long-run ) ]]; then
    echo "Usage: $0 pilot|long-run PREPARED_DIRECTORY RESULTS_DIRECTORY [check]" >&2
    exit 2
fi
if [[ $# == 4 && $4 != check ]]; then
    echo "The optional fourth argument must be check" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Specify --array}"
if [[ ${SLURM_CPUS_PER_TASK:-1} != 1 ]]; then
    echo "This sequential sampler uses one CPU per fit; request --cpus-per-task=1" >&2
    exit 2
fi

# sbatch runs a spool copy: --chdir must point to the repository, not the script's path.
[[ -x .venv/bin/python ]] || { echo "Run uv sync --locked in the repository first" >&2; exit 2; }
[[ -f "$2/manifest.json" ]] || { echo "Missing prepared manifest: $2/manifest.json" >&2; exit 2; }
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

args=(--stage "$1" --prepared "$2" --output "$3" --task-id "$SLURM_ARRAY_TASK_ID")
if [[ ${4:-} == check ]]; then
    args+=(--check-sweeps 20)
fi
exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
    .venv/bin/python -m factorregression.experiment run "${args[@]}"
