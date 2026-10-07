# Sampling implementation validation

Date: 2026-10-06. Current scope: common sampling, Normal, Horseshoe and Reference SSP.
Inference, Scalable SSP and the full 13-DGP pilot remain deferred.

## Tested environment

Python 3.13.15, uv 0.12.16, NumPy 2.5.3, SciPy 1.18.1, polyagamma 2.0.2,
ArviZ 0.22.0, pytest 9.1.1 and Ruff 0.16.10, on macOS ARM64.
`uv.lock` is generated for FactorRegression. BayesianCalibration supplied the uv/build
workflow reference; its lockfile and JAX/BlackJAX dependencies were not copied.
PG sampling explicitly selects Devroye. polyagamma built successfully from source locally.

## Results

- Full suite: **68 tests passed** in 6.54 seconds in the recorded local run.
- Ruff lint and format checks passed.
- `uv sync --locked --offline` succeeded with the tested resolution.
- `uv build --offline` produced both an sdist and a wheel from the sdist.
- The locked CLI entry point and configuration/observation replay passed.
- Nine smoke fits completed: 3 fixtures × 3 methods × 4 chains × 200 sweeps,
  totaling **7,200 complete sweeps**. No numerical failures were reported.
- Every smoke fit retained `mixing_flagged`. Their 100 post-burn-in draws per chain
  are execution checks, not evidence of adequate mixing or scientific accuracy.
- `.venv/`, `Archive/`, `outputs/` and `.DS_Store` are confirmed ignored by Git.
  `uv.lock` is present and not ignored, ready to commit with the source files.

Local smoke output is under `outputs/smoke-validated/` and excluded from Git. Each fit's
configuration, observed inputs, chain draws, completion status and diagnostic flags are saved.
No full pilot or main experiment was run. The GitHub Actions workflow is present but has
not been exercised on a remote repository.

## Independent checks

1. General dimensions: p=5,D=3 is the primary fixture; D=1, D=p and p=7,D=4
   also check generic loops and unequal order-specific ranks. Predictors and theta are
   compared with explicit sums over distinct unordered interaction tuples. Perturbing
   every loading independently confirms the conditional linear representation.
2. Gaussian/indicator conditionals: conditional-density differences match independent
   augmented joint evaluations. Precision-form Gaussian draws use `L^{-T} z`;
   50,000 draws verify means and the full covariance, including off-diagonal entries.
3. PG: 50,000 IID draws per tilt in {0,±1,±5,±20,±100}; analytic first/second moments
   and CDF probabilities from independent Jacobi-series density integration are compared
   using Monte Carlo uncertainty. This reference does not call polyagamma's PDF/CDF.
4. Truncated kernels: rate-zero uniform/power cases, tiny positive rates, ordinary bounds
   and underflowed Gamma truncation masses are checked with 50,000 draws per case.
   The Gamma underflow fallback is exact rejection from a bounded power or tangent-envelope
   proposal; no scale/CDF floor or substituted prior is used.
5. Horseshoe: the transformed local/component log densities match normal×half-Cauchy
   priors with their Jacobians. Local and component zero-rate boundaries and complete
   sweeps are checked. The newly drawn local scales feed component updates.
6. SSP: an independent explicit joint-mass scan matches z and gamma updates. Inactive
   gamma depends only on pi_gamma; inactive slabs are Gaussian prior draws without eta
   changes. Hyperparameter checks include every gamma and every slab, even when z=0.
7. Full sweeps: all three methods keep eta consistent with an independently reconstructed
   predictor across sequential updates. No hierarchy constraint is imposed.
8. Posterior references: the main-only intercept posterior is compared with Student-t
   prior/logistic-likelihood quadrature after analytically integrating sigma_beta2 and
   unused main effects. A fixed-beta/fixed-variance Normal loading pair is compared with
   independent two-dimensional Gaussian tensor quadrature. Both comparisons check
   R-hat<1.01, bulk/tail ESS≥400 and errors within 5 MCSE plus reference error. The second
   is explicitly a conditional-kernel check, not the full hierarchical interaction posterior.
9. Runtime contracts: deterministic replay, independent RNG streams, no mutation of
   observed input, thin/burn-in indexing, NPZ/JSON roundtrip, output overwrite rejection,
   draw-memory bounds, time bounds and numerical-failure status are checked. SSP boolean
   diagnostics have a regression test after the initial smoke exposed NumPy's unsupported
   boolean `ptp` subtraction.

## Numerical implementation notes

The predictor uses elementary-symmetric-polynomial dynamic programming. Following the
2026-10-07 numerical-cost review, each component builds coefficients once per coordinate
pass. Exclusion uses q_t=e_t−a_j q_{t−1}; after a draw, coefficients receive Δa_j q_{t−1}.
Ordinary cancellation is accepted. Polynomial/loading-pass cost is O(n p Σ_d d R_d),
with O(nd) temporary coefficient storage per component. The beta precision is still dense,
and no general range of computational feasibility or convergence is promised.

Numerical failures stop the affected fit and preserve completed retained draws with failure
context. Constant or incomplete chain diagnostics are marked undefined/incomplete.
Identifiable theta/prediction diagnostics and label-specific loading/indicator diagnostics
are recorded separately. Passing the tests does not prove global posterior exploration.

The editable pilot defaults and acceptance thresholds are in the implementation plan and
`configs/`. Their numeric choices are not final scientific settings.

## Numerical-cost review — 2026-10-07

The user explicitly removed machine-level agreement as a project objective. The refactor
keeps the model and Gibbs conditionals and uses ordinary float64 recurrences with errors
judged against inference scale and Monte Carlo uncertainty.

Changes:

- Build the full-variable coefficients e_0,...,e_{d-1} once per component/pass. Compute
  exclusions by q_t=e_t−a_j q_{t−1}, and update e_t by Δa_j q_{t−1} after every draw.
  Normal, Horseshoe and the active SSP indicator/slab passes all use this table.
- Batch the independent prior draws for inactive SSP components. Their gamma and slabs
  are still retained, updated and included in all hyperparameter counts.
- Reuse maintained observed-data eta and its likelihood in retained log posterior values.
  Only requested prediction patterns need another predictor calculation. Disabled derived
  outputs do not construct effective loadings or evaluate an empty predictor.
- Check full state at initialization, periodic cache boundaries and the final sweep;
  per-draw support/nonfinite checks and NumPy overflow/invalid detection remain. Avoid
  constructing effective SSP loadings or Python lists of every scalar during validation.
- Use Cholesky directly, avoiding the separate triangular copy, and skip repeated SciPy
  finite scans of already validated inputs. Positive-definiteness errors are still reported.
- Remove the separate tiny-positive-rate exponential path, using log1p/expm1 directly.
  Accept truncation-upper-bound values that round to the bound; values beyond the bound,
  zero/nonfinite draws and invalid parameters remain errors. The Gamma underflow rejection
  fallback is retained because replacing/removing it can materially invalidate sampling.

Algebra and predictor/cache comparisons now use atol=1e−8, rtol=1e−6. This is a declared
inference-scale numerical policy rather than an attempt to retain every float64 digit.
For comparison, a logit perturbation of 1e−6 changes a probability by at most 2.5e−7,
well below the pilot probability-MCSE target of 0.01. This is not a uniform error bound
for every degree or extreme loading configuration; material cache drift still fails.

Posterior quadrature uses epsabs=1e−7, epsrel=1e−6 and requires estimated numerical
error/refinement ≤0.1 MCSE. The Normal interaction reference uses 80/160 nodes per axis
instead of 320/640, giving 16 times fewer tensor grid cells. The two independent posterior
comparisons still pass their unchanged R-hat, ESS and 5-MCSE requirements. Exact checks
for unchanged inactive predictors, integer counts and deterministic replay are retained
because they test logical invariants/reproducibility rather than a numerical-accuracy goal.

Validation after the final refactor: **80 tests passed in 6.14 seconds**; lint/format and
diff checks passed. New tests cover cached coefficient updates across multiple passes and
degrees, one polynomial build per component rather than per coordinate, skipped observed
predictor reconstruction, harmless versus material drift, and rounded truncation endpoints.
A 100-variable fixture verifies conditional-mean differences below 1e−6 posterior conditional
SD and relative conditional-variance differences below 1e−6 against direct exclusion.
Nine smoke fits (7,200 sweeps) completed without numerical failures; all short-run mixing
statuses remain `mixing_flagged`. Results are in `outputs/smoke-recurrence-20261007/`.

### Local before/after measurements

Same Python/library environment as above, macOS ARM64. BLAS thread environment was limited
to one. Measurements compare the pre-refactor source with the new source, using the same
observed data, seed and initial-state type. Times are illustrative local medians, not a
performance or mixing guarantee. Sampler timings depend on active state and do not measure
ESS/second or imply identical floating-point trajectories.

| Operation | Before | After | Ratio |
| --- | --- | --- | --- |
| Slope pass: n=500,p=50,d=3 | 6.20 ms | 0.599 ms | 10.4× |
| Slope pass: n=500,p=100,d=3 | 24.54 ms | 1.191 ms | 20.6× |
| Normal sweep: n=500,p=80,D=3,R_2=R_3=1 | 26.98 ms | 2.878 ms | 9.4× |
| Horseshoe sweep: same dimensions | 27.86 ms | 4.218 ms | 6.6× |
| Reference SSP sweep: same dimensions | 27.89 ms | 0.690 ms | 40.4× |
| Normal fit: n=200,p=5,D=3,R_2=R_3=5 | 0.926 s | 0.707 s | 1.31× |
| Horseshoe fit: same dimensions | 1.222 s | 1.009 s | 1.21× |
| Reference SSP fit: same dimensions | 1.338 s | 0.755 s | 1.77× |

Slope-pass medians use seven repetitions with small sequential coordinate changes. Sweep
medians use twenty timed sweeps after five warm-up sweeps. Fit medians use three repetitions,
one chain, burn-in 100 and retained 500, including draw collection and insufficient-chain
diagnostic status but no file writing. At p=5 the isolated Normal/Horseshoe sweep times were
roughly unchanged (Horseshoe was about 7% slower in this short measurement); observed-data
snapshot and validation savings improve the measured full fit. The large SSP speedup also
includes avoiding inactive-component polynomial work. General performance remains limited
by the dense beta block and the actual posterior state.
