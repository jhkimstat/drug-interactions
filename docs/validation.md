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

The predictor uses elementary-symmetric-polynomial dynamic programming. A loading's
excluded polynomial is recomputed from the current remaining loadings rather than obtained
through subtractive polynomial division. This is deliberately a transparent reference
implementation; a loading sweep can cost O(n p² Σ_d d R_d), and the beta precision is dense.
The implementation supports general dimensions without promising scalability.

Numerical failures stop the affected fit and preserve completed retained draws with failure
context. Constant or incomplete chain diagnostics are marked undefined/incomplete.
Identifiable theta/prediction diagnostics and label-specific loading/indicator diagnostics
are recorded separately. Passing the tests does not prove global posterior exploration.

The editable pilot defaults and acceptance thresholds are in the implementation plan and
`configs/`. Their numeric choices are not final scientific settings.
