# Sampling implementation validation

Initial validation: 2026-10-06; follow-up reviews: 2026-10-07.
Current scope: common sampling, Normal, Horseshoe and Reference SSP.
Inference and Scalable SSP remain deferred. Unity pilot/long-run preparation was added on
2026-10-08; production server runs have not yet been submitted by this session.

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
4. Truncated kernels: rate-zero uniform/power cases, tiny positive rates and ordinary bounds
   are checked against independent CDFs/integrals with 50,000 draws per case. Truncated
   Gamma retains the original inverse-CDF path without rejection fallback. Underflowed
   masses that yield zero draws raise the existing positive/finite support error.
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

The predictor uses elementary-symmetric-polynomial dynamic programming. Loading, active
SSP gamma and slab slopes now use Sequential Prefix/Suffix DP, adopted on 2026-10-08.
Each component builds suffix coefficients once; a rolling prefix includes each newly
processed effective loading. The degree d-1 prefix/suffix sum avoids excluding a large
loading by subtraction. Loading-pass cost remains O(n p sum_d d R_d); per-component
coefficient storage is O(npd). The earlier exclusion recurrence/delta table update is
removed. Ordinary floating-point rounding is accepted and cache tolerance is unchanged.
The beta precision is still dense; no general feasibility/convergence range is promised.

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
  fallback was retained at this review. The latest decision below removes only that block
  while keeping the original inverse-CDF calculation.

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

## Library-function review — 2026-10-07

The custom inverse-Gamma sampler was removed. Common beta variance, Normal loading variance
and Reference SSP slab variance now call `scipy.stats.invgamma.rvs(a, scale=b,
random_state=rng)` directly, with existing positive/finite output checks. The project's
IG convention maps to SciPy `scale=b`, not `1/b`. Application tests verify shape, scale,
RNG propagation, the intercept count, and the full loading/inactive-slab counts; the redundant
test of the removed standalone sampler was replaced by those call-site checks.
[SciPy's parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html)

Further replacements:

- Positive-rate truncated exponentials delegate to `scipy.stats.truncexpon.rvs`, with
  standardized upper bound `rate*upper` and `scale=1/rate`; zero scaled rates use NumPy
  uniform draws. The adapter handles scalar/broadcast arrays, unit conversion and support
  checks. Horseshoe local scales are conditionally independent given current V,tau and
  share one library dispatch per order. This changes no loading scan or PG update order.
  [SciPy truncated exponential](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncexpon.html)
- At this review, zero-rate Gamma and bounded-power rejection proposals used
  `Generator.power(shape)`. The latest change below removes the rejection proposals.
- Bernoulli indicator prior densities use `scipy.stats.bernoulli.logpmf`.
- Logit-scale likelihood uses `scipy.special.log_expit` on signed logits. It stays finite
  for an unexpected outcome at logits ±1000, where first converting to an expit probability
  and calling a probability-scale PMF can produce negative infinity.
- Smoothed intercept initialization uses `scipy.special.logit`; the initialization formula
  and all scientific priors are unchanged.

### Justification for remaining custom numerical code and wrappers

The Gamma row reflects the latest decision below: retain the inverse-CDF wrapper and
remove only its rejection fallback.

| Code | Library alternative checked | Reason to retain |
| --- | --- | --- |
| Truncated Gamma wrapper | SciPy `gammainc`/`gammaincinv` | Retains the user-specified shape-rate/bound/RNG interface and rate-zero power conditional. Positive-rate sampling uses the existing SciPy inverse-CDF composition; no custom rejection algorithm remains. Existing support checks report invalid draws |
| Partial symmetric DP and sequential prefix/suffix slopes | Per-row `np.polynomial.polynomial.polyfromroots` | NumPy constructs full-degree polynomials, with no equivalent batched partial-degree/sequential prefix-suffix interface. The earlier partial-table measurement below motivates retaining DP; prefix/suffix adoption additionally addresses observed large-loading cancellation |
| Precision Gaussian composition | NumPy MVN or SciPy `Covariance.from_precision` | NumPy needs covariance; SciPy's precision factory factors Q again. The helper uses NumPy normal draws and SciPy triangular solves to reuse the existing precision Cholesky factor |
| PG adapter | polyagamma sampling | Already delegates sampling; the adapter records the model's shape=1, explicit RNG and Devroye choice, and checks valid output |
| Open uniform/support guards | NumPy Generator | Slice auxiliaries require strictly positive uniforms; NumPy's API is half-open. Helpers only enforce endpoint/support/error policy and contain no alternative PRNG |

Model-specific conditional parameters, interaction reconstruction and scan/state logic have
no equivalent generic distribution operation. Diagnostics already delegate R-hat/ESS/MCSE to
ArviZ; CLI/files use argparse, json and NumPy NPZ. NumPy JSON conversion and configuration
validation remain small application policies. Independent Jacobi/density/reference formulas
in tests are intentionally separate numerical oracles, not competing production algorithms.

### Checks and measurements

The library replacements change random-number consumption, so old and new seeded sample
paths are not expected to be identical. Deterministic replay of the new implementation and
independent posterior/CDF checks remain required. New checks include mixed zero/positive-rate
array truncation, open-uniform array endpoints, extreme-logit likelihood and direct IG
shape/scale/RNG dispatch. No numerical tolerance or scientific setting was changed in this review.

The partial-polynomial comparison used n=200,p=50,degree=3: 0.138 ms for the custom batched
partial table versus 29.571 ms for NumPy full polynomial construction per row (~214×).
Minimum of three timings, each with five repetitions, in the existing local environment.

An illustrative local fit comparison used n=200,p=5,D=3,R_2=R_3=5; one chain, burn-in 100,
retained 500, median of three repeats, one BLAS thread, and no file writing. Before→after:
Normal 0.635→0.668 s; Horseshoe 0.973→0.812 s; Reference SSP 0.725→0.793 s. Generic library
dispatch adds a modest cost to Normal/SSP; batched local-scale dispatch offsets that cost in
Horseshoe. These are implementation timings, not ESS/second or a general speed guarantee.

Final validation: **84 tests passed in 10.21 seconds**, including both independent posterior
comparisons. Ruff lint/format and diff checks passed. Nine smoke fits (7,200 sweeps) completed
without numerical failures; their short-run mixing status remains `mixing_flagged`. Saved
results are under `outputs/smoke-library-review-20261007/`. No full pilot was run.

## Truncated Gamma fallback restored — 2026-10-07

Historical step, superseded by the final scope clarification below.

The user requested restoration of the previous fallback implementation. The temporary
SciPy generic-truncation implementation and fail-on-mass-underflow policy are reverted.
`truncated_gamma(shape, rate, upper, rng)` again uses `gammainc`/`gammaincinv` for its ordinary
inverse-CDF path and the original exact rejection fallback for invalid/underflowed results.
Rate zero and bounded-power proposals retain `Generator.power(shape)` from the library
review. No rate, scale or CDF floor is introduced.

The original five independent quadrature-CDF fixtures, including shape=1000,upper=1 with
rate=1 or 0.01, are restored with 50,000 draws per case and the existing 5-MCSE criterion.
The rounded-upper-endpoint check is restored as well. Tests for the temporary SciPy
truncation dispatch and immediate underflow failure are removed. The temporary CLI stderr
change is reverted; existing MCMC failure metadata and nonzero CLI exit status remain.
The SciPy requirement is back to >=1.14 and uv.lock metadata is updated; the installed/tested
SciPy remains 1.18.1. Other library replacements from the prior review remain unchanged.

The temporary implementation had passed 86 tests and nine smoke fits, as recorded in the
previous review; those results describe the superseded implementation. Current restoration
results are recorded below.

Final restoration validation: **84 tests passed in 11.13 seconds**, including independent
underflow-fallback CDF and posterior checks. Ruff lint/format and diff checks passed.
Nine smoke fits completed **7,200 sweeps** without numerical failures; all retained
`mixing_flagged`. Outputs are under `outputs/smoke-gamma-fallback-restored-20261007/`.
No full pilot or main experiment was run.

## Truncated Gamma: remove only the fallback block — 2026-10-07

The user supplied the intended function and clarified that only rejection fallback should
be removed. The wrapper keeps its original `gammainc`/`gammaincinv` calculation and rate-zero
`upper*rng.power(shape)` draw. The retry/rejection block and `q>0` shortcut are removed.
Existing positive/finite and upper-bound checks report invalid draws. Dependencies, CLI,
truncated exponential and other samplers are unchanged by this edit.

The three independent Gamma CDF fixtures (rate=0,2,1e-12) retain 50,000 IID draws per case
and the same 5-MCSE criterion. The two former underflow-recovery fixtures now check the
existing `truncated Gamma draw must be positive and finite` error at shape=1000,upper=1
and rate=1 or 0.01. No numerical tolerance or scientific setting was changed.

Final validation: **84 tests passed in 10.17 seconds**, including independent CDF and
posterior checks. Ruff lint/format and diff checks passed. Nine smoke fits completed
**7,200 sweeps** without numerical failures; all retained `mixing_flagged`. Outputs are
under `outputs/smoke-gamma-inverse-cdf-only-20261007/`. No full pilot was run.

## Unity pilot/long-run preparation — 2026-10-08

The updated DGP signals are (1,1,1), (1,2,2), (1,3,3). The user confirmed ASC Unity's
`batch` partition. Production preparation generated **13 datasets, n=200 each** under
`data/pilot-long-run-20261008/`, with observations separate from truth, frozen stage configs,
and 39 pilot / 9 long-run task records. These files are Git-ignored and can be copied to
Unity or regenerated there from the preparation command. No remote connection or submission
was performed, and the full 2,000/2,000 pilot and 4,000/20,000 long-run budgets were not run.

Implemented the note supports and normalized truth generation, per-fit sequential-chain
runner, a small Slurm array entry script, and simulation-only CSV evaluation. The comparison
uses the same observed dataset/model/prior/ranks across stages, independent RNG streams,
combined MCSE and explicit long-run reference qualification. Failed, missing, unfinished
and mixing-flagged fits remain visible. No model priors, Gibbs kernels, numerical tolerances,
dependencies or locked package versions were changed for this preparation.

Validation on the existing macOS ARM64 environment:

- **103 tests passed in 10.56 seconds**, including all 13 DGP support/norm checks, independent
  probability reconstruction, deterministic data replay, stage seed separation, fitting with
  truth files removed, refusal to overwrite output, saved numerical-failure rows, long-run
  comparison qualification/combined MCSE, and existing posterior/quadrature tests.
- `uv run --locked --offline ruff check .`, Ruff format, `bash -n scripts/unity_experiment.sh`
  and `git diff --check` passed.
- Actual CLI preparation/run/summarize checks used n=32, 4 chains, burn-in=5, retained=20.
  Both stages ran all three methods on no-interaction, sparse-3way-s3 and dense-3way-s3:
  **18 completed fits / 1,800 sweeps**, with no numerical failures. Every short fit retained
  `mixing_flagged`; none was accepted as a qualified reference.
- Reports contain all 48 expected fit rows (18 completed, 30 missing), 1,044 functional rows
  and 522 same-target comparison rows. Results are in `outputs/unity-workflow-check-20261008/`.
- The separate `--check-sweeps` path also completed a 4-chain, 2+2-sweep check on the prepared
  n=200 no-interaction dataset and wrote to `check-pilot/`, leaving production paths empty.
- Generated observations/results are confirmed Git-ignored. Unity Linux dependency setup,
  actual Slurm submission, node memory use and full-budget convergence remain to be checked
  on the server following `docs/unity.md`.

The new truth-based risks are single-dataset simulation evaluations, not repeated-sample
coverage or proof of sampling correctness. Long-run results must satisfy their diagnostic
and reference-MCSE criteria before being used to assess a pilot's Monte Carlo stability.

## GitHub-to-Unity prepared-data transfer — 2026-10-08

At the user's request, `.gitignore` now includes a narrow exception for
`data/pilot-long-run-20261008/`. Its 29 files (13 observations, 13 truth fixtures, manifest
and two frozen configs; 52,032 bytes total) are eligible for Git transfer; all 29 were
already tracked at the final visibility check. Other data,
outputs, builds, environments and local credentials remain ignored. Git visibility checks
verified all 29 files and the unrelated exclusions. No prepared arrays/settings were changed.
Unity/README instructions now use the cloned prepared directory directly and reserve the
prepare command for a new output path. The earlier preparation-run observation that all
local data were ignored is superseded by this exception. The agent did not issue commit or push commands
during this change. **103 tests passed in 10.69 seconds**; Ruff and diff checks passed.

### 2026-10-08 — Array 동시 실행 상한 문서화

- docs/unity.md에 사용자 지정 동시 실행 상한을 두지 않는 정책을 기록하고 제출 예시의 `%3`을 제거했다. 샘플러 코드와 실험 설정은 변경하지 않았다.
- 검증: `uv run pytest` — 103 passed (22.97s); `uv run ruff check .` — All checks passed; `git diff --check` 통과. 문서 변경이므로 추가 sampling CLI smoke는 실행하지 않았다.

## Unity saved-result analysis — 2026-10-08

Analyzed the existing `outputs/unity-20261008/` results using the expanded prepared
manifest `data/pilot-long-run-all-settings-20261008/`: 39 pilot and 39 long-run fits.
Its observations/truth and frozen stage configs match the original preparation byte for
byte; original job mappings/seeds are unchanged. The actual summarize CLI verified saved
observations/seeds and reported **78 expected / 62 completed fits**, producing 3,596
functional rows and 1,350 comparison rows (1,334 functionals and 16 incomplete pairs).
Outputs are preserved under `outputs/unity-20261008/report-analysis-20261008/`.

- Pilot: 38 completed (11 diagnostics_ok, 27 mixing_flagged), 1 numerical failure.
- Long-run: 24 completed (23 diagnostics_ok, 1 mixing_flagged), 6 numerical failures,
  9 missing. Slurm accounting confirms the original nine-task array was cancelled before
  execution. The failed pilot prevents its recorded afterok dependency from being satisfied;
  the accounting record does not identify the cancellation cause/actor.
- All seven numerical failures are Horseshoe eta cache mismatches, max absolute errors
  6.10e-8 through 8.81e-6. They remain failures; no tolerance or algorithm was changed.
- The completed Horseshoe sparse-3way-s2 long run still fails theta[10] and prediction
  diagnostics. The corresponding interaction is present in 21 observed rows, all y=1;
  its sampled coefficient has a long right tail and differing chain means.
- Independent direct-combination reconstruction checked 744 retained snapshots across all
  248 completed chains. Maximum theta difference: 0; probability difference: 5.00e-16.
  Recomputing probability RMSE from all saved probability draws for all 62 completed fits
  differed by at most 1.96e-15. Existing atol=1e-8, rtol=1e-6 remained unchanged. This
  audits saved output consistency, not full trajectory correctness or failed chains.
- `uv run pytest`: **103 passed in 21.57s** on Linux/Python 3.13.15.
  `uv run ruff check .`: passed. Actual summarize CLI and analysis/plot script completed.
  No new sampling, submissions, cancellation, prior changes, or new inference scope.

See [the analysis report](unity-20261008-analysis.md) for method comparisons, reference
qualification, figures, failure details, and interpretation limits. Single generated datasets
and incomplete Horseshoe coverage do not support a general method ranking.

## Eta cache warning change reverted — 2026-10-08

At the user's request, reverted the immediately preceding warn-and-continue change
before investigating the mismatch cause. Restored the original sampler and tests:
`atol=1e-8`, `rtol=1e-6` mismatches cause numerical_failure; successful checks retain
the original `state.eta = eta` reset. The warning ceiling, logging, warning metadata and
new warning tests are removed. The plan again describes the original behavior.
Prior Unity analysis, raw results, and unrelated working changes remain intact.

Validation: `uv run pytest` — **103 passed in 20.69s**;
`uv run ruff check .` and `git diff --check` passed. Actual Horseshoe CLI smoke on
prepared sparse-3way-s2 data (one chain, 5 burn-in + 20 retained sweeps) completed;
diagnostics remain insufficient_draws. Output: `outputs/smoke-cache-warning-revert-20261008/`.
No production experiment was rerun, and the numerical failure cause remains unresolved.

## Horseshoe eta-cache investigation — 2026-10-08

The user supplied downloaded Unity results and explicitly requested comparison without
changing the sampler. Found one pilot and six long-run Horseshoe cache failures. Replayed
five available pre-failure windows from saved beta/V draws, using identical coordinate
values across cached recurrence, freshly rebuilt full-polynomial recurrence, and direct
combinatorial slopes. Both recurrence variants violated existing cache tolerance in all
five windows; direct combinations had no violations, with maximum eta error 3.02e-14.
Cached-recurrence residuals at the last retained sweep reproduce the logged failure errors
to about 1e-15. DP/combinatorial predictor reconstruction differs by <=6.7e-16 at those ends.

Also compared 32 retained-state fixtures and a labelled large-loading stress fixture.
Direct combinations removed the observed fixed-update discrepancies. Analysis-only seeded
reruns completed 4,000 and 24,000 sweeps with direct slopes; the local current-recurrence
long-run failed at sweep 300 (Unity failed at 1,800). The local pilot did not reproduce its
Unity failure. Two Unity failures occurred before retention and cannot be replayed from
stored draws. Cross-platform stochastic trajectories and full posterior convergence are
not claimed identical or validated by these comparisons.

Production src/config/test files, priors and cache tolerances remain unchanged. No downloaded
files were modified. **103 tests passed in 9.49 seconds**, Ruff passed, and source diff checks
confirm this is analysis/documentation only. Details and numerical tables are in
[the investigation report](eta-cache-investigation.md); reproducible analysis artifacts are
under `outputs/eta-cache-investigation-20261008/`. No sampler fix was implemented.

## Sequential Prefix/Suffix DP feasibility — 2026-10-08

Analysis-only prototype: suffix coefficients are built once per component, rolling prefix
coefficients use newly sampled coordinates, and each excluded slope is a single-degree
prefix/suffix convolution. All five recorded pre-failure windows pass unchanged cache
tolerance; maximum eta error is 3.109e-14. Generic sequential fixtures (5,2),(5,3),(5,5),
(7,4), large-loading stress through M=1e8, and a 24,000-sweep diagnostic chain also pass.

Complete loading blocks at n=200,p=5,R_2=R_3=5 take about 0.73 ms with current recurrence
and 0.72 ms with prefix/suffix, versus 1.25 ms with direct combinations, in the local
controlled timing. Isolated large-p passes retain O(npd) time but have a measured 65–67%
constant overhead; coefficient storage grows from O(nd) to O(npd). Measurement details
and limitations are in [the investigation report](eta-cache-investigation.md).

No production sampler/model/config/test changes, tolerance relaxation, clipping or prior
changes were made. **103 tests passed in 9.65 seconds**, Ruff passed. Analysis artifacts
remain in the ignored investigation output directory. This establishes feasibility for the
observed cache failures, not posterior convergence or a blanket accuracy guarantee.

## Sequential Prefix/Suffix DP production adoption — 2026-10-08

The user chose this as the final algorithm. Normal/Horseshoe loading scans and Reference
SSP active gamma/slab scans now build suffix once per component and advance prefix from
each processed effective coordinate. Common eta updates remain immediate. Removed the
production exclusion subtraction recurrence and delta polynomial updater. No prior,
conditioning/scan order, cache tolerance, dependency or prepared dataset changes were made.

Final checks: **112 tests passed in 10.25 seconds**. Independent combinations cover generic
orders and sequential prefix updates; large-loading regression fixtures preserve structural
zero slopes and eta; SSP tests explicitly cover births, deaths, unchanged gates and inactive
slab draws. Existing conditional-density, posterior-quadrature and deterministic replay tests
pass. `uv run ruff check .`, Ruff format checks and diff checks pass.

Replaying the five observed failure windows through production Horseshoe loading updates
with recorded draw values gives no violations and maximum eta error 3.109e-14. A production
24,000-sweep Horseshoe chain completes with maximum checked error 3.553e-14.
Nine smoke fits complete **7,200 sweeps** without numerical failures; all remain
`mixing_flagged`. Outputs are under `outputs/smoke-prefix-suffix-production-20261008/` and
`outputs/eta-cache-investigation-20261008/production-*.json`. These checks establish the
observed numerical fix, not posterior convergence for the Unity experiments. Actual Unity
resubmission remains to be performed in a fresh results directory using the shared data.
