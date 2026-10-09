# Horseshoe eta-cache investigation — 2026-10-08

The initial investigation/prototype sections below preceded production adoption and did not
modify sampler/model/config/test files. Production adoption is recorded at the end. The analysis uses the
Unity results supplied under `/Users/jaehoonkim/Downloads/unity-20261008/`. All calculations
use float64 and the existing cache tolerance atol=1e-8, rtol=1e-6.

## Finding

The exclusion recurrence can amplify cancellation when a component has a large loading
and small other loadings. Direct combinatorial slopes eliminated eta-cache discrepancies
in all five available recorded pre-failure trajectory windows. Rebuilding the full polynomial
for every coordinate but retaining exclusion recurrence did not eliminate them.

For d=3, q1=e1-a_j and q2=e2-a_j*q1. Rounding in the subtraction for q1 is multiplied by
a_j before a second subtraction. An h error then contributes (new-old)*h_error to eta.
This occurs even when the reconstructed predictor is moderate. The DP predictor and
combinatorial predictor agreed to <=6.7e-16 at the ends of the five replayed windows,
so predictor reconstruction is not the observed source in those windows.

## Replay of Unity draws

Unity metadata records seven Horseshoe cache failures: one pilot and six long-run fits.
Two failed before retained draws began. For the other five, start at the preceding successful
cache boundary and replay the saved beta and V values in the original d→k→j order.
All three variants use exactly the same sampled values; no new MCMC draws are made.
The failure sweep itself was not retained: each window ends one sweep before the failure.
The cached recurrence residual at that endpoint matches the logged failure residual to
about 1e-15, while direct slopes keep the residual near ordinary float64 rounding.

| Case | Chain (1-based) | Recorded window | Unity failure error | Cached recurrence peak | Fresh full-polynomial recurrence peak | Direct combinations peak |
| --- | --- | --- | --- | --- | --- | --- |
| dense-2way-s1 | 4 | 12000–12099 | 9.172e-08 | 9.192e-08 | 7.172e-08 | 7.772e-15 |
| dense-2way-s3 | 2 | 9400–9499 | 4.125e-06 | 4.125e-06 | 3.275e-06 | 2.087e-14 |
| dense-3way-s2 | 2 | 13300–13399 | 4.328e-06 | 4.330e-06 | 4.749e-06 | 3.020e-14 |
| sparse-2way-s2 | 4 | 21800–21899 | 1.447e-06 | 1.447e-06 | 7.282e-07 | 1.599e-14 |
| sparse-3way-s1 | 3 | 9600–9699 | 8.806e-06 | 9.884e-06 | 8.972e-06 | 5.773e-15 |

All five current/fresh recurrence replays exceeded the existing tolerance within their
windows; all five direct-combination replays passed at every replayed sweep. Max direct
error was 3.02e-14. No intermediate eta refresh or tolerance relaxation was added.

## Large-loading snapshots and controlled stress

At sweep 9,626 of sparse-3way-s1, one saved order-3 component has
v=(-0.000641643,-6828.713,-0.0106193,-0.0197329,0.000497557). For X=(1,1,1,1,0),
excluding the second loading gives h=0.000229018171154 by recurrence versus
0.000229024480980 by direct combinations: error=-6.31e-9. On structurally zero rows,
the same recurrence also produced spurious nonzero slopes.

Compared 32 retained snapshots (largest order-3 loading and last retained draw from each
available chain in the failed-fit directories). For each snapshot, apply the same fixed
coordinate replacements to all variants from a freshly reconstructed starting eta.
Four snapshots caused a cache-tolerance violation with either recurrence variant; none
did so with direct combinations. In the largest-loading snapshot above, the resulting
cache error was 8.01e-5 with cached recurrence versus 6.66e-16 with direct combinations.

An artificial, separately labelled stress fixture uses all 32 patterns, d=3 and
v=(M,1/M,1/M,1/M,1/M). Halving the first loading gives:

| M | Recurrence eta error | Direct eta error |
| --- | --- | --- |
| 1e+03 | 4.729e-08 | 1.735e-18 |
| 1e+04 | 1.414e-04 | 5.421e-20 |
| 1e+06 | 1.523e+01 | 4.235e-22 |

## Diagnostic MCMC reruns

Use the server observations, resolved seed and chain index with current installed library
versions. Initial loadings match Unity exactly. The analysis-only direct variant delegates
to the unchanged common/local/component updates; it replaces only loading-slope evaluation
and omits the unused polynomial cache. At this investigation stage it was not installed into the production sampler.

| Case | Variant | Result | Max checked eta error |
| --- | --- | --- | --- |
| pilot/sparse-3way-s2 | cached | completed 4000 sweeps | 1.114e-09 |
| pilot/sparse-3way-s2 | direct | completed 4000 sweeps | 8.837e-14 |
| long-run/sparse-2way-s3 | cached | cache failure at 300 | 3.555e-07 |
| long-run/sparse-2way-s3 | direct | completed 24000 sweeps | 3.642e-14 |

The local recurrence long-run failed at sweep 300; Unity failed at 1,800. The pilot
recurrence run did not reproduce its Unity failure locally. Floating-point arithmetic and
stochastic paths can differ across macOS ARM64 and Unity, and direct slopes change the
seeded trajectory. Consequently the identical-recorded-draw replay above provides the
stronger arithmetic isolation. These checks establish numerical consistency in the tested
windows/runs, not full posterior mixing or universal stability for arbitrary dimensions.

## Artifacts and verification

Raw JSON and the analysis-only script are in `outputs/eta-cache-investigation-20261008/`
(Git-ignored). Reproduce with `.venv/bin/python outputs/eta-cache-investigation-20261008/analyze.py
inspect` or `replay`, with the downloaded results present at the stated path.
No downloaded files were modified. No sampling priors, cache tolerance or production
implementation changes are part of this investigation.

## Sequential Prefix/Suffix DP — follow-up feasibility check

A sequential prefix/suffix prototype preserves the coordinate scan while avoiding the
large-loading subtraction. For a fixed component/order, let a_j=X_j*v_j. Precompute suffix
coefficients S[j,t]=e_t(a_j,...,a_{p-1}) through degree d-1. While scanning j=0,...,p-1,
maintain P[t]=e_t(new a_0,...,new a_{j-1}), starting at P[0]=1. Then

```
h_j = X_j * sum(P[t] * S[j+1, d-1-t] for t=0,...,d-1)
```

Only feasible degrees participate in the sum. After drawing the new v_j and updating eta,
advance the prefix with `P_new[t]=P_old[t]+(X_j*v_j_new)*P_old[t-1]`. Descending-degree
updates, or a materialized RHS using old prefix coefficients, preserve this identity.
The suffix stays valid because future coordinates have not yet changed. A static prefix
computed at the start of the pass would not reflect sequential Gibbs updates.

This evaluates only the required degree d-1 coefficient, not a full polynomial convolution.
Suffix construction, all slope queries and prefix updates each cost O(npd), so the total
loading-pass order remains O(n p sum_d d R_d). Per-component coefficient storage increases
from O(nd) to O(npd). At n=200,p=5,d=3, suffix plus rolling prefix tables occupy 33,600
bytes (32.8 KiB), versus 4,800 bytes for the current full-polynomial table. Tables are reused
component by component; this is coefficient storage, not a peak-RSS measurement.

### Numerical comparison

Replayed the same five recorded Unity windows with identical beta/V updates and no extra
eta refresh. Every prefix/suffix replay passed the existing atol=1e-8,rtol=1e-6 criterion.

| Case | Chain | Prefix/suffix peak eta error | Tolerance-violating sweeps |
| --- | --- | --- | --- |
| dense-2way-s1 | 4 | 7.772e-15 | 0 |
| dense-2way-s3 | 2 | 1.954e-14 | 0 |
| dense-3way-s2 | 2 | 3.109e-14 | 0 |
| sparse-2way-s2 | 4 | 1.599e-14 | 0 |
| sparse-3way-s1 | 3 | 5.773e-15 | 0 |

Maximum eta error was 3.109e-14, comparable to direct combinations. The prototype also
matched independent combinations for sequential fixtures (p,d)=(5,2),(5,3),(5,5),(7,4),
and passed the earlier large-loading stress fixtures through M=1e8. These are independent
analysis assertions, not changes to the production test suite.

An analysis-only Horseshoe chain on long-run/sparse-2way-s3 completed **24,000 sweeps**
with maximum checked eta error 3.553e-14. Its sampled trajectory differs
from other variants; no posterior convergence or universal floating-point stability claim
is inferred. The stability conclusion is supported primarily by identical-draw replay.

### Cost measurements

Local macOS ARM64, same installed versions, one BLAS/OpenMP thread. Isolated component
passes use fixed sequential replacements. Seven batches of 20 passes after warm-up; times
are medians and include constructing coefficient tables. Direct combinations are timed
only at p=5. Excluded-DP rebuild recomputes the polynomial on p-1 variables for every j.

| n | p | d | Current recurrence | Prefix/suffix | Excluded-DP rebuild | Direct combinations |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 5 | 3 | 0.053 ms | 0.059 ms | 0.057 ms | 0.117 ms |
| 500 | 50 | 3 | 0.607 ms | 1.011 ms | 6.139 ms | not timed |
| 500 | 100 | 3 | 1.233 ms | 2.040 ms | 24.978 ms | not timed |

At p=5 the isolated order-3 pass costs about 12% more than recurrence. At p=50/100 this
prototype costs about 65–67% more, although it remains linear in p and is about 6x/12x
faster than excluded-DP rebuilding. Thus equal asymptotic order does not imply equal
constant cost. At p=5, excluded-DP rebuilding is also inexpensive; prefix/suffix's scaling
advantage becomes clearer as p grows.

Timed complete loading blocks separately, including Gaussian conditional calculations and
eta updates for n=200,p=5,R_2=R_3=5. Each method starts from identical state and RNG values;
state copying and RNG setup are included equally. Rotate method order across nine batches
of 40 calls and report seven post-warm-up batch medians. The large-loading state is the
saved sparse-3way-s1 snapshot at sweep 9,626.

| Initial state | Current recurrence | Prefix/suffix | Direct combinations |
| --- | --- | --- | --- |
| ordinary | 0.731 ms | 0.721 ms | 1.250 ms |
| large_loading | 0.736 ms | 0.721 ms | 1.253 ms |

For the current pilot dimensions, complete loading-block cost is effectively unchanged
in these local measurements. These are loading-block timings, not full-sampler walltime,
ESS/sec or Unity performance measurements. The method is a promising replacement for
exclusion recurrence in the current setting, with the stated memory and large-p cost tradeoffs.

Prototype and JSON results are in `outputs/eta-cache-investigation-20261008/prefix_suffix.py`,
`prefix-suffix-*.json` and `benchmark_loading_pass.py`. Production sampler/model/config/test
files remain unchanged. Existing tests: **103 passed in 9.65 seconds**; Ruff passed.

## Production adoption — 2026-10-08

At the user's request, Sequential Prefix/Suffix DP is now the final production slope
algorithm. `model.py` supplies suffix construction, rolling-prefix advancement and the
single-degree convolution. Normal and Horseshoe loading scans and Reference SSP active
gamma/slab scans use it. SSP advances with the effective loading even for unchanged
indicators or inactive slab coordinates. The full-polynomial exclusion recurrence and
incremental delta-coefficient updater were removed from production.

Standalone `loading_slope(X,v,d,j)` calls remain supported; samplers pass keyword
`prefix`/`suffix` tables to avoid per-coordinate reconstruction. Common Gaussian updates
retain immediate eta updates. Gibbs ordering, priors, float64 and cache tolerance are
unchanged. Legacy recurrence formulas are frozen only in ignored analysis artifacts for
historical comparisons; they are not a production fallback.

Replayed the five saved failure windows through the actual production
`horseshoe.update_loadings`, supplying the original recorded coordinate draws. Every
window passed, maximum eta error 3.109e-14. Actual production `horseshoe.sweep` also
completed 24,000 sweeps on the sparse-2way-s3 diagnostic chain, maximum cache error
3.553e-14, with no numerical failures.

**112 tests passed in 10.25 seconds**, including independent combination/conditional checks,
large excluded loadings through 1e8, Normal/Horseshoe/SSP large-loading passes, SSP gamma
birth/death/unchanged gates and inactive slabs, replay/storage contracts, and the independent
quadrature posterior checks. Ruff lint/format checks passed. Nine smoke fits completed
7,200 sweeps without numerical failures; all retain short-run `mixing_flagged` status.
Production validation artifacts are `production-recorded.json`, `production-chain.json`
and `production_validation.py` in the ignored investigation directory. Smoke files are in
`outputs/smoke-prefix-suffix-production-20261008/`. No Unity jobs were resubmitted here.

A post-adoption loading-block timing using the same controlled setup as above measured
0.781/0.750 ms (legacy recurrence/production prefix-suffix) for ordinary state and
0.711/0.712 ms for the large-loading state. These local variations support similar cost
at current pilot dimensions; they are not a speedup or full-sampler efficiency guarantee.
Use a new Unity results directory when rerunning the same prepared observations.
