# Final manuscript claim–evidence map

The manuscript's central claim is an exact, computable tolerance to shared-window
withdrawal at a frozen neural–memory inference interface. It does not claim a
new state-of-the-art temporal encoder or a universal correctness improvement.

| Claim in the Abstract/Introduction | Evidence and location | Interpretation |
|---|---|---|
| The original prediction scores do not determine withdrawal stability. | Figure 1; `fig01_histories.csv` and `fig01_response.csv`; exhaustive conceptual example. | Two explicitly constructed histories have the same unedited distribution but different deletion responses. |
| Renormalized deletion admits exact additive certification. | Main Proposition 1, Eq. (6), Supplement S1; `tests/test_window_certificate.py`. | The sign of the pairwise numerator is optimized exactly; the surplus is not an unscaled worst margin. |
| Sparse competitors preserve the full-vocabulary guarantee. | Main Proposition 2; sparse/dense and exhaustive-oracle tests; analytical microbenchmark. | Supported objects and the strongest unsupported neural competitor suffice under fixed candidates and encoder. |
| Certification agrees with directly evaluated witnesses. | `revision/CORE_CHECKS.json`; 12 final runs, 60,000 query–seed rows, budgets 1–3. | 180,000 checks, zero observed certificate violations and zero witness mismatches. |
| Surplus improves intervention discrimination by 0.064–0.584 AUROC. | Figure 4; `results_nc/statistics/diagnostic_intervals.csv`. | All 16 paired intervals exceed zero, conditional on the three fitted models; alternative partitions are diagnostic tests. |
| Certified CTF selection lowers incorrect-or-flipping joint error at 40% coverage. | Results, Figure 3, Table 2; `paired_intervals.csv`. | Reductions are 11.07 and 4.05 percentage points; initial answer error increases slightly and is shown beside flip error. |
| Exact checking avoids conservatism and finite-probe misses. | Figure 5; extended controls on seed 0 and all 5,000 queries per group. | Exact-minus-bound supported coverage is 2.28–12.09 points; the 64-probe false-safe fraction reaches 11.35%. |
| Dependency locality reduces maintenance work. | Figure 6; 240 paired delete/restore transactions; independent 128-query rebuild per dataset. | Same writes, statistics and refresh kernel; fixed 5,000-query registrations; GDELT is a frequency-anchor scaling experiment. |

## Material qualifications retained in the main text

- Neural training uses the published interpolative split; only the editable
  memory is restricted to observations at or before a query timestamp.
- Strong confidence/ensemble selectors already perform well. Certificate-feature
  calibration has higher mean AURC in all four groups; certification supplies
  a separate invariance requirement.
- Exact guarantees apply to the declared shared-window family, frozen encoder
  and fixed candidates. Broader perturbations are tested as diagnostics.
- Empty-memory queries and zero-affected transactions are explicitly counted.
- TeRDy/ICEWS14 does not show a supported MRR improvement from memory fusion.
- Time-block uncertainty is conditional on the fitted seeds; seed SD is separate.

## Placement and terminology

All six principal figures and both principal tables are in the main manuscript.
The supplement contains complete selector, threshold, bootstrap, weight-sweep,
timing and cache tables. Detailed provenance and execution records reside in
the repository. This keeps the main argument readable without hiding changes
to its interpretation.

Canonical names are StableKG, TeRDy, compact temporal factorization (CTF),
neural anchor, editable temporal memory, certificate surplus, certified capacity,
answer error, edit-flip rate and joint error. Internal `TemporalComplEx` paths
refer to CTF and are not claims to reproduce an external leaderboard model.
