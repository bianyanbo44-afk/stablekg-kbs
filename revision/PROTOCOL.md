# Neurocomputing revision protocol — 22 September 2026

Baseline archive: KBS submission at git commit `18267a3`. Work proceeds on
`neurocomputing-revision-20260922`; the submitted manuscript is preserved there.
The author authorized a substantive revision, experiments, figures, self-review,
and a complete submission package, without subagents.

## Research question
Can a frozen temporal neural predictor expose an editable evidence channel
whose answer stability under shared temporal-window deletion is certified
exactly, while retaining useful link prediction and selective correctness?

## Model and guarantee
**Final protocol:** the primary method uses renormalized evidence,
with an exact ratio-to-affine certificate, as specified in `THEORY.md`. The
fixed-reference proposal is retained as a declared analytical control.
The new theorem and implementation pass exhaustive small-instance
checks before revised test scoring. Final outputs use
`results_nc/evaluation_renormalized`; earlier pilot/evaluation folders are not
paper evidence. Both the neural distribution and evidence distribution are
conditioned on the fixed candidate vocabulary that excludes training-observed
answers at the query timestamp. Empty memory falls back to the neural anchor.

Fuse neural object probabilities and normalized, decayed historical event mass.
Renormalize the memory after every deletion; complete withdrawal falls back to
the neural distribution. A window deletion affects every object in that window.
The retained-mass transformation yields an additive signed pairwise numerator.
All-competitor certification
must be checked against brute force; empty history and deterministic ties are
part of the specification. This concerns inference-time evidence edits with a
frozen encoder, not retraining, unlearning, or causal intervention effects.

## Validation and evaluation
Published interpolative Pe splits and existing deterministic 5,000-query panels.
Neural training uses the published training facts; the evidence channel retains
only training facts at or before the query time. Validation roles: 0:1000 model
checkpoint and fusion choice; 1000:3000 calibration fit; 3000:4000 selector choice;
4000:5000 operating thresholds. No test labels or test interventions select
hyperparameters. Earlier KBS test results are already known, so this is a
retrospective methodological extension, not an untouched preregistered study.

Primary datasets: ICEWS14 and ICEWS05-15, TeRDy and compact temporal ComplEx,
three training seeds (0, 1, 2). GDELT supplies an additional dense-graph check.
New TeRDy training: rank 6,000 / 2,000, up to 36 / 24 epochs respectively;
checkpoint interval 3 / 2, patience 4 validation checks. Published model and
regularizers are unchanged; resource-specific rank is recorded.

Fusion pilot uses validation only. Candidate temperatures 0.5, 1, 2, 4 and
evidence weights 0.1, 0.25, 0.5, 0.75; neural-only is a separate comparator.
Decay half-life 35 timestamp units; shared windows of 7 timestamp units.
Report the evidence-weight sensitivity including weight zero, rather than
interpreting a vanishing evidence channel as informative edit robustness.

Selectors: maximum probability, margin, negative entropy, temperature scaling,
ridge-calibrated neural/hybrid score, evidence descriptors without certificate,
and certificate-augmented selector. Include the three-model ensemble. Match
coverage on the same queries and predictions; deployed thresholds come from
validation only. Correctness and edit stability are separate outcomes.

Interventions: exact worst-case shared-window deletions, random window deletions,
alternate window boundary/width, and event-level deletions. A certificate for
one partition is not relabelled as a guarantee for another partition. Report
certificate overhead, update closure size and wall time, with recomputation
agreement checked against an independent direct calculation.

Statistics: MRR and Hits@1/3/10 use training-plus-validation truth for validation
filtering and complete known truth for final test filtering. Candidate selection
excludes only answers observed in training at the same query. Correctness uses
the corresponding split-safe known-answer set. Truth labels do not enter
score/selection features. The final strict-truth correction, made after an
exploratory test evaluation, is recorded in `STRICT_TRUTH_CORRECTION.md`.
AURC, Brier,
ECE, risk at 20%/40% coverage, and deployed coverage. Three-seed means and
standard deviations; paired timestamp-block bootstrap confidence intervals
(2,000 replicates), including margin and calibrated-score controls. Describe
at least one neutral or adverse result in proportion to its importance.

## Deliverables and review
Standalone reproduction commands; raw result tables; vector PDF/SVG and PNG
figures; complete Neurocomputing main text, supplementary methods/results,
title page, cover letter, highlights, declarations and source package. Relevant
references dated 2025–2026 are verified against primary metadata. Two inline
self-review passes are disclosed as self-review, not independent peer review.

## Local resource reuse
`data/processed_recent`, `external/TeRDy`, and `external/LTGQ` are junctions to
the original project's cached public data / pinned author code. The revision
does not modify those targets. Local environment:
`F:/CCFA/new_kbs_idea_20260907/.venv-gpu/Scripts/python.exe`.

## Validation-stage correction (before revised test evaluation)
The pilot exposed partial-answer false negatives in the original KBS selector:
it checked only the split-specific answer list. All revised results use the
unobserved-fact protocol above. The first exploratory pilot outputs were
superseded, not reused as paper results. ICEWS14 and ICEWS05-15 validation
targets have no overlap with training facts. GDELT contains such overlap and
requires explicit target deduplication before a completion experiment; its
full raw training graph remains suitable for the update-scaling experiment.
# Maintenance implementation note

Before final maintenance measurements, competitor calculations were vectorized
for supports of at least 16 competitors. This changes execution only, not the
score, edit family, outputs, sampled transactions or repetition count. The
same kernel is used for full and local refresh. An independent scalar
witness-producing implementation agrees on 150 randomized cases including
tiny retained masses. Test selection configurations are unaffected.
