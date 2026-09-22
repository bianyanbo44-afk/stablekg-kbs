# Neurocomputing revision self-review

This is an internal self-review in the same working context as manuscript
development. It is not independent peer review, and no subagents or simulated
reviewer identities were used.

## Review pass 1 and revisions

| ID | Concern and evidence | Action and resolution criterion |
|---|---|---|
| M1 | Globally complete validation answer sets could include test-only facts. Located in the public cache construction and validation use of field 4. | Introduced the strict split-truth adapter; retrained all six TeRDy fits with unchanged settings; refitted all 12 validation configurations and replaced all primary results. Regression test asserts exclusion of test-only answers. Closed by the strict final runs and CORE_CHECKS.json. |
| M2 | The former additive diagnostic did not certify the exposed neural prediction. Located in the preserved KBS manuscript. | Replace it with the renormalized neural–memory score, prove exact shared-window certification, and supply constructive witnesses. Main Method and Supplement S1 give the reduction, tie cases and complete-deletion fallback. |
| M3 | A generic correctness-improvement claim is not supported by strong confidence and ensemble controls. Located in the first complete revision comparison. | Separated correctness calibration from certificate gating; retained every control and reported answer error, edit flips and their union. The corrected results confirm a specific stability capability and joint-error reductions at declared coverage; the main text reports the correctness trade-off and higher mean certificate-calibrator AURC. Closed without claiming universal superiority. |
| M4 | Full-refresh controls could be made artificially slow by omitting shared sufficient statistics. | Use identical event writes, cached window statistics and the same certificate kernel on both paths. Time the difference in refreshed query sets. Verify against a raw-event rebuild. |
| M5 | GDELT does not supply an equivalent clean held-out neural panel in this distribution. | Use it explicitly as a relation-frequency-anchor maintenance experiment. Primary neural completion is on the two ICEWS datasets. |
| m1 | Some prototype figure legends crossed grid strokes; title styles inherited a colored border. | Remove the overlap and inherited border. Inspect final-size renders and retain PDF geometry checks. |
| m2 | Accuracy and filtered Hits@1 have different denominators and truth rules. | State the selected-answer and target-wise estimands in the main Experimental design and table captions. |
| m3 | Bootstrap intervals could be misread as accounting for model retraining. | State that paired time-block intervals are conditional on the fixed model fits, and report seed SD separately. |
| m4 | Original CTF checkpoint reuse could be mistaken for a fresh confirmatory fit. | Identify archived fixed-epoch checkpoints, retain their hashes, provide training reconstruction, and disclose reuse in the supplement. |

## Technical coverage

Originality is assessed against exact edit certification, sampled certification,
temporal conformal prediction and graph-path editing. Scientific importance is
the operational connection between prediction, a declared edit tolerance and
maintenance. Technical soundness requires the exact theorem, exhaustive small
oracles, sparse/dense consistency, strict validation isolation and equal-work
maintenance controls. Readability is assessed through the equal-score example,
explicit endpoint separation and the ordered figure sequence. The journal's
editor retains the decision on significance and fit.

Human/animal, clinical and private-data review axes are not applicable to these
public benchmark experiments. Public source attribution, data terms, author
metadata, no external funding, no competing interests and AI-use disclosure
are applicable and included.

## Review pass 2

Completed on 22 September 2026 using the strict final experiments, regenerated
tables/figures and the final manuscript sources.

- **Theory and numerical realization.** Reviewed both propositions, the
  complete-deletion fallback, tie ordering, empty memory and tiny retained
  masses. The meaningful test suite has 27 passing tests. All 180,000 final
  query–seed–budget checks agree with direct witness evaluation.
- **Experimental evidence.** All 12 primary evaluations contain 5,000 test
  queries. All 16 diagnostic interval comparisons are reported. Analytical
  controls retain every group, every weight and all/nonempty timing strata.
  Maintenance includes all 240 transactions, matching local/full outputs and
  three independent raw-event rebuild checks. No exploratory numbers remain
  in the revision's main text, supplement or figures.
- **Interpretation.** Abstract and Introduction claims map to proofs and
  measured evidence in CLAIM_EVIDENCE.md. Correctness, filtered ranking,
  intervention flips and their union have separate definitions. Confidence
  comparisons, empty-memory fractions and zero-affected transactions remain
  visible. No state-of-the-art encoder, chronological forecast or universal
  accuracy advantage is asserted.
- **Positioning and attribution.** All 31 cited references have verified
  metadata and publication years 2025–2026. Closest-work distinctions are tied
  to archived source abstracts in LITERATURE_POSITIONING.md. Data archives,
  source terms, pinned external code and checkpoint hashes are recorded.
- **Figures and pagination.** Six editable vector figures passed measured
  alignment, PDF font and collision checks; every figure was visually inspected.
  All principal figures/tables are in the 13-page main text. The 10-page
  supplement contains the complete tables. Page renders show no out-of-page
  text; final LaTeX logs contain no undefined references or overfull boxes.
  Compact supplement pagination replaced unnecessary isolated table pages.
- **Reproduction and packaging.** The documented build command completed in
  a clean directory without benchmark data, model weights or junctions.
  Regenerated statistics and figure CSVs agree at 1e-12 tolerance and all
  generated tables match; see REPRODUCTION_CHECK.json. Source-archive download
  has a SHA-256 check. FullRun was corrected to refit validation despite
  released summary files and to re-export scores from the supplied checkpoints.
  The flat LaTeX upload ZIP also compiled in a separate directory; its 13-page
  main text and 10-page supplement match the final PDFs page by page in extracted
  text. SOURCE_PACKAGE_CHECK.json records this check and source-data integrity.
- **Author declarations.** Name, official affiliation, email and programme
  agree across the manuscript and four editable submission documents. The
  author confirmed no external funding and no related competing interests.
  AI assistance is disclosed. Journal submission remains an author action.

The revised package supports its stated exact-certification and maintenance
contribution and is ready for author final reading and submission preparation.
This internal review establishes neither independent peer-review approval nor
an editorial acceptance probability. The current full online author guide
remains subject to a human-verification challenge, as recorded in
JOURNAL_REQUIREMENTS.md; live submission fields govern the final uploads.
