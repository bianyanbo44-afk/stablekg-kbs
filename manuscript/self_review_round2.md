# Self-review round 2: reviewer and KBS fit

Superseded by `self_review_20260909.md`. The readiness judgment below preceded
the mathematical, matched-control, numerical and figure-source checks of 9 September.

## Scientific story

The paper begins with the deployment decision that ranking alone does not answer: whether a conclusion is ready to use. The method then separates belief from local evidence stability, and the Results progress from public causal benchmarks to neural transfer, chronological replay, direct interventions and exact incremental updates.

## Claim discipline

- The paper does not claim TKGC state-of-the-art ranking.
- GDELT is described as a scale validation with localized rather than uniform gains.
- The compact neural model is described as a confirmation backbone.
- The certificate is defined relative to the declared evidence kernel and intervention family.
- The controlled drift recovery lag is not used as a headline result.

## KBS fit

The manuscript contributes a knowledge-based reasoning protocol, calibrated selective decisions, explicit abstention and incremental maintenance of derived conclusions. The implementation is modular and can wrap a learned or deterministic temporal KG scorer.

## Reproducibility

The repository contains fixed commands, dependencies, tests, source tables, vector figures, dataset hashes and seed records. The Supplementary Information records the operating protocols and all headline intervals.

## Presentation

Figures use consistent semantic colors, editable vector text and a restrained multi-panel layout. The figure source passes strict font, export, alignment and collision QA. Remaining dataset- and backbone-specific variation is stated in neutral, quantitative language.

## Decision

Ready for a KBS pre-submission package after the corresponding-author metadata and final repository URL are filled in.
