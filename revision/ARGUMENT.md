# Manuscript argument and terminology

## One-sentence argument
An editable temporal evidence channel can equip a frozen neural knowledge-graph
predictor with exact, efficiently maintained certificates against coherent
window deletions, while confidence calibration separately controls empirical
answer error.

This is an algorithmic/methods article for Neurocomputing. The contribution is
an exactly certifiable neural/evidence interface and its empirical operating
trade-offs; it is not a claim that a robustness feature universally improves
classification accuracy or that deleting an input fact unlearns the encoder.

## Reader-facing evidence sequence
1. Problem: a high-confidence answer need not be insensitive to revised evidence.
2. Construction: neural anchor + editable memory, normalization after deletion.
3. Theorems: exact shared-window adversary, sparse competitor reduction, update locality.
4. Prediction: two backbones, two public datasets, three seeds; neural-only vs hybrid.
5. Decision: strong confidence controls, exact certificate coverage, matched risk.
6. Mechanism: coupled vs winner-only deletion, empirical sampling vs exact adversary,
   held-out perturbation partitions, and fusion-weight trade-offs.
7. Practicality: certificate overhead, exact local updates, scaling and reproducibility.

## Terminology ledger
- **Neural anchor**: frozen learned categorical object distribution p_theta.
- **Editable evidence channel**: decayed training-event memory for the query's (s,r),
  restricted to event times <= query time.
- **Reference mass** S_q: total memory weight before the hypothetical deletions;
  post-edit evidence is renormalized by the retained mass.
- **Shared temporal window**: a group of events whose timestamps fall in one fixed bin;
  deletion removes every candidate's events in the selected bin.
- **Deletion budget** k: at most k nonempty shared windows.
- **Certificate surplus** R_k: smallest signed gap numerator after multiplying by
  retained mass; its sign certifies the post-edit answer.
- **Certificate radius**: maximum deletion budget that retains the same answer.
- **Belief**: validation-calibrated empirical probability of known-answer correctness.
- **Certified selection**: belief acceptance AND a valid k-window certificate.
- **Unobserved-fact completion**: exclude training-observed answers at the query time;
  evaluate top-1 against the complete published known-answer set.
- **Interpolative split**: published benchmark split, not a neural forecasting claim.

## Validation-stage interpretation
The corrected older-TeRDy pilot does not establish a general correctness advantage
from certificate features beyond strong confidence controls. Its intervention
diagnostics do support separating correctness calibration from edit stability.
The revised paper must retain this distinction. These pilot numbers will not be
used as final results; the main TeRDy runs have longer validation-selected training.

## Selection decision before new test scoring
Freeze each ridge calibrator and comparator choice on validation. Report all
controls, rather than replacing a baseline after inspecting test results. The
certificate gate is applied to the best validation-selected calibrated belief
among Score/Evidence/Certificate calibration. Raw confidence baselines remain
separate. This operating policy does not require certificate features to improve
correctness calibration. Correctness and edit-induced flips remain separate
endpoints; their union is the joint error endpoint.

## Main-text allocation
All figures needed for the above claims remain in the main text. Supplementary
material contains expanded run-level tables, proofs, reproducibility details,
and secondary sensitivity results. No conclusion-changing result is hidden there.
