# Closest-work positioning

The bibliography contains 31 verified records from 2025–2026. Metadata checks
are recorded separately from claim support. The following archived abstracts
support the central distinctions made in the Introduction and Discussion.

| Work | Direct source | Supported distinction |
|---|---|---|
| NCPNET, KDD 2025 | `literature/neurocomputing_20260922/wang2025nonexchangeable_openalex.json`, abstract | Conformal prediction for temporal graph neural networks, with diffusion-based nonconformity and temporal coverage control. Its prediction-set objective differs from deterministic invariance of one fixed answer under declared memory edits. |
| IGETR, Neural Networks 2026 | `literature/neurocomputing_20260922/fan2026editing_pubmed.xml`, PubMed abstract | Temporal GNN candidate paths, LLM-guided path editing and integration of refined paths. StableKG edits an explicit inference memory and proves a deletion guarantee for the resulting score. |
| GraphProt, IJCAI 2025 | `literature/neurocomputing_20260922/yang2025graphprot_openalex.json`, abstract | Black-box shielding with topology/feature filtration, sampled subgraphs and majority-vote ensembles. Its threat model and randomized construction differ from the shared-window adversary analyzed here. |
| AdaptDel, NeurIPS 2025 | `literature/neurocomputing_20260922/adaptdel_openalex.json`, abstract | Variable-rate deletion smoothing for sequence classifiers under edit-distance perturbations. StableKG obtains exact deterministic certification from its structured neural–memory mixture. |

The manuscript makes no claim that any of these methods is incapable of an
extension to this setting, and does not compare their task-specific numerical
results with the present query protocol. TeRDy is used as a pinned neural
backbone, not presented as a new architecture or a leaderboard reproduction.

The originality claim is the exact renormalized shared-window calculation,
the supported-competitor reduction and their integration with calibrated
selection and local maintenance. Broad claims of being the first robustness,
selection, graph-editing or temporal-uncertainty method are not made.
