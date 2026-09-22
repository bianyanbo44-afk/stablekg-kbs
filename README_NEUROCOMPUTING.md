# Exact stability certificates for neural temporal knowledge graph completion

This revision implements a frozen neural predictor with an editable,
renormalized temporal memory. Exact shared-window deletion certificates,
sparse full-vocabulary checks, calibrated selection and local maintenance are
evaluated with two backbones, two public completion datasets, three training
seeds and a separate GDELT maintenance benchmark.

The complete revised manuscript is `manuscript/neurocomputing/main.pdf`.
Its source is `main.tex`; the supplement contains proofs, all selectors and
paired intervals. Main figures are editable PDF/SVG files in
`figures/neurocomputing`, with CSV source data beside them. The original KBS
manuscript and historical outputs are preserved for provenance and do not
supply the revision's numerical results.

## Rebuild figures, statistics and manuscripts

Install `requirements-neurocomputing.txt` in Python 3.11 or newer. The tested
environment is recorded in `results_nc/provenance/environment.json`. To rebuild
from the supplied compressed query-level outputs, without training or a GPU:

```powershell
./run_neurocomputing.ps1 -CompilePaper
```

On first use, the source-data archive is downloaded from the public
`neurocomputing-v1` GitHub Release and checked against the SHA-256 manifest
tracked in this repository. The large archive is a release asset, so a normal
source checkout stays compact.

Set `STABLEKG_PYTHON` to a Python executable when using a dedicated environment.
Compilation requires `latexmk`, BibTeX, `elsarticle`, `algorithm`,
`algpseudocode`, `amsthm`, `xurl`, `microtype` and standard LaTeX packages.
Equivalent platform-independent commands are:

```text
python scripts/archive_nc_source.py --restore
python experiments/statistics_nc.py
python scripts/build_nc_tables.py
python scripts/build_nc_controls_tables.py
python figures/plot_neurocomputing.py
cd manuscript/neurocomputing
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

## Reconstruct the model experiments

Use an NVIDIA GPU with 8 GB or more and a suitable CUDA PyTorch installation:

```powershell
./run_neurocomputing.ps1 -FullRun -CompilePaper
```

This downloads and verifies the immutable public archives, obtains the pinned
TeRDy implementation, trains the compact factorization and TeRDy models,
freezes validation choices, exports test scores, and runs the intervention,
maintenance and statistical analyses. Existing checkpoints are reused.
Validation is refitted and inference arrays are exported from those checkpoints,
so the summaries tracked in a clean checkout cannot silently bypass evaluation.
The released compact-model results reuse archived fixed 30-epoch fits;
`experiments/train_compact_nc.py` recreates their training procedure from the
original source-order queries. Checkpoint hashes document the exact evaluated
artifacts. Hardware/library differences can change floating-point model fits;
archived query outputs reproduce the published tables without this variation.

Validation has four non-overlapping roles: checkpoint/fusion choice, calibrator
fit, selector tuning and operating thresholds. Every revised test inference
requires the hashed frozen configuration manifest. Prediction candidates
exclude observed training answers at the exact query timestamp. Test labels
are used only for evaluation in the final pipeline. Validation truth sets
contain training and validation facts only, enforced by a regression test.
The recorded development correction is in `revision/STRICT_TRUTH_CORRECTION.md`.
Final test evaluation uses complete known-answer sets for correctness
and designated held-out answers for filtered ranking.

## What is measured

- ICEWS14 and ICEWS05-15: 5,000 public test queries each; TeRDy and 96-dimensional
  compact temporal factorization; three training seeds per combination.
- Shared-window deletion: exact budgets 1, 2 and 3; random, shifted and wider
  partitions; single-event deletion; conservative and finite-probe controls.
- Maintenance: 5,000 registered queries on each of three public histories;
  identical event writes and window statistics in local and full refresh.
  GDELT uses a training relation-frequency anchor for scaling, not a claimed
  neural accuracy result.
- Statistics: 2,000 paired temporal-block bootstrap draws conditional on the
  fitted models, plus separate three-seed standard deviations.

The split is interpolative for the neural encoder. The certificate concerns
the frozen inference interface under defined memory deletions. It does not
certify changes to encoder weights. Correctness and edit stability are distinct
endpoints, and unavailable certified coverage remains explicitly unavailable.

## Tests and data terms

```text
python -m pytest tests -q
```

Tests include independent exhaustive deletion oracles, tiny retained masses,
tie handling, sparse/dense equivalence, local/full maintenance and weighted
statistics. `data/DATA_MANIFEST.md` records upstream attribution, source terms,
archive URLs and hashes. The Neurocomputing protocol and exact cache hashes
are recorded in `revision/PROTOCOL.md` and `results_nc/provenance`.

Original project code is MIT licensed. External code and public datasets
retain their own terms. No private data, credentials or local environments are
included in the release.
