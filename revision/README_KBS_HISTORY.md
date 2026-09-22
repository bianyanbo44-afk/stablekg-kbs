# StableKG

StableKG connects calibrated acceptance with evidence-deletion diagnostics.
The discrete certificate is the minimum number of whole winner-support windows
whose deletion erases a positive margin in an additive temporal evidence scorer.
Continuous descriptors support calibrated selection and intervention diagnosis.
A separate controlled rule-DAG experiment evaluates exact dependency updates.

The repository contains the transparent causal benchmark, a compact temporal
ComplEx-style neural backbone, prequential chronological audits, intervention
experiments, source-data tables, figure generation, and the KBS manuscript.

## Reproduce the reported results

1. Install Python dependencies with `python -m pip install -r requirements.txt`.
2. Download and verify the public archives with `python scripts/download_data.py --dataset all`.
3. Run the public benchmark, interventions, chronological audits and analysis:

```powershell
.\run_all.ps1
```

4. Add `-RunSynthetic` to regenerate the 20-seed controlled audit. Add
   `-RunNeural` to train the reported 3-seed neural backbones. Add
   `-CompilePaper` to compile both the Elsevier `elsarticle` manuscript and
   its supplementary information.

The corrected public-data results are in `results_final_v5/`; `results.csv`,
`public_ICEWS14.csv`, and earlier `results_final/` benchmark summaries are
historical artifacts and are excluded from the manuscript. The combined
statistics used by the figures are in `results_final_v5/analysis/`.

`-BuildOnly -CompilePaper` regenerates statistics, six figures, the main PDF
and supplementary PDF from the supplied experimental outputs. `-CompilePaper`
alone compiles the current LaTeX without rerunning experiments. Every external
process exit code is checked. For a complete run including model training:

```powershell
.\run_all.ps1 -RunSynthetic -RunNeural -CompilePaper
```

The manuscript entry point uses the reported 5,000-query, five-seed protocol.
Use individual experiment CLIs with separate output directories for smoke runs.
LaTeX requires `elsarticle`, `latexmk`, `xurl` and `placeins` in addition to the
math and graphics packages. Figure reproduction has a local geometry checker;
the development-only figure skills are optional for reproduction.

The review release includes matched controls, intervention AUROC comparisons
and equal accepted-count comparisons. `requirements-lock.txt` records tested
direct package versions; `results_review_20260909/environment.json` records the
Python/platform details. The earlier v4 results are preserved as historical
outputs and are superseded by the corrected full-mass deletion calculation.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover the causal time cutoff, standard filtered average-tie ranking,
multi-answer aggregation, validation-only threshold selection, and exact
incremental/full recomputation agreement.

## Reproducibility record

All reported runs record dataset hashes, split counts, entity and time counts,
random seeds, model dimensions, optimizer settings, calibration and operating
sample sizes, thresholds, and source-data paths. Public benchmark figures use
5 deterministic calibration seeds. Neural-backbone confirmation runs use 3
training seeds. Controlled dependency-closure audits use 20 seeds in four
five-seed batches to avoid long-process interruption on CPU-only machines.

## Citation and license

The dataset provenance, archive URLs, license metadata and SHA-256 values are
listed in `data/DATA_MANIFEST.md`. Code in this repository is released under
the MIT license in the accompanying submission package; the mirrored datasets
remain subject to their source terms.

## Recent backbone and public update extensions

Completed author-code TeRDy runs are in `results_recent_v2/` (ICEWS14) and
`results_recent_v3/` (ICEWS05-15). The public event-maintenance benchmark is in
`results_real_updates/`. Reproduction commands are:

The exact author-code checkouts are retained at commits `f3f47986adb97eee26ac2e59811dc0d02df570f4` (TeRDy) and `4617c8af7dfe1c12bc9f36f074923c2a73e3046b` (LTGQ adapter source).

```powershell
.\.venv-gpu\Scripts\python.exe experiments\recent_backbones.py --model TeRDy --dataset ICEWS14 --epochs 12 --eval-every 3 --batch 4096 --eval-batch 128 --device cuda --output results_recent_v2
.\.venv-gpu\Scripts\python.exe experiments\recent_backbones.py --model TeRDy --dataset ICEWS05-15 --rank 2000 --epochs 8 --eval-every 2 --batch 6000 --eval-batch 128 --device cuda --output results_recent_v3
python experiments\prepare_public_cache.py
python experiments\real_incremental.py
```
