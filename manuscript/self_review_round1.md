# Self-review round 1: integrity and consistency

## Result

The manuscript is internally consistent with `results_final_v4/analysis/`, `results_chronological/`, `results_neural/` and the 20-seed controlled audit. The historical intermediate result files are not used for reported claims.

## Checks

- Public headline metrics match `headline_public.csv` to four decimal places.
- Paired confidence intervals match `paired_block_bootstrap.csv`.
- Intervention rates and Spearman correlations match `intervention_statistics.csv`; underflowed p-values are not reported.
- Neural values match `results_final_v4/analysis/neural_summary.csv`.
- Chronological values match the two chronological summary JSON files.
- Incremental node reduction, runtime reduction and zero maximum error match `synthetic_summary_20.json`.
- Filtered ranking averages every labelled answer and removes sibling answers.
- Causal evidence uses `event_time <= query_time`; chronological replay withholds same-timestamp events.
- Operating thresholds are selected only from validation data.
- Figures use committed analysis tables and pass the strict source QA, panel-alignment gate and rendered collision audit.
- Dataset URLs, revisions, access date, license metadata and SHA-256 values match `data/DATA_MANIFEST.md`.

## Corrections made

The accepted-window sensitivity join was changed from sampled `query_id` to the semantic `(subject, relation, timestamp, winner)` key with one-to-one validation. The figure layout was revised until all six rendered collision audits returned PASS.
