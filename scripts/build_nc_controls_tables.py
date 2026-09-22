"""Summarize every prespecified analytical and maintenance control for the SI."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results_nc'
OUT = ROOT / 'manuscript/neurocomputing'
GROUPS = [(m, d) for m in ('TeRDy', 'TemporalComplEx')
          for d in ('ICEWS14', 'ICEWS05-15')]


def table_start(caption, columns, header):
    return (r'\begin{table}[H]\centering\small\caption{' + caption +
            r'}\begin{tabular}{' + columns + r'}\toprule ' + header + r'\\\midrule')


def main():
    exact_rows, timing_rows, weights_rows, update_rows, cache_rows = [], [], [], [], []
    facts = {'controls': [], 'maintenance': []}
    for model, dataset in GROUPS:
        stem = model + '_' + dataset
        control = pd.read_csv(BASE / 'extended' / (stem + '_controls.csv'))
        timing = pd.read_csv(BASE / 'extended' / (stem + '_runtime.csv'))
        sensitivity = pd.read_csv(BASE / 'extended' / (stem + '_sensitivity.csv'))
        label = model.replace('TemporalComplEx', 'CTF') + '/' + dataset
        supported = control[control.supported.astype(bool)]
        assert (supported.winner_only1 <= supported.robust1).all()
        assert len(control) == 5000 and len(timing) == 96
        row = {'model': model, 'dataset': dataset, 'queries': len(control),
               'supported_queries': len(supported),
               'exact_capacity_supported': float(supported.robust1.mean()),
               'conservative_capacity_supported': float(supported.winner_only1.mean()),
               'fixed_denominator_disagreement_all': float(
                   (control.fixed_denominator_robust1 != control.robust1).mean())}
        rates = []
        for n in (8, 32, 64):
            passed = supported[f'probe{n}_safe1'].astype(bool)
            rate = float((~supported.loc[passed, 'robust1'].astype(bool)).mean())
            row[f'probe{n}_passed'] = int(passed.sum())
            row[f'probe{n}_false_safe'] = rate
            rates.append(rate)
        exact_rows.append(label + ' & ' + str(len(supported)) + ' & ' + ' & '.join(
            f'{v * 100:.2f}' for v in [row['exact_capacity_supported'],
                                      row['conservative_capacity_supported'], *rates]) + r'\\')
        row['runtime'] = {}
        for scope, frame in (('All', timing), ('Nonempty', timing[timing.supported.astype(bool)])):
            values = {'queries': int(frame.query_id.nunique()),
                      'sparse_median_us': float(frame.sparse_us.median()),
                      'dense_median_us': float(frame.dense_us.median()),
                      'paired_speedup_median': float((frame.dense_us / frame.sparse_us).median()),
                      'competitors_median': float(frame.competitors.median()),
                      'candidates': int(frame.candidates.iloc[0])}
            row['runtime'][scope] = values
            timing_rows.append(label + ' & ' + scope + ' & ' + str(values['queries']) + ' & ' +
                               ' & '.join(f'{values[k]:.2f}' for k in
                               ('sparse_median_us', 'dense_median_us', 'paired_speedup_median')) + r'\\')
        row['weights'] = []
        for weight, frame in sensitivity.groupby('weight'):
            values = {'weight': float(weight),
                      'mrr': float(frame.reciprocal_sum.sum() / frame.answer_count.sum()),
                      'accuracy': float(frame.correct.mean()),
                      'robust1': float(frame.robust1.mean()),
                      'robust2': float(frame.robust2.mean()),
                      'changed_from_neural': float(frame.changed_from_neural.mean())}
            row['weights'].append(values)
            weights_rows.append(label + f' & {weight:g} & ' + ' & '.join(
                f'{values[k] * 100:.2f}' for k in
                ('mrr', 'accuracy', 'robust1', 'robust2', 'changed_from_neural')) + r'\\')
        facts['controls'].append(row)

    for dataset in ('ICEWS14', 'ICEWS05-15', 'GDELT'):
        trials = pd.read_csv(BASE / 'updates' / (dataset + '_trials.csv'))
        metadata = json.loads((BASE / 'updates' / (dataset + '_metadata.json')).read_text())
        assert len(trials) == 80 and metadata['oracle_discrete_mismatches'] == 0
        row = {'dataset': dataset, 'metadata': metadata, 'batches': [],
               'max_abs_local_full_error': float(trials.max_abs_error.max())}
        for batch, frame in trials.groupby('batch'):
            assert len(frame) == 20
            values = {'batch': int(batch),
                      'full_median_ms': float(frame.full_elapsed_ms.median()),
                      'local_median_ms': float(frame.incremental_elapsed_ms.median()),
                      'paired_speedup_median': float((frame.full_elapsed_ms / frame.incremental_elapsed_ms).median()),
                      'affected_median': float(frame.incremental_refreshed.median()),
                      'affected_max': int(frame.incremental_refreshed.max()),
                      'zero_affected_fraction': float((frame.incremental_refreshed == 0).mean())}
            row['batches'].append(values)
            update_rows.append(dataset + f' & {batch} & ' + ' & '.join(
                f'{values[k]:.2f}' for k in ('full_median_ms', 'local_median_ms', 'paired_speedup_median')) +
                f" & {values['affected_median']:g} & {100 * values['zero_affected_fraction']:.0f}" + r'\\')
        error = metadata['oracle_max_abs_error']
        cache_rows.append(dataset + ' & ' + f"{metadata['events']:,}" + ' & ' + ' & '.join(
            f'{metadata[k]:.2f}' for k in ('build_seconds', 'cache_numeric_arrays_mb', 'one_cache_rss_mb')) +
            (' & 0' if error == 0 else f' & {error:.2g}') + r'\\')
        facts['maintenance'].append(row)

    text = [r'\section{Analytical controls and computation}',
            table_start('One-window controls on nonempty-memory test queries, seed 0. Exact and bound columns are certified percentages. Probe columns give the percentage of passed queries that fail the exact certificate, not the proportion of all queries.',
                        'lrrrrrr', r'Group & $n$ & Exact & Bound & 8 probes & 32 probes & 64 probes'),
            *exact_rows, r'\bottomrule\end{tabular}\end{table}',
            table_start('Certificate microbenchmark. Times are microseconds; speedup is the median paired dense/sparse ratio over three repetitions per query. The sparse implementation returns budgets, radius and witnesses; the dense control returns budget indicators. Empty-memory queries are retained in the All stratum.',
                        'llrrrr', r'Group & Memory & Queries & Sparse & Dense & Speedup'),
            *timing_rows, r'\bottomrule\end{tabular}\end{table}',
            r'\subsection{Complete evidence-weight response}',
            r'All prespecified weights are reported for seed 0, with temperature held at its validation-selected value. Values other than $\lambda$ are percentages. Changed denotes disagreement with the neural-only answer.',
            r'\small\begin{longtable}{lrrrrrr}\caption{Prediction and stability over the evidence-weight grid.}\\\toprule Group & $\lambda$ & MRR & Accuracy & Stable 1 & Stable 2 & Changed\\\midrule\endhead',
            *weights_rows, r'\bottomrule\end{longtable}\normalsize',
            r'\clearpage\section{Complete maintenance measurements}',
            r'The timing table contains 20 transactions per row, comprising ten deletion/restoration pairs. Affected is the median number of registered queries refreshed locally out of 5,000. Zero is the percentage of sampled transactions affecting no registered query. Times include event writes and refresh, and speedup is the median paired full/local ratio.',
            r'\small\begin{longtable}{lrrrrrr}\caption{Full and local refresh for every transaction size.}\\\toprule Dataset & Events & Full (ms) & Local (ms) & Speedup & Affected & Zero (\%)\\\midrule\endhead',
            *update_rows, r'\bottomrule\end{longtable}\normalsize',
            table_start('One-time cache construction and independent raw-event rebuild checks. Array memory is MiB for probabilities, outputs, object/window indices, masses and multiplicities; it excludes Python containers and the reverse index. RSS is the measured process-memory increment in MiB. The final column is the largest absolute difference across the 128 rebuilt queries; discrete outputs agree exactly.',
                        'lrrrrr', r'Dataset & History entries & Build (s) & Arrays & RSS & Rebuild error'),
            *cache_rows, r'\bottomrule\end{tabular}\end{table}',
            r'\subsection{Fixed-denominator comparison}',
            r'The historical fixed-denominator certificate is evaluated on the same unedited score. Its one-window decision differs from the renormalized certificate for ' +
            ', '.join(f"{100*r['fixed_denominator_disagreement_all']:.2f}\\% ({r['model'].replace('TemporalComplEx','CTF')}, {r['dataset']})" for r in facts['controls']) +
            r' of all test queries. These decisions refer to different edited-score definitions; the renormalized definition is the one used for every primary result.']
    (OUT / 'generated_control_tables.tex').write_text('\n'.join(text), encoding='utf-8')
    (BASE / 'control_facts.json').write_text(json.dumps(facts, indent=2), encoding='utf-8')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
