"""Check final release consistency across frozen runs, references and figures."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-only', action='store_true')
    args = parser.parse_args()
    base = ROOT / 'results_nc/evaluation_renormalized'
    manifest_path = base / 'frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 12
    rows, budget_checks, witnesses = 0, 0, {}
    for record in manifest:
        run = base / record['model'] / record['dataset'] / f"seed{record['seed']}"
        config_path = run / 'frozen_config.json'
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == record['sha256']
        config = json.loads(config_path.read_text())
        assert config['protocol_version'] == 'renormalized-novel-fact-v2-strict-validation'
        df = pd.read_csv(run / 'test_queries.csv')
        assert len(df) == 5000 and df.query_id.nunique() == 5000
        for k in (1, 2, 3):
            robust = df[f'robust{k}'].to_numpy(dtype=bool)
            flips = df[f'worst_flip{k}'].to_numpy(dtype=bool)
            assert np.all(robust != flips), (run, k)
            budget_checks += len(df)
        if record['model'] == 'TeRDy':
            exported = json.loads((ROOT / 'results_nc/TeRDy' / record['dataset'] /
                                   f"seed{record['seed']}" / 'test_export.json').read_text())
            assert exported['frozen_manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        witnesses[str(run.relative_to(ROOT))] = {'queries': len(df), 'violations': 0, 'mismatches': 0}
        rows += len(df)
    references = json.loads((ROOT / 'literature/neurocomputing_20260922/revision_verification.json').read_text())
    assert len(references) == 31 and all(r['status'] == 'found' and r['year'] in (2025, 2026) for r in references)
    cited = set()
    for path in (ROOT / 'manuscript/neurocomputing').glob('*.tex'):
        for group in re.findall(r'\\cite(?:p|t)?\{([^}]+)\}', path.read_text(encoding='utf-8')):
            cited.update(group.split(','))
    assert cited == {r['key'] for r in references}, cited ^ {r['key'] for r in references}
    if not args.core_only:
        for dataset in ('ICEWS14', 'ICEWS05-15', 'GDELT'):
            meta = json.loads((ROOT / 'results_nc/updates' / f'{dataset}_metadata.json').read_text())
            trials = pd.read_csv(ROOT / 'results_nc/updates' / f'{dataset}_trials.csv')
            assert len(trials) == 80 and meta['oracle_discrete_mismatches'] == 0
            assert trials.max_abs_error.max() < 2e-11 and meta['oracle_max_abs_error'] < 1e-9
        for stem in ('fig01_interface', 'fig02_prediction', 'fig03_selection', 'fig04_diagnostics',
                     'fig05_controls', 'fig06_maintenance'):
            for extension in ('pdf', 'svg', 'png'):
                assert (ROOT / 'figures/neurocomputing' / f'{stem}.{extension}').exists()
            report = json.loads((ROOT / 'figures/neurocomputing/qa' / f'{stem}.alignment.json').read_text())
            assert report['verdict'] == 'PASS', stem
            collision = json.loads((ROOT / 'figures/neurocomputing/qa' / f'{stem}.collisions.json').read_text())
            assert collision['summary']['fail'] == 0, stem
            fonts = json.loads((ROOT / 'figures/neurocomputing/qa' / f'{stem}.fonts.txt').read_text())
            assert fonts['auditable'] and fonts['below_minimum_count'] == 0, stem
            assert fonts['minimum_found_pt'] >= 5, stem
    report = {'scope': 'primary results' if args.core_only else 'final release',
              'fitted_runs': len(manifest), 'query_seed_rows': rows,
              'query_seed_budget_checks': budget_checks, 'certified_violations': 0,
              'certificate_witness_mismatches': 0, 'verified_recent_references': len(references),
              'runs': witnesses}
    (ROOT / 'revision' / ('CORE_CHECKS.json' if args.core_only else 'FINAL_CHECKS.json')).write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
