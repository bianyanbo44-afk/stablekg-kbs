"""Rebuild the corrected evidence release without overwriting the prior run."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPECS = {"ICEWS14": ("ICEWS14_all", 7128, 2048), "ICEWS05-15": ("ICEWS05_15_all", 10488, 4096), "GDELT": ("GDELT_all", 500, 2048)}
p = argparse.ArgumentParser()
p.add_argument("dataset", choices=SPECS)
a = p.parse_args()
folder, entities, batch = SPECS[a.dataset]
common = ["--data-root", f"data/raw/{folder}", "--dataset", a.dataset, "--entity-count", str(entities)]
jobs = [
    ["experiments/stability_benchmark.py", *common, "--cap", "5000", "--seeds", "5", "--output-dir", "results_final_v5"],
    ["experiments/public_interventions_v2.py", *common, "--cap", "5000", "--output-dir", "results_interventions_v5"],
    ["experiments/matched_controls.py", *common, "--correct-prefix"],
]
if a.dataset != "GDELT":
    jobs.extend([
        ["experiments/chronological_benchmark.py", *common, "--cap", "5000", "--output-dir", "results_chronological_v5"],
        ["experiments/neural_backbone.py", *common, "--cap", "5000", "--seeds", "3", "--epochs", "30", "--dim", "96", "--negatives", "32", "--batch-size", str(batch), "--checkpoint-dir", "results_neural", "--output-dir", "results_neural_v5"],
    ])
logs = ROOT / "results_final_v5" / "logs"
logs.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
for i, job in enumerate(jobs):
    print(f"{a.dataset}: {job[0]}", flush=True)
    with (logs / f"{a.dataset}_{i}.log").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, *job], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
print(f"{a.dataset}: corrected release complete", flush=True)
