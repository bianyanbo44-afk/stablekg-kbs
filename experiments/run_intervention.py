"""Evidence-intervention experiment for stability certificates.

For each time step, remove one observed report at a time and record whether
the signed conclusion changes. The certificate is computed before the
intervention, so this evaluates prediction rather than fitting to the attack.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_benchmark as bench


def run(seed: int) -> dict[str, float]:
    stream = bench.DynamicKnowledgeStream(seed=seed)
    certs: list[float] = []
    flips: list[float] = []
    post_drift: list[float] = []
    work_full: list[float] = []
    work_inc: list[float] = []
    for t in range(stream.steps):
        stream.step(t)
        reasoner = bench.Reasoner(stream, "stability_aware")
        logits, stability = reasoner.infer(t)
        for fact in range(stream.n_base):
            observations = stream.evidence[fact]
            if len(observations) < 2 or abs(logits[fact]) < 1.35:
                continue
            original = logits[fact] >= 0
            changed = 0
            for index in range(len(observations)):
                edited = observations[:index] + observations[index + 1 :]
                old = stream.evidence[fact]
                stream.evidence[fact] = edited
                edited_logit, _ = reasoner._base_state(fact, t)
                changed += int((edited_logit >= 0) != original)
                stream.evidence[fact] = old
            certs.append(float(stability[fact]))
            flips.append(float(changed > 0))
            post_drift.append(float(t % stream.drift_period == 0 and t > 0))
        work_full.append(float(stream.n_nodes))
        work_inc.append(float(reasoner.recomputation_work(stream.changed_bases)))
    cert = np.asarray(certs)
    flip = np.asarray(flips)
    if len(cert) == 0:
        return {"seed": float(seed), "n_interventions": 0.0}
    low = cert < 0.10
    high = cert >= 0.10
    return {
        "seed": float(seed),
        "n_interventions": float(len(cert)),
        "flip_rate": float(flip.mean()),
        "low_certificate_flip_rate": float(flip[low].mean()) if low.any() else 0.0,
        "high_certificate_flip_rate": float(flip[high].mean()) if high.any() else 0.0,
        "certificate_flip_corr": float(np.corrcoef(cert, 1.0 - flip)[0, 1]) if len(cert) > 1 else 0.0,
        "full_nodes": float(np.mean(work_full)),
        "incremental_nodes": float(np.mean(work_inc)),
        "node_reduction": float(1.0 - np.mean(work_inc) / np.mean(work_full)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("intervention_results.csv"))
    args = parser.parse_args()
    rows = [run(seed) for seed in range(args.seeds)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("flip_rate", "low_certificate_flip_rate", "high_certificate_flip_rate", "certificate_flip_corr", "node_reduction"):
        values = [row[key] for row in rows if key in row]
        print(f"{key}: {np.mean(values):.4f}")


if __name__ == "__main__":
    main()
