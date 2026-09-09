"""Fixed-protocol evidence interventions on public temporal-KG queries.

For every evaluated query, the unedited winner and certificate are computed
first.  Three deterministic edits are then applied independently: removal of
the strongest single winner event, removal of the strongest seven-unit winner
window, and injection of one current-time event for the runner-up.  Labels are
consulted only to report prediction correctness, never to choose the edit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

import stability_benchmark as bench


def winner(events: Sequence[tuple[int, int]], timestamp: int, n_entities: int, half_life: float = 35.0) -> int:
    mass: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for event_time, entity in events:
        if event_time > timestamp:
            break
        weight = math.exp(-math.log(2.0) * (timestamp - event_time) / half_life)
        mass[entity] += weight; counts[entity] += 1
    return min(range(n_entities), key=lambda entity: (-mass.get(entity, 0.0), -counts.get(entity, 0), entity))


def run(root: Path, cap: int, entity_count: int | None) -> list[dict[str, object]]:
    index, inferred = bench.load_events(root / "train.jsonl")
    n_entities = int(entity_count or inferred)
    queries = bench.evenly_sample(list(bench.iter_pe(root / "test.jsonl")), cap)
    rows: list[dict[str, object]] = []
    for query_id, query in enumerate(queries):
        events = index.get((query.subject, query.relation), [])
        feat, mass, counts = bench.winner_features(query, events, n_entities, "decay", 35.0)
        original = int(feat["winner"])
        if feat["mass"] <= 0.0 or counts.get(original, 0) == 0:
            continue
        runner = min((e for e in range(n_entities) if e != original), key=lambda e: (-mass.get(e, 0.0), -counts.get(e, 0), e))
        causal_winner_events = [(t, e) for t, e in events if t <= query.timestamp and e == original]

        # Strongest event is the most recent under exponential decay.  Ties
        # follow source order, which is deterministic in the archive.
        strongest = max(causal_winner_events, key=lambda pair: pair[0])
        removed = False
        event_edit = []
        for pair in events:
            if not removed and pair == strongest:
                removed = True
                continue
            event_edit.append(pair)
        event_winner = winner(event_edit, query.timestamp, n_entities)

        window_mass: Counter[int] = Counter()
        for event_time, _ in causal_winner_events:
            window_mass[event_time // 7] += math.exp(-math.log(2.0) * (query.timestamp - event_time) / 35.0)
        strongest_window = min(window_mass, key=lambda bucket: (-window_mass[bucket], bucket))
        window_edit = [(t, e) for t, e in events if not (e == original and t <= query.timestamp and t // 7 == strongest_window)]
        window_winner = winner(window_edit, query.timestamp, n_entities)

        counter_edit = list(events) + [(query.timestamp, runner)]
        counter_edit.sort(key=lambda pair: pair[0])
        counter_winner = winner(counter_edit, query.timestamp, n_entities)

        rows.append({
            "query_id": query_id, "subject": query.subject, "relation": query.relation, "timestamp": query.timestamp,
            "winner": original, "runner": runner, "correct": int(original in set(query.answers)),
            "certificate": feat["certificate"], "fragility_cost": feat["fragility_cost"],
            "counter_evidence_cost": feat["counter_evidence_cost"], "diversity": feat["diversity"],
            "support": feat["support"], "event_flip": int(event_winner != original),
            "window_flip": int(window_winner != original), "counter_flip": int(counter_winner != original),
        })
    return rows


def summarize(dataset: str, rows: Sequence[dict[str, object]]) -> dict[str, object]:
    cert = np.asarray([float(r["certificate"]) for r in rows])
    q25, q75 = np.quantile(cert, [0.25, 0.75]) if len(cert) else (0.0, 0.0)
    out: dict[str, object] = {"dataset": dataset, "n_queries": len(rows), "certificate_q25": float(q25), "certificate_q75": float(q75)}
    for edit in ("event", "window", "counter"):
        flip = np.asarray([float(r[f"{edit}_flip"]) for r in rows])
        out[f"{edit}_flip_rate"] = float(flip.mean()) if len(flip) else 0.0
        out[f"low_certificate_{edit}_flip"] = float(flip[cert <= q25].mean()) if np.any(cert <= q25) else 0.0
        out[f"high_certificate_{edit}_flip"] = float(flip[cert >= q75].mean()) if np.any(cert >= q75) else 0.0
        rho, p = spearmanr(cert, 1.0 - flip) if len(flip) > 1 else (0.0, 1.0)
        out[f"certificate_{edit}_survival_spearman"] = float(rho) if np.isfinite(rho) else 0.0
        out[f"certificate_{edit}_survival_p"] = float(p) if np.isfinite(p) else 1.0
    out["winner_accuracy"] = float(np.mean([r["correct"] for r in rows])) if rows else 0.0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--cap", type=int, default=5000)
    p.add_argument("--entity-count", type=int)
    p.add_argument("--output-dir", type=Path, default=Path("results_final"))
    args = p.parse_args()
    rows = run(args.data_root, args.cap, args.entity_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.dataset}_interventions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = summarize(args.dataset, rows)
    with (args.output_dir / f"{args.dataset}_interventions.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
