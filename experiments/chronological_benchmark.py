"""Leakage-free prequential evaluation on chronological temporal-KG splits.

All published Pe facts are regrouped by timestamp.  The earliest 70% of
timestamps form the initial history, the next 15% form validation, and the
latest 15% form test.  At each validation or test timestamp, every query is
predicted before any fact from that timestamp is added to the evidence index.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, Sequence

import numpy as np

import stability_benchmark as bench


def merge_queries(paths: Iterable[Path]) -> list[bench.Query]:
    answers: DefaultDict[tuple[int, int, int], set[int]] = defaultdict(set)
    for path in paths:
        for query in bench.iter_pe(path):
            answers[query.key].update(query.answers)
    return [
        bench.Query(subject, relation, timestamp, tuple(sorted(values)))
        for (subject, relation, timestamp), values in sorted(
            answers.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
        )
    ]


def chronological_split(queries: Sequence[bench.Query]) -> tuple[list[bench.Query], list[bench.Query], list[bench.Query]]:
    timestamps = sorted({query.timestamp for query in queries})
    train_end = timestamps[max(0, int(round(0.70 * len(timestamps))) - 1)]
    validation_end = timestamps[max(0, int(round(0.85 * len(timestamps))) - 1)]
    train = [query for query in queries if query.timestamp <= train_end]
    validation = [query for query in queries if train_end < query.timestamp <= validation_end]
    test = [query for query in queries if query.timestamp > validation_end]
    return train, validation, test


def build_index(queries: Sequence[bench.Query]) -> DefaultDict[tuple[int, int], list[tuple[int, int]]]:
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for query in queries:
        for answer in query.answers:
            index[(query.subject, query.relation)].append((query.timestamp, answer))
    for events in index.values():
        events.sort()
    return index


def append_timestamp(index: DefaultDict[tuple[int, int], list[tuple[int, int]]], queries: Sequence[bench.Query]) -> None:
    for query in queries:
        index[query.subject, query.relation].extend((query.timestamp, answer) for answer in query.answers)


def evaluate_prequential(
    queries: Sequence[bench.Query],
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]],
    known: dict[tuple[int, int, int], set[int]],
    n_entities: int,
    cap: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    grouped: DefaultDict[int, list[bench.Query]] = defaultdict(list)
    for query in queries:
        grouped[query.timestamp].append(query)
    selected = set(bench.evenly_sample(list(queries), cap)) if cap > 0 else set(queries)
    rows: list[dict[str, float]] = []
    ranks: list[float] = []
    for timestamp in sorted(grouped):
        current = grouped[timestamp]
        for query in current:
            if query not in selected:
                continue
            feature, mass, _ = bench.winner_features(
                query,
                index.get((query.subject, query.relation), ()),
                n_entities,
                "decay",
                35.0,
            )
            winner = int(feature["winner"])
            answers = set(query.answers)
            rank_map = bench._filtered_ranks(answers, mass, known.get(query.key, answers), n_entities)
            answer_ranks = [rank_map[answer] for answer in answers]
            ranks.extend(answer_ranks)
            rows.append(
                {
                    **feature,
                    "correct": float(winner in answers),
                    "rank": float(np.mean(answer_ranks)),
                    "subject": float(query.subject),
                    "relation": float(query.relation),
                    "timestamp": float(query.timestamp),
                }
            )
        # Prequential order is deliberate: facts at t become available only
        # after all queries at t have been scored.
        append_timestamp(index, current)
    rank_array = np.asarray(ranks, dtype=float)
    metrics = {
        "mrr": float(np.mean(1.0 / rank_array)),
        "hits1": float(np.mean(rank_array <= 1.0)),
        "hits3": float(np.mean(rank_array <= 3.0)),
        "hits10": float(np.mean(rank_array <= 10.0)),
    }
    return rows, metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = tuple(args.data_root / f"{split}.jsonl" for split in ("train", "valid", "test"))
    all_queries = merge_queries(paths)
    train, validation, test = chronological_split(all_queries)
    n_entities = int(args.entity_count or (1 + max(max(q.subject, *q.answers) for q in all_queries)))
    known: DefaultDict[tuple[int, int, int], set[int]] = defaultdict(set)
    for query in all_queries:
        known[query.key].update(query.answers)

    # The validation stream starts from the initial-history state.  Its first
    # half in time fits the calibrator; its second half selects operating points.
    index = build_index(train)
    validation_rows, validation_rank = evaluate_prequential(validation, index, known, n_entities, args.cap)
    validation_times = sorted({int(row["timestamp"]) for row in validation_rows})
    midpoint = validation_times[len(validation_times) // 2]
    calibration = [row for row in validation_rows if int(row["timestamp"]) < midpoint]
    operating = [row for row in validation_rows if int(row["timestamp"]) >= midpoint]

    # ``index`` now contains train and all validation facts.  Test continues
    # from exactly that state and retains the same predict-then-update order.
    test_rows, test_rank = evaluate_prequential(test, index, known, n_entities, args.cap)
    baseline_features = ["margin", "log_support", "recency"]
    stable_features = [
        "margin",
        "log_support",
        "recency",
        "certificate",
        "diversity",
        "fragility_cost",
        "counter_evidence_cost",
    ]
    baseline = bench.RidgeLogistic(baseline_features).fit(calibration)
    stable = bench.RidgeLogistic(stable_features).fit(calibration)
    bench.add_prediction(operating, baseline, "belief_baseline")
    bench.add_prediction(operating, stable, "belief_stability")
    bench.add_prediction(test_rows, baseline, "belief_baseline")
    bench.add_prediction(test_rows, stable, "belief_stability")

    summary: dict[str, object] = {
        "dataset": args.dataset,
        "protocol": "70/15/15 timestamps; predict all queries at t before update",
        "n_entities": n_entities,
        "train_timestamp_min": min(q.timestamp for q in train),
        "train_timestamp_max": max(q.timestamp for q in train),
        "validation_timestamp_min": min(q.timestamp for q in validation),
        "validation_timestamp_max": max(q.timestamp for q in validation),
        "test_timestamp_min": min(q.timestamp for q in test),
        "test_timestamp_max": max(q.timestamp for q in test),
        "n_train_queries": len(train),
        "n_validation_queries_available": len(validation),
        "n_test_queries_available": len(test),
        "n_calibration": len(calibration),
        "n_operating": len(operating),
        "n_test": len(test_rows),
        "validation_mrr": validation_rank["mrr"],
        **{f"test_{key}": value for key, value in test_rank.items()},
        "baseline_brier": bench.brier(test_rows, "belief_baseline"),
        "stability_brier": bench.brier(test_rows, "belief_stability"),
        "baseline_ece": bench.ece(test_rows, "belief_baseline"),
        "stability_ece": bench.ece(test_rows, "belief_stability"),
        "baseline_aurc": bench.aurc(test_rows, "belief_baseline"),
        "stability_aurc": bench.aurc(test_rows, "belief_stability"),
    }
    for target in (0.20, 0.40, 0.60, 0.80):
        label = f"{target:.2f}"
        for prefix, key in (("baseline", "belief_baseline"), ("stability", "belief_stability")):
            threshold = bench.choose_threshold(operating, key, target)
            result = bench.selective(test_rows, key, threshold)
            summary[f"{prefix}_threshold@{label}"] = threshold
            summary[f"{prefix}_risk@{label}"] = result["risk"]
            summary[f"{prefix}_coverage@{label}"] = result["coverage"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bench.write_rows(args.output_dir / f"{args.dataset}_chronological_predictions.csv", test_rows)
    (args.output_dir / f"{args.dataset}_chronological_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--entity-count", type=int)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("results_chronological"))
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
