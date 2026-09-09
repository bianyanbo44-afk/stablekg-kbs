"""Reproducible public temporal-KG evaluation.

The public ICEWS/GDELT mirror stores temporal knowledge-graph queries in JSONL
files. This script evaluates the point query ``Pe`` and keeps the protocol
strictly split-aware: train records provide evidence, validation records choose
operating thresholds, and test records are evaluated once with those frozen
thresholds. Ranking is filtered over the complete entity vocabulary; selective
prediction uses a confidence-only baseline or the proposed confidence-stability
dual score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Query:
    subject: int
    relation: int
    timestamp: int
    answers: tuple[int, ...]

    @property
    def key(self) -> tuple[int, int, int]:
        return self.subject, self.relation, self.timestamp


def iter_pe(path: Path, limit: int | None = None) -> Iterator[Query]:
    """Stream Pe records from a published JSONL split."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("query_name") != "Pe":
                continue
            query = row.get("query", [])
            answers = tuple(dict.fromkeys(int(x) for x in row.get("answer", [])))
            if len(query) != 3 or not answers:
                continue
            yield Query(int(query[0]), int(query[1]), int(query[2]), answers)
            count += 1
            if limit is not None and count >= limit:
                return


def load_events(train_path: Path) -> tuple[DefaultDict[tuple[int, int], list[tuple[int, int]]], int]:
    """Index training facts by subject and relation."""
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    max_entity = -1
    for query in iter_pe(train_path):
        for answer in query.answers:
            index[(query.subject, query.relation)].append((query.timestamp, answer))
            max_entity = max(max_entity, query.subject, answer)
    return index, max_entity + 1


def load_known(paths: Iterable[Path]) -> dict[tuple[int, int, int], set[int]]:
    """Collect known positives for filtered ranking from all published splits."""
    known: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for path in paths:
        for query in iter_pe(path):
            known[query.key].update(query.answers)
    return known


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _event_scores(
    query: Query,
    events: Sequence[tuple[int, int]],
    mode: str,
    half_life: float,
) -> tuple[dict[int, float], dict[int, Counter[int]]]:
    scores: dict[int, float] = defaultdict(float)
    bins: dict[int, Counter[int]] = defaultdict(Counter)
    for event_time, entity in events:
        # Causal replay: evidence published after the query time cannot affect
        # the decision.  The previous implementation clipped negative ages to
        # zero, which silently leaked future training facts into past queries.
        if event_time > query.timestamp:
            continue
        age = max(0, query.timestamp - event_time)
        weight = 1.0 if mode == "static" else math.exp(-math.log(2.0) * age / half_life)
        scores[entity] += weight
        bins[entity][event_time // 7] += weight
    return scores, bins


def _certificate(winner_score: float, runner_score: float, evidence: Counter[int]) -> float:
    """Return the evidence fraction that survives the cheapest sign-flip attack."""
    if winner_score <= 0:
        return 0.0
    margin = max(0.0, winner_score - runner_score)
    if margin <= 0 or not evidence:
        return 0.0
    removed = 0.0
    for mass in sorted(evidence.values(), reverse=True):
        removed += float(mass)
        if removed >= margin:
            return float(np.clip(1.0 - removed / (winner_score + 1e-12), 0.0, 1.0))
    return 1.0


def evaluate_queries(
    queries: Sequence[Query],
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]],
    known: Mapping[tuple[int, int, int], set[int]],
    n_entities: int,
    mode: str,
    half_life: float = 35.0,
) -> dict[str, object]:
    """Evaluate complete-entity filtered ranking and per-query decisions."""
    rows: list[dict[str, float]] = []
    reciprocal_ranks: list[float] = []
    hits1: list[float] = []
    hits3: list[float] = []
    hits10: list[float] = []
    for query in queries:
        scores, bins = _event_scores(query, index.get((query.subject, query.relation), []), mode, half_life)
        winner = min(range(n_entities), key=lambda entity: (-scores.get(entity, 0.0), entity))
        winner_score = float(scores.get(winner, 0.0))
        runner_score = max((value for entity, value in scores.items() if entity != winner), default=0.0)
        margin = winner_score - runner_score
        confidence = sigmoid(margin)
        stability = _certificate(winner_score, runner_score, bins.get(winner, Counter()))
        dual_score = confidence * (0.5 + 0.5 * stability)
        answer_set = set(query.answers)
        correct = float(winner in answer_set)

        filtered_other = set(known.get(query.key, set())) - answer_set
        best_rank = float("inf")
        for answer in answer_set:
            answer_score = float(scores.get(answer, 0.0))
            greater = sum(value > answer_score for entity, value in scores.items() if entity not in filtered_other and entity != answer)
            equal_seen = sum(value == answer_score for entity, value in scores.items() if entity not in filtered_other and entity != answer)
            zero_entities = n_entities - len(scores) - len(filtered_other - set(scores))
            if answer_score == 0.0:
                equal_seen += max(0, zero_entities)
            rank = 1.0 + greater + 0.5 * equal_seen
            best_rank = min(best_rank, rank)
        if not math.isfinite(best_rank):
            continue
        reciprocal_ranks.append(1.0 / best_rank)
        hits1.append(float(best_rank <= 1.0))
        hits3.append(float(best_rank <= 3.0))
        hits10.append(float(best_rank <= 10.0))
        rows.append({
            "correct": correct,
            "confidence": confidence,
            "stability": stability,
            "dual_score": dual_score,
            "margin": margin,
            "winner": float(winner),
            "rank": best_rank,
            "evidence_mass": winner_score,
        })
    return {
        "rows": rows,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hits1": float(np.mean(hits1)) if hits1 else 0.0,
        "hits3": float(np.mean(hits3)) if hits3 else 0.0,
        "hits10": float(np.mean(hits10)) if hits10 else 0.0,
    }


def ece(rows: Sequence[dict[str, float]], score_key: str = "confidence", bins: int = 10) -> float:
    if not rows:
        return 0.0
    prob = np.asarray([row[score_key] for row in rows], dtype=float)
    truth = np.asarray([row["correct"] for row in rows], dtype=float)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if mask.any():
            value += float(mask.mean()) * abs(float(prob[mask].mean()) - float(truth[mask].mean()))
    return value


def aurc(rows: Sequence[dict[str, float]], score_key: str) -> float:
    if not rows:
        return 0.0
    order = np.argsort(-np.asarray([row[score_key] for row in rows], dtype=float), kind="mergesort")
    errors = np.asarray([1.0 - row["correct"] for row in rows], dtype=float)[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def choose_threshold(rows: Sequence[dict[str, float]], score_key: str, target: float) -> float:
    """Choose a validation threshold with the closest attainable coverage."""
    if not rows:
        return 1.0
    scores = sorted({float(row[score_key]) for row in rows}, reverse=True)
    candidates = [scores[0] + 1e-9] + scores
    best_key = (float("inf"), float("inf"), float("inf"))
    best_threshold = 1.0
    for threshold in candidates:
        coverage = sum(row[score_key] >= threshold for row in rows) / len(rows)
        key = (abs(coverage - target), -coverage, -threshold)
        if key < best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def selective_summary(rows: Sequence[dict[str, float]], score_key: str, threshold: float) -> dict[str, float]:
    accepted = [row for row in rows if row[score_key] >= threshold]
    coverage = len(accepted) / len(rows) if rows else 0.0
    risk = float(np.mean([1.0 - row["correct"] for row in accepted])) if accepted else 0.0
    brier = float(np.mean([(row[score_key] - row["correct"]) ** 2 for row in accepted])) if accepted else 0.0
    return {"risk": risk, "coverage": coverage, "brier": brier, "n_accepted": float(len(accepted))}


def run_dataset(name: str, root: Path, cap: int, seeds: int) -> list[dict[str, float]]:
    train, valid, test = (root / "train.jsonl", root / "valid.jsonl", root / "test.jsonl")
    started = time.time()
    index, n_entities = load_events(train)
    valid_queries = list(iter_pe(valid, cap))
    test_queries = list(iter_pe(test, cap))
    # For a causal test query, filtering may use all labelled positives for the
    # same query key, but evidence scoring above remains training-only and
    # time-causal.  Keeping this distinction explicit prevents label leakage.
    known = load_known((train, valid, test))
    rows: list[dict[str, float]] = []
    targets = (0.20, 0.40, 0.60, 0.80)
    for seed in range(seeds):
        val_static = evaluate_queries(valid_queries, index, known, n_entities, "static")
        val_decay = evaluate_queries(valid_queries, index, known, n_entities, "decay")
        test_static = evaluate_queries(test_queries, index, known, n_entities, "static")
        test_decay = evaluate_queries(test_queries, index, known, n_entities, "decay")
        row: dict[str, float] = {
            "dataset": name,
            "seed": float(seed),
            "n_entities": float(n_entities),
            "n_train_events": float(sum(len(events) for events in index.values())),
            "n_validation": float(len(val_static["rows"])),
            "n_test": float(len(test_static["rows"])),
            "static_mrr": float(test_static["mrr"]),
            "static_hits1": float(test_static["hits1"]),
            "static_hits3": float(test_static["hits3"]),
            "static_hits10": float(test_static["hits10"]),
            "decay_mrr": float(test_decay["mrr"]),
            "decay_hits1": float(test_decay["hits1"]),
            "decay_hits3": float(test_decay["hits3"]),
            "decay_hits10": float(test_decay["hits10"]),
            "static_ece": ece(test_static["rows"]),
            "decay_ece": ece(test_decay["rows"]),
            "static_aurc": aurc(test_static["rows"], "confidence"),
            "decay_aurc": aurc(test_decay["rows"], "confidence"),
            "stability_aurc": aurc(test_decay["rows"], "dual_score"),
            "elapsed_seconds": float(time.time() - started),
        }
        for target in targets:
            label = f"{target:.2f}"
            thresholds = (
                ("static", test_static["rows"], "confidence", choose_threshold(val_static["rows"], "confidence", target)),
                ("decay", test_decay["rows"], "confidence", choose_threshold(val_decay["rows"], "confidence", target)),
                ("stability", test_decay["rows"], "dual_score", choose_threshold(val_decay["rows"], "dual_score", target)),
            )
            for prefix, rows_eval, key, threshold in thresholds:
                summary = selective_summary(rows_eval, key, threshold)
                row[f"{prefix}_risk@{label}"] = summary["risk"]
                row[f"{prefix}_coverage@{label}"] = summary["coverage"]
                row[f"{prefix}_brier@{label}"] = summary["brier"]
                row[f"{prefix}_threshold@{label}"] = threshold
        rows.append(row)
        print(f"{name} seed={seed} train={row['n_train_events']:.0f} test={row['n_test']:.0f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run_dataset(args.dataset, args.data_root, args.cap, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("static_risk@0.40", "decay_risk@0.40", "stability_risk@0.40", "static_coverage@0.40", "decay_coverage@0.40", "stability_coverage@0.40"):
        values = [row[key] for row in rows]
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{key}: {statistics.mean(values):.4f} +/- {spread:.4f}")


if __name__ == "__main__":
    main()
