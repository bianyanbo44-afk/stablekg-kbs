"""Reproducible experiments for stability certificates on temporal KGs.

The benchmark is deliberately model-agnostic.  A causal evidence scorer turns
training events into a ranked candidate list; a validation-only calibrator then
estimates the probability that the selected candidate is correct.  StableKG
adds two evidence diagnostics: the mass removed by a descending-window attack
and temporal diversity. The attack also yields an exact minimum window count
for erasing a positive margin under winner-only window deletion.

The script writes both aggregate metrics and query-level source data.  It never
uses test labels to fit the scorer, half-life, calibrator or operating
thresholds. Validation labels fit the calibrator. ``--cap`` is a deterministic,
evenly spaced sample after the
split has been parsed; ``0`` evaluates every Pe query.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from bisect import bisect_right
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


def iter_pe(path: Path) -> Iterator[Query]:
    """Yield the published point queries, preserving their source order."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("query_name") != "Pe":
                continue
            query = row.get("query", [])
            answers = tuple(dict.fromkeys(int(x) for x in row.get("answer", [])))
            if len(query) == 3 and answers:
                yield Query(int(query[0]), int(query[1]), int(query[2]), answers)


def evenly_sample(queries: Sequence[Query], cap: int) -> list[Query]:
    """Select a deterministic spread across a split without using labels."""
    if cap <= 0 or len(queries) <= cap:
        return list(queries)
    # np.linspace gives coverage over the complete source order and avoids a
    # convenient early-time slice.  The source order is fixed by the archive.
    indices = np.linspace(0, len(queries) - 1, cap, dtype=int)
    return [queries[int(i)] for i in indices]


def load_events(train_path: Path) -> tuple[DefaultDict[tuple[int, int], list[tuple[int, int]]], int]:
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    max_entity = -1
    for query in iter_pe(train_path):
        for answer in query.answers:
            index[(query.subject, query.relation)].append((query.timestamp, answer))
            max_entity = max(max_entity, query.subject, answer)
    for events in index.values():
        events.sort(key=lambda pair: pair[0])
    return index, max_entity + 1


def load_known(paths: Iterable[Path]) -> dict[tuple[int, int, int], set[int]]:
    known: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for path in paths:
        for query in iter_pe(path):
            known[query.key].update(query.answers)
    return known


def load_known_for_queries(
    paths: Iterable[Path],
    queries: Sequence[Query],
    index: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
) -> dict[tuple[int, int, int], set[int]]:
    """Build the filtered-answer map only for evaluated query keys.

    The full GDELT mirror contains more than twenty million JSONL rows.  A
    full three-split scan for every smoke run is unnecessary because filtered
    ranking only needs answers at the evaluated ``(subject, relation, time)``
    keys.  Training answers are recovered from the already-built event index;
    validation and test files are scanned once and discarded.
    """
    keys = {query.key for query in queries}
    known: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for query in queries:
        # Recover all training objects observed at this exact time.
        for event_time, entity in index.get((query.subject, query.relation), ()):
            if event_time == query.timestamp:
                known[query.key].add(int(entity))
        known[query.key].update(query.answers)
    for path in paths:
        if path is None or not path.exists():
            continue
        for query in iter_pe(path):
            if query.key in keys:
                known[query.key].update(query.answers)
    return known


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(x, -40.0, 40.0))))


def causal_features(
    query: Query,
    events: Sequence[tuple[int, int]],
    half_life: float,
    mode: str,
) -> tuple[dict[int, float], dict[int, Counter[int]], dict[int, int], dict[int, int]]:
    """Return weighted causal mass, seven-timestamp-unit bins and last times."""
    # Events are sorted once at ingestion.  The bisect is material for GDELT,
    # where a single subject-relation stream can contain many observations.
    # Tuple ordering makes this an exact upper bound over the time coordinate
    # without allocating a second timestamp list for every query.
    cutoff = bisect_right(events, (query.timestamp, 10**18))
    mass: dict[int, float] = defaultdict(float)
    bins: dict[int, Counter[int]] = defaultdict(Counter)
    counts: dict[int, int] = defaultdict(int)
    last: dict[int, int] = {}
    for event_time, entity in events[:cutoff]:
        age = query.timestamp - event_time
        weight = 1.0 if mode == "static" else math.exp(-math.log(2.0) * age / half_life)
        mass[entity] += weight
        bins[entity][event_time // 7] += weight
        counts[entity] += 1
        last[entity] = event_time
    return mass, bins, counts, last


def _filtered_rank(
    answer: int,
    mass: Mapping[int, float],
    counts: Mapping[int, int],
    known_other: set[int],
    n_entities: int,
) -> float:
    """Average filtered rank with score-only ties, including the zero tail."""
    value = float(mass.get(answer, 0.0))
    # This helper is retained for unit tests and small callers.  The evaluator
    # uses ``_filtered_ranks`` below, which groups equal scores once per query.
    observed = set(mass)
    greater = sum(float(score) > value for entity, score in mass.items() if entity != answer and entity not in known_other)
    equal = sum(float(score) == value for entity, score in mass.items() if entity != answer and entity not in known_other)
    filtered_unobserved = sum(1 for entity in known_other if entity not in observed)
    unobserved = max(0, n_entities - len(observed) - filtered_unobserved - (0 if answer in observed else 1))
    if value == 0.0:
        equal += unobserved
    return 1.0 + greater + 0.5 * equal


def _filtered_ranks(
    answers: set[int], mass: Mapping[int, float], known_answers: set[int], n_entities: int
) -> dict[int, float]:
    """Compute standard filtered average-tie ranks for every labelled answer.

    Each answer is ranked in its own filtered candidate set.  All other known
    answers at the same query key, including sibling answers in the current
    record, are removed before counting candidates.
    """
    return {
        answer: _filtered_rank(
            answer,
            mass,
            {},
            set(known_answers) - {answer},
            n_entities,
        )
        for answer in answers
    }


def window_deletion_certificate(bin_masses: Iterable[float], margin: float) -> tuple[int, float]:
    """Return minimum window count to erase a margin and greedy removed fraction.

    With descending positive masses v, k is the first prefix with sum >= margin.
    Any deletion of fewer than k whole winner windows leaves a positive margin.
    The fraction is the cost of that particular attack, NOT the minimum mass
    over arbitrary subsets. A zero count means there is no positive margin (or
    no usable evidence). Reaching equality erases the strict margin; it need
    not change a tie-broken winner. Competitor evidence is held fixed.
    """
    values = sorted((float(value) for value in bin_masses if value > 0.0), reverse=True)
    total = float(sum(values))
    if margin <= 0.0 or total <= 0.0:
        return 0, 0.0
    removed = 0.0
    for count, value in enumerate(values, 1):
        removed += value
        if removed >= margin:
            return count, float(np.clip(removed / total, 0.0, 1.0))
    # Accumulation order can make a total-mass margin exceed this sum by ulps.
    if math.isclose(removed, margin, rel_tol=1e-12, abs_tol=1e-15):
        return len(values), 1.0
    raise ValueError("The margin exceeds all available winner-window mass")


def winner_features(
    query: Query,
    events: Sequence[tuple[int, int]],
    n_entities: int,
    mode: str,
    half_life: float,
) -> tuple[dict[str, float], dict[int, float], dict[int, int]]:
    mass, bins, counts, last = causal_features(query, events, half_life, mode)
    ranked = sorted(range(n_entities), key=lambda e: (-mass.get(e, 0.0), -counts.get(e, 0), e))
    winner = ranked[0]
    winner_mass = float(mass.get(winner, 0.0))
    runner_mass = float(mass.get(ranked[1], 0.0)) if len(ranked) > 1 else 0.0
    margin = winner_mass - runner_mass
    support = int(counts.get(winner, 0))
    winner_bins = bins.get(winner, Counter())
    total_mass = float(sum(winner_bins.values()))
    diversity = 0.0 if total_mass <= 0 else float(1.0 - max(winner_bins.values()) / total_mass)

    # Descending-prefix attack cost, not minimum removable mass over subsets.
    # Its prefix length has the discrete guarantee proved in the manuscript.
    window_count, fragility_cost = window_deletion_certificate(winner_bins.values(), margin)
    counter_evidence_cost = float(np.clip(margin / (winner_mass + runner_mass + 1e-12), 0.0, 1.0))
    certificate = float(np.clip(0.5 * fragility_cost + 0.5 * diversity, 0.0, 1.0))
    recency = 0.0 if winner not in last else math.exp(
        -math.log(2.0) * (query.timestamp - last[winner]) / half_life
    )
    feat = {
        "winner": float(winner),
        "margin": float(margin),
        "confidence_raw": float(sigmoid(margin)),
        "support": float(support),
        "log_support": float(math.log1p(support)),
        "diversity": diversity,
        "fragility_cost": fragility_cost,
        "window_count": float(window_count),
        "counter_evidence_cost": counter_evidence_cost,
        "certificate": certificate,
        "recency": float(recency),
        "mass": winner_mass,
        "runner_mass": runner_mass,
    }
    return feat, mass, counts


def evaluate(
    queries: Sequence[Query],
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]],
    known: Mapping[tuple[int, int, int], set[int]],
    n_entities: int,
    mode: str,
    half_life: float = 35.0,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    reciprocal_ranks: list[float] = []
    all_ranks: list[float] = []
    for query in queries:
        feat, mass, counts = winner_features(
            query, index.get((query.subject, query.relation), []), n_entities, mode, half_life
        )
        answers = set(query.answers)
        winner = int(feat["winner"])
        feat["correct"] = float(winner in answers)
        known_answers = set(known.get(query.key, set())) | answers
        rank_map = _filtered_ranks(answers, mass, known_answers, n_entities)
        answer_ranks = [rank_map[answer] for answer in answers]
        # Aggregate over every labelled answer, as in standard filtered
        # temporal-KG evaluation.  The query-level source row stores the mean
        # rank only for convenient plotting; no optimistic best-answer shortcut
        # is used in MRR or Hits.
        feat["rank"] = float(np.mean(answer_ranks))
        feat["subject"] = float(query.subject)
        feat["relation"] = float(query.relation)
        feat["timestamp"] = float(query.timestamp)
        feat["answer_count"] = float(len(answers))
        reciprocal_ranks.extend(1.0 / np.asarray(answer_ranks, dtype=float))
        all_ranks.extend(answer_ranks)
        rows.append(feat)
    return rows, {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hits1": float(np.mean(np.asarray(all_ranks) <= 1.0)) if all_ranks else 0.0,
        "hits3": float(np.mean(np.asarray(all_ranks) <= 3.0)) if all_ranks else 0.0,
        "hits10": float(np.mean(np.asarray(all_ranks) <= 10.0)) if all_ranks else 0.0,
    }


class RidgeLogistic:
    """Deterministic Newton logistic calibration with a small ridge penalty."""

    def __init__(self, feature_names: Sequence[str], l2: float = 1e-2) -> None:
        self.feature_names = list(feature_names)
        self.l2 = float(l2)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weight: np.ndarray | None = None

    def _matrix(self, rows: Sequence[Mapping[str, float]], fit: bool = False) -> np.ndarray:
        x = np.asarray([[float(row[name]) for name in self.feature_names] for row in rows], dtype=float)
        if fit or self.mean is None:
            self.mean = x.mean(axis=0) if len(x) else np.zeros(len(self.feature_names))
            self.scale = x.std(axis=0) if len(x) else np.ones(len(self.feature_names))
            self.scale[self.scale < 1e-8] = 1.0
        return (x - self.mean) / self.scale

    def fit(self, rows: Sequence[Mapping[str, float]]) -> "RidgeLogistic":
        if not rows:
            self.mean = np.zeros(len(self.feature_names)); self.scale = np.ones(len(self.feature_names))
            self.weight = np.zeros(len(self.feature_names) + 1)
            return self
        x = np.c_[np.ones(len(rows)), self._matrix(rows, fit=True)]
        y = np.asarray([float(row["correct"]) for row in rows], dtype=float)
        w = np.zeros(x.shape[1], dtype=float)
        penalty = np.eye(x.shape[1], dtype=float) * self.l2
        penalty[0, 0] = 0.0
        previous = float("inf")
        for _ in range(80):
            z = np.clip(x @ w, -40.0, 40.0)
            p = 1.0 / (1.0 + np.exp(-z))
            # Negative log likelihood plus ridge, normalized by sample count.
            objective = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * w @ penalty @ w)
            hessian = (x.T * (p * (1.0 - p))) @ x / len(y) + penalty
            gradient = (x.T @ (p - y)) / len(y) + penalty @ w
            try:
                step = np.linalg.solve(hessian + 1e-8 * np.eye(hessian.shape[0]), gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            # Backtracking keeps the fit stable for near-separable validation
            # subsets while remaining deterministic.
            scale = 1.0
            for _ in range(12):
                candidate = w - scale * step
                cz = np.clip(x @ candidate, -40.0, 40.0)
                candidate_obj = float(np.mean(np.logaddexp(0.0, cz) - y * cz) + 0.5 * candidate @ penalty @ candidate)
                if candidate_obj <= objective + 1e-12:
                    w = candidate
                    objective = candidate_obj
                    break
                scale *= 0.5
            if abs(previous - objective) < 1e-9:
                break
            previous = objective
        self.weight = w
        return self

    def predict(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("fit must be called before predict")
        x = np.c_[np.ones(len(rows)), self._matrix(rows)]
        return 1.0 / (1.0 + np.exp(-np.clip(x @ self.weight, -40.0, 40.0)))


def add_prediction(rows: Sequence[dict[str, float]], model: RidgeLogistic, key: str) -> None:
    values = model.predict(rows)
    for row, value in zip(rows, values):
        row[key] = float(value)


def choose_threshold(rows: Sequence[Mapping[str, float]], key: str, target: float) -> float:
    if not rows:
        return 1.0
    values = sorted({float(row[key]) for row in rows}, reverse=True)
    candidates = [values[0] + 1e-9] + values
    best = (float("inf"), float("inf"), float("inf"), 1.0)
    for threshold in candidates:
        coverage = sum(float(row[key]) >= threshold for row in rows) / len(rows)
        # Closest coverage wins; among ties retain more coverage, then the
        # higher threshold.  This rule is fixed before reading test labels.
        score = (abs(coverage - target), -coverage, -threshold, threshold)
        if score < best:
            best = score
    return float(best[3])


def selective(rows: Sequence[Mapping[str, float]], key: str, threshold: float) -> dict[str, float]:
    accepted = [row for row in rows if float(row[key]) >= threshold]
    errors = [1.0 - float(row["correct"]) for row in accepted]
    return {
        "risk": float(np.mean(errors)) if errors else 0.0,
        "coverage": float(len(accepted) / len(rows)) if rows else 0.0,
        "brier_accepted": float(np.mean([(float(row[key]) - float(row["correct"])) ** 2 for row in accepted])) if accepted else 0.0,
        "n": float(len(accepted)),
    }


def brier(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return float(np.mean([(float(row[key]) - float(row["correct"])) ** 2 for row in rows])) if rows else 0.0


def ece(rows: Sequence[Mapping[str, float]], key: str, bins: int = 10) -> float:
    if not rows:
        return 0.0
    p = np.asarray([float(row[key]) for row in rows]); y = np.asarray([float(row["correct"]) for row in rows])
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1.0 else p <= high)
        if mask.any():
            value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return value


def aurc(rows: Sequence[Mapping[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    order = np.argsort(-np.asarray([float(row[key]) for row in rows]), kind="mergesort")
    errors = np.asarray([1.0 - float(row["correct"]) for row in rows])[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def bootstrap_mean_ci(values: Sequence[float], seed: int = 1729, repeats: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(x, size=(repeats, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def paired_bootstrap_delta(
    rows: Sequence[Mapping[str, float]], baseline_key: str, proposed_key: str, threshold_base: float, threshold_prop: float,
    repeats: int = 2000, seed: int = 1729,
) -> tuple[float, float, float]:
    """CI for proposed minus baseline error at their validation-set operating points."""
    if not rows:
        return 0.0, 0.0, 0.0
    base = np.asarray([1.0 - float(row["correct"]) if float(row[baseline_key]) >= threshold_base else np.nan for row in rows])
    prop = np.asarray([1.0 - float(row["correct"]) if float(row[proposed_key]) >= threshold_prop else np.nan for row in rows])
    # Compare risk at each method's selected coverage using a common accepted
    # prefix.  The primary metric below is the direct paired risk difference at
    # the nearest common coverage, avoiding a denominator mismatch.
    b_order = np.argsort(-np.asarray([float(row[baseline_key]) for row in rows]), kind="mergesort")
    p_order = np.argsort(-np.asarray([float(row[proposed_key]) for row in rows]), kind="mergesort")
    n = min(int(np.sum(~np.isnan(base))), int(np.sum(~np.isnan(prop))))
    if n == 0:
        return 0.0, 0.0, 0.0
    common = max(1, n)
    b_err = np.asarray([1.0 - float(rows[i]["correct"]) for i in b_order[:common]])
    p_err = np.asarray([1.0 - float(rows[i]["correct"]) for i in p_order[:common]])
    delta = p_err - b_err
    return bootstrap_mean_ci(delta, seed=seed, repeats=repeats)


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def run_dataset(name: str, root: Path, cap: int, seeds: int, entity_count: int | None, output_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    train, valid, test = root / "train.jsonl", root / "valid.jsonl", root / "test.jsonl"
    index, inferred_entities = load_events(train)
    n_entities = int(entity_count or inferred_entities)
    valid_all = list(iter_pe(valid)); test_all = list(iter_pe(test))
    valid_queries = evenly_sample(valid_all, cap); test_queries = evenly_sample(test_all, cap)
    known = load_known_for_queries((valid, test), list(set(valid_queries) | set(test_queries)), index)
    val_rows, val_rank_decay = evaluate(valid_queries, index, known, n_entities, "decay")
    test_rows, test_rank_decay = evaluate(test_queries, index, known, n_entities, "decay")
    _, test_rank_static = evaluate(test_queries, index, known, n_entities, "static")

    baseline_features = ["margin", "log_support", "recency"]
    ablation_sets = {
        "belief_only": ["margin"],
        "plus_certificate": ["margin", "certificate"],
        "plus_diversity": ["margin", "certificate", "diversity"],
        "plus_fragility": ["margin", "certificate", "diversity", "fragility_cost"],
        "full_certificate": ["margin", "log_support", "recency", "certificate", "diversity", "fragility_cost", "counter_evidence_cost"],
    }
    targets = (0.20, 0.40, 0.60, 0.80)
    aggregate: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(val_rows)); split = max(1, len(order) // 2)
        cal_rows = [val_rows[int(i)] for i in order[:split]]
        op_rows = [val_rows[int(i)] for i in order[split:]]
        models: dict[str, RidgeLogistic] = {}
        for label, features in [("baseline", baseline_features), ("stability", ablation_sets["full_certificate"])]:
            models[label] = RidgeLogistic(features).fit(cal_rows)
        add_prediction(op_rows, models["baseline"], "belief_baseline")
        add_prediction(op_rows, models["stability"], "belief_stability")
        add_prediction(test_rows, models["baseline"], "belief_baseline")
        add_prediction(test_rows, models["stability"], "belief_stability")
        row: dict[str, object] = {
            "dataset": name, "seed": seed, "n_entities": n_entities,
            "n_train_events": sum(len(v) for v in index.values()),
            "n_validation_available": len(valid_all), "n_validation": len(val_rows),
            "n_test_available": len(test_all), "n_test": len(test_rows),
            "decay_mrr": test_rank_decay["mrr"], "decay_hits1": test_rank_decay["hits1"],
            "decay_hits3": test_rank_decay["hits3"], "decay_hits10": test_rank_decay["hits10"],
            "static_mrr": test_rank_static["mrr"], "static_hits1": test_rank_static["hits1"],
            "static_hits3": test_rank_static["hits3"], "static_hits10": test_rank_static["hits10"],
            "baseline_brier": brier(test_rows, "belief_baseline"),
            "stability_brier": brier(test_rows, "belief_stability"),
            "baseline_ece": ece(test_rows, "belief_baseline"),
            "stability_ece": ece(test_rows, "belief_stability"),
            "baseline_aurc": aurc(test_rows, "belief_baseline"),
            "stability_aurc": aurc(test_rows, "belief_stability"),
            "calibration_n": len(cal_rows), "operating_n": len(op_rows),
        }
        for target in targets:
            label = f"{target:.2f}"
            for prefix, key in (("baseline", "belief_baseline"), ("stability", "belief_stability")):
                threshold = choose_threshold(op_rows, key, target)
                summary = selective(test_rows, key, threshold)
                row[f"{prefix}_threshold@{label}"] = threshold
                row[f"{prefix}_risk@{label}"] = summary["risk"]
                row[f"{prefix}_coverage@{label}"] = summary["coverage"]
                row[f"{prefix}_brier_accepted@{label}"] = summary["brier_accepted"]
        # The ablation table is fitted and selected by the same validation
        # protocol.  Only its 40% point and AURC are needed for the main table.
        for label, features in ablation_sets.items():
            model = RidgeLogistic(features).fit(cal_rows)
            val_key = f"belief_{label}_val"; test_key = f"belief_{label}"
            add_prediction(op_rows, model, val_key); add_prediction(test_rows, model, test_key)
            threshold = choose_threshold(op_rows, val_key, 0.40)
            summary = selective(test_rows, test_key, threshold)
            row[f"{label}_aurc"] = aurc(test_rows, test_key)
            row[f"{label}_risk@0.40"] = summary["risk"]
            row[f"{label}_coverage@0.40"] = summary["coverage"]
            if seed == 0:
                for t in targets:
                    th = choose_threshold(op_rows, val_key, t)
                    s = selective(test_rows, test_key, th)
                    curve_rows.append({"dataset": name, "ablation": label, "target": t, "risk": s["risk"], "coverage": s["coverage"], "threshold": th})
        aggregate.append(row)
        if seed == 0:
            for q, pred in zip(test_queries, test_rows):
                prediction_rows.append({
                    "dataset": name, "query_id": len(prediction_rows), "subject": int(q.subject), "relation": int(q.relation), "timestamp": int(q.timestamp),
                    "answers": ";".join(map(str, q.answers)), "winner": int(pred["winner"]), "correct": int(pred["correct"]), "rank": pred["rank"],
                    **{k: pred[k] for k in ("margin", "support", "recency", "diversity", "fragility_cost", "counter_evidence_cost", "certificate", "belief_baseline", "belief_stability")},
                })
        print(f"{name} seed={seed} n_test={len(test_rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    # Query-level paired uncertainty for the headline operating point uses the
    # seed-0 predictions and thresholds.  Seed variability remains visible in
    # the aggregate CSV and summary CIs.
    summary: dict[str, object] = {
        "dataset": name, "n_seeds": seeds, "n_entities": n_entities,
        "n_train_events": sum(len(v) for v in index.values()),
        "n_validation_available": len(valid_all), "n_validation": len(valid_queries),
        "n_test_available": len(test_all), "n_test": len(test_queries),
        "elapsed_seconds": time.perf_counter() - started,
    }
    for metric in ("decay_mrr", "decay_hits1", "decay_hits3", "decay_hits10", "static_mrr", "static_hits1", "static_hits3", "static_hits10", "baseline_brier", "stability_brier", "baseline_ece", "stability_ece", "baseline_aurc", "stability_aurc"):
        vals = [float(r[metric]) for r in aggregate]
        mean, lo, hi = bootstrap_mean_ci(vals, seed=7000 + len(metric))
        summary[metric] = mean; summary[metric + "_lo"] = lo; summary[metric + "_hi"] = hi
    for target in targets:
        label = f"{target:.2f}"
        for prefix in ("baseline", "stability"):
            for metric in ("risk", "coverage", "brier_accepted"):
                vals = [float(r[f"{prefix}_{metric}@{label}"]) for r in aggregate]
                mean, lo, hi = bootstrap_mean_ci(vals, seed=8000 + int(target * 100) + len(prefix) + len(metric))
                summary[f"{prefix}_{metric}@{label}"] = mean; summary[f"{prefix}_{metric}@{label}_lo"] = lo; summary[f"{prefix}_{metric}@{label}_hi"] = hi
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / f"{name}_seeds.csv", aggregate)
    write_rows(output_dir / f"{name}_predictions.csv", prediction_rows)
    write_rows(output_dir / f"{name}_ablation_curve.csv", curve_rows)
    with (output_dir / f"{name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return {"summary": summary, "seeds": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--entity-count", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    result = run_dataset(args.dataset, args.data_root, args.cap, args.seeds, args.entity_count, args.output_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
