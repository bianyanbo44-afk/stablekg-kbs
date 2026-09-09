"""Causal, calibration-first evaluation for the stability-certificate paper.

This implementation is intentionally self-contained and CPU friendly.  It
uses only the published ``Pe`` point queries from the ICEWS/GDELT JSONL mirror:
training facts are evidence, validation is split into calibration and operating
halves, and the test split is untouched until the final report.  The proposed
selector calibrates correctness from the model margin together with an
intervention certificate and temporal diversity.  A confidence-only calibration
is the matched baseline.
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
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    max_entity = -1
    for query in iter_pe(train_path):
        for answer in query.answers:
            index[(query.subject, query.relation)].append((query.timestamp, answer))
            max_entity = max(max_entity, query.subject, answer)
    return index, max_entity + 1


def load_known(paths: Iterable[Path]) -> dict[tuple[int, int, int], set[int]]:
    known: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for path in paths:
        for query in iter_pe(path):
            known[query.key].update(query.answers)
    return known


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-x))


def _causal_features(
    query: Query,
    events: Sequence[tuple[int, int]],
    half_life: float,
    mode: str,
) -> tuple[dict[int, float], dict[int, Counter[int]], dict[int, int], dict[int, int]]:
    """Return causal weighted mass, temporal bins and raw support counts."""
    mass: dict[int, float] = defaultdict(float)
    bins: dict[int, Counter[int]] = defaultdict(Counter)
    count: dict[int, int] = defaultdict(int)
    last: dict[int, int] = {}
    for event_time, entity in events:
        # Strict temporal causality is central to the benchmark.  The original
        # prototype clipped negative ages and therefore leaked future evidence.
        if event_time > query.timestamp:
            continue
        age = query.timestamp - event_time
        weight = 1.0 if mode == "static" else math.exp(-math.log(2.0) * age / half_life)
        mass[entity] += weight
        bins[entity][event_time // 7] += weight
        count[entity] += 1
        last[entity] = max(last.get(entity, -1), event_time)
    return mass, bins, count, last


def _winner_features(
    query: Query,
    events: Sequence[tuple[int, int]],
    n_entities: int,
    mode: str,
    half_life: float,
) -> dict[str, float]:
    mass, bins, counts, last = _causal_features(query, events, half_life, mode)
    # Relation-specific empirical prior is computed only from causal evidence.
    # It provides deterministic tie-breaking for candidates with equal mass.
    ranked = sorted(range(n_entities), key=lambda e: (-mass.get(e, 0.0), -counts.get(e, 0), e))
    winner = ranked[0]
    winner_mass = float(mass.get(winner, 0.0))
    runner_mass = float(mass.get(ranked[1], 0.0)) if len(ranked) > 1 else 0.0
    margin = winner_mass - runner_mass
    support = int(counts.get(winner, 0))
    winner_bins = bins.get(winner, Counter())
    total_mass = float(sum(winner_bins.values()))
    max_bin = float(max(winner_bins.values())) if winner_bins else 0.0
    diversity = 0.0 if total_mass <= 0 else float(1.0 - max_bin / total_mass)
    # Cheapest event-window deletion that erases the current margin.  This is
    # an operational certificate, not a post-hoc uncertainty label.
    removed = 0.0
    intervention_cost = 0.0
    if winner_mass > 0 and margin > 0 and winner_bins:
        for value in sorted(winner_bins.values(), reverse=True):
            removed += float(value)
            if removed >= margin:
                intervention_cost = float(np.clip(removed / winner_mass, 0.0, 1.0))
                break
        if intervention_cost == 0.0:
            intervention_cost = 1.0
    certificate = float(np.clip(0.5 * intervention_cost + 0.5 * diversity, 0.0, 1.0))
    recency = 0.0 if winner not in last else math.exp(-math.log(2.0) * (query.timestamp - last[winner]) / half_life)
    confidence = sigmoid(margin)
    return {
        "winner": float(winner),
        "margin": float(margin),
        "confidence": float(confidence),
        "support": float(support),
        "log_support": float(math.log1p(support)),
        "diversity": diversity,
        "intervention_cost": intervention_cost,
        "certificate": certificate,
        "recency": float(recency),
        "mass": winner_mass,
        "runner_mass": runner_mass,
    }


def evaluate(
    queries: Sequence[Query],
    index: DefaultDict[tuple[int, int], list[tuple[int, int]]],
    known: Mapping[tuple[int, int, int], set[int]],
    n_entities: int,
    mode: str,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    ranks: list[float] = []
    for query in queries:
        events = index.get((query.subject, query.relation), [])
        feat = _winner_features(query, events, n_entities, mode, 35.0)
        winner = int(feat["winner"])
        answers = set(query.answers)
        feat["correct"] = float(winner in answers)

        # Filtered rank over the entire entity vocabulary.  The score is the
        # causal mass, with raw support and id used only for deterministic ties.
        mass, _, counts, _ = _causal_features(query, events, 35.0, mode)
        filtered_other = set(known.get(query.key, set())) - answers
        # Only entities observed before the query can have a positive score.
        # Counting the zero-score tail analytically avoids an O(|V|) loop per
        # query while preserving the exact filtered rank and tie convention.
        observed = set(mass)
        ranked_observed = sorted(observed, key=lambda e: (-mass[e], -counts.get(e, 0), e))
        best_rank = float("inf")
        for answer in answers:
            value = float(mass.get(answer, 0.0))
            if answer in observed:
                greater = 0
                equal = 0
                answer_count = counts.get(answer, 0)
                for entity in ranked_observed:
                    if entity == answer or entity in filtered_other:
                        continue
                    other = float(mass[entity])
                    if other > value:
                        greater += 1
                    elif other == value:
                        if counts.get(entity, 0) > answer_count or (counts.get(entity, 0) == answer_count and entity < answer):
                            greater += 1
                        else:
                            equal += 1
                best_rank = min(best_rank, 1.0 + greater + 0.5 * equal)
            else:
                # Unobserved answers tie after all remaining observed entities;
                # the deterministic id order gives a reproducible representative
                # rank and the half-tie correction keeps the metric standard.
                remaining_observed = sum(entity not in filtered_other for entity in observed)
                zero_entities = n_entities - len(observed) - sum(entity not in observed for entity in filtered_other)
                best_rank = min(best_rank, 1.0 + remaining_observed + 0.5 * max(0, zero_entities - 1))
        feat["rank"] = best_rank
        ranks.append(1.0 / best_rank)
        rows.append(feat)
    return rows, {"mrr": float(np.mean(ranks)) if ranks else 0.0, "hits1": float(np.mean([r <= 1 for r in [x["rank"] for x in rows]])) if rows else 0.0, "hits3": float(np.mean([r <= 3 for r in [x["rank"] for x in rows]])) if rows else 0.0, "hits10": float(np.mean([r <= 10 for r in [x["rank"] for x in rows]])) if rows else 0.0}


class LogisticCalibrator:
    """Small deterministic ridge logistic model, avoiding a heavy dependency."""
    def __init__(self, feature_names: Sequence[str], l2: float = 1e-2) -> None:
        self.feature_names = list(feature_names)
        self.l2 = l2
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weight: np.ndarray | None = None

    def _matrix(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        x = np.asarray([[float(row[name]) for name in self.feature_names] for row in rows], dtype=float)
        if self.mean is None:
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
            self.scale[self.scale < 1e-8] = 1.0
        return (x - self.mean) / self.scale

    def fit(self, rows: Sequence[dict[str, float]]) -> "LogisticCalibrator":
        if not rows:
            self.weight = np.zeros(len(self.feature_names) + 1)
            self.mean = np.zeros(len(self.feature_names)); self.scale = np.ones(len(self.feature_names))
            return self
        x = self._matrix(rows)
        x = np.c_[np.ones(len(x)), x]
        y = np.asarray([float(row["correct"]) for row in rows], dtype=float)
        w = np.zeros(x.shape[1], dtype=float)
        for _ in range(500):
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -40, 40)))
            grad = (x.T @ (p - y)) / len(y)
            grad[1:] += self.l2 * w[1:]
            # A fixed step and clipped gradient make the fit reproducible on all
            # three data scales.
            w -= 0.25 * np.clip(grad, -1.0, 1.0)
        self.weight = w
        return self

    def predict(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("fit must be called before predict")
        x = self._matrix(rows)
        x = np.c_[np.ones(len(x)), x]
        return 1.0 / (1.0 + np.exp(-np.clip(x @ self.weight, -40, 40)))


def add_predictions(rows: list[dict[str, float]], baseline: LogisticCalibrator, proposed: LogisticCalibrator) -> None:
    base = baseline.predict(rows)
    prop = proposed.predict(rows)
    for row, b, p in zip(rows, base, prop):
        row["belief_baseline"] = float(b)
        row["belief_stability"] = float(p)


def choose_threshold(rows: Sequence[dict[str, float]], key: str, target: float) -> float:
    values = sorted({float(r[key]) for r in rows}, reverse=True)
    if not values:
        return 1.0
    candidates = [values[0] + 1e-9] + values
    best = (float("inf"), float("inf"), float("inf"), 1.0)
    for t in candidates:
        cov = sum(r[key] >= t for r in rows) / len(rows)
        score = (abs(cov - target), -cov, -t, t)
        if score < best:
            best = score
    return float(best[3])


def selective(rows: Sequence[dict[str, float]], key: str, threshold: float) -> dict[str, float]:
    accepted = [r for r in rows if r[key] >= threshold]
    return {
        "risk": float(np.mean([1.0 - r["correct"] for r in accepted])) if accepted else 0.0,
        "coverage": float(len(accepted) / len(rows)) if rows else 0.0,
        "brier": float(np.mean([(r[key] - r["correct"]) ** 2 for r in accepted])) if accepted else 0.0,
        "n": float(len(accepted)),
    }


def ece(rows: Sequence[dict[str, float]], key: str, bins: int = 10) -> float:
    if not rows:
        return 0.0
    p = np.asarray([r[key] for r in rows]); y = np.asarray([r["correct"] for r in rows]); out = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any(): out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return out


def aurc(rows: Sequence[dict[str, float]], key: str) -> float:
    if not rows: return 0.0
    order = np.argsort(-np.asarray([r[key] for r in rows]), kind="mergesort")
    errors = np.asarray([1-r["correct"] for r in rows])[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors)+1)))


def run_dataset(name: str, root: Path, cap: int, seeds: int, entity_count: int | None) -> list[dict[str, float]]:
    train, valid, test = root/"train.jsonl", root/"valid.jsonl", root/"test.jsonl"
    started = time.time()
    index, inferred = load_events(train)
    n_entities = entity_count or inferred
    valid_queries = list(iter_pe(valid, cap)); test_queries = list(iter_pe(test, cap))
    known = load_known((train, valid, test))
    out: list[dict[str, float]] = []
    targets = (0.20, 0.40, 0.60, 0.80)
    for seed in range(seeds):
        val_rows, val_rank = evaluate(valid_queries, index, known, n_entities, "decay")
        test_rows, test_rank = evaluate(test_queries, index, known, n_entities, "decay")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(val_rows)); split = max(1, len(order)//2)
        cal_rows = [val_rows[i] for i in order[:split]]; op_rows = [val_rows[i] for i in order[split:]]
        baseline_features = ["margin", "log_support", "recency"]
        proposed_features = ["margin", "log_support", "recency", "certificate", "diversity", "intervention_cost"]
        baseline = LogisticCalibrator(baseline_features).fit(cal_rows)
        proposed = LogisticCalibrator(proposed_features).fit(cal_rows)
        # Fit on calibration rows only. The operating half chooses thresholds;
        # test rows remain untouched until prediction is finalized.
        add_predictions(op_rows, baseline, proposed)
        add_predictions(test_rows, baseline, proposed)
        row: dict[str, float] = {"dataset":name,"seed":float(seed),"n_entities":float(n_entities),"n_train_events":float(sum(map(len,index.values()))),"n_validation":float(len(val_rows)),"n_test":float(len(test_rows)),"decay_mrr":test_rank["mrr"],"decay_hits1":test_rank["hits1"],"decay_hits3":test_rank["hits3"],"decay_hits10":test_rank["hits10"],"baseline_ece":ece(test_rows,"belief_baseline"),"stability_ece":ece(test_rows,"belief_stability"),"baseline_aurc":aurc(test_rows,"belief_baseline"),"stability_aurc":aurc(test_rows,"belief_stability"),"elapsed_seconds":float(time.time()-started)}
        for target in targets:
            label=f"{target:.2f}"
            for prefix,key in (("baseline","belief_baseline"),("stability","belief_stability")):
                threshold=choose_threshold(op_rows,key,target); summary=selective(test_rows,key,threshold)
                row[f"{prefix}_threshold@{label}"]=threshold; row[f"{prefix}_risk@{label}"]=summary["risk"]; row[f"{prefix}_coverage@{label}"]=summary["coverage"]; row[f"{prefix}_brier@{label}"]=summary["brier"]
        out.append(row)
        print(f"{name} seed={seed} train={row['n_train_events']:.0f} test={row['n_test']:.0f} elapsed={row['elapsed_seconds']:.1f}s",flush=True)
    return out


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--dataset',required=True); p.add_argument('--cap',type=int,default=5000); p.add_argument('--seeds',type=int,default=3); p.add_argument('--entity-count',type=int); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=run_dataset(a.dataset,a.data_root,a.cap,a.seeds,a.entity_count); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for key in ('baseline_risk@0.40','stability_risk@0.40','baseline_coverage@0.40','stability_coverage@0.40'):
        v=[float(r[key]) for r in rows]; print(f'{key}: {statistics.mean(v):.4f} +/- {(statistics.stdev(v) if len(v)>1 else 0):.4f}')

if __name__=='__main__': main()
