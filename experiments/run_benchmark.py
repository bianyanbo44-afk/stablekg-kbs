"""Reproducible benchmark for stability-aware temporal reasoning.

The script has two modes. ``--mode synthetic`` runs a controlled feasibility
study with known truth and perturbations. ``--mode raw`` parses a canonical
ICEWS-style ``*.txt`` event file supplied by the user and reports data
inventory statistics without making unsupported labels. The public benchmark
download instructions and hashes live in ``data/DATA_MANIFEST.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Observation:
    time: int
    polarity: int
    source: int


@dataclass(frozen=True)
class Rule:
    premises: Tuple[int, ...]
    conclusion: int


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class DynamicKnowledgeStream:
    """A finite acyclic knowledge graph with a hidden time-varying truth state."""

    def __init__(
        self,
        seed: int,
        n_base: int = 120,
        n_derived: int = 100,
        n_sources: int = 4,
        drift_period: int = 10,
        drift_fraction: float = 0.12,
        steps: int = 90,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_base = n_base
        self.n_derived = n_derived
        self.n_sources = n_sources
        self.n_nodes = n_base + n_derived
        self.drift_period = drift_period
        self.drift_fraction = drift_fraction
        self.steps = steps
        self.reliabilities = np.array([0.97, 0.86, 0.71, 0.58], dtype=float)[:n_sources]
        self.source_names = [f"source_{i+1}" for i in range(n_sources)]

        self.rules = self._make_rules()
        self.by_conclusion: Dict[int, List[Rule]] = {}
        self.children: Dict[int, List[int]] = {i: [] for i in range(self.n_nodes)}
        for rule in self.rules:
            self.by_conclusion.setdefault(rule.conclusion, []).append(rule)
            for premise in rule.premises:
                self.children[premise].append(rule.conclusion)

        self.truth = self.rng.random(self.n_base) < 0.5
        self.evidence: List[List[Observation]] = [[] for _ in range(self.n_base)]
        self.changed_bases: List[int] = []

    def _make_rules(self) -> List[Rule]:
        rules: List[Rule] = []
        for offset in range(self.n_derived):
            conclusion = self.n_base + offset
            max_available = self.n_base + offset
            k = int(self.rng.integers(2, 4))
            premises = tuple(
                sorted(
                    self.rng.choice(max_available, size=k, replace=False).tolist()
                )
            )
            rules.append(Rule(premises=premises, conclusion=conclusion))
            # A fraction of conclusions receive an independent redundant proof.
            if offset >= 8 and self.rng.random() < 0.32:
                k2 = int(self.rng.integers(2, 4))
                alt = tuple(
                    sorted(self.rng.choice(max_available, size=k2, replace=False).tolist())
                )
                if set(alt) != set(premises):
                    rules.append(Rule(premises=alt, conclusion=conclusion))
        return rules

    def _hidden_derived_truth(self) -> np.ndarray:
        truth = np.zeros(self.n_nodes, dtype=bool)
        truth[: self.n_base] = self.truth
        for node in range(self.n_base, self.n_nodes):
            candidates = self.by_conclusion.get(node, [])
            truth[node] = any(all(truth[p] for p in r.premises) for r in candidates)
        return truth

    def _emit_observations(self, time: int) -> None:
        for source, reliability in enumerate(self.reliabilities):
            rate = 0.16 if source == 0 else 0.12
            selected = np.flatnonzero(self.rng.random(self.n_base) < rate)
            for fact in selected.tolist():
                # ``polarity`` is always the signed report: +1 means the
                # source reports the fact as true and -1 as false. A reliable
                # source agrees with the latent truth; an error flips it.
                correct = self.rng.random() < reliability
                report_true = bool(self.truth[fact]) if correct else not bool(self.truth[fact])
                polarity = 1 if report_true else -1
                self.evidence[fact].append(
                    Observation(time=time, polarity=polarity, source=source)
                )

    def step(self, time: int) -> np.ndarray:
        self.changed_bases = []
        if time > 0 and time % self.drift_period == 0:
            n_flip = max(1, int(round(self.n_base * self.drift_fraction)))
            self.changed_bases = self.rng.choice(self.n_base, size=n_flip, replace=False).tolist()
            self.truth[self.changed_bases] = ~self.truth[self.changed_bases]
        self._emit_observations(time)
        return self._hidden_derived_truth()


def local_drift_score(observations: Sequence[Observation], time: int) -> float:
    """Estimate a change signal from recent-vs-older source reports."""
    if len(observations) < 4:
        return 0.0
    recent = np.array([o.polarity for o in observations if time - o.time < 4], dtype=float)
    older = np.array([o.polarity for o in observations if 4 <= time - o.time < 14], dtype=float)
    if recent.size == 0 or older.size == 0:
        return 0.0
    return float(np.clip(abs(recent.mean() - older.mean()) / 2.0, 0.0, 1.0))


def source_logit(reliability: float) -> float:
    return math.log(reliability / (1.0 - reliability))


class Reasoner:
    def __init__(self, stream: DynamicKnowledgeStream, mode: str) -> None:
        self.stream = stream
        self.mode = mode
        self.last_state: Tuple[np.ndarray, np.ndarray] | None = None

    def _decay(self, age: int, drift: float) -> float:
        if self.mode == "static":
            rate = 0.0
        elif self.mode == "global_decay":
            rate = 0.045
        else:
            rate = 0.045 * (1.0 + 4.5 * drift)
        return math.exp(-rate * age)

    def _base_state(self, fact: int, time: int) -> Tuple[float, float]:
        obs = self.stream.evidence[fact]
        drift = local_drift_score(obs, time)
        contributions = []
        by_source: Dict[int, float] = {}
        for item in obs:
            weight = source_logit(float(self.stream.reliabilities[item.source]))
            value = item.polarity * weight * self._decay(time - item.time, drift)
            contributions.append(value)
            by_source[item.source] = by_source.get(item.source, 0.0) + value
        if not contributions:
            return 0.0, 0.0
        logit = float(np.sum(contributions))
        total_mass = float(np.sum(np.abs(contributions))) + 1e-9
        # Leave-one-source-out margin is an intervention certificate: if one
        # source can remove the sign, the conclusion is fragile.
        leave_one = [abs(logit - amount) for amount in by_source.values()]
        certificate = min(leave_one) / (abs(logit) + total_mass)
        certificate *= math.exp(-1.4 * drift)
        return logit, float(np.clip(certificate, 0.0, 1.0))

    def infer(self, time: int) -> Tuple[np.ndarray, np.ndarray]:
        logits = np.zeros(self.stream.n_nodes, dtype=float)
        stability = np.zeros(self.stream.n_nodes, dtype=float)
        for fact in range(self.stream.n_base):
            logits[fact], stability[fact] = self._base_state(fact, time)
        for node in range(self.stream.n_base, self.stream.n_nodes):
            candidates = self.stream.by_conclusion.get(node, [])
            scored = []
            for rule in candidates:
                vals = [logits[p] for p in rule.premises]
                # A conjunction is limited by its weakest premise while the
                # average preserves a useful confidence scale.
                score = 0.5 * min(vals) + 0.5 * float(np.mean(vals))
                path_stability = min(stability[p] for p in rule.premises)
                scored.append((score, path_stability))
            if scored:
                best_score, best_stability = max(scored, key=lambda item: item[0])
                logits[node] = best_score
                positive_paths = [s for s, _ in scored if s > 0]
                if len(positive_paths) > 1:
                    best_stability = min(1.0, best_stability * (1.0 + 0.35 * (len(positive_paths) - 1)))
                stability[node] = best_stability
        self.last_state = logits, stability
        return logits, stability

    def recomputation_work(self, changed_bases: Iterable[int]) -> int:
        if self.mode != "stability_aware":
            return self.stream.n_nodes
        seen = set(int(x) for x in changed_bases)
        frontier = list(seen)
        while frontier:
            node = frontier.pop()
            for child in self.stream.children.get(node, []):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return len(seen)

    def accepts(self, logits: np.ndarray, stability: np.ndarray) -> np.ndarray:
        confidence_gate = np.abs(logits) >= 1.35
        if self.mode == "stability_aware":
            return confidence_gate & (stability >= 0.22)
        return confidence_gate


def ece(prob: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(prob)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if np.any(mask):
            value += mask.mean() * abs(float(prob[mask].mean()) - float(truth[mask].mean()))
    return float(value)


def aurc(prob: np.ndarray, truth: np.ndarray) -> float:
    """Area under the risk-coverage curve for confidence-ranked decisions."""
    confidence = np.abs(prob - 0.5)
    order = np.argsort(-confidence)
    errors = ((prob[order] >= 0.5) != truth[order]).astype(float)
    cumulative = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.mean(cumulative))


def bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 1000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_one(seed: int) -> Dict[str, float]:
    stream = DynamicKnowledgeStream(seed=seed)
    records: Dict[str, List[float]] = {
        key: []
        for key in (
            "static",
            "global_decay",
            "stability_aware",
        )
    }
    work: Dict[str, List[float]] = {key: [] for key in records}
    for time in range(stream.steps):
        truth = stream.step(time)
        query_nodes = stream.rng.choice(stream.n_nodes, size=90, replace=False)
        for mode in records:
            reasoner = Reasoner(stream, mode)
            logits, stability = reasoner.infer(time)
            nodes = query_nodes
            prob = np.array([sigmoid(float(x)) for x in logits[nodes]])
            y = truth[nodes].astype(float)
            pred = prob >= 0.5
            accepted = reasoner.accepts(logits[nodes], stability[nodes])
            if np.any(accepted):
                risk = float(np.mean(pred[accepted] != y[accepted]))
                coverage = float(np.mean(accepted))
                records[mode].append((risk, coverage, float(np.mean((prob - y) ** 2)), ece(prob, y), aurc(prob, y)))
            else:
                records[mode].append((0.0, 0.0, float(np.mean((prob - y) ** 2)), ece(prob, y), aurc(prob, y)))
            work[mode].append(float(reasoner.recomputation_work(stream.changed_bases)))

    result: Dict[str, float] = {"seed": float(seed)}
    for mode, rows in records.items():
        matrix = np.asarray(rows, dtype=float)
        # Risk is averaged over time so each drift episode contributes equally.
        result[f"{mode}_risk"] = float(matrix[:, 0].mean())
        result[f"{mode}_coverage"] = float(matrix[:, 1].mean())
        result[f"{mode}_brier"] = float(matrix[:, 2].mean())
        result[f"{mode}_ece"] = float(matrix[:, 3].mean())
        result[f"{mode}_aurc"] = float(matrix[:, 4].mean())
        result[f"{mode}_work"] = float(np.mean(work[mode]))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_raw(path: Path) -> Dict[str, object]:
    """Inventory a tab-separated ICEWS-style file without training a model."""
    rows = 0
    times = set()
    entities = set()
    relations = set()
    sample: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                fields = line.rstrip("\n").split()
            if len(fields) < 4:
                continue
            rows += 1
            if len(sample) < 3:
                sample.append(fields[:8])
            times.add(fields[0])
            entities.update(fields[1:3])
            relations.add(fields[3])
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "events": rows,
        "unique_timestamps": len(times),
        "unique_entities": len(entities),
        "unique_relations": len(relations),
        "sample_rows": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.csv"))
    parser.add_argument("--mode", choices=("synthetic", "raw"), default="synthetic")
    parser.add_argument("--raw-file", type=Path)
    args = parser.parse_args()
    if args.mode == "raw":
        if args.raw_file is None:
            parser.error("--raw-file is required when --mode raw")
        print(json.dumps(inventory_raw(args.raw_file), indent=2, ensure_ascii=False))
        return
    rows = [run_one(seed) for seed in range(args.seeds)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} seeds to {args.output}")
    for mode in ("static", "global_decay", "stability_aware"):
        vals = [row[f"{mode}_risk"] for row in rows]
        cov = [row[f"{mode}_coverage"] for row in rows]
        aurcs = [row[f"{mode}_aurc"] for row in rows]
        lo, hi = bootstrap_ci(vals, seed=1729 + len(mode))
        print(f"{mode:16s} risk={np.mean(vals):.3f} [{lo:.3f},{hi:.3f}] coverage={np.mean(cov):.3f} aurc={np.mean(aurcs):.3f} work={np.mean([r[f'{mode}_work'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
