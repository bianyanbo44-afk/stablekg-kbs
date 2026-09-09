"""Attach StableKG certificates to a temporal ComplEx-style backbone.

The module trains an interpolative temporal link-prediction model on the
published split and evaluates a post-hoc selective layer.  Model confidence is
calibrated on one half of validation data, operating thresholds are selected
on the other half, and test labels are used only for final reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import stability_benchmark as bench


def complex_product(
    a_re: torch.Tensor,
    a_im: torch.Tensor,
    b_re: torch.Tensor,
    b_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re


class TemporalComplEx(nn.Module):
    """A compact TComplEx-style model with reciprocal-relation training."""

    def __init__(self, n_entities: int, n_relations: int, n_times: int, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.entity_re = nn.Embedding(n_entities, dim)
        self.entity_im = nn.Embedding(n_entities, dim)
        self.relation_re = nn.Embedding(2 * n_relations, dim)
        self.relation_im = nn.Embedding(2 * n_relations, dim)
        self.time_re = nn.Embedding(n_times, dim)
        self.time_im = nn.Embedding(n_times, dim)
        for embedding in (
            self.entity_re,
            self.entity_im,
            self.relation_re,
            self.relation_im,
            self.time_re,
            self.time_im,
        ):
            nn.init.xavier_uniform_(embedding.weight)

    def query(self, subject: torch.Tensor, relation: torch.Tensor, timestamp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sr_re, sr_im = complex_product(
            self.entity_re(subject),
            self.entity_im(subject),
            self.relation_re(relation),
            self.relation_im(relation),
        )
        q_re, q_im = complex_product(
            sr_re,
            sr_im,
            self.time_re(timestamp),
            self.time_im(timestamp),
        )
        return q_re, q_im

    def score(self, subject: torch.Tensor, relation: torch.Tensor, timestamp: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        q_re, q_im = self.query(subject, relation, timestamp)
        o_re = self.entity_re(objects)
        o_im = self.entity_im(objects)
        if objects.ndim == 2:
            q_re = q_re[:, None, :]
            q_im = q_im[:, None, :]
        return torch.sum(q_re * o_re + q_im * o_im, dim=-1) * math.sqrt(self.dim)

    def score_all(self, subject: torch.Tensor, relation: torch.Tensor, timestamp: torch.Tensor) -> torch.Tensor:
        q_re, q_im = self.query(subject, relation, timestamp)
        return (q_re @ self.entity_re.weight.T + q_im @ self.entity_im.weight.T) * math.sqrt(self.dim)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def flatten_training(queries: Sequence[bench.Query], n_relations: int) -> np.ndarray:
    rows: list[tuple[int, int, int, int]] = []
    for query in queries:
        rows.extend((query.subject, query.relation, query.timestamp, answer) for answer in query.answers)
        rows.extend((answer, query.relation + n_relations, query.timestamp, query.subject) for answer in query.answers)
    return np.asarray(rows, dtype=np.int64)


def train_model(
    model: TemporalComplEx,
    examples: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    negatives: int,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tensor = torch.from_numpy(examples)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        started = time.perf_counter()
        permutation = torch.randperm(len(tensor), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(tensor), batch_size):
            batch = tensor[permutation[start : start + batch_size]]
            s, r, t, o = (batch[:, i] for i in range(4))
            negative = torch.randint(
                model.entity_re.num_embeddings,
                (len(batch), negatives),
                generator=generator,
            )
            if negatives:
                collision = negative == o[:, None]
                negative[collision] = (negative[collision] + 1) % model.entity_re.num_embeddings
            positive_score = model.score(s, r, t, o)
            negative_score = model.score(s, r, t, negative)
            loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        row = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(1, seen),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(f"epoch={epoch + 1} loss={row['loss']:.5f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    return history


def evidence_for_candidate(
    query: bench.Query,
    events: Sequence[tuple[int, int]],
    candidate: int,
    half_life: float,
) -> dict[str, float]:
    mass, bins, counts, last = bench.causal_features(query, events, half_life, "decay")
    candidate_mass = float(mass.get(candidate, 0.0))
    competitor_mass = max((float(value) for entity, value in mass.items() if entity != candidate), default=0.0)
    evidence_margin = max(0.0, candidate_mass - competitor_mass)
    candidate_bins: Counter[int] = bins.get(candidate, Counter())
    total_mass = float(sum(candidate_bins.values()))
    diversity = 0.0 if total_mass <= 0.0 else float(1.0 - max(candidate_bins.values()) / total_mass)
    _, fragility_cost = bench.window_deletion_certificate(candidate_bins.values(), evidence_margin)
    counter_cost = float(np.clip(evidence_margin / (candidate_mass + competitor_mass + 1e-12), 0.0, 1.0))
    certificate = float(np.clip(0.5 * fragility_cost + 0.5 * diversity, 0.0, 1.0))
    recency = 0.0 if candidate not in last else math.exp(
        -math.log(2.0) * (query.timestamp - last[candidate]) / half_life
    )
    return {
        "support": float(counts.get(candidate, 0)),
        "log_support": float(math.log1p(counts.get(candidate, 0))),
        "recency": float(recency),
        "diversity": diversity,
        "fragility_cost": fragility_cost,
        "counter_evidence_cost": counter_cost,
        "certificate": certificate,
        "evidence_alignment": float(candidate_mass > 0.0 and candidate_mass >= competitor_mass),
    }


def filtered_rank(scores: np.ndarray, answer: int, known_answers: set[int]) -> float:
    target = float(scores[answer])
    excluded = set(known_answers) - {answer}
    greater = 0
    equal = 0
    for entity, score in enumerate(scores):
        if entity == answer or entity in excluded:
            continue
        if float(score) > target:
            greater += 1
        elif float(score) == target:
            equal += 1
    return 1.0 + greater + 0.5 * equal


@torch.inference_mode()
def evaluate(
    model: TemporalComplEx,
    queries: Sequence[bench.Query],
    index: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
    known: Mapping[tuple[int, int, int], set[int]],
    batch_size: int,
    half_life: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, float]] = []
    ranks: list[float] = []
    for start in range(0, len(queries), batch_size):
        query_batch = queries[start : start + batch_size]
        s = torch.tensor([q.subject for q in query_batch], dtype=torch.long)
        r = torch.tensor([q.relation for q in query_batch], dtype=torch.long)
        t = torch.tensor([q.timestamp for q in query_batch], dtype=torch.long)
        matrix = model.score_all(s, r, t).cpu().numpy()
        for query, scores in zip(query_batch, matrix):
            order = np.argsort(-scores, kind="mergesort")
            winner = int(order[0])
            runner = int(order[1])
            maximum = float(scores[winner])
            margin = maximum - float(scores[runner])
            shifted = scores.astype(np.float64) - float(np.max(scores))
            log_partition = float(np.log(np.exp(shifted).sum()))
            softmax_max = float(math.exp(-log_partition))
            evidence = evidence_for_candidate(
                query,
                index.get((query.subject, query.relation), ()),
                winner,
                half_life,
            )
            known_answers = set(known.get(query.key, set())) | set(query.answers)
            answer_ranks = [filtered_rank(scores, answer, known_answers) for answer in query.answers]
            ranks.extend(answer_ranks)
            rows.append(
                {
                    "subject": float(query.subject),
                    "relation": float(query.relation),
                    "timestamp": float(query.timestamp),
                    "winner": float(winner),
                    "runner": float(runner),
                    "correct": float(winner in query.answers),
                    "rank": float(np.mean(answer_ranks)),
                    "neural_margin": float(margin),
                    "neural_max": maximum,
                    "neural_softmax": softmax_max,
                    **evidence,
                }
            )
    rank_array = np.asarray(ranks, dtype=float)
    metrics = {
        "mrr": float(np.mean(1.0 / rank_array)),
        "hits1": float(np.mean(rank_array <= 1.0)),
        "hits3": float(np.mean(rank_array <= 3.0)),
        "hits10": float(np.mean(rank_array <= 10.0)),
    }
    return rows, metrics


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    root = args.data_root
    train_queries = list(bench.iter_pe(root / "train.jsonl"))
    validation_all = list(bench.iter_pe(root / "valid.jsonl"))
    test_all = list(bench.iter_pe(root / "test.jsonl"))
    validation = bench.evenly_sample(validation_all, args.cap)
    test = bench.evenly_sample(test_all, args.cap)
    all_queries = train_queries + validation + test
    n_entities = args.entity_count or 1 + max(max(q.subject, *q.answers) for q in all_queries)
    n_relations = 1 + max(q.relation for q in all_queries)
    n_times = 1 + max(q.timestamp for q in all_queries)
    index, _ = bench.load_events(root / "train.jsonl")
    known = bench.load_known_for_queries((root / "valid.jsonl", root / "test.jsonl"), validation + test, index)
    examples = flatten_training(train_queries, n_relations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    baseline_features = ["neural_margin", "neural_max", "neural_softmax"]
    stable_features = baseline_features + [
        "log_support",
        "recency",
        "certificate",
        "diversity",
        "fragility_cost",
        "counter_evidence_cost",
        "evidence_alignment",
    ]
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        set_seed(seed)
        model = TemporalComplEx(n_entities, n_relations, n_times, args.dim)
        started = time.perf_counter()
        if args.checkpoint_dir:
            checkpoint = torch.load(args.checkpoint_dir / f"{args.dataset}_seed{seed}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            history = []
        else:
            history = train_model(model, examples, seed, args.epochs, args.batch_size,
                                  args.negatives, args.learning_rate, args.weight_decay)
        validation_rows, validation_rank = evaluate(model, validation, index, known, args.eval_batch_size, args.half_life)
        test_rows, test_rank = evaluate(model, test, index, known, args.eval_batch_size, args.half_life)
        rng = np.random.default_rng(50_000 + seed)
        order = rng.permutation(len(validation_rows))
        split = len(order) // 2
        calibration = [validation_rows[int(i)] for i in order[:split]]
        operating = [validation_rows[int(i)] for i in order[split:]]
        baseline = bench.RidgeLogistic(baseline_features).fit(calibration)
        stable = bench.RidgeLogistic(stable_features).fit(calibration)
        bench.add_prediction(operating, baseline, "belief_baseline")
        bench.add_prediction(operating, stable, "belief_stability")
        bench.add_prediction(test_rows, baseline, "belief_baseline")
        bench.add_prediction(test_rows, stable, "belief_stability")
        record: dict[str, object] = {
            "dataset": args.dataset,
            "seed": seed,
            "n_entities": n_entities,
            "n_relations": n_relations,
            "n_times": n_times,
            "n_train_quadruples_reciprocal": len(examples),
            "epochs": args.epochs,
            "dimension": args.dim,
            "validation_mrr": validation_rank["mrr"],
            **{f"test_{key}": value for key, value in test_rank.items()},
            "baseline_brier": bench.brier(test_rows, "belief_baseline"),
            "stability_brier": bench.brier(test_rows, "belief_stability"),
            "baseline_ece": bench.ece(test_rows, "belief_baseline"),
            "stability_ece": bench.ece(test_rows, "belief_stability"),
            "baseline_aurc": bench.aurc(test_rows, "belief_baseline"),
            "stability_aurc": bench.aurc(test_rows, "belief_stability"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        for target in (0.20, 0.40, 0.60, 0.80):
            label = f"{target:.2f}"
            for prefix, key in (("baseline", "belief_baseline"), ("stability", "belief_stability")):
                threshold = bench.choose_threshold(operating, key, target)
                metrics = bench.selective(test_rows, key, threshold)
                record[f"{prefix}_threshold@{label}"] = threshold
                record[f"{prefix}_risk@{label}"] = metrics["risk"]
                record[f"{prefix}_coverage@{label}"] = metrics["coverage"]
        seed_rows.append(record)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": vars(args),
                "n_entities": n_entities,
                "n_relations": n_relations,
                "n_times": n_times,
            },
            args.output_dir / f"{args.dataset}_seed{seed}.pt",
        )
        if history:
            write_rows(args.output_dir / f"{args.dataset}_seed{seed}_history.csv", history)
        if seed == args.seed_start:
            write_rows(args.output_dir / f"{args.dataset}_predictions.csv", test_rows)
        print(json.dumps(record, indent=2), flush=True)
    write_rows(args.output_dir / f"{args.dataset}_seeds.csv", seed_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results_neural"))
    parser.add_argument("--checkpoint-dir", type=Path, help="Re-evaluate trusted local checkpoints without retraining")
    parser.add_argument("--entity-count", type=int)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--half-life", type=float, default=35.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
