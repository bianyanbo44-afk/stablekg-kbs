"""Matched evidence controls using the original validation protocol.

This is a prespecified diagnostic grid, not a test-set hyperparameter search.
Original headline predictions are retained. All feature rows and contrasts are
saved so alternative explanations can be assessed without retraining rankers.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import stability_benchmark as b
import analyze_results as analysis

BASE = ["margin", "log_support", "recency"]
FD = ["fragility_cost", "diversity"]
Z = ["counter_evidence_cost"]
SETS = {
    "margin_only": ["margin"],
    "matched_base": BASE,
    "base_plus_c": BASE + ["certificate"],
    "base_plus_z": BASE + Z,
    "base_plus_fd": BASE + FD,
    "without_diversity": BASE + Z + ["fragility_cost"],
    "without_fragility": BASE + Z + ["diversity"],
    "nonredundant": BASE + Z + FD,
    "full_original": BASE + ["certificate", "diversity", "fragility_cost"] + Z,
    "base_plus_window_count": BASE + ["window_count"],
    "base_plus_softmax_entropy": BASE + ["score_softmax", "negative_entropy"],
}

def feature_rows(root, split, cap, n):
    index, _ = b.load_events(root / "train.jsonl")
    for split_name in split:
        queries = b.evenly_sample(list(b.iter_pe(root / f"{split_name}.jsonl")), cap)
        rows = []
        for q in queries:
            f, mass, _ = b.winner_features(q, index.get((q.subject, q.relation), ()), n, "decay", 35.)
            _, bins, _, _ = b.causal_features(q, index.get((q.subject, q.relation), ()), 35., "decay")
            k, cost = b.window_deletion_certificate(bins.get(int(f["winner"]), {}).values(), f["margin"])
            # Zero-score candidates are included in the softmax partition.
            shift = max(mass.values(), default=0.)
            exp_values = np.exp(np.asarray(list(mass.values())) - shift)
            zero = math.exp(-shift)
            partition = exp_values.sum() + (n - len(mass)) * zero
            probabilities = exp_values / partition
            pzero = zero / partition
            entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum()
            if pzero > 0:
                entropy -= (n - len(mass)) * pzero * math.log(pzero)
            rows.append({**f, "correct": float(int(f["winner"]) in q.answers),
                "timestamp": q.timestamp, "subject": q.subject, "relation": q.relation,
                "window_count": k, "prefix_cost_checked": cost,
                "score_softmax": 1. / partition, "negative_entropy": -float(entropy)})
        yield split_name, rows

def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = {s: args.output_dir / f"{args.dataset}_{s}_features.csv" for s in ("valid", "test")}
    if not all(p.exists() for p in cache.values()):
        for split, rows in feature_rows(args.data_root, cache, args.cap, args.entity_count):
            pd.DataFrame(rows).to_csv(cache[split], index=False)
            print(f"{args.dataset} cached {split}: {len(rows)}", flush=True)
    val, test = [pd.read_csv(cache[s]).to_dict("records") for s in ("valid", "test")]
    if args.correct_prefix:
        for row in val + test:
            row["fragility_cost"] = row["prefix_cost_checked"]
            row["certificate"] = .5 * (row["fragility_cost"] + row["diversity"])
        args.output_dir = args.output_dir / "corrected"
        args.output_dir.mkdir(parents=True, exist_ok=True)
    results, seed0 = [], None
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(val))
        cal = [val[i] for i in order[:len(order)//2]]
        op = [val[i] for i in order[len(order)//2:]]
        for name, features in SETS.items():
            model = b.RidgeLogistic(features).fit(cal)
            key = name
            b.add_prediction(op, model, key)
            b.add_prediction(test, model, key)
            row = {"dataset": args.dataset, "seed": seed, "model": name,
                   "aurc": b.aurc(test, key), "brier": b.brier(test, key), "ece": b.ece(test, key)}
            for target in (.2, .4):
                th = b.choose_threshold(op, key, target)
                v = b.selective(test, key, th)
                row[f"risk@{target}"] = v["risk"]
                row[f"coverage@{target}"] = v["coverage"]
                order_test = np.argsort(-np.asarray([x[key] for x in test]), kind="mergesort")
                count = max(1, round(target * len(test)))
                row[f"fixed_count_risk@{target}"] = float(np.mean([1. - test[i]["correct"] for i in order_test[:count]]))
            results.append(row)
        if seed == 0:
            seed0 = pd.DataFrame(test).copy()
        print(f"{args.dataset} matched controls seed {seed} complete", flush=True)
    pd.DataFrame(results).to_csv(args.output_dir / f"{args.dataset}_seeds.csv", index=False)
    seed0.to_csv(args.output_dir / f"{args.dataset}_predictions.csv", index=False)
    contrasts = []
    for control in ("matched_base", "base_plus_z", "base_plus_fd", "without_diversity", "without_fragility", "nonredundant", "base_plus_softmax_entropy"):
        def stat(frame):
            correct = frame.correct.to_numpy(float)
            return analysis.aurc(correct, frame.full_original.to_numpy(float)) - analysis.aurc(correct, frame[control].to_numpy(float))
        point, low, high = analysis.block_bootstrap(seed0, stat, 319, repeats=args.bootstrap)
        contrasts.append({"dataset": args.dataset, "contrast": f"full_original - {control}", "delta_aurc": point, "ci_low": low, "ci_high": high})
    pd.DataFrame(contrasts).to_csv(args.output_dir / f"{args.dataset}_contrasts.csv", index=False)
    differences = np.abs(seed0.fragility_cost - seed0.prefix_cost_checked) > 1e-10
    summary = {"dataset": args.dataset, "feature_sets": SETS, "seeds": list(range(5)),
               "ridge": .01, "half_life": 35, "window_timestamp_units": 7,
               "bootstrap_resamples": args.bootstrap,
               "numerical_prefix_disagreements": int(differences.sum()),
               "zero_support_fraction": float((seed0.support == 0).mean())}
    (args.output_dir / f"{args.dataset}_protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).groupby("model").aurc.mean().to_string(), flush=True)
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--entity-count", type=int, required=True)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("results_review_20260909"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--correct-prefix", action="store_true")
    run(parser.parse_args())
