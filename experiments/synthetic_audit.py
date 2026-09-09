"""Controlled drift, intervention and dependency-closure audit.

This companion experiment is intentionally separate from the public-data
benchmark.  It supplies a known, time-varying truth state so that a certificate
can be tested against interventions and update latency without assigning
unobserved labels to a real knowledge graph.
"""

from __future__ import annotations

import argparse
import csv
import copy
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import run_benchmark as bench


def _bootstrap(values: Iterable[float], seed: int, repeats: int = 4000) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if not len(x):
        return 0.0, 0.0, 0.0
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(x, size=(repeats, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def _counter_injection(stream: bench.DynamicKnowledgeStream, fact: int, t: int, winner: bool) -> int:
    """Minimum number of opposite reports needed to reverse a signed logit."""
    reasoner = bench.Reasoner(stream, "stability_aware")
    original = bool(winner)
    old = stream.evidence[fact]
    # Injecting a report from the least reliable source is a conservative unit
    # intervention.  The source is fixed before looking at the outcome.
    source = len(stream.reliabilities) - 1
    current, _ = reasoner._base_state(fact, t)
    if current == 0.0:
        return 0
    unit = abs(bench.source_logit(float(stream.reliabilities[source])))
    if original:
        # A positive prediction flips only after crossing strictly below zero.
        return int(math.floor(abs(current) / unit + 1e-12) + 1)
    # A negative prediction flips when it reaches zero.
    return int(math.ceil(max(0.0, abs(current) / unit - 1e-12)))


def _incremental_runtime(stream: bench.DynamicKnowledgeStream, seed: int, repeats: int = 2000) -> tuple[float, float, float, float]:
    """Time exact full and dependency-closure recomputation on the rule DAG."""
    rng = np.random.default_rng(100_000 + seed)
    logits = rng.normal(size=stream.n_nodes)
    changed = sorted(rng.choice(stream.n_base, size=max(1, stream.n_base // 10), replace=False).tolist())
    closure = set(changed)
    frontier = list(changed)
    while frontier:
        node = frontier.pop()
        for child in stream.children.get(node, []):
            if child not in closure:
                closure.add(child); frontier.append(child)
    ordered_closure = sorted(closure)
    perturbation = rng.normal(scale=0.25, size=len(changed))

    def propagate_inplace(out: np.ndarray, nodes: Iterable[int]) -> None:
        for node in nodes:
            if node < stream.n_base:
                continue
            scores = []
            for rule in stream.by_conclusion.get(node, []):
                vals = [out[p] for p in rule.premises]
                scores.append(0.5 * min(vals) + 0.5 * float(np.mean(vals)))
            if scores:
                out[node] = max(scores)

    base = logits.copy()
    propagate_inplace(base, range(stream.n_base, stream.n_nodes))
    edited = base.copy()
    edited[changed] += perturbation
    full_result = edited.copy(); propagate_inplace(full_result, range(stream.n_base, stream.n_nodes))
    incremental_result = edited.copy(); propagate_inplace(incremental_result, ordered_closure)
    max_error = float(np.max(np.abs(full_result - incremental_result)))

    full_work = edited.copy()
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        propagate_inplace(full_work, range(stream.n_base, stream.n_nodes))
    full_ms = (time.perf_counter_ns() - t0) / repeats / 1e6
    incremental_work = edited.copy()
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        propagate_inplace(incremental_work, ordered_closure)
    incremental_ms = (time.perf_counter_ns() - t0) / repeats / 1e6
    return full_ms, incremental_ms, 1.0 - incremental_ms / full_ms, max_error


def run(seed: int, steps: int = 90) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    stream = bench.DynamicKnowledgeStream(seed=seed, steps=steps)
    modes = ("static", "global_decay", "stability_aware")
    time_records: list[dict[str, float]] = []
    intervention_cert: list[float] = []
    intervention_event_flip: list[float] = []
    intervention_window_flip: list[float] = []
    intervention_counter: list[float] = []
    drift_rows: list[dict[str, float]] = []
    full_work = []
    incremental_work = []
    for t in range(steps):
        truth = stream.step(t)
        states = {}
        for mode in modes:
            reasoner = bench.Reasoner(stream, mode)
            logits, stability = reasoner.infer(t)
            prob = np.asarray([bench.sigmoid(float(x)) for x in logits])
            pred = prob >= 0.5
            accepted = reasoner.accepts(logits, stability)
            mask = accepted
            risk = float(np.mean(pred[mask] != truth[mask])) if np.any(mask) else 0.0
            coverage = float(np.mean(mask))
            states[mode] = (logits, stability, pred, accepted, risk, coverage)
            time_records.append({"seed": seed, "time": t, "mode": mode, "risk": risk, "coverage": coverage, "drift": float(bool(stream.changed_bases))})
        # Exact closure accounting is independent of model output.
        aware = bench.Reasoner(stream, "stability_aware")
        if stream.changed_bases:
            full_work.append(float(stream.n_nodes))
            incremental_work.append(float(aware.recomputation_work(stream.changed_bases)))

        # A certificate/intervention record is collected on base facts that
        # have enough evidence for a meaningful deletion operation.
        logits, stability, _, _, _, _ = states["stability_aware"]
        for fact in range(stream.n_base):
            obs = stream.evidence[fact]
            if len(obs) < 2 or abs(float(logits[fact])) < 1.35:
                continue
            original_sign = float(logits[fact]) >= 0
            event_flip = 0
            for i in range(len(obs)):
                old = stream.evidence[fact]
                stream.evidence[fact] = old[:i] + old[i + 1 :]
                edited, _ = aware._base_state(fact, t)
                event_flip += int((edited >= 0) != original_sign)
                stream.evidence[fact] = old
            # Window deletion removes all reports in each occupied two-step
            # bucket.  The bucket width is fixed before evaluating flips.
            buckets = sorted({o.time // 7 for o in obs})
            window_flip = 0
            for bucket in buckets:
                old = stream.evidence[fact]
                stream.evidence[fact] = [o for o in old if o.time // 7 != bucket]
                edited, _ = aware._base_state(fact, t)
                window_flip += int((edited >= 0) != original_sign)
                stream.evidence[fact] = old
            intervention_cert.append(float(stability[fact]))
            intervention_event_flip.append(float(event_flip > 0))
            intervention_window_flip.append(float(window_flip > 0))
            intervention_counter.append(float(_counter_injection(stream, fact, t, original_sign)))

        if stream.changed_bases:
            # Measure how quickly each mode returns to a correct accepted state
            # for the facts changed by this drift event.  The first post-drift
            # step with >=80% correctness among accepted changed facts is the
            # recovery time; a missing crossing is recorded as the horizon.
            drift_set = np.asarray(list(stream.changed_bases), dtype=int)
            horizon = min(stream.drift_period, steps - t)
            # Each mode receives an isolated copy.  The main trajectory is
            # therefore unchanged while recovery is replayed from exactly the
            # same post-drift state.
            for mode in modes:
                replay = copy.deepcopy(stream)
                recovered = False
                for lag in range(horizon):
                    tt = t + lag
                    if lag > 0:
                        truth_replay = replay.step(tt)
                    else:
                        truth_replay = replay._hidden_derived_truth()
                    rr = bench.Reasoner(replay, mode)
                    ll, ss = rr.infer(tt)
                    aa = rr.accepts(ll[drift_set], ss[drift_set])
                    pp = ll[drift_set] >= 0
                    yy = truth_replay[drift_set]
                    if np.any(aa):
                        acc = float(np.mean(pp[aa] == yy[aa]))
                        if acc >= 0.8:
                            drift_rows.append({"seed": seed, "drift_time": t, "mode": mode, "recovery_lag": lag, "post_drift_accuracy": acc})
                            recovered = True
                            break
                if not recovered:
                    drift_rows.append({"seed": seed, "drift_time": t, "mode": mode, "recovery_lag": horizon, "post_drift_accuracy": 0.0})

    cert = np.asarray(intervention_cert)
    event = np.asarray(intervention_event_flip)
    window = np.asarray(intervention_window_flip)
    counter = np.asarray(intervention_counter)
    med = float(np.median(cert)) if len(cert) else 0.0
    full_ms, incremental_ms, runtime_reduction, max_error = _incremental_runtime(stream, seed)
    result = {
        "seed": seed,
        "n_interventions": int(len(cert)),
        "event_flip_rate": float(event.mean()) if len(event) else 0.0,
        "window_flip_rate": float(window.mean()) if len(window) else 0.0,
        "low_certificate_event_flip": float(event[cert < med].mean()) if np.any(cert < med) else 0.0,
        "high_certificate_event_flip": float(event[cert >= med].mean()) if np.any(cert >= med) else 0.0,
        "low_certificate_window_flip": float(window[cert < med].mean()) if np.any(cert < med) else 0.0,
        "high_certificate_window_flip": float(window[cert >= med].mean()) if np.any(cert >= med) else 0.0,
        "certificate_nonflip_corr": float(np.corrcoef(cert, 1.0 - window)[0, 1]) if len(cert) > 1 else 0.0,
        "median_counter_events": float(np.median(counter)) if len(counter) else 0.0,
        "full_nodes": float(np.mean(full_work)),
        "incremental_nodes": float(np.mean(incremental_work)),
        "node_reduction": float(1.0 - np.mean(incremental_work) / np.mean(full_work)),
        "full_update_ms": full_ms,
        "incremental_update_ms": incremental_ms,
        "runtime_reduction": runtime_reduction,
        "incremental_max_error": max_error,
    }
    return result, drift_rows, time_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--steps", type=int, default=90)
    p.add_argument("--output-dir", type=Path, default=Path("results_final"))
    args = p.parse_args()
    seed_rows = []
    drift_rows = []
    time_rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        result, drifts, times = run(seed, args.steps)
        seed_rows.append(result); drift_rows.extend(drifts); time_rows.extend(times)
        print(f"seed={seed} interventions={result['n_interventions']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "synthetic_seeds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0])); w.writeheader(); w.writerows(seed_rows)
    if drift_rows:
        with (args.output_dir / "synthetic_drift.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(drift_rows[0])); w.writeheader(); w.writerows(drift_rows)
    if time_rows:
        with (args.output_dir / "synthetic_time_series.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(time_rows[0])); w.writeheader(); w.writerows(time_rows)
    summary = {}
    for key in seed_rows[0]:
        if key == "seed":
            continue
        m, lo, hi = _bootstrap([float(r[key]) for r in seed_rows], 9000 + len(key))
        summary[key] = {"mean": m, "lo": lo, "hi": hi}
    with (args.output_dir / "synthetic_summary.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)
    # Drift summaries are grouped by mode and reported with seed-level means.
    if drift_rows:
        drift_summary = {}
        for mode in sorted({r["mode"] for r in drift_rows}):
            vals = [r["recovery_lag"] for r in drift_rows if r["mode"] == mode]
            drift_summary[mode] = {"n": len(vals), "mean_recovery_lag": float(np.mean(vals)), "median_recovery_lag": float(np.median(vals))}
        with (args.output_dir / "synthetic_drift_summary.json").open("w", encoding="utf-8") as f:
            import json
            json.dump(drift_summary, f, indent=2)


if __name__ == "__main__":
    main()
