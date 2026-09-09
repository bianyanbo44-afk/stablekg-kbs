"""Create the auditable statistics and source-data tables used by the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATASETS = ("ICEWS14", "ICEWS05-15", "GDELT")
METHODS = {"baseline": "belief_baseline", "StableKG": "belief_stability"}


def percentile_ci(values: Iterable[float], seed: int, repeats: int = 5000) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    if array.size == 1:
        value = float(array[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def contiguous_blocks(timestamp: pd.Series, target_blocks: int = 30) -> np.ndarray:
    values = timestamp.to_numpy(dtype=int)
    low = int(values.min())
    width = max(1, int(math.ceil((int(values.max()) - low + 1) / target_blocks)))
    return (values - low) // width


def aurc(correct: np.ndarray, belief: np.ndarray) -> float:
    order = np.argsort(-belief, kind="mergesort")
    errors = 1.0 - correct[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    seed: int,
    repeats: int = 1000,
) -> tuple[float, float, float]:
    work = frame.copy()
    work["_block"] = contiguous_blocks(work["timestamp"])
    groups = [part.index.to_numpy() for _, part in work.groupby("_block", sort=True)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[int(index)] for index in chosen])
        estimates[repeat] = statistic(work.loc[indices])
    point = statistic(work)
    finite = estimates[np.isfinite(estimates)]
    return point, float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def risk_at_threshold(frame: pd.DataFrame, key: str, threshold: float) -> float:
    accepted = frame[key].to_numpy(float) >= threshold
    if not accepted.any():
        return math.nan
    return float(np.mean(1.0 - frame.loc[accepted, "correct"].to_numpy(float)))


def make_public_tables(public_dir: Path, intervention_dir: Path, output: Path) -> dict[str, object]:
    headline_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    temporal_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    summary_bundle: dict[str, object] = {}

    for dataset_index, dataset in enumerate(DATASETS):
        summary = json.loads((public_dir / f"{dataset}_summary.json").read_text(encoding="utf-8"))
        seeds = pd.read_csv(public_dir / f"{dataset}_seeds.csv")
        predictions = pd.read_csv(public_dir / f"{dataset}_predictions.csv")
        summary_bundle[dataset] = summary
        headline_rows.append(
            {
                "dataset": dataset,
                "n_entities": summary["n_entities"],
                "n_train_events": summary["n_train_events"],
                "n_validation": summary["n_validation"],
                "n_test": summary["n_test"],
                "decay_mrr": summary["decay_mrr"],
                "decay_hits1": summary["decay_hits1"],
                "decay_hits3": summary["decay_hits3"],
                "decay_hits10": summary["decay_hits10"],
                "baseline_brier": summary["baseline_brier"],
                "stability_brier": summary["stability_brier"],
                "baseline_ece": summary["baseline_ece"],
                "stability_ece": summary["stability_ece"],
                "baseline_aurc": summary["baseline_aurc"],
                "stability_aurc": summary["stability_aurc"],
                "aurc_relative_reduction": (summary["baseline_aurc"] - summary["stability_aurc"]) / summary["baseline_aurc"],
                "baseline_risk@0.20": summary["baseline_risk@0.20"],
                "stability_risk@0.20": summary["stability_risk@0.20"],
                "risk_delta@0.20": summary["stability_risk@0.20"] - summary["baseline_risk@0.20"],
                "baseline_risk@0.40": summary["baseline_risk@0.40"],
                "stability_risk@0.40": summary["stability_risk@0.40"],
                "risk_delta@0.40": summary["stability_risk@0.40"] - summary["baseline_risk@0.40"],
            }
        )

        for method, key in METHODS.items():
            correct = predictions["correct"].to_numpy(float)
            belief = predictions[key].to_numpy(float)
            order = np.argsort(-belief, kind="mergesort")
            errors = 1.0 - correct[order]
            cumulative = np.cumsum(errors) / np.arange(1, len(errors) + 1)
            for coverage in np.linspace(0.02, 1.0, 50):
                count = max(1, int(round(coverage * len(predictions))))
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "coverage": count / len(predictions),
                        "risk": float(cumulative[count - 1]),
                    }
                )
            edges = np.linspace(0.0, 1.0, 11)
            for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (belief >= low) & (belief < high if high < 1.0 else belief <= high)
                if mask.any():
                    reliability_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "bin": bin_index,
                            "belief": float(belief[mask].mean()),
                            "accuracy": float(correct[mask].mean()),
                            "count": int(mask.sum()),
                        }
                    )

        seed0 = seeds.loc[seeds["seed"] == 0].iloc[0]
        block_values = contiguous_blocks(predictions["timestamp"], target_blocks=10)
        predictions = predictions.assign(time_block=block_values)
        for target in (0.20, 0.40):
            label = f"{target:.2f}"
            for method, key in METHODS.items():
                prefix = "baseline" if method == "baseline" else "stability"
                threshold = float(seed0[f"{prefix}_threshold@{label}"])
                for block, part in predictions.groupby("time_block", sort=True):
                    accepted = part[key].to_numpy(float) >= threshold
                    temporal_rows.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "method": method,
                            "time_block": int(block),
                            "timestamp_min": int(part["timestamp"].min()),
                            "timestamp_max": int(part["timestamp"].max()),
                            "n_queries": len(part),
                            "coverage": float(accepted.mean()),
                            "risk": float(np.mean(1.0 - part.loc[accepted, "correct"])) if accepted.any() else math.nan,
                        }
                    )

            base_threshold = float(seed0[f"baseline_threshold@{label}"])
            stable_threshold = float(seed0[f"stability_threshold@{label}"])

            def risk_delta(sample: pd.DataFrame) -> float:
                stable_risk = risk_at_threshold(sample, "belief_stability", stable_threshold)
                base_risk = risk_at_threshold(sample, "belief_baseline", base_threshold)
                return stable_risk - base_risk

            estimate, low, high = block_bootstrap(
                predictions,
                risk_delta,
                seed=12_000 + dataset_index * 10 + int(target * 10),
            )
            bootstrap_rows.append(
                {
                    "dataset": dataset,
                    "metric": f"risk_delta@{label}",
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "unit": "absolute risk",
                    "bootstrap_unit": "contiguous timestamp block",
                }
            )

        def aurc_delta(sample: pd.DataFrame) -> float:
            correct = sample["correct"].to_numpy(float)
            return aurc(correct, sample["belief_stability"].to_numpy(float)) - aurc(
                correct, sample["belief_baseline"].to_numpy(float)
            )

        estimate, low, high = block_bootstrap(predictions, aurc_delta, seed=13_000 + dataset_index)
        bootstrap_rows.append(
            {
                "dataset": dataset,
                "metric": "aurc_delta",
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "unit": "AURC",
                "bootstrap_unit": "contiguous timestamp block",
            }
        )

        ablations = ("belief_only", "plus_certificate", "plus_diversity", "plus_fragility", "full_certificate")
        for ablation_index, ablation in enumerate(ablations):
            for metric in ("aurc", "risk@0.40"):
                values = seeds[f"{ablation}_{metric}"].to_numpy(float)
                mean, low, high = percentile_ci(values, 14_000 + dataset_index * 100 + ablation_index * 10 + len(metric))
                ablation_rows.append(
                    {
                        "dataset": dataset,
                        "ablation": ablation,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

        interventions = pd.read_csv(intervention_dir / f"{dataset}_interventions.csv")
        intervention_summary = json.loads(
            (intervention_dir / f"{dataset}_interventions.json").read_text(encoding="utf-8")
        )
        q25 = float(intervention_summary["certificate_q25"])
        q75 = float(intervention_summary["certificate_q75"])
        for edit_index, edit in enumerate(("event", "window", "counter")):
            survival = 1.0 - interventions[f"{edit}_flip"].to_numpy(float)
            rho, pvalue = spearmanr(interventions["certificate"].to_numpy(float), survival)
            for group, mask in (
                ("low certificate", interventions["certificate"].to_numpy(float) <= q25),
                ("high certificate", interventions["certificate"].to_numpy(float) >= q75),
            ):
                values = interventions.loc[mask, f"{edit}_flip"].to_numpy(float)
                rng = np.random.default_rng(15_000 + dataset_index * 100 + edit_index * 10 + len(group))
                sample = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
                intervention_rows.append(
                    {
                        "dataset": dataset,
                        "intervention": edit,
                        "certificate_group": group,
                        "n": len(values),
                        "flip_rate": float(values.mean()),
                        "ci_low": float(np.quantile(sample, 0.025)),
                        "ci_high": float(np.quantile(sample, 0.975)),
                        "survival_spearman": float(rho),
                        "spearman_p": float(pvalue),
                    }
                )

        # The intervention runner retains the source-order query id before
        # dropping queries with no causal evidence, whereas the benchmark
        # prediction table numbers only retained rows.  Joining on the
        # semantic query key prevents silent row misalignment in this
        # secondary sensitivity analysis.
        join_keys = ["subject", "relation", "timestamp", "winner"]
        intervention_keys = interventions[join_keys + ["window_flip"]].copy()
        duplicate_keys = intervention_keys.duplicated(join_keys, keep=False)
        if bool(duplicate_keys.any()):
            raise RuntimeError(f"Duplicate intervention query keys for {dataset}")
        joined = predictions.merge(intervention_keys, on=join_keys, how="inner", validate="one_to_one")
        for target in (0.20, 0.40):
            label = f"{target:.2f}"
            for method, key in METHODS.items():
                prefix = "baseline" if method == "baseline" else "stability"
                threshold = float(seed0[f"{prefix}_threshold@{label}"])
                accepted = joined[key].to_numpy(float) >= threshold
                sensitivity_rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "method": method,
                        "n_supported": len(joined),
                        "n_accepted": int(accepted.sum()),
                        "accepted_window_flip_rate": float(joined.loc[accepted, "window_flip"].mean()) if accepted.any() else math.nan,
                    }
                )

    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "headline_public.csv": headline_rows,
        "risk_coverage.csv": curve_rows,
        "reliability.csv": reliability_rows,
        "temporal_blocks.csv": temporal_rows,
        "paired_block_bootstrap.csv": bootstrap_rows,
        "intervention_statistics.csv": intervention_rows,
        "accepted_window_sensitivity.csv": sensitivity_rows,
        "ablation_statistics.csv": ablation_rows,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / name, index=False)
    (output / "public_summary_bundle.json").write_text(json.dumps(summary_bundle, indent=2), encoding="utf-8")
    return summary_bundle


def merge_synthetic(batch_root: Path, output: Path) -> dict[str, object]:
    seed_frames = []
    drift_frames = []
    time_frames = []
    for batch_index in range(4):
        root = batch_root / f"batch{batch_index}"
        seed_frames.append(pd.read_csv(root / "synthetic_seeds.csv"))
        drift_frames.append(pd.read_csv(root / "synthetic_drift.csv"))
        time_frames.append(pd.read_csv(root / "synthetic_time_series.csv"))
    seeds = pd.concat(seed_frames, ignore_index=True).sort_values("seed")
    if seeds["seed"].tolist() != list(range(20)):
        raise RuntimeError("Synthetic batches must contain exactly seeds 0 through 19")
    if float(seeds["incremental_max_error"].max()) != 0.0:
        raise RuntimeError("Incremental recomputation diverged from full recomputation")
    drift = pd.concat(drift_frames, ignore_index=True).sort_values(["seed", "drift_time", "mode"])
    time_series = pd.concat(time_frames, ignore_index=True).sort_values(["seed", "time", "mode"])
    output.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(output / "synthetic_seeds_20.csv", index=False)
    drift.to_csv(output / "synthetic_drift_20.csv", index=False)
    time_series.to_csv(output / "synthetic_time_series_20.csv", index=False)
    summary: dict[str, object] = {"n_seeds": 20}
    for column_index, column in enumerate(seeds.columns):
        if column == "seed":
            continue
        mean, low, high = percentile_ci(seeds[column].to_numpy(float), 20_000 + column_index)
        summary[column] = {"mean": mean, "ci_low": low, "ci_high": high}
    (output / "synthetic_summary_20.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_neural(neural_dir: Path, output: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(("ICEWS14", "ICEWS05-15")):
        path = neural_dir / f"{dataset}_seeds.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        dataset_summary: dict[str, object] = {"n_seeds": len(frame)}
        metrics = (
            "test_mrr",
            "test_hits1",
            "test_hits3",
            "test_hits10",
            "baseline_brier",
            "stability_brier",
            "baseline_ece",
            "stability_ece",
            "baseline_aurc",
            "stability_aurc",
            "baseline_risk@0.20",
            "stability_risk@0.20",
            "baseline_risk@0.40",
            "stability_risk@0.40",
        )
        for metric_index, metric in enumerate(metrics):
            mean, low, high = percentile_ci(
                frame[metric].to_numpy(float),
                30_000 + dataset_index * 100 + metric_index,
            )
            dataset_summary[metric] = {"mean": mean, "ci_low": low, "ci_high": high}
        result[dataset] = dataset_summary
        rows.append(
            {
                "dataset": dataset,
                "n_seeds": len(frame),
                **{metric: dataset_summary[metric]["mean"] for metric in metrics},
                "aurc_delta": dataset_summary["stability_aurc"]["mean"] - dataset_summary["baseline_aurc"]["mean"],
                "risk_delta@0.20": dataset_summary["stability_risk@0.20"]["mean"] - dataset_summary["baseline_risk@0.20"]["mean"],
                "risk_delta@0.40": dataset_summary["stability_risk@0.40"]["mean"] - dataset_summary["baseline_risk@0.40"]["mean"],
            }
        )
    pd.DataFrame(rows).to_csv(output / "neural_summary.csv", index=False)
    (output / "neural_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", type=Path, default=Path("results_final_v4"))
    parser.add_argument("--intervention-dir", type=Path, default=Path("results_final"))
    parser.add_argument("--batch-root", type=Path, default=Path("results_final"))
    parser.add_argument("--neural-dir", type=Path, default=Path("results_neural"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_final_v4/analysis"))
    args = parser.parse_args()
    merge_synthetic(args.batch_root, args.output_dir)
    make_public_tables(args.public_dir, args.intervention_dir, args.output_dir)
    summarize_neural(args.neural_dir, args.output_dir)
    print(f"Wrote analysis tables to {args.output_dir}")


if __name__ == "__main__":
    main()
