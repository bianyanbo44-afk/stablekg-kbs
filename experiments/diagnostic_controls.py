"""Compare stability diagnostics with margin/support controls on held-out edits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import analyze_results as a
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_review_20260909/corrected"

def auc(y, score):
    n1 = int(y.sum()); n0 = len(y)-n1
    if min(n1, n0) == 0:
        return np.nan
    return float((rankdata(score)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))

rows, intervals = [], []
for dataset in ("ICEWS14","ICEWS05-15","GDELT"):
    frame = pd.read_csv(ROOT / f"results_interventions_v5/{dataset}_interventions.csv")
    features = pd.read_csv(OUT / f"{dataset}_predictions.csv")
    frame["margin"] = features.margin.to_numpy()[frame.query_id]
    frame["window_count"] = features.window_count.to_numpy()[frame.query_id]
    for edit in ("event", "window", "counter"):
        y = 1 - frame[f"{edit}_flip"].to_numpy(int)
        for key in ("certificate","counter_evidence_cost","margin","support","diversity","window_count"):
            rows.append({"dataset":dataset,"edit":edit,"feature":key,"n":len(frame),
                "survival_auroc":auc(y, frame[key].to_numpy()),
                "survival_spearman":spearmanr(y, frame[key]).statistic})
        def contrast(sample):
            y = 1-sample[f"{edit}_flip"].to_numpy(int)
            return auc(y,sample.certificate.to_numpy())-auc(y,sample.counter_evidence_cost.to_numpy())
        point, low, high = a.block_bootstrap(frame, contrast, 981, repeats=1000)
        intervals.append({"dataset":dataset,"edit":edit,"delta_auroc_c_minus_z":point,"ci_low":low,"ci_high":high})
pd.DataFrame(rows).to_csv(OUT / "diagnostic_controls.csv",index=False)
pd.DataFrame(intervals).to_csv(OUT / "diagnostic_contrasts.csv",index=False)
print(pd.DataFrame(rows).pivot(index=["dataset","edit"],columns="feature",values="survival_auroc").round(4).to_string())
