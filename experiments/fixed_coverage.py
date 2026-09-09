"""Evaluate exactly equal accepted counts at prespecified test coverage budgets."""
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_results as a
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results_final_v5/analysis"
def risk(frame,key,target):
    order=np.argsort(-frame[key].to_numpy(),kind="mergesort")
    count=max(1,round(target*len(frame)))
    return float(1-frame.correct.to_numpy()[order[:count]].mean())
rows=[]
for dataset in ("ICEWS14","ICEWS05-15","GDELT"):
    f=pd.read_csv(ROOT/f"results_final_v5/{dataset}_predictions.csv")
    for target in (.2,.4):
        def contrast(sample):
            return risk(sample,"belief_stability",target)-risk(sample,"belief_baseline",target)
        point,low,high=a.block_bootstrap(f,contrast,614+int(target*100),repeats=1000)
        rows.append({"dataset":dataset,"coverage":target,"accepted_count":round(target*len(f)),
            "baseline_risk":risk(f,"belief_baseline",target),"stability_risk":risk(f,"belief_stability",target),
            "delta_risk":point,"ci_low":low,"ci_high":high})
pd.DataFrame(rows).to_csv(OUT/"fixed_count_comparison.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
