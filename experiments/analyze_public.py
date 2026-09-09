"""Aggregate public benchmark CSVs and write publication-ready summary tables."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def mean_ci(values, z=1.96):
    x=np.asarray(values,dtype=float); m=float(x.mean()) if len(x) else 0.0
    if len(x)<2: return m,m,m
    se=float(x.std(ddof=1)/np.sqrt(len(x))); return m,m-z*se,m+z*se

def main():
    p=argparse.ArgumentParser(); p.add_argument('--inputs',nargs='+',required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    rows=[]
    for path in args.inputs:
        with open(path,encoding='utf-8') as f: rows.extend(csv.DictReader(f))
    datasets=sorted(set(r['dataset'] for r in rows)); out=[]
    for ds in datasets:
        sub=[r for r in rows if r['dataset']==ds]
        rec={'dataset':ds,'n_seeds':len(sub),'n_entities':float(sub[0]['n_entities']),'n_train_events':float(sub[0]['n_train_events']),'n_validation':float(sub[0]['n_validation']),'n_test':float(sub[0]['n_test'])}
        for metric in ['decay_mrr','decay_hits1','decay_hits3','decay_hits10','baseline_ece','stability_ece','baseline_aurc','stability_aurc']:
            m,lo,hi=mean_ci([float(r[metric]) for r in sub]); rec[metric]=m; rec[metric+'_lo']=lo; rec[metric+'_hi']=hi
        for target in ['0.20','0.40','0.60','0.80']:
            for prefix in ['baseline','stability']:
                for metric in ['risk','coverage','brier']:
                    m,lo,hi=mean_ci([float(r[f'{prefix}_{metric}@{target}']) for r in sub]); rec[f'{prefix}_{metric}@{target}']=m; rec[f'{prefix}_{metric}@{target}_lo']=lo; rec[f'{prefix}_{metric}@{target}_hi']=hi
        out.append(rec)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
