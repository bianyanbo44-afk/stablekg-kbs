"""Event-window intervention audit on the public temporal KG splits."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import run_public_benchmark_v2 as bench

def winner(events, timestamp, mode='decay', half_life=35.0):
    mass=defaultdict(float); bins=defaultdict(lambda: defaultdict(float))
    for t,e in events:
        if t>timestamp: continue
        w=1.0 if mode=='static' else math.exp(-math.log(2.0)*(timestamp-t)/half_life)
        mass[e]+=w; bins[e][t//7]+=w
    if not mass: return None,mass,bins
    w=min(mass,key=lambda e:(-mass[e],e)); return w,mass,bins

def run(root:Path, cap:int, entity_count:int|None, seed:int=0):
    train,valid,test=root/'train.jsonl',root/'valid.jsonl',root/'test.jsonl'
    index,n=bench.load_events(train); n=entity_count or n
    queries=list(bench.iter_pe(test,cap)); records=[]
    for q in queries:
        events=index.get((q.subject,q.relation),[])
        w,mass,bins=winner(events,q.timestamp)
        if w is None or not bins.get(w): continue
        runner=max((v for e,v in mass.items() if e!=w),default=0.0); margin=mass[w]-runner
        # certificate is the pre-intervention surviving evidence fraction.
        vals=sorted(bins[w].values(),reverse=True); removed=0.0; cost=1.0
        for v in vals:
            removed+=v
            if removed>=margin:
                cost=float(np.clip(1.0-removed/(mass[w]+1e-12),0,1)); break
        cert=float(np.clip(0.5*cost+0.5*(1-max(vals)/sum(vals)),0,1))
        flips=[]
        for bid in bins[w]:
            edited=[(t,e) for t,e in events if not (e==w and t//7==bid and t<=q.timestamp)]
            new_w,_,_=winner(edited,q.timestamp)
            flips.append(float(new_w!=w))
        records.append({'certificate':cert,'flip':float(any(flips)),'bin_flip_rate':float(np.mean(flips)),'correct':float(w in set(q.answers)),'n_bins':float(len(bins[w]))})
    return records

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--dataset',required=True); p.add_argument('--cap',type=int,default=2000); p.add_argument('--entity-count',type=int); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=run(a.data_root,a.cap,a.entity_count)
    cert=np.array([r['certificate'] for r in rows]); flip=np.array([r['flip'] for r in rows]);
    med=float(np.median(cert)) if len(cert) else 0
    summary={'dataset':a.dataset,'n_queries':len(rows),'overall_flip_rate':float(flip.mean()) if len(flip) else 0.0,'low_certificate_flip_rate':float(flip[cert<med].mean()) if np.any(cert<med) else 0.0,'high_certificate_flip_rate':float(flip[cert>=med].mean()) if np.any(cert>=med) else 0.0,'certificate_flip_corr':float(np.corrcoef(cert,1-flip)[0,1]) if len(rows)>1 else 0.0,'median_certificate':med,'winner_accuracy':float(np.mean([r['correct'] for r in rows])) if rows else 0.0,'mean_bin_flip_rate':float(np.mean([r['bin_flip_rate'] for r in rows])) if rows else 0.0}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
