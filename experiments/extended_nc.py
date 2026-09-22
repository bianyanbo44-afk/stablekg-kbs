"""Predefined final controls: sharpness, probes, fusion weight and runtime."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from prepare_nc_cache import load
from evaluate_nc import observed_answers,admissible_probability,rank_row
from window_certificate import make_index,evidence_at,fused_score,certify
from certificate_controls import winner_only_certificate,random_probe_survival,dense_budget_certificate


def run(model,dataset,output):
    root=Path('results_nc/evaluation_renormalized')/model/dataset/'seed0'
    config=json.loads((root/'frozen_config.json').read_text());events,panel=load(dataset);index=make_index(events)
    matrix=np.load(Path('results_nc')/model/dataset/'seed0/test_scores.npy',mmap_mode='r')
    rng=np.random.default_rng(20260922);controls=[];sensitivity=[];timings=[]
    runtime_ids=set(np.linspace(0,len(panel['test'])-1,32,dtype=int))
    for i,(logits,q) in enumerate(zip(matrix,panel['test'])):
        obs=observed_answers(index,q);p=admissible_probability(logits,config['temperature'],obs)
        ev=evidence_at(index,q[0],q[1],q[2],excluded=obs)
        g=fused_score(p,ev,config['weight'],'renormalized');g[obs]=-np.inf
        exact=certify(g,ev,config['weight'],normalization='renormalized',probability=p)
        row={'model':model,'dataset':dataset,'query_id':i,'timestamp':q[2],'windows':len(ev.windows),
             'supported':ev.event_count>0,'winner':exact['winner'],'robust1':exact['robust1'],'robust2':exact['robust2']}
        for k in (1,2):
            row[f'winner_only{k}']=winner_only_certificate(g,ev,p,config['weight'],k)
            survival=random_probe_survival(g,ev,p,config['weight'],k,rng,repeats=64)
            for n in (8,32,64):row[f'probe{n}_safe{k}']=bool(survival[:n].all())
        fixed=certify(g,ev,config['weight'],normalization='fixed')
        row['fixed_denominator_robust1']=fixed['robust1'];controls.append(row)
        for weight in (0.,.1,.25,.5,.75):
            scores=fused_score(p,ev,weight,'renormalized');scores[obs]=-np.inf
            c=certify(scores,ev,weight,normalization='renormalized',probability=p)
            ranks=rank_row(scores,q)
            sensitivity.append({'model':model,'dataset':dataset,'query_id':i,'timestamp':q[2],'weight':weight,
                'mrr':float(np.mean(1/ranks)),'reciprocal_sum':float(np.sum(1/ranks)),'answer_count':len(q[3]),
                'correct':int(c['winner'] in q[4]),'robust1':c['robust1'],
                'robust2':c['robust2'],'changed_from_neural':int(c['winner']!=np.argmax(p)),'supported':ev.event_count>0})
        if i in runtime_ids:
            for repeat in range(3):
                def timed_dense():
                    start=time.perf_counter();value=dense_budget_certificate(g,ev,p,config['weight'])
                    return value,(time.perf_counter()-start)*1e6
                def timed_sparse():
                    start=time.perf_counter();value=certify(g,ev,config['weight'],normalization='renormalized',probability=p)
                    return value,(time.perf_counter()-start)*1e6
                if (i+repeat)%2:
                    dense,dense_us=timed_dense();sparse,sparse_us=timed_sparse()
                else:
                    sparse,sparse_us=timed_sparse();dense,dense_us=timed_dense()
                assert all(dense[k]==sparse[f'robust{k}'] for k in (1,2,3))
                timings.append({'model':model,'dataset':dataset,'query_id':i,'repeat':repeat,'supported':ev.event_count>0,
                                'windows':len(ev.windows),'candidates':len(p),'competitors':sparse['competitors'],
                                'dense_us':dense_us,'sparse_us':sparse_us})
        if (i+1)%1000==0:print(model,dataset,'extended',i+1,flush=True)
    stem=model+'_'+dataset
    pd.DataFrame(controls).to_csv(output/f'{stem}_controls.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(output/f'{stem}_sensitivity.csv',index=False)
    pd.DataFrame(timings).to_csv(output/f'{stem}_runtime.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results_nc/extended');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for model in ('TeRDy','TemporalComplEx'):
            for dataset in ('ICEWS14','ICEWS05-15'):run(model,dataset,out)
