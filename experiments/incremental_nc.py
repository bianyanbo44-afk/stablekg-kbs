"""Exact local maintenance of renormalized neural/evidence predictions."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from prepare_nc_cache import load
from window_certificate import make_index,evidence_at,fused_score,certify
from evaluate_nc import observed_answers,admissible_probability
from maintenance_certificate import certificate_state


class NeuralEvidenceCache:
    def __init__(self,events,queries,logits,temperature,weight):
        index=make_index(events);self.queries=queries;self.weight=weight
        self.reverse={};self.evidence=[];self.observed=[];self.probabilities=[]
        for i,q in enumerate(queries):
            obs=observed_answers(index,q);self.observed.append(set(map(int,obs)))
            self.evidence.append(evidence_at(index,q[0],q[1],q[2],excluded=obs))
            self.probabilities.append(admissible_probability(logits[i],temperature,obs))
            self.reverse.setdefault((q[0],q[1]),[]).append(i)
        self.outputs=np.zeros((len(queries),9));self.refresh(range(len(queries)))

    def refresh(self,ids):
        for i in ids:
            p,ev=self.probabilities[i],self.evidence[i]
            g=fused_score(p,ev,self.weight,'renormalized');g[list(self.observed[i])]=-np.inf
            self.outputs[i]=certificate_state(g,ev,p,self.weight)

    def write(self,event,delta):
        s,r,o,t=map(int,event);affected=[]
        for i in self.reverse.get((s,r),[]):
            q=self.queries[i]
            if t>q[2] or o in self.observed[i]:continue
            ev=self.evidence[i];window=t//7
            if o not in ev.objects:
                if delta<0:raise ValueError('Removing absent object')
                ev.objects=np.r_[ev.objects,o];ev.mass=np.c_[ev.mass,np.zeros(len(ev.windows))]
                ev.counts=np.c_[ev.counts,np.zeros(len(ev.windows),dtype=np.int64)] if ev.counts is not None else np.zeros(ev.mass.shape,dtype=np.int64)
            if window not in ev.windows:
                if delta<0:raise ValueError('Removing absent window')
                ev.windows=np.r_[ev.windows,window];ev.mass=np.r_[ev.mass,np.zeros((1,len(ev.objects)))]
                ev.counts=np.r_[ev.counts,np.zeros((1,len(ev.objects)),dtype=np.int64)]
            oi=int(np.flatnonzero(ev.objects==o)[0]);wi=int(np.flatnonzero(ev.windows==window)[0])
            if ev.counts[wi,oi]+delta<0:raise ValueError('Removing absent event multiplicity')
            ev.counts[wi,oi]+=delta;ev.mass[wi,oi]+=delta*np.exp2(-(q[2]-t)/35.)
            if ev.counts[wi,oi]==0:ev.mass[wi,oi]=0.
            ev.event_count+=delta
            valid_w=ev.counts.sum(axis=1)>0;valid_o=ev.counts.sum(axis=0)>0
            ev.windows=ev.windows[valid_w];ev.objects=ev.objects[valid_o]
            ev.mass=ev.mass[np.ix_(valid_w,valid_o)];ev.counts=ev.counts[np.ix_(valid_w,valid_o)]
            # Canonical ordering yields deterministic tie decisions in witnesses.
            wi_order=np.argsort(ev.windows);oi_order=np.argsort(ev.objects)
            ev.windows=ev.windows[wi_order];ev.objects=ev.objects[oi_order]
            ev.mass=ev.mass[np.ix_(wi_order,oi_order)];ev.counts=ev.counts[np.ix_(wi_order,oi_order)]
            ev.reference_mass=float(ev.mass.sum());affected.append(i)
        return affected

    def apply(self,edits,incremental):
        start=time.perf_counter();affected=set()
        for event,delta in edits:affected.update(self.write(event,delta))
        written=time.perf_counter();ids=sorted(affected) if incremental else range(len(self.queries))
        self.refresh(ids);end=time.perf_counter()
        return {'elapsed_ms':(end-start)*1000,'write_ms':(written-start)*1000,'refresh_ms':(end-written)*1000,
                'affected':len(affected),'refreshed':len(ids)}


def benchmark(dataset,repeats,output):
    events,panel=load(dataset);source=Path('results_nc/TeRDy')/dataset/'seed0'
    queries=panel['validation']
    if dataset=='GDELT':
        # Runtime-only control: a measured public graph with a fixed, empirical
        # relation-frequency prior. This is not reported as a neural benchmark.
        counts=np.ones((panel['n_relations'],panel['n_entities']),dtype=np.float64)
        np.add.at(counts,(events[:,1],events[:,2]),1)
        logits=np.log(counts[[q[1] for q in queries]])
        config={'temperature':1.,'weight':.5};anchor='Training relation-frequency prior; cache-scaling experiment only.'
    else:
        config=json.loads((Path('results_nc/evaluation_renormalized/TeRDy')/dataset/'seed0/frozen_config.json').read_text())
        logits=np.load(source/'validation_scores.npy',mmap_mode='r');anchor='Validation-selected TeRDy seed 0.'
    start=time.perf_counter();rss=psutil.Process().memory_info().rss
    inc=NeuralEvidenceCache(events,queries,logits,config['temperature'],config['weight'])
    build=time.perf_counter()-start;memory=(psutil.Process().memory_info().rss-rss)/2**20
    array_bytes=inc.outputs.nbytes+sum(p.nbytes for p in inc.probabilities)
    array_bytes+=sum(ev.objects.nbytes+ev.windows.nbytes+ev.mass.nbytes+(ev.counts.nbytes if ev.counts is not None else 0) for ev in inc.evidence)
    full=copy.deepcopy(inc);rng=np.random.default_rng(20260922);records=[]
    # Each delete/restore pair starts from the same complete public history.
    # Query keys and timestamps, rather than outcomes, determine affected sets.
    for batch in (1,10,100,1000):
        for repeat in range(repeats):
            ids=rng.choice(len(events),batch,replace=False)
            for kind,delta in [('delete',-1),('restore',1)]:
                edits=[(events[i],delta) for i in ids]
                if repeat%2:
                    f=full.apply(edits,False);v=inc.apply(edits,True)
                else:v=inc.apply(edits,True);f=full.apply(edits,False)
                error=float(np.max(np.abs(full.outputs-inc.outputs)))
                if not np.allclose(full.outputs,inc.outputs,atol=2e-11,rtol=0):raise AssertionError(error)
                records.append({'dataset':dataset,'batch':batch,'repeat':repeat,'kind':kind,'max_abs_error':error,
                    **{'full_'+k:x for k,x in f.items()},**{'incremental_'+k:x for k,x in v.items()}})
        pd.DataFrame(records).to_csv(output/f'{dataset}_trials.csv',index=False)
        print(dataset,batch,'complete',flush=True)
    # An independent raw-event rebuild after restored edits must agree as well.
    index=make_index(events);oracle_error=0.;oracle_discrete=0
    for i in np.linspace(0,len(queries)-1,128,dtype=int):
        q=queries[i];ev=evidence_at(index,q[0],q[1],q[2],excluded=list(inc.observed[i]))
        p=inc.probabilities[i];g=fused_score(p,ev,config['weight'],'renormalized');g[list(inc.observed[i])]=-np.inf
        c=certify(g,ev,config['weight'],normalization='renormalized',probability=p)
        expected=np.array([c['winner'],c['min_delete'],c['robust1'],c['robust2'],c['robust3'],
                           c['residual1'],c['residual2'],c['residual3'],g[c['winner']]])
        oracle_error=max(oracle_error,float(np.max(np.abs(expected-inc.outputs[i]))))
        oracle_discrete+=int(np.any(expected[:5]!=inc.outputs[i,:5]))
    assert oracle_discrete==0 and oracle_error<1e-9
    (output/f'{dataset}_metadata.json').write_text(json.dumps({'dataset':dataset,'events':len(events),
        'registered_queries':len(queries),'build_seconds':build,'one_cache_rss_mb':memory,
        'cache_numeric_arrays_mb':array_bytes/2**20,'cache_numeric_scope':'Probability, output, window, object, mass and multiplicity arrays; excludes Python containers and reverse-index objects.','anchor':anchor,
        'oracle_max_abs_error':oracle_error,'oracle_discrete_mismatches':oracle_discrete,'repeats':repeats,
        'control':'Both paths apply identical event writes and maintain identical cached window sufficient statistics.',
        'timed_scope':'Event writes, reverse lookup, affected-query construction, probability fusion, winner and certificate refresh; encoder forward and cache construction excluded.'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['ICEWS14','ICEWS05-15','GDELT']);p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--output',default='results_nc/updates');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        for dataset in a.datasets:benchmark(dataset,a.repeats,out)
