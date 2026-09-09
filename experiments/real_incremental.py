"""End-to-end public-event maintenance with an equally indexed full-refresh control.

Both engines maintain exactly the same raw time multiplicities and weighted
candidate/bin sufficient statistics. Full refresh re-evaluates every registered
subject/relation decision from these summaries (it does NOT rescan the raw graph).
Incremental refresh finds affected decisions during writes. Advancing the clock
refreshes all decisions, because unnormalised margins and recency both decay.
"""
from __future__ import annotations
import argparse, gc, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
import stability_benchmark as b
from prepare_public_cache import load

FEATURES=['margin','log_support','recency','certificate','diversity','fragility_cost','counter_evidence_cost']

class Candidate:
    __slots__=('times','bins','mass','count','last')
    def __init__(self):
        self.times=Counter();self.bins=Counter();self.mass=0.;self.count=0;self.last=-1
    def edit(self,t,delta,w):
        if self.times[t]+delta<0: raise ValueError('Deleting absent event')
        self.times[t]+=delta
        if not self.times[t]: del self.times[t]
        self.count+=delta; self.mass+=delta*w; self.bins[t//7]+=delta*w
        if delta<0 and not any(k//7==t//7 for k in self.times): self.bins.pop(t//7,None)
        if not self.count:
            self.mass=0.;self.bins.clear();self.last=-1
        elif delta>0: self.last=max(self.last,t)
        elif t==self.last and t not in self.times: self.last=max(self.times)

class IndexedEngine:
    def __init__(self,events,keys,anchor,calibrator,threshold):
        self.anchor=anchor;self.clock=anchor;self.rate=math.log(2)/35
        self.keys=list(keys);self.reverse={k:i for i,k in enumerate(keys)}
        self.streams={k:{} for k in keys};self.calibrator=calibrator;self.threshold=threshold
        self.outputs=np.zeros((len(keys),11));self.event_count=0
        for s,r,o,t in events: self._edit((int(s),int(r),int(o),int(t)),1)
        self.refresh(range(len(self.keys)))
    def _edit(self,event,delta):
        s,r,o,t=event;key=(s,r)
        if key not in self.reverse: raise ValueError('Unregistered query key')
        if t>self.clock: raise ValueError('Future event must be preceded by clock advance')
        state=self.streams[key]
        if o not in state:
            if delta<0: raise ValueError('Deleting absent candidate')
            state[o]=Candidate()
        state[o].edit(t,delta,math.exp(self.rate*(t-self.anchor)))
        if state[o].count==0: del state[o]
        self.event_count+=delta
        return self.reverse[key]
    def features(self,i):
        state=self.streams[self.keys[i]]
        ranked=sorted(state,key=lambda o:(-state[o].mass,-state[o].count,o))
        if not ranked: return [0.,0.,0.,0.,0.,0.,0.],0,0
        winner=ranked[0];w=state[winner]
        runner=state[ranked[1]].mass if len(ranked)>1 else 0.
        margin=max(0.,w.mass-runner);factor=math.exp(-self.rate*(self.clock-self.anchor))
        k,f=b.window_deletion_certificate((v*factor for v in w.bins.values()),margin*factor)
        d=1.-max(w.bins.values())/sum(w.bins.values()) if w.mass>0 else 0.
        z=margin/(w.mass+runner+1e-12/factor)
        recency=math.exp(-self.rate*(self.clock-w.last))
        return [margin*factor,math.log1p(w.count),recency,.5*(f+d),d,f,z],winner,k
    def refresh(self,indices):
        ids=list(indices)
        if not ids:return
        rows=[];winners=[];counts=[]
        for i in ids:
            features,w,k=self.features(i);rows.append(features);winners.append(w);counts.append(k)
        x=np.asarray(rows)
        model=self.calibrator
        raw=np.c_[np.ones(len(ids)),(x-model.mean)/model.scale]@model.weight
        belief=1/(1+np.exp(-np.clip(raw,-40,40)))
        self.outputs[ids]=np.c_[winners,counts,x,belief,belief>=self.threshold]
    def apply(self,edits,clock,incremental):
        start=time.perf_counter_ns()
        advanced=clock!=self.clock
        if clock<self.clock:raise ValueError('Clock cannot go backwards')
        self.clock=clock
        touched=set()
        for event,delta in edits: touched.add(self._edit(event,delta))
        write_end=time.perf_counter_ns()
        ids=range(len(self.keys)) if advanced or not incremental else sorted(touched)
        self.refresh(ids)
        end=time.perf_counter_ns()
        return {'elapsed_ms':(end-start)/1e6,'write_lookup_ms':(write_end-start)/1e6,
                'refresh_ms':(end-write_end)/1e6,'refreshed':len(ids),'affected':len(touched)}

def calibrator(name):
    path=Path('results_review_20260909')/f'{name}_valid_features.csv'
    df=pd.read_csv(path)
    if 'prefix_cost_checked' in df:
        df['fragility_cost']=df.prefix_cost_checked
        df['certificate']=.5*(df.fragility_cost+df.diversity)
    rows=df.to_dict('records');order=np.random.default_rng(10000).permutation(len(rows));mid=len(order)//2
    model=b.RidgeLogistic(FEATURES).fit([rows[i] for i in order[:mid]])
    operating=[rows[i] for i in order[mid:]];p=model.predict(operating)
    threshold=float(np.sort(p)[::-1][max(0,int(.2*len(p))-1)])
    return model,threshold

def run(name,args):
    events,panel=load(name);pool=np.asarray(panel['validation_event_pool'],dtype=np.int64)
    keys=sorted({(int(s),int(r)) for s,r,_,_ in events}|{(int(s),int(r)) for s,r,_,_ in pool})
    anchor=max(int(events[:,3].max()),int(pool[:,3].max()))
    model,threshold=calibrator(name)
    proc=psutil.Process();gc.collect();rss0=proc.memory_info().rss;start=time.perf_counter()
    inc=IndexedEngine(events,keys,anchor,model,threshold)
    build_seconds=time.perf_counter()-start;rss1=proc.memory_info().rss
    full=IndexedEngine(events,keys,anchor,model,threshold)
    assert np.array_equal(inc.outputs,full.outputs)
    rng=np.random.default_rng(args.seed);records=[];trial=0
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for kind in ['insert','delete','counter','clock']:
        for batch in ([0] if kind=='clock' else args.batches):
            for repeat in range(args.repeats):
                if kind=='insert':
                    edits=[(tuple(map(int,e)),1) for e in pool[rng.choice(len(pool),batch,replace=False)]]
                elif kind=='delete':
                    # Pick distinct available original facts; earlier inserts/deletes remain.
                    edits=[];used=set()
                    while len(edits)<batch:
                        e=tuple(map(int,events[rng.integers(len(events))]))
                        if e in used:continue
                        s,r,o,t=e;candidate=inc.streams[(s,r)].get(o)
                        if candidate is not None and candidate.times.get(t,0)>0:
                            edits.append((e,-1));used.add(e)
                elif kind=='counter':
                    selected=rng.choice(len(keys),batch,replace=False)
                    edits=[]
                    for i in selected:
                        s,r=keys[i];winner=int(inc.outputs[i,0]);o=(winner+1+int(rng.integers(panel['n_entities']-1)))%panel['n_entities']
                        edits.append(((s,r,o,inc.clock),1))
                else: edits=[]
                clock=inc.clock+(1 if kind=='clock' else 0)
                # Alternate order to reduce systematic cache/thermal ordering effects.
                if trial%2:
                    f=full.apply(edits,clock,False);v=inc.apply(edits,clock,True)
                else:
                    v=inc.apply(edits,clock,True);f=full.apply(edits,clock,False)
                difference=np.abs(inc.outputs-full.outputs)
                mismatches=int(np.count_nonzero(inc.outputs[:,[0,1,10]]!=full.outputs[:,[0,1,10]]))
                if mismatches or difference.max()>1e-10:raise AssertionError((mismatches,float(difference.max())))
                row={'dataset':name,'kind':kind,'batch':batch,'repeat':repeat,'n_keys':len(keys),'events':inc.event_count,
                     'max_abs_error':float(difference.max()),'discrete_mismatches':mismatches,
                     'process_rss_mb':proc.memory_info().rss/1024**2,
                     **{'incremental_'+k:x for k,x in v.items()},**{'full_'+k:x for k,x in f.items()}}
                records.append(row);trial+=1
            pd.DataFrame(records).to_csv(output/f'{name}_trials.csv',index=False)
            print(f'{name} {kind} batch={batch}: full={f["elapsed_ms"]:.2f}ms inc={v["elapsed_ms"]:.2f}ms nodes={v["refreshed"]}/{len(keys)}',flush=True)
    # Independent raw-event oracle on a deterministic sample after all operations.
    oracle_error=0.;oracle_discrete=0
    for i in np.linspace(0,len(keys)-1,min(128,len(keys)),dtype=int):
        raw=sorted((t,o) for o,c in inc.streams[keys[i]].items() for t,n in c.times.items() for _ in range(n))
        s,r=keys[i];feat,_,_=b.winner_features(b.Query(s,r,inc.clock,(0,)),raw,panel['n_entities'],'decay',35.)
        actual=inc.outputs[i]
        oracle_error=max(oracle_error,float(np.max(np.abs(actual[2:9]-[feat[k] for k in FEATURES]))))
        mismatch=int(actual[0]!=feat['winner'])+int(actual[1]!=feat['window_count'])
        oracle_discrete+=mismatch
        if mismatch: print('oracle mismatch',keys[i],actual[:2],feat['winner'],feat['window_count'],flush=True)
    meta={'dataset':name,'initial_events':len(events),'registered_decisions':len(keys),'build_seconds':build_seconds,
          'incremental_engine_resident_mb':max(0,rss1-rss0)/1024**2,'memory_measurement':'Process RSS delta for one engine; raw data and Python baseline excluded',
          'oracle_max_feature_error':oracle_error,'oracle_discrete_mismatches':oracle_discrete,
          'threshold':threshold,'seed':args.seed,'repeats':args.repeats,'clock_policy':'refresh all on global clock advance',
          'scope':'Event writes, reverse lookup, affected-set construction, summary maintenance, candidate ranking, certificate, belief and acceptance; excludes file I/O and one-time build'}
    if oracle_discrete or oracle_error>1e-8:raise AssertionError(meta)
    (output/f'{name}_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['ICEWS14','ICEWS05-15','GDELT'])
    p.add_argument('--batches',nargs='+',type=int,default=[1,10,100,1000]);p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--seed',type=int,default=20260909);p.add_argument('--output',default='results_real_updates')
    args=p.parse_args()
    with threadpool_limits(limits=1):
        for name in args.datasets: run(name,args)
