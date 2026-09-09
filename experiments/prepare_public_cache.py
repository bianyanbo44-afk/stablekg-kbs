"""Lossless Pe training cache and the fixed 5,000-query evaluation panels."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import stability_benchmark as b

SPECS = {'ICEWS14':('ICEWS14_all',7128), 'ICEWS05-15':('ICEWS05_15_all',10488), 'GDELT':('GDELT_all',500)}

def prepare(name):
    out=Path('data/processed_recent')/name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'panel.json').exists() and (out/'train.npy').exists(): return out
    folder,ne=SPECS[name]; root=Path('data/raw')/folder
    ts=time.perf_counter()
    train=list(b.iter_pe(root/'train.jsonl'))
    valid_all=list(b.iter_pe(root/'valid.jsonl'))
    test_all=list(b.iter_pe(root/'test.jsonl'))
    valid=b.evenly_sample(valid_all,5000); test=b.evenly_sample(test_all,5000)
    keys={q.key for q in valid+test}; known={k:set() for k in keys}
    for q in train+valid_all+test_all:
        if q.key in keys: known[q.key].update(q.answers)
    arr=np.array([(q.subject,q.relation,o,q.timestamp) for q in train for o in q.answers],dtype=np.int64)
    np.save(out/'train.npy',arr)
    def pack(q): return [q.subject,q.relation,q.timestamp,list(q.answers),sorted(known[q.key])]
    panel={'dataset':name,'n_entities':ne,'n_relations':1+max(q.relation for q in train+valid_all+test_all),
           'n_times':1+max(q.timestamp for q in train+valid_all+test_all),
           'training_events':len(arr),'validation':[pack(q) for q in valid],'test':[pack(q) for q in test],
           'validation_event_pool':[[q.subject,q.relation,o,q.timestamp] for q in valid_all for o in q.answers],
           'protocol':'Published Pe queries; fixed evenly spaced cap=5000; score-only filtered average ties'}
    (out/'panel.json').write_text(json.dumps(panel),encoding='utf-8')
    print(f'{name}: cached {len(arr)} train events in {time.perf_counter()-ts:.1f}s',flush=True)
    return out

def load(name):
    root=prepare(name)
    return np.load(root/'train.npy'),json.loads((root/'panel.json').read_text(encoding='utf-8'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=list(SPECS));a=p.parse_args()
    for name in a.datasets: prepare(name)
