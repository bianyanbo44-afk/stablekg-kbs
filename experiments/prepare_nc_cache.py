"""Strict split-specific truth sets for Neurocomputing model selection.

Training and validation facts define validation labels/filtering. Full known
truth is used only for final test evaluation. The upstream cache is read-only.
"""
from pathlib import Path
import copy,json
from prepare_public_cache import load as load_public

def load(dataset):
    events,panel=load_public(dataset)
    panel=copy.deepcopy(panel)
    keys={tuple(q[:3]) for q in panel['validation']}
    known={k:set() for k in keys}
    for s,r,o,t in events:
        key=(int(s),int(r),int(t))
        if key in known:known[key].add(int(o))
    for s,r,o,t in panel['validation_event_pool']:
        key=(int(s),int(r),int(t))
        if key in known:known[key].add(int(o))
    changed=0;removed=0
    for q in panel['validation']:
        safe=known[tuple(q[:3])];assert set(q[3])<=safe
        difference=set(q[4])-safe;changed+=bool(difference);removed+=len(difference)
        q[4]=sorted(safe)
    panel['truth_protocol']='Validation: training plus validation only. Test: complete published known truth.'
    panel['validation_removed_test_only_answers']=removed
    panel['validation_affected_queries']=changed
    # This compact record makes the effective label sets hashable without
    # mutating or copying the large shared training cache.
    out=Path('data/processed_nc')/dataset;out.mkdir(parents=True,exist_ok=True)
    dest=out/'panel.json'
    content=json.dumps(panel)
    if not dest.exists() or dest.read_text(encoding='utf-8')!=content:dest.write_text(content,encoding='utf-8')
    return events,panel
