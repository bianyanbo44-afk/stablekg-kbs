import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import stability_benchmark as b
from real_incremental import FEATURES, IndexedEngine

def calibrator():
    rng=np.random.default_rng(21)
    rows=[dict(zip(FEATURES,rng.random(7)),correct=float(i%2)) for i in range(40)]
    return b.RidgeLogistic(FEATURES).fit(rows)

def test_local_update_candidate_replacement_and_clock():
    events=np.array([[0,0,1,7],[0,0,1,8],[0,0,2,10],[1,0,2,9]])
    keys=[(0,0),(1,0),(2,0)];cal=calibrator()
    inc=IndexedEngine(events,keys,10,cal,.5);full=IndexedEngine(events,keys,10,cal,.5)
    for edits,clock in [([((0,0,1,8),-1)],10),([((2,0,3,10),1)],10),
                        ([((0,0,1,7),-1)],10),([],30),([((0,0,1,30),1)],30)]:
        result=inc.apply(edits,clock,True);full.apply(edits,clock,False)
        np.testing.assert_allclose(inc.outputs,full.outputs,rtol=0,atol=1e-12)
        if not edits:assert result['refreshed']==len(keys)
    for i,(s,r) in enumerate(keys):
        raw=sorted((t,o) for o,c in inc.streams[(s,r)].items() for t,n in c.times.items() for _ in range(n))
        feat,_,_=b.winner_features(b.Query(s,r,inc.clock,(0,)),raw,5,'decay',35.)
        np.testing.assert_allclose(inc.outputs[i,2:9],[feat[k] for k in FEATURES],atol=1e-12)
        assert inc.outputs[i,0]==feat['winner']

def test_removing_last_timestamp_updates_recency_and_deletes_empty_bin():
    cal=calibrator();events=np.array([[0,0,1,1],[0,0,1,20]])
    engine=IndexedEngine(events,[(0,0)],20,cal,.5)
    engine.apply([((0,0,1,20),-1)],20,True)
    state=engine.streams[(0,0)][1]
    assert state.last==1 and set(state.bins)=={0}
    assert 0<engine.outputs[0,4]<1

def test_future_write_rejected():
    import pytest
    engine=IndexedEngine(np.array([[0,0,1,1]]),[(0,0)],1,calibrator(),.5)
    with pytest.raises(ValueError,match='Future event'):
        engine.apply([((0,0,1,2),1)],1,True)
