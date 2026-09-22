import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import prepare_nc_cache as cache

def test_validation_truth_never_uses_test_only_answer(monkeypatch,tmp_path):
    events=np.array([[0,0,1,4]])
    original={'validation':[[0,0,4,[2],[1,2,3]]],'test':[[0,0,4,[3],[1,2,3]]],
              'validation_event_pool':[[0,0,2,4]]}
    monkeypatch.setattr(cache,'load_public',lambda name:(events,original));monkeypatch.chdir(tmp_path)
    _,panel=cache.load('sample')
    assert panel['validation'][0][4]==[1,2]
    assert panel['test'][0][4]==[1,2,3]
    assert original['validation'][0][4]==[1,2,3]
    assert panel['validation_removed_test_only_answers']==1
