import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from incremental_nc import NeuralEvidenceCache
from window_certificate import make_index,evidence_at,fused_score,certify


def test_local_updates_match_independent_raw_rebuild():
    events=np.array([[0,0,1,1],[0,0,1,2],[0,0,2,8],[0,0,3,9],[1,0,1,3]])
    queries=[[0,0,5,[2],[2]],[0,0,10,[1],[1]],[1,0,5,[3],[3]]]
    logits=np.array([[.1,.5,.2,.1],[.2,.3,.8,.4],[.3,.4,.1,.7]])
    cache=NeuralEvidenceCache(events,queries,logits,1.,.6)
    before=cache.outputs.copy();result=cache.apply([(events[2],-1)],True)
    assert result['affected']==1 and result['refreshed']==1
    assert np.array_equal(cache.outputs[[0,2]],before[[0,2]])
    independent=NeuralEvidenceCache(np.delete(events,2,axis=0),queries,logits,1.,.6)
    assert np.allclose(cache.outputs,independent.outputs)
    cache.apply([(events[2],1)],True)
    assert np.allclose(cache.outputs,before)


def test_insert_new_window_object_and_skip_ineligible_answer():
    events=np.array([[0,0,1,4]])
    queries=[[0,0,5,[2],[2]],[0,0,4,[2],[1,2]]]
    logits=np.array([[0.,1.,2.],[0.,1.,2.]])
    cache=NeuralEvidenceCache(events,queries,logits,1.,.7)
    before=cache.outputs.copy()
    result=cache.apply([((0,0,1,4),1)],True)
    assert result['affected']==1  # query at time 4 excludes observed object 1
    assert np.array_equal(cache.outputs[1],before[1])
    result=cache.apply([((0,0,2,3),1)],True)
    assert result['affected']==2


def test_long_history_retains_tiny_old_mass_after_recent_window_deletion():
    from window_certificate import WindowEvidence,edited_score
    ev=WindowEvidence(np.array([0,1]),np.array([0,1]),np.array([[1e-35,0.],[0.,1.]]),1.,2)
    p=np.array([.6,.4]);score=fused_score(p,ev,.75,'renormalized')
    c=certify(score,ev,.75,normalization='renormalized',probability=p)
    assert c['min_delete']==1 and not c['robust1']
    assert np.argmax(edited_score(score,ev,.75,c['witness1'],'renormalized',p))==0
