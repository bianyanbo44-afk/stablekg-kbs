import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from window_certificate import (WindowEvidence, certify, brute_force, edited_score,
                                evidence_at, make_index, fused_score)


def test_shared_deletion_removes_both_sides():
    ev = WindowEvidence(np.array([0, 1]), np.arange(2), np.array([[5., 4.], [1., 0.]]), 10., 10)
    score = np.array([.5625, .4375])
    result = certify(score, ev, 1., budgets=(1, 2))
    assert result['robust1']  # winner-only subtraction would falsely fail
    assert not result['robust2']  # removing both makes entity 1 strictly larger
    assert not brute_force(score, ev, 1., 2)


def test_sparse_and_full_equal_brute_force():
    rng = np.random.default_rng(1097)
    for _ in range(180):
        n, b = int(rng.integers(2, 12)), int(rng.integers(0, 7))
        objects = np.sort(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False))
        mass = rng.uniform(.1, 4, (b, len(objects)))
        mass[rng.random(mass.shape) < .4] = 0
        ev = WindowEvidence(objects, np.arange(b), mass, max(1., mass.sum()), 1)
        probability = rng.dirichlet(np.ones(n))
        weight = float(rng.uniform(.05, .95))
        score = fused_score(probability, ev, weight)
        sparse = certify(score, ev, weight)
        full = certify(score, ev, weight, sparse=False)
        assert sparse['min_delete'] == full['min_delete']
        for k in (1, 2, 3):
            assert sparse[f'robust{k}'] == full[f'robust{k}'] == brute_force(score, ev, weight, k)
            if not sparse[f'robust{k}']:
                assert np.argmax(edited_score(score, ev, weight, sparse[f'witness{k}'])) != sparse['winner']


def test_ties_obey_entity_id():
    ev = WindowEvidence(np.array([0, 1]), np.array([0]), np.array([[0., .5]]), 1., 1)
    assert not certify(np.array([.25, .75]), ev, 1.)['robust1']
    ev = WindowEvidence(np.array([0, 1]), np.array([0]), np.array([[.5, 0.]]), 1., 1)
    assert certify(np.array([.75, .25]), ev, 1.)['robust1']


def test_only_past_evidence_and_fixed_reference_denominator():
    events = np.array([[0, 0, 2, 1], [0, 0, 3, 4], [0, 0, 4, 10]])
    ev = evidence_at(make_index(events), 0, 0, 5, width=2)
    assert ev.event_count == 2
    assert 4 not in ev.objects
    score = fused_score(np.ones(5) / 5, ev, .5)
    removed = edited_score(score, ev, .5, [0, 1])
    assert np.allclose(removed, .1)
    assert ev.reference_mass > 0


def test_empty_history_is_neural_score():
    ev = evidence_at({}, 0, 0, 5)
    score = fused_score(np.array([.1, .9]), ev, .5)
    assert np.argmax(score) == 1
    result = certify(score, ev, .5)
    assert result['all_stable'] and result['robust3'] and result['windows'] == 0


def test_renormalized_certificate_matches_every_small_deletion_subset():
    rng=np.random.default_rng(71931)
    for _ in range(220):
        n,b=int(rng.integers(2,13)),int(rng.integers(0,7))
        objects=np.sort(rng.choice(n,size=int(rng.integers(1,n+1)),replace=False))
        mass=rng.uniform(.1,4,(b,len(objects)))
        ev=WindowEvidence(objects,np.arange(b),mass,float(mass.sum()),b)
        p=rng.dirichlet(np.ones(n));weight=float(rng.uniform(.05,.95))
        score=fused_score(p,ev,weight,'renormalized')
        cert=certify(score,ev,weight,budgets=tuple(range(7)),normalization='renormalized',probability=p)
        full=certify(score,ev,weight,normalization='renormalized',probability=p,sparse=False)
        assert cert['min_delete']==full['min_delete']
        for k in range(7):
            assert cert[f'robust{k}']==brute_force(score,ev,weight,k,'renormalized',p)
            if not cert[f'robust{k}']:
                changed=edited_score(score,ev,weight,cert[f'witness{k}'],'renormalized',p)
                assert np.argmax(changed)!=cert['winner']


def test_complete_deletion_falls_back_to_anchor():
    ev=WindowEvidence(np.array([0,1]),np.array([0]),np.array([[10.,0.]]),10.,1)
    p=np.array([.1,.9]);score=fused_score(p,ev,.75,'renormalized')
    cert=certify(score,ev,.75,normalization='renormalized',probability=p)
    assert cert['winner']==0 and cert['min_delete']==1 and not cert['robust1']
    assert np.argmax(edited_score(score,ev,.75,[0],'renormalized',p))==1


def test_observed_answers_stay_excluded_after_edit():
    ev=WindowEvidence(np.array([0,1,2]),np.array([0,1]),np.array([[2.,3.,1.],[1.,0.,2.]]),9.,6)
    p=np.array([0.,.4,.6]);score=fused_score(p,ev,.75,'renormalized');score[0]=-np.inf
    cert=certify(score,ev,.75,normalization='renormalized',probability=p)
    assert cert['robust1']==brute_force(score,ev,.75,1,'renormalized',p)
    assert edited_score(score,ev,.75,[0,1],'renormalized',p)[0]==-np.inf
