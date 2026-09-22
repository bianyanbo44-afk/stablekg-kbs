import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from window_certificate import WindowEvidence,certify,fused_score
from certificate_controls import winner_only_certificate,dense_budget_certificate


def test_dense_reference_and_conservative_bound():
    rng=np.random.default_rng(60091)
    for _ in range(180):
        n,b=int(rng.integers(2,17)),int(rng.integers(1,9))
        objects=np.sort(rng.choice(n,size=int(rng.integers(1,n+1)),replace=False))
        mass=rng.lognormal(0,4,(b,len(objects)))
        ev=WindowEvidence(objects,np.arange(b),mass,float(mass.sum()),b)
        p=rng.dirichlet(np.ones(n));weight=float(rng.uniform(.05,.95))
        score=fused_score(p,ev,weight,'renormalized')
        exact=certify(score,ev,weight,normalization='renormalized',probability=p)
        dense=dense_budget_certificate(score,ev,p,weight)
        for k in (1,2,3):
            assert exact[f'robust{k}']==dense[k]
            assert not winner_only_certificate(score,ev,p,weight,k) or exact[f'robust{k}']
