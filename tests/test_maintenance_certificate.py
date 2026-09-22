import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from maintenance_certificate import certificate_state
from window_certificate import WindowEvidence,fused_score,certify

def test_vectorized_state_matches_independent_witness_path():
    rng=np.random.default_rng(7753)
    for trial in range(150):
        b=int(rng.integers(1,25));n=int(rng.integers(25,80));mass=rng.lognormal(0,4,(b,n-2))
        mass[rng.random(mass.shape)<.65]=0
        # Every retained row must have positive mass.
        mass=mass[mass.sum(axis=1)>0];b=len(mass)
        if trial%7==0:mass[0]*=1e-35
        ev=WindowEvidence(np.arange(n-2),np.arange(b),mass,float(mass.sum()),int((mass>0).sum()))
        p=rng.dirichlet(np.ones(n));weight=float(rng.choice([0,.1,.25,.5,.75]))
        g=fused_score(p,ev,weight,'renormalized');c=certify(g,ev,weight,normalization='renormalized',probability=p)
        expected=np.array([c['winner'],c['min_delete'],c['robust1'],c['robust2'],c['robust3'],c['residual1'],c['residual2'],c['residual3'],g[c['winner']]])
        actual=certificate_state(g,ev,p,weight)
        assert np.array_equal(actual[:5],expected[:5])
        assert np.allclose(actual[5:],expected[5:],atol=1e-14,rtol=1e-12)
