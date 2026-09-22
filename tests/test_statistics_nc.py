import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from statistics_nc import weighted_aurc,weighted_risk,weighted_auc
from evaluate_nc import aurc,roc_auc_score


def test_weighted_metrics_equal_explicit_bootstrap_expansion():
    rng=np.random.default_rng(9012)
    for _ in range(50):
        n=30;p=rng.random(n);y=rng.integers(2,size=n);weights=rng.integers(5,size=n)
        order=np.argsort(-p,kind='stable');idx=np.repeat(np.arange(n),weights)
        h=np.r_[0.,np.cumsum(1/np.arange(1,weights.sum()+1))]
        assert np.isclose(weighted_aurc(1-y,order,weights,h),aurc(y[idx],p[idx]))
        expanded=idx[np.argsort(-p[idx],kind='stable')]
        for coverage in (.2,.4,.8):
            m=int(len(expanded)*coverage)
            assert np.isclose(weighted_risk(1-y,order,weights,coverage),np.mean(1-y[expanded[:m]]))
        assert np.isclose(weighted_auc(y,p,weights),roc_auc_score(y[idx],p[idx]))
