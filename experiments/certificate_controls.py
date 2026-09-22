"""Conservative, sampled and dense controls for shared-window certification."""
import numpy as np
from window_certificate import edited_score,relevant_competitors


def winner_only_certificate(score,ev,p,weight,k):
    w=int(np.argmax(score));b=len(ev.windows)
    if not b:return True
    if k>=b and np.argmax(p)!=w:return False
    pos={int(o):i for i,o in enumerate(ev.objects)}
    wm=weight*ev.mass[:,pos[w]]/ev.reference_mass if w in pos else np.zeros(b)
    fractions=ev.mass.sum(axis=1)/ev.reference_mass
    for c in relevant_competitors(score,ev,w):
        if not np.isfinite(score[c]):continue
        upper=wm+(1-weight)*(p[w]-p[c])*fractions
        top=np.sort(np.maximum(upper,0))[::-1][:min(k,b-1)]
        if score[w]-score[c]<=top.sum()+2e-12:return False
    return True


def random_probe_survival(score,ev,p,weight,k,rng,repeats=64):
    w=int(np.argmax(score));b=len(ev.windows);survival=[]
    for _ in range(repeats):
        size=int(rng.integers(1,min(k,b)+1)) if b else 0
        rows=rng.choice(b,size,replace=False)
        changed=edited_score(score,ev,weight,rows,'renormalized',p)
        survival.append(bool(np.argmax(changed)==w))
    return np.asarray(survival)


def dense_budget_certificate(score,ev,p,weight,budgets=(1,2,3)):
    """Vectorized full-vocabulary reference, using stable retained-mass sums.

    This is a stronger runtime control than a Python loop over all entities.
    It returns budget certificates, not the full radius computed by the sparse
    implementation. It intentionally does not use unsupported-class reduction.
    """
    w=int(np.argmax(score));b=len(ev.windows);n=len(score)
    if not b:return {k:True for k in budgets}
    mass=np.zeros((b,n));mass[:,ev.objects]=ev.mass/ev.reference_mass
    totals=mass.sum(axis=1)
    delta=weight*(mass[:,w,None]-mass)+(1-weight)*(p[w]-p)[None,:]*totals[:,None]
    order=np.argsort(-delta,axis=0,kind='stable')
    sorted_delta=np.take_along_axis(delta,order,axis=0)
    candidates=np.broadcast_to(np.arange(n),(b,n))
    sorted_mass=mass[order,candidates]
    sorted_w=mass[:,w][order]
    sorted_total=totals[order]
    def suffix(a):return np.cumsum(a[::-1],axis=0)[::-1]
    sw,sc,sm=suffix(sorted_w),suffix(sorted_mass),suffix(sorted_total)
    result={}
    for k in budgets:
        take=np.minimum((sorted_delta>0).sum(axis=0),min(k,b-1))
        ww=(1-weight)*p[w]+weight*sw[take,np.arange(n)]/sm[take,np.arange(n)]
        cc=(1-weight)*p+weight*sc[take,np.arange(n)]/sm[take,np.arange(n)]
        flips=(cc>ww)|((cc==ww)&(np.arange(n)<w))
        flips[w]=False;flips[~np.isfinite(score)]=False
        result[k]=bool(not np.any(flips) and (k<b or np.argmax(p)==w))
    return result
