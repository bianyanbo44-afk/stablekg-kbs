"""Vectorized sparse certificate state for repeated cache maintenance.

Returns the same nine cached fields as the witness-producing implementation.
The query predictor and edit family are unchanged; competitor calculations are
batched in NumPy to avoid repeated Python dispatch on dense public histories.
"""
import numpy as np
from window_certificate import certify,relevant_competitors

def certificate_state(g,ev,p,weight):
    w=int(np.argmax(g));ids=relevant_competitors(g,ev,w)
    ids=ids[np.isfinite(g[ids])];b=len(ev.windows)
    if b==0 or len(ids)<16:
        c=certify(g,ev,weight,normalization='renormalized',probability=p)
        return np.array([w,c['min_delete'],c['robust1'],c['robust2'],c['robust3'],c['residual1'],c['residual2'],c['residual3'],g[w]])
    norm=(weight/ev.reference_mass)*ev.mass;pos={int(o):i for i,o in enumerate(ev.objects)}
    wm=norm[:,pos[w]] if w in pos else np.zeros(b)
    cm=np.zeros((b,len(ids)))
    for j,c in enumerate(ids):
        if int(c) in pos:cm[:,j]=norm[:,pos[int(c)]]
    fractions=ev.mass.sum(axis=1)/ev.reference_mass;anchor=(1-weight)*(p[w]-p[ids])
    delta=wm[:,None]-cm+fractions[:,None]*anchor[None,:]
    order=np.argsort(-delta,axis=0,kind='stable');columns=np.arange(len(ids))
    positive=np.minimum((delta>0).sum(axis=0),b-1)
    def suffix(values):return np.cumsum(values[::-1],axis=0)[::-1]
    sw=suffix(wm[order]);sc=suffix(np.take_along_axis(cm,order,axis=0));sm=suffix(fractions[order])
    ww=(1-weight)*p[w]+sw/sm;cc=(1-weight)*p[ids][None,:]+sc/sm
    flip=(cc>ww)|((cc==ww)&(ids[None,:]<w))
    prefixes=np.arange(b)[:,None];valid=prefixes<=positive[None,:]
    where=np.argwhere(flip&valid);minimum=int(where[:,0].min()) if len(where) else b+1
    fallback=p.copy();fallback[~np.isfinite(g)]=-np.inf;nw=int(np.argmax(fallback))
    if nw!=w:minimum=min(minimum,b)
    robust=[];residual=[]
    for k in (1,2,3):
        take=np.minimum(positive,k);bad=bool(flip[take,columns].any())
        surplus=float(np.min(anchor*sm[take,columns]+sw[take,columns]-sc[take,columns]))
        if k>=b and nw!=w:
            bad=True;surplus=min(surplus,float((1-weight)*(p[w]-p[nw])))
        robust.append(not bad);residual.append(surplus)
    return np.array([w,minimum,*robust,*residual,g[w]])
