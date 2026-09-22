"""Validation-only configuration and frozen test evaluation for the revision."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from prepare_nc_cache import load
from window_certificate import (make_index, evidence_at, neural_probability, fused_score,
                                certify, edited_score)
from stability_benchmark import RidgeLogistic

FIT=slice(1000,3000); TUNE=slice(3000,4000); OPER=slice(4000,5000)
SCORE=['max_probability','candidate_probability','neural_margin','negative_entropy','hybrid_max','hybrid_margin']
EVIDENCE=SCORE+['log_events','winner_evidence','diversity','alignment','history_fraction']
CERT=EVIDENCE+['residual1','residual2','radius_fraction','robust1','robust2']
FEATURES={'Probability calibration':['hybrid_max'],'Score calibration':SCORE,'Evidence calibration':EVIDENCE,'Certificate calibration':CERT}


def roc_auc_score(y,score):
    y=np.asarray(y,dtype=bool);n=int(y.sum());m=len(y)-n
    return float((rankdata(np.asarray(score))[y].sum()-n*(n+1)/2)/(n*m))


def rank_row(scores,q):
    mask=np.ones(len(scores),dtype=bool);mask[q[4]]=False
    background=np.sort(scores[mask]);values=scores[q[3]]
    left=np.searchsorted(background,values,side='left')
    right=np.searchsorted(background,values,side='right')
    return 1.+len(background)-right+.5*(right-left)


def aurc(correct, score):
    errors=1.-np.asarray(correct)[np.argsort(-np.asarray(score),kind='stable')]
    return float(np.mean(np.cumsum(errors)/np.arange(1,len(errors)+1)))


def probability_metrics(y,p):
    p=np.clip(p,0.,1.);y=np.asarray(y);ece=0.
    bins=np.minimum((p*10).astype(int),9)
    for b in range(10):
        m=bins==b
        if m.any():ece+=m.mean()*abs(p[m].mean()-y[m].mean())
    return {'brier':float(np.mean((p-y)**2)),'ece':float(ece)}


def observed_answers(index,q):
    events=index.get((q[0],q[1]),np.empty((0,2),dtype=np.int64))
    lo,hi=np.searchsorted(events[:,0],q[2],side='left'),np.searchsorted(events[:,0],q[2],side='right')
    return np.unique(events[lo:hi,1])


def admissible_probability(logits,temperature,observed):
    scores=np.asarray(logits,dtype=np.float64).copy();scores[observed]=-np.inf
    return neural_probability(scores,temperature)


def choose_fusion(matrix,queries,index):
    # This function is called only on the checkpoint/fusion validation role.
    observed=[observed_answers(index,q) for q in queries]
    ev=np.stack([evidence_at(index,q[0],q[1],q[2],excluded=obs).totals(matrix.shape[1]) for q,obs in zip(queries,observed)])
    records=[]
    for temperature in (.5,1.,2.,4.):
        probs=np.stack([admissible_probability(logits,temperature,obs) for logits,obs in zip(matrix,observed)])
        for weight in (0.,.1,.25,.5,.75):
            rr=[];correct=[]
            for scores,q,obs in zip((1-weight)*probs+weight*ev,queries,observed):
                scores[obs]=-np.inf
                rr.extend(rank_row(scores,q));correct.append(int(np.argmax(scores)) in q[4])
            records.append({'temperature':temperature,'weight':weight,'mrr':float(np.mean(1/np.array(rr))),
                            'accuracy':float(np.mean(correct))})
    # Positive evidence weights define the editable model. Neural-only is reported separately.
    best=max((r for r in records if r['weight']>0),key=lambda r:(r['mrr'],r['accuracy'],-r['weight']))
    return best,records


def describe(matrix,queries,index,config,split,stress=True):
    rows=[];ranking=[];neural_ranking=[];rng=np.random.default_rng(20260922)
    for i,(logits,q) in enumerate(zip(matrix,queries)):
        ts=time.perf_counter()
        obs=observed_answers(index,q)
        assert not (set(q[3])&set(obs)), 'Evaluation targets must be unobserved training facts.'
        normalization=config.get('normalization','renormalized')
        ev=evidence_at(index,q[0],q[1],q[2],excluded=obs);p=admissible_probability(logits,config['temperature'],obs)
        g=fused_score(p,ev,config['weight'],normalization);g[obs]=-np.inf;w=int(np.argmax(g));nw=int(np.argmax(p))
        cstart=time.perf_counter();certificate=certify(g,ev,config['weight'],normalization=normalization,probability=p);certificate_us=(time.perf_counter()-cstart)*1e6
        top=np.partition(g,-2)[-2:];p_top=np.partition(p,-2)[-2:]
        mass=ev.mass.sum(axis=0);total=ev.reference_mass
        emass=float(mass[np.searchsorted(ev.objects,w)]) if w in ev.objects else 0.
        windowmass=ev.mass.sum(axis=1)/(total or 1.)
        diversity=float(1.-np.sum(windowmass**2))
        current=ev.windows[-1] if len(ev.windows) else 0
        history_fraction=float(np.sum(ev.mass[ev.windows>=current-3])/total) if total else 0.
        row={'query_id':i,'subject':q[0],'relation':q[1],'timestamp':q[2],
             'winner':w,'neural_winner':nw,'correct':float(w in q[4]),'neural_correct':float(nw in q[4]),
             'max_probability':float(p.max()),'candidate_probability':float(p[w]),'neural_margin':float(p_top[1]-p_top[0]),
             'negative_entropy':float(np.sum(p*np.log(np.maximum(p,1e-300)))),
             'hybrid_max':float(g[w]),'hybrid_margin':float(top[1]-top[0]),
             'log_events':float(np.log1p(ev.event_count)),'winner_evidence':emass/(total or 1.),
             'diversity':diversity,'alignment':float(w==nw),'history_fraction':history_fraction,
             'radius_fraction':certificate['radius']/max(1,len(ev.windows)),
             'certificate_us':certificate_us,'evidence_events':ev.event_count}
        row.update({k:v for k,v in certificate.items() if not k.startswith(('witness','adversary'))})
        rr=rank_row(g,q);nrr=rank_row(p,q)
        ranking.extend(rr);neural_ranking.extend(nrr)
        row['query_mrr']=float(np.mean(1/rr));row['neural_query_mrr']=float(np.mean(1/nrr))
        row['answer_count']=len(q[3]);row['reciprocal_sum']=float(np.sum(1/rr));row['neural_reciprocal_sum']=float(np.sum(1/nrr))
        if stress:
            for k in (1,2,3):
                adv=edited_score(g,ev,config['weight'],certificate[f'witness{k}'],normalization,p)
                row[f'worst_flip{k}']=float(np.argmax(adv)!=w)
            # Out-of-design checks use independently sampled windows and partitions.
            random_flips=[]
            for _ in range(8):
                selected=rng.choice(len(ev.windows),min(2,len(ev.windows)),replace=False)
                random_flips.append(np.argmax(edited_score(g,ev,config['weight'],selected,normalization,p))!=w)
            row['random_flip2']=float(np.mean(random_flips))
            for label,width,offset in [('shifted',7,3),('wide',14,0),('event_day',1,0)]:
                alt=evidence_at(index,q[0],q[1],q[2],width=width,offset=offset,excluded=obs)
                ca=certify(g,alt,config['weight'],budgets=(1,),normalization=normalization,probability=p)
                altered=edited_score(g,alt,config['weight'],ca['witness1'],normalization,p)
                row[f'{label}_flip']=float(np.argmax(altered)!=w)
            # Strongest single event removed, with all other events retained.
            source=index.get((q[0],q[1]),np.empty((0,2),dtype=np.int64))
            source=source[:np.searchsorted(source[:,0],q[2],side='right')]
            support=source[source[:,1]==w]
            single=g.copy()
            if len(support):
                removed=np.exp2(-(q[2]-support[-1,0])/35.)
                if normalization=='fixed':single[w]-=config['weight']*removed/(total or 1.)
                else:
                    remaining=ev.totals(len(p))*total;remaining[w]=max(0.,remaining[w]-removed)
                    if remaining.sum()>0:single=(1-config['weight'])*p+config['weight']*remaining/remaining.sum()
                    else:single=p.copy()
                    single[obs]=-np.inf
            row['event_flip']=float(np.argmax(single)!=w)
        row['total_us']=(time.perf_counter()-ts)*1e6
        rows.append(row)
        if (i+1)%1000==0:print(f'{split}: {i+1}/{len(queries)}',flush=True)
    metrics={}
    for prefix,rr in [('',ranking),('neural_',neural_ranking)]:
        r=np.array(rr);metrics[prefix+'mrr']=float(np.mean(1/r))
        for k in (1,3,10):metrics[prefix+f'hits{k}']=float(np.mean(r<=k))
    return rows,metrics


def fit_selectors(rows):
    settings={}
    for label,columns in FEATURES.items():
        candidates=[];y=np.array([r['correct'] for r in rows[TUNE]])
        for penalty in (.0001,.001,.01,.1,1.):
            model=RidgeLogistic(columns,l2=penalty).fit(rows[FIT]);p=model.predict(rows[TUNE])
            nll=float(-np.mean(y*np.log(np.maximum(p,1e-12))+(1-y)*np.log(np.maximum(1-p,1e-12))))
            candidates.append((nll,model))
        _,model=min(candidates,key=lambda x:x[0])
        settings[label]={'features':columns,'l2':model.l2,'mean':model.mean.tolist(),
                         'scale':model.scale.tolist(),'weight':model.weight.tolist()}
    apply_selectors(rows,settings)
    baseline=['max_probability','candidate_probability','neural_margin','negative_entropy','hybrid_max','hybrid_margin','Probability calibration','Score calibration','Evidence calibration']
    winner=min(baseline,key=lambda label:aurc([r['correct'] for r in rows[TUNE]],[r[label] for r in rows[TUNE]]))
    thresholds={}
    for label in baseline+['Certificate calibration']:
        scores=np.array([r[label] for r in rows[OPER]])
        thresholds[label]={str(c):float(np.sort(scores)[-int(len(scores)*c)]) for c in (.2,.4)}
    belief=min(FEATURES,key=lambda label:aurc([r['correct'] for r in rows[TUNE]],[r[label] for r in rows[TUNE]]))
    return {'calibrators':settings,'best_baseline':winner,'belief_selector':belief,'thresholds':thresholds}


def apply_selectors(rows,settings):
    for label,setting in settings.items():
        model=RidgeLogistic(setting['features'],l2=setting['l2'])
        model.mean=np.array(setting['mean']);model.scale=np.array(setting['scale']);model.weight=np.array(setting['weight'])
        for row,p in zip(rows,model.predict(rows)):row[label]=float(p)


def add_temperature_feature(rows,queries,index,matrix,temperature):
    for row,q,logits in zip(rows,queries,matrix):
        p=admissible_probability(logits,temperature,observed_answers(index,q))
        row['Temperature scaling']=float(p[row['winner']])


def add_ensemble_features(rows,queries,index,config,split):
    matrices=[np.load(Path(source)/f'{split}_scores.npy',mmap_mode='r') for source in config['sources']]
    for i,(row,q) in enumerate(zip(rows,queries)):
        obs=observed_answers(index,q)
        probs=np.stack([admissible_probability(m[i],t,obs) for m,t in zip(matrices,config['temperatures'])])
        mean=probs.mean(axis=0);w=row['winner'];entropy=-np.sum(mean*np.log(np.maximum(mean,1e-300)))
        member_entropy=-np.sum(probs*np.log(np.maximum(probs,1e-300)),axis=1)
        row.update(ensemble_probability=float(mean[w]),ensemble_variance=float(probs[:,w].var()),
                   ensemble_entropy=float(-entropy),ensemble_vote=float(np.mean(np.argmax(probs,axis=1)==w)),
                   ensemble_mi=float(entropy-member_entropy.mean()))


def summarize(rows,settings,ranking):
    df=pd.DataFrame(rows);y=df.correct.to_numpy();report=[]
    for label,thresholds in settings['thresholds'].items():
        score=df[label].to_numpy();order=np.argsort(-score,kind='stable')
        row={'selector':label,'aurc':aurc(y,score),**ranking}
        if label in ('max_probability','candidate_probability','hybrid_max','ensemble_probability','Temperature scaling') or label.endswith('calibration'):
            row.update(probability_metrics(y,score))
        for coverage in (.2,.4):
            key=str(coverage);n=int(len(df)*coverage);accepted=order[:n]
            row[f'risk{int(coverage*100)}']=float(1-y[accepted].mean())
            for target in ('worst_flip1','random_flip2','shifted_flip','wide_flip','event_flip'):
                if target in df:row[f'{target}_{int(coverage*100)}']=float(df[target].iloc[accepted].mean())
            deployed=score>=thresholds[key]
            row[f'deployed_coverage{int(coverage*100)}']=float(deployed.mean())
            row[f'deployed_risk{int(coverage*100)}']=float(1-y[deployed].mean()) if deployed.any() else None
        report.append(row)
    for k in (1,2):
        label=f'Certified {k}-window';belief_label=settings.get('belief_selector','Certificate calibration');score=df[belief_label].to_numpy()
        eligible=df[f'robust{k}'].to_numpy(bool);order=np.argsort(-score,kind='stable');order=order[eligible[order]]
        row={'selector':label,'certified_coverage':float(eligible.mean()),**ranking}
        for coverage in (.2,.4):
            n=int(len(df)*coverage)
            if len(order)<n:continue
            accepted=order[:n];row[f'risk{int(coverage*100)}']=float(1-y[accepted].mean())
            for target in ('worst_flip1','random_flip2','shifted_flip','wide_flip','event_flip'):
                if target in df:row[f'{target}_{int(coverage*100)}']=float(df[target].iloc[accepted].mean())
            deployed=eligible&(score>=settings['thresholds'][belief_label][str(coverage)])
            row[f'deployed_coverage{int(coverage*100)}']=float(deployed.mean())
            row[f'deployed_risk{int(coverage*100)}']=float(1-y[deployed].mean()) if deployed.any() else None
        report.append(row)
    diagnostics={'queries':len(df),'accuracy':float(y.mean()),'neural_accuracy':float(df.neural_correct.mean()),
                 'empty_history_fraction':float((df.evidence_events==0).mean()),
                 'robust1_fraction':float(df.robust1.mean()),'robust2_fraction':float(df.robust2.mean()),
                 'certificate_us_median':float(df.certificate_us.median()),'certificate_us_p95':float(df.certificate_us.quantile(.95))}
    supported=df.evidence_events>0
    diagnostics['supported_robust1_fraction']=float(df.loc[supported,'robust1'].mean()) if supported.any() else None
    if 'worst_flip1' in df:
        diagnostics['certified_violations']=int(((df.robust1==True)&(df.worst_flip1>0)).sum())
        diagnostics['witness_mismatches']=int((df.robust1.astype(int)+df.worst_flip1.astype(int)!=1).sum())
        for outcome in ('random_flip2','shifted_flip','wide_flip','event_flip'):
            target=(df[outcome].to_numpy()>0).astype(int)
            for feature in ('hybrid_margin','residual1','residual2'):
                diagnostics[f'{outcome}_auc_{feature}']=float(roc_auc_score(target,-df[feature])) if len(np.unique(target))==2 else None
    return report,diagnostics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--model',default='TeRDy')
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--score-root',default='results_nc')
    parser.add_argument('--output',default='results_nc/evaluation_renormalized');parser.add_argument('--test',action='store_true')
    parser.add_argument('--normalization',choices=['fixed','renormalized'],default='renormalized')
    parser.add_argument('--reuse-validation',action='store_true');a=parser.parse_args()
    source=Path(a.score_root)/a.model/a.dataset/f'seed{a.seed}';out=Path(a.output)/a.model/a.dataset/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True);events,panel=load(a.dataset);index=make_index(events)
    config_path=out/'frozen_config.json'
    if not a.reuse_validation:
        matrix=np.load(source/'validation_scores.npy',mmap_mode='r')
        config,grid=choose_fusion(matrix[:1000],panel['validation'][:1000],index)
        config['normalization']=a.normalization
        pd.DataFrame(grid).to_csv(out/'validation_fusion_grid.csv',index=False)
        print('Fusion:',config,flush=True)
        rows,ranking=describe(matrix,panel['validation'],index,config,'validation')
        selectors=fit_selectors(rows);config.update(selectors=selectors,validation_ranking=ranking,
          dataset=a.dataset,model=a.model,seed=a.seed,source=str(source),width=7,half_life=35,
          roles={'checkpoint_fusion':[0,1000],'fit':[1000,3000],'tune':[3000,4000],'operating':[4000,5000]})
        # Configuration is frozen and serialized before any test scores are opened.
        config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        pd.DataFrame(rows).to_csv(out/'validation_queries.csv',index=False)
        report,diagnostics=summarize(rows[3000:4000],selectors,ranking)
        pd.DataFrame(report).to_csv(out/'validation_metrics.csv',index=False)
        (out/'validation_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    if a.test:
        config=json.loads(config_path.read_text(encoding='utf-8'))
        matrix=np.load(source/'test_scores.npy',mmap_mode='r')
        rows,ranking=describe(matrix,panel['test'],index,config,'test')
        if 'temperature_scale' in config:add_temperature_feature(rows,panel['test'],index,matrix,config['temperature_scale'])
        if 'ensemble' in config:add_ensemble_features(rows,panel['test'],index,config['ensemble'],'test')
        apply_selectors(rows,config['selectors']['calibrators'])
        report,diagnostics=summarize(rows,config['selectors'],ranking)
        pd.DataFrame(rows).to_csv(out/'test_queries.csv',index=False)
        pd.DataFrame(report).to_csv(out/'test_metrics.csv',index=False)
        (out/'test_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        print(json.dumps(diagnostics),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
