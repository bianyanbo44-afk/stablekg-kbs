"""Paired timestamp-block intervals and paper-ready summaries (no test tuning)."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path('results_nc/evaluation_renormalized')


def weighted_aurc(errors,order,weights,harmonic):
    e=errors[order];m=weights[order].astype(int);n=int(m.sum())
    if not n:return np.nan
    stop=np.cumsum(m);start=stop-m
    previous_error=np.cumsum(m*e)-m*e
    return float(np.sum(e*m+(previous_error-e*start)*(harmonic[stop]-harmonic[start]))/n)


def weighted_risk(errors,order,weights,coverage):
    m=weights[order];target=int(np.floor(weights.sum()*coverage))
    if target<=0 or m.sum()<target:return np.nan
    cumulative=np.cumsum(m);j=int(np.searchsorted(cumulative,target,side='left'))
    before=int(cumulative[j-1]) if j else 0
    return float((np.dot(errors[order[:j]],m[:j])+errors[order[j]]*(target-before))/target)


def weighted_auc(y,score,weights):
    order=np.argsort(score,kind='stable');s=np.asarray(score)[order];yy=np.asarray(y)[order];ww=np.asarray(weights)[order]
    group=np.cumsum(np.r_[True,s[1:]!=s[:-1]])-1
    pos=np.bincount(group,weights=ww*yy);neg=np.bincount(group,weights=ww*(1-yy))
    denom=pos.sum()*neg.sum()
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/denom) if denom else np.nan


def blocks(timestamps,count=20):
    unique=np.unique(timestamps);lookup={int(t):b for b,ts in enumerate(np.array_split(unique,min(count,len(unique)))) for t in ts}
    return np.array([lookup[int(t)] for t in timestamps])


def interval(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return (float(np.quantile(x,.025)),float(np.quantile(x,.975)),len(x)) if len(x) else (None,None,0)


def analyze_group(model,dataset,repetitions,output):
    runs=[];meta=[];metrics=[]
    for seed in range(3):
        path=ROOT/model/dataset/f'seed{seed}';df=pd.read_csv(path/'test_queries.csv')
        config=json.loads((path/'frozen_config.json').read_text());runs.append(df);meta.append(config)
        metric=pd.read_csv(path/'test_metrics.csv');metric['seed']=seed;metric['model']=model;metric['dataset']=dataset;metrics.append(metric)
    pd.concat(metrics).to_csv(output/f'{model}_{dataset}_all_metrics.csv',index=False)
    n=len(runs[0]);b=blocks(runs[0].timestamp.to_numpy());nb=b.max()+1
    harmonic=np.r_[0.,np.cumsum(1/np.arange(1,n*nb+1))]
    items=[]
    for df,cfg in zip(runs,meta):
        order=lambda name:np.argsort(-df[name].to_numpy(),kind='stable')
        belief=cfg['selectors']['belief_selector'];base=cfg['selectors']['best_baseline']
        gate=order(belief);gate=gate[df.robust1.to_numpy(bool)[gate]]
        items.append({'df':df,'error':1-df.correct.to_numpy(),'orders':{'baseline':order(base),
            'certificate':order('Certificate calibration'),'belief':order(belief),'gate':gate},
            'baseline':base,'belief':belief})
    def calculate(weights):
        vals={}
        for item in items:
            df=item['df'];e=item['error'];o=item['orders'];a=df.answer_count.to_numpy()
            def add(key,value):vals.setdefault(key,[]).append(value)
            add('mrr_gain',float(np.dot(weights,df.reciprocal_sum-df.neural_reciprocal_sum)/np.dot(weights,a)))
            add('accuracy_gain',float(np.average(df.correct-df.neural_correct,weights=weights)))
            add('aurc_certificate_minus_best',weighted_aurc(e,o['certificate'],weights,harmonic)-weighted_aurc(e,o['baseline'],weights,harmonic))
            add('certified_coverage',float(np.average(df.robust1,weights=weights)))
            for coverage in (.2,.4,.8):
                c=int(100*coverage)
                add(f'risk{c}_certificate_minus_best',weighted_risk(e,o['certificate'],weights,coverage)-weighted_risk(e,o['baseline'],weights,coverage))
                add(f'risk{c}_gate_minus_belief',weighted_risk(e,o['gate'],weights,coverage)-weighted_risk(e,o['belief'],weights,coverage))
                joint=np.maximum(e,df.worst_flip1.to_numpy())
                add(f'joint{c}_gate_minus_belief',weighted_risk(joint,o['gate'],weights,coverage)-weighted_risk(joint,o['belief'],weights,coverage))
        # A matched-coverage comparison is defined only if every seed reaches it.
        return {key:float(np.mean(v)) if np.isfinite(v).all() else np.nan for key,v in vals.items()}
    observed=calculate(np.ones(n,dtype=int));samples={key:[] for key in observed};rng=np.random.default_rng(713119)
    for _ in range(repetitions):
        counts=np.bincount(rng.integers(nb,size=nb),minlength=nb);values=calculate(counts[b])
        for key,value in values.items():samples[key].append(value)
    report=[]
    for key,value in observed.items():
        lo,hi,valid=interval(samples[key]);report.append({'model':model,'dataset':dataset,'comparison':key,
            'estimate':value,'ci_low':lo,'ci_high':hi,'valid_replicates':valid,'replicates':repetitions,'time_blocks':nb})
    pd.DataFrame(report).to_csv(output/f'{model}_{dataset}_paired_intervals.csv',index=False)
    # Intervention discrimination: pool the three fixed fits, resampling timestamp
    # blocks jointly across seeds. No independently fitted model is re-trained.
    pooled=pd.concat(runs,ignore_index=True);pb=np.tile(b,3);diag=[]
    for outcome in ('random_flip2','shifted_flip','wide_flip','event_flip'):
        y=(pooled[outcome].to_numpy()>0).astype(int)
        a=-pooled.hybrid_margin.to_numpy();c=-pooled.residual1.to_numpy()
        obs_a=weighted_auc(y,a,np.ones(len(y)));obs_c=weighted_auc(y,c,np.ones(len(y)));sample=[]
        for _ in range(repetitions):
            counts=np.bincount(rng.integers(nb,size=nb),minlength=nb);w=counts[pb]
            sample.append(weighted_auc(y,c,w)-weighted_auc(y,a,w))
        lo,hi,valid=interval(sample)
        diag.append({'model':model,'dataset':dataset,'outcome':outcome,'margin_auc':obs_a,'certificate_auc':obs_c,
                     'auc_gain':obs_c-obs_a,'ci_low':lo,'ci_high':hi,'valid_replicates':valid,'positive_cases':int(y.sum()),'queries_times_seeds':len(y)})
    pd.DataFrame(diag).to_csv(output/f'{model}_{dataset}_diagnostic_intervals.csv',index=False)
    (output/f'{model}_{dataset}_comparators.json').write_text(json.dumps([{'seed':i,'baseline':x['baseline'],'belief':x['belief']} for i,x in enumerate(items)],indent=2))
    print('Statistical summary:',model,dataset,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=2000);p.add_argument('--output',default='results_nc/statistics');a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    for model in ('TeRDy','TemporalComplEx'):
        for dataset in ('ICEWS14','ICEWS05-15'):analyze_group(model,dataset,a.replicates,output)
    all_metrics=pd.concat([pd.read_csv(p) for p in output.glob('*_all_metrics.csv')],ignore_index=True)
    all_metrics.to_csv(output/'all_metrics.csv',index=False)
    numeric=all_metrics.select_dtypes(include='number').columns.drop('seed')
    all_metrics.groupby(['model','dataset','selector'])[numeric].agg(['mean','std']).to_csv(output/'seed_summary.csv')
    pd.concat([pd.read_csv(p) for p in output.glob('*_paired_intervals.csv')]).to_csv(output/'paired_intervals.csv',index=False)
    pd.concat([pd.read_csv(p) for p in output.glob('*_diagnostic_intervals.csv')]).to_csv(output/'diagnostic_intervals.csv',index=False)
    (output/'statistical_protocol.json').write_text(json.dumps({'replicates':a.replicates,'blocks':20,
      'definition':'Contiguous, equal-count groups of observed query timestamps; block resampling is paired across all methods and all three fixed training seeds.',
      'uncertainty':'Intervals describe timestamp-block sampling variation conditional on the three fitted models. Training variability is separately summarized as sample SD over three seeds.',
      'multiplicity':'Predefined comparison family is reported completely; no significance-star or confirmatory familywise claim is made.',
      'missing_coverage':'NA when the certified pool cannot reach the requested coverage; no silent replacement by a lower coverage.'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
