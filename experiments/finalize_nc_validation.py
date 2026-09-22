"""Freeze strong ensemble/temperature controls before any revised test scoring."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from prepare_nc_cache import load
from window_certificate import make_index
from evaluate_nc import (SCORE,FIT,TUNE,OPER,FEATURES,observed_answers,admissible_probability,
                         aurc,summarize,add_ensemble_features,add_temperature_feature)
from stability_benchmark import RidgeLogistic

ROOT=Path('results_nc/evaluation_renormalized')


def main():
    frozen=[]
    for model in ('TemporalComplEx','TeRDy'):
        for dataset in ('ICEWS14','ICEWS05-15'):
            events,panel=load(dataset);index=make_index(events)
            sources=[Path('results_nc')/model/dataset/f'seed{s}' for s in range(3)]
            configs=[json.loads((ROOT/model/dataset/f'seed{s}'/'frozen_config.json').read_text()) for s in range(3)]
            for seed,config in enumerate(configs):
                out=ROOT/model/dataset/f'seed{seed}';rows=pd.read_csv(out/'validation_queries.csv').to_dict('records')
                matrix=np.load(sources[seed]/'validation_scores.npy',mmap_mode='r')
                x=np.array(matrix[FIT],dtype=np.float64)
                for logits,q in zip(x,panel['validation'][FIT]):logits[observed_answers(index,q)]=-np.inf
                target=np.array([logits[q[3]].mean() for logits,q in zip(x,panel['validation'][FIT])])
                def loss(log_temp):
                    temperature=np.exp(log_temp)
                    return float(np.mean(logsumexp(x/temperature,axis=1)-target/temperature))
                temp=float(np.exp(minimize_scalar(loss,bounds=(-3,4),method='bounded').x))
                config['temperature_scale']=temp
                add_temperature_feature(rows,panel['validation'],index,matrix,temp)
                config['ensemble']={'sources':list(map(str,sources)),
                    'temperatures':[c['temperature'] for c in configs],
                    'scope':'Three-seed uncertainty selector evaluated on the same fixed hybrid predictions.'}
                add_ensemble_features(rows,panel['validation'],index,config['ensemble'],'validation')
                columns=SCORE+['ensemble_probability','ensemble_variance','ensemble_entropy','ensemble_vote','ensemble_mi']
                candidates=[];y=np.array([r['correct'] for r in rows[TUNE]])
                for penalty in (.0001,.001,.01,.1,1.):
                    cal=RidgeLogistic(columns,l2=penalty).fit(rows[FIT]);p=cal.predict(rows[TUNE])
                    nll=float(-np.mean(y*np.log(np.maximum(p,1e-12))+(1-y)*np.log(np.maximum(1-p,1e-12))))
                    candidates.append((nll,cal))
                _,cal=min(candidates,key=lambda x:x[0]);label='Ensemble calibration'
                config['selectors']['calibrators'][label]={'features':columns,'l2':cal.l2,'mean':cal.mean.tolist(),
                                                         'scale':cal.scale.tolist(),'weight':cal.weight.tolist()}
                for row,p in zip(rows,cal.predict(rows)):row[label]=float(p)
                for name in (label,'ensemble_probability','Temperature scaling'):
                    scores=np.array([r[name] for r in rows[OPER]])
                    config['selectors']['thresholds'][name]={str(c):float(np.sort(scores)[-int(len(scores)*c)]) for c in (.2,.4)}
                candidates=[name for name in config['selectors']['thresholds'] if name!='Certificate calibration']
                config['selectors']['best_baseline']=min(candidates,key=lambda name:aurc(y,[r[name] for r in rows[TUNE]]))
                # The deployed method uses one encoder. Ensemble is a compute-rich comparator.
                config['selectors']['belief_selector']=min(FEATURES,key=lambda name:aurc(y,[r[name] for r in rows[TUNE]]))
                config['protocol_version']='renormalized-novel-fact-v2-strict-validation'
                config['truth_protocol']=panel['truth_protocol']
                config['validation_scores_sha256']=hashlib.sha256((sources[seed]/'validation_scores.npy').read_bytes()).hexdigest()
                dest=out/'frozen_config.json';dest.write_text(json.dumps(config,indent=2),encoding='utf-8')
                pd.DataFrame(rows).to_csv(out/'validation_queries.csv',index=False)
                report,diag=summarize(rows[TUNE],config['selectors'],config['validation_ranking'])
                pd.DataFrame(report).to_csv(out/'validation_metrics.csv',index=False)
                (out/'validation_diagnostics.json').write_text(json.dumps(diag,indent=2))
                frozen.append({'model':model,'dataset':dataset,'seed':seed,'configuration':str(dest),
                               'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
                print('Frozen final controls:',model,dataset,seed,flush=True)
    (ROOT/'frozen_manifest.json').write_text(json.dumps(frozen,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
