"""Author-code TKGC backbones under a shared Pe protocol.

TeRDy uses the unmodified author model and regularizers. LTGQ uses the author
forward path with constructor-only, unreferenced modules removed to avoid
allocating >1 billion unused parameters. The transformed source is exported.
All checkpoint/selector choices use disjoint validation roles; no test score
is generated until checkpoint selection has finished.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, math, os, random, sys, time
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import stability_benchmark as b
from neural_backbone import evidence_for_candidate, filtered_rank
from prepare_public_cache import load

ROOT=Path(__file__).resolve().parents[1]
BASE=['neural_margin','neural_max','neural_softmax']
STABLE=BASE+['log_support','recency','certificate','diversity','fragility_cost','counter_evidence_cost','evidence_alignment']

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def ltgq_source(output):
    path=ROOT/'external/LTGQ/model.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LTGQ')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name!='__init__']
    used={n.attr for m in methods for n in ast.walk(m) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'}
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    # Also preserve attributes read later in construction.
    used|={n.attr for n in ast.walk(init) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Load) and isinstance(n.value,ast.Name) and n.value.id=='self'}
    removed=[];body=[]
    for n in init.body:
        target=n.targets[0] if isinstance(n,ast.Assign) and len(n.targets)==1 else None
        if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self' and target.attr not in used:
            removed.append(target.attr)
        else:body.append(n)
    init.body=body;ast.fix_missing_locations(tree)
    dest=output/'ltgq_pruned_author_model.py';dest.write_text(ast.unparse(tree),encoding='utf-8')
    (output/'ltgq_adapter.json').write_text(json.dumps({'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'removed_unused_attributes':removed,'forward_modified':False},indent=2))
    return module(dest,'author_ltgq').LTGQ

def make_model(kind,panel,args,output):
    ne,nr,nt=[panel[k] for k in ('n_entities','n_relations','n_times')]
    if kind=='TeRDy':
        mod=module(ROOT/'external/TeRDy/models.py','author_terdy')
        rank=args.rank or (6000 if panel['dataset']=='ICEWS14' else 8000)
        model=mod.TeRDy((ne,2*nr,ne,nt),rank,alpha=10)
        return model.to(args.device),{'rank':rank,'alpha':10},None
    Model=ltgq_source(output)
    name=panel['dataset'];y0=2014 if name=='ICEWS14' else 2005
    # TFLEX lexicographically indexes all 365 / 4,017 dates; verify cardinality.
    expected=(date(2015,1,1)-date(2014,1,1)).days if y0==2014 else (date(2016,1,1)-date(2005,1,1)).days
    assert nt==expected
    dates=[date(y0,1,1)+timedelta(days=i) for i in range(nt)]
    mapping=torch.tensor([[d.year-y0,d.month-1,d.day-1] for d in dates],device=args.device)
    ent=np.random.permutation(400);rel=np.random.permutation(400)+400;perm=[]
    for i in range(20):
        for j in range(20):
            k=i*20+j;perm.extend([ent[k],rel[k]] if i%2==0 else [rel[k],ent[k]])
    params=SimpleNamespace(num_ent=ne,num_rel=nr,n_year=1 if y0==2014 else 11,n_month=12,n_day=31,
        embed_dim=400,chequer_perm=np.array([perm]),inp_drop=.1,feat_drop=.3,hid_drop=.4,
        num_filt=64,ker_sz=3,k_h=20,k_w=20,dataset='icews14' if y0==2014 else 'icews05-15')
    return Model(params).to(args.device),{k:v for k,v in vars(params).items() if k!='chequer_perm'},mapping

def score(model,kind,x,mapping):
    if kind=='TeRDy':return model.get_queries(x)@model.get_rhs(0,model.sizes[0])
    dates=mapping[x[:,3]]
    probs,_=model(x[:,0],x[:,1],dates[:,0],dates[:,1],dates[:,2],None,'one_to_n')
    return torch.logit(probs.clamp(1e-7,1-1e-7))

@torch.no_grad()
def inference(model,kind,queries,mapping,batch,device):
    model.eval();scores=[]
    for start in range(0,len(queries),batch):
        rows=queries[start:start+batch]
        x=torch.tensor([[q[0],q[1],q[3][0],q[2]] for q in rows],device=device)
        scores.append(score(model,kind,x,mapping).cpu().numpy())
    return np.concatenate(scores)

def ranks_metrics(matrix,queries):
    rr=[]
    for scores,q in zip(matrix,queries):rr.extend(filtered_rank(scores,o,set(q[4])) for o in q[3])
    rr=np.asarray(rr)
    return {'mrr':float(np.mean(1/rr)),**{f'hits{k}':float(np.mean(rr<=k)) for k in (1,3,10)}}

def train(model,kind,events,panel,mapping,args,out):
    nr=panel['n_relations'];inverse=events[:,[2,1,0,3]].copy();inverse[:,1]+=nr
    train=np.concatenate([events,inverse]);tensor=torch.from_numpy(train)
    name=panel['dataset'];is14=name=='ICEWS14'
    lr=args.learning_rate or (.02 if is14 else .008) if kind=='TeRDy' else (args.learning_rate or .001)
    opt=torch.optim.Adagrad(model.parameters(),lr=lr) if kind=='TeRDy' else torch.optim.Adam(model.parameters(),lr=lr)
    if kind=='TeRDy':
        reg=module(ROOT/'external/TeRDy/regularizers.py','terdy_reg')
        emb=reg.N3(.005 if is14 else .002);temporal=reg.Lambda3(.005 if is14 else .1)
    else:temporal=module(ROOT/'external/LTGQ/regularizers.py','ltgq_reg').TL(1e-5)
    selection=panel['validation'][:1000]
    best=-1.;history=[];stale=0;started=time.perf_counter()
    # Filtering of negative LTGQ samples uses training facts only.
    positives={}
    if kind=='LTGQ':
        for s,r,o,t in train:positives.setdefault((int(s),int(r),int(t)),set()).add(int(o))
    for epoch in range(1,args.epochs+1):
        ts=time.perf_counter();model.train();permutation=torch.randperm(len(tensor));total=0.;seen=0
        for start in range(0,len(tensor),args.batch):
            x=tensor[permutation[start:start+args.batch]].to(args.device)
            if len(x)==1:continue
            if kind=='TeRDy':
                predictions,factors,time_emb,freq=model(x)
                loss=F.cross_entropy(predictions,x[:,2])+emb(factors)+temporal(time_emb)+freq*.0005
            else:
                xx=x.cpu().numpy();negative=[]
                for s,r,o,t in xx:
                    excluded=positives[(int(s),int(r),int(t))]
                    candidates=np.setdiff1d(np.arange(panel['n_entities']),np.fromiter(excluded,dtype=np.int64),assume_unique=True)
                    negative.append(np.r_[o,np.random.choice(candidates,1000,replace=False)])
                candidates=torch.from_numpy(np.asarray(negative)).to(args.device)
                dates=mapping[x[:,3]]
                pred,times=model(x[:,0],x[:,1],dates[:,0],dates[:,1],dates[:,2],candidates,'one_to_x')
                label=torch.full_like(pred,1/panel['n_entities']);label[:,0]+=.9
                loss=F.binary_cross_entropy(pred,label)+temporal(times)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            if kind=='LTGQ':torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            nonfinite=0
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    # Matches the author's published optimizer safeguard;
                    # retain a count in the training log for auditability.
                    nonfinite += int((~torch.isfinite(p.grad)).sum().item())
                    p.grad=torch.nan_to_num(p.grad)
            opt.step();total+=loss.detach().item()*len(x);seen+=len(x)
        torch.cuda.synchronize() if args.device=='cuda' else None
        row={'epoch':epoch,'loss':total/seen,'seconds':time.perf_counter()-ts,'nonfinite_gradient_values':nonfinite}
        if epoch%args.eval_every==0 or epoch==args.epochs:
            matrix=inference(model,kind,selection,mapping,args.eval_batch,args.device)
            val=ranks_metrics(matrix,selection)['mrr'];row['validation_mrr']=val
            if val>best+1e-5:
                best=val;stale=0
                torch.save({'state_dict':model.state_dict(),'epoch':epoch,'validation_mrr':val},out/'best.pt')
            else:stale+=1
        history.append(row);pd.DataFrame(history).to_csv(out/'training.csv',index=False)
        print(f'{kind} {name} seed={args.seed} epoch={epoch} loss={row["loss"]:.5f} seconds={row["seconds"]:.1f} val={row.get("validation_mrr","-")}',flush=True)
        if stale>=args.patience:break
    return {'best_validation_mrr':best,'trained_epochs':epoch,'training_seconds':time.perf_counter()-started,'learning_rate':lr}

def features(matrix,queries,index):
    rows=[]
    for scores,q in zip(matrix,queries):
        order=np.argsort(-scores,kind='stable');win=int(order[0]);runner=int(order[1]);z=scores.astype(float)-scores.max()
        prob=np.exp(z-logsumexp(z));query=b.Query(q[0],q[1],q[2],tuple(q[3]))
        rows.append({'subject':q[0],'relation':q[1],'timestamp':q[2],'winner':win,'correct':float(win in q[3]),
            'neural_margin':float(scores[win]-scores[runner]),'neural_max':float(scores[win]),'neural_softmax':float(prob[win]),
            'negative_entropy':float(np.sum(prob*np.log(np.maximum(prob,1e-300)))),
            **evidence_for_candidate(query,index.get((q[0],q[1]),[]),win,35)})
    return rows

def selective_scores(vscore,tscore,panel,vr,tr):
    # 1000 checkpoint / 2000 fit / 1000 selector tune / 1000 threshold queries.
    fit=slice(1000,3000);tune=slice(3000,4000);oper=slice(4000,5000)
    vm=vscore[fit].astype(float);target=np.array([np.mean(row[q[3]]) for row,q in zip(vm,panel['validation'][fit])])
    def nll(logtemp):
        temp=np.exp(logtemp);return float(np.mean(logsumexp(vm/temp,axis=1)-target/temp))
    temperature=float(np.exp(minimize_scalar(nll,bounds=(-4,4),method='bounded').x))
    for matrix,rows in [(vscore,vr),(tscore,tr)]:
        scaled=matrix.astype(float)/temperature;lse=logsumexp(scaled,axis=1);maximum=np.max(scaled,axis=1)
        prob=np.exp(scaled-lse[:,None]);entropy=np.sum(prob*(scaled-lse[:,None]),axis=1)
        for i,row in enumerate(rows):row['temperature']=float(np.exp(maximum[i]-lse[i]));row['temperature_entropy']=float(entropy[i])
    choices={'Calibrated score':BASE,'Entropy':['temperature_entropy'],'StableKG':STABLE}
    settings={'temperature':temperature,'validation_roles':{'checkpoint':[0,1000],'fit':[1000,3000],'tune':[3000,4000],'operating':[4000,5000]}}
    for name,cols in choices.items():
        candidates=[]
        for penalty in (.0001,.001,.01,.1,1.):
            cal=b.RidgeLogistic(cols,l2=penalty).fit(vr[fit]);probs=cal.predict(vr[tune]);y=np.array([r['correct'] for r in vr[tune]])
            loss=-np.mean(y*np.log(np.maximum(probs,1e-12))+(1-y)*np.log(np.maximum(1-probs,1e-12)))
            candidates.append((loss,penalty,cal))
        _,penalty,cal=min(candidates,key=lambda x:x[0]);settings[name]={'l2':penalty,'features':cols}
        for rows in [vr,tr]:
            for row,value in zip(rows,cal.predict(rows)):row[name]=float(value)
    selectors=['temperature',*choices];report=[]
    for selector in selectors:
        row={'selector':selector,'aurc':b.aurc(tr,selector),'brier':b.brier(tr,selector),'ece':b.ece(tr,selector)}
        for coverage in (.2,.4):
            scores=np.array([r[selector] for r in tr]);correct=np.array([r['correct'] for r in tr]);n=int(len(tr)*coverage)
            row[f'exact_risk{int(coverage*100)}']=float(1-correct[np.argsort(-scores,kind='stable')[:n]].mean())
            th=b.choose_threshold(vr[oper],selector,coverage);metrics=b.selective(tr,selector,th)
            row[f'threshold{int(coverage*100)}']=th
            row[f'deployed_risk{int(coverage*100)}']=metrics['risk'];row[f'deployed_coverage{int(coverage*100)}']=metrics['coverage']
        report.append(row)
    settings['best_comparator_on_validation']=min(['temperature','Calibrated score','Entropy'],key=lambda k:b.aurc(vr[tune],k))
    return report,settings

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['TeRDy','LTGQ'],required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=101);p.add_argument('--patience',type=int,default=6)
    p.add_argument('--eval-every',type=int,default=5);p.add_argument('--batch',type=int,default=512);p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--learning-rate',type=float,default=0.)
    p.add_argument('--device',default='cuda');p.add_argument('--output',default='results_recent');p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    if a.device=='cuda':
        assert torch.cuda.is_available();torch.cuda.manual_seed_all(a.seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    out=Path(a.output)/a.model/a.dataset/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    events,panel=load(a.dataset);model,config,mapping=make_model(a.model,panel,a,out)
    runmeta={'args':vars(a),'model_config':config,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else 'cpu','data_source':'data/processed_recent/'+a.dataset,
        'full_candidate_vocabulary':panel['n_entities'],'cap':5000}
    (out/'run.json').write_text(json.dumps(runmeta,indent=2))
    if not a.evaluate_only:
        runmeta.update(train(model,a.model,events,panel,mapping,a,out));(out/'run.json').write_text(json.dumps(runmeta,indent=2))
    state=torch.load(out/'best.pt',map_location=a.device,weights_only=True);model.load_state_dict(state['state_dict'])
    matrices={split:inference(model,a.model,panel[split],mapping,a.eval_batch,a.device) for split in ['validation','test']}
    for split,scores in matrices.items():np.save(out/f'{split}_scores.npy',scores)
    index={}
    for s,r,o,t in events:index.setdefault((int(s),int(r)),[]).append((int(t),int(o)))
    for v in index.values():v.sort()
    vr=features(matrices['validation'],panel['validation'],index);tr=features(matrices['test'],panel['test'],index)
    report,settings=selective_scores(matrices['validation'],matrices['test'],panel,vr,tr)
    metrics=ranks_metrics(matrices['test'],panel['test'])
    for row in report:row.update({'dataset':a.dataset,'model':a.model,'seed':a.seed,'checkpoint_epoch':state['epoch'],**metrics})
    pd.DataFrame(report).to_csv(out/'metrics.csv',index=False);pd.DataFrame(vr).to_csv(out/'validation_predictions.csv',index=False);pd.DataFrame(tr).to_csv(out/'test_predictions.csv',index=False)
    (out/'selection.json').write_text(json.dumps(settings,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
