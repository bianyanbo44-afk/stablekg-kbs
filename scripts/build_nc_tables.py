"""Generate manuscript tables and a numeric writing record from final results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/neurocomputing'
BASE=ROOT/'results_nc/evaluation_renormalized'
GROUPS=[('TeRDy','ICEWS14'),('TeRDy','ICEWS05-15'),('TemporalComplEx','ICEWS14'),('TemporalComplEx','ICEWS05-15')]
def pm(values,scale=100,digits=2):
    v=np.asarray(values,dtype=float)
    return '--' if not np.isfinite(v).all() else f'${np.mean(v)*scale:.{digits}f}\\pm{np.std(v,ddof=1)*scale:.{digits}f}$'
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%')

def calibration_metrics(frame,column):
    y=frame.correct.to_numpy();p=frame[column].to_numpy();bins=np.minimum((p*10).astype(int),9)
    ece=sum(float((bins==k).mean()*abs(y[bins==k].mean()-p[bins==k].mean())) for k in range(10) if np.any(bins==k))
    return {'brier':float(np.mean((p-y)**2)),'ece':ece}
def main():
    OUT.mkdir(parents=True,exist_ok=True);prediction=[];select=[];supp=[];facts=[];deployment=[];seed_diagnostics=[]
    pred_header=r'\begin{table*}[tbp]\centering\small\caption{Predictive performance on the fixed public test panels. Values are mean $\pm$ sample SD over three training seeds, in percent. MRR and Hits average over designated held-out answers; accuracy evaluates the selected object against the complete known-answer set. CTF denotes compact temporal factorization.}\label{tab:prediction}\begin{tabular}{lllrrrrr}\toprule Backbone & Dataset & Interface & MRR & Hits@1 & Hits@3 & Hits@10 & Accuracy\\\midrule'
    sel_header=r'\begin{table*}[tbp]\centering\small\caption{Calibrated belief and one-window certified selection on the same predictions. Error and edit-flip rates are percentages at exactly 20\% or 40\% query coverage. Values are three-seed mean $\pm$ SD. Certified capacity is the fraction of all queries passing the certificate. Unavailable coverage is denoted by a dash.}\label{tab:selection}\begin{tabular}{lllrrrr}\toprule Backbone & Dataset & Policy & Error at 20\% & Error at 40\% & Flips at 40\% & Capacity\\\midrule'
    for model,dataset in GROUPS:
        label='CTF' if model=='TemporalComplEx' else model;frames=[];configs=[];diags=[];metrics=[]
        for seed in range(3):
            path=BASE/model/dataset/f'seed{seed}'
            frames.append(pd.read_csv(path/'test_queries.csv'));configs.append(json.loads((path/'frozen_config.json').read_text()))
            diags.append(json.loads((path/'test_diagnostics.json').read_text()));metrics.append(pd.read_csv(path/'test_metrics.csv').set_index('selector'))
        for interface,prefix in (('Neural','neural_'),('Hybrid','')):
            values=[pm([m.iloc[0][prefix+k] for m in metrics]) for k in ('mrr','hits1','hits3','hits10')]
            acc=pm([d[prefix+'accuracy'] for d in diags]);prediction.append(f'{label} & {dataset} & {interface} & '+' & '.join(values+[acc])+r'\\')
        for policy in ('Belief','Certified'):
            rows=[m.loc[c['selectors']['belief_selector'] if policy=='Belief' else 'Certified 1-window'] for m,c in zip(metrics,configs)]
            vals=[pm([r.get(k,np.nan) for r in rows]) for k in ('risk20','risk40','worst_flip1_40')]
            vals.append('--' if policy=='Belief' else pm([d['robust1_fraction'] for d in diags]))
            select.append(f'{label} & {dataset} & {policy} & '+' & '.join(vals)+r'\\')
            for coverage in (20,40):
                values=[pm([r.get(k,np.nan) for r in rows]) for k in
                        (f'deployed_coverage{coverage}',f'deployed_risk{coverage}')]
                deployment.append(f'{label}/{dataset} & {policy} & {coverage} & '+' & '.join(values)+r'\\')
        for outcome in ('random_flip2','shifted_flip','wide_flip','event_flip'):
            values=[pm([d[f'{outcome}_auc_{feature}'] for d in diags],scale=1,digits=3)
                    for feature in ('hybrid_margin','residual1')]
            seed_diagnostics.append(f'{label}/{dataset} & '+tex(outcome)+' & '+' & '.join(values)+r'\\')
        # Every predefined selector and each gate is retained in the supplement.
        supp.append(r'\begin{table}[H]\centering\small\caption{'+label+' on '+dataset+r'. Three-seed means. AURC is on its natural scale; Brier and ECE assess binary answer correctness. Risk is a percentage.}\begin{tabular}{lrrrrr}\toprule Selector & AURC & Brier & ECE & Risk 20\% & Risk 40\%\\\midrule')
        for selector in metrics[0].index:
            vals=[]
            for k in ('aurc','brier','ece','risk20','risk40'):
                v=np.array([m.loc[selector].get(k,np.nan) for m in metrics]);vals.append(f'{np.mean(v)*(100 if k.startswith("risk") else 1):.3f}' if np.isfinite(v).all() else '--')
            supp.append(tex(selector)+' & '+' & '.join(vals)+r'\\')
        supp.append(r'\bottomrule\end{tabular}\end{table}')
        calibration={}
        for label_name in ('raw','belief'):
            values=[calibration_metrics(df,'hybrid_max' if label_name=='raw' else cfg['selectors']['belief_selector'])
                    for df,cfg in zip(frames,configs)]
            calibration.update({label_name+'_'+key:float(np.mean([v[key] for v in values])) for key in ('brier','ece')})
        selection40={}
        for policy in ('belief','gate'):
            selected=[m.loc[c['selectors']['belief_selector'] if policy=='belief' else 'Certified 1-window'] for m,c in zip(metrics,configs)]
            selection40.update({policy+'_'+key:float(np.mean([r[key] for r in selected])) for key in ('risk40','worst_flip1_40')})
        facts.append({'model':model,'dataset':dataset,'calibration':calibration,'selection40':selection40,
          'mean_diagnostics':{k:float(np.mean([d[k] for d in diags])) for k in diags[0] if isinstance(diags[0][k],(int,float))},
          'frozen_choices':[{'seed':i,'temperature':c['temperature'],'weight':c['weight'],'belief':c['selectors']['belief_selector'],'best_control':c['selectors']['best_baseline']} for i,c in enumerate(configs)],
          'mrr_neural':float(np.mean([m.iloc[0].neural_mrr for m in metrics])),
          'mrr_hybrid':float(np.mean([m.iloc[0].mrr for m in metrics]))})
    (OUT/'generated_prediction_table.tex').write_text(pred_header+'\n'+'\n'.join(prediction)+r'\bottomrule\end{tabular}\end{table*}',encoding='utf-8')
    (OUT/'generated_selection_table.tex').write_text(sel_header+'\n'+'\n'.join(select)+r'\bottomrule\end{tabular}\end{table*}',encoding='utf-8')
    supp.insert(0,r'\clearpage\section{Complete selector results}')
    supp.append(r'\section{Frozen choices}\begin{longtable}{llrrll}\caption{Validation-selected fusion and confidence models.}\\\toprule Backbone / data & Seed & $T$ & $\lambda$ & Belief & Best control\\\midrule\endhead')
    abbreviate={'Probability calibration':'Probability','Score calibration':'Score','Evidence calibration':'Evidence','Certificate calibration':'Certificate','Ensemble calibration':'Ensemble','candidate_probability':'Candidate prob.','hybrid_margin':'Hybrid margin','ensemble_probability':'Ensemble prob.','Temperature scaling':'Temperature'}
    for f in facts:
        for c in f['frozen_choices']:
            name=f['model'].replace('TemporalComplEx','CTF')+' / '+f['dataset']
            supp.append(f"{name} & {c['seed']} & {c['temperature']:g} & {c['weight']:g} & {tex(abbreviate.get(c['belief'],c['belief']))} & {tex(abbreviate.get(c['best_control'],c['best_control']))}"+r'\\')
    supp.append(r'\bottomrule\end{longtable}')
    paired=pd.read_csv(ROOT/'results_nc/statistics/paired_intervals.csv')
    supp.append(r'\section{Paired temporal-block intervals}\small\begin{longtable}{llrrrr}\caption{The complete prespecified paired comparison family.}\\\toprule Group & Comparison & Estimate & Lower & Upper & Valid\\\midrule\endhead')
    for _,r in paired.iterrows():
        vals=[f'{r[k]:.4f}' if pd.notna(r[k]) else '--' for k in ('estimate','ci_low','ci_high')]
        supp.append(r.model.replace('TemporalComplEx','CTF')+'/'+r.dataset+' & '+tex(r.comparison)+' & '+' & '.join(vals)+f' & {int(r.valid_replicates)}'+r'\\')
    supp.append(r'\bottomrule\end{longtable}\normalsize')
    supp.append(r'Valid is the number of usable bootstrap draws out of 2,000. Gated operating points require sufficient certified queries in every seed. Estimates and intervals use natural metric units, not percentage points.')
    supp.extend([r'\section{Validation-threshold deployment}',
      r'Thresholds are chosen on the operating-validation subset at nominal 20\% and 40\% coverage and then applied unchanged to the test panel. Certified selection adds the one-window gate to the same belief threshold. Values are percentages, mean $\pm$ sample SD over three fits.',
      r'\small\begin{longtable}{llrrr}\caption{Test performance at frozen operating-validation thresholds.}\\\toprule Group & Policy & Nominal & Test coverage & Test error\\\midrule\endhead',
      *deployment,r'\bottomrule\end{longtable}\normalsize',
      r'\section{Intervention discrimination by fitted seed}',
      r'AUROC is computed independently within each fitted seed, then summarized as mean $\pm$ sample SD. The paired intervals in the main figure instead pool query--seed predictions and resample timestamp blocks jointly. The two summaries describe different sources of variation.',
      r'\small\begin{longtable}{llrr}\caption{Within-seed intervention discrimination and between-seed variation.}\\\toprule Group & Intervention & Margin AUROC & Surplus AUROC\\\midrule\endhead',
      *seed_diagnostics,r'\bottomrule\end{longtable}\normalsize'])
    (OUT/'generated_supplementary_tables.tex').write_text('\n'.join(supp),encoding='utf-8')
    (ROOT/'results_nc/manuscript_facts.json').write_text(json.dumps(facts,indent=2))
    print(json.dumps(facts,indent=2))

if __name__=='__main__':main()
