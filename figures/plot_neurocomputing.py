"""Rebuild all Neurocomputing vector figures from archived experiment outputs."""
from pathlib import Path
import argparse, json, itertools, os, runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/neurocomputing'
DATA=OUT/'source_data'
QA=OUT/'qa'
TEAL='#167D8D'; INK='#37434E'; COPPER='#B66B42'; VIOLET='#8576A6'
GREY='#9BA5AC'; LIGHT='#E8ECEE'; PALE='#E5F2F2'
GROUPS=[('TeRDy','ICEWS14'),('TeRDy','ICEWS05-15'),('TemporalComplEx','ICEWS14'),('TemporalComplEx','ICEWS05-15')]
NAMES=['TeRDy / ICEWS14','TeRDy / ICEWS05-15','CTF / ICEWS14','CTF / ICEWS05-15']
# Optional author-side rendered QA; ordinary reproduction has no skill dependency.
require_matplotlib_panel_alignment = (runpy.run_path(os.environ['STABLEKG_ALIGNMENT_QA'])['require_matplotlib_panel_alignment']
                  if os.environ.get('STABLEKG_ALIGNMENT_QA') else None)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,
 'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
 'lines.linewidth':1.5,'lines.markersize':4,'legend.frameon':False,'savefig.facecolor':'white',
 'mathtext.fontset':'dejavusans'})

def letter(ax,label):
    width_pt=ax.get_position().width*ax.figure.get_figwidth()*72
    ax.text(-10/width_pt,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')

def clean(ax,ygrid=True):
    if ygrid:ax.grid(axis='y',color=LIGHT,lw=.55);ax.set_axisbelow(True)

def save(fig,name,axes):
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
    fig.canvas.draw();w,h=fig.get_size_inches()*72
    panels=[]
    for label,ax,row,col,span in axes:
        box=ax.get_position();panels.append({'id':label,'bbox_pt':[box.x0*w,box.y0*h,box.x1*w,box.y1*h],
            'grid_id':'main','row_start':row,'row_stop':row+1,'col_start':col,'col_stop':col+span})
    (QA/(name+'.layout.json')).write_text(json.dumps({'schema_version':1,'backend':'python-matplotlib',
       'figure':{'width_pt':w,'height_pt':h},'panels':panels,'exemptions':[]},indent=2))
    if require_matplotlib_panel_alignment:
        require_matplotlib_panel_alignment(fig, axes=[x[1] for x in axes], panel_ids=[x[0] for x in axes],
                       json_out=QA/(name+'.alignment.json'))
    for ext in ('.pdf','.svg','.png'):fig.savefig(OUT/(name+ext),dpi=600)
    plt.close(fig)

def concept():
    fig=plt.figure(figsize=(183/25.4,116/25.4))
    gs=fig.add_gridspec(2,2,left=.085,right=.97,bottom=.12,top=.94,hspace=.62,wspace=.48,height_ratios=[.85,1.])
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1])
    a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
    letter(a,'a')
    boxes=[(.015,.16,.25,.66,INK,'Neural anchor','Frozen temporal encoder\nCandidate distribution p'),
           (.365,.16,.26,.66,TEAL,'Editable memory','Shared temporal windows\nRenormalized evidence'),
           (.735,.16,.25,.66,INK,'Certified answer','Calibrated belief\nExact deletion guarantee')]
    for x,y,w,h,color,title,body in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.025',
                         linewidth=.8,edgecolor=color,facecolor='white'))
        a.text(x+w/2,y+h*.73,title,ha='center',va='center',fontweight='bold',color=color,fontsize=9)
        a.text(x+w/2,y+h*.39,body,ha='center',va='center',linespacing=1.8,fontsize=7.5)
    for x1,x2 in ((.277,.35),(.637,.72)):
        a.annotate('',xy=(x2,.5),xytext=(x1,.5),arrowprops={'arrowstyle':'->','lw':1.1,'color':INK})
    a.text(.5,-.06,'Event edit  →  affected queries only  →  refreshed answer and certificate',ha='center',color=INK,fontsize=8)
    letter(b,'b');b.set_title('Same scores, different histories',loc='left',pad=12)
    b.set_xlim(-.6,7.2);b.set_ylim(-.55,3.5);b.axis('off')
    histories={'Concentrated':np.array([[6,0],[0,2],[0,2.]]),'Distributed':np.tile([2,4/3],(3,1))}
    records=[]
    for j,(name,m) in enumerate(histories.items()):
        x0=j*4.2
        b.text(x0+.9,3.22,name,ha='center',color=COPPER if j==0 else TEAL,fontsize=8)
        for i in range(3):
            for o in range(2):
                value=m[i,o];alpha=.09+.8*value/6
                b.add_patch(Rectangle((x0+o*1.1,2.2-i*.73),.85,.63,facecolor=TEAL if o==0 else COPPER,alpha=alpha,lw=0))
                b.text(x0+o*1.1+.425,2.515-i*.73,f'{value:.3g}',ha='center',va='center',fontsize=7.5)
                records.append({'history':name,'window':i+1,'object':o,'mass':value,'kind':'conceptual'})
        b.text(x0+.425,-.03,'Winner',ha='center',fontsize=7)
        b.text(x0+1.525,-.03,'Rival',ha='center',fontsize=7)
    b.text(3.1,-.48,'Both predict the winner with score 0.54',ha='center',fontsize=7.5,color=INK)
    letter(c,'c');c.set_title('Worst answer margin after deletion',loc='left',pad=12)
    anchor=np.array([.45,.35,.2]);response=[]
    for j,(name,m) in enumerate(histories.items()):
        margins=[]
        for k in range(4):
            candidates=[]
            for n in range(k+1):
                for removed in itertools.combinations(range(3),n):
                    rem=m.copy();rem[list(removed)]=0
                    score=anchor.copy() if rem.sum()==0 else .4*anchor+.6*np.r_[rem.sum(axis=0)/rem.sum(),0.]
                    candidates.append(score[0]-max(score[1:]))
            margins.append(min(candidates));response.append({'history':name,'budget':k,'worst_margin':min(candidates),'kind':'conceptual'})
        c.plot(range(4),margins,'o-',color=COPPER if j==0 else TEAL,label=name)
    c.axhline(0,color=GREY,lw=.7,ls='--');c.set(xlabel='Deletion budget (windows)',ylabel='Winner minus best rival',xticks=range(4),ylim=(-.65,.30))
    c.legend(loc='center right');clean(c,False)
    save(fig,'fig01_interface',[('a',a,0,0,2),('b',b,1,0,1),('c',c,1,1,1)])
    pd.DataFrame(records).to_csv(DATA/'fig01_histories.csv',index=False)
    pd.DataFrame(response).to_csv(DATA/'fig01_response.csv',index=False)

def runs(model,dataset):
    base=ROOT/'results_nc/evaluation_renormalized'/model/dataset
    return [(pd.read_csv(base/f'seed{s}/test_queries.csv'),json.loads((base/f'seed{s}/frozen_config.json').read_text())) for s in range(3)]

def prediction():
    stats=pd.read_csv(ROOT/'results_nc/statistics/paired_intervals.csv')
    fig=plt.figure(figsize=(183/25.4,126/25.4))
    gs=fig.add_gridspec(2,4,left=.13,right=.97,bottom=.13,top=.92,wspace=.65,hspace=.92)
    axs=[fig.add_subplot(gs[0,:2]),fig.add_subplot(gs[0,2:])]
    calibration=[fig.add_subplot(gs[1,j]) for j in range(4)]
    rows=[];reliability=[]
    for j,(model,dataset) in enumerate(GROUPS):
        values=[]
        for seed,(df,cfg) in enumerate(runs(model,dataset)):
            neu=df.neural_reciprocal_sum.sum()/df.answer_count.sum();hyb=df.reciprocal_sum.sum()/df.answer_count.sum()
            values.append((neu,hyb));rows.append({'model':model,'dataset':dataset,'seed':seed,'neural_mrr':neu,'hybrid_mrr':hyb})
        v=np.array(values)*100
        for k,color in ((0,INK),(1,TEAL)):
            axs[0].errorbar(v[:,k].mean(),j+(k-.5)*.16,xerr=v[:,k].std(ddof=1),fmt='o',color=color,capsize=2,label=['Neural','Hybrid'][k] if j==0 else None)
        axs[0].plot(v.mean(axis=0),[j-.08,j+.08],color=GREY,lw=.8,zorder=0)
        row=stats[(stats.model==model)&(stats.dataset==dataset)&(stats.comparison=='mrr_gain')].iloc[0]
        x=row.estimate*100
        axs[1].errorbar(x,j,xerr=[[x-row.ci_low*100],[row.ci_high*100-x]],fmt='o',color=TEAL,capsize=2)
        values=[]
        for df,cfg in runs(model,dataset):
            values.append(pd.DataFrame({'correct':df.correct,'Uncalibrated':df.hybrid_max,'Calibrated':df[cfg['selectors']['belief_selector']]}))
        pooled=pd.concat(values,ignore_index=True);ax=calibration[j]
        ax.plot([0,1],[0,1],color=GREY,ls='--',lw=.7)
        for method,color in [('Uncalibrated',INK),('Calibrated',TEAL)]:
            binid=np.minimum((pooled[method]*10).astype(int),9);xs=[];ys=[]
            for k in range(10):
                part=pooled[binid==k]
                if not len(part):continue
                x=part[method].mean();y=part.correct.mean();xs.append(x);ys.append(y)
                reliability.append({'model':model,'dataset':dataset,'method':method,'bin':k,'n':len(part),'mean_probability':x,'accuracy':y})
            ax.plot(xs,ys,'s--' if method=='Uncalibrated' else 'o-',color=color,markersize=2.5,lw=1,label=method)
        letter(ax,chr(99+j));ax.set_title(NAMES[j].replace(' / ','\n'),pad=10)
        ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted probability')
        if j==0:ax.set_ylabel('Observed accuracy')
        else:ax.set_yticklabels([])
    for ax,label in zip(axs,'ab'):
        letter(ax,label);ax.set_yticks(range(4),[n.replace(' / ','\n') for n in NAMES] if ax==axs[0] else []);ax.invert_yaxis();ax.grid(axis='x',color=LIGHT,lw=.55);ax.set_axisbelow(True)
    axs[0].set(xlabel='Filtered MRR (%)',title='Temporal memory and neural prediction');axs[0].legend(loc='upper center',bbox_to_anchor=(.55,-.26),ncol=2)
    axs[1].axvline(0,color=GREY,ls='--',lw=.8);axs[1].set(xlabel='Paired MRR gain (percentage points)',title='Gain with temporal sampling uncertainty')
    handles,labels=calibration[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.57,-.008),ncol=2)
    save(fig,'fig02_prediction',[(l,a,0,i*2,2) for i,(l,a) in enumerate(zip('ab',axs))]+[(chr(99+i),a,1,i,1) for i,a in enumerate(calibration)])
    pd.DataFrame(rows).to_csv(DATA/'fig02_prediction.csv',index=False)
    pd.DataFrame(reliability).to_csv(DATA/'fig02_reliability.csv',index=False)
    stats[stats.comparison=='mrr_gain'].to_csv(DATA/'fig02_paired_gains.csv',index=False)

def selection():
    fig,axs=plt.subplots(2,4,figsize=(183/25.4,112/25.4),gridspec_kw={'left':.08,'right':.985,'bottom':.13,'top':.81,'wspace':.3,'hspace':.4})
    record=[];handles=[];coverage=np.linspace(.02,1,99)
    colors={'Best control':GREY,'Calibrated belief':INK,'Certified belief':TEAL}
    for col,(model,dataset) in enumerate(GROUPS):
        data=runs(model,dataset)
        for label in colors:
            values=[[],[]]
            for row,(df,cfg) in enumerate(data):
                base=cfg['selectors']['best_baseline'];belief=cfg['selectors']['belief_selector']
                order=np.argsort(-df[base if label=='Best control' else belief].to_numpy(),kind='stable')
                if label=='Certified belief':order=order[df.robust1.to_numpy(bool)[order]]
                for endpoint,target in enumerate((1-df.correct.to_numpy(),df.worst_flip1.to_numpy())):
                    curve=[float(np.mean(target[order[:int(c*len(df))]])) if int(c*len(df))<=len(order) else np.nan for c in coverage]
                    values[endpoint].append(curve)
                    record.extend({'model':model,'dataset':dataset,'seed':row,'selector':label,'endpoint':['error','worst_flip1'][endpoint],'coverage':c,'risk':y} for c,y in zip(coverage,curve))
            for endpoint in (0,1):
                v=np.array(values[endpoint]);valid=np.isfinite(v).all(axis=0);mu=v[:,valid].mean(axis=0)*100;sd=v[:,valid].std(axis=0,ddof=1)*100
                line,=axs[endpoint,col].plot(coverage[valid]*100,mu,color=colors[label],label=label,lw=1.5 if label=='Certified belief' else 1.2,ls='--' if label=='Best control' else '-')
                axs[endpoint,col].fill_between(coverage[valid]*100,np.maximum(0,mu-sd),mu+sd,color=colors[label],alpha=.10,lw=0)
                if col==0 and endpoint==0:handles.append(line)
        for row in (0,1):
            ax=axs[row,col];letter(ax,chr(97+row*4+col));clean(ax);ax.set(xlim=(0,100),ylim=(-2,100),xticks=[0,50,100])
            if col:ax.set_yticklabels([])
        axs[0,col].set_title(NAMES[col].replace(' / ','\n'),pad=12)
        axs[1,col].set_xlabel('Coverage (%)')
    axs[0,0].set_ylabel('Answer error (%)');axs[1,0].set_ylabel('Worst-window flips (%)')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.985),ncol=3)
    save(fig,'fig03_selection',[(chr(97+i*4+j),axs[i,j],i,j,1) for i in range(2) for j in range(4)])
    pd.DataFrame(record).to_csv(DATA/'fig03_curves.csv',index=False)

def diagnostics():
    df=pd.read_csv(ROOT/'results_nc/statistics/diagnostic_intervals.csv')
    left=min(-.03,float(np.floor(df.ci_low.min()*10)/10));right=max(.65,float(np.ceil(df.ci_high.max()*10)/10))
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,105/25.4),gridspec_kw={'left':.19,'right':.965,'bottom':.16,'top':.88,'hspace':.85,'wspace':.38})
    labels={'random_flip2':'Random windows','shifted_flip':'Shifted windows','wide_flip':'Wider windows','event_flip':'Single event'}
    for j,((model,dataset),ax) in enumerate(zip(GROUPS,axs.ravel())):
        d=df[(df.model==model)&(df.dataset==dataset)];letter(ax,chr(97+j));ax.set_title(NAMES[j],pad=8,loc='left')
        for i,key in enumerate(labels):
            r=d[d.outcome==key].iloc[0]
            ax.errorbar(r.auc_gain,i,xerr=[[r.auc_gain-r.ci_low],[r.ci_high-r.auc_gain]],fmt='o',color=TEAL,capsize=2)
        ax.set_yticks(range(4),list(labels.values()) if j%2==0 else []);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.8,ls='--');ax.grid(axis='x',color=LIGHT,lw=.55);ax.set_axisbelow(True)
        ax.set_xlim(left,right);ax.set_xticks(np.arange(0,right+.01,.2));ax.set_xlabel('AUROC gain over score margin')
    save(fig,'fig04_diagnostics',[(chr(97+i),a,i//2,i%2,1) for i,a in enumerate(axs.ravel())]);df.to_csv(DATA/'fig04_diagnostics.csv',index=False)

def controls():
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,116/25.4),gridspec_kw={'left':.115,'right':.96,'bottom':.15,'top':.90,'hspace':.72,'wspace':.36})
    record=[];sensitivity=[]
    for j,(model,dataset) in enumerate(GROUPS):
        stem=model+'_'+dataset;base=ROOT/'results_nc/extended';df=pd.read_csv(base/f'{stem}_controls.csv');s=pd.read_csv(base/f'{stem}_sensitivity.csv')
        # Evidence-supported queries isolate non-vacuous deletion certificates.
        supported=df[df.supported]
        exact=supported.robust1.mean()*100;bound=supported.winner_only1.mean()*100
        axs[0,0].plot([bound,exact],[j,j],color=GREY,lw=1);axs[0,0].scatter([bound,exact],[j,j],c=[GREY,TEAL],s=22,zorder=3)
        for k,n in enumerate((8,32,64)):
            safe=supported[f'probe{n}_safe1'].astype(bool);f=float((~supported.loc[safe,'robust1'].astype(bool)).mean()) if safe.any() else np.nan
            axs[0,1].scatter(f*100,j+(k-1)*.14,color=[COPPER,VIOLET,INK][k],marker=['o','s','^'][k],s=18,label=f'{n} probes' if j==0 else None)
            record.append({'model':model,'dataset':dataset,'supported_queries':len(supported),'exact_coverage':exact/100,'bound_coverage':bound/100,'probes':n,'false_safe_among_passed':f})
        v=[]
        for weight,part in s.groupby('weight'):
            v.append({'weight':weight,'mrr':part.reciprocal_sum.sum()/part.answer_count.sum(),'robust':part.robust1.mean()})
        v=pd.DataFrame(v);color=[TEAL,COPPER,INK,VIOLET][j]
        for ax,key in ((axs[1,0],'mrr'),(axs[1,1],'robust')):
            ax.plot(v.weight,v[key]*100,color=color,marker=['o','s','^','D'][j],
                    linestyle=['-','--',':','-.'][j],label=NAMES[j],markersize=3)
        v['model']=model;v['dataset']=dataset;sensitivity.extend(v.to_dict('records'))
    for i,ax in enumerate(axs.ravel()):letter(ax,chr(97+i));clean(ax)
    axs[0,0].set_yticks(range(4),[n.replace(' / ','\n') for n in NAMES]);axs[0,0].invert_yaxis();axs[0,0].set(xlabel='Certified supported queries (%)',title='Exactness recovers admissible answers',xlim=(0,100))
    axs[0,0].scatter([],[],c=GREY,label='Conservative',s=20);axs[0,0].scatter([],[],c=TEAL,label='Exact',s=20);axs[0,0].legend(loc='lower left',bbox_to_anchor=(-.05,-.50),ncol=2)
    axs[0,1].set_yticks(range(4),[]);axs[0,1].invert_yaxis();axs[0,1].set(xlabel='False-safe rate among passed probes (%)',title='Finite probes can miss a deletion');axs[0,1].legend(loc='lower left',bbox_to_anchor=(-.05,-.50),ncol=3,columnspacing=.6)
    axs[1,0].set(xlabel='Evidence mixture weight',ylabel='Filtered MRR (%)',title='Predictive response')
    axs[1,1].set(xlabel='Evidence mixture weight',ylabel='Certified queries (%)',title='Stability response')
    handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.56,-.006))
    save(fig,'fig05_controls',[(chr(97+i),a,i//2,i%2,1) for i,a in enumerate(axs.ravel())]);pd.DataFrame(record).to_csv(DATA/'fig05_controls.csv',index=False);pd.DataFrame(sensitivity).to_csv(DATA/'fig05_weights.csv',index=False)

def maintenance():
    fig,axs=plt.subplots(2,3,figsize=(183/25.4,111/25.4),gridspec_kw={'left':.1,'right':.98,'bottom':.16,'top':.86,'hspace':.60,'wspace':.40})
    records=[]
    for col,dataset in enumerate(('ICEWS14','ICEWS05-15','GDELT')):
        df=pd.read_csv(ROOT/f'results_nc/updates/{dataset}_trials.csv')
        assert (df[['batch','full_elapsed_ms','incremental_elapsed_ms']].to_numpy()>0).all(), 'Log axes require strictly positive observations.'
        for kind,color,label in (('full',INK,'Full refresh'),('incremental',TEAL,'Local refresh')):
            groups=df.groupby('batch')[kind+'_elapsed_ms'];means=groups.median();lo=groups.quantile(.25);hi=groups.quantile(.75)
            axs[0,col].plot(means.index,means.values,'s--' if kind=='full' else 'o-',color=color,label=label)
            axs[0,col].fill_between(means.index,lo,hi,color=color,alpha=.12,lw=0)
        g=df.groupby('batch');work=g.incremental_refreshed.median()/5000*100
        axs[1,col].plot(work.index,work.values,'o-',color=TEAL);axs[1,col].axhline(100,color=INK,lw=1,ls='--')
        for row in (0,1):
            ax=axs[row,col];letter(ax,chr(97+row*3+col));ax.set_xscale('log');ax.set_xticks([1,10,100,1000],['1','10','100','1,000']);clean(ax)
        axs[0,col].set_yscale('log');axs[0,col].tick_params(axis='y',labelsize=8)
        axs[0,col].set_title(dataset+('\nFrequency anchor' if dataset=='GDELT' else '\nNeural anchor'),pad=11)
        axs[1,col].set(xlabel='Edited events per transaction',ylim=(-2,105))
        records.extend(df.to_dict('records'))
    axs[0,0].set_ylabel('Update time (ms; log scale)');axs[1,0].set_ylabel('Registered queries refreshed (%)')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.995),ncol=2)
    save(fig,'fig06_maintenance',[(chr(97+i*3+j),axs[i,j],i,j,1) for i in range(2) for j in range(3)]);pd.DataFrame(records).to_csv(DATA/'fig06_updates.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--concept-only',action='store_true');args=parser.parse_args()
    concept()
    if not args.concept_only:
        prediction();selection();diagnostics();controls();maintenance()
