"""Record the exact public cache and evaluation-panel provenance."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
from prepare_nc_cache import load
from window_certificate import make_index
from evaluate_nc import observed_answers

def main():
    output=Path('results_nc/provenance');output.mkdir(parents=True,exist_ok=True);records=[]
    for dataset in ('ICEWS14','ICEWS05-15','GDELT'):
        events,panel=load(dataset);index=make_index(events);folder=Path('data/processed_recent')/dataset
        row={'dataset':dataset,'events':len(events),'unique_events':len(np.unique(events,axis=0)),
             **{k:panel[k] for k in ('n_entities','n_relations','n_times')}}
        for split in ('validation','test'):
            queries=panel[split];row[split+'_queries']=len(queries)
            row[split+'_answers']=sum(len(q[3]) for q in queries)
            row[split+'_training_overlap']=sum(len(set(q[3])&set(observed_answers(index,q))) for q in queries)
            row[split+'_timestamps']=len({q[2] for q in queries})
        row['cache_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('train.npy','panel.json')}
        row['strict_panel_sha256']=hashlib.sha256((Path('data/processed_nc')/dataset/'panel.json').read_bytes()).hexdigest()
        row['truth_protocol']=panel['truth_protocol']
        row['validation_removed_test_only_answers']=panel['validation_removed_test_only_answers']
        records.append(row);print(dataset,row['events'],row['unique_events'],flush=True)
    (output/'data_record.json').write_text(json.dumps(records,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k!='cache_sha256'} for r in records]).to_csv(output/'data_record.csv',index=False)
    import torch,scipy,matplotlib,psutil
    cpu_name=platform.processor()
    if platform.system()=='Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu_name=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    (output/'environment.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),
       'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
       'matplotlib':matplotlib.__version__,'gpu':torch.cuda.get_device_name(0),
       'cpu':cpu_name,'physical_cores':psutil.cpu_count(logical=False),'logical_processors':psutil.cpu_count(),
       'ram_gb':psutil.virtual_memory().total/2**30},indent=2))

if __name__=='__main__':main()
