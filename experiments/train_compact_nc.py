"""Recreate the archived fixed-budget CTF fits without evaluating test labels."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from neural_backbone import TemporalComplEx,set_seed,train_model,flatten_training
from stability_benchmark import iter_pe
from prepare_nc_cache import load

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results_neural_v5');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    for dataset in ('ICEWS14','ICEWS05-15'):
        events,panel=load(dataset);nr=panel['n_relations'];batch=2048 if dataset=='ICEWS14' else 4096
        folder='ICEWS14_all' if dataset=='ICEWS14' else 'ICEWS05_15_all'
        examples=flatten_training(list(iter_pe(Path('data/raw')/folder/'train.jsonl')),nr)
        for seed in range(3):
            dest=out/f'{dataset}_seed{seed}.pt'
            if dest.exists():continue
            set_seed(seed);model=TemporalComplEx(panel['n_entities'],nr,panel['n_times'],96)
            history=train_model(model,np.asarray(examples,dtype=np.int64),seed,30,batch,32,.002,1e-6)
            torch.save({'state_dict':model.state_dict(),'n_entities':panel['n_entities'],'n_relations':nr,
               'n_times':panel['n_times'],'config':{'dim':96,'epochs':30,'negatives':32,'batch_size':batch,
               'learning_rate':.002,'weight_decay':1e-6,'seed':seed}},dest)
            (out/f'{dataset}_seed{seed}_training.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__':main()
