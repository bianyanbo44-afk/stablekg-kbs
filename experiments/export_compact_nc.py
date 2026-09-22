"""Reuse the archived, fixed-budget temporal ComplEx checkpoints on new panels."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from neural_backbone import TemporalComplEx
from prepare_nc_cache import load


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--output',default='results_nc');p.add_argument('--test',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    for name in ('ICEWS14','ICEWS05-15'):
        _,panel=load(name)
        for seed in (0,1,2):
            checkpoint=Path(a.checkpoint_root)/f'{name}_seed{seed}.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=TemporalComplEx(state['n_entities'],state['n_relations'],state['n_times'],state['config']['dim'])
            model.load_state_dict(state['state_dict']);model.eval()
            out=Path(a.output)/'TemporalComplEx'/name/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
            meta={k:v for k,v in state.items() if k!='state_dict'}
            meta.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        checkpoint_origin=checkpoint.as_posix(),reuse='Supplied fixed 30-epoch checkpoint; no new test-based model selection.')
            (out/'run.json').write_text(json.dumps(meta,default=str,indent=2))
            for split in (['test'] if a.test else ['validation']):
                # Re-export from the supplied checkpoint: a previous score file
                # may belong to another fit with the same dataset/seed path.
                scores=[]
                with torch.inference_mode():
                    for start in range(0,len(panel[split]),128):
                        q=torch.tensor([x[:3] for x in panel[split][start:start+128]])
                        scores.append(model.score_all(q[:,0],q[:,1],q[:,2]).numpy())
                np.save(out/f'{split}_scores.npy',np.concatenate(scores))
                print(name,seed,split,'exported',flush=True)


if __name__=='__main__':main()
