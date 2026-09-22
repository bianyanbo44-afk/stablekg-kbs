"""Test inference starts only after all model/selector configurations are frozen."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from prepare_nc_cache import load
from recent_backbones import make_model,inference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint-root',default='results_neural_v5');args_cli=parser.parse_args()
    manifest=Path('results_nc/evaluation_renormalized/frozen_manifest.json')
    frozen=json.loads(manifest.read_text())
    for record in frozen:
        assert hashlib.sha256(Path(record['configuration']).read_bytes()).hexdigest()==record['sha256']
    torch.set_num_threads(4)
    for dataset in ('ICEWS14','ICEWS05-15'):
        _,panel=load(dataset)
        for seed in range(3):
            source=Path('results_nc/TeRDy')/dataset/f'seed{seed}'
            # Export after this manifest is frozen, even if an older run left
            # test scores at the same path. The provenance stamp must match.
            metadata=json.loads((source/'run.json').read_text());args=SimpleNamespace(**metadata['args'])
            model,_,mapping=make_model('TeRDy',panel,args,source)
            checkpoint=torch.load(source/'best.pt',map_location='cuda',weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            scores=inference(model,'TeRDy',panel['test'],mapping,128,'cuda')
            np.save(source/'test_scores.npy',scores)
            (source/'test_export.json').write_text(json.dumps({'checkpoint_epoch':checkpoint['epoch'],
                'checkpoint_sha256':hashlib.sha256((source/'best.pt').read_bytes()).hexdigest(),
                'frozen_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},indent=2))
            del model;torch.cuda.empty_cache()
            print('TEST INFERENCE:',dataset,seed,flush=True)
    subprocess.run([sys.executable,'-u','experiments/export_compact_nc.py','--test','--checkpoint-root',
                    args_cli.checkpoint_root],check=True)


if __name__=='__main__':main()
