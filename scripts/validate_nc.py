"""Finite CPU queue: configure each model using validation, while GPUs train."""
import argparse, subprocess, sys, time, os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='2'
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',required=True)
p.add_argument('--force',action='store_true',help='Refit validation even when released summaries are present.')
a=p.parse_args()
subprocess.run([sys.executable,'-u','experiments/export_compact_nc.py','--checkpoint-root',a.checkpoint_root],check=True)
pending=[(model,dataset,seed) for model in ('TemporalComplEx','TeRDy')
         for dataset in ('ICEWS14','ICEWS05-15') for seed in (0,1,2)]
while pending:
    progressed=False
    for model,dataset,seed in pending[:]:
        source=Path('results_nc')/model/dataset/f'seed{seed}'/'validation_scores.npy'
        dest=Path('results_nc/evaluation_renormalized')/model/dataset/f'seed{seed}'/'validation_metrics.csv'
        if not a.force and dest.exists() and (dest.parent/'validation_queries.csv').exists():
            pending.remove((model,dataset,seed));continue
        if not source.exists():continue
        # The training exporter closes the array file before finishing its process;
        # np.load catches an incomplete header/file rather than silently truncating.
        subprocess.run([sys.executable,'-u','experiments/evaluate_nc.py','--model',model,
                        '--dataset',dataset,'--seed',str(seed)],check=True)
        pending.remove((model,dataset,seed));progressed=True
        print('VALIDATION FROZEN:',model,dataset,seed,flush=True)
    if pending and not progressed:time.sleep(15)
print('All twelve validation configurations are frozen.',flush=True)
