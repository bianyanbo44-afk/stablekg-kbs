"""Finite dependency queue: freeze controls, infer once, then evaluate all runs."""
import argparse,subprocess,sys,time,os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='2'
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--checkpoint-root',default='results_neural_v5');a=p.parse_args()
root=Path('results_nc/evaluation_renormalized')
required=[root/m/d/f'seed{s}'/'validation_metrics.csv' for m in ('TeRDy','TemporalComplEx')
          for d in ('ICEWS14','ICEWS05-15') for s in range(3)]
deadline=time.monotonic()+3*60*60
while not all(path.exists() for path in required):
    if time.monotonic()>deadline:raise TimeoutError('Validation queue did not complete in three hours; inspect training/validation logs.')
    time.sleep(15)
subprocess.run([sys.executable,'-u','experiments/finalize_nc_validation.py'],check=True)
subprocess.run([sys.executable,'-u','experiments/export_test_nc.py','--checkpoint-root',a.checkpoint_root],check=True)
for m in ('TeRDy','TemporalComplEx'):
    for d in ('ICEWS14','ICEWS05-15'):
        for s in range(3):
            subprocess.run([sys.executable,'-u','experiments/evaluate_nc.py','--model',m,'--dataset',d,
                            '--seed',str(s),'--reuse-validation','--test'],check=True)
print('All twelve frozen test evaluations completed.',flush=True)
