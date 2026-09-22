"""Run independent final controls after all final test evaluations complete."""
import subprocess,sys,time,os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='2'
from pathlib import Path

root=Path('results_nc/evaluation_renormalized')
required=[root/m/d/f'seed{s}'/'test_metrics.csv' for m in ('TeRDy','TemporalComplEx')
          for d in ('ICEWS14','ICEWS05-15') for s in range(3)]
deadline=time.monotonic()+4*60*60
while not all(path.exists() for path in required):
    if time.monotonic()>deadline:raise TimeoutError('Test evaluation did not finish; inspect stage logs.')
    time.sleep(15)
for script in ('experiments/statistics_nc.py','experiments/extended_nc.py','experiments/incremental_nc.py'):
    subprocess.run([sys.executable,'-u',script],check=True)
print('Final controls, maintenance measurements, and paired intervals are complete.',flush=True)
