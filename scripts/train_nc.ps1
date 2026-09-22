$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$pythonPath = if ($env:STABLEKG_PYTHON) { $env:STABLEKG_PYTHON } else { 'python' }
foreach ($datasetName in @('ICEWS14','ICEWS05-15')) {
    foreach ($seedNumber in @(0,1,2)) {
        $modelDirectory = "results_nc/TeRDy/$datasetName/seed$seedNumber"
        if (Test-Path "$modelDirectory/validation_scores.npy") { continue }
        $rankValue = if ($datasetName -eq 'ICEWS14') { 6000 } else { 2000 }
        $epochCount = if ($datasetName -eq 'ICEWS14') { 36 } else { 24 }
        $evalInterval = if ($datasetName -eq 'ICEWS14') { 3 } else { 2 }
        $batchSize = if ($datasetName -eq 'ICEWS14') { 4096 } else { 6000 }
        & $pythonPath -u experiments/recent_backbones.py --model TeRDy --dataset $datasetName --seed $seedNumber --rank $rankValue --epochs $epochCount --eval-every $evalInterval --patience 4 --batch $batchSize --eval-batch 128 --device cuda --output results_nc --validation-only --scores-only
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $datasetName seed $seedNumber (exit $LASTEXITCODE)" }
    }
}
