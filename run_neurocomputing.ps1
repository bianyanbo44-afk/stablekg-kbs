param(
    [switch]$FullRun,
    [switch]$CompilePaper,
    [string]$CheckpointRoot = 'results_neural_v5'
)
$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$pythonPath = if ($env:STABLEKG_PYTHON) { $env:STABLEKG_PYTHON } else { 'python' }
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
Push-Location $PSScriptRoot
try {
    if ($FullRun) {
        Invoke-Checked $pythonPath @('scripts/download_data.py','--dataset','all')
        Invoke-Checked $pythonPath @('experiments/prepare_public_cache.py')
        if (-not (Test-Path 'external/TeRDy/.git')) {
            Invoke-Checked 'git' @('clone','https://github.com/Young0222/TeRDy.git','external/TeRDy')
            Invoke-Checked 'git' @('-C','external/TeRDy','checkout','f3f47986adb97eee26ac2e59811dc0d02df570f4')
        }
        $actualRevision = & git -C external/TeRDy rev-parse HEAD
        if ($actualRevision -ne 'f3f47986adb97eee26ac2e59811dc0d02df570f4') { throw 'TeRDy must use the recorded revision.' }
        Invoke-Checked $pythonPath @('experiments/train_compact_nc.py','--output',$CheckpointRoot)
        & ./scripts/train_nc.ps1
        if (-not $?) { throw 'TeRDy training failed.' }
        Invoke-Checked $pythonPath @('scripts/validate_nc.py','--checkpoint-root',$CheckpointRoot,'--force')
        Invoke-Checked $pythonPath @('scripts/finish_nc.py','--checkpoint-root',$CheckpointRoot)
        Invoke-Checked $pythonPath @('scripts/postprocess_nc.py')
        Invoke-Checked $pythonPath @('experiments/data_record_nc.py')
        Invoke-Checked $pythonPath @('scripts/archive_nc_source.py')
    } else {
        Invoke-Checked $pythonPath @('scripts/archive_nc_source.py','--restore')
        Invoke-Checked $pythonPath @('experiments/statistics_nc.py')
    }
    Invoke-Checked $pythonPath @('scripts/build_nc_tables.py')
    Invoke-Checked $pythonPath @('scripts/build_nc_controls_tables.py')
    Invoke-Checked $pythonPath @('figures/plot_neurocomputing.py')
    if ($CompilePaper) {
        Push-Location manuscript/neurocomputing
        try {
            Invoke-Checked 'latexmk' @('-pdf','-interaction=nonstopmode','-halt-on-error','main.tex')
            Invoke-Checked 'latexmk' @('-pdf','-interaction=nonstopmode','-halt-on-error','supplementary.tex')
        } finally { Pop-Location }
    }
} finally { Pop-Location }
