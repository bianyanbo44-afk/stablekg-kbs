param(
    [int]$Cap = 5000,
    [int]$PublicSeeds = 5,
    [switch]$RunSynthetic,
    [switch]$RunNeural,
    [switch]$CompilePaper,
    [switch]$BuildOnly
)
$ErrorActionPreference = "Stop"
$python = (Get-Command python).Source
if ($Cap -ne 5000 -or $PublicSeeds -ne 5) {
    throw "This entry point reproduces the fixed 5000-query, five-seed manuscript. Use experiment CLIs and a separate output directory for exploratory runs."
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
Push-Location $PSScriptRoot
try {
    $compileOnly = $CompilePaper -and -not $BuildOnly -and -not $RunSynthetic -and -not $RunNeural
    if (-not $compileOnly) {
        if (-not $BuildOnly) {
            if ($RunSynthetic) {
                for ($batch = 0; $batch -lt 4; $batch++) {
                    Invoke-Checked $python @("experiments/synthetic_audit.py", "--seeds", "5", "--seed-start", (5*$batch), "--steps", "90", "--output-dir", "results_final/batch$batch")
                }
            }
            foreach ($spec in @(
                @{Data="data/raw/ICEWS14_all"; Name="ICEWS14"; Entities=7128; Batch=2048},
                @{Data="data/raw/ICEWS05_15_all"; Name="ICEWS05-15"; Entities=10488; Batch=4096},
                @{Data="data/raw/GDELT_all"; Name="GDELT"; Entities=500; Batch=2048}
            )) {
                $common = @("--data-root", $spec.Data, "--dataset", $spec.Name, "--cap", $Cap, "--entity-count", $spec.Entities)
                Invoke-Checked $python (@("experiments/stability_benchmark.py") + $common + @("--seeds", $PublicSeeds, "--output-dir", "results_final_v5"))
                Invoke-Checked $python (@("experiments/public_interventions_v2.py") + $common + @("--output-dir", "results_interventions_v5"))
                Invoke-Checked $python (@("experiments/matched_controls.py") + $common + @("--correct-prefix"))
                if ($spec.Name -ne "GDELT") {
                    Invoke-Checked $python (@("experiments/chronological_benchmark.py") + $common + @("--output-dir", "results_chronological_v5"))
                    if ($RunNeural) {
                        Invoke-Checked $python (@("experiments/neural_backbone.py") + $common + @("--seeds", "3", "--epochs", "30", "--dim", "96", "--negatives", "32", "--batch-size", $spec.Batch, "--output-dir", "results_neural_v5"))
                    }
                }
            }
        }
        Invoke-Checked $python @("experiments/analyze_results.py", "--public-dir", "results_final_v5", "--intervention-dir", "results_interventions_v5", "--batch-root", "results_final", "--neural-dir", "results_neural_v5", "--output-dir", "results_final_v5/analysis")
        Invoke-Checked $python @("experiments/diagnostic_controls.py")
        Invoke-Checked $python @("experiments/fixed_coverage.py")
        Invoke-Checked $python @("scripts/revise_manuscript.py")
        Invoke-Checked $python @("figures/make_figures.py")
    }
    if ($CompilePaper) {
        Push-Location manuscript
        try {
            Invoke-Checked "latexmk" @("-g", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex")
            Invoke-Checked "latexmk" @("-g", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex")
        } finally { Pop-Location }
    }
    Write-Host "StableKG requested stages completed."
} finally { Pop-Location }
