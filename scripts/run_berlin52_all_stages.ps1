param(
    [string]$PythonExe = "python",
    [string]$RepoRoot = ".",
    [string]$CppExe = "svrap.exe",
    [int]$PretrainEpochs = 200,
    [int]$FinetuneEpochs = 100,
    [int]$LabelRuns = 10,
    [int]$CppTimeout = 1800,
    [int]$Seed = 42,
    [double]$Alpha = 7.0,
    [double]$SupWeight = 1.0,
    [double]$CfWeight = 1.0,
    [double]$CfTemp = 500.0,
    [double]$EntropyWeight = 0.01,
    [switch]$SkipLabelGeneration
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path $RepoRoot
Set-Location $repo

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repo "results"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$trainLog = Join-Path $logDir "berlin52_all_stages_$ts.log"
$preCsv = Join-Path $logDir "berlin52_pretrained_proute_$ts.csv"
$fineCsv = Join-Path $logDir "berlin52_finetuned_proute_$ts.csv"
$allFineCsv = Join-Path $logDir "all_finetuned_proute_$ts.csv"

Write-Host "[A] Train pipeline on berlin52 (updated mode)"

$trainArgs = @(
    "scripts/train_with_pretraining_and_gumbel.py",
    "--repo-root", ".",
    "--cpp-exe", $CppExe,
    "--alpha", $Alpha,
    "--label-runs", $LabelRuns,
    "--cpp-timeout", $CppTimeout,
    "--pretrain-datasets", "berlin52",
    "--finetune-datasets", "berlin52",
    "--pretrain-epochs", $PretrainEpochs,
    "--finetune-epochs", $FinetuneEpochs,
    "--sup-weight", $SupWeight,
    "--cf-weight", $CfWeight,
    "--cf-temp", $CfTemp,
    "--entropy-weight", $EntropyWeight,
    "--seed", $Seed,
    "--log-file", $trainLog
)

if ($SkipLabelGeneration) {
    $trainArgs += "--skip-label-generation"
}

& $PythonExe @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training pipeline failed, exit code=$LASTEXITCODE"
}

Write-Host "[B] Report p_route for pretrained model on berlin52"
& $PythonExe "scripts/report_proute_diversification.py" `
    --repo-root . `
    --dataset-dir main_dataset `
    --datasets berlin52 `
    --shared-model-path models/svrap_pretrained_30graphs_sup_rl.pth `
    --title pretrained_on_berlin52 `
    --csv-out $preCsv
if ($LASTEXITCODE -ne 0) {
    throw "Pretrained report failed, exit code=$LASTEXITCODE"
}

Write-Host "[C1] Report p_route for finetuned berlin52"
& $PythonExe "scripts/report_proute_diversification.py" `
    --repo-root . `
    --dataset-dir main_dataset `
    --datasets berlin52 `
    --model-path-template 'models/svrap_finetuned_{dataset}.pth' `
    --title finetuned_on_berlin52 `
    --csv-out $fineCsv
if ($LASTEXITCODE -ne 0) {
    throw "Finetuned berlin52 report failed, exit code=$LASTEXITCODE"
}

Write-Host "[C2] Report p_route for all existing finetuned datasets (single command view)"
& $PythonExe "scripts/report_proute_diversification.py" `
    --repo-root . `
    --dataset-dir main_dataset `
    --model-path-template 'models/svrap_finetuned_{dataset}.pth' `
    --title all_existing_finetuned_models `
    --csv-out $allFineCsv
if ($LASTEXITCODE -ne 0) {
    throw "All-dataset report failed, exit code=$LASTEXITCODE"
}

Write-Host "Done."
Write-Host "train_log=$trainLog"
Write-Host "pretrained_csv=$preCsv"
Write-Host "finetuned_berlin52_csv=$fineCsv"
Write-Host "all_finetuned_csv=$allFineCsv"
