param(
    [string]$PythonExe = "python",
    [string]$RepoRoot = ".",
    [string]$CppExe = "svrap.exe",
    [double]$Alpha = 7.0,
    [int]$PretrainEpochs = 200,
    [int]$FinetuneEpochs = 100,
    [int]$LabelRuns = 10,
    [int]$GuidedVsNoNnRuns = 10,
    [int]$CppTimeout = 1800,
    [int]$Seed = 42,
    [double]$SupWeight = 1.0,
    [double]$CfWeight = 1.0,
    [double]$CfTemp = 500.0,
    [double]$EntropyWeight = 0.01,
    [switch]$SkipLabelGeneration
)

$ErrorActionPreference = "Stop"

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    if ([string]::IsNullOrWhiteSpace($Name)) {
        throw "Executable name cannot be empty."
    }

    if ([System.IO.Path]::IsPathRooted($Name)) {
        if (Test-Path $Name) {
            return (Resolve-Path $Name).Path
        }
    } else {
        $candidate = Join-Path $RepoRoot $Name
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }

    throw "Executable not found: $Name"
}

function Invoke-Step {
    param(
        [string]$Title,
        [string]$Exe,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "==== $Title ===="
    Write-Host "$Exe $($CommandArgs -join ' ')"
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit code=$LASTEXITCODE)"
    }
}

$repo = Resolve-Path $RepoRoot
Set-Location $repo

$resolvedPythonExe = Resolve-Executable -Name $PythonExe -RepoRoot $repo
$resolvedCppExe = Resolve-Executable -Name $CppExe -RepoRoot $repo

Write-Host "Using Python executable: $resolvedPythonExe"
Write-Host "Using C++ executable: $resolvedCppExe"

$resultDir = Join-Path $repo "results"
if (-not (Test-Path $resultDir)) {
    New-Item -ItemType Directory -Path $resultDir | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"

$trainLog = Join-Path $resultDir "pipeline_train_$ts.log"
$divCsv = Join-Path $resultDir "pipeline_proute_diversification_$ts.csv"
$guidedRaw = Join-Path $resultDir "guided_vs_no_nn_raw_$ts.csv"
$guidedSummaryCsv = Join-Path $resultDir "guided_vs_no_nn_summary_$ts.csv"
$guidedSummaryMd = Join-Path $resultDir "guided_vs_no_nn_summary_$ts.md"
$inferCsv = Join-Path $resultDir "main_dataset_p_route_stats_$ts.csv"
$heatmapDir = Join-Path $resultDir "p_route_heatmaps_$ts"
$heatmapMd = Join-Path $resultDir "main_dataset_p_route_summary_$ts.md"

# Mitigate common OpenMP duplicate runtime conflicts in plotting/torch stacks.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"

$trainArgs = @(
    "scripts/train_with_pretraining_and_gumbel.py",
    "--repo-root", ".",
    "--cpp-exe", $CppExe,
    "--alpha", "$Alpha",
    "--label-runs", "$LabelRuns",
    "--cpp-timeout", "$CppTimeout",
    "--pretrain-epochs", "$PretrainEpochs",
    "--finetune-epochs", "$FinetuneEpochs",
    "--sup-weight", "$SupWeight",
    "--cf-weight", "$CfWeight",
    "--cf-temp", "$CfTemp",
    "--entropy-weight", "$EntropyWeight",
    "--seed", "$Seed",
    "--log-file", $trainLog
)
if ($SkipLabelGeneration) {
    $trainArgs += "--skip-label-generation"
}

Invoke-Step -Title "Stage A: Label + Pretrain + Finetune (all main_dataset)" -Exe $resolvedPythonExe -CommandArgs $trainArgs

Invoke-Step -Title "Stage B: p_route diversification summary" -Exe $resolvedPythonExe -CommandArgs @(
    "scripts/report_proute_diversification.py",
    "--repo-root", ".",
    "--dataset-dir", "main_dataset",
    "--model-path-template", "models/svrap_finetuned_{dataset}.pth",
    "--title", "main_dataset_finetuned_models",
    "--csv-out", $divCsv
)

Invoke-Step -Title "Stage C: Guided(full) vs no_nn, 10 runs each" -Exe $resolvedPythonExe -CommandArgs @(
    "scripts/compare_guided_vs_no_nn_gap.py",
    "--repo-root", ".",
    "--main-dataset-dir", "main_dataset",
    "--models-dir", "models",
    "--model-prefix", "svrap_finetuned_",
    "--exe", $resolvedCppExe,
    "--alpha", "$Alpha",
    "--runs", "$GuidedVsNoNnRuns",
    "--timeout", "$CppTimeout",
    "--raw-out", $guidedRaw,
    "--summary-out", $guidedSummaryCsv,
    "--summary-md", $guidedSummaryMd,
    "--seed", "$Seed"
)

Invoke-Step -Title "Stage D: Inference stats export (all main_dataset)" -Exe $resolvedPythonExe -CommandArgs @(
    "scripts/infer_main_dataset_p_route.py",
    "--repo-root", ".",
    "--main-dataset-dir", "main_dataset",
    "--models-dir", "models",
    "--model-prefix", "svrap_finetuned_",
    "--output-csv", $inferCsv,
    "--topk-ratio", "0.2",
    "--seed", "$Seed"
)

Invoke-Step -Title "Stage E: Heatmap export (all main_dataset)" -Exe $resolvedPythonExe -CommandArgs @(
    "scripts/export_p_route_heatmaps_and_md.py",
    "--repo-root", ".",
    "--main-dataset-dir", "main_dataset",
    "--models-dir", "models",
    "--model-prefix", "svrap_finetuned_",
    "--summary-csv", $inferCsv,
    "--output-heatmap-dir", $heatmapDir,
    "--output-md", $heatmapMd,
    "--topk-ratio", "0.2",
    "--seed", "$Seed"
)

Write-Host ""
Write-Host "Pipeline completed successfully."
Write-Host "train_log=$trainLog"
Write-Host "diversification_csv=$divCsv"
Write-Host "guided_vs_no_nn_raw_csv=$guidedRaw"
Write-Host "guided_vs_no_nn_summary_csv=$guidedSummaryCsv"
Write-Host "guided_vs_no_nn_summary_md=$guidedSummaryMd"
Write-Host "inference_csv=$inferCsv"
Write-Host "heatmap_dir=$heatmapDir"
Write-Host "heatmap_summary_md=$heatmapMd"
