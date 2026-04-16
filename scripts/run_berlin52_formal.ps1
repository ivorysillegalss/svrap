param(
    [string]$PythonExe = "C:/Users/chenz/miniconda3/envs/altr-py310/python.exe",
    [string]$RepoRoot = ".",
    [string]$CppExe = "svrap.exe",
    [switch]$SkipLabelGeneration
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path $RepoRoot
Set-Location $repo

$argsForAllStages = @(
    "-PythonExe", $PythonExe,
    "-RepoRoot", ".",
    "-CppExe", $CppExe,
    "-PretrainEpochs", 200,
    "-FinetuneEpochs", 100,
    "-LabelRuns", 10,
    "-CppTimeout", 1800,
    "-Seed", 42,
    "-Alpha", 7.0,
    "-SupWeight", 1.0,
    "-CfWeight", 1.0,
    "-CfTemp", 500.0,
    "-EntropyWeight", 0.01
)

if ($SkipLabelGeneration) {
    $argsForAllStages += "-SkipLabelGeneration"
}

Write-Host "Running Berlin52 formal pipeline with production-aligned settings:"
Write-Host "pretrain_epochs=200, finetune_epochs=100, label_runs=10"

& powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_berlin52_all_stages.ps1" @argsForAllStages
if ($LASTEXITCODE -ne 0) {
    throw "Berlin52 formal pipeline failed, exit code=$LASTEXITCODE"
}

Write-Host "Berlin52 formal pipeline finished successfully."
