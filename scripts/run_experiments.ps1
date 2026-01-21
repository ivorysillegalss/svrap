# run_experiments.ps1
# Batch execution script for SVRAP experiments
# Steps: Python Policy Network -> attention_probs.csv -> C++ Tabu Search

$ErrorActionPreference = "Continue"

# ================= Configuration =================
$DatasetDir = "formatted_dataset"
$LogFile = "experiment_results.log"
$Alpha = 7.0
$PythonScript = "svrap_solver.py"
$CppExe = ".\svrap.exe"

# Ensure we use the python from the current environment if possible
$PythonExe = (Get-Command python).Source
Write-Host "Using Python: $PythonExe"
# =============================================

# Check required files
if (-not (Test-Path $DatasetDir)) {
    Write-Error "Error: Dataset directory '$DatasetDir' not found."
    exit 1
}
if (-not (Test-Path $PythonScript)) {
    Write-Error "Error: Python script '$PythonScript' not found."
    exit 1
}
if (-not (Test-Path $CppExe)) {
    Write-Error "Error: C++ executable '$CppExe' not found. Please compile first (make)."
    exit 1
}

# Initialize log file
$StartTime = Get-Date
"=== SVRAP Batch Experiments Started at $StartTime ===" | Out-File -FilePath $LogFile -Encoding utf8

# Get all .txt dataset files
$Datasets = Get-ChildItem -Path $DatasetDir -Filter "*.txt"

Write-Host "Found $( $Datasets.Count ) datasets. Starting processing..."
Write-Host "Logs will be written to: $LogFile"

foreach ($File in $Datasets) {
    $DatasetPath = $File.FullName
    $DatasetName = $File.Name
    
    Write-Host "--------------------------------------------------"
    Write-Host "Processing: $DatasetName"
    
    # Write separator to log
    "`n`n==================================================" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "DATASET: $DatasetName" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "TIME: $(Get-Date)" | Out-File -FilePath $LogFile -Append -Encoding utf8
    "==================================================" | Out-File -FilePath $LogFile -Append -Encoding utf8

    # 1. Run Python (Train/Inference + Generate Initial Solution)
    Write-Host "  [1/2] Running Python Policy Network..." -NoNewline
    
    "COMMAND: $PythonExe $PythonScript --dataset `"$DatasetPath`" --train" | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    # Run command, capture output, check status, THEN write to file.
    $pyOutput = & $PythonExe $PythonScript --dataset "$DatasetPath" --train 2>&1
    $pyStatus = $LASTEXITCODE
    
    $pyOutput | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    if ($pyStatus -eq 0) {
        Write-Host " Done" -ForegroundColor Green
    } else {
        Write-Host " Failed (Exit Code: $pyStatus)" -ForegroundColor Red
        "ERROR: Python script failed with exit code $pyStatus" | Out-File -FilePath $LogFile -Append -Encoding utf8
        continue # Skip C++ if Python fails
    }

    # 2. Run C++ (Read Initial Solution + Search)
    Write-Host "  [2/2] Running C++ Tabu Search..." -NoNewline
    
    "COMMAND: $CppExe $Alpha `"$DatasetPath`"" | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    $LASTEXITCODE = 0
    $cppOutput = & $CppExe $Alpha "$DatasetPath" 2>&1
    $cppStatus = $LASTEXITCODE
    
    $cppOutput | Out-File -FilePath $LogFile -Append -Encoding utf8
    
    if ($cppStatus -eq 0) {
        Write-Host " Done" -ForegroundColor Green
    } else {
        Write-Host " Failed (Exit Code: $cppStatus)" -ForegroundColor Red
        "ERROR: C++ executable failed with exit code $cppStatus" | Out-File -FilePath $LogFile -Append -Encoding utf8
    }
}

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "--------------------------------------------------"
Write-Host "All experiments finished."
Write-Host "Total Duration: $Duration"
Write-Host "Detailed logs: $LogFile"
