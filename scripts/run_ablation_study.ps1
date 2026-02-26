$datasets = @(
    # Clustered / Drilling Problems (Highly structured, non-uniform)
    "d198.txt",
    "d493.txt",

    # Geographical / Real-world (Uneven density, natural clusters)
    "u159.txt",
    "gr96.txt",
    "gr120.txt",
    "gr137.txt",
    "bier127.txt",
    "berlin52.txt",

    # Other potentially interesting ones
    "rat783.txt", # Large grid/curve
    "pr107.txt",
    "pr152.txt"
)

$strategies = @("baseline", "no_nn", "no_entropy", "no_knn", "simple_div", "full")
$alpha = 7
$numRuns = 30
$root = ".."  # Assuming running from scripts/ directory
$exePath = "$root\svrap.exe"
$resultsDir = "$root\results"
$resultsFile = "$resultsDir\ablation_results.csv"

# Ensure results directory exists
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# Initialize CSV if it doesn't exist (overwrite if cleaning)
"Dataset,Strategy,Run,BestCost,Time" | Out-File -FilePath $resultsFile -Encoding utf8

foreach ($dataset in $datasets) {
    $datasetPath = "$root\formatted_dataset\$dataset"
    
    # Check if dataset exists
    if (-not (Test-Path $datasetPath)) {
        Write-Host "Dataset $dataset not found, skipping..."
        continue
    }

    # Generate neural probabilities ONCE per dataset
    Write-Host "Generating probabilities for $dataset..."
    # Using python from path, ensure environment is active
    & "C:\Users\chenz\miniconda3\envs\altr-py310\python.exe" "$root\svrap_solver.py" --dataset $datasetPath --no-train
    
    foreach ($strategy in $strategies) {
        Write-Host "Running $dataset with $strategy ($numRuns runs)..."
        
        for ($i = 1; $i -le $numRuns; $i++) {
            # Run the executable and capture output
            $output = & $exePath $alpha $datasetPath $strategy 2>&1
            
            # Parse output for Best Cost and Time
            $bestCost = "N/A"
            $time = "N/A"
            
            foreach ($line in $output) {
                if ($line -match "Best cost for .* = ([\d\.]+)") {
                    $bestCost = $matches[1]
                }
                # Fallback for simpler output
                if ($bestCost -eq "N/A" -and $line -match "Best cost.*?=\s*([\d\.]+)") {
                    $bestCost = $matches[1]
                }

                if ($line -match "Tabu search finished in ([\d\.]+)s") {
                    $time = $matches[1]
                }
                 # Fallback for simpler output
                 if ($time -eq "N/A" -and $line -match "finished in\s*([\d\.]+)s") {
                    $time = $matches[1]
                }
            }
            
            "$dataset,$strategy,$i,$bestCost,$time" | Out-File -FilePath $resultsFile -Append -Encoding utf8
            
            # Optional: Print progress every 5 runs
            if ($i % 5 -eq 0) {
                Write-Host "  -> Run $i/${numRuns}: Cost=$bestCost, Time=${time}s"
            }
        }
    }
}
Write-Host "Ablation study complete. Results saved to $resultsFile"
