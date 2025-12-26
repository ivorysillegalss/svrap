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
$exePath = ".\svrap_test.exe"
$resultsFile = "ablation_results.csv"

# Initialize CSV if it doesn't exist
if (-not (Test-Path $resultsFile)) {
    "Dataset,Strategy,Run,BestCost,Time" | Out-File -FilePath $resultsFile -Encoding utf8
}

foreach ($dataset in $datasets) {
    $datasetPath = "formatted_dataset/$dataset"
    
    # Check if dataset exists
    if (-not (Test-Path $datasetPath)) {
        Write-Host "Dataset $dataset not found, skipping..."
        continue
    }

    # Generate neural probabilities ONCE per dataset (or per run if we want to test NN stability too?)
    # Usually, we want to test the search stability given a fixed NN output, OR the whole system stability.
    # Given the NN training might be deterministic or not, let's generate it once per dataset to save time,
    # unless you want to test NN training variance too.
    # Assuming we use the pre-trained model or train once.
    # Let's run inference once per dataset to get attention_probs.csv.
    
    Write-Host "Generating probabilities for $dataset..."
    # Ensure we use the correct python environment with torch
    python svrap_solver.py --dataset $datasetPath --no-train
    
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
                if ($line -match "Tabu search finished in ([\d\.]+)s") {
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
