import subprocess
import re
import time
import os
import sys
import pandas as pd

# Configuration
EXE_PATH = "../svrap.exe"  # Adjust if needed (e.g. "Release/svrap.exe")
PYTHON_SOLVER = "../svrap_solver.py"
ALPHA = 7.0  # Standard test alpha value
STRATEGY = "full" # Default strategy
DATASETS = [
    "../formatted_dataset/kroA100.txt",
    "../formatted_dataset/kroB100.txt",
    "../formatted_dataset/kroC100.txt",
    "../formatted_dataset/kroD100.txt",
    "../formatted_dataset/kroE100.txt"
]
OUTPUT_FILE = "../results/stability_test_results.csv"
NUM_RUNS = 30  # Run each dataset 3 times

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

def run_solver(dataset, alpha, strategy):
    cmd = [EXE_PATH, str(alpha), dataset, strategy]
    # print(f"Running: {' '.join(cmd)}") # Reduce verbosity
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stdout
    except Exception as e:
        print(f"Error running solver: {e}")
        return None, 0
    
    elapsed = time.time() - start_time
    
    # Parse best cost
    best_cost = None
    matches = BEST_COST_PATTERN.findall(output)
    if matches:
        best_cost = float(matches[-1])
        
    return best_cost, elapsed

def run_inference(dataset_path):
    """Runs svrap_solver.py in inference mode to generate attention_probs.csv"""
    cmd = [sys.executable, PYTHON_SOLVER, "--dataset", dataset_path, "--no-train"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating probs for {dataset_path}: {e}")
        return False

def main():
    if not os.path.exists(EXE_PATH):
        print(f"Error: Executable not found at {EXE_PATH}")
        return

    results = []
    
    print(f"Starting Stability Test on kro*100 datasets...")
    print(f"Strategy: {STRATEGY}, Alpha: {ALPHA}, Runs per dataset: {NUM_RUNS}")
    
    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"Warning: Dataset {dataset} not found, skipping.")
            continue
            
        dataset_name = os.path.basename(dataset)
        print(f"Processing {dataset_name}...")
        
        # Generate neural network probabilities first
        print(f"  Generating attention probabilities...")
        run_inference(dataset)
        
        dataset_costs = []
        dataset_times = []
        
        for i in range(NUM_RUNS):
            cost, time_taken = run_solver(dataset, ALPHA, STRATEGY)
            if cost is not None:
                dataset_costs.append(cost)
                dataset_times.append(time_taken)
                results.append({
                    "Dataset": dataset_name,
                    "Run": i + 1,
                    "Cost": cost,
                    "Time": time_taken,
                    "Strategy": STRATEGY
                })
            else:
                print(f"  Run {i+1} failed.")
        
        if dataset_costs:
            avg_cost = sum(dataset_costs) / len(dataset_costs)
            print(f"  Avg Cost: {avg_cost:.2f} (over {len(dataset_costs)} runs)")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        
        # Calculate statistics
        print("\n--- Stability Analysis (Aggregated) ---")
        summary = df.groupby("Dataset")["Cost"].agg(["mean", "std", "min", "max"])
        summary["CV (%)"] = (summary["std"] / summary["mean"]) * 100
        print(summary)
        print("---------------------------------------")

if __name__ == "__main__":
    main()
