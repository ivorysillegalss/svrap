import subprocess
import re
import time
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
EXE_PATH = "../svrap.exe"
STRATEGY = "full"
DATASET_DIR = "../formatted_dataset"
ALPHAS = [3.0, 5.0, 7.0, 9.0]
OUTPUT_FILE = "../results/constraint_sensitivity_results.csv"

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

def get_all_datasets(directory):
    datasets = []
    if not os.path.exists(directory):
        return datasets
    for f in os.listdir(directory):
        if f.endswith(".txt"):
            datasets.append(os.path.join(directory, f))
    return sorted(datasets)

def run_solver(dataset, alpha, strategy):
    cmd = [EXE_PATH, str(alpha), dataset, strategy]
    # print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stdout
    except Exception as e:
        print(f"Error running solver: {e}")
        return None, 0
    
    elapsed = time.time() - start_time
    
    best_cost = None
    matches = BEST_COST_PATTERN.findall(output)
    if matches:
        best_cost = float(matches[-1])
        
    return best_cost, elapsed

def main():
    if not os.path.exists(EXE_PATH):
        print(f"Error: Executable not found at {EXE_PATH}")
        return

    datasets = get_all_datasets(DATASET_DIR)
    if not datasets:
        print(f"No datasets found in {DATASET_DIR}")
        return

    results = []
    
    print(f"Starting Constraint Sensitivity Test (Varying Alpha)...")
    print(f"Alphas: {ALPHAS}")
    print(f"Datasets: {len(datasets)} found.")
    
    for dataset in datasets:
        dataset_name = os.path.basename(dataset)
        print(f"Processing {dataset_name}...")

        for alpha in ALPHAS:
            cost, time_taken = run_solver(dataset, alpha, STRATEGY)
            
            if cost is not None:
                # print(f"  Alpha={alpha}: Cost={cost}, Time={time_taken:.2f}s")
                results.append({
                    "Dataset": dataset_name,
                    "Alpha": alpha,
                    "Cost": cost,
                    "Time": time_taken
                })
            else:
                print(f"  Alpha={alpha}: Failed.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        
        # Simple plot generation (optional, if matplotlib is installed)
        try:
            # Plot average cost across all datasets for each alpha
            avg_costs = df.groupby("Alpha")["Cost"].mean()
            
            plt.figure()
            avg_costs.plot(marker='o')
            plt.xlabel("Alpha")
            plt.ylabel("Average Cost")
            plt.title("Average Cost vs Alpha (All Datasets)")
            plt.grid(True)
            plt.savefig("../results/constraint_sensitivity_plot.png")
            print("Plot saved to ../results/constraint_sensitivity_plot.png")
        except Exception as e:
            print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
