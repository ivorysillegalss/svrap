import subprocess
import re
import time
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
EXE_PATH = "../svrap.exe"
ALPHA = 1.0
STRATEGY = "full"
OUTPUT_FILE = "../results/entropy_sensitivity_summary.csv"

# Datasets to test
DATASETS = [
    "../formatted_dataset/eil51.txt",
    "../formatted_dataset/kroA100.txt",
    "../formatted_dataset/pr152.txt",
    "../formatted_dataset/rat195.txt"
]

# Hyperparameter Ranges
ENTROPY_WEIGHTS = [0, 1, 10, 100, 1000]

# Defaults for other params
DEFAULT_K = 20
DEFAULT_TABU = 15
DEFAULT_DIV = 2
DEFAULT_PR = 50

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

def generate_probs(dataset_path):
    """Generates dummy attention_probs.csv for the given dataset."""
    cmd = ["python", "generate_dummy_probs.py", dataset_path, "attention_probs.csv"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # print(f"Generated dummy probs for {os.path.basename(dataset_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating probs for {dataset_path}: {e}")

def run_solver(dataset, alpha, strategy, k, tabu_len, div_times, pr_times, entropy_weight):
    # Command format: svrap.exe <alpha> <dataset> <strategy> <k> <tabu_len> <div_times> <pr_times> <entropy_weight>
    cmd = [EXE_PATH, str(alpha), dataset, strategy, str(k), str(tabu_len), str(div_times), str(pr_times), str(entropy_weight)]
    
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

    results = []
    
    print(f"Starting Comprehensive Entropy Weight Sensitivity Test...")
    
    for dataset_path in DATASETS:
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset {dataset_path} not found, skipping.")
            continue
            
        dataset_name = os.path.basename(dataset_path)
        print(f"\nProcessing Dataset: {dataset_name}")
        
        # Generate matching dummy probabilities
        generate_probs(dataset_path)
        
        for ew in ENTROPY_WEIGHTS:
            cost, time_taken = run_solver(dataset_path, ALPHA, STRATEGY, DEFAULT_K, DEFAULT_TABU, DEFAULT_DIV, DEFAULT_PR, ew)
            if cost is not None:
                results.append({
                    "Dataset": dataset_name,
                    "EntropyWeight": ew,
                    "Cost": cost,
                    "Time": time_taken
                })
                print(f"  EntropyWeight={ew}: Cost={cost}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        
        # Plotting
        datasets = df['Dataset'].unique()
        plt.figure(figsize=(12, 8))
        
        for ds in datasets:
            subset = df[df['Dataset'] == ds]
            # Normalize cost to see relative improvement/degradation
            base_cost = subset[subset['EntropyWeight'] == 0]['Cost'].values
            if len(base_cost) > 0:
                base_val = base_cost[0]
                normalized_costs = subset['Cost'] / base_val
                plt.plot(subset['EntropyWeight'], normalized_costs, marker='o', label=ds)
            else:
                plt.plot(subset['EntropyWeight'], subset['Cost'], marker='o', label=ds)

        plt.title('Entropy Weight Sensitivity (Normalized Cost)')
        plt.xlabel('Entropy Weight')
        plt.ylabel('Normalized Cost (Relative to Weight=0)')
        plt.xscale('symlog') # symlog handles 0 better than log
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_FILE.replace('.csv', '_normalized.png'))
        print(f"Normalized plot saved to {OUTPUT_FILE.replace('.csv', '_normalized.png')}")

if __name__ == "__main__":
    main()
