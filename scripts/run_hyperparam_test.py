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
DATASET = "../formatted_dataset/kroA100.txt" # Focus on one dataset for hyperparam tuning
OUTPUT_FILE = "../results/hyperparam_sensitivity_results.csv"

# Hyperparameter Ranges
K_VALUES = [10, 20, 30]
TABU_LENGTHS = [10, 15, 20]
DIV_TIMES = [1, 2, 3]
PR_TIMES = [25, 50, 75]

# Defaults
DEFAULT_K = 20
DEFAULT_TABU = 15
DEFAULT_DIV = 2
DEFAULT_PR = 50

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

def run_solver(dataset, alpha, strategy, k, tabu_len, div_times, pr_times):
    # Command format: svrap.exe <alpha> <dataset> <strategy> <k> <tabu_len> <div_times> <pr_times>
    cmd = [EXE_PATH, str(alpha), dataset, strategy, str(k), str(tabu_len), str(div_times), str(pr_times)]
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

    if not os.path.exists(DATASET):
        print(f"Error: Dataset {DATASET} not found.")
        return

    results = []
    
    print(f"Starting Comprehensive Hyperparameter Sensitivity Test...")
    print(f"Dataset: {DATASET}")
    
    # 1. Test K (KNN Neighbors)
    print("\nTesting K (KNN Neighbors)...")
    for k in K_VALUES:
        cost, time_taken = run_solver(DATASET, ALPHA, STRATEGY, k, DEFAULT_TABU, DEFAULT_DIV, DEFAULT_PR)
        if cost is not None:
            results.append({"Parameter": "K", "Value": k, "Cost": cost, "Time": time_taken})
            print(f"  K={k}: Cost={cost}")

    # 2. Test Tabu List Length
    print("\nTesting Tabu List Length...")
    for tl in TABU_LENGTHS:
        cost, time_taken = run_solver(DATASET, ALPHA, STRATEGY, DEFAULT_K, tl, DEFAULT_DIV, DEFAULT_PR)
        if cost is not None:
            results.append({"Parameter": "TabuLength", "Value": tl, "Cost": cost, "Time": time_taken})
            print(f"  TabuLength={tl}: Cost={cost}")

    # 3. Test Diversification Times
    print("\nTesting Diversification Times...")
    for dt in DIV_TIMES:
        cost, time_taken = run_solver(DATASET, ALPHA, STRATEGY, DEFAULT_K, DEFAULT_TABU, dt, DEFAULT_PR)
        if cost is not None:
            results.append({"Parameter": "Diversification", "Value": dt, "Cost": cost, "Time": time_taken})
            print(f"  Diversification={dt}: Cost={cost}")

    # 4. Test Path Relinking Times
    print("\nTesting Path Relinking Times...")
    for pr in PR_TIMES:
        cost, time_taken = run_solver(DATASET, ALPHA, STRATEGY, DEFAULT_K, DEFAULT_TABU, DEFAULT_DIV, pr)
        if cost is not None:
            results.append({"Parameter": "PathRelinking", "Value": pr, "Cost": cost, "Time": time_taken})
            print(f"  PathRelinking={pr}: Cost={cost}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        
        # Plotting
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            params = ["K", "TabuLength", "Diversification", "PathRelinking"]
            for i, param in enumerate(params):
                ax = axes[i//2, i%2]
                data = df[df["Parameter"] == param]
                ax.plot(data["Value"], data["Cost"], marker='o')
                ax.set_title(f"Effect of {param}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Cost")
                ax.grid(True)
                
            plt.tight_layout()
            plt.savefig("../results/hyperparam_sensitivity_plot.png")
            print("Plot saved to ../results/hyperparam_sensitivity_plot.png")
        except Exception as e:
            print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
