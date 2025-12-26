import subprocess
import re
import time
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
SOLVER_PY = "svrap_solver.py"
EXE_PATH = os.path.abspath(os.path.join("..", "svrap.exe"))
DATASET_DIR = "../formatted_dataset" # Used for listing files, relative to script location
OUTPUT_FILE = "../results/epoch_sensitivity_results.csv"

# Epoch values to test
EPOCH_VALUES = [100, 500, 1000, 2000]

# Representative Datasets by Size
# We select 2 datasets for each category to get a balanced view without taking forever.
DATASET_GROUPS = {
    "Small (<100)": ["berlin52.txt", "st70.txt"],
    "Medium (100-200)": ["kroA100.txt", "u159.txt"],
    "Large (>200)": ["kroA200.txt", "d493.txt"] 
}

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

def run_python_training(dataset_path, epochs):
    """Runs the Python script to train the model and generate attention_probs.csv"""
    # Note: svrap_solver.py writes attention_probs.csv to CWD.
    # We run it from root (..) so it writes to root.
    cmd = ["python", SOLVER_PY, "--dataset", dataset_path, "--train", "--epochs", str(epochs)]
    # print(f"    Training Model (Epochs={epochs})...")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd="..", capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"    Training failed: {result.stderr[:200]}...") # Print first 200 chars of error
            return False
    except Exception as e:
        print(f"    Error running python training: {e}")
        return False
    
    elapsed = time.time() - start_time
    # print(f"    Training finished in {elapsed:.2f}s")
    return True

def run_cpp_solver(dataset_path):
    """Runs the C++ solver which uses the generated attention_probs.csv"""
    # C++ solver reads attention_probs.csv from CWD.
    # We run from root (..)
    cmd = [EXE_PATH, "1.0", dataset_path, "full"] # Alpha=1.0, Strategy=full
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd="..", capture_output=True, text=True, check=False)
        output = result.stdout
    except Exception as e:
        print(f"    Error running C++ solver: {e}")
        return None, 0
    
    elapsed = time.time() - start_time
    
    best_cost = None
    matches = BEST_COST_PATTERN.findall(output)
    if matches:
        best_cost = float(matches[-1])
        
    return best_cost, elapsed

def main():
    if not os.path.exists(os.path.join("..", "svrap.exe")):
        print(f"Error: Executable not found at ../svrap.exe")
        return

    results = []
    
    print(f"Starting Comprehensive Epoch Sensitivity Test...")
    print(f"Epochs to test: {EPOCH_VALUES}")
    
    for group_name, filenames in DATASET_GROUPS.items():
        print(f"\n=== Group: {group_name} ===")
        
        for filename in filenames:
            dataset_path = os.path.join(DATASET_DIR, filename)
            # Fix path for subprocess (needs to be relative to root if we run from root, or absolute)
            # Since we run subprocess with cwd="..", we need the path relative to ".."
            # DATASET_DIR is "../formatted_dataset", so relative to ".." it is "formatted_dataset"
            rel_dataset_path = os.path.join("formatted_dataset", filename)
            
            print(f"  Processing {filename}...")
            
            for epochs in EPOCH_VALUES:
                print(f"    [Epochs={epochs}] Training...", end="", flush=True)
                
                # 1. Train
                success = run_python_training(rel_dataset_path, epochs)
                if not success:
                    print(" Failed.")
                    continue
                
                # 2. Solve
                print(" Solving...", end="", flush=True)
                cost, time_taken = run_cpp_solver(rel_dataset_path)
                
                if cost is not None:
                    print(f" Done. Cost={cost:.1f}")
                    results.append({
                        "Group": group_name,
                        "Dataset": filename,
                        "Epochs": epochs,
                        "Cost": cost,
                        "SolverTime": time_taken
                    })
                else:
                    print(" Solver Failed.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")
        
        # Plotting
        try:
            # Create a plot for each group
            groups = df["Group"].unique()
            fig, axes = plt.subplots(1, len(groups), figsize=(15, 5))
            if len(groups) == 1: axes = [axes]
            
            for i, group in enumerate(groups):
                ax = axes[i]
                group_data = df[df["Group"] == group]
                
                # Plot each dataset in the group
                for dataset in group_data["Dataset"].unique():
                    ds_data = group_data[group_data["Dataset"] == dataset]
                    ax.plot(ds_data["Epochs"], ds_data["Cost"], marker='o', label=dataset)
                
                ax.set_title(group)
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Cost")
                ax.legend()
                ax.grid(True)
                
            plt.tight_layout()
            plt.savefig("../results/epoch_sensitivity_plot.png")
            print("Plot saved to ../results/epoch_sensitivity_plot.png")
        except Exception as e:
            print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
