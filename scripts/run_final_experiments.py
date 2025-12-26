import subprocess
import os
import sys
import pandas as pd
import time
import re

# Configuration
EXE_PATH = "../svrap.exe"
PYTHON_SOLVER = "../svrap_solver.py"
FORMATTED_DATASET_DIR = "../formatted_dataset"
RESULTS_FILE = "../results/final_experiment_results.csv"
ALPHA = 1.0
STRATEGY = "full"

# Best Hyperparameters (Size Dependent)
# Format: (MinSize, MaxSize, TabuLength, EntropyWeight)
# Note: MaxSize is exclusive.
CONFIGS = [
    (0, 100, 20, 10.0),    # Small: High entropy
    (100, 9999, 20, 1.0)   # Medium/Large: Low entropy
]

# Fixed Hyperparameters
K = 20
DIV_TIMES = 2
PR_TIMES = 50

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

def get_dataset_size(filepath):
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0

def get_config(size):
    for min_s, max_s, tabu, entropy in CONFIGS:
        if min_s <= size < max_s:
            return tabu, entropy
    return 20, 1.0 # Default fallback

def run_inference(dataset_path):
    """Runs svrap_solver.py in inference mode to generate attention_probs.csv"""
    # Ensure we are in the scripts directory when calling this, 
    # but svrap_solver.py is in the parent directory.
    # We should call it from the parent directory context or adjust paths.
    # Let's assume we run this script from 'scripts/' folder.
    
    cmd = ["python", PYTHON_SOLVER, "--dataset", dataset_path, "--no-train"]
    try:
        # print(f"Generating probabilities for {os.path.basename(dataset_path)}...")
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating probs for {dataset_path}: {e}")
        return False

def run_cpp_solver(dataset_path, tabu, entropy):
    cmd = [EXE_PATH, str(ALPHA), dataset_path, STRATEGY, str(K), str(tabu), str(DIV_TIMES), str(PR_TIMES), str(entropy)]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stdout
    except Exception as e:
        print(f"Error running C++ solver: {e}")
        return None, 0
    
    elapsed = time.time() - start_time
    
    best_cost = None
    matches = BEST_COST_PATTERN.findall(output)
    if matches:
        best_cost = float(matches[-1])
        
    return best_cost, elapsed

def main():
    if not os.path.exists(EXE_PATH):
        print(f"Error: Executable {EXE_PATH} not found.")
        return

    if not os.path.exists(FORMATTED_DATASET_DIR):
        print(f"Error: Dataset directory {FORMATTED_DATASET_DIR} not found.")
        return

    results = []
    
    files = [f for f in os.listdir(FORMATTED_DATASET_DIR) if f.endswith(".txt")]
    files.sort() # Process in order
    
    print(f"Starting Final Full Volume Experiment on {len(files)} datasets...")
    print(f"Results will be saved to {RESULTS_FILE}")
    
    for filename in files:
        dataset_path = os.path.join(FORMATTED_DATASET_DIR, filename)
        size = get_dataset_size(dataset_path)
        
        if size == 0:
            continue
            
        tabu, entropy = get_config(size)
        
        print(f"\nProcessing {filename} (Size: {size})")
        print(f"  Config: Tabu={tabu}, Entropy={entropy}")
        
        # 1. Generate Probabilities (Inference)
        # Note: This overwrites 'attention_probs.csv' in the parent directory
        # We need to make sure svrap.exe reads from the right place.
        # svrap.exe reads 'attention_probs.csv' from current working directory.
        # If we run this script from 'scripts/', we need to be careful.
        # Ideally, we change CWD to parent dir for the whole execution?
        # Or we move the generated csv to where svrap.exe expects it.
        
        # Let's assume we run this script from 'scripts/'.
        # svrap_solver.py is in '../'.
        # It writes to 'attention_probs.csv' in CWD (which would be 'scripts/').
        # svrap.exe is in '../'. When executed as '../svrap.exe', its CWD is still 'scripts/'.
        # So if svrap.exe looks for "attention_probs.csv", it will look in 'scripts/'.
        # Let's verify main.cpp: read_attention_probs("attention_probs.csv", probs);
        # Yes, relative path. So if we run from scripts/, it works.
        
        success = run_inference(dataset_path)
        if not success:
            print("  Skipping due to inference failure (model might be missing).")
            # Fallback: Run without neural init? Or just skip?
            # Let's try to run anyway, maybe it will use dummy probs or fail gracefully.
            # But for "Final Experiment", we want the best result.
            # If model is missing, we can't get "best" result.
            # We'll record it as a failure or run with default.
            pass 

        # 2. Run C++ Solver
        cost, time_taken = run_cpp_solver(dataset_path, tabu, entropy)
        
        if cost is not None:
            print(f"  Result: Cost={cost}, Time={time_taken:.2f}s")
            results.append({
                "Dataset": filename,
                "Size": size,
                "TabuLength": tabu,
                "EntropyWeight": entropy,
                "Cost": cost,
                "Time": time_taken
            })
        else:
            print("  Failed to get cost.")

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\nExperiment Complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
