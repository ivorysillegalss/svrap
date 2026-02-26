#!/usr/bin/env python
"""
SVRAP 完整测试流水线 (简化版)
直接在当前 Python 环境运行，请确保已激活 altr-py310

使用方法:
    conda activate altr-py310
    cd d:\svrap1\svrap\scripts
    python run_full_test.py
"""

import subprocess
import os
import sys
import time
import re
import csv
from datetime import datetime

# ========================================
# Configuration
# ========================================

DATASETS = [
    "eil51", "eil76", "eil101",
    "berlin52", "st70", "pr76", "rat99", "rd100",
    "kroA100", "kroB100", "kroC100", "kroD100", "kroE100",
    "lin105", "pr107", "gr96", "gr120", "bier127",
    "ch130", "pr124", "pr136", "gr137", "pr144",
    "ch150", "kroA150", "kroB150", "pr152", "u159",
    "rat195", "d198", "kroA200", "kroB200",
    "d493", "rat783"
]

ALPHA = 7.0
EPOCHS = 2000

# Resolve project root relative to this script so that execution works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXE_PATH = os.path.join(PROJECT_ROOT, "svrap.exe")
PYTHON_SOLVER = os.path.join(PROJECT_ROOT, "svrap_solver.py")
HEARTBEAT_INTERVAL = 600  # 10 minutes

BEST_COST_PATTERN = re.compile(r"Best cost.*?=\s*([0-9.]+)")

# Global timing
start_time = None
last_heartbeat = None

def heartbeat(phase, task, progress):
    """Print heartbeat if interval elapsed"""
    global last_heartbeat
    now = time.time()
    
    if last_heartbeat is None or (now - last_heartbeat) >= HEARTBEAT_INTERVAL:
        elapsed = now - start_time
        hours, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        
        print("\n" + "=" * 60)
        print(f"[HEARTBEAT] {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Phase: {phase}")
        print(f"  Task: {task}")
        print(f"  Progress: {progress}")
        print(f"  Elapsed: {hours:02d}:{mins:02d}:{secs:02d}")
        print("=" * 60 + "\n", flush=True)
        
        last_heartbeat = now

def train_model(dataset_name):
    """Train model for one dataset"""
    dataset_path = os.path.join(PROJECT_ROOT, "formatted_dataset", f"{dataset_name}.txt")
    
    if not os.path.exists(dataset_path):
        return False, "Not found"
    
    cmd = [
        sys.executable, PYTHON_SOLVER,
        "--dataset", dataset_path,
        "--train", "--epochs", str(EPOCHS)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Extract best cost
        for line in result.stdout.split('\n'):
            if 'Training finished' in line:
                return True, line.strip()
        
        if result.returncode != 0:
            return False, result.stderr[:100]
        return True, "OK"
        
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C detected, stopping...")
        raise
    except Exception as e:
        return False, str(e)[:50]

def run_inference(dataset_path):
    """Run inference to generate backbone"""
    cmd = [sys.executable, PYTHON_SOLVER, "--dataset", dataset_path, "--no-train"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        return True
    except:
        return False

def run_cpp_solver(dataset_path, strategy="full"):
    """Run C++ solver"""
    cmd = [EXE_PATH, str(ALPHA), dataset_path, strategy]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        matches = BEST_COST_PATTERN.findall(result.stdout)
        if matches:
            return float(matches[-1])
    except:
        pass
    return None

def phase1_train():
    """Phase 1: Train all models"""
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING MODELS")
    print(f"Datasets: {len(DATASETS)}, Epochs: {EPOCHS}")
    print("=" * 70 + "\n")
    
    results = []
    for i, ds in enumerate(DATASETS, 1):
        progress = f"{i}/{len(DATASETS)}"
        heartbeat("Training", ds, progress)
        
        print(f"[{progress}] Training {ds}...", end=" ", flush=True)
        t0 = time.time()
        ok, msg = train_model(ds)
        elapsed = time.time() - t0
        
        status = "OK" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s) - {msg}")
        results.append((ds, ok, elapsed))
    
    success = sum(1 for _, ok, _ in results if ok)
    print(f"\nPhase 1 Complete: {success}/{len(DATASETS)} models trained")
    return results

def phase2_test():
    """Phase 2: Run ablation study"""
    print("\n" + "=" * 70)
    print("PHASE 2: ABLATION STUDY")
    print("=" * 70 + "\n")
    
    strategies = ["full", "baseline", "no_nn", "no_entropy", "no_knn"]
    test_datasets = ["eil51", "kroA100", "ch130", "pr152", "rat195"]
    
    results = []
    total = len(test_datasets) * len(strategies)
    count = 0
    
    for ds in test_datasets:
        ds_path = os.path.join(PROJECT_ROOT, "formatted_dataset", f"{ds}.txt")
        
        # Generate backbone first
        print(f"\n[{ds}] Generating backbone...", flush=True)
        run_inference(ds_path)
        
        for strategy in strategies:
            count += 1
            heartbeat("Ablation", f"{ds}/{strategy}", f"{count}/{total}")
            
            print(f"  {strategy}: ", end="", flush=True)
            cost = run_cpp_solver(ds_path, strategy)
            
            if cost:
                print(f"{cost:.2f}")
                results.append({"Dataset": ds, "Strategy": strategy, "Cost": cost})
            else:
                print("FAILED")
    
    # Save results
    if results:
          output_path = os.path.join(PROJECT_ROOT, "results", "ablation_results.csv")
          os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
          with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Dataset", "Strategy", "Cost"])
            writer.writeheader()
            writer.writerows(results)
          print(f"\nResults saved to {output_path}")
    
    return results

def phase3_stability():
    """Phase 3: Stability test"""
    print("\n" + "=" * 70)
    print("PHASE 3: STABILITY TEST (5 runs per dataset)")
    print("=" * 70 + "\n")
    
    test_datasets = ["kroA100", "kroB100", "kroC100"]
    num_runs = 5
    results = []
    
    for ds in test_datasets:
        ds_path = os.path.join(PROJECT_ROOT, "formatted_dataset", f"{ds}.txt")
        
        print(f"\n[{ds}]", flush=True)
        run_inference(ds_path)
        
        costs = []
        for run in range(num_runs):
            heartbeat("Stability", ds, f"Run {run+1}/{num_runs}")
            cost = run_cpp_solver(ds_path, "full")
            if cost:
                costs.append(cost)
                results.append({"Dataset": ds, "Run": run+1, "Cost": cost})
                print(f"  Run {run+1}: {cost:.2f}")
        
        if costs:
            avg = sum(costs) / len(costs)
            std = (sum((c - avg)**2 for c in costs) / len(costs)) ** 0.5
            print(f"  Avg: {avg:.2f} ± {std:.2f}")
    
    # Save
    if results:
          stability_path = os.path.join(PROJECT_ROOT, "results", "stability_results.csv")
          with open(stability_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Dataset", "Run", "Cost"])
            writer.writeheader()
            writer.writerows(results)
          print(f"\nResults saved to {stability_path}")

def main():
    global start_time, last_heartbeat
    start_time = time.time()
    last_heartbeat = start_time  # Print first heartbeat immediately
    
    print("=" * 70)
    print("SVRAP FULL TEST PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check prerequisites
    if not os.path.exists(EXE_PATH):
        print(f"ERROR: {EXE_PATH} not found. Please compile first.")
        return
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch not found. Please activate altr-py310 environment.")
        return
    
    # Run phases
    phase1_train()
    phase2_test()
    phase3_stability()
    
    # Summary
    total_time = time.time() - start_time
    hours, rem = divmod(int(total_time), 3600)
    mins, secs = divmod(rem, 60)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"Total time: {hours:02d}:{mins:02d}:{secs:02d}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
