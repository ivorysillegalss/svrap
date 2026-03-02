"""
SVRAP 完整测试流水线
- 阶段1: 为所有数据集训练模型
- 阶段2: 运行完整测试套件 (内部采用单任务带参模式，规避了 C++ 内部循环对 CSV 预读的问题)
- 包含心跳监控 (每10分钟输出状态)
"""

import subprocess
import os
import sys
import time
import threading
from datetime import datetime, timedelta

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
EXE_PATH = "../svrap.exe"
PYTHON_SOLVER = "../svrap_solver.py"
HEARTBEAT_INTERVAL = 600  # 10 minutes in seconds

# Conda environment configuration
CONDA_ENV = "altr-py310"
CONDA_PATH = r"C:\Users\chenz\miniconda3\Scripts\conda.exe"

# Global state for heartbeat
heartbeat_info = {
    "phase": "Initializing",
    "current_task": "",
    "progress": "",
    "start_time": None,
    "running": True
}

# ========================================
# Heartbeat Monitor
# ========================================

def heartbeat_monitor():
    """Background thread that prints heartbeat every 10 minutes"""
    last_beat = time.time()
    
    while heartbeat_info["running"]:
        time.sleep(10)  # Check every 10 seconds
        
        if time.time() - last_beat >= HEARTBEAT_INTERVAL:
            elapsed = datetime.now() - heartbeat_info["start_time"]
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "=" * 60)
            print(f"[HEARTBEAT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Phase: {heartbeat_info['phase']}")
            print(f"  Current Task: {heartbeat_info['current_task']}")
            print(f"  Progress: {heartbeat_info['progress']}")
            print(f"  Total Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print("=" * 60 + "\n")
            sys.stdout.flush()
            
            last_beat = time.time()

def update_heartbeat(phase=None, task=None, progress=None):
    """Update heartbeat information"""
    if phase:
        heartbeat_info["phase"] = phase
    if task:
        heartbeat_info["current_task"] = task
    if progress:
        heartbeat_info["progress"] = progress

# ========================================
# Phase 1: Train Models
# ========================================

def train_model(dataset_name):
    """Train model for a single dataset"""
    dataset_path = f"../formatted_dataset/{dataset_name}.txt"
    
    if not os.path.exists(dataset_path):
        print(f"  [SKIP] Dataset not found: {dataset_path}")
        return False
    
    # Use conda run to execute in the correct environment
    cmd = [
        CONDA_PATH, "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", PYTHON_SOLVER,
        "--dataset", dataset_path,
        "--train",
        "--epochs", str(EPOCHS)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=3600  # 1 hour timeout per dataset
        )
        
        if result.returncode == 0:
            # Extract best cost from output
            for line in result.stdout.split('\n'):
                if 'Best Cost' in line or 'Training finished' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"  [ERROR] Training failed: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Training exceeded 1 hour limit")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def phase1_train_all_models():
    """Train models for all datasets"""
    update_heartbeat(phase="Phase 1: Training Models")
    
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING NEURAL NETWORK MODELS")
    print("=" * 70)
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Epochs per dataset: {EPOCHS}")
    print(f"Estimated time: ~{len(DATASETS) * 2} - {len(DATASETS) * 5} minutes")
    print("-" * 70)
    
    successful = 0
    failed = []
    
    for i, dataset in enumerate(DATASETS, 1):
        progress = f"{i}/{len(DATASETS)} ({100*i//len(DATASETS)}%)"
        update_heartbeat(task=f"Training {dataset}", progress=progress)
        
        print(f"\n[{progress}] Training model for {dataset}...")
        
        if train_model(dataset):
            successful += 1
            print(f"  [OK] {dataset} completed")
        else:
            failed.append(dataset)
            print(f"  [FAIL] {dataset}")
    
    print("\n" + "-" * 70)
    print(f"Training Summary: {successful}/{len(DATASETS)} successful")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")
    print("-" * 70)
    
    return successful, failed

# ========================================
# Phase 2: Run Tests
# ========================================

def run_inference(dataset_path):
    """Run Python inference to generate backbone"""
    cmd = [
        CONDA_PATH, "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", PYTHON_SOLVER, "--dataset", dataset_path, "--no-train"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
        return result.returncode == 0
    except:
        return False

def run_cpp_solver(dataset_path, strategy="full"):
    """Run C++ solver"""
    cmd = [EXE_PATH, str(ALPHA), dataset_path, strategy]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
        
        # Extract best cost
        for line in result.stdout.split('\n'):
            if 'Best cost' in line:
                import re
                match = re.search(r'Best cost.*?=\s*([0-9.]+)', line)
                if match:
                    return float(match.group(1))
        return None
    except:
        return None

def phase2_run_tests():
    """Run ablation and comparison tests"""
    update_heartbeat(phase="Phase 2: Running Tests")
    
    print("\n" + "=" * 70)
    print("PHASE 2: RUNNING TESTS")
    print("=" * 70)
    
    results = []
    strategies = ["full", "baseline", "no_entropy", "no_knn"]
    
    # Use a subset for testing (can expand later)
    test_datasets = [
        "eil51", "eil76", "eil101",
        "kroA100", "kroB100", "kroC100", "kroD100", "kroE100",
        "berlin52", "st70", "pr76", "rat99", "rd100"
    ]
    
    total_tests = len(test_datasets) * len(strategies)
    current_test = 0
    
    for dataset in test_datasets:
        dataset_path = f"../formatted_dataset/{dataset}.txt"
        
        if not os.path.exists(dataset_path):
            print(f"[SKIP] {dataset} not found")
            continue
        
        print(f"\n>>> 测试 {dataset}...")
        
        # 步骤 1: 首先运行 Python 策略网络推理，生成/覆盖属于当前 dataset 的初始解 attention_probs.csv
        update_heartbeat(task=f"推理: {dataset}")
        if not run_inference(dataset_path):
            print(f"  [WARN] 推理失败, 退回使用基线策略")
        
        for strategy in strategies:
            current_test += 1
            progress = f"{current_test}/{total_tests} ({100*current_test//total_tests}%)"
            update_heartbeat(
                task=f"{dataset} - {strategy}",
                progress=progress
            )
            
            # 步骤 2: 将 C++ 作为一个“单次任务执行器”进行调用，它会自动读取刚刚生成的当前 dataset 的结果文件
            cost = run_cpp_solver(dataset_path, strategy)
            
            if cost is not None:
                results.append({
                    "Dataset": dataset,
                    "Strategy": strategy,
                    "BestCost": cost,
                    "Alpha": ALPHA
                })
                print(f"  {strategy}: {cost:.2f}")
            else:
                print(f"  {strategy}: FAILED")
    
    # Save results
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        output_path = "../results/full_pipeline_results.csv"
        os.makedirs("../results", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Print summary
        print("\n" + "-" * 70)
        print("RESULTS SUMMARY")
        print("-" * 70)
        
        summary = df.groupby('Strategy')['BestCost'].agg(['mean', 'std', 'min', 'max'])
        print(summary.to_string())
        
        # Calculate improvement
        if 'full' in df['Strategy'].values and 'baseline' in df['Strategy'].values:
            full_mean = df[df['Strategy'] == 'full']['BestCost'].mean()
            baseline_mean = df[df['Strategy'] == 'baseline']['BestCost'].mean()
            improvement = (baseline_mean - full_mean) / baseline_mean * 100
            print(f"\nFull vs Baseline Improvement: {improvement:.2f}%")
    
    return results

# ========================================
# Main Pipeline
# ========================================

def main():
    print("=" * 70)
    print("SVRAP FULL PIPELINE - TRAINING & TESTING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Heartbeat interval: {HEARTBEAT_INTERVAL // 60} minutes")
    print("=" * 70)
    
    heartbeat_info["start_time"] = datetime.now()
    
    # Start heartbeat monitor thread
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()
    
    try:
        # Phase 1: Train models
        successful, failed = phase1_train_all_models()
        
        if successful == 0:
            print("\n[ERROR] No models trained successfully. Aborting.")
            return
        
        # Phase 2: Run tests
        results = phase2_run_tests()
        
        # Final summary
        elapsed = datetime.now() - heartbeat_info["start_time"]
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Models trained: {successful}/{len(DATASETS)}")
        print(f"Test results: {len(results)} entries")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline stopped by user.")
    finally:
        heartbeat_info["running"] = False

if __name__ == "__main__":
    main()
