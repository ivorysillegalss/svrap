# -*- coding: utf-8 -*-
"""Batch runner for all datasets in formatted_dataset/.

For each dataset file (x,y per line):
  1) Train + infer the SVRAP policy network via run_pipeline(...).
  2) Copy the dataset into dataset.txt for the C++ solver.
  3) Run app.exe and measure its runtime, parsing the reported best cost.
  4) Append a row with all metrics to experiment_results.csv.

Run this from the project root:
  python run_all_datasets.py
"""

import os
import glob
import time
import csv
import shutil
import subprocess
import threading
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Dict, Any

from svrap_solver import run_pipeline


def ensure_cpp_built(project_root: str) -> str:
    """Ensure C++ binary exists; build with `make` if needed. Returns exe path."""
    exe_name = "app.exe" if os.name == "nt" else "app"
    exe_path = os.path.join(project_root, exe_name)
    if os.path.exists(exe_path):
        return exe_path

    print("[C++] app binary not found, invoking `make` to build...")
    result = subprocess.run(["make"], cwd=project_root)
    if result.returncode != 0 or not os.path.exists(exe_path):
        raise RuntimeError("Failed to build C++ solver (make/app.exe)")
    return exe_path


def run_cpp_solver(exe_path: str, cwd: str, log_file) -> Dict[str, Any]:
    """Run app.exe and measure wall-clock time; parse best cost from stdout.

    All stdout/stderr from the C++ solver are appended to the given log_file.
    """
    t0 = time.time()
    proc = subprocess.run(
        [exe_path],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    t1 = time.time()
    cpp_time = t1 - t0

    # 追加写入 C++ 日志
    log_file.write("\n[C++] run at " + datetime.now().isoformat(timespec="seconds") + "\n")
    log_file.write(proc.stdout)
    if proc.stderr:
        log_file.write("\n[stderr]\n" + proc.stderr + "\n")
    log_file.flush()

    if proc.returncode != 0:
        log_file.write(f"[C++] Solver exited with non-zero code: {proc.returncode}\n")

    best_cost_cpp = None
    for line in proc.stdout.splitlines()[::-1]:
        if "Best cost:" in line:
            try:
                # Expected format: "Best cost: <value>"
                parts = line.split(":", 1)
                best_cost_cpp = float(parts[1].strip())
            except Exception:
                pass
            break

    return {
        "cpp_time_sec": cpp_time,
        "cpp_best_cost": best_cost_cpp,
    }


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "formatted_dataset")
    results_csv = os.path.join(project_root, "experiment_results.csv")
    batch_log = os.path.join(project_root, "run_all_datasets.log")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"formatted_dataset directory not found: {dataset_dir}")

    exe_path = ensure_cpp_built(project_root)

    dataset_files = sorted(glob.glob(os.path.join(dataset_dir, "*.txt")))
    if not dataset_files:
        raise FileNotFoundError(f"No .txt datasets found in {dataset_dir}")

    print(f"Found {len(dataset_files)} datasets under formatted_dataset/.")

    # Prepare CSV
    header = [
        "dataset_name",
        "n_nodes",
        "py_best_cost",
        "py_assign_ratio",
        "py_route_ratio",
        "py_loss_ratio",
        "py_assign_count",
        "py_route_count",
        "py_loss_count",
        "py_train_time_sec",
        "py_train_epochs_run",
        "py_best_found_epoch",
        "py_best_found_time_from_start_sec",
        "py_inference_time_sec",
        "py_greedy_backbone_size",
        "py_greedy_cost",
        "cpp_best_cost",
        "cpp_time_sec",
    ]

    write_header = not os.path.exists(results_csv)

    with open(results_csv, "a", newline="", encoding="utf-8") as result_f, \
         open(batch_log, "a", encoding="utf-8") as log_f:

        writer = csv.writer(result_f)
        if write_header:
            writer.writerow(header)

        total_datasets = len(dataset_files)

        for idx, ds_path in enumerate(dataset_files, start=1):
            ds_name = os.path.basename(ds_path)

            # 控制台只简要输出当前数据集及索引
            print(f"[DATASET {idx}/{total_datasets}] {ds_name} - started")

            # 1) Copy dataset to dataset.txt for C++
            target_dataset_txt = os.path.join(project_root, "dataset.txt")
            shutil.copyfile(ds_path, target_dataset_txt)

            # 日志头
            log_f.write("\n" + "=" * 70 + "\n")
            log_f.write(f"[DATASET] {ds_name} (index {idx}/{total_datasets})\n")
            log_f.write("Started at: " + datetime.now().isoformat(timespec="seconds") + "\n")
            log_f.flush()

            metrics: Dict[str, Any] = {}
            cpp_metrics: Dict[str, Any] = {}
            worker_error: Dict[str, Any] = {"error": None}

            def worker() -> None:
                try:
                    # Python 训练与推理日志全部重定向到批量日志文件
                    with redirect_stdout(log_f), redirect_stderr(log_f):
                        nonlocal metrics, cpp_metrics
                        metrics = run_pipeline(train_model=True, dataset_path=ds_path)
                        cpp_metrics = run_cpp_solver(exe_path=exe_path, cwd=project_root, log_file=log_f)
                except Exception as e:  # noqa: BLE001
                    worker_error["error"] = e

            start_time = time.time()
            last_heartbeat = start_time
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            # 每隔 5 分钟输出一次保活信息
            heartbeat_interval = 300.0
            while thread.is_alive():
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed = now - start_time
                    msg = (
                        f"[HEARTBEAT] dataset {idx}/{total_datasets} ({ds_name}) "
                        f"running for {elapsed/60.0:.1f} min"
                    )
                    print(msg)
                    log_f.write(msg + "\n")
                    log_f.flush()
                    last_heartbeat = now
                time.sleep(5.0)

            thread.join()

            if worker_error["error"] is not None:
                raise worker_error["error"]

            end_time = time.time()
            total_elapsed = end_time - start_time

            row = [
                metrics.get("dataset_name"),
                metrics.get("n_nodes"),
                metrics.get("best_cost"),
                metrics.get("assign_ratio"),
                metrics.get("route_ratio"),
                metrics.get("loss_ratio"),
                metrics.get("assign_count"),
                metrics.get("route_count"),
                metrics.get("loss_count"),
                metrics.get("train_time_sec"),
                metrics.get("train_epochs_run"),
                metrics.get("best_found_epoch"),
                metrics.get("best_found_time_from_start_sec"),
                metrics.get("inference_time_sec"),
                metrics.get("greedy_backbone_size"),
                metrics.get("greedy_cost"),
                cpp_metrics.get("cpp_best_cost"),
                cpp_metrics.get("cpp_time_sec"),
            ]

            writer.writerow(row)
            result_f.flush()

            py_best = metrics.get("best_cost")
            cpp_best = cpp_metrics.get("cpp_best_cost")
            print(
                f"[DONE] {ds_name} | py_best={py_best:.4f} "
                f"cpp_best={cpp_best:.4f} | total_time={total_elapsed/60.0:.1f} min"
            )



if __name__ == "__main__":
    main()
