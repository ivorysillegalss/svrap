import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


BEST_COST_PATTERN = re.compile(r"Best Cost:\s*([0-9eE+\-.]+)")


def parse_best_cost(output: str) -> Optional[float]:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def run_training(
    python_exe: str,
    solver_path: Path,
    dataset_path: Path,
    variant_tag: str,
    seed: int,
    epochs: int,
    repo_root: Path,
) -> Dict[str, object]:
    cmd = [
        python_exe,
        str(solver_path),
        "--dataset",
        str(dataset_path),
        "--train",
        "--variant-tag",
        variant_tag,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
    ]

    start_time = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start_time
    output = proc.stdout or ""

    dataset_name = dataset_path.stem
    history_name = f"training_log_{dataset_name}_{variant_tag}.csv"
    history_path = repo_root / history_name

    return {
        "return_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "best_cost": parse_best_cost(output),
        "stdout": output,
        "history_path": history_path,
        "history_exists": history_path.exists(),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run berlin52 / pr152 / d198 10 times each and collect convergence logs."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["berlin52", "pr152", "d198"],
        help="Dataset names to test (default: berlin52 pr152 d198)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "main_dataset"),
        help="Directory containing the dataset files",
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per dataset")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs for each run")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed; actual seed = seed_base + run_index",
    )
    parser.add_argument(
        "--variant-prefix",
        default="conv",
        help="Prefix used in the variant tag for each run",
    )
    parser.add_argument(
        "--raw-out",
        default=str(repo_root / "results" / "convergence_curve_test_raw.csv"),
        help="Raw per-run CSV output",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(repo_root / "results" / "convergence_curve_logs"),
        help="Directory where per-run training logs are copied",
    )
    args = parser.parse_args()

    solver_path = Path(args.solver)
    dataset_dir = Path(args.dataset_dir)
    raw_out = Path(args.raw_out)
    logs_dir = Path(args.logs_dir)

    if not solver_path.exists():
        print(f"[ERROR] Solver not found: {solver_path}")
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}")
        return 1

    dataset_paths: List[Path] = []
    for dataset_name in args.datasets:
        dataset_path = dataset_dir / f"{dataset_name}.txt"
        if dataset_path.exists():
            dataset_paths.append(dataset_path)
        else:
            print(f"[WARN] Dataset not found, skipping: {dataset_path}")

    if not dataset_paths:
        print("[ERROR] No valid datasets to run")
        return 1

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "run",
        "seed",
        "variant_tag",
        "return_code",
        "elapsed_seconds",
        "best_cost",
        "history_path",
        "copied_history_path",
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_jobs = len(dataset_paths) * args.runs
    job_index = 0

    print(f"Datasets: {', '.join(p.stem for p in dataset_paths)}")
    print(f"Runs per dataset: {args.runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Raw CSV: {raw_out}")
    print(f"Logs dir: {logs_dir}")

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.stem
        dataset_log_dir = logs_dir / dataset_name
        dataset_log_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(1, args.runs + 1):
            job_index += 1
            seed = args.seed_base + run_idx - 1
            variant_tag = f"{args.variant_prefix}_r{run_idx:02d}"

            print(
                f"[{job_index}/{total_jobs}] {dataset_name} run={run_idx:02d} "
                f"seed={seed} tag={variant_tag}"
            )

            result = run_training(
                python_exe=args.python,
                solver_path=solver_path,
                dataset_path=dataset_path,
                variant_tag=variant_tag,
                seed=seed,
                epochs=args.epochs,
                repo_root=repo_root,
            )

            copied_history_path: Optional[Path] = dataset_log_dir / f"run_{run_idx:02d}.csv"
            if result["history_exists"]:
                shutil.copy2(result["history_path"], copied_history_path)
            else:
                copied_history_path = None

            stdout_path = dataset_log_dir / f"run_{run_idx:02d}.stdout.txt"
            stdout_path.write_text(str(result["stdout"]), encoding="utf-8")

            row = {
                "dataset": dataset_name,
                "run": run_idx,
                "seed": seed,
                "variant_tag": variant_tag,
                "return_code": result["return_code"],
                "elapsed_seconds": round(float(result["elapsed_seconds"]), 6),
                "best_cost": "" if result["best_cost"] is None else result["best_cost"],
                "history_path": str(result["history_path"]),
                "copied_history_path": "" if copied_history_path is None else str(copied_history_path),
            }
            with raw_out.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            status = "ok" if int(result["return_code"]) == 0 else f"fail({result['return_code']})"
            print(
                f"  {status}, time={float(result['elapsed_seconds']):.1f}s, "
                f"best_cost={result['best_cost']}"
            )

    print(f"Done. Raw results written to {raw_out}")
    print(f"Per-run logs copied to {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())