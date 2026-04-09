import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


def run_cmd(cmd: List[str], timeout: int) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as e:
        stdout_obj = e.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        out = stdout_text + "\n[TIMEOUT]"
        return 124, out


def build_solver_cmd(
    python_exe: str,
    solver_py: Path,
    dataset_path: Path,
    variant_tag: str,
    seed: int,
    epochs: int,
    reinforce_only_dataset: str,
) -> List[str]:
    cmd = [
        python_exe,
        str(solver_py),
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
    if reinforce_only_dataset:
        cmd.extend(["--reinforce-only-datasets", reinforce_only_dataset])
    return cmd


def build_cpp_cmd(exe_path: Path, alpha: float, dataset_path: Path, strategy: str) -> List[str]:
    return [str(exe_path), str(alpha), str(dataset_path), strategy]


def mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [v for v in values if v == v]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def parse_dataset_name_set(raw: str) -> set:
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Run each dataset for N times, collect full/no_nn best costs, "
            "and compute mean/std per dataset and strategy."
        )
    )
    parser.add_argument("--python", default="python", help="Python executable for svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ executable")
    parser.add_argument("--dataset-dir", default=str(repo_root / "formatted_dataset"), help="Dataset directory")
    parser.add_argument("--runs", type=int, default=10, help="Runs per dataset")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA for C++ solver")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs per run")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed. Effective seed = seed_base + run_index",
    )
    parser.add_argument(
        "--raw-out",
        default=str(repo_root / "results" / "full_no_nn_10runs_raw.csv"),
        help="Raw per-run output CSV",
    )
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "full_no_nn_10runs_summary.csv"),
        help="Summary output CSV",
    )
    parser.add_argument("--train-timeout", type=int, default=7200, help="Timeout seconds for one training run")
    parser.add_argument("--solve-timeout", type=int, default=1800, help="Timeout seconds for one C++ solve")
    parser.add_argument(
        "--reinforce-only-datasets",
        default="",
        help="Comma-separated dataset names that should use pure REINFORCE (default: none)",
    )

    args = parser.parse_args()

    solver_py = Path(args.solver)
    exe_path = Path(args.exe)
    dataset_dir = Path(args.dataset_dir)
    raw_out = Path(args.raw_out)
    summary_out = Path(args.summary_out)
    reinforce_only_set = parse_dataset_name_set(args.reinforce_only_datasets)

    if not solver_py.exists():
        print(f"[ERROR] Solver not found: {solver_py}", file=sys.stderr)
        return 1
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    datasets = sorted(p for p in dataset_dir.glob("*.txt"))
    if not datasets:
        print(f"[ERROR] No dataset files in {dataset_dir}", file=sys.stderr)
        return 1

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    # 先初始化 Raw CSV 文件，写入表头
    fieldnames_raw = [
        "dataset", "run", "seed", "reinforce_only", 
        "train_return_code", "full_return_code", "no_nn_return_code", 
        "full_cost", "no_nn_cost"
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writeheader()

    rows: List[Dict[str, Any]] = []
    start = time.time()

    total_jobs = len(datasets) * args.runs
    current_job = 0

    print(f"Datasets: {len(datasets)}, runs per dataset: {args.runs}, total runs: {total_jobs}")

    for dataset_path in datasets:
        dataset_name = dataset_path.stem

        reinforce_only_dataset = dataset_name if dataset_name in reinforce_only_set else ""

        for run_idx in range(1, args.runs + 1):
            current_job += 1
            seed = args.seed_base + run_idx
            variant_tag = f"stats_{dataset_name}_r{run_idx}"

            print(
                f"[{current_job}/{total_jobs}] {dataset_name} run={run_idx} "
                f"seed={seed} reinforce_only={bool(reinforce_only_dataset)}"
            )

            solver_cmd = build_solver_cmd(
                python_exe=args.python,
                solver_py=solver_py,
                dataset_path=dataset_path,
                variant_tag=variant_tag,
                seed=seed,
                epochs=args.epochs,
                reinforce_only_dataset=reinforce_only_dataset,
            )
            code_train, out_train = run_cmd(solver_cmd, timeout=args.train_timeout)
            if code_train != 0:
                print(f"  [WARN] Training/inference failed for {dataset_name} run={run_idx} (code={code_train})")

            full_cmd = build_cpp_cmd(exe_path, args.alpha, dataset_path, "full")
            code_full, out_full = run_cmd(full_cmd, timeout=args.solve_timeout)
            full_cost = parse_best_cost(out_full)

            no_nn_cmd = build_cpp_cmd(exe_path, args.alpha, dataset_path, "no_nn")
            code_no_nn, out_no_nn = run_cmd(no_nn_cmd, timeout=args.solve_timeout)
            no_nn_cost = parse_best_cost(out_no_nn)


            # 在执行完 code_no_nn 之后，构建这一行的数据
            current_row = {
                "dataset": dataset_name,
                "run": run_idx,
                "seed": seed,
                "reinforce_only": int(bool(reinforce_only_dataset)),
                "train_return_code": code_train,
                "full_return_code": code_full,
                "no_nn_return_code": code_no_nn,
                "full_cost": full_cost,
                "no_nn_cost": no_nn_cost,
            }
            rows.append(current_row)

            # --- 修改部分：立即追加写入 Raw CSV ---
            with raw_out.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
                writer.writerow(current_row)
            

    # Aggregate summary
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        ds = str(r["dataset"])
        grouped.setdefault(ds, {"full": [], "no_nn": []})
        grouped[ds]["full"].append(float(r["full_cost"]))
        grouped[ds]["no_nn"].append(float(r["no_nn_cost"]))

    summary_rows = []
    for ds in sorted(grouped.keys()):
        full_mean, full_std = mean_std(grouped[ds]["full"])
        no_nn_mean, no_nn_std = mean_std(grouped[ds]["no_nn"])
        summary_rows.append(
            {
                "dataset": ds,
                "runs": args.runs,
                "full_mean": full_mean,
                "full_std": full_std,
                "no_nn_mean": no_nn_mean,
                "no_nn_std": no_nn_std,
            }
        )

    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "runs",
                "full_mean",
                "full_std",
                "no_nn_mean",
                "no_nn_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    print(f"Raw results: {raw_out}")
    print(f"Summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
