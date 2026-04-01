import argparse
import csv
import math
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


def run_cmd(cmd: Sequence[str], timeout: int) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]"


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


def parse_int_or_default(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_float_or_nan(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def is_valid_cost_pair(full_cost: Any, no_nn_cost: Any) -> bool:
    full_val = parse_float_or_nan(full_cost)
    no_nn_val = parse_float_or_nan(no_nn_cost)
    return (not math.isnan(full_val)) and (not math.isnan(no_nn_val))


def load_existing_valid_rows(
    dataset_raw_out: Path,
    dataset_name: str,
    runs: int,
    seed_base: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    if not dataset_raw_out.exists():
        missing_runs = list(range(1, runs + 1))
        return [], missing_runs

    by_run: Dict[int, Dict[str, Any]] = {}
    with dataset_raw_out.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_idx = parse_int_or_default(row.get("run"), -1)
            if run_idx < 1 or run_idx > runs:
                continue
            if run_idx in by_run:
                # Keep first valid record for each run index.
                continue
            if not is_valid_cost_pair(row.get("full_cost"), row.get("no_nn_cost")):
                continue

            by_run[run_idx] = {
                "dataset": str(row.get("dataset") or dataset_name),
                "run": run_idx,
                "seed": parse_int_or_default(row.get("seed"), seed_base + run_idx),
                "reinforce_only": parse_int_or_default(row.get("reinforce_only"), 0),
                "train_return_code": parse_int_or_default(row.get("train_return_code"), 0),
                "full_return_code": parse_int_or_default(row.get("full_return_code"), 0),
                "no_nn_return_code": parse_int_or_default(row.get("no_nn_return_code"), 0),
                "full_cost": parse_float_or_nan(row.get("full_cost")),
                "no_nn_cost": parse_float_or_nan(row.get("no_nn_cost")),
            }

    existing_rows = [by_run[i] for i in sorted(by_run.keys())]
    missing_runs = [i for i in range(1, runs + 1) if i not in by_run]
    return existing_rows, missing_runs


def is_reinforce_only_dataset(dataset_name: str) -> bool:
    # Keep the same rule as existing script (d493 / rat783).
    return ("493" in dataset_name) or ("783" in dataset_name)


def run_dataset_jobs(
    dataset_path: Path,
    runs: int,
    seed_base: int,
    epochs: int,
    python_exe: str,
    solver_py: Path,
    exe_path: Path,
    alpha: float,
    train_timeout: int,
    solve_timeout: int,
    per_dataset_out_dir: Path,
) -> List[Dict[str, Any]]:
    dataset_name = dataset_path.stem
    reinforce_only_dataset = dataset_name if is_reinforce_only_dataset(dataset_name) else ""

    fieldnames_raw = [
        "dataset",
        "run",
        "seed",
        "reinforce_only",
        "train_return_code",
        "full_return_code",
        "no_nn_return_code",
        "full_cost",
        "no_nn_cost",
    ]
    dataset_raw_out = per_dataset_out_dir / f"{dataset_name}.csv"
    existing_rows, missing_runs = load_existing_valid_rows(
        dataset_raw_out=dataset_raw_out,
        dataset_name=dataset_name,
        runs=runs,
        seed_base=seed_base,
    )

    if not missing_runs:
        print(f"[SKIP {dataset_name}] already has {runs} valid runs")
        return existing_rows

    # Rewrite per-dataset CSV with only valid rows, removing NaN/incomplete rows.
    with dataset_raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writeheader()
        if existing_rows:
            writer.writerows(existing_rows)

    print(
        f"[RESUME {dataset_name}] valid={len(existing_rows)}, missing={len(missing_runs)} "
        f"target_runs={runs}"
    )

    rows: List[Dict[str, Any]] = list(existing_rows)
    for run_idx in missing_runs:
        seed = seed_base + run_idx
        variant_tag = f"stats_parallel_{dataset_name}_r{run_idx}"

        print(
            f"[DATASET {dataset_name}] run={run_idx}/{runs} "
            f"seed={seed} reinforce_only={bool(reinforce_only_dataset)}"
        )

        solver_cmd = build_solver_cmd(
            python_exe=python_exe,
            solver_py=solver_py,
            dataset_path=dataset_path,
            variant_tag=variant_tag,
            seed=seed,
            epochs=epochs,
            reinforce_only_dataset=reinforce_only_dataset,
        )
        code_train, _ = run_cmd(solver_cmd, timeout=train_timeout)
        if code_train != 0:
            print(
                f"  [WARN] Training/inference failed for {dataset_name} "
                f"run={run_idx} (code={code_train})"
            )

        full_cmd = build_cpp_cmd(exe_path, alpha, dataset_path, "full")
        code_full, out_full = run_cmd(full_cmd, timeout=solve_timeout)
        full_cost = parse_best_cost(out_full)

        no_nn_cmd = build_cpp_cmd(exe_path, alpha, dataset_path, "no_nn")
        code_no_nn, out_no_nn = run_cmd(no_nn_cmd, timeout=solve_timeout)
        no_nn_cost = parse_best_cost(out_no_nn)

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

        # Flush one row immediately after each run finishes.
        with dataset_raw_out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
            writer.writerow(current_row)

    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Parallel runner by dataset for full/no_nn stats. "
            "Uses main_dataset by default and skips berlin52."
        )
    )
    parser.add_argument("--python", default="python", help="Python executable for svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ executable")
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "main_dataset"),
        help="Dataset directory (default: main_dataset)",
    )
    parser.add_argument(
        "--skip-datasets",
        default="berlin52",
        help="Comma-separated dataset names to skip (default: berlin52)",
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per dataset")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA for C++ solver")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs per run")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed. Effective seed = seed_base + run_index",
    )
    parser.add_argument(
        "--max-parallel-datasets",
        type=int,
        default=3,
        help="Maximum number of datasets to run in parallel",
    )
    parser.add_argument(
        "--raw-out",
        default=str(repo_root / "results" / "main_dataset_no_berlin_parallel_raw.csv"),
        help="Raw per-run output CSV",
    )
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "main_dataset_no_berlin_parallel_summary.csv"),
        help="Summary output CSV",
    )
    parser.add_argument(
        "--per-dataset-out-dir",
        default=str(repo_root / "results" / "main_dataset_no_berlin_parallel_raw_by_dataset"),
        help="Directory to store one raw CSV per dataset",
    )
    parser.add_argument("--train-timeout", type=int, default=7200, help="Timeout seconds for one training run")
    parser.add_argument("--solve-timeout", type=int, default=1800, help="Timeout seconds for one C++ solve")

    args = parser.parse_args()

    solver_py = Path(args.solver)
    exe_path = Path(args.exe)
    dataset_dir = Path(args.dataset_dir)
    raw_out = Path(args.raw_out)
    summary_out = Path(args.summary_out)
    per_dataset_out_dir = Path(args.per_dataset_out_dir)

    if not solver_py.exists():
        print(f"[ERROR] Solver not found: {solver_py}", file=sys.stderr)
        return 1
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1
    if args.max_parallel_datasets < 1:
        print("[ERROR] --max-parallel-datasets must be >= 1", file=sys.stderr)
        return 1

    skip_set = {x.strip() for x in str(args.skip_datasets).split(",") if x.strip()}
    all_datasets = sorted(p for p in dataset_dir.glob("*.txt"))
    datasets = [p for p in all_datasets if p.stem not in skip_set]

    if not datasets:
        print(f"[ERROR] No dataset files to run in {dataset_dir} after filtering", file=sys.stderr)
        return 1

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    per_dataset_out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames_raw = [
        "dataset",
        "run",
        "seed",
        "reinforce_only",
        "train_return_code",
        "full_return_code",
        "no_nn_return_code",
        "full_cost",
        "no_nn_cost",
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writeheader()

    start = time.time()
    print(
        f"Datasets selected: {len(datasets)} / {len(all_datasets)} (skip={sorted(skip_set)}), "
        f"runs per dataset: {args.runs}, max parallel datasets: {args.max_parallel_datasets}"
    )
    print(f"Per-dataset raw CSV dir: {per_dataset_out_dir}")

    all_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel_datasets) as executor:
        future_to_dataset = {
            executor.submit(
                run_dataset_jobs,
                dataset_path,
                args.runs,
                args.seed_base,
                args.epochs,
                args.python,
                solver_py,
                exe_path,
                args.alpha,
                args.train_timeout,
                args.solve_timeout,
                per_dataset_out_dir,
            ): dataset_path.stem
            for dataset_path in datasets
        }

        finished = 0
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                dataset_rows = future.result()
            except Exception as exc:
                print(f"[ERROR] Dataset worker failed for {dataset_name}: {exc}", file=sys.stderr)
                dataset_rows = []

            all_rows.extend(dataset_rows)
            finished += 1

            print(f"[DONE {finished}/{len(datasets)}] {dataset_name}, rows={len(dataset_rows)}")

    # Keep a combined raw file for convenience in downstream analysis.
    with raw_out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writerows(all_rows)

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in all_rows:
        ds = str(row["dataset"])
        grouped.setdefault(ds, {"full": [], "no_nn": []})
        grouped[ds]["full"].append(float(row["full_cost"]))
        grouped[ds]["no_nn"].append(float(row["no_nn_cost"]))

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
    print(f"Per-dataset raw dir: {per_dataset_out_dir}")
    print(f"Raw results: {raw_out}")
    print(f"Summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())