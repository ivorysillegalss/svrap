import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
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


def run_cmd(cmd: Sequence[str], cwd: Path, timeout: int) -> Tuple[int, str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", (time.time() - t0)
    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]", (time.time() - t0)


def mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [v for v in values if v == v]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def parse_intervals(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            val = int(s)
        except ValueError:
            continue
        if val >= 1:
            out.append(val)
    return sorted(set(out))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"

    parser = argparse.ArgumentParser(
        description=(
            "Sweep sparse counterfactual schedule on large datasets. "
            "Node loss is computed every N epochs; other epochs use pure REINFORCE."
        )
    )
    parser.add_argument("--python", default="python", help="Python executable")
    parser.add_argument(
        "--runner",
        default=str(scripts_dir / "run_u159_sparse_node_loss_test.py"),
        help="Path to sparse counterfactual training runner",
    )
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ executable")
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "main_dataset"),
        help="Dataset directory containing d493.txt and rat783.txt",
    )
    parser.add_argument(
        "--datasets",
        default="d493,rat783",
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--node-loss-intervals",
        default="10,20",
        help="Comma-separated node-loss intervals (e.g. 10,20)",
    )
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs per run")
    parser.add_argument("--runs", type=int, default=1, help="Runs per (dataset, interval)")
    parser.add_argument("--seed-base", type=int, default=42, help="Effective seed = seed_base + run")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA for C++ solver")
    parser.add_argument(
        "--variant-base",
        default="cf_sparse_large",
        help="Variant base tag. Effective tag includes dataset/interval/run",
    )
    parser.add_argument(
        "--raw-out",
        default=str(repo_root / "results" / "large_counterfactual_sparse_raw.csv"),
        help="Raw output CSV path",
    )
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "large_counterfactual_sparse_summary.csv"),
        help="Summary output CSV path",
    )
    parser.add_argument("--train-timeout", type=int, default=7200, help="Timeout seconds for one training run")
    parser.add_argument("--solve-timeout", type=int, default=1800, help="Timeout seconds for one C++ solve")

    args = parser.parse_args()

    runner = Path(args.runner)
    exe_path = Path(args.exe)
    dataset_dir = Path(args.dataset_dir)
    raw_out = Path(args.raw_out)
    summary_out = Path(args.summary_out)

    if not runner.exists():
        print(f"[ERROR] Runner not found: {runner}", file=sys.stderr)
        return 1
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    dataset_names = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    if not dataset_names:
        print("[ERROR] --datasets is empty", file=sys.stderr)
        return 1

    intervals = parse_intervals(str(args.node_loss_intervals))
    if not intervals:
        print("[ERROR] --node-loss-intervals has no valid integer >= 1", file=sys.stderr)
        return 1

    dataset_paths: List[Tuple[str, Path]] = []
    for name in dataset_names:
        p = dataset_dir / f"{name}.txt"
        if not p.exists():
            print(f"[ERROR] Dataset file not found: {p}", file=sys.stderr)
            return 1
        dataset_paths.append((name, p))

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    raw_fields = [
        "dataset",
        "node_loss_interval",
        "run",
        "seed",
        "variant_tag",
        "epochs",
        "alpha",
        "train_return_code",
        "train_elapsed_sec",
        "full_return_code",
        "no_nn_return_code",
        "full_cost",
        "no_nn_cost",
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()

    total_jobs = len(dataset_paths) * len(intervals) * args.runs
    current_job = 0
    all_rows: List[Dict[str, Any]] = []

    for dataset_name, dataset_path in dataset_paths:
        for interval in intervals:
            for run_idx in range(1, args.runs + 1):
                current_job += 1
                seed = args.seed_base + run_idx
                variant_tag = f"{args.variant_base}_{dataset_name}_i{interval}_r{run_idx}"
                print(
                    f"[{current_job}/{total_jobs}] dataset={dataset_name} "
                    f"interval={interval} run={run_idx} seed={seed}"
                )

                train_cmd = [
                    args.python,
                    str(runner),
                    "--train",
                    "--dataset",
                    str(dataset_path),
                    "--epochs",
                    str(args.epochs),
                    "--node-loss-interval",
                    str(interval),
                    "--variant-tag",
                    variant_tag,
                    "--seed",
                    str(seed),
                    "--alpha",
                    str(args.alpha),
                ]
                code_train, _, train_elapsed = run_cmd(train_cmd, cwd=scripts_dir, timeout=args.train_timeout)

                full_cmd = [str(exe_path), str(args.alpha), str(dataset_path), "full"]
                code_full, out_full, _ = run_cmd(full_cmd, cwd=repo_root, timeout=args.solve_timeout)
                full_cost = parse_best_cost(out_full)

                no_nn_cmd = [str(exe_path), str(args.alpha), str(dataset_path), "no_nn"]
                code_no_nn, out_no_nn, _ = run_cmd(no_nn_cmd, cwd=repo_root, timeout=args.solve_timeout)
                no_nn_cost = parse_best_cost(out_no_nn)

                row = {
                    "dataset": dataset_name,
                    "node_loss_interval": interval,
                    "run": run_idx,
                    "seed": seed,
                    "variant_tag": variant_tag,
                    "epochs": args.epochs,
                    "alpha": args.alpha,
                    "train_return_code": code_train,
                    "train_elapsed_sec": train_elapsed,
                    "full_return_code": code_full,
                    "no_nn_return_code": code_no_nn,
                    "full_cost": full_cost,
                    "no_nn_cost": no_nn_cost,
                }
                all_rows.append(row)

                with raw_out.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=raw_fields)
                    writer.writerow(row)

    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    for row in all_rows:
        key = (str(row["dataset"]), int(row["node_loss_interval"]))
        grouped.setdefault(key, {"full": [], "no_nn": []})
        grouped[key]["full"].append(float(row["full_cost"]))
        grouped[key]["no_nn"].append(float(row["no_nn_cost"]))

    summary_fields = [
        "dataset",
        "node_loss_interval",
        "runs",
        "epochs",
        "alpha",
        "full_mean",
        "full_std",
        "no_nn_mean",
        "no_nn_std",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for (dataset_name, interval), values in sorted(grouped.items()):
        full_mean, full_std = mean_std(values["full"])
        no_nn_mean, no_nn_std = mean_std(values["no_nn"])
        summary_rows.append(
            {
                "dataset": dataset_name,
                "node_loss_interval": interval,
                "runs": args.runs,
                "epochs": args.epochs,
                "alpha": args.alpha,
                "full_mean": full_mean,
                "full_std": full_std,
                "no_nn_mean": no_nn_mean,
                "no_nn_std": no_nn_std,
            }
        )

    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Raw: {raw_out}")
    print(f"Done. Summary: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())