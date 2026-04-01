import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


def run_cmd(cmd: List[str], timeout: int, cwd: Path) -> Tuple[int, str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Run strict paper_baseline on large datasets (d493, rat783) for 10 runs, "
            "and export raw + summary CSV outputs."
        )
    )
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ executable")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA parameter")
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "main_dataset"),
        help="Dataset directory containing d493.txt and rat783.txt",
    )
    parser.add_argument(
        "--datasets",
        default="d493,rat783",
        help="Comma-separated dataset names to run",
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per dataset")
    parser.add_argument(
        "--raw-out",
        default=str(repo_root / "results" / "paper_baseline_large_10runs_raw.csv"),
        help="Raw output CSV path",
    )
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "paper_baseline_large_10runs_summary.csv"),
        help="Summary output CSV path",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout seconds per C++ run")

    args = parser.parse_args()

    exe_path = Path(args.exe)
    dataset_dir = Path(args.dataset_dir)
    raw_out = Path(args.raw_out)
    summary_out = Path(args.summary_out)

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

    dataset_paths = []
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
        "run",
        "strategy",
        "alpha",
        "return_code",
        "elapsed_sec",
        "best_cost",
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()

    all_rows: List[Dict[str, float]] = []
    total_jobs = len(dataset_paths) * args.runs
    done_jobs = 0

    for dataset_name, dataset_path in dataset_paths:
        for run_idx in range(1, args.runs + 1):
            done_jobs += 1
            print(f"[{done_jobs}/{total_jobs}] {dataset_name} run={run_idx} strategy=paper_baseline")

            cmd = [str(exe_path), str(args.alpha), str(dataset_path), "paper_baseline"]
            code, out, elapsed = run_cmd(cmd=cmd, timeout=args.timeout, cwd=repo_root)
            best_cost = parse_best_cost(out)

            row = {
                "dataset": dataset_name,
                "run": run_idx,
                "strategy": "paper_baseline",
                "alpha": args.alpha,
                "return_code": code,
                "elapsed_sec": elapsed,
                "best_cost": best_cost,
            }
            all_rows.append(row)

            with raw_out.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writerow(row)

            if code != 0:
                print(f"  [WARN] return_code={code}")

    grouped: Dict[str, List[float]] = {}
    for row in all_rows:
        ds = str(row["dataset"])
        grouped.setdefault(ds, [])
        grouped[ds].append(float(row["best_cost"]))

    summary_fields = ["dataset", "runs", "strategy", "alpha", "best_cost_mean", "best_cost_std"]
    summary_rows = []
    for ds in sorted(grouped.keys()):
        m, s = mean_std(grouped[ds])
        summary_rows.append(
            {
                "dataset": ds,
                "runs": args.runs,
                "strategy": "paper_baseline",
                "alpha": args.alpha,
                "best_cost_mean": m,
                "best_cost_std": s,
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