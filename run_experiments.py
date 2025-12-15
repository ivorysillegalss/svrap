import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def run_instance(exe_path: str, alpha: float, dataset_path: str) -> Tuple[float, float, str]:
    """Run svrap.exe on a single dataset with given alpha.

    Returns (best_cost, elapsed_seconds, raw_stdout).
    If parsing fails, best_cost is float('nan').
    """
    cmd = [exe_path, str(alpha), dataset_path]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as e:
        print(f"[ERROR] Failed to run {cmd}: {e}", file=sys.stderr)
        return float("nan"), 0.0, ""
    end = time.perf_counter()

    stdout = proc.stdout or ""
    elapsed = end - start

    # Parse last occurrence of "Best cost ... = value"
    best_cost = float("nan")
    for match in BEST_COST_PATTERN.finditer(stdout):
        try:
            best_cost = float(match.group(1))
        except ValueError:
            continue

    if proc.returncode != 0:
        print(f"[WARN] Process exited with code {proc.returncode} for {dataset_path}", file=sys.stderr)

    return best_cost, elapsed, stdout


def discover_datasets(dataset_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.lower().endswith(".txt"):
            files.append(os.path.join(dataset_dir, name))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SVRAP tabu search executable over formatted_dataset instances "
            "for given alpha values, collecting best cost and runtime."
        )
    )
    parser.add_argument(
        "--exe",
        default=os.path.join(os.path.dirname(__file__), "svrap.exe"),
        help="Path to compiled svrap executable (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(os.path.dirname(__file__), "formatted_dataset"),
        help="Directory containing dataset .txt files (default: %(default)s)",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[3.0, 5.0, 7.0, 9.0],
        help="List of alpha values to test (default: 3 5 7 9)",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="CSV file to write aggregated results (default: %(default)s)",
    )
    parser.add_argument(
        "--run-log",
        default="run.log",
        help="Log file to append per-run results during execution (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-log",
        default="summary.log",
        help="Text log file to write final aggregated summary (default: %(default)s)",
    )

    args = parser.parse_args()

    exe_path = os.path.abspath(args.exe)
    dataset_dir = os.path.abspath(args.dataset_dir)

    if not os.path.isfile(exe_path):
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    datasets = discover_datasets(dataset_dir)
    if not datasets:
        print(f"[ERROR] No .txt datasets found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Using executable: {exe_path}")
    print(f"Using dataset dir: {dataset_dir}")
    print(f"Datasets: {len(datasets)} files")
    print(f"Alphas: {', '.join(str(a) for a in args.alphas)}")

    # Prepare job list: all (alpha, dataset) combinations
    jobs = []  # type: List[Tuple[float, str]]
    for alpha in args.alphas:
        for ds_path in datasets:
            jobs.append((alpha, ds_path))

    # results[(dataset_name, alpha)] = (best_cost, elapsed)
    results: Dict[Tuple[str, float], Tuple[float, float]] = {}

    # Open run-log for appending; each line is written as runs complete
    run_log_path = os.path.abspath(args.run_log)
    print(f"Per-run log: {run_log_path}")
    with open(run_log_path, "a", encoding="utf-8") as run_log:
        run_log.write("\n=== New experiment session ===\n")
        run_log.write(
            f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        run_log.write(
            f"Executable: {exe_path}\nDataset dir: {dataset_dir}\nAlphas: {', '.join(str(a) for a in args.alphas)}\n"
        )
        total_jobs = len(jobs)
        completed_jobs = 0
        session_start = time.perf_counter()
        last_heartbeat = session_start

        for alpha, ds_path in jobs:
            ds_name = os.path.basename(ds_path)
            start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            best_cost, elapsed, _ = run_instance(exe_path, alpha, ds_path)

            # Console output
            if not (best_cost == best_cost):  # NaN check
                print(
                    f"[RUN] {start_wall} dataset={ds_name}, alpha={alpha} -> failed (no best cost parsed), time={elapsed:.3f}s"
                )
            else:
                print(
                    f"[RUN] {start_wall} dataset={ds_name}, alpha={alpha} -> best_cost={best_cost:.4f}, time={elapsed:.3f}s"
                )

            # Append to per-run log immediately
            best_str = "nan" if not (best_cost == best_cost) else f"{best_cost:.6f}"
            run_log.write(
                f"time={start_wall}, dataset={ds_name}, alpha={alpha}, best_cost={best_str}, time_seconds={elapsed:.6f}\n"
            )
            run_log.flush()

            results[(ds_name, alpha)] = (best_cost, elapsed)

            completed_jobs += 1

            # Heartbeat: every ~10 minutes, print a progress line
            now = time.perf_counter()
            if now - last_heartbeat >= 600.0:
                elapsed_total = now - session_start
                minutes = elapsed_total / 60.0
                hb_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[HEARTBEAT] {hb_wall} elapsed={minutes:.1f} min, completed {completed_jobs}/{total_jobs} runs"
                )
                run_log.write(
                    f"[HEARTBEAT] time={hb_wall}, elapsed_minutes={minutes:.1f}, completed={completed_jobs}/{total_jobs}\n"
                )
                run_log.flush()
                last_heartbeat = now

    # Write CSV summary
    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("dataset,alpha,best_cost,time_seconds\n")
        for (ds_name, alpha), (best_cost, elapsed) in sorted(results.items()):
            best_str = "" if not (best_cost == best_cost) else f"{best_cost:.6f}"
            f.write(f"{ds_name},{alpha},{best_str},{elapsed:.6f}\n")

    # Write human-readable summary log
    summary_path = os.path.abspath(args.summary_log)
    with open(summary_path, "w", encoding="utf-8") as fsum:
        fsum.write("SVRAP experiment summary\n")
        fsum.write(f"Executable: {exe_path}\n")
        fsum.write(f"Dataset dir: {dataset_dir}\n")
        fsum.write(f"Alphas: {', '.join(str(a) for a in args.alphas)}\n\n")
        for (ds_name, alpha), (best_cost, elapsed) in sorted(results.items()):
            best_str = "nan" if not (best_cost == best_cost) else f"{best_cost:.6f}"
            fsum.write(
                f"dataset={ds_name}, alpha={alpha}, best_cost={best_str}, time_seconds={elapsed:.6f}\n"
            )

    print(f"\nCSV summary written to {out_path}")
    print(f"Text summary written to {summary_path}")


if __name__ == "__main__":
    main()
