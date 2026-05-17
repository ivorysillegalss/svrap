import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


def run_cmd(cmd: List[str], cwd: Path, timeout: int) -> Tuple[int, str, float]:
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
        return proc.returncode, proc.stdout or "", time.time() - t0
    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]", time.time() - t0


def mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def format_float(value: float, digits: int = 2) -> str:
    if value != value:
        return "NaN"
    return f"{value:.{digits}f}"


def ensure_attention_probs(
    repo_root: Path,
    python_exe: str,
    solver_py: Path,
    dataset_path: Path,
    variant_tag: str,
    epochs: int,
    seed: int,
    timeout: int,
) -> None:
    cmd = [
        python_exe,
        str(solver_py),
        "--dataset",
        str(dataset_path),
        "--variant-tag",
        variant_tag,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
    ]
    code, output, elapsed = run_cmd(cmd, cwd=repo_root, timeout=timeout)
    if code != 0:
        raise RuntimeError(
            f"Failed to prepare attention_probs.csv for {dataset_path.name} (code={code}, elapsed={elapsed:.1f}s).\n"
            f"Output:\n{output}"
        )
    print(f"[PREP] attention_probs.csv ready for {dataset_path.stem} in {elapsed:.1f}s")


def run_solver_once(
    repo_root: Path,
    exe_path: Path,
    alpha: float,
    dataset_path: Path,
    strategy: str,
    timeout: int,
) -> Tuple[int, float, float, str]:
    cmd = [str(exe_path), str(alpha), str(dataset_path), strategy]
    code, output, elapsed = run_cmd(cmd, cwd=repo_root, timeout=timeout)
    best_cost = parse_best_cost(output)
    return code, elapsed, best_cost, output


def write_markdown_table(summary_rows: List[Dict[str, object]], out_md: Path) -> None:
    lines = [
        "| Strategy | KNN | Runs | Successful Runs | Mean Time (s) | Std Time (s) | Mean Best Cost | Std Best Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {strategy} | {knn} | {runs} | {valid_runs} | {mean_time} | {std_time} | {mean_cost} | {std_cost} |".format(
                strategy=row["Strategy"],
                knn=row["KNN"],
                runs=row["Runs"],
                valid_runs=row["Successful Runs"],
                mean_time=format_float(float(row["Mean Time (s)"]), 2),
                std_time=format_float(float(row["Std Time (s)"]), 2),
                mean_cost=format_float(float(row["Mean Best Cost"]), 2),
                std_cost=format_float(float(row["Std Best Cost"]), 2),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary_rows: List[Dict[str, object]], out_png: Path) -> None:
    labels = [str(row["Strategy"]) for row in summary_rows]
    mean_times = [float(row["Mean Time (s)"]) for row in summary_rows]
    std_times = [float(row["Std Time (s)"]) for row in summary_rows]
    mean_costs = [float(row["Mean Best Cost"]) for row in summary_rows]
    std_costs = [float(row["Std Best Cost"]) for row in summary_rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    colors = ["#0F766E", "#B45309"]
    x = range(len(labels))

    time_bars = axes[0].bar(x, mean_times, yerr=std_times, capsize=6, color=colors, edgecolor="#1f2937")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Solve Time (s)")
    axes[0].set_title("Mean Solve Time")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, value in zip(time_bars, mean_times):
        axes[0].annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    cost_bars = axes[1].bar(x, mean_costs, yerr=std_costs, capsize=6, color=colors, edgecolor="#1f2937")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("Best Cost")
    axes[1].set_title("Mean Best Cost")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, value in zip(cost_bars, mean_costs):
        axes[1].annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    fig.suptitle("pr152 KNN Comparison over 10 Runs", fontsize=15)
    fig.text(0.5, 0.01, "Error bars show one standard deviation", ha="center", fontsize=9)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Run pr152 10 times with KNN on/off, export a summary table, raw CSV, and a figure."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ solver executable")
    parser.add_argument("--dataset", default=str(repo_root / "main_dataset" / "pr152.txt"), help="Dataset path")
    parser.add_argument("--alpha", type=float, default=7.0, help="Alpha parameter for C++ solver")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per strategy")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Epochs used when svrap_solver.py needs to train a pr152 model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to svrap_solver.py when preparing attention_probs.csv",
    )
    parser.add_argument("--prep-timeout", type=int, default=7200, help="Timeout seconds for model prep")
    parser.add_argument("--solve-timeout", type=int, default=3600, help="Timeout seconds for each C++ solve")
    parser.add_argument(
        "--variant-tag",
        default="pr152_knn_comparison",
        help="Variant tag used for model/log isolation during preparation",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(repo_root / "results" / "pr152_knn_comparison"),
        help="Output prefix for CSV/MD/PNG files",
    )

    args = parser.parse_args()

    solver_py = Path(args.solver)
    exe_path = Path(args.exe)
    dataset_path = Path(args.dataset)
    out_prefix = Path(args.out_prefix)

    if not solver_py.exists():
        print(f"[ERROR] Solver not found: {solver_py}", file=sys.stderr)
        return 1
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_prefix.with_name(out_prefix.name + "_raw.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary_md = out_prefix.with_name(out_prefix.name + "_summary.md")
    plot_png = out_prefix.with_name(out_prefix.name + "_summary.png")

    ensure_attention_probs(
        repo_root=repo_root,
        python_exe=args.python,
        solver_py=solver_py,
        dataset_path=dataset_path,
        variant_tag=args.variant_tag,
        epochs=args.epochs,
        seed=args.seed,
        timeout=args.prep_timeout,
    )

    strategy_specs = [
        ("full", "KNN On", 1),
        ("no_knn", "KNN Off", 0),
    ]

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    raw_fields = [
        "dataset",
        "run",
        "strategy",
        "knn_enabled",
        "return_code",
        "elapsed_sec",
        "best_cost",
    ]
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()

    total_jobs = len(strategy_specs) * args.runs
    job_idx = 0
    for strategy, label, knn_enabled in strategy_specs:
        for run_idx in range(1, args.runs + 1):
            job_idx += 1
            print(
                f"[{job_idx}/{total_jobs}] dataset={dataset_path.stem} run={run_idx} "
                f"strategy={strategy}"
            )
            code, elapsed, best_cost, output = run_solver_once(
                repo_root=repo_root,
                exe_path=exe_path,
                alpha=args.alpha,
                dataset_path=dataset_path,
                strategy=strategy,
                timeout=args.solve_timeout,
            )
            if code != 0:
                print(f"  [WARN] return_code={code}")
            row = {
                "dataset": dataset_path.stem,
                "run": run_idx,
                "strategy": strategy,
                "knn_enabled": knn_enabled,
                "return_code": code,
                "elapsed_sec": elapsed,
                "best_cost": best_cost,
            }
            raw_rows.append(row)
            with raw_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writerow(row)

    for strategy, label, knn_enabled in strategy_specs:
        strategy_rows = [row for row in raw_rows if str(row["strategy"]) == strategy]
        times = [float(row["elapsed_sec"]) for row in strategy_rows if float(row["elapsed_sec"]) == float(row["elapsed_sec"])]
        costs = [float(row["best_cost"]) for row in strategy_rows if float(row["best_cost"]) == float(row["best_cost"])]
        successful_runs = sum(
            1
            for row in strategy_rows
            if int(row["return_code"]) == 0
            and float(row["elapsed_sec"]) == float(row["elapsed_sec"])
            and float(row["best_cost"]) == float(row["best_cost"])
        )
        mean_time, std_time = mean_std(times)
        mean_cost, std_cost = mean_std(costs)
        summary_rows.append(
            {
                "Strategy": label,
                "KNN": "On" if knn_enabled else "Off",
                "Runs": args.runs,
                "Successful Runs": successful_runs,
                "Mean Time (s)": mean_time,
                "Std Time (s)": std_time,
                "Mean Best Cost": mean_cost,
                "Std Best Cost": std_cost,
            }
        )

    summary_fields = [
        "Strategy",
        "KNN",
        "Runs",
        "Successful Runs",
        "Mean Time (s)",
        "Std Time (s)",
        "Mean Best Cost",
        "Std Best Cost",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown_table(summary_rows, summary_md)
    plot_summary(summary_rows, plot_png)

    print(f"Saved raw CSV: {raw_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary table: {summary_md}")
    print(f"Saved figure: {plot_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
