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


def discover_datasets(dataset_dir: Path, dataset_names: str | None) -> List[Path]:
    if dataset_names:
        names = [name.strip() for name in dataset_names.split(",") if name.strip()]
        datasets = [dataset_dir / f"{name}.txt" for name in names]
    else:
        datasets = sorted(dataset_dir.glob("*.txt"))

    missing = [path for path in datasets if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing dataset file(s): {missing_text}")

    return datasets


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
        "| Dataset | Runs | Full Time (s) | No-KNN Time (s) | Time Improvement (%) | Full Cost | No-KNN Cost | Cost Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {runs} | {full_time} | {no_knn_time} | {improvement} | {full_cost} | {no_knn_cost} | {cost_delta} |".format(
                dataset=row["Dataset"],
                runs=row["Runs"],
                full_time=format_float(float(row["Full Mean Time (s)"]), 2),
                no_knn_time=format_float(float(row["No-KNN Mean Time (s)"]), 2),
                improvement=format_float(float(row["Time Improvement (%)"]), 2),
                full_cost=format_float(float(row["Full Mean Best Cost"]), 2),
                no_knn_cost=format_float(float(row["No-KNN Mean Best Cost"]), 2),
                cost_delta=format_float(float(row["Cost Delta"]), 2),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary_rows: List[Dict[str, object]], out_png: Path) -> None:
    ordered_rows = sorted(summary_rows, key=lambda row: float(row["Time Improvement (%)"]), reverse=True)
    labels = [str(row["Dataset"]) for row in ordered_rows]
    improvements = [float(row["Time Improvement (%)"]) for row in ordered_rows]
    full_times = [float(row["Full Mean Time (s)"]) for row in ordered_rows]
    no_knn_times = [float(row["No-KNN Mean Time (s)"]) for row in ordered_rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    height = max(6.0, 0.42 * len(ordered_rows) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, height), constrained_layout=True)

    x = range(len(ordered_rows))
    colors = ["#0F766E", "#B45309"]
    axes[0].bar(x, full_times, width=0.4, label="KNN On", color=colors[0])
    axes[0].bar([i + 0.4 for i in x], no_knn_times, width=0.4, label="KNN Off", color=colors[1])
    axes[0].set_xticks([i + 0.2 for i in x], labels, rotation=45, ha="right")
    axes[0].set_ylabel("Solve Time (s)")
    axes[0].set_title("Mean Solve Time by Dataset")
    axes[0].legend(loc="best")
    axes[0].grid(axis="y", alpha=0.3)

    y = list(range(len(ordered_rows)))
    bar_colors = ["#059669" if value >= 0 else "#DC2626" for value in improvements]
    bars = axes[1].barh(y, improvements, color=bar_colors, edgecolor="#1f2937")
    axes[1].axvline(0, color="#111827", linewidth=1.0)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Time Improvement (%) = (No-KNN - KNN) / No-KNN")
    axes[1].set_title("KNN Time Improvement")
    axes[1].grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, improvements):
        axes[1].annotate(
            f"{value:.1f}%",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    best_row = ordered_rows[0] if ordered_rows else None
    title = "All-Dataset KNN Comparison (2 runs per dataset)"
    if best_row is not None:
        title += f" | best: {best_row['Dataset']} ({float(best_row['Time Improvement (%)']):.1f}%)"
    fig.suptitle(title, fontsize=15)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Run all datasets with KNN on/off, 2 runs each by default, and export a summary table, raw CSV, and a figure."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ solver executable")
    parser.add_argument("--dataset-dir", default=str(repo_root / "main_dataset"), help="Directory containing dataset files")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to run; default is all .txt files in --dataset-dir",
    )
    parser.add_argument("--alpha", type=float, default=7.0, help="Alpha parameter for C++ solver")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per dataset and strategy")
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
        default=str(repo_root / "results" / "all_datasets_knn_comparison"),
        help="Output prefix for CSV/MD/PNG files",
    )

    args = parser.parse_args()

    solver_py = Path(args.solver)
    exe_path = Path(args.exe)
    dataset_dir = Path(args.dataset_dir)
    out_prefix = Path(args.out_prefix)

    if not solver_py.exists():
        print(f"[ERROR] Solver not found: {solver_py}", file=sys.stderr)
        return 1
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_prefix.with_name(out_prefix.name + "_raw.csv")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    summary_md = out_prefix.with_name(out_prefix.name + "_summary.md")
    plot_png = out_prefix.with_name(out_prefix.name + "_summary.png")

    dataset_paths = discover_datasets(dataset_dir, args.datasets)

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

    total_jobs = len(dataset_paths) * len(strategy_specs) * args.runs
    job_idx = 0
    summary_fields = [
        "Dataset",
        "Runs",
        "Full Mean Time (s)",
        "Full Std Time (s)",
        "No-KNN Mean Time (s)",
        "No-KNN Std Time (s)",
        "Time Delta (s)",
        "Time Improvement (%)",
        "Full Mean Best Cost",
        "Full Std Best Cost",
        "No-KNN Mean Best Cost",
        "No-KNN Std Best Cost",
        "Cost Delta",
        "Best Improvement Flag",
    ]

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.stem
        ensure_attention_probs(
            repo_root=repo_root,
            python_exe=args.python,
            solver_py=solver_py,
            dataset_path=dataset_path,
            variant_tag=f"{args.variant_tag}_{dataset_name}" if args.variant_tag else dataset_name,
            epochs=args.epochs,
            seed=args.seed,
            timeout=args.prep_timeout,
        )

        per_dataset_rows: Dict[str, List[Dict[str, object]]] = {strategy: [] for strategy, _, _ in strategy_specs}

        for strategy, label, knn_enabled in strategy_specs:
            for run_idx in range(1, args.runs + 1):
                job_idx += 1
                print(
                    f"[{job_idx}/{total_jobs}] dataset={dataset_name} run={run_idx} "
                    f"strategy={strategy}"
                )
                code, elapsed, best_cost, _ = run_solver_once(
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
                    "dataset": dataset_name,
                    "run": run_idx,
                    "strategy": strategy,
                    "knn_enabled": knn_enabled,
                    "return_code": code,
                    "elapsed_sec": elapsed,
                    "best_cost": best_cost,
                }
                raw_rows.append(row)
                per_dataset_rows[strategy].append(row)
                with raw_csv.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=raw_fields)
                    writer.writerow(row)

        full_rows = per_dataset_rows["full"]
        no_knn_rows = per_dataset_rows["no_knn"]
        full_times = [float(row["elapsed_sec"]) for row in full_rows if float(row["elapsed_sec"]) == float(row["elapsed_sec"])]
        no_knn_times = [float(row["elapsed_sec"]) for row in no_knn_rows if float(row["elapsed_sec"]) == float(row["elapsed_sec"])]
        full_costs = [float(row["best_cost"]) for row in full_rows if float(row["best_cost"]) == float(row["best_cost"])]
        no_knn_costs = [float(row["best_cost"]) for row in no_knn_rows if float(row["best_cost"]) == float(row["best_cost"])]

        full_time_mean, full_time_std = mean_std(full_times)
        no_knn_time_mean, no_knn_time_std = mean_std(no_knn_times)
        full_cost_mean, full_cost_std = mean_std(full_costs)
        no_knn_cost_mean, no_knn_cost_std = mean_std(no_knn_costs)

        time_delta = no_knn_time_mean - full_time_mean
        time_improvement_pct = float("nan")
        if no_knn_time_mean == no_knn_time_mean and no_knn_time_mean != 0:
            time_improvement_pct = time_delta / no_knn_time_mean * 100.0

        cost_delta = full_cost_mean - no_knn_cost_mean
        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Runs": args.runs,
                "Full Mean Time (s)": full_time_mean,
                "Full Std Time (s)": full_time_std,
                "No-KNN Mean Time (s)": no_knn_time_mean,
                "No-KNN Std Time (s)": no_knn_time_std,
                "Time Delta (s)": time_delta,
                "Time Improvement (%)": time_improvement_pct,
                "Full Mean Best Cost": full_cost_mean,
                "Full Std Best Cost": full_cost_std,
                "No-KNN Mean Best Cost": no_knn_cost_mean,
                "No-KNN Std Best Cost": no_knn_cost_std,
                "Cost Delta": cost_delta,
                "Best Improvement Flag": "",
            }
        )

    valid_improvements = [row for row in summary_rows if row["Time Improvement (%)"] == row["Time Improvement (%)"]]
    if valid_improvements:
        best_row = max(valid_improvements, key=lambda row: float(row["Time Improvement (%)"]))
        best_row["Best Improvement Flag"] = "BEST"

    summary_rows = sorted(summary_rows, key=lambda row: float(row["Time Improvement (%)"]), reverse=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown_table(summary_rows, summary_md)
    plot_summary(summary_rows, plot_png)

    if summary_rows:
        best_row = summary_rows[0]
        print(
            f"Best improvement: {best_row['Dataset']} ({float(best_row['Time Improvement (%)']):.2f}%), "
            f"full={float(best_row['Full Mean Time (s)']):.2f}s, no_knn={float(best_row['No-KNN Mean Time (s)']):.2f}s"
        )

    print(f"Saved raw CSV: {raw_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary table: {summary_md}")
    print(f"Saved figure: {plot_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
