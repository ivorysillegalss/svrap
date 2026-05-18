import argparse
import csv
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    cpp_strategy: str
    needs_python: bool = False
    counterfactual_node_loss: bool = False
    entropy_mode: str = "off"
    entropy_bonus_weight: float = 0.01
    entropy_bonus_start: float = 0.01
    entropy_bonus_end: float = 0.0


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
        return proc.returncode, proc.stdout or "", time.time() - t0

    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]", time.time() - t0

def rebuild_svrap(repo_root: Path, timeout: int) -> Tuple[int, str, float]:
    return run_cmd(["make", "svrap"], cwd=repo_root, timeout=timeout)


def safe_mean(values: Iterable[float]) -> float:
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan")
    return statistics.mean(clean)


def safe_std(values: Iterable[float]) -> float:
    clean = [value for value in values if value == value]
    if len(clean) <= 1:
        return 0.0 if len(clean) == 1 else float("nan")
    return statistics.stdev(clean)


def format_float(value: float, digits: int = 2) -> str:
    if value != value:
        return "NaN"
    return f"{value:.{digits}f}"


def discover_datasets(dataset_dir: Path, requested: Optional[str]) -> List[Tuple[str, Path]]:
    if requested:
        names = [part.strip() for part in requested.split(",") if part.strip()]
    else:
        preferred_order = ["berlin52", "bier127", "d198", "gr96", "pr107", "pr152"]
        discovered = {path.stem: path for path in dataset_dir.glob("*.txt")}
        names = [name for name in preferred_order if name in discovered]
        leftovers = sorted(name for name in discovered if name not in names)
        names.extend(leftovers)

    dataset_paths: List[Tuple[str, Path]] = []
    for name in names:
        path = dataset_dir / f"{name}.txt"
        if path.exists():
            dataset_paths.append((name, path))
    return dataset_paths


def run_python_prep(
    repo_root: Path,
    python_exe: str,
    solver_py: Path,
    dataset_path: Path,
    variant_tag: str,
    seed: int,
    epochs: int,
    variant: VariantSpec,
    timeout: int,
) -> Tuple[int, str, float]:
    cmd: List[str] = [
        python_exe,
        str(solver_py),
        "--dataset",
        str(dataset_path),
        "--train",
        "--epochs",
        str(epochs),
        "--variant-tag",
        variant_tag,
        "--seed",
        str(seed),
    ]

    if variant.counterfactual_node_loss:
        cmd.append("--counterfactual-node-loss")
    else:
        cmd.append("--no-counterfactual-node-loss")

    if variant.entropy_mode == "anneal":
        cmd.extend(
            [
                "--entropy-mode",
                "anneal",
                "--entropy-bonus-start",
                str(variant.entropy_bonus_start),
                "--entropy-bonus-end",
                str(variant.entropy_bonus_end),
            ]
        )
    elif variant.entropy_mode == "fixed":
        cmd.extend(
            [
                "--entropy-mode",
                "fixed",
                "--entropy-bonus-weight",
                str(variant.entropy_bonus_weight),
            ]
        )
    else:
        cmd.extend(["--entropy-mode", "off"])

    return run_cmd(cmd, cwd=repo_root, timeout=timeout)


def run_cpp_solver(
    repo_root: Path,
    exe_path: Path,
    alpha: float,
    dataset_path: Path,
    strategy: str,
    timeout: int,
) -> Tuple[int, str, float, float]:
    cmd = [str(exe_path), str(alpha), str(dataset_path), strategy]
    code, output, elapsed = run_cmd(cmd, cwd=repo_root, timeout=timeout)
    best_cost = parse_best_cost(output)
    return code, output, elapsed, best_cost


def write_markdown(summary_rows: List[Dict[str, object]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# Paper Dataset Ablation Suite")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = [
        "Dataset",
        "Variant",
        "Runs",
        "Mean Cost",
        "Std Cost",
        "Mean Time (s)",
        "Std Time (s)",
        "Mean Gap (%)",
        "Std Gap (%)",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in summary_rows:
        lines.append(
            "| {dataset} | {variant} | {runs} | {mean_cost} | {std_cost} | {mean_time} | {std_time} | {mean_gap} | {std_gap} |".format(
                dataset=row["dataset"],
                variant=row["variant"],
                runs=row["runs"],
                mean_cost=format_float(float(row["mean_cost"]), 2),
                std_cost=format_float(float(row["std_cost"]), 2),
                mean_time=format_float(float(row["mean_time_sec"]), 2),
                std_time=format_float(float(row["std_time_sec"]), 2),
                mean_gap=format_float(float(row["mean_gap_pct"]), 2),
                std_gap=format_float(float(row["std_gap_pct"]), 2),
            )
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_gap_bars(summary_rows: List[Dict[str, object]], variants: List[VariantSpec], out_png: Path) -> None:
    dataset_names = []
    for row in summary_rows:
        if row["dataset"] not in dataset_names:
            dataset_names.append(str(row["dataset"]))

    dataset_to_rows: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in summary_rows:
        dataset_to_rows.setdefault(str(row["dataset"]), {})[str(row["variant"])] = row

    n_datasets = len(dataset_names)
    cols = min(3, max(1, n_datasets))
    rows = math.ceil(n_datasets / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    x_labels = [variant.label for variant in variants]
    x_positions = list(range(len(variants)))
    colors = [
        "#334155",
        "#0F766E",
        "#2563EB",
        "#B45309",
        "#7C3AED",
    ]

    for index, dataset_name in enumerate(dataset_names):
        ax = axes[index // cols][index % cols]
        rows_for_dataset = dataset_to_rows.get(dataset_name, {})
        gaps = []
        gap_stds = []
        for variant in variants:
            row = rows_for_dataset.get(variant.label)
            if row is None:
                gaps.append(float("nan"))
                gap_stds.append(float("nan"))
            else:
                gaps.append(float(row["mean_gap_pct"]))
                gap_stds.append(float(row["std_gap_pct"]))

        bars = ax.bar(
            x_positions,
            gaps,
            yerr=gap_stds,
            capsize=5,
            color=colors[: len(variants)],
            edgecolor="#1f2937",
        )
        ax.axhline(0.0, color="#1f2937", linewidth=1.0, alpha=0.65)
        ax.set_title(dataset_name)
        ax.set_xticks(x_positions, x_labels, rotation=22, ha="right")
        ax.set_ylabel("Gap (%)")
        ax.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, gaps):
            if value == value:
                ax.annotate(
                    f"{value:.1f}",
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                    xytext=(0, 3 if value >= 0 else -3),
                    textcoords="offset points",
                )

    for empty_idx in range(n_datasets, rows * cols):
        axes[empty_idx // cols][empty_idx % cols].axis("off")

    fig.suptitle("Paper Dataset Ablation Gaps", fontsize=15)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    solver_py = repo_root / "svrap_solver.py"
    exe_path = repo_root / "svrap.exe"
    default_dataset_dir = repo_root / "paper_dataset"

    parser = argparse.ArgumentParser(
        description=(
            "Run paper_dataset ablation experiments for the paper baseline, p_route, KNN, entropy annealing, "
            "and counterfactual variants."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run svrap_solver.py")
    parser.add_argument("--solver", default=str(solver_py), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(exe_path), help="Path to svrap.exe")
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir), help="Directory containing paper datasets")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names. Leave empty to auto-discover paper_dataset/*.txt",
    )
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA parameter for the C++ solver")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per dataset and variant")
    parser.add_argument("--epochs", type=int, default=1200, help="Training epochs for neural variants")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--prep-timeout", type=int, default=7200, help="Timeout seconds for each Python prep run")
    parser.add_argument("--solve-timeout", type=int, default=3600, help="Timeout seconds for each C++ run")
    parser.add_argument(
        "--variant-tag-prefix",
        default="paper_ablation",
        help="Prefix used to isolate model/log artifacts per variant run",
    )
    parser.add_argument("--raw-out", default=str(repo_root / "results" / "paper_dataset_ablation_raw.csv"), help="Raw CSV output")
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "paper_dataset_ablation_summary.csv"),
        help="Summary CSV output",
    )
    parser.add_argument(
        "--summary-md",
        default=str(repo_root / "results" / "paper_dataset_ablation_summary.md"),
        help="Summary Markdown output",
    )
    parser.add_argument(
        "--plot-out",
        default=str(repo_root / "results" / "paper_dataset_ablation_gap.png"),
        help="Gap bar chart output",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without executing them")
    parser.add_argument(
        "--skip-rebuild-svrap",
        action="store_true",
        help="Skip the automatic svrap rebuild step before running the suite",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print the variant plan and exit",
    )
    args = parser.parse_args()

    solver_py_path = Path(args.solver)
    exe_path = Path(args.exe)
    dataset_dir = Path(args.dataset_dir)
    raw_out = Path(args.raw_out)
    summary_out = Path(args.summary_out)
    summary_md = Path(args.summary_md)
    plot_out = Path(args.plot_out)

    variants = [
        VariantSpec(key="baseline", label="original", cpp_strategy="paper_baseline"),
        VariantSpec(key="p_route", label="p_route", cpp_strategy="nn_only", needs_python=True),
        VariantSpec(key="knn", label="KN", cpp_strategy="knn_only"),
        VariantSpec(
            key="entropy_anneal",
            label="entropy_anneal",
            cpp_strategy="nn_only",
            needs_python=True,
            entropy_mode="anneal",
            entropy_bonus_start=0.01,
            entropy_bonus_end=0.0,
        ),
        VariantSpec(
            key="counterfactual",
            label="counterfactual",
            cpp_strategy="nn_only",
            needs_python=True,
            counterfactual_node_loss=True,
        ),
    ]

    if args.list_variants:
        for variant in variants:
            print(f"{variant.label}: cpp_strategy={variant.cpp_strategy}, needs_python={variant.needs_python}, "
                  f"counterfactual={variant.counterfactual_node_loss}, entropy_mode={variant.entropy_mode}")
        return 0

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1
    if not solver_py_path.exists():
        print(f"[ERROR] Solver not found: {solver_py_path}", file=sys.stderr)
        return 1
    if not args.dry_run and not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not args.skip_rebuild_svrap and not args.dry_run:
        print("Rebuilding svrap.exe from the current source tree...")
        build_code, build_output, build_elapsed = rebuild_svrap(repo_root, timeout=args.solve_timeout)
        if build_code != 0:
            print(f"[ERROR] svrap rebuild failed after {build_elapsed:.1f}s (code={build_code})", file=sys.stderr)
            print(build_output)
            return build_code
        print(f"svrap rebuild finished in {build_elapsed:.1f}s")

    dataset_paths = discover_datasets(dataset_dir, args.datasets)
    if not dataset_paths:
        print(f"[ERROR] No dataset files found in {dataset_dir}", file=sys.stderr)
        return 1

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using datasets: {', '.join(name for name, _ in dataset_paths)}")
    print(f"Using variants: {', '.join(variant.label for variant in variants)}")

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    total_jobs = len(dataset_paths) * args.runs * len(variants)
    current_job = 0

    for dataset_name, dataset_path in dataset_paths:
        print(f"\n=== Dataset: {dataset_name} ===")

        baseline_costs: List[float] = []
        baseline_times: List[float] = []
        baseline_return_codes: List[int] = []

        for run_idx in range(1, args.runs + 1):
            current_job += 1
            print(f"[{current_job}/{total_jobs}] {dataset_name} run={run_idx} variant=original")
            if args.dry_run:
                print(f"  C++: {exe_path} {args.alpha} {dataset_path} paper_baseline")
                continue

            code, output, elapsed, best_cost = run_cpp_solver(
                repo_root=repo_root,
                exe_path=exe_path,
                alpha=args.alpha,
                dataset_path=dataset_path,
                strategy="paper_baseline",
                timeout=args.solve_timeout,
            )
            baseline_costs.append(best_cost)
            baseline_times.append(elapsed)
            baseline_return_codes.append(code)
            raw_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": "original",
                    "run": run_idx,
                    "return_code": code,
                    "elapsed_sec": elapsed,
                    "best_cost": best_cost,
                    "gap_vs_baseline_pct": 0.0,
                    "cpp_strategy": "paper_baseline",
                    "python_enabled": False,
                    "counterfactual_node_loss": False,
                    "entropy_mode": "off",
                }
            )

        if args.dry_run:
            for variant in variants[1:]:
                for run_idx in range(1, args.runs + 1):
                    current_job += 1
                    print(f"[{current_job}/{total_jobs}] {dataset_name} run={run_idx} variant={variant.label}")
                    if variant.needs_python:
                        seed = args.seed + run_idx
                        variant_tag = f"{args.variant_tag_prefix}_{dataset_name}_{variant.key}_r{run_idx}"
                        print(
                            f"  Python: {args.python} {solver_py_path} --dataset {dataset_path} --train --epochs {args.epochs} "
                            f"--variant-tag {variant_tag} --seed {seed} "
                            f"{'--counterfactual-node-loss' if variant.counterfactual_node_loss else '--no-counterfactual-node-loss'} "
                            f"--entropy-mode {variant.entropy_mode}"
                        )
                    print(f"  C++: {exe_path} {args.alpha} {dataset_path} {variant.cpp_strategy}")
            continue

        if len(baseline_costs) != args.runs:
            print(f"[ERROR] Baseline produced {len(baseline_costs)} successful runs for {dataset_name}; expected {args.runs}", file=sys.stderr)
            return 1

        for variant in variants[1:]:
            variant_costs: List[float] = []
            variant_times: List[float] = []
            variant_gaps: List[float] = []
            variant_return_codes: List[int] = []

            for run_idx in range(1, args.runs + 1):
                current_job += 1
                seed = args.seed + run_idx
                variant_tag = f"{args.variant_tag_prefix}_{dataset_name}_{variant.key}_r{run_idx}"
                print(f"[{current_job}/{total_jobs}] {dataset_name} run={run_idx} variant={variant.label}")

                if variant.needs_python:
                    prep_code, prep_output, prep_elapsed = run_python_prep(
                        repo_root=repo_root,
                        python_exe=args.python,
                        solver_py=solver_py_path,
                        dataset_path=dataset_path,
                        variant_tag=variant_tag,
                        seed=seed,
                        epochs=args.epochs,
                        variant=variant,
                        timeout=args.prep_timeout,
                    )
                    if prep_code != 0:
                        print(
                            f"  [WARN] Python prep returned {prep_code} after {prep_elapsed:.1f}s for {variant.label}"
                        )
                        print(prep_output)

                code, output, elapsed, best_cost = run_cpp_solver(
                    repo_root=repo_root,
                    exe_path=exe_path,
                    alpha=args.alpha,
                    dataset_path=dataset_path,
                    strategy=variant.cpp_strategy,
                    timeout=args.solve_timeout,
                )

                baseline_cost = baseline_costs[run_idx - 1]
                gap_pct = float("nan")
                if baseline_cost == baseline_cost and abs(baseline_cost) > 1e-12 and best_cost == best_cost:
                    gap_pct = (best_cost - baseline_cost) / baseline_cost * 100.0

                variant_costs.append(best_cost)
                variant_times.append(elapsed)
                variant_gaps.append(gap_pct)
                variant_return_codes.append(code)

                raw_rows.append(
                    {
                        "dataset": dataset_name,
                        "variant": variant.label,
                        "run": run_idx,
                        "return_code": code,
                        "elapsed_sec": elapsed,
                        "best_cost": best_cost,
                        "gap_vs_baseline_pct": gap_pct,
                        "cpp_strategy": variant.cpp_strategy,
                        "python_enabled": variant.needs_python,
                        "counterfactual_node_loss": variant.counterfactual_node_loss,
                        "entropy_mode": variant.entropy_mode,
                    }
                )

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant.label,
                    "cpp_strategy": variant.cpp_strategy,
                    "runs": args.runs,
                    "mean_cost": safe_mean(variant_costs),
                    "std_cost": safe_std(variant_costs),
                    "mean_time_sec": safe_mean(variant_times),
                    "std_time_sec": safe_std(variant_times),
                    "mean_gap_pct": safe_mean(variant_gaps),
                    "std_gap_pct": safe_std(variant_gaps),
                    "successful_runs": sum(1 for code in variant_return_codes if code == 0),
                }
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "variant": "original",
                "cpp_strategy": "paper_baseline",
                "runs": args.runs,
                "mean_cost": safe_mean(baseline_costs),
                "std_cost": safe_std(baseline_costs),
                "mean_time_sec": safe_mean(baseline_times),
                "std_time_sec": safe_std(baseline_times),
                "mean_gap_pct": 0.0,
                "std_gap_pct": 0.0,
                "successful_runs": sum(1 for code in baseline_return_codes if code == 0),
            }
        )

    if args.dry_run:
        print("\nDry run finished. No commands were executed.")
        return 0

    raw_fields = [
        "dataset",
        "variant",
        "run",
        "return_code",
        "elapsed_sec",
        "best_cost",
        "gap_vs_baseline_pct",
        "cpp_strategy",
        "python_enabled",
        "counterfactual_node_loss",
        "entropy_mode",
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_fields = [
        "dataset",
        "variant",
        "cpp_strategy",
        "runs",
        "successful_runs",
        "mean_cost",
        "std_cost",
        "mean_time_sec",
        "std_time_sec",
        "mean_gap_pct",
        "std_gap_pct",
    ]
    summary_rows_sorted = sorted(summary_rows, key=lambda row: (str(row["dataset"]), str(row["variant"])) )
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows_sorted)

    write_markdown(summary_rows_sorted, summary_md)
    plot_gap_bars(summary_rows_sorted, variants, plot_out)

    print(f"Saved raw results: {raw_out}")
    print(f"Saved summary CSV: {summary_out}")
    print(f"Saved summary Markdown: {summary_md}")
    print(f"Saved gap chart: {plot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())