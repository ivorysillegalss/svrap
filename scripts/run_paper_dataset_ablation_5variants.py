#!/usr/bin/env python3
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


def run_cmd(cmd: List[str], timeout: int, cwd: Path = None) -> Tuple[int, str, float]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
            cwd=cwd,
        )
        return proc.returncode, proc.stdout or "", time.time() - start
    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]", time.time() - start


def build_solver_cmd(
    python_exe: str,
    solver_py: Path,
    dataset_path: Path,
    variant_tag: str,
    seed: int,
    epochs: int,
    extra_flags: List[str],
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
    cmd.extend(extra_flags)
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


VARIANT_ORDER = ["origin", "p_route", "counterfactual", "K-NN", "full"]


VARIANT_CONFIG = {
    "origin": {
        "needs_python": False,
        "cpp_strategy": "paper_baseline",
        "py_flags": [],
    },
    "p_route": {
        "needs_python": True,
        "cpp_strategy": "nn_only",
        "py_flags": ["--no-counterfactual-node-loss", "--entropy-mode", "off"],
    },
    "counterfactual": {
        "needs_python": True,
        "cpp_strategy": "nn_only",
        "py_flags": ["--counterfactual-node-loss", "--entropy-mode", "off"],
    },
    "K-NN": {
        "needs_python": True,
        "cpp_strategy": "full",
        "py_flags": ["--counterfactual-node-loss", "--entropy-mode", "off"],
    },
    "full": {
        "needs_python": True,
        "cpp_strategy": "full",
        "py_flags": ["--counterfactual-node-loss", "--entropy-mode", "anneal"],
    },
}


def rebuild_svrap(repo_root: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would run: make svrap (cwd={repo_root})")
        return
    subprocess.run(["make", "svrap"], cwd=repo_root, check=True)


def render_markdown(summary_map: Dict[str, Dict[str, List[float]]]) -> str:
    lines: List[str] = []
    lines.append("# Paper dataset ablation summary")
    lines.append("")
    for ds in sorted(summary_map):
        lines.append(f"## {ds}")
        lines.append("| Variant | Mean Cost | Std Cost | Gap vs origin (%) |")
        lines.append("|---|---:|---:|---:|")
        origin_vals = summary_map[ds].get("origin", [])
        origin_mean = mean_std(origin_vals)[0]
        for variant in VARIANT_ORDER:
            vals = summary_map[ds].get(variant, [])
            mean_cost, std_cost = mean_std(vals)
            gap = float("nan")
            if origin_mean == origin_mean and origin_mean != 0 and mean_cost == mean_cost:
                gap = ((mean_cost - origin_mean) / origin_mean) * 100.0
            lines.append(f"| {variant} | {mean_cost:.2f} | {std_cost:.2f} | {gap:.2f} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return repo_root / path

    parser = argparse.ArgumentParser(
        description="Run 5-variant paper-dataset ablation study and export CSV/MD/PNG results."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to svrap.exe")
    parser.add_argument("--dataset-dir", default=str(repo_root / "paper_dataset"), help="Dataset directory")
    parser.add_argument("--runs", type=int, default=5, help="Runs per dataset")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA parameter for C++ solver")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs per run")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed")
    parser.add_argument("--train-timeout", type=int, default=7200, help="Timeout for solver training runs")
    parser.add_argument("--solve-timeout", type=int, default=1800, help="Timeout for C++ solve runs")
    parser.add_argument("--skip-rebuild-svrap", action="store_true", help="Skip automatic `make svrap` rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--list-variants", action="store_true", help="Print variants and exit")
    parser.add_argument(
        "--raw-out",
        default=str(repo_root / "results" / "paper_dataset_ablation_5variants_raw.csv"),
        help="Raw CSV output",
    )
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "paper_dataset_ablation_5variants_summary.csv"),
        help="Summary CSV output",
    )
    parser.add_argument(
        "--summary-md",
        default=str(repo_root / "results" / "paper_dataset_ablation_5variants_summary.md"),
        help="Summary markdown output",
    )
    parser.add_argument(
        "--gap-plot",
        default=str(repo_root / "results" / "paper_dataset_ablation_5variants_gap.png"),
        help="Gap chart output",
    )
    args = parser.parse_args()

    if args.list_variants:
        print("Variants:")
        for variant in VARIANT_ORDER:
            cfg = VARIANT_CONFIG[variant]
            print(f"- {variant}: cpp_strategy={cfg['cpp_strategy']}, needs_python={cfg['needs_python']}, py_flags={cfg['py_flags']}")
        return 0

    solver_py = resolve_path(args.solver)
    exe_path = resolve_path(args.exe)
    dataset_dir = resolve_path(args.dataset_dir)
    if not dataset_dir.exists() and dataset_dir.name == "paper_dataset":
        fallback = repo_root / "main_dataset"
        if fallback.exists():
            print(f"[WARN] {dataset_dir} not found, falling back to {fallback}")
            dataset_dir = fallback
    raw_out = Path(args.raw_out)
    summary_out = Path(args.summary_out)
    summary_md = Path(args.summary_md)
    gap_plot = Path(args.gap_plot)

    if not solver_py.exists():
        print(f"[ERROR] Solver not found: {solver_py}", file=sys.stderr)
        return 1
    if not exe_path.exists() and not args.dry_run:
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    datasets = sorted(dataset_dir.glob("*.txt"))
    if not datasets:
        print(f"[ERROR] No dataset files in {dataset_dir}", file=sys.stderr)
        return 1

    if not args.skip_rebuild_svrap:
        rebuild_svrap(repo_root, args.dry_run)

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    gap_plot.parent.mkdir(parents=True, exist_ok=True)

    raw_fields = [
        "dataset",
        "variant",
        "run",
        "seed",
        "train_return_code",
        "train_time_s",
        "solve_return_code",
        "solve_time_s",
        "cost",
    ]

    rows: List[Dict[str, Any]] = []
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()

        total_jobs = len(datasets) * args.runs * len(VARIANT_ORDER)
        job_idx = 0
        print(f"Datasets={len(datasets)} runs={args.runs} variants={len(VARIANT_ORDER)} total_jobs={total_jobs}")

        for dataset_path in datasets:
            dataset_name = dataset_path.stem
            for run_idx in range(1, args.runs + 1):
                seed = args.seed_base + run_idx

                for variant in VARIANT_ORDER:
                    job_idx += 1
                    cfg = VARIANT_CONFIG[variant]
                    print(f"[{job_idx}/{total_jobs}] {dataset_name} run={run_idx} variant={variant}")

                    train_return_code = 0
                    train_time = 0.0

                    if cfg["needs_python"]:
                        solver_cmd = build_solver_cmd(
                            python_exe=args.python,
                            solver_py=solver_py,
                            dataset_path=dataset_path,
                            variant_tag=f"ablation_{dataset_name}_{variant}_r{run_idx}",
                            seed=seed,
                            epochs=args.epochs,
                            extra_flags=list(cfg["py_flags"]),
                        )
                        if args.dry_run:
                            print("[DRY-RUN] " + " ".join(solver_cmd))
                        else:
                            train_return_code, _, train_time = run_cmd(solver_cmd, timeout=args.train_timeout, cwd=repo_root)

                    solve_cmd = build_cpp_cmd(exe_path, args.alpha, dataset_path, cfg["cpp_strategy"])
                    if args.dry_run:
                        print("[DRY-RUN] " + " ".join(solve_cmd))
                        solve_return_code = 0
                        solve_time = 0.0
                        cost = float("nan")
                    else:
                        solve_return_code, solve_output, solve_time = run_cmd(solve_cmd, timeout=args.solve_timeout, cwd=repo_root)
                        cost = parse_best_cost(solve_output)

                    row = {
                        "dataset": dataset_name,
                        "variant": variant,
                        "run": run_idx,
                        "seed": seed,
                        "train_return_code": int(train_return_code),
                        "train_time_s": round(train_time, 3),
                        "solve_return_code": int(solve_return_code),
                        "solve_time_s": round(solve_time, 3),
                        "cost": cost,
                    }
                    rows.append(row)
                    writer.writerow(row)

    summary_map: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        summary_map.setdefault(row["dataset"], {})
        summary_map[row["dataset"]].setdefault(row["variant"], [])
        summary_map[row["dataset"]][row["variant"]].append(float(row["cost"]))

    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "mean_cost", "std_cost", "n"])
        writer.writeheader()
        for dataset_name in sorted(summary_map):
            for variant in VARIANT_ORDER:
                mean_cost, std_cost = mean_std(summary_map[dataset_name].get(variant, []))
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "variant": variant,
                        "mean_cost": mean_cost,
                        "std_cost": std_cost,
                        "n": len(summary_map[dataset_name].get(variant, [])),
                    }
                )

    summary_md.write_text(render_markdown(summary_map), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.14
        x = list(range(len(summary_map)))
        datasets_sorted = sorted(summary_map)

        for idx, variant in enumerate(VARIANT_ORDER):
            gaps = []
            for dataset_name in datasets_sorted:
                origin_mean = mean_std(summary_map[dataset_name].get("origin", []))[0]
                variant_mean = mean_std(summary_map[dataset_name].get(variant, []))[0]
                if origin_mean == origin_mean and origin_mean != 0 and variant_mean == variant_mean:
                    gaps.append(((variant_mean - origin_mean) / origin_mean) * 100.0)
                else:
                    gaps.append(float("nan"))
            offsets = [pos + (idx - 2) * width for pos in x]
            ax.bar(offsets, gaps, width=width, label=variant)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets_sorted, rotation=30, ha="right")
        ax.set_ylabel("Gap vs origin (%)")
        ax.set_title("Paper dataset ablation gap comparison")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(str(gap_plot), dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to render plot: {exc}", file=sys.stderr)

    print(f"Saved raw CSV: {raw_out}")
    print(f"Saved summary CSV: {summary_out}")
    print(f"Saved summary MD: {summary_md}")
    print(f"Saved gap plot: {gap_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
