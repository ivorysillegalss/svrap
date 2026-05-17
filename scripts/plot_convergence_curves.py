import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["Cost", "Baseline", "Best"]


def load_run_logs(dataset_dir: Path) -> List[pd.DataFrame]:
    run_logs: List[pd.DataFrame] = []
    for csv_path in sorted(dataset_dir.glob("run_*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        required = {"Epoch", "Cost", "Baseline", "Best"}
        if not required.issubset(df.columns):
            continue
        run_logs.append(df[["Epoch", "Cost", "Baseline", "Best"]].copy())
    return run_logs


def load_runtime_summary(raw_csv: Path) -> Dict[str, Dict[str, float]]:
    if not raw_csv.exists():
        return {}
    df = pd.read_csv(raw_csv)
    if df.empty or "dataset" not in df.columns or "elapsed_seconds" not in df.columns:
        return {}
    grouped = df.groupby("dataset")["elapsed_seconds"].agg(["mean", "std", "min", "max"])
    summary: Dict[str, Dict[str, float]] = {}
    for dataset, row in grouped.iterrows():
        summary[str(dataset)] = {
            "runtime_mean": float(row["mean"]),
            "runtime_std": float(0.0 if pd.isna(row["std"]) else row["std"]),
            "runtime_min": float(row["min"]),
            "runtime_max": float(row["max"]),
        }
    return summary


def align_metric(run_logs: List[pd.DataFrame], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_epoch = max(int(df["Epoch"].max()) for df in run_logs)
    values = np.full((len(run_logs), max_epoch + 1), np.nan, dtype=float)

    for run_idx, df in enumerate(run_logs):
        for _, row in df.iterrows():
            epoch = int(row["Epoch"])
            values[run_idx, epoch] = float(row[metric])

    epochs = np.arange(max_epoch + 1)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return epochs, mean, std


def plot_dataset(
    dataset: str,
    dataset_dir: Path,
    out_dir: Path,
    runtime_summary: Dict[str, float] | None = None,
) -> Dict[str, float]:
    run_logs = load_run_logs(dataset_dir)
    if not run_logs:
        raise RuntimeError(f"No valid run logs found in {dataset_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, float] = {
        "dataset": dataset,
        "runs": float(len(run_logs)),
    }
    if runtime_summary:
        summary.update(runtime_summary)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 12), sharex=True)
    title = f"Convergence Curves - {dataset}"
    if runtime_summary and "runtime_mean" in runtime_summary:
        title += f" | runtime {runtime_summary['runtime_mean']:.1f}s ± {runtime_summary.get('runtime_std', 0.0):.1f}s"
    fig.suptitle(title, fontsize=16)

    for ax, metric in zip(axes, METRICS):
        epochs, mean, std = align_metric(run_logs, metric)
        ax.plot(epochs, mean, label=f"{metric} (mean)", linewidth=2.0)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.18, label=f"{metric} ±1 std")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        valid_mean = mean[~np.isnan(mean)]
        valid_std = std[~np.isnan(std)]
        summary[f"{metric.lower()}_final_mean"] = float(valid_mean[-1])
        summary[f"{metric.lower()}_final_std"] = float(valid_std[-1])

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"{dataset}_convergence_curve.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return summary


def plot_overview(
    dataset_summaries: Dict[str, List[pd.DataFrame]],
    out_dir: Path,
    runtime_map: Dict[str, Dict[str, float]],
) -> None:
    datasets = list(dataset_summaries.keys())
    if not datasets:
        return

    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4 * len(datasets)), sharex=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        run_logs = dataset_summaries[dataset]
        epochs, mean, std = align_metric(run_logs, "Best")
        ax.plot(epochs, mean, linewidth=2.2, label=f"{dataset} best (mean)")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.18)
        ax.set_ylabel("Best")
        ax.grid(True, alpha=0.3)
        runtime_text = ""
        if dataset in runtime_map:
            runtime_text = (
                f" | runtime {runtime_map[dataset]['runtime_mean']:.1f}s"
                f" ± {runtime_map[dataset]['runtime_std']:.1f}s"
            )
        ax.set_title(f"{dataset}{runtime_text}")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    out_path = out_dir / "all_datasets_best_convergence_curve.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot final convergence curves from per-run training logs.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["berlin52", "pr152", "d198"],
        help="Dataset names to plot",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(repo_root / "results" / "convergence_curve_logs"),
        help="Directory containing copied per-run logs",
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "results" / "convergence_curves"),
        help="Directory for generated plots",
    )
    parser.add_argument(
        "--summary-out",
        default=str(repo_root / "results" / "convergence_curve_summary.csv"),
        help="CSV summary output path",
    )
    parser.add_argument(
        "--raw-csv",
        default=str(repo_root / "results" / "convergence_curve_test_raw.csv"),
        help="Raw per-run CSV output from the test runner",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    summary_out = Path(args.summary_out)
    raw_csv = Path(args.raw_csv)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    runtime_map = load_runtime_summary(raw_csv)

    rows: List[Dict[str, float]] = []
    dataset_runs: Dict[str, List[pd.DataFrame]] = {}

    for dataset in args.datasets:
        dataset_dir = logs_dir / dataset
        if not dataset_dir.exists():
            print(f"[WARN] Missing log directory, skipping: {dataset_dir}")
            continue
        run_logs = load_run_logs(dataset_dir)
        if not run_logs:
            print(f"[WARN] No valid run logs found, skipping: {dataset_dir}")
            continue
        dataset_runs[dataset] = run_logs
        summary = plot_dataset(dataset, dataset_dir, out_dir, runtime_map.get(dataset))
        rows.append(summary)
        print(f"Saved plot for {dataset}: {out_dir / f'{dataset}_convergence_curve.png'}")

    if not rows:
        raise RuntimeError("No dataset plots were generated.")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_overview(dataset_runs, out_dir, runtime_map)
    print(f"Saved overview plot: {out_dir / 'all_datasets_best_convergence_curve.png'}")
    print(f"Saved summary CSV: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())