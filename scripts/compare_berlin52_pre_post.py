import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


@dataclass
class VariantResult:
    tag: str
    label: str
    baseline_cost: float
    guided_cost: float
    gap_pct: float
    p_off_mean: float
    p_route_mean: float
    p_loss_mean: float
    p_route_min: float
    p_route_max: float
    p_route_std: float


def run_cmd(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Exit: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output)
    if not matches:
        raise ValueError("Failed to parse best cost from solver output")
    return float(matches[-1])


def score_label(gap_pct: float) -> str:
    if gap_pct <= -2.0:
        return "Significant Improvement"
    if gap_pct < 0.0:
        return "Slight Improvement"
    return "Regression"


def render_variant_plots(df_probs: pd.DataFrame, history_path: str, out_dir: str, tag: str):
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df_probs["p_route"], bins=20, kde=True)
    plt.title(f"Distribution of Attention Probabilities (p_route) for berlin52 [{tag}]")
    plt.xlabel("Probability (p_route)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"berlin52_p_route_distribution_{tag}.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_probs["x_norm"], df_probs["y_norm"], c=df_probs["p_route"], cmap="viridis", s=95, alpha=0.85)
    plt.colorbar(sc, label="P_Route Probability")
    top5 = df_probs.sort_values("p_route", ascending=False).head(5)
    for _, row in top5.iterrows():
        plt.annotate(f"#{int(row['node_id'])}", (row["x_norm"], row["y_norm"]), xytext=(4, 4), textcoords="offset points")
    plt.title(f"Spatial Distribution of P_Route for berlin52 [{tag}]")
    plt.xlabel("X (Normalized)")
    plt.ylabel("Y (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"berlin52_spatial_heatmap_{tag}.png"), dpi=160)
    plt.close()

    if os.path.exists(history_path):
        df_history = pd.read_csv(history_path)
        plt.figure(figsize=(10, 6))
        plt.plot(df_history["Epoch"], df_history["Cost"], label="Cost", alpha=0.6)
        plt.plot(df_history["Epoch"], df_history["Baseline"], label="Baseline (Greedy)", alpha=0.6)
        plt.plot(df_history["Epoch"], df_history["Best"], label="Best Cost", linewidth=2.0)
        plt.title(f"Training Cost Curve - Berlin52 [{tag}]")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"berlin52_cost_curve_{tag}.png"), dpi=160)
        plt.close()


def evaluate_variant(root_dir: str, dataset_path: str, tag: str, label: str, baseline_cost: float) -> VariantResult:
    python_exe = os.environ.get("SVRAP_PYTHON_EXE", "python")

    run_cmd(
        [
            python_exe,
            "svrap_solver.py",
            "--dataset",
            dataset_path,
            "--no-train",
            "--variant-tag",
            tag,
        ],
        cwd=root_dir,
    )

    attention_path = os.path.join(root_dir, "attention_probs.csv")
    out_dir = os.path.join(root_dir, "results", "berlin52_compare")
    os.makedirs(out_dir, exist_ok=True)

    tagged_attention = os.path.join(out_dir, f"attention_probs_{tag}.csv")
    shutil.copyfile(attention_path, tagged_attention)

    cpp_output = run_cmd([".\\svrap.exe", "7", dataset_path], cwd=root_dir)
    guided_cost = parse_best_cost(cpp_output)

    raw = pd.read_csv(tagged_attention, header=None)
    if raw.shape[1] >= 5:
        raw = raw.iloc[:, :5]
        raw.columns = ["x", "y", "p_assign", "p_route", "p_loss"]
        raw["p_off"] = raw["p_assign"] + raw["p_loss"]
    elif raw.shape[1] >= 4:
        raw = raw.iloc[:, :4]
        raw.columns = ["x", "y", "p_off", "p_route"]
        raw["p_assign"] = raw["p_off"]
        raw["p_loss"] = 0.0
    else:
        raise ValueError(f"Unexpected attention_probs format with {raw.shape[1]} columns")
    df = raw
    for col in ["x", "y", "p_assign", "p_route", "p_loss", "p_off"]:
        df[col] = pd.to_numeric(df[col])

    max_coord = max(df["x"].max(), df["y"].max())
    scale = max_coord if max_coord > 0 else 1.0
    df["x_norm"] = df["x"] / scale
    df["y_norm"] = df["y"] / scale
    df["node_id"] = range(len(df))

    history_path = os.path.join(root_dir, f"training_log_berlin52_{tag}.csv")
    render_variant_plots(df, history_path, out_dir, tag)

    p_off_mean = float(df["p_off"].mean())
    p_route_mean = float(df["p_route"].mean())
    p_loss_mean = float(df["p_loss"].mean())

    print(f"[{tag}] guided_cost={guided_cost:.2f}, baseline_cost={baseline_cost:.2f}, gap={(guided_cost - baseline_cost) / baseline_cost * 100.0:+.2f}%")
    print(f"[{tag}] class means: off={p_off_mean:.6f}, route={p_route_mean:.6f}, loss={p_loss_mean:.6f}")

    gap_pct = (guided_cost - baseline_cost) / baseline_cost * 100.0

    return VariantResult(
        tag=tag,
        label=label,
        baseline_cost=baseline_cost,
        guided_cost=guided_cost,
        gap_pct=gap_pct,
        p_off_mean=p_off_mean,
        p_route_mean=p_route_mean,
        p_loss_mean=p_loss_mean,
        p_route_min=float(df["p_route"].min()),
        p_route_max=float(df["p_route"].max()),
        p_route_std=float(df["p_route"].std()),
    )


def write_report(root_dir: str, results):
    out_dir = os.path.join(root_dir, "results", "berlin52_compare")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for r in results:
        rows.append(
            {
                "Version": r.label,
                "Dataset Class": "Small",
                "Dataset": "berlin52",
                "N": 52,
                "Baseline Cost": r.baseline_cost,
                "SVRAP (Full) Cost": r.guided_cost,
                "Gap (%)": r.gap_pct,
                "Verdict": score_label(r.gap_pct),
                "Mean P(off_route)": r.p_off_mean,
                "Mean P(route)": r.p_route_mean,
                "Mean P(loss)": r.p_loss_mean,
                "P(route) Min": r.p_route_min,
                "P(route) Max": r.p_route_max,
                "P(route) Std": r.p_route_std,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "berlin52_pre_post_comparison.csv")
    md_path = os.path.join(out_dir, "berlin52_pre_post_comparison.md")
    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Berlin52 Pre/Post Fix Comparison\n\n")
        f.write("## Cost Comparison\n\n")
        f.write("| Version | Dataset Class | Dataset | N | Baseline Cost | SVRAP (Full) Cost | Gap (%) | Verdict |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---|\n")
        for r in results:
            f.write(
                f"| {r.label} | Small | berlin52 | 52 | {r.baseline_cost:,.2f} | {r.guided_cost:,.2f} | {r.gap_pct:+.2f}% | {score_label(r.gap_pct)} |\n"
            )

        f.write("\n## Probability Diagnostics\n\n")
        f.write("| Version | Mean P(off_route) | Mean P(route) | Mean P(loss) | P(route) Min | P(route) Max | P(route) Std |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r.label} | {r.p_off_mean:.6f} | {r.p_route_mean:.6f} | {r.p_loss_mean:.6f} | {r.p_route_min:.6f} | {r.p_route_max:.6f} | {r.p_route_std:.6f} |\n"
            )

        f.write("\n## Generated Plots\n\n")
        for r in results:
            f.write(f"- berlin52_p_route_distribution_{r.tag}.png\n")
            f.write(f"- berlin52_spatial_heatmap_{r.tag}.png\n")
            f.write(f"- berlin52_cost_curve_{r.tag}.png\n")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison Markdown: {md_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join("formatted_dataset", "berlin52.txt")

    baseline_output = run_cmd([".\\svrap.exe", "7", dataset_path, "baseline"], cwd=root_dir)
    baseline_cost = parse_best_cost(baseline_output)
    print(f"Baseline cost: {baseline_cost:.2f}")

    pre = evaluate_variant(root_dir, dataset_path, "pre_fix", "Before Fix", baseline_cost)
    post = evaluate_variant(root_dir, dataset_path, "post_fix", "After Fix", baseline_cost)

    write_report(root_dir, [pre, post])


if __name__ == "__main__":
    main()
