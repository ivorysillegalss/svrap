import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Package SVRAP variant outputs into a comparison folder")
    parser.add_argument("--training-log", required=True, help="Path to training log csv")
    parser.add_argument("--attention-probs", default="attention_probs.csv", help="Path to attention probs csv")
    parser.add_argument("--backbone", default="backbone_indices.txt", help="Path to backbone indices file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--tag", required=True, help="Variant tag for plot titles")
    parser.add_argument("--final-greedy-cost", type=float, default=None, help="Optional final greedy cost to record")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_history = pd.read_csv(args.training_log)
    df_probs = pd.read_csv(args.attention_probs, header=None)
    df_probs.columns = ["x", "y", "p_off", "p_route"]

    out_attention = os.path.join(args.out_dir, "attention_probs.csv")
    df_probs.to_csv(out_attention, index=False, header=False)

    out_backbone = os.path.join(args.out_dir, "backbone_indices.txt")
    shutil.copyfile(args.backbone, out_backbone)

    out_log = os.path.join(args.out_dir, os.path.basename(args.training_log))
    shutil.copyfile(args.training_log, out_log)

    plt.figure(figsize=(12, 7))
    plt.plot(df_history["Epoch"], df_history["Cost"], label="Cost", alpha=0.6)
    plt.plot(df_history["Epoch"], df_history["Baseline"], label="Baseline (Greedy)", alpha=0.8)
    plt.plot(df_history["Epoch"], df_history["Best"], label="Best Cost", linewidth=2.0)
    plt.title(f"Training Cost Curve - Berlin52 [{args.tag}]")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "berlin52_cost_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.histplot(df_probs["p_route"].astype(float), bins=20, kde=True)
    plt.title(f"Distribution of Attention Probabilities (p_route) for berlin52 [{args.tag}]")
    plt.xlabel("Probability (p_route)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "berlin52_p_route_distribution.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(11, 8))
    sc = plt.scatter(
        df_probs["x"].astype(float),
        df_probs["y"].astype(float),
        c=df_probs["p_route"].astype(float),
        cmap="viridis",
        s=120,
        alpha=0.85,
    )
    plt.colorbar(sc, label="p_route")

    top = df_probs.assign(idx=np.arange(len(df_probs))).sort_values("p_route", ascending=False).head(5)
    for _, row in top.iterrows():
        plt.annotate(
            f"#{int(row.idx)}",
            (float(row.x), float(row.y)),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title(f"Spatial Distribution of p_route for berlin52 [{args.tag}]")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "berlin52_spatial_heatmap.png"), dpi=160)
    plt.close()

    stats = {
        "mean_route": float(df_probs["p_route"].astype(float).mean()),
        "std_route": float(df_probs["p_route"].astype(float).std()),
        "min_route": float(df_probs["p_route"].astype(float).min()),
        "max_route": float(df_probs["p_route"].astype(float).max()),
        "best_train_cost": float(df_history["Best"].min()),
    }
    if args.final_greedy_cost is not None:
        stats["final_greedy_cost"] = float(args.final_greedy_cost)

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.10f}\n")

    print("packaged", args.out_dir)
    print(stats)


if __name__ == "__main__":
    main()
