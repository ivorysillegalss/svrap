import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for heatmap export. Install it with: pip install matplotlib"
    ) from exc

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from svrap_solver import SVRAPConfig, SVRAPEnvironment, SVRAPNetwork


def prepare_features(env: SVRAPEnvironment, device: torch.device):
    d_max = env.d_matrix.max()
    c_max = env.c_matrix.max()
    d_norm = env.d_matrix / d_max if d_max > 0 else env.d_matrix
    c_norm = env.c_matrix / c_max if c_max > 0 else env.c_matrix
    edge_feat = torch.stack([d_norm, c_norm], dim=-1).unsqueeze(0).to(device)
    node_feat = env.tensor_locs.unsqueeze(0).to(device)
    return node_feat, edge_feat


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().cpu()
    return {
        "p_route_mean": x.mean().item(),
        "p_route_std": x.std(unbiased=False).item(),
        "p_route_min": x.min().item(),
        "p_route_max": x.max().item(),
        "p_route_q10": torch.quantile(x, torch.tensor(0.10)).item(),
        "p_route_q25": torch.quantile(x, torch.tensor(0.25)).item(),
        "p_route_q50": torch.quantile(x, torch.tensor(0.50)).item(),
        "p_route_q75": torch.quantile(x, torch.tensor(0.75)).item(),
        "p_route_q90": torch.quantile(x, torch.tensor(0.90)).item(),
    }


def render_heatmap(
    dataset_name: str,
    coords: List[tuple],
    p_route: torch.Tensor,
    out_path: Path,
    cmap: str,
) -> None:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    vals = p_route.detach().cpu().numpy()

    plt.figure(figsize=(8, 6), dpi=140)
    sc = plt.scatter(xs, ys, c=vals, cmap=cmap, s=44, edgecolors="black", linewidths=0.2)
    cbar = plt.colorbar(sc)
    cbar.set_label("p_route", rotation=90)
    plt.title(f"{dataset_name} p_route heatmap")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_summary_csv(summary_csv: Path) -> List[Dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_summary_csv(summary_csv: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = [
        "dataset",
        "n_nodes",
        "k_route",
        "greedy_cost",
        "n_route",
        "n_assign",
        "n_loss",
        "p_route_mean",
        "p_route_std",
        "p_route_min",
        "p_route_max",
        "p_route_q10",
        "p_route_q25",
        "p_route_q50",
        "p_route_q75",
        "p_route_q90",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(v: str) -> str:
    try:
        f = float(v)
        if math.isnan(f):
            return "nan"
        if abs(f) >= 1000:
            return f"{f:.2f}"
        return f"{f:.6f}" if abs(f) < 1 else f"{f:.4f}"
    except Exception:
        return str(v)


def export_markdown_table(
    rows: List[Dict[str, str]],
    output_md: Path,
    output_heatmap_dir: Path,
) -> None:
    if not rows:
        raise RuntimeError("No rows available to export markdown table.")

    rows = sorted(rows, key=lambda x: x.get("dataset", ""))

    cols = [
        "dataset",
        "n_nodes",
        "k_route",
        "greedy_cost",
        "p_route_mean",
        "p_route_std",
        "p_route_min",
        "p_route_max",
        "p_route_q10",
        "p_route_q50",
        "p_route_q90",
        "n_assign",
        "n_loss",
        "heatmap",
    ]

    lines: List[str] = []
    lines.append("# main_dataset p_route Summary")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for r in rows:
        ds = r.get("dataset", "")
        heatmap_file = output_heatmap_dir / f"{ds}_p_route_heatmap.png"
        rel = heatmap_file.relative_to(output_md.parent)
        heatmap_md = f"![{ds}]({rel.as_posix()})" if heatmap_file.exists() else "N/A"

        vals = {
            "dataset": ds,
            "n_nodes": r.get("n_nodes", ""),
            "k_route": r.get("k_route", ""),
            "greedy_cost": format_value(r.get("greedy_cost", "")),
            "p_route_mean": format_value(r.get("p_route_mean", "")),
            "p_route_std": format_value(r.get("p_route_std", "")),
            "p_route_min": format_value(r.get("p_route_min", "")),
            "p_route_max": format_value(r.get("p_route_max", "")),
            "p_route_q10": format_value(r.get("p_route_q10", "")),
            "p_route_q50": format_value(r.get("p_route_q50", "")),
            "p_route_q90": format_value(r.get("p_route_q90", "")),
            "n_assign": r.get("n_assign", ""),
            "n_loss": r.get("n_loss", ""),
            "heatmap": heatmap_md,
        }

        line = "| " + " | ".join(str(vals[c]) for c in cols) + " |"
        lines.append(line)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate p_route heatmaps for main_dataset and export summary CSV to markdown table."
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--main-dataset-dir", type=str, default="main_dataset")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model-prefix", type=str, default="svrap_finetuned_")
    parser.add_argument("--summary-csv", type=str, default="results/main_dataset_p_route_stats.csv")
    parser.add_argument("--output-heatmap-dir", type=str, default="results/p_route_heatmaps")
    parser.add_argument("--output-md", type=str, default="results/main_dataset_p_route_summary.md")
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(args.repo_root).resolve()
    dataset_dir = (repo_root / args.main_dataset_dir).resolve()
    models_dir = (repo_root / args.models_dir).resolve()
    summary_csv = (repo_root / args.summary_csv).resolve()
    output_heatmap_dir = (repo_root / args.output_heatmap_dir).resolve()
    output_md = (repo_root / args.output_md).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    output_heatmap_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows: List[Dict[str, object]] = []
    missing_models: List[str] = []

    dataset_files = sorted(dataset_dir.glob("*.txt"))
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found in: {dataset_dir}")

    for ds_path in dataset_files:
        ds_name = ds_path.stem
        model_path = models_dir / f"{args.model_prefix}{ds_name}.pth"
        if not model_path.exists():
            missing_models.append(str(model_path))
            print(f"[skip] missing model for {ds_name}: {model_path.name}")
            continue

        env = SVRAPEnvironment(str(ds_path)).to(device)
        node_feat, edge_feat = prepare_features(env, device)

        model = SVRAPNetwork(SVRAPConfig.EMBED_DIM, SVRAPConfig.N_HEADS).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            logits = model(node_feat, edge_feat).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            p_route = probs[:, 1]
            p_stats = tensor_stats(p_route)

            k = max(2, int(env.n * args.topk_ratio))
            topk_idx = torch.topk(p_route, k=k).indices
            actions = torch.zeros(env.n, dtype=torch.long, device=device)
            actions[topk_idx] = 1
            greedy_cost, details = env.evaluate_solution(actions)

        heatmap_path = output_heatmap_dir / f"{ds_name}_p_route_heatmap.png"
        render_heatmap(
            dataset_name=ds_name,
            coords=env.original_locations,
            p_route=p_route,
            out_path=heatmap_path,
            cmap=args.cmap,
        )

        row: Dict[str, object] = {
            "dataset": ds_name,
            "n_nodes": env.n,
            "k_route": k,
            "greedy_cost": float(greedy_cost),
            "n_route": int(details["n_route"]),
            "n_assign": int(details["n_assign"]),
            "n_loss": int(details["n_loss"]),
        }
        row.update(p_stats)
        rows.append(row)

        print(
            f"[done] {ds_name}: heatmap={heatmap_path.name}, "
            f"p_route std={row['p_route_std']:.4f}, greedy_cost={row['greedy_cost']:.2f}"
        )

    if not rows:
        raise RuntimeError("No rows generated. Check model names and directories.")

    write_summary_csv(summary_csv, rows)
    csv_rows = load_summary_csv(summary_csv)
    export_markdown_table(csv_rows, output_md, output_heatmap_dir)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved markdown table: {output_md}")

    if missing_models:
        print("\nMissing model files:")
        for m in missing_models:
            print(f"- {m}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
