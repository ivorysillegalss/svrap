import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

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
    q10 = torch.quantile(x, torch.tensor(0.10)).item()
    q25 = torch.quantile(x, torch.tensor(0.25)).item()
    q50 = torch.quantile(x, torch.tensor(0.50)).item()
    q75 = torch.quantile(x, torch.tensor(0.75)).item()
    q90 = torch.quantile(x, torch.tensor(0.90)).item()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
    }


def run_inference_for_dataset(
    dataset_path: Path,
    model_path: Path,
    device: torch.device,
    topk_ratio: float,
) -> Dict[str, float]:
    env = SVRAPEnvironment(str(dataset_path)).to(device)
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

        k = max(2, int(env.n * topk_ratio))
        topk_indices = torch.topk(p_route, k=k).indices
        actions = torch.zeros(env.n, dtype=torch.long, device=device)
        actions[topk_indices] = 1

        greedy_cost, details = env.evaluate_solution(actions)

        row = {
            "dataset": dataset_path.stem,
            "n_nodes": env.n,
            "k_route": k,
            "greedy_cost": float(greedy_cost),
            "n_route": int(details["n_route"]),
            "n_assign": int(details["n_assign"]),
            "n_loss": int(details["n_loss"]),
        }
        row.update({f"p_route_{k}": v for k, v in p_stats.items()})
        return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch inference for all main_dataset instances using finetuned models and export p_route differentiation stats."
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--main-dataset-dir", type=str, default="main_dataset")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="svrap_finetuned_",
        help="Model filename prefix. Expected pattern: <prefix><dataset>.pth",
    )
    parser.add_argument("--output-csv", type=str, default="results/main_dataset_p_route_stats.csv")
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(args.repo_root).resolve()
    dataset_dir = (repo_root / args.main_dataset_dir).resolve()
    models_dir = (repo_root / args.models_dir).resolve()
    output_csv = (repo_root / args.output_csv).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows: List[Dict[str, float]] = []
    missing_models: List[str] = []

    dataset_files = sorted(dataset_dir.glob("*.txt"))
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found in: {dataset_dir}")

    for ds_path in dataset_files:
        dataset_name = ds_path.stem
        model_path = models_dir / f"{args.model_prefix}{dataset_name}.pth"
        if not model_path.exists():
            missing_models.append(str(model_path))
            print(f"[skip] missing model for {dataset_name}: {model_path.name}")
            continue

        row = run_inference_for_dataset(
            dataset_path=ds_path,
            model_path=model_path,
            device=device,
            topk_ratio=args.topk_ratio,
        )
        rows.append(row)

        print(
            f"[infer] {dataset_name}: "
            f"p_route(mean={row['p_route_mean']:.4f}, std={row['p_route_std']:.4f}, "
            f"min={row['p_route_min']:.4f}, max={row['p_route_max']:.4f}, "
            f"q10={row['p_route_q10']:.4f}, q90={row['p_route_q90']:.4f}), "
            f"greedy_cost={row['greedy_cost']:.2f}"
        )

    if not rows:
        raise RuntimeError("No inference rows produced. Check model naming and directories.")

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

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved p_route stats to: {output_csv}")

    if missing_models:
        print("\nMissing model files:")
        for p in missing_models:
            print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
