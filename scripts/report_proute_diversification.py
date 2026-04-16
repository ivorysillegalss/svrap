import argparse
import sys
from pathlib import Path
from typing import List, Optional

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


def greedy_topk_actions(probs: torch.Tensor, ratio: float) -> torch.Tensor:
    n = probs.size(0)
    k = max(2, int(n * ratio))
    topk_indices = torch.topk(probs[:, 1], k=k).indices
    actions = torch.zeros(n, dtype=torch.long, device=probs.device)
    actions[topk_indices] = 1
    return actions


def quantile(v: torch.Tensor, q: float) -> float:
    if v.numel() == 0:
        return 0.0
    return torch.quantile(v, q).item()


def parse_dataset_names(dataset_dir: Path, requested: Optional[List[str]]) -> List[str]:
    if requested:
        names = []
        for n in requested:
            names.append(n[:-4] if n.endswith(".txt") else n)
        return names
    return sorted(p.stem for p in dataset_dir.glob("*.txt"))


def resolve_model_path(
    dataset_name: str,
    shared_model_path: Optional[Path],
    model_path_template: str,
) -> Path:
    if shared_model_path is not None:
        return shared_model_path
    return Path(model_path_template.format(dataset=dataset_name))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report p_route diversification stats from trained model(s) on dataset(s)."
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--dataset-dir", type=str, default="main_dataset")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names, e.g. berlin52 d198")
    parser.add_argument(
        "--model-path-template",
        type=str,
        default="models/svrap_finetuned_{dataset}.pth",
        help="Template used when --shared-model-path is not set.",
    )
    parser.add_argument(
        "--shared-model-path",
        type=str,
        default="",
        help="If set, use the same model file for all datasets.",
    )
    parser.add_argument("--csv-out", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--title", type=str, default="", help="Optional title shown before table")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    shared_model_path = None
    if args.shared_model_path:
        shared_model_path = Path(args.shared_model_path)
        if not shared_model_path.is_absolute():
            shared_model_path = (repo_root / shared_model_path).resolve()
        if not shared_model_path.exists():
            raise FileNotFoundError(f"Shared model not found: {shared_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = parse_dataset_names(dataset_dir, args.datasets)

    if args.title:
        print(f"=== {args.title} ===")
    print(f"device={device}")
    print(
        "dataset,n,model,mean,std,min,p10,p50,p90,max,k,greedy_cost,n_route,n_assign,n_loss"
    )

    rows = []
    for name in names:
        dataset_path = dataset_dir / f"{name}.txt"
        if not dataset_path.exists():
            print(f"{name},NA,NA,SKIP_DATASET_NOT_FOUND")
            continue

        model_path = resolve_model_path(
            dataset_name=name,
            shared_model_path=shared_model_path,
            model_path_template=args.model_path_template,
        )
        if not model_path.is_absolute():
            model_path = (repo_root / model_path).resolve()
        if not model_path.exists():
            print(f"{name},NA,{model_path},SKIP_MODEL_NOT_FOUND")
            continue

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

            greedy_actions = greedy_topk_actions(probs, SVRAPConfig.TOP_K_ROUTE_RATIO)
            greedy_cost, details = env.evaluate_solution(greedy_actions)

            k = int(greedy_actions.sum().item())
            row = {
                "dataset": name,
                "n": env.n,
                "model": str(model_path),
                "mean": p_route.mean().item(),
                "std": p_route.std(unbiased=False).item(),
                "min": p_route.min().item(),
                "p10": quantile(p_route, 0.10),
                "p50": quantile(p_route, 0.50),
                "p90": quantile(p_route, 0.90),
                "max": p_route.max().item(),
                "k": k,
                "greedy_cost": greedy_cost,
                "n_route": details.get("n_route", 0),
                "n_assign": details.get("n_assign", 0),
                "n_loss": details.get("n_loss", 0),
            }
            rows.append(row)
            print(
                f"{row['dataset']},{row['n']},{row['model']},"
                f"{row['mean']:.6f},{row['std']:.6f},{row['min']:.6f},"
                f"{row['p10']:.6f},{row['p50']:.6f},{row['p90']:.6f},{row['max']:.6f},"
                f"{row['k']},{row['greedy_cost']:.2f},{row['n_route']},{row['n_assign']},{row['n_loss']}"
            )

    if args.csv_out:
        csv_path = Path(args.csv_out)
        if not csv_path.is_absolute():
            csv_path = (repo_root / csv_path).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write("dataset,n,model,mean,std,min,p10,p50,p90,max,k,greedy_cost,n_route,n_assign,n_loss\n")
            for r in rows:
                f.write(
                    f"{r['dataset']},{r['n']},{r['model']},"
                    f"{r['mean']:.6f},{r['std']:.6f},{r['min']:.6f},"
                    f"{r['p10']:.6f},{r['p50']:.6f},{r['p90']:.6f},{r['max']:.6f},"
                    f"{r['k']},{r['greedy_cost']:.2f},{r['n_route']},{r['n_assign']},{r['n_loss']}\n"
                )
        print(f"saved_csv={csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
