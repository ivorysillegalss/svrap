import argparse
import csv
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Suppress FutureWarnings from torch.load to keep logs cleaner.
warnings.filterwarnings("ignore", category=FutureWarning)


def _append_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _append_repo_root_to_path()

# Reuse core environment/model/config from existing solver.
from svrap_solver import SVRAPConfig, SVRAPEnvironment, SVRAPNetwork  # noqa: E402


def run_sparse_pipeline(
    dataset_path: str,
    train_model: bool,
    node_loss_interval: int,
) -> None:
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    if node_loss_interval < 1:
        raise ValueError("node_loss_interval must be >= 1")

    env = SVRAPEnvironment(dataset_path)

    d_max = env.d_matrix.max()
    c_max = env.c_matrix.max()
    d_norm = env.d_matrix / d_max if d_max > 0 else env.d_matrix
    c_norm = env.c_matrix / c_max if c_max > 0 else env.c_matrix

    edge_feat = torch.stack([d_norm, c_norm], dim=-1).unsqueeze(0)
    node_feat = env.tensor_locs.unsqueeze(0)

    random.seed(SVRAPConfig.SEED)
    torch.manual_seed(SVRAPConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SVRAPConfig.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env.to(device)

    model = SVRAPNetwork(SVRAPConfig.EMBED_DIM, SVRAPConfig.N_HEADS).to(device)
    model_path = SVRAPConfig.get_model_path(dataset_name)

    if SVRAPConfig.VARIANT_TAG:
        print(f"Variant Tag: {SVRAPConfig.VARIANT_TAG}")

    print(
        f"Training schedule for {dataset_name}: "
        f"node loss every {node_loss_interval} epochs; "
        "other epochs use pure REINFORCE"
    )

    if train_model:
        print(f"Starting Training for {dataset_name}...")
        optimizer = optim.Adam(model.parameters(), lr=SVRAPConfig.LR)

        best_cost = float("inf")
        best_actions: Optional[torch.Tensor] = None
        no_improve_steps = 0

        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)

        training_history = []
        node_loss_epoch_count = 0

        for epoch in range(SVRAPConfig.EPOCHS):
            model.train()
            optimizer.zero_grad()

            logits = model(node_feat, edge_feat).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            actions = dist.sample()

            cost, _ = env.evaluate_solution(actions)

            with torch.no_grad():
                k_baseline = min(env.n, max(2, int(env.n * SVRAPConfig.TOP_K_ROUTE_RATIO)))
                route_scores = probs[:, 1]
                topk_indices = torch.topk(route_scores, k=k_baseline).indices
                greedy_actions = torch.zeros(env.n, dtype=torch.long, device=probs.device)
                greedy_actions[topk_indices] = 1
                baseline_cost, _ = env.evaluate_solution(greedy_actions)

            log_probs = dist.log_prob(actions)
            denom = max(abs(baseline_cost), 1.0)
            advantage = (cost - baseline_cost) / denom
            reinforce_loss = log_probs.mean() * advantage

            node_loss = torch.tensor(0.0, device=probs.device)
            cf_gain_std = 0.0
            should_compute_node_loss = (epoch % node_loss_interval == 0)

            if should_compute_node_loss:
                node_loss_epoch_count += 1
                cf_gains = []
                cf_base_actions = greedy_actions.clone()
                for i in range(env.n):
                    actions_on = cf_base_actions.clone()
                    actions_on[i] = 1
                    cost_on, _ = env.evaluate_solution(actions_on)

                    actions_off = cf_base_actions.clone()
                    actions_off[i] = 0
                    cost_off, _ = env.evaluate_solution(actions_off)

                    cf_gains.append(cost_off - cost_on)

                cf_gains_t = torch.tensor(cf_gains, dtype=torch.float32, device=probs.device)
                cf_gain_std = cf_gains_t.std(unbiased=False).item()

                target_route_prob = torch.sigmoid(cf_gains_t / SVRAPConfig.COUNTERFACTUAL_SIGMOID_TEMP)
                node_loss = F.binary_cross_entropy(probs[:, 1], target_route_prob)

            entropy_bonus = dist.entropy().mean()

            # User requested schedule:
            # - 1 epoch: node-loss-enhanced objective
            # - middle 4 epochs: pure REINFORCE only
            if should_compute_node_loss:
                loss = (
                    SVRAPConfig.RL_LOSS_WEIGHT * reinforce_loss
                    + SVRAPConfig.NODE_LOSS_WEIGHT * node_loss
                    - SVRAPConfig.ENTROPY_BONUS_WEIGHT * entropy_bonus
                )
                step_mode = "node+rl"
            else:
                loss = reinforce_loss
                step_mode = "reinforce_only"

            loss.backward()
            optimizer.step()

            if cost < best_cost:
                best_cost = cost
                best_actions = actions.clone()
                no_improve_steps = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_cost": best_cost,
                        "best_actions": best_actions,
                        "epoch": epoch,
                    },
                    model_path,
                )
            else:
                no_improve_steps += 1

            training_history.append(
                (
                    epoch,
                    cost,
                    baseline_cost,
                    best_cost,
                    step_mode,
                    float(node_loss.item()),
                )
            )

            if epoch % 100 == 0:
                mean_route_prob = probs[:, 1].mean().item()
                std_route_prob = probs[:, 1].std(unbiased=False).item()
                print(
                    f"Epoch {epoch}: Cost {cost:.2f}, Best {best_cost:.2f}, "
                    f"Baseline {baseline_cost:.2f}, Mode {step_mode}, "
                    f"Mean p_route {mean_route_prob:.4f}, Std p_route {std_route_prob:.4f}, "
                    f"CF std {cf_gain_std:.2f}, NodeLoss {node_loss.item():.4f}"
                )

            if epoch > SVRAPConfig.MIN_EPOCHS and no_improve_steps >= SVRAPConfig.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Training finished. Best Cost: {best_cost:.2f}")
        print(f"Node-loss epochs executed: {node_loss_epoch_count}")

        history_path = SVRAPConfig.get_training_log_path(dataset_name)
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Cost", "Baseline", "Best", "Mode", "NodeLoss"])
            writer.writerows(training_history)
        print(f"Saved training history to {history_path}")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No model found. Using random initialization (not recommended).")

    model.eval()
    with torch.no_grad():
        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)
        logits = model(node_feat, edge_feat).squeeze(0)
        final_probs = F.softmax(logits, dim=-1)

        p_route = final_probs[:, 1]
        mean_off = final_probs[:, 0].mean().item()
        mean_route = final_probs[:, 1].mean().item()
        print(f"Mean probabilities -> off_route: {mean_off:.6f}, route: {mean_route:.6f}")
        print(
            f"p_route range -> min: {p_route.min().item():.6f}, "
            f"max: {p_route.max().item():.6f}, std: {p_route.std().item():.6f}"
        )

        sorted_indices = torch.argsort(p_route, descending=True)
        k = max(2, int(env.n * SVRAPConfig.TOP_K_ROUTE_RATIO))
        backbone_indices = sorted_indices[:k].tolist()

        greedy_actions = torch.zeros(env.n, dtype=torch.long)
        greedy_actions[backbone_indices] = 1
        greedy_cost, details = env.evaluate_solution(greedy_actions)

        print("\n=== Inference Results ===")
        print(f"Dataset: {dataset_name}")
        print(f"Greedy Backbone Size: {k}")
        print(f"Greedy Cost (Approx): {greedy_cost:.2f}")
        print(f"Details: {details}")

        print(f"\nExporting to {SVRAPConfig.CSV_OUTPUT}...")
        with open(SVRAPConfig.CSV_OUTPUT, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(env.n):
                x, y = env.original_locations[i]
                poff = final_probs[i, 0].item()
                pr = final_probs[i, 1].item()
                writer.writerow([x, y, f"{poff:.6f}", f"{pr:.6f}"])

        with open(SVRAPConfig.BACKBONE_OUTPUT, "w") as f:
            for idx in backbone_indices:
                f.write(f"{idx}\n")

        print("Export complete.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sparse node-loss test: compute node loss every N epochs, "
            "use pure REINFORCE on the other epochs."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "main_dataset" / "u159.txt"),
        help="Path to dataset file",
    )
    parser.add_argument("--train", action="store_true", help="Force training even if model exists")
    parser.add_argument("--no-train", action="store_true", help="Skip training, only inference")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--variant-tag", type=str, default="u159_sparse5", help="Variant tag for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--node-loss-interval",
        type=int,
        default=5,
        help="Compute node loss every N epochs (default: 5)",
    )
    parser.add_argument("--alpha", type=float, default=7.0, help="SVRAP parameter A")

    args = parser.parse_args()

    SVRAPConfig.EPOCHS = args.epochs
    SVRAPConfig.VARIANT_TAG = args.variant_tag.strip()
    SVRAPConfig.SEED = args.seed
    SVRAPConfig.PARAM_A = float(args.alpha)

    if args.train and args.no_train:
        print("[ERROR] --train and --no-train cannot be used together")
        return 1

    dataset_path = args.dataset
    dataset_name = Path(dataset_path).stem

    model_path = SVRAPConfig.get_model_path(dataset_name)
    if args.train:
        should_train = True
    elif args.no_train:
        should_train = False
    else:
        should_train = not os.path.exists(model_path)

    run_sparse_pipeline(
        dataset_path=dataset_path,
        train_model=should_train,
        node_loss_interval=args.node_loss_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())