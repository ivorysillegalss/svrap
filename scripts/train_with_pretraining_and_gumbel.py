import argparse
import io
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from svrap_solver import SVRAPConfig, SVRAPEnvironment, SVRAPNetwork

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

PRETRAIN_FILES = [
    "att48.txt",
    "berlin52.txt",
    "bier127.txt",
    "ch130.txt",
    "ch150.txt",
    "d198.txt",
    "eil101.txt",
    "eil51.txt",
    "eil76.txt",
    "gr120.txt",
    "gr137.txt",
    "gr96.txt",
    "kroA100.txt",
    "kroA150.txt",
    "kroB100.txt",
    "kroB150.txt",
    "kroC100.txt",
    "kroD100.txt",
    "kroE100.txt",
    "lin105.txt",
    "pr107.txt",
    "pr124.txt",
    "pr136.txt",
    "pr144.txt",
    "pr152.txt",
    "pr76.txt",
    "rat99.txt",
    "rd100.txt",
    "st70.txt",
    "u159.txt",
]


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            if getattr(st, "closed", False):
                continue
            try:
                st.write(s)
                st.flush()
            except ValueError:
                # Can happen during interpreter teardown if a stream closes early.
                continue
        return len(s)

    def flush(self):
        for st in self._streams:
            if getattr(st, "closed", False):
                continue
            try:
                st.flush()
            except ValueError:
                # Benign at process shutdown for already-closed file handles.
                continue


def resolve_repo_root(repo_root_arg: str) -> Path:
    candidate = Path(repo_root_arg).resolve()
    if (candidate / "svrap_solver.py").exists():
        return candidate

    # Fallback: script parent is repository root in this project layout.
    if (REPO_DIR / "svrap_solver.py").exists():
        print(
            f"[warn] repo-root '{candidate}' does not look like project root; "
            f"fallback to '{REPO_DIR}'"
        )
        return REPO_DIR

    return candidate


def resolve_existing_dir(path_arg: str, repo_root: Path, must_exist: bool = True) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        resolved = p.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {resolved}")
        return resolved

    candidates = [
        (repo_root / p).resolve(),
        (Path.cwd() / p).resolve(),
        (REPO_DIR / p).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    final_path = (repo_root / p).resolve()
    if must_exist:
        raise FileNotFoundError(f"Directory not found: {final_path}")
    return final_path


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("inf")
    return float(matches[-1])


def load_binary_labels(label_path: Path, expected_n: int) -> List[int]:
    labels: List[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s not in {"0", "1"}:
                raise ValueError(f"Invalid label '{s}' in {label_path}")
            labels.append(int(s))
    if len(labels) != expected_n:
        raise ValueError(
            f"Label count mismatch for {label_path}: got {len(labels)}, expected {expected_n}"
        )
    return labels


def prepare_features(env: SVRAPEnvironment, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    d_max = env.d_matrix.max()
    c_max = env.c_matrix.max()
    d_norm = env.d_matrix / d_max if d_max > 0 else env.d_matrix
    c_norm = env.c_matrix / c_max if c_max > 0 else env.c_matrix
    edge_feat = torch.stack([d_norm, c_norm], dim=-1).unsqueeze(0).to(device)
    node_feat = env.tensor_locs.unsqueeze(0).to(device)
    return node_feat, edge_feat


def sample_actions_gumbel_topk(route_probs: torch.Tensor, k: int) -> torch.Tensor:
    eps = 1e-12
    u = torch.rand_like(route_probs).clamp_(min=eps, max=1.0 - eps)
    gumbel = -torch.log(-torch.log(u))
    scores = torch.log(route_probs.clamp_min(eps)) + gumbel
    topk_indices = torch.topk(scores, k=k).indices
    actions = torch.zeros(route_probs.size(0), dtype=torch.long, device=route_probs.device)
    actions[topk_indices] = 1
    return actions


def greedy_topk_actions(probs: torch.Tensor, ratio: float) -> torch.Tensor:
    n = probs.size(0)
    k = max(2, int(n * ratio))
    route_scores = probs[:, 1]
    topk_indices = torch.topk(route_scores, k=k).indices
    actions = torch.zeros(n, dtype=torch.long, device=probs.device)
    actions[topk_indices] = 1
    return actions


def get_counterfactual_interval(n_nodes: int) -> int:
    if n_nodes > 500:
        return 10
    if n_nodes > 200:
        return 5
    return 1


def compute_counterfactual_node_loss(
    env: SVRAPEnvironment,
    probs: torch.Tensor,
    base_actions: torch.Tensor,
    cf_temp: float,
) -> Tuple[torch.Tensor, float]:
    cf_gains = []
    for i in range(env.n):
        actions_on = base_actions.clone()
        actions_on[i] = 1
        cost_on, _ = env.evaluate_solution(actions_on)

        actions_off = base_actions.clone()
        actions_off[i] = 0
        cost_off, _ = env.evaluate_solution(actions_off)

        cf_gains.append(cost_off - cost_on)

    cf_gains_t = torch.tensor(cf_gains, dtype=torch.float32, device=probs.device)
    target_route_prob = torch.sigmoid(cf_gains_t / cf_temp)
    node_loss = F.binary_cross_entropy(probs[:, 1], target_route_prob)
    return node_loss, cf_gains_t.std(unbiased=False).item()


def select_dataset_files(base_dir: Path, requested: Optional[List[str]], default_files: Optional[List[str]]) -> List[str]:
    if requested:
        selected = []
        for name in requested:
            fname = name if name.endswith(".txt") else f"{name}.txt"
            p = base_dir / fname
            if not p.exists():
                raise FileNotFoundError(f"Requested dataset not found: {p}")
            selected.append(fname)
        return selected
    if default_files is not None:
        return list(default_files)
    return sorted(p.name for p in base_dir.glob("*.txt"))


def run_cpp_no_nn_once(
    exe_path: Path,
    alpha: float,
    dataset_path: Path,
    label_output_path: Path,
    timeout_sec: int,
) -> Tuple[float, str]:
    cmd = [
        str(exe_path),
        str(alpha),
        str(dataset_path),
        "no_nn",
        "20",
        "15",
        "2",
        "50",
        "1.0",
        str(label_output_path),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"C++ no_nn failed for {dataset_path.name} (code={proc.returncode})\n{proc.stdout}"
        )
    return parse_best_cost(proc.stdout), proc.stdout


def generate_labels_with_cpp(
    pretraining_dir: Path,
    labels_dir: Path,
    exe_path: Path,
    alpha: float,
    runs_per_graph: int,
    timeout_sec: int,
    pretrain_files: List[str],
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)

    for fname in pretrain_files:
        dataset_path = pretraining_dir / fname
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing pretraining dataset: {dataset_path}")

        final_label_path = labels_dir / f"{dataset_path.stem}.labels.txt"
        best_cost = float("inf")
        best_temp = None

        print(f"[label] {fname}: running C++ no_nn {runs_per_graph} times")
        for r in range(1, runs_per_graph + 1):
            temp_label_path = labels_dir / f"{dataset_path.stem}.run{r}.txt"
            cost, out_text = run_cpp_no_nn_once(
                exe_path=exe_path,
                alpha=alpha,
                dataset_path=dataset_path,
                label_output_path=temp_label_path,
                timeout_sec=timeout_sec,
            )

            if not temp_label_path.exists():
                # Most common cause: using an old svrap executable that does not support
                # the 9th CLI argument for label output.
                raise RuntimeError(
                    "Label file was not generated by C++ solver. "
                    f"Expected: {temp_label_path}\n"
                    "Likely cause: executable does not include label-export patch. "
                    "Please rebuild svrap.exe from current source (main.cpp/tabu_search.h updates).\n"
                    f"Solver output tail:\n{out_text[-1200:]}"
                )

            print(f"  run {r:02d}: best_cost={cost:.2f}")
            if cost < best_cost:
                best_cost = cost
                best_temp = temp_label_path

        if best_temp is None or not best_temp.exists():
            raise RuntimeError(f"Failed to generate labels for {fname}")

        shutil.copyfile(best_temp, final_label_path)
        print(f"[label] keep best for {fname}: cost={best_cost:.2f}, file={final_label_path}")


def build_pretraining_items(
    pretraining_dir: Path,
    labels_dir: Path,
    device: torch.device,
    pretrain_files: List[str],
):
    items = []
    for fname in pretrain_files:
        dataset_path = pretraining_dir / fname
        label_path = labels_dir / f"{dataset_path.stem}.labels.txt"
        env = SVRAPEnvironment(str(dataset_path)).to(device)
        labels = load_binary_labels(label_path, env.n)
        node_feat, edge_feat = prepare_features(env, device)
        fixed_k = max(2, int(env.n * SVRAPConfig.TOP_K_ROUTE_RATIO))
        cf_interval = get_counterfactual_interval(env.n)
        items.append((dataset_path.stem, env, node_feat, edge_feat, labels, fixed_k, cf_interval))
    return items


def run_supervised_plus_rl_pretraining(
    model: SVRAPNetwork,
    items,
    pretrain_epochs: int,
    lr: float,
    sup_weight: float,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, pretrain_epochs + 1):
        random.shuffle(items)
        total_loss = 0.0
        total_sup = 0.0
        total_cost = 0.0

        for _, env, node_feat, edge_feat, labels, _, _ in items:
            model.train()
            optimizer.zero_grad()

            logits = model(node_feat, edge_feat).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            labels_t = torch.tensor(labels, dtype=torch.long, device=probs.device)

            sup_loss = F.cross_entropy(logits, labels_t)

            # Pretraining uses pure supervised loss only.
            loss = sup_weight * sup_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_sup += float(sup_loss.item())
            with torch.no_grad():
                greedy_actions = greedy_topk_actions(probs, SVRAPConfig.TOP_K_ROUTE_RATIO)
                greedy_cost, _ = env.evaluate_solution(greedy_actions)
            total_cost += float(greedy_cost)

        n_graphs = len(items)
        print(
            f"[pretrain] epoch={epoch:03d} "
            f"loss={total_loss / n_graphs:.4f} "
            f"sup={total_sup / n_graphs:.4f} "
            f"greedy_cost={total_cost / n_graphs:.2f}"
        )


def run_gumbel_reinforce_finetune(
    pretrained_path: Path,
    main_dataset_dir: Path,
    output_dir: Path,
    finetune_epochs: int,
    lr: float,
    cf_weight: float,
    cf_temp: float,
    entropy_weight: float,
    device: torch.device,
    finetune_files: Optional[List[str]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(pretrained_path, map_location=device)
    base_state = ckpt["model_state_dict"]

    if finetune_files:
        main_files = [main_dataset_dir / x for x in finetune_files]
    else:
        main_files = sorted(main_dataset_dir.glob("*.txt"))
    if not main_files:
        raise FileNotFoundError(f"No .txt datasets found in {main_dataset_dir}")

    for dataset_path in main_files:
        dataset_name = dataset_path.stem
        print(f"\n[finetune] dataset={dataset_name}")

        env = SVRAPEnvironment(str(dataset_path)).to(device)
        node_feat, edge_feat = prepare_features(env, device)
        cf_interval = get_counterfactual_interval(env.n)

        model = SVRAPNetwork(SVRAPConfig.EMBED_DIM, SVRAPConfig.N_HEADS).to(device)
        model.load_state_dict(base_state)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, finetune_epochs + 1):
            model.train()
            optimizer.zero_grad()

            logits = model(node_feat, edge_feat).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            actions = dist.sample()
            cost, _ = env.evaluate_solution(actions)

            with torch.no_grad():
                greedy_actions = greedy_topk_actions(probs, SVRAPConfig.TOP_K_ROUTE_RATIO)
                baseline_cost, _ = env.evaluate_solution(greedy_actions)

            log_probs = dist.log_prob(actions)

            denom = max(abs(baseline_cost), 1.0)
            advantage = (cost - baseline_cost) / denom
            reinforce_loss = log_probs.mean() * advantage
            entropy = Categorical(probs).entropy().mean()

            should_compute_cf = (epoch % cf_interval == 0)
            if should_compute_cf:
                node_loss, cf_std = compute_counterfactual_node_loss(
                    env=env,
                    probs=probs,
                    base_actions=greedy_actions,
                    cf_temp=cf_temp,
                )
            else:
                node_loss = torch.tensor(0.0, device=probs.device)
                cf_std = 0.0

            loss = reinforce_loss + cf_weight * node_loss - entropy_weight * entropy
            loss.backward()
            optimizer.step()

            p_route = probs[:, 1].detach()
            print(
                f"[finetune:{dataset_name}] epoch={epoch:03d} "
                f"cost={cost:.2f} baseline={baseline_cost:.2f} "
                f"p_route(mean={p_route.mean().item():.4f}, "
                f"std={p_route.std(unbiased=False).item():.4f}, "
                f"min={p_route.min().item():.4f}, max={p_route.max().item():.4f}) "
                f"cf_interval={cf_interval} cf_step={should_compute_cf} "
                f"cf_loss={node_loss.item():.4f} cf_std={cf_std:.4f}"
            )

        save_path = output_dir / f"svrap_finetuned_{dataset_name}.pth"
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print(f"[finetune] saved: {save_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_cpp_exe(repo_root: Path, arg_exe: str) -> Path:
    candidate = Path(arg_exe)
    if candidate.exists():
        return candidate

    fallback = repo_root / "svrap.exe"
    if fallback.exists():
        return fallback

    cmake_fallback = repo_root / "build" / "app.exe"
    if cmake_fallback.exists():
        return cmake_fallback

    raise FileNotFoundError(
        "C++ solver executable not found. Build one first (e.g., svrap.exe or build/app.exe)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "1) Generate 0/1 pseudo-labels on 30 pretraining graphs via C++ no_nn (10 runs each, keep best); "
            "2) pretrain with pure supervised loss over all 30 graphs per epoch (no batching); "
            "3) fine-tune each main_dataset graph for 100 epochs with categorical REINFORCE + counterfactual."
        )
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--pretraining-dir", type=str, default="pretraining_dataset")
    parser.add_argument("--main-dataset-dir", type=str, default="main_dataset")
    parser.add_argument("--labels-dir", type=str, default="pretraining_dataset/labels_no_nn")
    parser.add_argument("--cpp-exe", type=str, default="svrap.exe")
    parser.add_argument("--alpha", type=float, default=7.0)
    parser.add_argument("--label-runs", type=int, default=10)
    parser.add_argument("--cpp-timeout", type=int, default=1800)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--sup-weight", type=float, default=1.0)
    parser.add_argument("--rl-weight", type=float, default=0.25)
    parser.add_argument("--cf-weight", type=float, default=1.0)
    parser.add_argument("--cf-temp", type=float, default=500.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-datasets", nargs="*", default=None, help="Optional subset for pretraining/label generation, e.g. berlin52 d198")
    parser.add_argument("--finetune-datasets", nargs="*", default=None, help="Optional subset for finetuning, e.g. berlin52")
    parser.add_argument("--log-file", type=str, default="", help="Optional log file path. If set, script logs to console and this file.")
    parser.add_argument("--skip-label-generation", action="store_true")

    args = parser.parse_args()
    repo_root = resolve_repo_root(args.repo_root)

    log_handle = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = (repo_root / log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        sys.stdout = _TeeStream(orig_stdout, log_handle)
        sys.stderr = _TeeStream(orig_stderr, log_handle)
        print(f"Logging to: {log_path}")

    try:
        pretraining_dir = resolve_existing_dir(args.pretraining_dir, repo_root, must_exist=True)
        main_dataset_dir = resolve_existing_dir(args.main_dataset_dir, repo_root, must_exist=True)
        labels_dir = resolve_existing_dir(args.labels_dir, repo_root, must_exist=False)
        output_dir = (repo_root / "models").resolve()
        pretrain_files = select_dataset_files(pretraining_dir, args.pretrain_datasets, PRETRAIN_FILES)
        finetune_files = select_dataset_files(main_dataset_dir, args.finetune_datasets, None)

        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        exe_path = resolve_cpp_exe(repo_root, args.cpp_exe)
        print(f"Using C++ solver: {exe_path}")

        if not args.skip_label_generation:
            generate_labels_with_cpp(
                pretraining_dir=pretraining_dir,
                labels_dir=labels_dir,
                exe_path=exe_path,
                alpha=args.alpha,
                runs_per_graph=args.label_runs,
                timeout_sec=args.cpp_timeout,
                pretrain_files=pretrain_files,
            )

        model = SVRAPNetwork(SVRAPConfig.EMBED_DIM, SVRAPConfig.N_HEADS).to(device)
        items = build_pretraining_items(pretraining_dir, labels_dir, device, pretrain_files)

        run_supervised_plus_rl_pretraining(
            model=model,
            items=items,
            pretrain_epochs=args.pretrain_epochs,
            lr=args.lr_pretrain,
            sup_weight=args.sup_weight,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        pretrained_path = output_dir / "svrap_pretrained_30graphs_sup_rl.pth"
        torch.save({"model_state_dict": model.state_dict()}, pretrained_path)
        print(f"Saved pretrained model: {pretrained_path}")

        run_gumbel_reinforce_finetune(
            pretrained_path=pretrained_path,
            main_dataset_dir=main_dataset_dir,
            output_dir=output_dir,
            finetune_epochs=args.finetune_epochs,
            lr=args.lr_finetune,
            cf_weight=args.cf_weight,
            cf_temp=args.cf_temp,
            entropy_weight=args.entropy_weight,
            device=device,
            finetune_files=finetune_files,
        )

        print("All done.")
        return 0
    finally:
        if log_handle is not None:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            if not log_handle.closed:
                log_handle.flush()
                log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
