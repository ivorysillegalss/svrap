import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from svrap_solver import SVRAPConfig, SVRAPEnvironment, SVRAPNetwork

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


def run_cmd(cmd: List[str], timeout: int) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]"


def prepare_features(env: SVRAPEnvironment, device: torch.device):
    d_max = env.d_matrix.max()
    c_max = env.c_matrix.max()
    d_norm = env.d_matrix / d_max if d_max > 0 else env.d_matrix
    c_norm = env.c_matrix / c_max if c_max > 0 else env.c_matrix
    edge_feat = torch.stack([d_norm, c_norm], dim=-1).unsqueeze(0).to(device)
    node_feat = env.tensor_locs.unsqueeze(0).to(device)
    return node_feat, edge_feat


def export_attention_probs_for_dataset(
    dataset_path: Path,
    model_path: Path,
    attention_csv_path: Path,
    device: torch.device,
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

    attention_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with attention_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(env.n):
            x, y = env.original_locations[i]
            p_off = probs[i, 0].item()
            p_route = probs[i, 1].item()
            writer.writerow([x, y, f"{p_off:.6f}", f"{p_route:.6f}"])

    p_route = probs[:, 1].detach().cpu()
    return {
        "p_route_mean": p_route.mean().item(),
        "p_route_std": p_route.std(unbiased=False).item(),
        "p_route_min": p_route.min().item(),
        "p_route_max": p_route.max().item(),
        "n_nodes": env.n,
    }


def safe_mean(xs: List[float]) -> float:
    ys = [x for x in xs if x == x]
    return statistics.mean(ys) if ys else float("nan")


def safe_std(xs: List[float]) -> float:
    ys = [x for x in xs if x == x]
    if len(ys) <= 1:
        return 0.0 if len(ys) == 1 else float("nan")
    return statistics.stdev(ys)


def to_md_table(rows: List[Dict[str, object]], md_path: Path) -> None:
    headers = [
        "dataset",
        "runs",
        "guided_cost_mean",
        "guided_cost_std",
        "no_nn_cost_mean",
        "no_nn_cost_std",
        "difference_mean",
        "gap_pct_vs_no_nn_mean",
        "improve_pct_vs_no_nn_mean",
        "p_route_mean",
        "p_route_std",
    ]
    lines = []
    lines.append("# Guided vs no_nn Gap Summary")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                if h == "runs":
                    vals.append(str(int(v)))
                elif abs(v) >= 1000:
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare neural-guided mode (full) vs no_nn mode on main_dataset, "
            "and export difference/gap metrics."
        )
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--main-dataset-dir", type=str, default="main_dataset")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--model-prefix", type=str, default="svrap_finetuned_")
    parser.add_argument("--exe", type=str, default="svrap.exe")
    parser.add_argument("--alpha", type=float, default=7.0)
    parser.add_argument("--runs", type=int, default=1, help="C++ runs per dataset per mode")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--attention-csv", type=str, default="attention_probs.csv")
    parser.add_argument("--raw-out", type=str, default="results/guided_vs_no_nn_raw.csv")
    parser.add_argument("--summary-out", type=str, default="results/guided_vs_no_nn_summary.csv")
    parser.add_argument("--summary-md", type=str, default="results/guided_vs_no_nn_summary.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = Path(args.repo_root).resolve()
    dataset_dir = (repo_root / args.main_dataset_dir).resolve()
    models_dir = (repo_root / args.models_dir).resolve()
    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = (repo_root / exe_path).resolve()

    attention_csv_path = (repo_root / args.attention_csv).resolve()
    raw_out = (repo_root / args.raw_out).resolve()
    summary_out = (repo_root / args.summary_out).resolve()
    summary_md = (repo_root / args.summary_md).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")
    if not exe_path.exists():
        raise FileNotFoundError(f"C++ executable not found: {exe_path}")

    datasets = sorted(dataset_dir.glob("*.txt"))
    if not datasets:
        raise FileNotFoundError(f"No dataset files found in {dataset_dir}")

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using executable: {exe_path}")

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    missing_models: List[str] = []

    for ds_path in datasets:
        ds_name = ds_path.stem
        model_path = models_dir / f"{args.model_prefix}{ds_name}.pth"
        if not model_path.exists():
            missing_models.append(str(model_path))
            print(f"[skip] missing model: {model_path.name}")
            continue

        p_stats = export_attention_probs_for_dataset(
            dataset_path=ds_path,
            model_path=model_path,
            attention_csv_path=attention_csv_path,
            device=device,
        )

        guided_costs: List[float] = []
        no_nn_costs: List[float] = []

        for run_idx in range(1, args.runs + 1):
            cmd_full = [str(exe_path), str(args.alpha), str(ds_path), "full"]
            rc_full, out_full = run_cmd(cmd_full, timeout=args.timeout)
            full_cost = parse_best_cost(out_full)

            cmd_no_nn = [str(exe_path), str(args.alpha), str(ds_path), "no_nn"]
            rc_no_nn, out_no_nn = run_cmd(cmd_no_nn, timeout=args.timeout)
            no_nn_cost = parse_best_cost(out_no_nn)

            difference = full_cost - no_nn_cost
            gap_pct_vs_no_nn = (
                (difference / no_nn_cost) * 100.0 if no_nn_cost == no_nn_cost and abs(no_nn_cost) > 1e-12 else float("nan")
            )
            improve_pct_vs_no_nn = (
                ((no_nn_cost - full_cost) / no_nn_cost) * 100.0 if no_nn_cost == no_nn_cost and abs(no_nn_cost) > 1e-12 else float("nan")
            )

            guided_costs.append(full_cost)
            no_nn_costs.append(no_nn_cost)

            raw_rows.append(
                {
                    "dataset": ds_name,
                    "run": run_idx,
                    "full_return_code": rc_full,
                    "no_nn_return_code": rc_no_nn,
                    "guided_cost": full_cost,
                    "no_nn_cost": no_nn_cost,
                    "difference": difference,
                    "gap_pct_vs_no_nn": gap_pct_vs_no_nn,
                    "improve_pct_vs_no_nn": improve_pct_vs_no_nn,
                    "p_route_mean": p_stats["p_route_mean"],
                    "p_route_std": p_stats["p_route_std"],
                    "p_route_min": p_stats["p_route_min"],
                    "p_route_max": p_stats["p_route_max"],
                }
            )

        guided_mean = safe_mean(guided_costs)
        no_nn_mean = safe_mean(no_nn_costs)
        diff_mean = guided_mean - no_nn_mean
        gap_mean = (diff_mean / no_nn_mean) * 100.0 if no_nn_mean == no_nn_mean and abs(no_nn_mean) > 1e-12 else float("nan")
        improve_mean = ((no_nn_mean - guided_mean) / no_nn_mean) * 100.0 if no_nn_mean == no_nn_mean and abs(no_nn_mean) > 1e-12 else float("nan")

        summary_rows.append(
            {
                "dataset": ds_name,
                "runs": args.runs,
                "n_nodes": p_stats["n_nodes"],
                "guided_cost_mean": guided_mean,
                "guided_cost_std": safe_std(guided_costs),
                "no_nn_cost_mean": no_nn_mean,
                "no_nn_cost_std": safe_std(no_nn_costs),
                "difference_mean": diff_mean,
                "gap_pct_vs_no_nn_mean": gap_mean,
                "improve_pct_vs_no_nn_mean": improve_mean,
                "p_route_mean": p_stats["p_route_mean"],
                "p_route_std": p_stats["p_route_std"],
                "p_route_min": p_stats["p_route_min"],
                "p_route_max": p_stats["p_route_max"],
            }
        )

        print(
            f"[done] {ds_name}: guided={guided_mean:.2f}, no_nn={no_nn_mean:.2f}, "
            f"diff={diff_mean:.2f}, gap={gap_mean:.3f}%, improve={improve_mean:.3f}%"
        )

    if not raw_rows:
        raise RuntimeError("No result rows generated. Check model names and dataset paths.")

    raw_fields = [
        "dataset",
        "run",
        "full_return_code",
        "no_nn_return_code",
        "guided_cost",
        "no_nn_cost",
        "difference",
        "gap_pct_vs_no_nn",
        "improve_pct_vs_no_nn",
        "p_route_mean",
        "p_route_std",
        "p_route_min",
        "p_route_max",
    ]
    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows_sorted = sorted(summary_rows, key=lambda x: str(x["dataset"]))
    summary_fields = [
        "dataset",
        "runs",
        "n_nodes",
        "guided_cost_mean",
        "guided_cost_std",
        "no_nn_cost_mean",
        "no_nn_cost_std",
        "difference_mean",
        "gap_pct_vs_no_nn_mean",
        "improve_pct_vs_no_nn_mean",
        "p_route_mean",
        "p_route_std",
        "p_route_min",
        "p_route_max",
    ]
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows_sorted)

    to_md_table(summary_rows_sorted, summary_md)

    print(f"Saved raw results: {raw_out}")
    print(f"Saved summary CSV: {summary_out}")
    print(f"Saved summary MD: {summary_md}")

    if missing_models:
        print("\nMissing model files:")
        for p in missing_models:
            print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
