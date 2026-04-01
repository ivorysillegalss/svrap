import argparse
import csv
import shutil
import math
import re
import subprocess
import sys
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


def run_cmd(cmd: List[str], cwd: Path, timeout: int) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
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


def parse_int_or_default(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_float_or_nan(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def is_valid_row(row: Dict[str, Any], runs: int) -> bool:
    run_idx = parse_int_or_default(row.get("run"), -1)
    if run_idx < 1 or run_idx > runs:
        return False
    full_cost = parse_float_or_nan(row.get("full_cost"))
    no_nn_cost = parse_float_or_nan(row.get("no_nn_cost"))
    return (not math.isnan(full_cost)) and (not math.isnan(no_nn_cost))


def load_existing_rows(
    csv_path: Path,
    runs: int,
    infer_dataset_names: List[str],
) -> Tuple[List[Dict[str, Any]], set]:
    clean_rows: List[Dict[str, Any]] = []
    existing_pairs = set()
    if not csv_path.exists():
        return clean_rows, existing_pairs

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_valid_row(row, runs):
                continue
            run_idx = parse_int_or_default(row.get("run"), -1)
            infer_name = str(row.get("infer_dataset") or "").strip()
            if run_idx < 1 or infer_name not in infer_dataset_names:
                continue
            key = (run_idx, infer_name)
            if key in existing_pairs:
                continue

            normalized = {
                "train_dataset": str(row.get("train_dataset") or "pr152"),
                "infer_dataset": infer_name,
                "run": run_idx,
                "seed": parse_int_or_default(row.get("seed"), 0),
                "variant_tag": str(row.get("variant_tag") or ""),
                "alpha": parse_float_or_nan(row.get("alpha")),
                "train_dataset_path": str(row.get("train_dataset_path") or ""),
                "infer_dataset_path": str(row.get("infer_dataset_path") or ""),
                "train_return_code": parse_int_or_default(row.get("train_return_code"), 0),
                "infer_return_code": parse_int_or_default(row.get("infer_return_code"), 0),
                "full_return_code": parse_int_or_default(row.get("full_return_code"), 0),
                "no_nn_return_code": parse_int_or_default(row.get("no_nn_return_code"), 0),
                "full_cost": parse_float_or_nan(row.get("full_cost")),
                "no_nn_cost": parse_float_or_nan(row.get("no_nn_cost")),
                "node_loss_interval": parse_int_or_default(row.get("node_loss_interval"), 5),
            }
            clean_rows.append(normalized)
            existing_pairs.add(key)
    return clean_rows, existing_pairs


def parse_infer_datasets(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_model_path(models_dir: Path, dataset_name: str, variant_tag: str) -> Path:
    suffix = f"_{variant_tag}" if variant_tag else ""
    return models_dir / f"svrap_best_model_{dataset_name}{suffix}.pth"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    models_dir = scripts_dir / "models"

    parser = argparse.ArgumentParser(
        description=(
            "Run transfer experiment for multiple runs: "
            "train on pr152 with sparse node-loss schedule, then infer on d493/rat783 without training"
        )
    )
    parser.add_argument("--python", default="python", help="Python executable")
    parser.add_argument(
        "--runner",
        default=str(scripts_dir / "run_u159_sparse_node_loss_test.py"),
        help="Path to single-run sparse node-loss script",
    )
    parser.add_argument(
        "--train-dataset",
        default=str(repo_root / "main_dataset" / "pr152.txt"),
        help="Training dataset path (default: pr152)",
    )
    parser.add_argument(
        "--infer-datasets",
        default="d493,rat783",
        help="Comma-separated inference-only dataset names under main_dataset",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "main_dataset"),
        help="Dataset directory for inference-only datasets",
    )
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to C++ executable")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA for C++ solver")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--runs", type=int, default=10, help="Total target runs")
    parser.add_argument("--seed-base", type=int, default=42, help="Effective seed = seed_base + run")
    parser.add_argument(
        "--variant-base",
        default="pr152_transfer_sparse5",
        help="Variant base tag. Effective tag = <variant-base>_r<run>",
    )
    parser.add_argument("--node-loss-interval", type=int, default=5, help="Compute node loss every N epochs")
    parser.add_argument(
        "--out",
        default=str(repo_root / "results" / "pr152_transfer_large_10runs.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--train-timeout", type=int, default=7200, help="Timeout for training command")
    parser.add_argument("--solve-timeout", type=int, default=1800, help="Timeout for each C++ solve")

    args = parser.parse_args()

    runner = Path(args.runner)
    train_dataset_path = Path(args.train_dataset)
    dataset_dir = Path(args.dataset_dir)
    exe_path = Path(args.exe)
    out_csv = Path(args.out)
    infer_dataset_names = parse_infer_datasets(args.infer_datasets)
    infer_dataset_paths = [dataset_dir / f"{name}.txt" for name in infer_dataset_names]

    if not runner.exists():
        print(f"[ERROR] Runner not found: {runner}", file=sys.stderr)
        return 1
    if not train_dataset_path.exists():
        print(f"[ERROR] Training dataset not found: {train_dataset_path}", file=sys.stderr)
        return 1
    if train_dataset_path.stem != "pr152":
        print(f"[ERROR] This workflow expects pr152 training dataset, got: {train_dataset_path.stem}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1
    if not infer_dataset_paths:
        print("[ERROR] --infer-datasets is empty", file=sys.stderr)
        return 1
    for p in infer_dataset_paths:
        if not p.exists():
            print(f"[ERROR] Inference dataset not found: {p}", file=sys.stderr)
            return 1
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}", file=sys.stderr)
        return 1
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}", file=sys.stderr)
        return 1

    fieldnames = [
        "train_dataset",
        "infer_dataset",
        "run",
        "seed",
        "variant_tag",
        "alpha",
        "train_dataset_path",
        "infer_dataset_path",
        "train_return_code",
        "infer_return_code",
        "full_return_code",
        "no_nn_return_code",
        "full_cost",
        "no_nn_cost",
        "node_loss_interval",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows, existing_pairs = load_existing_rows(
        csv_path=out_csv,
        runs=args.runs,
        infer_dataset_names=infer_dataset_names,
    )

    # Rewrite with clean valid rows only (drop NaN/incomplete duplicates).
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(existing_rows, key=lambda r: (int(r["run"]), str(r["infer_dataset"]))):
            writer.writerow(row)

    target_pairs = []
    for run_idx in range(1, args.runs + 1):
        for infer_name in infer_dataset_names:
            target_pairs.append((run_idx, infer_name))

    missing_pairs = [p for p in target_pairs if p not in existing_pairs]
    missing_runs = sorted({run_idx for run_idx, _ in missing_pairs})

    if not missing_runs:
        print(f"[SKIP] all target pairs already completed in {out_csv}")
        return 0

    print(
        f"[RESUME] completed_pairs={len(existing_pairs)}, missing_pairs={len(missing_pairs)}, "
        f"target_runs={args.runs}, infer_datasets={infer_dataset_names}, out={out_csv}"
    )

    for run_idx in missing_runs:
        seed = args.seed_base + run_idx
        variant_tag = f"{args.variant_base}_r{run_idx}"

        print(f"[RUN {run_idx}/{args.runs}] seed={seed} variant={variant_tag} train=pr152")

        train_cmd = [
            args.python,
            str(runner),
            "--train",
            "--dataset",
            str(train_dataset_path),
            "--epochs",
            str(args.epochs),
            "--node-loss-interval",
            str(args.node_loss_interval),
            "--variant-tag",
            variant_tag,
            "--seed",
            str(seed),
            "--alpha",
            str(args.alpha),
        ]
        code_train, _ = run_cmd(train_cmd, cwd=scripts_dir, timeout=args.train_timeout)
        if code_train != 0:
            print(f"  [WARN] Train failed (code={code_train})")

        src_model = build_model_path(models_dir=models_dir, dataset_name="pr152", variant_tag=variant_tag)
        if not src_model.exists():
            print(f"  [WARN] Source model missing after training: {src_model}")

        for infer_name, infer_path in zip(infer_dataset_names, infer_dataset_paths):
            if (run_idx, infer_name) in existing_pairs:
                continue

            dst_model = build_model_path(models_dir=models_dir, dataset_name=infer_name, variant_tag=variant_tag)
            if src_model.exists():
                shutil.copyfile(src_model, dst_model)

            infer_cmd = [
                args.python,
                str(runner),
                "--no-train",
                "--dataset",
                str(infer_path),
                "--epochs",
                str(args.epochs),
                "--node-loss-interval",
                str(args.node_loss_interval),
                "--variant-tag",
                variant_tag,
                "--seed",
                str(seed),
                "--alpha",
                str(args.alpha),
            ]
            code_infer, _ = run_cmd(infer_cmd, cwd=scripts_dir, timeout=args.solve_timeout)

            full_cmd = [str(exe_path), str(args.alpha), str(infer_path), "full"]
            code_full, out_full = run_cmd(full_cmd, cwd=scripts_dir, timeout=args.solve_timeout)
            full_cost = parse_best_cost(out_full)

            no_nn_cmd = [str(exe_path), str(args.alpha), str(infer_path), "no_nn"]
            code_no_nn, out_no_nn = run_cmd(no_nn_cmd, cwd=scripts_dir, timeout=args.solve_timeout)
            no_nn_cost = parse_best_cost(out_no_nn)

            row = {
                "train_dataset": "pr152",
                "infer_dataset": infer_name,
                "run": run_idx,
                "seed": seed,
                "variant_tag": variant_tag,
                "alpha": args.alpha,
                "train_dataset_path": str(train_dataset_path),
                "infer_dataset_path": str(infer_path),
                "train_return_code": code_train,
                "infer_return_code": code_infer,
                "full_return_code": code_full,
                "no_nn_return_code": code_no_nn,
                "full_cost": full_cost,
                "no_nn_cost": no_nn_cost,
                "node_loss_interval": args.node_loss_interval,
            }

            with out_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            existing_pairs.add((run_idx, infer_name))
            print(
                f"  [OK] infer={infer_name} run={run_idx} full={full_cost} no_nn={no_nn_cost}"
            )

    print(f"Done. Output: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())