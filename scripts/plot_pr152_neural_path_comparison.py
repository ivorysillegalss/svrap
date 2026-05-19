#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for path comparison plotting. Install it with: pip install matplotlib"
    ) from exc


BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def run_cmd(cmd: List[str], cwd: Path, timeout: int) -> Tuple[int, str, float]:
    start = time.time()
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
        return proc.returncode, proc.stdout or "", time.time() - start
    except subprocess.TimeoutExpired as exc:
        stdout_obj = exc.stdout or ""
        if isinstance(stdout_obj, bytes):
            stdout_text = stdout_obj.decode("utf-8", errors="replace")
        else:
            stdout_text = str(stdout_obj)
        return 124, stdout_text + "\n[TIMEOUT]", time.time() - start


def parse_best_cost(output: str) -> float:
    matches = BEST_COST_PATTERN.findall(output or "")
    if not matches:
        return float("nan")
    try:
        return float(matches[-1])
    except ValueError:
        return float("nan")


def load_route(route_csv: Path) -> List[Tuple[int, int, int]]:
    if not route_csv.exists():
        raise FileNotFoundError(f"route output not found: {route_csv}")

    rows: List[Tuple[int, int, int]] = []
    with route_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                order = int(row["order"])
                x = int(row["x"])
                y = int(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((order, x, y))
    rows.sort(key=lambda item: item[0])
    return rows


def derive_route_csv(label_csv: Path) -> Path:
    stem = label_csv
    if stem.suffix == ".txt":
        stem = stem.with_suffix("")
    return stem.parent / f"{stem.name}_route.csv"


def prepare_neural_guidance(
    repo_root: Path,
    python_exe: str,
    solver_py: Path,
    dataset_path: Path,
    variant_tag: str,
    epochs: int,
    seed: int,
    timeout: int,
) -> Tuple[Path, str]:
    model_path = repo_root / "models" / f"svrap_best_model_{dataset_path.stem}_{variant_tag}.pth"
    train_flag = "--train" if not model_path.exists() else "--no-train"
    cmd = [
        python_exe,
        str(solver_py),
        "--dataset",
        str(dataset_path),
        train_flag,
        "--variant-tag",
        variant_tag,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
    ]
    code, output, elapsed = run_cmd(cmd, cwd=repo_root, timeout=timeout)
    if code != 0:
        raise RuntimeError(
            f"Failed to prepare neural guidance for {dataset_path.name} (code={code}, elapsed={elapsed:.1f}s).\n"
            f"Output:\n{output}"
        )

    attention_csv = repo_root / "attention_probs.csv"
    if not attention_csv.exists():
        raise FileNotFoundError(f"Expected attention_probs.csv after solver run: {attention_csv}")
    return attention_csv, output


def run_cpp_mode(
    repo_root: Path,
    exe_path: Path,
    alpha: float,
    dataset_path: Path,
    strategy: str,
    label_output: Path,
    route_output: Path,
    timeout: int,
) -> Tuple[float, str]:
    cmd = [
        str(exe_path),
        str(alpha),
        str(dataset_path),
        strategy,
        "20",
        "15",
        "2",
        "50",
        "0.0",
        str(label_output),
        str(route_output),
    ]
    code, output, elapsed = run_cmd(cmd, cwd=repo_root, timeout=timeout)
    if code != 0:
        raise RuntimeError(
            f"C++ solver failed for strategy={strategy} (code={code}, elapsed={elapsed:.1f}s).\n"
            f"Output:\n{output}"
        )
    return parse_best_cost(output), output


def plot_routes(
    dataset_name: str,
    coords: List[Tuple[int, int]],
    guided_route: List[Tuple[int, int, int]],
    no_nn_route: List[Tuple[int, int, int]],
    guided_cost: float,
    no_nn_cost: float,
    out_png: Path,
) -> None:
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True)

    def draw_panel(ax, title: str, route: List[Tuple[int, int, int]], line_color: str, cost: float) -> None:
        ax.scatter(
            xs,
            ys,
            c="#d1d5db",
            s=26,
            edgecolors="#9ca3af",
            linewidths=0.25,
            zorder=2,
        )

        if route:
            route_xy = [(x, y) for _, x, y in route]
            line_x = [x for x, _ in route_xy] + [route_xy[0][0]]
            line_y = [y for _, y in route_xy] + [route_xy[0][1]]
            ax.plot(line_x, line_y, color=line_color, linewidth=1.5, alpha=0.9, zorder=3)
            for order, x, y in route[:: max(1, len(route) // 14)]:
                ax.annotate(
                    str(order),
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=line_color,
                    zorder=4,
                )
            ax.scatter(
                [route_xy[0][0]],
                [route_xy[0][1]],
                s=95,
                marker="*",
                color="#111827",
                zorder=4,
                label="start",
            )

        ax.set_title(f"{title}\nBest cost = {cost:.2f} | route size = {len(route)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    draw_panel(axes[0], f"{dataset_name} - neural guidance on", guided_route, "#2563eb", guided_cost)
    draw_panel(axes[1], f"{dataset_name} - neural guidance off", no_nn_route, "#ea580c", no_nn_cost)

    fig.suptitle(f"{dataset_name} final path comparison on node coordinate plane", fontsize=15)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary_path: Path, guided_cost: float, no_nn_cost: float, guided_route: Path, no_nn_route: Path) -> None:
    lines = [
        f"dataset: {summary_path.stem.replace('_path_comparison', '')}",
        f"guided_cost: {guided_cost:.4f}",
        f"no_nn_cost: {no_nn_cost:.4f}",
        f"guided_route_csv: {guided_route}",
        f"no_nn_route_csv: {no_nn_route}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare neural-guided and no_nn final routes for a dataset and plot both paths on the x/y plane."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for svrap_solver.py")
    parser.add_argument("--solver", default=str(repo_root / "svrap_solver.py"), help="Path to svrap_solver.py")
    parser.add_argument("--exe", default=str(repo_root / "svrap.exe"), help="Path to svrap.exe")
    parser.add_argument("--dataset", default=str(repo_root / "main_dataset" / "berlin52.txt"), help="Dataset path")
    parser.add_argument("--variant-tag", default="berlin52_path_comparison", help="Model tag used to prepare p_route")
    parser.add_argument("--alpha", type=float, default=7.0, help="ALPHA parameter for the C++ solver")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs if the model does not exist")
    parser.add_argument("--seed", type=int, default=42, help="Seed for Python/Torch runs")
    parser.add_argument("--prep-timeout", type=int, default=7200, help="Timeout seconds for neural prep")
    parser.add_argument("--solve-timeout", type=int, default=3600, help="Timeout seconds for each C++ solve")
    parser.add_argument("--out-dir", default=str(repo_root / "results" / "berlin52_path_comparison"), help="Output directory")
    args = parser.parse_args()

    solver_py = Path(args.solver)
    exe_path = Path(args.exe)
    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)

    if not solver_py.exists():
        raise FileNotFoundError(f"Solver not found: {solver_py}")
    if not exe_path.exists():
        raise FileNotFoundError(f"C++ executable not found: {exe_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    attention_csv, prep_output = prepare_neural_guidance(
        repo_root=repo_root,
        python_exe=args.python,
        solver_py=solver_py,
        dataset_path=dataset_path,
        variant_tag=args.variant_tag,
        epochs=args.epochs,
        seed=args.seed,
        timeout=args.prep_timeout,
    )

    copied_attention_csv = out_dir / f"{dataset_path.stem}_attention_probs.csv"
    shutil.copy2(attention_csv, copied_attention_csv)

    guided_label = out_dir / "guided_labels.txt"
    guided_route_csv = derive_route_csv(guided_label)
    no_nn_label = out_dir / "no_nn_labels.txt"
    no_nn_route_csv = derive_route_csv(no_nn_label)

    guided_cost, guided_output = run_cpp_mode(
        repo_root=repo_root,
        exe_path=exe_path,
        alpha=args.alpha,
        dataset_path=dataset_path,
        strategy="full",
        label_output=guided_label,
        route_output=guided_route_csv,
        timeout=args.solve_timeout,
    )

    no_nn_cost, no_nn_output = run_cpp_mode(
        repo_root=repo_root,
        exe_path=exe_path,
        alpha=args.alpha,
        dataset_path=dataset_path,
        strategy="no_nn",
        label_output=no_nn_label,
        route_output=no_nn_route_csv,
        timeout=args.solve_timeout,
    )

    if not guided_route_csv.exists():
        raise FileNotFoundError(f"guided route output not found: {guided_route_csv}")
    if not no_nn_route_csv.exists():
        raise FileNotFoundError(f"no_nn route output not found: {no_nn_route_csv}")

    guided_route = load_route(guided_route_csv)
    no_nn_route = load_route(no_nn_route_csv)

    coords: List[Tuple[int, int]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if len(parts) < 2:
                continue
            try:
                coords.append((int(float(parts[0])), int(float(parts[1]))))
            except ValueError:
                continue

    if not coords:
        raise RuntimeError(f"No coordinates loaded from {dataset_path}")

    out_png = out_dir / f"{dataset_path.stem}_path_comparison.png"
    plot_routes(
        dataset_name=dataset_path.stem,
        coords=coords,
        guided_route=guided_route,
        no_nn_route=no_nn_route,
        guided_cost=guided_cost,
        no_nn_cost=no_nn_cost,
        out_png=out_png,
    )

    summary_path = out_dir / f"{dataset_path.stem}_path_comparison.md"
    write_summary(summary_path, guided_cost, no_nn_cost, guided_route_csv, no_nn_route_csv)

    print(f"Prepared neural guidance using: {prep_output.splitlines()[-1] if prep_output else 'solver output'}")
    print(f"Saved attention probabilities: {copied_attention_csv}")
    print(f"Saved guided route CSV: {guided_route_csv}")
    print(f"Saved no_nn route CSV: {no_nn_route_csv}")
    print(f"Saved comparison figure: {out_png}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())