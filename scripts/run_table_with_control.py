import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


@dataclass
class DatasetSpec:
    category: str
    name: str


DATASETS: List[DatasetSpec] = [
    DatasetSpec("Small", "berlin52"),
    DatasetSpec("Small", "gr96"),
    DatasetSpec("Medium", "pr107"),
    DatasetSpec("Medium", "bier127"),
    DatasetSpec("Medium", "gr137"),
    DatasetSpec("Medium", "pr152"),
    DatasetSpec("Medium", "u159"),
    DatasetSpec("Medium", "d198"),
    DatasetSpec("Large", "d493"),
    DatasetSpec("Large", "rat783"),
]


def count_nodes(dataset_path: str) -> int:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def hyperparams_by_size(n: int) -> Tuple[int, int, int, float]:
    # Keep aligned with previous final experiment defaults.
    k_neighbors = 20
    tabu_len = 20
    div_times = 2
    entropy_weight = 10.0 if n < 100 else 1.0
    return k_neighbors, tabu_len, div_times, entropy_weight


def parse_best_cost(stdout: str) -> float:
    matches = BEST_COST_PATTERN.findall(stdout)
    if not matches:
        return float("nan")
    return float(matches[-1])


def run_cmd(cmd: List[str], cwd: str, timeout_sec: int | None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        # Emulate CompletedProcess shape for downstream parsing.
        return subprocess.CompletedProcess(cmd, returncode=124, stdout=e.stdout or "")


def run_solver(
    exe_path: str,
    dataset_path: str,
    strategy: str,
    k_neighbors: int,
    tabu_len: int,
    div_times: int,
    entropy_weight: float,
    cwd: str,
) -> Tuple[float, float, str]:
    cmd = [
        exe_path,
        "7",
        dataset_path,
        strategy,
        str(k_neighbors),
        str(tabu_len),
        str(div_times),
        "50",
        str(entropy_weight),
    ]
    t0 = time.time()
    proc = run_cmd(cmd, cwd, timeout_sec=1800)
    elapsed = time.time() - t0
    best = parse_best_cost(proc.stdout)
    return best, elapsed, proc.stdout


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_exe = sys.executable
    py_solver = os.path.join(repo_root, "svrap_solver.py")
    cpp_exe = os.path.join(repo_root, "svrap.exe")

    if not os.path.exists(cpp_exe):
        raise FileNotFoundError(f"Missing solver executable: {cpp_exe}")

    out_dir = os.path.join(repo_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "table_with_control_20260320_temp500_retrain.csv")
    out_md = os.path.join(out_dir, "table_with_control_20260320_temp500_retrain.md")

    rows: List[Dict[str, object]] = []

    def flush_outputs() -> None:
        if not rows:
            return

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        with open(out_md, "w", encoding="utf-8") as f:
            f.write("| Dataset Category | Dataset | N | Baseline Cost | SVRAP (Full) Cost | No-Neural Cost (Control) | Gap (%) |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(
                    "| {cat} | {ds} | {n} | {b:.2f} | {full:.2f} | {nn:.2f} | {gap:.2f}% |\n".format(
                        cat=r["Dataset Category"],
                        ds=r["Dataset"],
                        n=r["N"],
                        b=r["Baseline Cost"],
                        full=r["SVRAP (Full) Cost"],
                        nn=r["No-Neural Cost (Control)"],
                        gap=r["Gap (%)"],
                    )
                )

    for spec in DATASETS:
        dataset_rel = os.path.join("formatted_dataset", f"{spec.name}.txt")
        dataset_abs = os.path.join(repo_root, dataset_rel)
        if not os.path.exists(dataset_abs):
            print(f"[SKIP] missing dataset: {dataset_rel}")
            continue

        n = count_nodes(dataset_abs)
        k, tabu, div, entropy_w = hyperparams_by_size(n)

        print(f"\n=== {spec.name} (N={n}) ===", flush=True)

        # Step 1: retrain and run inference with current solver parameters.
        infer_cmd = [
            py_exe,
            py_solver,
            "--dataset",
            dataset_rel,
            "--train",
            "--epochs",
            "5000",
            "--variant-tag",
            f"table_ctrl_temp500_20260320_{spec.name}",
        ]
        infer_proc = run_cmd(infer_cmd, repo_root, timeout_sec=None)
        if infer_proc.returncode != 0:
            print(f"[WARN] inference failed for {spec.name}, continue with current attention_probs", flush=True)

        # Step 2: baseline
        baseline_cost, baseline_t, _ = run_solver(
            cpp_exe,
            dataset_rel,
            "baseline",
            k,
            tabu,
            div,
            entropy_w,
            repo_root,
        )

        # Step 3: full (neural enabled)
        full_cost, full_t, _ = run_solver(
            cpp_exe,
            dataset_rel,
            "full",
            k,
            tabu,
            div,
            entropy_w,
            repo_root,
        )

        # Step 4: control variable - no neural init
        nonn_cost, nonn_t, _ = run_solver(
            cpp_exe,
            dataset_rel,
            "no_nn",
            k,
            tabu,
            div,
            entropy_w,
            repo_root,
        )

        gap = float("nan")
        if baseline_cost == baseline_cost and baseline_cost != 0 and full_cost == full_cost:
            gap = (full_cost - baseline_cost) / baseline_cost * 100.0

        print(
            f"baseline={baseline_cost:.2f}, full={full_cost:.2f}, no_nn={nonn_cost:.2f}, gap={gap:.2f}%",
            flush=True,
        )

        rows.append(
            {
                "Dataset Category": spec.category,
                "Dataset": spec.name,
                "N": n,
                "Baseline Cost": baseline_cost,
                "SVRAP (Full) Cost": full_cost,
                "No-Neural Cost (Control)": nonn_cost,
                "Gap (%)": gap,
                "Baseline Time (s)": baseline_t,
                "Full Time (s)": full_t,
                "No-Neural Time (s)": nonn_t,
            }
        )

        # Persist after each dataset to avoid losing progress on later failures.
        flush_outputs()

    if not rows:
        raise RuntimeError("No dataset results were produced; nothing to write.")

    flush_outputs()

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
