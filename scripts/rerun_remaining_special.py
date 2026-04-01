import csv
import os
import re
import subprocess
import sys
import time

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

TARGET_DATASETS = ["d198", "d493", "rat783"]
ORDER = ["berlin52", "gr96", "pr107", "bier127", "gr137", "pr152", "u159", "d198", "d493", "rat783"]


def run_cmd(cmd, cwd, timeout=None):
    try:
        return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, stdout=e.stdout or "")


def parse_best_cost(text):
    m = BEST_COST_PATTERN.findall(text or "")
    if not m:
        return float("nan")
    return float(m[-1])


def count_nodes(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def hyperparams_by_size(n):
    k_neighbors = 20
    tabu_len = 20
    div_times = 2
    entropy_weight = 10.0 if n < 100 else 1.0
    return k_neighbors, tabu_len, div_times, entropy_weight


def run_solver(exe_path, dataset_rel, strategy, k, tabu, div, entropy_w, cwd):
    cmd = [exe_path, "7", dataset_rel, strategy, str(k), str(tabu), str(div), "50", str(entropy_w)]
    t0 = time.time()
    proc = run_cmd(cmd, cwd=cwd, timeout=1800)
    return parse_best_cost(proc.stdout), time.time() - t0


def load_existing_rows(csv_path):
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    return rows


def write_outputs(rows, out_csv, out_md):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Dataset Category | Dataset | N | Baseline Cost | SVRAP (Full) Cost | No-Neural Cost (Control) | Gap (%) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['Dataset Category']} | {r['Dataset']} | {int(float(r['N']))} | {float(r['Baseline Cost']):.2f} | {float(r['SVRAP (Full) Cost']):.2f} | {float(r['No-Neural Cost (Control)']):.2f} | {float(r['Gap (%)']):.2f}% |\\n"
            )


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py = sys.executable
    py_solver = os.path.join(repo, "svrap_solver.py")
    exe = os.path.join(repo, "svrap.exe")

    out_csv = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain.csv")
    out_md = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain.md")

    existing = load_existing_rows(out_csv)
    keep = [r for r in existing if r.get("Dataset") not in TARGET_DATASETS]

    new_rows = []
    for ds in TARGET_DATASETS:
        dataset_rel = os.path.join("formatted_dataset", f"{ds}.txt")
        dataset_abs = os.path.join(repo, dataset_rel)
        n = count_nodes(dataset_abs)
        k, tabu, div, entropy_w = hyperparams_by_size(n)
        category = "Large" if n >= 300 else ("Medium" if n >= 100 else "Small")

        print(f"=== {ds} (N={n}) ===", flush=True)

        train_cmd = [
            py,
            py_solver,
            "--dataset", dataset_rel,
            "--train",
            "--epochs", "5000",
            "--variant-tag", f"table_ctrl_temp500_20260320_{ds}",
        ]
        train_proc = run_cmd(train_cmd, cwd=repo, timeout=None)
        if train_proc.returncode != 0:
            print(f"[WARN] train failed for {ds}", flush=True)

        baseline_cost, baseline_t = run_solver(exe, dataset_rel, "baseline", k, tabu, div, entropy_w, repo)
        full_cost, full_t = run_solver(exe, dataset_rel, "full", k, tabu, div, entropy_w, repo)
        nonn_cost, nonn_t = run_solver(exe, dataset_rel, "no_nn", k, tabu, div, entropy_w, repo)

        gap = float("nan")
        if baseline_cost == baseline_cost and baseline_cost != 0 and full_cost == full_cost:
            gap = (full_cost - baseline_cost) / baseline_cost * 100.0

        print(f"baseline={baseline_cost:.2f}, full={full_cost:.2f}, no_nn={nonn_cost:.2f}, gap={gap:.2f}%", flush=True)

        new_rows.append(
            {
                "Dataset Category": category,
                "Dataset": ds,
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

    merged = keep + new_rows
    order_index = {name: idx for idx, name in enumerate(ORDER)}
    merged.sort(key=lambda r: order_index.get(r["Dataset"], 999))

    write_outputs(merged, out_csv, out_md)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
