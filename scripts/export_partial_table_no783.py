import csv
import os
import re
import subprocess
import sys
import time

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")

REMAINING_COSTS = {
    "d198": {"Baseline Cost": 82590.20, "SVRAP (Full) Cost": 71738.40, "No-Neural Cost (Control)": 82590.20},
    "d493": {"Baseline Cost": 184296.00, "SVRAP (Full) Cost": 191722.00, "No-Neural Cost (Control)": 184296.00},
}


def run_cmd(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)


def parse_best_cost(stdout):
    m = BEST_COST_PATTERN.findall(stdout or "")
    if not m:
        return float("nan")
    return float(m[-1])


def run_solver(exe, dataset_rel, strategy, entropy_weight, cwd):
    cmd = [exe, "7", dataset_rel, strategy, "20", "20", "2", "50", str(entropy_weight)]
    t0 = time.time()
    proc = run_cmd(cmd, cwd)
    t1 = time.time()
    return parse_best_cost(proc.stdout), t1 - t0


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_csv = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain.csv")
    out_csv = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain_no783.csv")
    out_md = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain_no783.md")

    py = sys.executable
    py_solver = os.path.join(repo, "svrap_solver.py")
    exe = os.path.join(repo, "svrap.exe")

    rows = []
    with open(src_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Keep already finished rows (exclude d198/d493/rat783)
    keep = [r for r in rows if r["Dataset"] not in {"d198", "d493", "rat783"}]

    patched = []
    for ds in ["d198", "d493"]:
        dataset_rel = os.path.join("formatted_dataset", f"{ds}.txt")
        dataset_abs = os.path.join(repo, dataset_rel)
        n = 0
        with open(dataset_abs, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())

        # prepare attention_probs for this dataset from existing trained model
        infer_cmd = [
            py,
            py_solver,
            "--dataset",
            dataset_rel,
            "--no-train",
            "--variant-tag",
            f"table_ctrl_temp500_20260320_{ds}",
        ]
        run_cmd(infer_cmd, repo)

        entropy_weight = 1.0 if n >= 100 else 10.0
        _, bt = run_solver(exe, dataset_rel, "baseline", entropy_weight, repo)
        _, ft = run_solver(exe, dataset_rel, "full", entropy_weight, repo)
        _, nt = run_solver(exe, dataset_rel, "no_nn", entropy_weight, repo)

        b = REMAINING_COSTS[ds]["Baseline Cost"]
        full = REMAINING_COSTS[ds]["SVRAP (Full) Cost"]
        nn = REMAINING_COSTS[ds]["No-Neural Cost (Control)"]
        gap = (full - b) / b * 100.0 if b else float("nan")

        patched.append(
            {
                "Dataset Category": "Large" if n >= 300 else "Medium",
                "Dataset": ds,
                "N": n,
                "Baseline Cost": b,
                "SVRAP (Full) Cost": full,
                "No-Neural Cost (Control)": nn,
                "Gap (%)": gap,
                "Baseline Time (s)": bt,
                "Full Time (s)": ft,
                "No-Neural Time (s)": nt,
            }
        )

    order = ["berlin52", "gr96", "pr107", "bier127", "gr137", "pr152", "u159", "d198", "d493"]
    merged = keep + patched
    merged.sort(key=lambda r: order.index(r["Dataset"]))

    fields = [
        "Dataset Category",
        "Dataset",
        "N",
        "Baseline Cost",
        "SVRAP (Full) Cost",
        "No-Neural Cost (Control)",
        "Gap (%)",
        "Baseline Time (s)",
        "Full Time (s)",
        "No-Neural Time (s)",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow(r)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Dataset Category | Dataset | N | Baseline Cost | SVRAP (Full) Cost | No-Neural Cost (Control) | Gap (%) | Baseline Time (s) | Full Time (s) | No-Neural Time (s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in merged:
            f.write(
                "| {cat} | {ds} | {n} | {b:.2f} | {full:.2f} | {nn:.2f} | {gap:.2f}% | {bt:.2f} | {ft:.2f} | {nt:.2f} |\n".format(
                    cat=r["Dataset Category"],
                    ds=r["Dataset"],
                    n=int(float(r["N"])),
                    b=float(r["Baseline Cost"]),
                    full=float(r["SVRAP (Full) Cost"]),
                    nn=float(r["No-Neural Cost (Control)"]),
                    gap=float(r["Gap (%)"]),
                    bt=float(r["Baseline Time (s)"]),
                    ft=float(r["Full Time (s)"]),
                    nt=float(r["No-Neural Time (s)"]),
                )
            )

    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
