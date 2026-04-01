import csv
import os
import re
import subprocess
from typing import Dict, List

BEST_COST_PATTERN = re.compile(r"Best cost(?: for .*?)?=\s*([0-9eE+\-.]+)")


def parse_best_cost(stdout: str) -> float:
    m = BEST_COST_PATTERN.findall(stdout or "")
    if not m:
        return float("nan")
    return float(m[-1])


def run_solver(exe: str, dataset_rel: str, cwd: str) -> float:
    # Keep the same CLI shape as existing scripts, but paper_baseline internally locks
    # to paper-aligned settings and ignores override arguments.
    cmd = [exe, "7", dataset_rel, "paper_baseline", "20", "20", "2", "50", "1.0"]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return parse_best_cost(proc.stdout)


def fmt2(v: float) -> str:
    return f"{v:.2f}"


def main() -> None:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exe = os.path.join(repo, "svrap.exe")

    src_csv = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain_no783.csv")
    out_csv = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain_no783_paper_baseline.csv")
    out_md = os.path.join(repo, "results", "table_with_control_20260320_temp500_retrain_no783_paper_baseline.md")

    rows: List[Dict[str, str]] = []
    with open(src_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        ds = row["Dataset"]
        dataset_rel = os.path.join("formatted_dataset", f"{ds}.txt")
        new_base = run_solver(exe, dataset_rel, repo)

        row["Baseline Cost"] = fmt2(new_base)

        full_cost = float(row["SVRAP (Full) Cost"])
        gap = (full_cost - new_base) / new_base * 100.0 if new_base and new_base == new_base else float("nan")
        row["Gap (%)"] = fmt2(gap)
        row.pop("Baseline Time (s)", None)
        row.pop("Full Time (s)", None)
        row.pop("No-Neural Time (s)", None)

        print(f"{ds}: paper_baseline={new_base:.2f}, full={full_cost:.2f}, gap={gap:.2f}%")

    fields = [
        "Dataset Category",
        "Dataset",
        "N",
        "Baseline Cost",
        "SVRAP (Full) Cost",
        "No-Neural Cost (Control)",
        "Gap (%)",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Dataset Category | Dataset | N | Baseline Cost | SVRAP (Full) Cost | No-Neural Cost (Control) | Gap (%) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {cat} | {ds} | {n} | {b} | {full} | {nn} | {gap} |\n".format(
                    cat=r["Dataset Category"],
                    ds=r["Dataset"],
                    n=int(float(r["N"])),
                    b=r["Baseline Cost"],
                    full=fmt2(float(r["SVRAP (Full) Cost"])),
                    nn=fmt2(float(r["No-Neural Cost (Control)"])),
                    gap=r["Gap (%)"],
                )
            )

    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD: {out_md}")


if __name__ == "__main__":
    main()
