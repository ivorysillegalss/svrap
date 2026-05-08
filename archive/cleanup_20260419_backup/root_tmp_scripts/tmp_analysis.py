import pandas as pd
import os

d = r"C:\Users\chenz\OneDrive\桌面\预训练微调前结果"

main = pd.read_csv(os.path.join(d, "main_dataset_p_route_stats_20260416_212459.csv"))
sumdf = pd.read_csv(os.path.join(d, "guided_vs_no_nn_summary_20260416_212459.csv"))
raw = pd.read_csv(os.path.join(d, "guided_vs_no_nn_raw_20260416_212459.csv"))

merge = main[["dataset", "p_route_std", "p_route_mean", "p_route_q10", "p_route_q90"]].merge(
    sumdf[["dataset", "improve_pct_vs_no_nn_mean"]], on="dataset", how="inner"
)
merge["spread_q90_q10"] = merge["p_route_q90"] - merge["p_route_q10"]

c_std = merge["p_route_std"].corr(merge["improve_pct_vs_no_nn_mean"])
c_mean = merge["p_route_mean"].corr(merge["improve_pct_vs_no_nn_mean"])
c_spread = merge["spread_q90_q10"].corr(merge["improve_pct_vs_no_nn_mean"])

merge["improve_z"] = (
    (merge["improve_pct_vs_no_nn_mean"] - merge["improve_pct_vs_no_nn_mean"].mean())
    / merge["improve_pct_vs_no_nn_mean"].std(ddof=0)
)
merge["spread_z"] = (
    (merge["spread_q90_q10"] - merge["spread_q90_q10"].mean())
    / merge["spread_q90_q10"].std(ddof=0)
)

n = len(main)
frac1 = (main["p_route_std"] < 1e-6).mean()
frac2 = (main["p_route_std"] < 1e-5).mean()
frac3 = (main["p_route_std"] < 1e-4).mean()
spread = main["p_route_q90"] - main["p_route_q10"]
med_sp = spread.median()
max_sp = spread.max()

raw2 = raw.copy()
raw2["win"] = (raw2["guided_cost"] < raw2["no_nn_cost"]).astype(float)
rob = raw2.groupby("dataset").agg(
    runs=("run", "count"),
    win_rate=("win", "mean"),
    improve_mean=("improve_pct_vs_no_nn", "mean"),
    improve_std=("improve_pct_vs_no_nn", "std"),
).reset_index()

rob_rounded = rob.copy()
rob_rounded["win_rate"] = rob_rounded["win_rate"].round(6)
rob_rounded["improve_mean"] = rob_rounded["improve_mean"].round(6)
rob_rounded["improve_std"] = rob_rounded["improve_std"].round(6)

robust_win = rob_rounded.loc[rob_rounded["win_rate"] >= 0.8, "dataset"].tolist()
unstable = rob_rounded.loc[(rob_rounded["win_rate"] >= 0.4) & (rob_rounded["win_rate"] <= 0.6), "dataset"].tolist()
robust_loss = rob_rounded.loc[rob_rounded["win_rate"] <= 0.2, "dataset"].tolist()

print("CORRELATIONS")
print(f"corr(p_route_std,improve_mean)={c_std:.12f}")
print(f"corr(p_route_mean,improve_mean)={c_mean:.12f}")
print(f"corr(spread_q90_q10,improve_mean)={c_spread:.12f}")
print()

print("COLLAPSE")
print(f"fraction p_route_std <1e-6: {frac1:.12f} ({int((main['p_route_std'] < 1e-6).sum())}/{n})")
print(f"fraction p_route_std <1e-5: {frac2:.12f} ({int((main['p_route_std'] < 1e-5).sum())}/{n})")
print(f"fraction p_route_std <1e-4: {frac3:.12f} ({int((main['p_route_std'] < 1e-4).sum())}/{n})")
print(f"median(q90-q10): {med_sp:.12f}")
print(f"max(q90-q10): {max_sp:.12f}")
print()

print("ZSCORES dataset improve_z spread_z")
print(
    merge[["dataset", "improve_pct_vs_no_nn_mean", "spread_q90_q10", "improve_z", "spread_z"]]
    .sort_values("dataset")
    .to_string(index=False, float_format=lambda x: f"{x:.6f}")
)
print()

print("ROBUSTNESS per dataset")
print(rob_rounded.sort_values("dataset").to_string(index=False))
print()
print("ROBUST_WIN>=0.8:", robust_win)
print("UNSTABLE_0.4_to_0.6:", unstable)
print("ROBUST_LOSS<=0.2:", robust_loss)
