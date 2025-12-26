import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def analyze_results(file_path="../results/ablation_results.csv"):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    output_file = "../results/ablation_summary_report.txt"
    with open(output_file, "w") as f:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            # Write to file as well, converting args to string
            print(*args, file=f, **kwargs)

        log_print("Results Summary (Mean +/- Std):")
        grouped = df.groupby(['Dataset', 'Strategy'])[['BestCost', 'Time']].agg(['mean', 'std'])
        log_print(grouped)

        # Pivot table for mean costs
        pivot_mean = df.pivot_table(index='Dataset', columns='Strategy', values='BestCost', aggfunc='mean')
        log_print("\nMean Costs:")
        log_print(pivot_mean)
        
        # Calculate improvement over baseline
        if 'baseline' in df['Strategy'].unique():
            # Calculate mean cost per dataset/strategy first
            means = df.groupby(['Dataset', 'Strategy'])['BestCost'].mean().reset_index()
            
            baseline_costs = means[means['Strategy'] == 'baseline'][['Dataset', 'BestCost']].rename(columns={'BestCost': 'BaselineCost'})
            
            merged = means.merge(baseline_costs, on='Dataset')
            merged['Gap (%)'] = (merged['BestCost'] - merged['BaselineCost']) / merged['BaselineCost'] * 100
            
            log_print("\nImprovement over Baseline (%):")
            pivot_gap = merged.pivot(index='Dataset', columns='Strategy', values='Gap (%)')
            log_print(pivot_gap)
        # Calculate improvement over Full Model (%) for ablation studies
        if 'full' in df['Strategy'].unique():
            means = df.groupby(['Dataset', 'Strategy'])['BestCost'].mean().reset_index()
            full_costs = means[means['Strategy'] == 'full'][['Dataset', 'BestCost']].rename(columns={'BestCost': 'FullCost'})
            
            merged_full = means.merge(full_costs, on='Dataset')
            merged_full['Gap vs Full (%)'] = (merged_full['BestCost'] - merged_full['FullCost']) / merged_full['FullCost'] * 100
            
            log_print("\nGap vs Full Model (%):")
            pivot_gap_full = merged_full.pivot(index='Dataset', columns='Strategy', values='Gap vs Full (%)')
            log_print(pivot_gap_full)
    
    print(f"\nReport saved to {output_file}")
if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        analyze_results()
