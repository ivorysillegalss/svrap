import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
from scipy import stats

def analyze_ablation_results(file_path="../results/ablation_results.csv"):
    """分析消融实验结果"""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    output_file = "../results/ablation_summary_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, file=f, **kwargs)

        log_print("=" * 70)
        log_print("SVRAP 消融实验分析报告")
        log_print("=" * 70)

        log_print("\n1. 各策略各数据集统计 (Mean ± Std):")
        log_print("-" * 70)
        grouped = df.groupby(['Dataset', 'Strategy'])['BestCost'].agg(['mean', 'std', 'min', 'max'])
        log_print(grouped.to_string())

        # Pivot table for mean costs
        pivot_mean = df.pivot_table(index='Dataset', columns='Strategy', values='BestCost', aggfunc='mean')
        log_print("\n2. 成本矩阵 (Mean):")
        log_print("-" * 70)
        log_print(pivot_mean.to_string())
        
        # Calculate improvement over baseline
        if 'baseline' in df['Strategy'].unique():
            means = df.groupby(['Dataset', 'Strategy'])['BestCost'].mean().reset_index()
            baseline_costs = means[means['Strategy'] == 'baseline'][['Dataset', 'BestCost']].rename(columns={'BestCost': 'BaselineCost'})
            
            merged = means.merge(baseline_costs, on='Dataset')
            merged['Gap (%)'] = (merged['BestCost'] - merged['BaselineCost']) / merged['BaselineCost'] * 100
            
            log_print("\n3. 相对Baseline的改进 (%):")
            log_print("-" * 70)
            pivot_gap = merged.pivot(index='Dataset', columns='Strategy', values='Gap (%)')
            log_print(pivot_gap.to_string())
            
            # 汇总统计
            log_print("\n4. 平均改进率汇总:")
            log_print("-" * 70)
            avg_gaps = merged.groupby('Strategy')['Gap (%)'].mean()
            for strategy, gap in avg_gaps.items():
                status = "✓ 改进" if gap < 0 else "✗ 退化"
                log_print(f"  {strategy}: {gap:+.2f}% {status}")

        # 统计显著性检验
        if 'full' in df['Strategy'].unique() and 'baseline' in df['Strategy'].unique():
            log_print("\n5. 统计显著性检验 (full vs baseline):")
            log_print("-" * 70)
            
            datasets = df['Dataset'].unique()
            p_values = []
            
            for dataset in datasets:
                full_costs = df[(df['Dataset'] == dataset) & (df['Strategy'] == 'full')]['BestCost'].values
                baseline_costs = df[(df['Dataset'] == dataset) & (df['Strategy'] == 'baseline')]['BestCost'].values
                
                if len(full_costs) > 1 and len(baseline_costs) > 1:
                    # 配对t检验
                    min_len = min(len(full_costs), len(baseline_costs))
                    t_stat, p_val = stats.ttest_rel(full_costs[:min_len], baseline_costs[:min_len])
                    p_values.append(p_val)
                    sig = "**显著**" if p_val < 0.05 else "不显著"
                    log_print(f"  {dataset}: p={p_val:.4f} {sig}")
            
            if p_values:
                avg_p = np.mean(p_values)
                sig_count = sum(1 for p in p_values if p < 0.05)
                log_print(f"\n  平均p值: {avg_p:.4f}")
                log_print(f"  显著改进数据集: {sig_count}/{len(p_values)}")

        log_print("\n" + "=" * 70)
    
    print(f"\n报告已保存到 {output_file}")
    
    # 生成可视化图表
    try:
        generate_ablation_plots(df)
    except Exception as e:
        print(f"生成图表失败: {e}")

def generate_ablation_plots(df):
    """生成消融实验可视化图表"""
    output_dir = "../results"
    
    # 图1: 各策略平均成本对比
    plt.figure(figsize=(12, 6))
    means = df.groupby('Strategy')['BestCost'].mean().sort_values()
    colors = ['green' if s == 'full' else 'steelblue' for s in means.index]
    means.plot(kind='bar', color=colors)
    plt.title('Average Cost by Strategy')
    plt.ylabel('Mean Cost')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_strategy_comparison.png", dpi=150)
    plt.close()
    print(f"保存图表: {output_dir}/ablation_strategy_comparison.png")
    
    # 图2: 各数据集各策略箱线图
    plt.figure(figsize=(14, 8))
    df_sorted = df.sort_values('Dataset')
    sns.boxplot(data=df_sorted, x='Dataset', y='BestCost', hue='Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.title('Cost Distribution by Dataset and Strategy')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_boxplot.png", dpi=150)
    plt.close()
    print(f"保存图表: {output_dir}/ablation_boxplot.png")

def analyze_constraint_results(file_path="../results/constraint_sensitivity_results.csv"):
    """分析α参数敏感性结果"""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    df = pd.read_csv(file_path)
    
    output_file = "../results/constraint_analysis_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, file=f, **kwargs)
        
        log_print("=" * 70)
        log_print("SVRAP α参数敏感性分析报告")
        log_print("=" * 70)
        
        # 按α分组统计
        log_print("\n各α值下的平均成本:")
        log_print("-" * 70)
        alpha_stats = df.groupby('Alpha')['Cost'].agg(['mean', 'std', 'min', 'max'])
        log_print(alpha_stats.to_string())
        
        # 透视表
        pivot = df.pivot_table(index='Dataset', columns='Alpha', values='Cost', aggfunc='mean')
        log_print("\n成本矩阵 (Dataset × Alpha):")
        log_print("-" * 70)
        log_print(pivot.to_string())
    
    print(f"\n报告已保存到 {output_file}")
    
    # 生成图表
    try:
        plt.figure(figsize=(10, 6))
        alpha_means = df.groupby('Alpha')['Cost'].mean()
        alpha_means.plot(marker='o', linewidth=2, markersize=8)
        plt.xlabel('Alpha (α)')
        plt.ylabel('Average Cost')
        plt.title('Cost Sensitivity to Alpha Parameter')
        plt.grid(True, alpha=0.3)
        plt.savefig("../results/alpha_sensitivity.png", dpi=150)
        plt.close()
        print("保存图表: ../results/alpha_sensitivity.png")
    except Exception as e:
        print(f"生成图表失败: {e}")

def analyze_all():
    """分析所有可用的结果文件"""
    results_dir = "../results"
    
    print("=" * 70)
    print("SVRAP 结果综合分析")
    print("=" * 70)
    
    # 消融实验
    ablation_file = f"{results_dir}/ablation_results.csv"
    if os.path.exists(ablation_file):
        print("\n>>> 分析消融实验结果...")
        analyze_ablation_results(ablation_file)
    
    # α敏感性
    constraint_file = f"{results_dir}/constraint_sensitivity_results.csv"
    if os.path.exists(constraint_file):
        print("\n>>> 分析α参数敏感性结果...")
        analyze_constraint_results(constraint_file)
    
    # 稳定性测试
    stability_file = f"{results_dir}/stability_test_results.csv"
    if os.path.exists(stability_file):
        print("\n>>> 分析稳定性测试结果...")
        df = pd.read_csv(stability_file)
        summary = df.groupby('Dataset')['Cost'].agg(['mean', 'std', 'min', 'max'])
        summary['CV(%)'] = (summary['std'] / summary['mean']) * 100
        print(summary.to_string())
    
    print("\n" + "=" * 70)
    print("分析完成！")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if 'ablation' in file_path.lower():
            analyze_ablation_results(file_path)
        elif 'constraint' in file_path.lower():
            analyze_constraint_results(file_path)
        else:
            analyze_ablation_results(file_path)
    else:
        analyze_all()
