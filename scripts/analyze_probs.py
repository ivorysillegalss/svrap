import csv
import pandas as pd
import numpy as np

def analyze_attention_probs(file_path):
    print(f"Reading data from {file_path}...")
    
    # Check if file exists
    try:
        raw = pd.read_csv(file_path, header=None)
        if raw.shape[1] >= 5:
            # Legacy format: x, y, p_assign, p_route, p_loss
            raw = raw.iloc[:, :5]
            raw.columns = ['x', 'y', 'p_assign', 'p_route', 'p_loss']
            raw['p_off'] = raw['p_assign'] + raw['p_loss']
        elif raw.shape[1] >= 4:
            # Binary format: x, y, p_off, p_route
            raw = raw.iloc[:, :4]
            raw.columns = ['x', 'y', 'p_off', 'p_route']
            raw['p_assign'] = raw['p_off']
            raw['p_loss'] = 0.0
        else:
            raise ValueError(f"Unexpected attention_probs format with {raw.shape[1]} columns")
        df = raw
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Add Node ID
    df['node_id'] = df.index

    # Get p_route column
    p_route = df['p_route'].values
    
    # 1. Raw Probabilities (from Softmax in model)
    df['raw_p_route'] = p_route

    # 2. Normalized Probabilities (Sum to 1)
    # Interpretation: If we pick exactly one node based on this distribution, what is the prob?
    df['norm_prob'] = p_route / np.sum(p_route)

    # 3. Min-Max Scaling (Scale to 0-1 range)
    # Interpretation: Relative strength, where the weakest node is 0 and strongest is 1.
    min_p = np.min(p_route)
    max_p = np.max(p_route)
    if max_p > min_p:
        df['scaled_score'] = (p_route - min_p) / (max_p - min_p)
    else:
        df['scaled_score'] = 0.0

    # Sort by p_route descending
    df_sorted = df.sort_values(by='p_route', ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print(f"ANALYSIS OF P_ROUTE (Total Nodes: {len(df)})")
    print("="*80)
    print(f"{'Rank':<5} {'Node ID':<8} {'Coordinates':<15} {'Raw P(Route)':<15} {'Norm Prob':<15} {'Scaled (0-1)':<15}")
    print("-" * 80)

    for i, row in df_sorted.iterrows():
        coords = f"({int(row['x'])},{int(row['y'])})"
        # Mark top 10 with a star
        mark = "*" if i < 10 else " "
        print(f"{i+1:<5} {int(row['node_id']):<8} {coords:<15} {row['raw_p_route']:.6f} {mark}      {row['norm_prob']:.6f}        {row['scaled_score']:.6f}")

    print("-" * 80)
    print("Note: ")
    print("  - Raw P(Route): The probability output by the Softmax layer for class 'ROUTE'.")
    print("  - Norm Prob: The probability of this node being selected relative to others (sum=1).")
    print("  - Scaled (0-1): Min-Max scaling to highlight the strongest vs weakest nodes.")
    print("="*80)

    # Save the analyzed report
    output_file = "analyzed_p_route.csv"
    df_sorted[['node_id', 'x', 'y', 'raw_p_route', 'norm_prob', 'scaled_score']].to_csv(output_file, index=False)
    print(f"\nAnalyzed data saved to {output_file}")

if __name__ == "__main__":
    analyze_attention_probs("attention_probs.csv")
