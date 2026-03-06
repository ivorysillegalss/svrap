import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import sys
import os

# Fix OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add parent directory to path to import svrap_solver
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the main class and function, assuming they are available
try:
    from svrap_solver import run_pipeline, SVRAPNetwork, SVRAPEnvironment, SVRAPConfig
except ImportError:
    # If SVRAPDataset changed name to SVRAPEnvironment (based on read_file), adjust here
    from svrap_solver import run_pipeline, SVRAPNetwork, SVRAPEnvironment, SVRAPConfig
    print("Imported SVRAPEnvironment instead of SVRAPDataset")

def visualize_berlin52():
    dataset_path = os.path.join(parent_dir, "formatted_dataset", "berlin52.txt")
    print(f"Running visualization for {dataset_path}...")
    
    # 1. Run pipeline 
    # run_pipeline in svrap_solver.py usually returns the trained model and environment/dataset
    try:
        # Based on typical usage, we force training to ensure the model is fresh and compatible
        # This will save the model to disk as well
        # We need to make sure we catch the return values correctly. 
        # Inspecting svrap_solver.py shows run_pipeline returns (model, env) if called as library?
        # Actually it seems to return nothing. Let's see if we can just load the model cleanly after pipeline runs.
        
        run_pipeline(train_model=True, dataset_path=dataset_path) 
        
        # Now instantiate model and env manually since run_pipeline doesn't return them
        env = SVRAPEnvironment(dataset_path)
        # Assuming SVRAPNetwork is the class name, we need to initialize it.
        # It usually takes input_dim, hidden_dim, but we need to know the defaults from config or code.
        # Let's peek at SVRAPNetwork definition or usage in svrap_solver.py.
        # But wait, run_pipeline saves the model. We can load it.
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        model_path = SVRAPConfig.get_model_path(dataset_name)
        
        # Initialize model structure (assuming default params from Config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SVRAPNetwork().to(device) 
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    except Exception as e:
        print(f"Pipeline or Loading failed: {e}")
        return
        # For now, let's assume it failed and we can't proceed without a model
        return

    # 2. Extract Probability Map (p_route)
    print("\n[Analysis] Extracting Probability Map (p_route)...")
    
    model.eval()
    
    # We need to pass the device, since model might not have .device attribute exposed directly 
    # (it is a nn.Module which doesn't have .device by default, users often add it)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Get coordinates from environment
        if hasattr(env, 'tensor_locs'):
            coords = env.tensor_locs.to(device).unsqueeze(0) # (1, N, 2)
        else:
            # Fallback if property name is different
            coords = torch.tensor(env.locations, dtype=torch.float32).to(device).unsqueeze(0)
            
        # Prepare Edge Features (d_ij, c_ij) - Matches svrap_solver.py logic
        # d_matrix and c_matrix should be available in env
        if hasattr(env, 'd_matrix') and hasattr(env, 'c_matrix'):
             d_matrix = env.d_matrix.to(device)
             c_matrix = env.c_matrix.to(device)
             
             d_max = d_matrix.max()
             c_max = c_matrix.max()
             
             d_norm = d_matrix / d_max if d_max > 0 else d_matrix
             c_norm = c_matrix / c_max if c_max > 0 else c_matrix
             
             edge_feat = torch.stack([d_norm, c_norm], dim=-1) # (N, N, 2)
             edge_feat = edge_feat.unsqueeze(0) # (1, N, N, 2)
        else:
             print("Error: Environment missing d_matrix or c_matrix")
             return

        # Forward pass
        # model(x, edge_feat) -> returns logits (Batch, N, 3)
        logits = model(coords, edge_feat)
        
        # Squeeze batch dim
        logits = logits.squeeze(0) # (N, 3)
        
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Extract p_route (Class 1)
        p_route = probs[:, 1]
            
        # Convert to numpy
        p_route_np = p_route.cpu().numpy() # (N,)
        
        # Create DataFrame
        # Use env.locations which are normalized, or original_locations for plotting?
        # Let's use env.locations for consistency with the model input
        locs_np = coords.squeeze().cpu().numpy()
        
        df_probs = pd.DataFrame({
            'Node_ID': range(len(p_route_np)),
            'P_Route': p_route_np,
            'X': locs_np[:, 0],
            'Y': locs_np[:, 1]
        })
        
        # Sort by P_Route descending
        df_sorted = df_probs.sort_values(by='P_Route', ascending=False).reset_index(drop=True)
        
        print("\n[Top 10 Nodes by P_Route]")
        print(df_sorted.head(10))
        
        print("\n[Bottom 5 Nodes by P_Route]")
        print(df_sorted.tail(5))
        
        # 3. Plot Distribution of P_Route
        plt.figure(figsize=(10, 6))
        sns.histplot(df_sorted['P_Route'], bins=20, kde=True)
        plt.title(f'Distribution of Attention Probabilities (p_route) for berlin52')
        plt.xlabel('Probability (p_route)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plot_path_dist = os.path.join(current_dir, 'berlin52_p_route_distribution.png')
        plt.savefig(plot_path_dist)
        print(f"Saved distribution plot to {plot_path_dist}")
        
        # 4. Plot Training History if available
        history_file = f"training_log_berlin52.csv"
        # Search in current directory, or parent directory (where svrap_solver might have written it)
        # But wait, svrap_solver writes to "current working directory" when run.
        # Since I am running from scripts/, it should be in scripts/
        if os.path.exists(history_file):
            print(f"Found training log: {history_file}")
            df_history = pd.read_csv(history_file)
            
            plt.figure(figsize=(10, 6))
            plt.plot(df_history['Epoch'], df_history['Cost'], label='Cost', alpha=0.5)
            plt.plot(df_history['Epoch'], df_history['Baseline'], label='Baseline (Greedy)', alpha=0.5)
            plt.plot(df_history['Epoch'], df_history['Best'], label='Best Cost', linewidth=2)
            plt.title('Training Cost Curve - Berlin52')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path_cost = os.path.join(current_dir, 'berlin52_cost_curve.png')
            plt.savefig(plot_path_cost)
            print(f"Saved cost curve to {plot_path_cost}")
        else:
            print("No training log found. Run svrap_solver.py or run_pipeline(train_model=True) to generate it.")

        # 4. Plot P_Route Heatmap on Coordinates
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(df_probs['X'], df_probs['Y'], c=df_probs['P_Route'], cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(sc, label='P_Route Probability')
        
        # Annotate top 5 nodes
        for i in range(5):
            node = df_sorted.iloc[i]
            plt.annotate(f"#{int(node['Node_ID'])}", (node['X'], node['Y']), 
                         xytext=(5, 5), textcoords='offset points')
            
        plt.title(f'Spatial Distribution of P_Route for berlin52')
        plt.xlabel('X (Normalized)')
        plt.ylabel('Y (Normalized)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plot_path_spatial = os.path.join(current_dir, 'berlin52_spatial_heatmap.png')
        plt.savefig(plot_path_spatial)
        print(f"Saved spatial heatmap to {plot_path_spatial}")

if __name__ == "__main__":
    visualize_berlin52()
