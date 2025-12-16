import os
import math
import random
import time
import csv
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

# Suppress FutureWarnings from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Optional

# ==========================================
# 1. Configuration & Constants
# ==========================================

class SVRAPConfig:
    # Default dataset (51 points) - used as fallback
    RAW_DATA = [
        (37,52), (49,49), (52,64), (20,26), (40,30), (21,47), (17,63), (31,62), (52,33), (51,21),
        (42,41), (31,32), (5,25), (12,42), (36,16), (52,41), (27,23), (17,33), (13,13), (57,58),
        (62,42), (42,57), (16,57), (8,52), (7,38), (27,68), (30,48), (43,67), (58,48), (58,27),
        (37,69), (38,46), (46,10), (61,33), (62,63), (63,69), (32,22), (45,35), (59,15), (5,6),
        (10,17), (21,10), (5,64), (30,15), (39,10), (32,39), (25,32), (25,55), (48,28), (56,37),
        (30,40)
    ]

    # Model Hyperparameters
    EMBED_DIM = 128
    N_HEADS = 8
    LR = 1e-3
    EPOCHS = 2000
    EARLY_STOP_PATIENCE = 100
    MIN_EPOCHS = 100

    # SVRAP Problem Parameters
    PARAM_A = 7.0  # Control routing/allocation cost weight
    LAMBDA_TOUR = 1.0
    LAMBDA_ALLOC = 1.0
    
    # Dynamic Isolation Penalty
    USE_DYNAMIC_LAMBDA_ISOL = True
    
    ALLOW_ISOLATED_VERTICES = True
    ISOLATED_VERTEX_PENALTY = 100.0 

    # Backbone Construction
    TOP_K_ROUTE_RATIO = 0.2

    # Paths
    MODEL_DIR = "models"
    CSV_OUTPUT = "attention_probs.csv"
    BACKBONE_OUTPUT = "backbone_indices.txt"

    @staticmethod
    def get_model_path(dataset_name="default"):
        if not os.path.exists(SVRAPConfig.MODEL_DIR):
            os.makedirs(SVRAPConfig.MODEL_DIR)
        return os.path.join(SVRAPConfig.MODEL_DIR, f"svrap_best_model_{dataset_name}.pth")

# ==========================================
# 2. Environment
# ==========================================

class SVRAPEnvironment:
    def __init__(self, dataset_path: Optional[str] = None):
        self.locations = []
        self.original_locations = [] # Keep integer coordinates for export
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            with open(dataset_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            # Handle float strings by converting to float first
                            self.original_locations.append((int(float(parts[0])), int(float(parts[1]))))
                        except ValueError:
                            continue
        else:
            print("Using built-in RAW_DATA")
            self.original_locations = SVRAPConfig.RAW_DATA

        self.n = len(self.original_locations)
        if self.n == 0:
            raise ValueError(f"No valid data found in {dataset_path}")

        # Normalize coordinates to [0, 1]
        max_val = 0
        for x, y in self.original_locations:
            max_val = max(max_val, x, y)
        
        scale = max_val if max_val > 0 else 1.0
        self.locations = [(x/scale, y/scale) for x, y in self.original_locations]
        self.tensor_locs = torch.tensor(self.locations, dtype=torch.float32)

        # Calculate Distance Matrix
        self.dist_matrix = torch.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                d = math.sqrt((self.original_locations[i][0] - self.original_locations[j][0])**2 + 
                              (self.original_locations[i][1] - self.original_locations[j][1])**2)
                self.dist_matrix[i][j] = d

        # Calculate Cost Matrices based on PARAM_A
        # c_ij = a * l_ij
        # d_ij = (10 - a) * l_ij
        self.c_matrix = SVRAPConfig.PARAM_A * self.dist_matrix
        self.d_matrix = (10.0 - SVRAPConfig.PARAM_A) * self.dist_matrix

        # Dynamic lambda_isol
        if SVRAPConfig.USE_DYNAMIC_LAMBDA_ISOL:
            self.lambda_isol = 0.5 + 0.0004 * (SVRAPConfig.PARAM_A**2) * self.n
        else:
            self.lambda_isol = 2.0 
        
        print(f"Lambda Isolation: {self.lambda_isol:.4f}")

        # Pre-calculate Isolation Cost D_i = min_{j!=i} d_ij
        self.D_i = torch.zeros(self.n)
        for i in range(self.n):
            # Mask self distance with infinity
            d_row = self.d_matrix[i].clone()
            d_row[i] = float('inf')
            self.D_i[i] = torch.min(d_row)

    def to(self, device):
        """Move all environment tensors to the specified device."""
        self.tensor_locs = self.tensor_locs.to(device)
        self.dist_matrix = self.dist_matrix.to(device)
        self.c_matrix = self.c_matrix.to(device)
        self.d_matrix = self.d_matrix.to(device)
        self.D_i = self.D_i.to(device)
        return self

    def evaluate_solution(self, actions: torch.Tensor) -> Tuple[float, dict]:
        """
        actions: tensor of shape (N,) with values 0 (ASSIGN), 1 (ROUTE), 2 (LOSS)
        Returns: total_cost, details_dict
        """
        route_indices = (actions == 1).nonzero(as_tuple=True)[0]
        assign_indices = (actions == 0).nonzero(as_tuple=True)[0]
        loss_indices = (actions == 2).nonzero(as_tuple=True)[0]

        tour_cost = 0.0
        alloc_cost = 0.0
        isol_cost = 0.0
        penalty = 0.0

        # 1. Tour Cost
        if len(route_indices) > 0:
            # Approximate tour cost using the order of indices
            current_tour = route_indices.tolist()
            if len(current_tour) > 1:
                for k in range(len(current_tour)):
                    u = current_tour[k]
                    v = current_tour[(k + 1) % len(current_tour)]
                    tour_cost += self.c_matrix[u, v].item()
        else:
            # No route nodes
            if len(assign_indices) > 0:
                penalty += 1e5 # Invalid state: ASSIGN nodes need a ROUTE backbone

        # 2. Allocation Cost
        if len(assign_indices) > 0:
            if len(route_indices) > 0:
                for idx in assign_indices:
                    # Find min d_ij to any node in route_indices
                    d_vals = self.d_matrix[idx, route_indices]
                    min_d = torch.min(d_vals).item()
                    alloc_cost += min_d

        # 3. Isolation Cost
        for idx in loss_indices:
            isol_cost += self.D_i[idx].item() * self.lambda_isol
            if not SVRAPConfig.ALLOW_ISOLATED_VERTICES:
                penalty += SVRAPConfig.ISOLATED_VERTEX_PENALTY

        total_cost = (SVRAPConfig.LAMBDA_TOUR * tour_cost + 
                      SVRAPConfig.LAMBDA_ALLOC * alloc_cost + 
                      isol_cost + penalty)
        
        return total_cost, {
            "tour": tour_cost,
            "alloc": alloc_cost,
            "isol": isol_cost,
            "penalty": penalty,
            "n_route": len(route_indices),
            "n_assign": len(assign_indices),
            "n_loss": len(loss_indices)
        }

# ==========================================
# 3. Neural Network Modules
# ==========================================

class GatedEdgeFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc_edge = nn.Linear(2, embed_dim) # Input: (d_ij, c_ij)
        self.gate = nn.Linear(embed_dim, 1)
        self.proj = nn.Linear(embed_dim, 1) # Project to scalar bias

    def forward(self, edge_feat):
        # edge_feat: (N, N, 2)
        x = F.relu(self.fc_edge(edge_feat)) # (N, N, embed_dim)
        g = torch.sigmoid(self.gate(x))
        out = self.proj(x * g) # (N, N, 1)
        return out.squeeze(-1) # (N, N)

class SVRAPNetwork(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Node embedding
        self.node_embed = nn.Linear(2, embed_dim)
        
        # Edge fusion
        self.edge_fusion = GatedEdgeFusion(embed_dim)
        
        # Attention
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Output heads (logits for ASSIGN, ROUTE, LOSS)
        self.classifier = nn.Linear(embed_dim, 3)

    def forward(self, x, edge_feat):
        # x: (1, N, 2)
        # edge_feat: (1, N, N, 2)
        
        # 1. Node Embeddings
        h = self.node_embed(x) # (1, N, embed_dim)
        
        # 2. Edge Bias
        attn_bias = self.edge_fusion(edge_feat[0]) # (N, N)
        
        # 3. Self Attention
        # attn_mask in PyTorch MHA is additive if float
        attn_out, _ = self.mha(h, h, h, attn_mask=attn_bias)
        h = self.ln1(h + attn_out)
        
        # 4. FFN
        ffn_out = self.ffn(h)
        h = self.ln2(h + ffn_out)
        
        # 5. Output
        logits = self.classifier(h) # (1, N, 3)
        return logits

# ==========================================
# 4. Pipeline & Training
# ==========================================

def run_pipeline(train_model: bool = True, dataset_path: Optional[str] = None):
    # Determine dataset name for model saving
    if dataset_path:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    else:
        dataset_name = "default"
        
    # 1. Setup Environment
    env = SVRAPEnvironment(dataset_path)
    
    # Prepare Edge Features (d_ij, c_ij)
    edge_feat = torch.stack([env.d_matrix, env.c_matrix], dim=-1) # (N, N, 2)
    edge_feat = edge_feat / 100.0 # Simple scaling
    edge_feat = edge_feat.unsqueeze(0) # (1, N, N, 2)
    
    node_feat = env.tensor_locs.unsqueeze(0) # (1, N, 2)

    # 2. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env.to(device)
    
    model = SVRAPNetwork(SVRAPConfig.EMBED_DIM, SVRAPConfig.N_HEADS).to(device)
    
    model_path = SVRAPConfig.get_model_path(dataset_name)
    
    # 3. Training Loop
    if train_model:
        print(f"Starting Training for {dataset_name}...")
        optimizer = optim.Adam(model.parameters(), lr=SVRAPConfig.LR)
        
        best_cost = float('inf')
        best_actions = None
        no_improve_steps = 0
        
        # Baseline for REINFORCE
        avg_cost = 0.0
        alpha_baseline = 0.9
        
        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)
        
        for epoch in range(SVRAPConfig.EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            logits = model(node_feat, edge_feat) # (1, N, 3)
            logits = logits.squeeze(0) # (N, 3)
            
            # Sample actions
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            actions = dist.sample() # (N,)
            
            # Evaluate
            cost, _ = env.evaluate_solution(actions)
            
            # REINFORCE Loss
            log_probs = dist.log_prob(actions)
            
            if epoch == 0:
                avg_cost = cost
            
            advantage = cost - avg_cost
            loss = (log_probs * advantage).mean()
            
            loss.backward()
            optimizer.step()
            
            # Update baseline
            avg_cost = alpha_baseline * avg_cost + (1 - alpha_baseline) * cost
            
            # Track Best
            if cost < best_cost:
                best_cost = cost
                best_actions = actions.clone()
                no_improve_steps = 0
                # Save Model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_cost': best_cost,
                    'best_actions': best_actions,
                    'epoch': epoch
                }, model_path)
            else:
                no_improve_steps += 1
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost {cost:.2f}, Best {best_cost:.2f}, Avg {avg_cost:.2f}")
            
            # Early Stopping
            if epoch > SVRAPConfig.MIN_EPOCHS and no_improve_steps >= SVRAPConfig.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
                
        print(f"Training finished. Best Cost: {best_cost:.2f}")

    # 4. Inference & Export
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No model found. Using random initialization (not recommended).")

    model.eval()
    with torch.no_grad():
        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)
        logits = model(node_feat, edge_feat).squeeze(0)
        final_probs = F.softmax(logits, dim=-1) # (N, 3)
        
        # Extract Backbone
        p_route = final_probs[:, 1]
        
        # Sort by p_route descending
        sorted_indices = torch.argsort(p_route, descending=True)
        
        # Select Top K
        k = max(2, int(env.n * SVRAPConfig.TOP_K_ROUTE_RATIO))
        backbone_indices = sorted_indices[:k].tolist()
        
        # Construct Greedy Solution for Evaluation
        greedy_actions = torch.zeros(env.n, dtype=torch.long) # Default ASSIGN (0)
        greedy_actions[backbone_indices] = 1 # Set ROUTE
        
        greedy_cost, details = env.evaluate_solution(greedy_actions)
        
        print("\n=== Inference Results ===")
        print(f"Dataset: {dataset_name}")
        print(f"Greedy Backbone Size: {k}")
        print(f"Greedy Cost (Approx): {greedy_cost:.2f}")
        print(f"Details: {details}")
        
        # Export to CSV
        print(f"\nExporting to {SVRAPConfig.CSV_OUTPUT}...")
        with open(SVRAPConfig.CSV_OUTPUT, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header not strictly needed by C++ but good for debug
            # writer.writerow(["x", "y", "p_assign", "p_route", "p_loss", "is_backbone"])
            
            for i in range(env.n):
                x, y = env.original_locations[i]
                pa = final_probs[i, 0].item()
                pr = final_probs[i, 1].item()
                pl = final_probs[i, 2].item()
                is_bb = "YES" if i in backbone_indices else "NO"
                
                # Format: x, y, p_assign, p_route, p_loss
                writer.writerow([x, y, f"{pa:.6f}", f"{pr:.6f}", f"{pl:.6f}"])
                
                print(f"Node {i}: ({x},{y}) P(R)={pr:.4f} Backbone={is_bb}")

        # Export Backbone Indices
        with open(SVRAPConfig.BACKBONE_OUTPUT, 'w') as f:
            for idx in backbone_indices:
                f.write(f"{idx}\n")
        
        print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVRAP Solver with Policy Network")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset file (e.g., formatted_dataset/berlin52.txt)")
    parser.add_argument("--train", action="store_true", help="Force training even if model exists")
    parser.add_argument("--no-train", action="store_true", help="Skip training, only inference")
    
    args = parser.parse_args()
    
    dataset_path = args.dataset
    
    # Determine if we should train
    # Default: Train if model doesn't exist
    # If --train is set, force train
    # If --no-train is set, skip train
    
    if dataset_path:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    else:
        dataset_name = "default"
        
    model_path = SVRAPConfig.get_model_path(dataset_name)
    
    if args.train:
        should_train = True
    elif args.no_train:
        should_train = False
    else:
        should_train = not os.path.exists(model_path)
    
    run_pipeline(train_model=should_train, dataset_path=dataset_path)
