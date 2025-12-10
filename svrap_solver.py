import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os # 新增: 用于处理文件路径

# ==========================================
# 1. 配置与环境 (Configuration & Environment)
# ==========================================

class SVRAPConfig:
    RAW_DATA = """
37,52
49,49
52,64
20,26
40,30
21,47
17,63
31,62
52,33
51,21
42,41
31,32
5,25
12,42
36,16
52,41
27,23
17,33
13,13
57,58
62,42
42,57
16,57
8,52
7,38
27,68
30,48
43,67
58,48
58,27
37,69
38,46
46,10
61,33
62,63
63,69
32,22
45,35
59,15
5,6
10,17
21,10
5,64
30,15
39,10
32,39
25,32
25,55
48,28
56,37
30,40
"""
    # 强化学习参数
    SEED = 42
    EMBED_DIM = 128
    N_HEADS = 4
    LR = 1e-4
    EPOCHS = 1000
    
    # SVRAP 目标函数参数 (基于您的公式和业务定义)
    GAMMA = 2.0      # 分配成本系数 (C_ij = D_ij * GAMMA)
    LAMBDA_ISOL = 0.5  # **隔离成本权重因子 (λ_isol)**

    # **新增配置：模型保存路径**
    MODEL_PATH = "svrap_best_model.pth"

class SVRAPEnvironment:
    def __init__(self, raw_data):
        self.coords, self.norm_coords = self._parse_and_normalize(raw_data)
        self.n_nodes = len(self.coords)
        
        # 根据节点到中心点的距离模拟 D_i (固有隔离成本)
        center = self.coords.mean(dim=0)
        dist_to_center = torch.norm(self.coords - center, dim=1)
        max_dist = dist_to_center.max()
        if max_dist > 0:
            norm_dist = dist_to_center / max_dist
        else:
            norm_dist = torch.zeros_like(dist_to_center)

        self.node_isolation_cost = 5.0 + 15.0 * norm_dist # (N,)
        torch.manual_seed(SVRAPConfig.SEED)
        
        self.dist_matrix, self.cost_matrix = self._compute_matrices()
        
    def _parse_and_normalize(self, raw_data):
        coords = []
        for line in raw_data.strip().split('\n'):
            parts = line.strip().split(',')
            coords.append([float(parts[0]), float(parts[1])])
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        
        min_vals, _ = coords_tensor.min(dim=0)
        max_vals, _ = coords_tensor.max(dim=0)
        range_vals = max_vals - min_vals
        norm_coords = (coords_tensor - min_vals) / torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
        return coords_tensor, norm_coords

    def _compute_matrices(self):
        x = self.norm_coords
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        cost = dist * SVRAPConfig.GAMMA
        return dist.unsqueeze(0), cost.unsqueeze(0)

    def evaluate_solution(self, actions):
        """
        计算目标函数总成本 (最小化):
        总成本 = 路由成本 + 分配成本 + 隔离成本
        Actions: 0=Assign, 1=Route, 2=Loss (对应 v_i=1)
        """
        actions = actions.cpu().numpy()
        assign_indices = [i for i, a in enumerate(actions) if a == 0]
        route_indices = [i for i, a in enumerate(actions) if a == 1]
        loss_indices = [i for i, a in enumerate(actions) if a == 2] # 对应 v_i=1
        backbone_indices = route_indices
        
        d_mat = self.dist_matrix[0]
        c_mat = self.cost_matrix[0]
        
        # 1. 路由成本 (Route Cost)
        if len(backbone_indices) < 2:
            route_cost = 1000.0 
        else:
            route_cost = 0
            for k in range(len(backbone_indices)):
                u, v = backbone_indices[k], backbone_indices[(k + 1) % len(backbone_indices)]
                route_cost += d_mat[u, v].item()
            
        # 2. 分配成本 (Assignment Cost)
        assign_cost = 0
        if len(backbone_indices) > 0:
            for nb_idx in assign_indices:
                min_c_to_backbone = min([c_mat[nb_idx, b].item() for b in backbone_indices])
                assign_cost += min_c_to_backbone
        
        # 3. 隔离成本 (Isolation Cost)
        isolation_cost_sum = sum(self.node_isolation_cost[i].item() for i in loss_indices)
        isolation_cost = SVRAPConfig.LAMBDA_ISOL * isolation_cost_sum

        total_cost = route_cost + assign_cost + isolation_cost
        
        if len(assign_indices) > 0 and len(backbone_indices) == 0:
             total_cost += 1000.0
            
        return total_cost

# ==========================================
# 2. 模型架构 (Model Architecture) - 保持不变
# ==========================================

class GatedEdgeFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.proj = nn.Linear(2, 1) 
    def forward(self, d, c):
        feat = torch.stack([d, c], dim=-1)
        z = self.gate(feat)
        bias = self.proj(feat * z)
        return bias.squeeze(-1)

class SVRAPNetwork(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        dim = SVRAPConfig.EMBED_DIM
        self.node_emb = nn.Linear(2, dim)
        self.edge_fusion = GatedEdgeFusion(32)
        self.attn = nn.MultiheadAttention(dim, SVRAPConfig.N_HEADS, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3) 

    def forward(self, x, d, c):
        h = self.node_emb(x)
        attn_bias = self.edge_fusion(d, c)
        h_key_bias = h + attn_bias.mean(dim=-1).unsqueeze(-1) 
        h2 = self.norm1(h)
        attn_out, _ = self.attn(h2, h_key_bias, h_key_bias)
        h = h + attn_out
        h2 = self.norm2(h)
        h = h + self.ff(h2)
        logits = self.head(h)
        return logits

# ==========================================
# 3. 训练与运行流程 (Training & Workflow)
# ==========================================

def run_pipeline(train_model=True):
    # 1. 初始化
    torch.manual_seed(SVRAPConfig.SEED)
    env = SVRAPEnvironment(SVRAPConfig.RAW_DATA)
    model = SVRAPNetwork(env.n_nodes)
    
    x = env.norm_coords.unsqueeze(0)
    d = env.dist_matrix
    c = env.cost_matrix
    
    # --- 加载/训练逻辑 ---
    best_cost = float('inf')
    best_actions_tensor = None
    
    if os.path.exists(SVRAPConfig.MODEL_PATH) and not train_model:
        # **加载已保存的模型**
        print(f"✅ 发现已保存模型: {SVRAPConfig.MODEL_PATH}。跳过训练，直接加载...")
        checkpoint = torch.load(SVRAPConfig.MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_cost = checkpoint['best_cost']
        best_actions_tensor = checkpoint['best_actions_tensor']
        
    elif train_model:
        # **开始训练**
        optimizer = optim.Adam(model.parameters(), lr=SVRAPConfig.LR)
        print(f"🚀 开始训练: 节点数 {env.n_nodes}, 目标: 最小化总成本")
        
        baseline = 0
        model.train()
        for epoch in range(SVRAPConfig.EPOCHS):
            optimizer.zero_grad()
            logits = model(x, d, c)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
            cost = env.evaluate_solution(actions[0])
            
            if cost < best_cost:
                best_cost = cost
                # 保存最优状态
                best_actions_tensor = actions.clone().detach() 
                
                # **新增: 实时保存最佳模型**
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_cost': best_cost,
                    'best_actions_tensor': best_actions_tensor,
                }, SVRAPConfig.MODEL_PATH)
            
            reward = -cost
            if epoch == 0: baseline = reward
            else: baseline = 0.95 * baseline + 0.05 * reward
            
            advantage = reward - baseline
            log_probs = dist.log_prob(actions)
            loss = -(log_probs * advantage).mean()
            
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                route_count = (actions == 1).sum().item()
                loss_count = (actions == 2).sum().item()
                print(f"Epoch {epoch:04d} | Cost: {cost:.4f} | Best: {best_cost:.4f} | R/L Count: {route_count}/{loss_count}")
        
        print(f"\n🎉 训练完成。最优模型已保存到 {SVRAPConfig.MODEL_PATH}")

    # 4. 最终结果展示
    print("\n" + "="*70)
    print("最终结果展示 (Final Results)")
    print("="*70)
    print(f"最优总成本 (Best Cost): {best_cost:.4f}")

    # 确保使用保存的最优动作进行评估展示
    if best_actions_tensor is not None:
        final_actions = best_actions_tensor.squeeze().cpu().tolist()
        final_route = [i for i, a in enumerate(final_actions) if a == 1]
        final_loss = [i for i, a in enumerate(final_actions) if a == 2]
        
        print(f"最优解 - ROUTE 节点: {final_route}")
        print(f"最优解 - LOSS 节点:   {final_loss}")
    
    model.eval()
    with torch.no_grad():
        final_logits = model(x, d, c)
        final_probs = F.softmax(final_logits, dim=-1)
    
    final_probs_np = final_probs[0].cpu().numpy()
    
    print("\n" + "="*70)
    print("节点最终概率策略与最优解状态对比")
    print("="*70)
    print("ID | X,Y 坐标 | P_Assign | P_Route | P_Loss | 最优解状态")
    print("-" * 70)
    
    status_map = {0: 'ASSIGN', 1: 'ROUTE', 2: 'LOSS'}
    for i in range(env.n_nodes):
        p_assign, p_route, p_loss = final_probs_np[i]
        coord = env.coords[i].numpy()
        
        if best_actions_tensor is not None:
            action_status = status_map[final_actions[i]]
        else:
            action_status = "N/A"
        
        print(f"{i:2d} | {coord[0]:.0f},{coord[1]:.0f} | {p_assign:.4f} | {p_route:.4f} | {p_loss:.4f} | {action_status:10s}")
        
    print("-" * 70)


if __name__ == "__main__":
    # 第一次运行：进行训练并保存模型
    # run_pipeline(train_model=True) 
    
    # 第二次运行：直接加载已保存的模型，跳过训练
    # run_pipeline(train_model=False) 
    
    # 默认行为：如果文件存在则加载，否则训练
    if os.path.exists(SVRAPConfig.MODEL_PATH):
        run_pipeline(train_model=False)
    else:
        run_pipeline(train_model=True)