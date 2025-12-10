import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os 

# ==========================================
# 1. 配置与环境 (Configuration & Environment)
# ==========================================

class SVRAPConfig:
    # 客户坐标数据 (51个客户，对应 n=51)
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
    
    # **核心 SVRAP 问题参数 (基于原文)**
    PARAM_A = 7.0  # 偏向因子 a (影响 c_ij, d_ij 和 lambda_isol)
    
    # 权重因子 (基于原文 - 允许孤立顶点)
    LAMBDA_TOUR = 1.0  # lambda_tour = 1
    LAMBDA_ALLOC = 1.0 # lambda_alloc = 1
    # LAMBDA_ISOL 将在环境初始化时动态计算: 0.5 + 0.0004 * a^2 * n

    # 贪心/模型配置
    TOP_K_ROUTE_RATIO = 0.2  # 选取 P_Route 最高的 20% 节点作为初始骨干
    MODEL_PATH = "svrap_best_model.pth"

class SVRAPEnvironment:
    def __init__(self, raw_data):
        self.coords, self.norm_coords = self._parse_and_normalize(raw_data)
        self.n_nodes = len(self.coords) # 包含所有客户节点
        self.param_a = SVRAPConfig.PARAM_A
        torch.manual_seed(SVRAPConfig.SEED)
        
        # 1. 计算所有矩阵 (距离, 路由成本, 分配成本)
        # 这里的距离 l_ij 对应代码中的 dist
        self.dist_matrix, self.route_cost_matrix, self.alloc_cost_matrix = self._compute_matrices()
        
        # 2. 计算隔离成本权重 (lambda_isol)
        # n 是客户顶点数量，这里 self.n_nodes 就是客户数 N=51
        # 公式: lambda_isol = 0.5 + 0.0004 * a^2 * n
        self.lambda_isol = 0.5 + 0.0004 * (self.param_a ** 2) * self.n_nodes
        
        # 3. 计算隔离成本 D_i (客户 i 分配给任何其他顶点 j 的最低成本)
        # D_i = min(d_ij | j != i)
        
        # alloc_cost_matrix[0] 是 d_ij = (10 - a) * l_ij
        temp_alloc_cost = self.alloc_cost_matrix[0].clone() 
        
        # 排除对角线 d_ii，即自己分配给自己不计算在内
        diag_val = torch.inf
        temp_alloc_cost.fill_diagonal_(diag_val) 
        
        # D_i 是每一行（客户 i）到所有其他客户的最小分配成本
        self.node_isolation_cost, _ = temp_alloc_cost.min(dim=1) 
        
        print(f"SVRAP 环境初始化完成: a={self.param_a}, 客户节点 n={self.n_nodes}")
        print(f"动态计算的 Lambda_isol: {self.lambda_isol:.4f}")
        
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
        dist = torch.norm(diff, dim=-1) # l_ij (欧氏距离)
        
        # 路由成本 c_ij = a * l_ij
        route_cost = dist * self.param_a 
        
        # 分配成本 d_ij = (10 - a) * l_ij
        alloc_cost = dist * (10.0 - self.param_a)
        
        return dist.unsqueeze(0), route_cost.unsqueeze(0), alloc_cost.unsqueeze(0)

    def evaluate_solution(self, actions):
        """
        计算目标函数总成本 (最小化):
        Total Cost = lambda_tour * Tour + lambda_alloc * Allocation + lambda_isol * Isolation
        Actions: 0=Assign, 1=Route, 2=Loss
        """
        # 修正：确保 actions 是一维的 (N,)
        actions = actions.squeeze().cpu().numpy()
        if actions.ndim == 0:
             actions = np.array([actions.item()])
        
        assign_indices = [i for i, a in enumerate(actions) if a == 0]
        route_indices = [i for i, a in enumerate(actions) if a == 1]
        loss_indices = [i for i, a in enumerate(actions) if a == 2] 
        backbone_indices = route_indices
        
        route_mat = self.route_cost_matrix[0] # c_ij
        alloc_mat = self.alloc_cost_matrix[0] # d_ij
        
        # --- 1. 路由成本 (Tour Cost) ---
        # lambda_tour * sum(c_ij * x_ij)
        route_cost_sum = 0
        if len(backbone_indices) < 2:
            route_cost_sum = 1000.0 # 惩罚无效路径
        else:
            for k in range(len(backbone_indices)):
                u, v = backbone_indices[k], backbone_indices[(k + 1) % len(backbone_indices)]
                # route_mat[u, v] 已经是 c_ij = a * l_ij
                route_cost_sum += route_mat[u, v].item()
        
        tour_cost = SVRAPConfig.LAMBDA_TOUR * route_cost_sum
            
        # --- 2. 分配成本 (Allocation Cost) ---
        # lambda_alloc * sum(d_ij * y_ij)
        assign_cost_sum = 0
        if len(backbone_indices) > 0:
            for nb_idx in assign_indices:
                # 寻找最小分配成本 d_ij。alloc_mat[nb_idx, b] 已经是 d_ij = (10-a)*l_ij
                min_d_to_backbone = min([alloc_mat[nb_idx, b].item() for b in backbone_indices])
                assign_cost_sum += min_d_to_backbone

        allocation_cost = SVRAPConfig.LAMBDA_ALLOC * assign_cost_sum
        
        # --- 3. 隔离成本 (Isolation Cost) ---
        # lambda_isol * sum(D_i * v_i)
        isolation_cost_sum = 0
        for i in loss_indices:
             # D_i 是预先计算的最低分配成本 min(d_ij)
            isolation_cost_sum += self.node_isolation_cost[i].item()
            
        # lambda_isol 是动态计算的 self.lambda_isol
        isolation_cost = self.lambda_isol * isolation_cost_sum

        # 最终总成本
        total_cost = tour_cost + allocation_cost + isolation_cost
        
        # 额外惩罚：如果存在 ASSIGN 节点但没有 ROUTE 骨干
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
        # 路由成本矩阵 d 和 分配成本矩阵 c 用于指导注意力
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
    # 将路由成本矩阵和分配成本矩阵作为特征输入
    route_cost_tensor = env.route_cost_matrix 
    alloc_cost_tensor = env.alloc_cost_matrix
    
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
            # 使用路由和分配成本作为注意力输入特征
            logits = model(x, route_cost_tensor, alloc_cost_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
            cost = env.evaluate_solution(actions) # actions 已经是 (B, N)
            
            if cost < best_cost:
                best_cost = cost
                best_actions_tensor = actions.clone().detach() 
                
                # 实时保存最佳模型
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
                actions_np = actions.squeeze().cpu().numpy()
                route_count = (actions_np == 1).sum().item()
                loss_count = (actions_np == 2).sum().item()
                print(f"Epoch {epoch:04d} | Cost: {cost:.4f} | Best: {best_cost:.4f} | R/L Count: {route_count}/{loss_count}")
        
        print(f"\n🎉 训练完成。最优模型已保存到 {SVRAPConfig.MODEL_PATH}")

    # 4. 最终结果展示和贪心选择
    print("\n" + "="*70)
    print("最终结果展示 (Final Results)")
    print("="*70)
    print(f"最优总成本 (Best Cost): {best_cost:.4f}")

    final_actions = best_actions_tensor.squeeze().cpu().tolist() if best_actions_tensor is not None else []
    
    if best_actions_tensor is not None:
        final_route = [i for i, a in enumerate(final_actions) if a == 1]
        final_loss = [i for i, a in enumerate(final_actions) if a == 2]
        
        print(f"最优解 - ROUTE 节点: {final_route}")
        print(f"最优解 - LOSS 节点:   {final_loss}")
    
    model.eval()
    with torch.no_grad():
        final_logits = model(x, route_cost_tensor, alloc_cost_tensor)
        final_probs = F.softmax(final_logits, dim=-1)
    
    final_probs_np = final_probs[0].cpu().numpy()
    
    # --- 基于贪心策略的初始骨干构建 (Greedy Backbone Selection) ---
    
    p_route = final_probs_np[:, 1] # 获取 On-route 概率 (动作 1)
    sorted_indices = np.argsort(p_route)[::-1] # 节点按 P_Route 降序排序
    
    n_nodes = env.n_nodes
    k = max(2, int(n_nodes * SVRAPConfig.TOP_K_ROUTE_RATIO)) # 确定贪心选择的数量 K
    greedy_backbone_indices = sorted_indices[:k].tolist() # 选取前 K 个节点作为初始骨干
    
    # 评估这个贪心解的完整成本 (ROUTE=1, ASSIGN=0)
    greedy_actions = np.zeros(n_nodes, dtype=int)
    greedy_actions[greedy_backbone_indices] = 1 # ROUTE
    greedy_cost = env.evaluate_solution(torch.tensor(greedy_actions).unsqueeze(0)) 

    # -----------------------------------------------------------------
    
    print("\n" + "="*70)
    print("节点最终概率策略与最优解状态对比")
    print("="*70)
    print("ID | X,Y 坐标 | P_Assign | P_Route | P_Loss | 最优解状态 | 贪心骨干?")
    print("-" * 80)
    
    status_map = {0: 'ASSIGN', 1: 'ROUTE', 2: 'LOSS'}
    for i in range(env.n_nodes):
        p_assign, p_route_val, p_loss = final_probs_np[i]
        coord = env.coords[i].numpy()
        
        action_status = status_map[final_actions[i]] if best_actions_tensor is not None else "N/A"
        is_greedy_backbone = "✅" if i in greedy_backbone_indices else " "
        
        print(f"{i:2d} | {coord[0]:.0f},{coord[1]:.0f} | {p_assign:.4f} | {p_route_val:.4f} | {p_loss:.4f} | {action_status:10s} | {is_greedy_backbone:^8s}")
        
    print("-" * 80)
    print(f"**贪心选择结果 (K={k}, 阈值: P_Route 最高的 {SVRAPConfig.TOP_K_ROUTE_RATIO*100:.0f}%)**")
    print(f"贪心初始骨干节点 (ROUTE): {greedy_backbone_indices}")
    print(f"评估贪心解总成本: {greedy_cost:.4f}")
    print("="*70)


if __name__ == "__main__":
    # 默认行为：如果文件存在则加载，否则训练
    if os.path.exists(SVRAPConfig.MODEL_PATH):
        run_pipeline(train_model=False)
    else:
        run_pipeline(train_model=True)