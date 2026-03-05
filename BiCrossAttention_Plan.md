# 策略网络特征提取模块重构计划 (Feature Extraction Module Refactoring Plan)

## 1. 背景与目标
根据最新理论研究（笔记推导补充），我们要将现有的 `svrap_solver.py` 中的特征提取和融合部分进行一次大规模的架构升级。

**核心核心诉求**：**废除原先的 MLP“门控 (Gating)” 或“简单的晚期拼接融合 (Late Fusion / Concat)”**。转而使用更加符合特征交互逻辑的 **双向交叉注意力 (Bi-directional Cross-Attention)** 机制来进行“路由特征 (Route)”与“分配特征 (Alloc)”深度通信。

根据用户指示，本次重构**分阶段（Step-by-Step）**进行，当前先**只实现单向：“路由查分配 (Route to Alloc)”** 这一半功能。

---

## 2. 现状分析
当前代码（基于最近修改的单流统一架构）：
- 直接把 `d_matrix` 和 `c_matrix` 叠加在最后一维 `(N, N, 2)`。
- 送入单个 `EdgeBiasProjector` 用感知机投射成单个 Attention Bias 矩阵。
- 然后丢进普通的单流 `ContextEncoder` 里面。

**问题**：这种把两条通道生硬绑在一起的做法，抹杀了“分配”和“路由”作为两种不同的物理空间视角独立编码的能力，也无法实现真正的“交互查询”。

---

## 3. 具体修改计划 (阶段一：只做 $H_{r \rightarrow a}$)

### 3.1 恢复双视角独立编码 ($X \rightarrow H$)
必须把两个矩阵拆开，恢复两套独有的 Self-Attention 编码器。

- **操作思路**：
  在 `SVRAPNetwork` 的初始化中定义两个 `ContextEncoder` (可以重用现有的 Encoder，但输入维度调整)：
  - `routing_encoder`：专门接收 `c_matrix` (对应图中 $X_{route} \rightarrow H_{route}$)
  - `alloc_encoder`：专门接收 `d_matrix` (对应图中 $X_{alloc} \rightarrow H_{alloc}$)
- **附带改动**：
  把 `EdgeBiasProjector` 退回能接收单个通道 `(N,N)` 并扩充投射的格式。

### 3.2 增加【交叉注意力】层 (Cross-Attention: Route $\rightarrow$ Alloc)
这是本阶段重构的灵魂。我们要写一个新的模块，实现**路由查分配**（从 Alloc 中收集满足 Route 需求的信息）。

- **理论公式**：
  $$H^{cross}_{r \rightarrow a} = Softmax(\frac{Q_{route} K_{alloc}^T}{\sqrt{d_k}}) \cdot V_{route}$$
  *(注：严格遵从笔记要求，我们把 $V$ 设为 $H_{route}$。虽然通常交叉注意力中 $K,V$ 同源，这里尊重推导图中的设计)*。

- **代码操作**：
  1. 新建 `RouteToAllocCrossAttention(nn.Module)` 类。
  2. 内部实例化一个 `nn.MultiheadAttention`。
  3. `forward(h_route, h_alloc)` 中：
     - `query` = `h_route`
     - `key` = `h_alloc`
     - `value` = `h_route` (按图设计!)
  4. 加入标准的残差连接（Residual Connection）和 LayerNorm，甚至 FFN。

### 3.3 网络主干 (Forward) 串联
现在我们将上述组件拼装。
```python
def forward(self, x, edge_feat):
    d_matrix = edge_feat[:, :, :, 0]
    c_matrix = edge_feat[:, :, :, 1]
    
    # 1. 独立自编码
    h_base = self.node_embed(x)
    h_route = self.routing_encoder(h_base, c_matrix) # -> H_route
    h_alloc = self.alloc_encoder(h_base, d_matrix)   # -> H_alloc
    
    # 2. 交叉注意力 （目前仅单向）
    h_fused = self.cross_attn_r_to_a(h_route, h_alloc) # Q:Route, K:Alloc, V:Route
    
    # 附注：未来阶段2将会加入 H_a_to_r，并在此时进行 h_fused = h_r_to_a + h_a_to_r
    
    # 3. 分类输出
    logits = self.classifier(h_fused)
    return logits
```

---

## 4. 相关文档修正记录
为避免混乱，我已经顺手修复了项目中的相关指代：
- 修改 `README.md`，去除了过时的机制描述。
- 修正 `scripts/TEST_PLAN.md` 中对于网络架构的描述，将 `Dual-Stream` / 拼接（Concat）的字眼，提前更新为 `Bi-directional Cross-Attention` 架构描述，与新的发展蓝图保持一致。