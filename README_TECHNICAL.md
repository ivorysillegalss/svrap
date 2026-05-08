# SVRAP 技术归档（README_TECHNICAL，最终论文版）

## 1. 目标与范围

本文档描述当前仓库最终采用的技术方案（2026-04）：

1. Python 输出节点 `p_route` 概率先验
2. C++ 读取概率并执行 Tabu Search
3. baseline 统一为 `top-k`（按 `p_route` 选骨干点）
4. gumbel 相关分支不再作为主流程

历史实验与分叉版本位于 `archive/`，仅用于追溯，不参与当前默认流程。

## 2. 端到端流程

### 2.1 主链路

1. 读取数据集（`formatted_dataset/*.txt`）
2. Python 侧训练/推理，输出 `attention_probs.csv`
3. C++ 侧读取概率并构建初始化解（top-k backbone）
4. 执行 Tabu Search（邻域 + 多样化 + 路径重连）
5. 输出最优成本与运行时间

### 2.2 baseline 定义（当前）

Python 训练中 baseline 的计算方式：

1. 使用 `probs[:, 1]` 作为 `p_route`
2. 取 top-k 形成 `greedy_actions`
3. 评估得到 `baseline_cost`
4. 与采样动作成本比较得到 advantage

说明：当前不使用 gumbel top-k 采样分支作为默认训练路径。

## 3. 业务规则与状态语义

### 3.1 成本构成

SVRAP 近似目标由三部分组成：

1. Routing：`c_ij = a * l_ij`
2. Allocation：`d_ij = (10-a) * l_ij`
3. Isolation：`lambda_isol * D_i`

其中动态系数：

`lambda_isol = 0.5 + 0.0004 * a^2 * n`

### 3.2 off-route 决策

对每个 off-route 节点，按以下规则即时决策：

`min(assignment_cost, isolation_cost)`

### 3.3 状态字段

- `Y`：on-tour
- `N`：off-tour

“分配/隔离”不是固定状态字段，而是在成本评估时动态选择。

## 4. 实现结构

### 4.1 语言与职责

1. C++17：搜索核心（邻域评估、Tabu 管理、路径重连）
2. Python + PyTorch：策略网络训练与推理
3. 脚本层：批量实验、结果汇总、表格导出

### 4.2 关键模块

1. `main.cpp`：参数解析、实例调度
2. `input.cpp` / `input.h`：数据与概率文件读取
3. `greedy.cpp` / `greedy.h`：初始化与代价计算
4. `tabu_search.cpp` / `tabu_search.h`：Tabu 主循环
5. `svrap_solver.py`：训练、推理、导出

## 5. 外部接口与协议

### 5.1 C++ CLI

`svrap.exe [ALPHA] [dataset_path] [strategy] [k] [tbl] [q] [t] [entropy_weight]`

说明：

1. `paper_baseline` 策略会锁定论文基线参数
2. `attention_probs.csv` 默认从仓库根目录读取

### 5.2 Python CLI

`python svrap_solver.py --dataset <path> [--train|--no-train] [--epochs N] [--variant-tag TAG]`

### 5.3 文件协议

1. 数据集：每行 `x,y`
2. 概率文件：`attention_probs.csv`（`x,y,p_off,p_route`）
3. 骨干输出：`backbone_indices.txt`

## 6. 防御性设计

### 6.1 容错行为

1. 概率文件缺失：回退传统初始化
2. 行格式异常：逐行跳过，不中断全流程
3. 单实例异常：实例级隔离，不影响批量其他实例
4. 候选为空：安全退出当前循环

### 6.2 约束与假设

1. 运行目录需可访问 `attention_probs.csv`、数据集目录与 `models/`
2. Python 与 C++ 成本口径存在近似差异，结果解释以 C++ 最终成本为准
3. 随机种子可控但非跨平台严格位级复现

## 7. 复杂度摘要

设节点数为 $n$：

1. 距离矩阵构建：$O(n^2)$
2. 单次完整代价评估：最坏 $O(n^2)$
3. Tabu 单迭代：由候选数量与每候选评估复杂度共同决定，最坏受 $O(n^2)$ 主导
4. 神经前向（注意力层）：典型 $O(n^2 d)$
5. 反事实节点引导项（训练）：在大图上是主要开销项之一

## 8. 结果与工件管理

1. `results/`：仅保留与当前流程有耦合的核心结果
2. `archive/`：历史日志、分化版本、已下线实验分支
3. 通过 `variant-tag` 隔离模型与日志，避免覆盖

## 9. 当前结论

当前仓库是“神经先验 + 元启发式优化”的稳定论文交付形态：

1. 训练口径清晰（top-k baseline）
2. 推理与搜索解耦明确（Python 先验，C++ 优化）
3. 历史分支与主流程已隔离，仓库可复现性更高