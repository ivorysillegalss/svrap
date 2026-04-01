# SVRAP 混合求解器技术归档（README_TECHNICAL）

## 1. 项目编写目的（Objectives）

### 1.1 业务痛点与技术问题

本项目针对 SVRAP（Selective Vehicle Routing and Allocation Problem）类问题，核心痛点是：

- 纯启发式方法（仅贪婪 + Tabu）在大规模或结构复杂实例上容易陷入局部最优。
- 纯神经网络方法可以提供全局偏好，但难以稳定输出严格可行且低成本的最终路径。
- 多数据集批量实验下，模型训练、推理、启发式搜索、日志归档之间容易出现流程割裂与结果不可复现。

因此本项目采用 Python + C++ 的混合架构：

- Python 策略网络负责学习“哪些点更应进入骨干路径”的概率先验。
- C++ 禁忌搜索负责高效、可控、可解释的局部优化与全局逃逸。
- 脚本层负责批量实验、消融测试和结果固化。

### 1.2 预期交付状态

目标交付形态是一个可复现实验框架，而非单次脚本：

- 给定数据集文件，能够稳定输出路径优化结果（Best cost）。
- 支持策略切换（full/baseline/paper_baseline/no_nn/no_entropy/no_knn/simple_div）。
- 支持自动化批处理、实验日志、模型工件隔离（variant tag）。
- 在防御性层面，输入异常、文件缺失、格式漂移等情况不会导致整批实验崩溃。

---

## 2. 核心业务逻辑（Business Logic）

### 2.1 端到端流程

1. 数据加载：读取坐标数据（formatted_dataset/*.txt），计算 TSPLIB EUC_2D 距离矩阵。
2. 神经先验（可选）：
   - Python 模型读取同一数据集，输出每个节点二分类概率：off_route / route。
   - 结果写入 attention_probs.csv。
3. 初始解构建：
   - C++ 读取 attention_probs.csv。
   - 若启用 neural_init 且匹配成功，按 p_route 选 Top-K（20%，至少 2）构建 ontour。
   - 否则退化为全点在路径上的传统初始化。
4. 贪婪局部搜索：基于 ADD/DROP/交换（swap）思路生成可用初始可行解。
5. 禁忌搜索主循环：
   - 生成候选邻域（ADD/DROP/TWOOPT）。
   - 优先选择改进移动；否则执行非改进移动或触发多样化。
   - 新 Champion 出现时可触发路径重连（Path Relinking）。
6. 输出：打印每个实例的最优成本和运行耗时。

### 2.2 关键业务规则

- 目标函数采用 SVRAP 近似实现：
  - Routing：基于 c_ij = a * l_ij。
  - Allocation：基于 d_ij = (10-a) * l_ij。
  - Isolation：基于 lambda_isol * D_i，其中 lambda_isol = 0.5 + 0.0004 * a^2 * n。
- 对于 off-route 节点，按 min(分配成本, 隔离成本) 做逐点贪心决策。
- 若未提供 isolation 文件，C++ 端默认 D_i = (10-a) * min_{j!=i} l_ij。
- 为避免“全点离路”退化，Python 训练阶段加入 NO_ROUTE_PENALTY。
- 非改进移动阶段引入熵退火项：adjusted_cost = raw_cost - lambda(t) * s。
- 多样化策略使用频率项 + 神经偏差项混合评分，优先翻转“长期固化且与模型偏好冲突”的节点状态。

### 2.3 状态语义

- VertexInfo.status:
  - Y: 当前在路径上（on-tour）
  - N: 当前不在路径上（off-tour）
- 注意：是否“分配”或“隔离”并非独立状态机变量，而是在成本计算时按最小代价即时决策。

---

## 3. 底层实现细节（Implementation Details）

### 3.1 关键语言与库

- C++17（核心搜索引擎）：
  - 原因：低开销循环、细粒度内存控制、适合高频邻域评估。
- Python + PyTorch（策略网络）：
  - 原因：快速迭代模型结构、方便做 REINFORCE 与反事实损失组合训练。
- PowerShell/Python 脚本：
  - 原因：适配 Windows 环境下批处理实验、日志和结果聚合。

### 3.2 模型与搜索协同机制

- 神经网络输出：每节点二分类概率 (p_off, p_route)。
- C++ 利用方式：
  - 初始化时：高 p_route 节点优先入 Backbone。
  - 邻域操作时：
    - ADD 倾向采样高 p_route 的 off-route 节点。
    - DROP 倾向采样低 p_route 的 on-route 节点。
  - 多样化时：将频率记忆与 p_route 偏差共同作为翻转优先级。

### 3.3 并发模型

- 当前主程序与训练流程均为单进程串行执行。
- 不使用多线程并发搜索，设计上偏向可复现与行为可解释。
- 随机性来源于 C++ mt19937 与 PyTorch/CUDA。
- Python 侧可通过 --seed 控制随机种子；C++ 侧当前使用 random_device 初始化 mt19937，未提供命令行 seed 注入。

### 3.4 核心数据结构

- Point: 节点 ID + 整数坐标。
- VertexInfo: 节点索引、路径状态、最近路径节点、最近代价、隔离代价、是否高熵点。
- distance: N x N 距离矩阵。
- point_probs: 节点 ID 到 p_route 的映射。
- TabuInfo:
  - tabu_list + tabu_time，用于保存禁忌移动及剩余禁忌时长。
- OpKey:
  - 支持单点操作键（ADD/DROP）与点对操作键（TWOOPT）。

### 3.5 设计模式与组织方式

- 策略配置对象模式：
  - StrategyConfig 聚合策略开关与参数，避免散落的全局 flag。
- 管道分层模式：
  - 数据层（input）-> 初始解层（greedy）-> 元启发层（tabu）-> 训练层（solver.py）-> 编排层（scripts）。
- 近似“工厂式”流程选择：
  - 通过 CLI strategy 字符串分派不同组合策略（paper_baseline/full 等）。

### 3.6 接口定义与协议约定

#### C++ 可执行接口

- 命令（位置参数，均可选）：svrap.exe [ALPHA] [dataset_path] [strategy] [k] [tbl] [q] [t] [entropy_weight]
- 关键约定：
  - 未提供 dataset_path 时，会按内置实例列表批量运行 formatted_dataset 下多个数据集。
  - 若 strategy 为 paper_baseline，后续参数覆盖被忽略并锁定论文基线值。
  - attention_probs.csv 固定从工作目录根读取。

#### Python 求解接口

- 命令：python svrap_solver.py --dataset <path> [--train|--no-train] [--epochs N] [--variant-tag TAG] [--seed N] [--reinforce-only-datasets ...]
- 关键约定：
  - 输出 attention_probs.csv 与 backbone_indices.txt。
  - 模型路径按 dataset + variant-tag 隔离，防止覆盖。

#### 文件协议

- 数据集输入：每行 x,y。
- attention_probs.csv：x,y,p_off,p_route（二分类格式；兼容旧 5 列格式）。
- isolation 文件：与数据集同名后缀 _iso.txt，每行 x,y,iso_cost。

---

## 4. 防御性记录与边界（Defensive Documentation）

### 4.1 异常处理与失败点兜底

- 文件不可读：
  - 坐标文件读取失败抛异常并中止当前实例。
  - isolation/probability 文件缺失仅告警，走默认逻辑继续。
- 行格式错误：
  - 逐行容错，非法行跳过，不阻断全文件。
- 概率格式漂移：
  - 兼容 4 列（二分类）与 5 列（旧格式）并做归一化。
- Python Backbone 不可用：
  - 自动回退至全点 on-tour 初始化。
- 邻域无有效候选：
  - 退出当前迭代循环，不出现未定义行为。
- 实例级异常隔离：
  - main 对每个数据集单独 try/catch，单实例失败不影响后续实例。

### 4.2 输入约束与环境依赖

- 输入约束：
  - 数据集应为纯文本坐标 CSV（x,y）。
  - 点数至少应支持形成非空路径；过小实例会导致某些操作降级为 ADD。
- 环境依赖：
  - C++17 编译器。
  - Python 3.8+ 与 torch。
  - Windows PowerShell 脚本中可能写死解释器路径（需按环境调整）。
- 路径约束：
  - 运行目录需保证能访问 attention_probs.csv、formatted_dataset、models。

### 4.3 当前权衡与遗留风险

- 搜索效率权衡：
  - 虽有部分增量化（如 TWOOPT 复用 allocation 成本、ADD 仅重算 routing+一次 allocation），但 DROP 与若干流程仍需完整成本评估。
- 混合策略耦合风险：
  - C++ 行为对 Python 输出分布敏感，模型塌缩会直接影响初始化与邻域偏置。
- 成本一致性风险：
  - Python 训练环境中的 tour cost 为启发式近似，和 C++ 最终 Tabu 优化成本口径存在差异。
- 可移植性风险：
  - 脚本中存在 Windows/特定 Conda 路径假设。
- 随机性风险：
  - Python 可控 seed 仍会受硬件与库版本影响；C++ 当前未暴露 seed 参数，跨次运行会有随机扰动。

---

## 5. 技术规格（Technical Specs）

### 5.1 复杂度与资源开销（核心函数）

设节点数为 n。

- 距离矩阵构建 compute_distances:
  - 时间复杂度 O(n^2)
  - 空间复杂度 O(n^2)
- 成本评估 compute_cost:
  - routing 部分 O(|route|)
  - allocation/isolation 部分最坏 O(n^2)
  - 合并后最坏 O(n^2)
- 贪婪初始化 nearest_neighbour:
  - 时间复杂度 O(k^2)（k 为初始 on-tour 点数）
- Tabu 单次迭代（近似）:
  - 通过最多 200 次尝试生成至多 50 个有效候选（MAX_NEIGHBORS=50）。
  - 单候选评估在不同操作下复杂度不同：DROP 更接近完整 O(n^2)，ADD/TWOOPT 存在部分复用。
  - 总迭代上限 MAX_TOTAL_ITER=1000，最坏时间量级仍受 O(n^2) 成本评估主导。
- 神经前向（多头注意力）:
  - 每层典型 O(n^2 * d)
- 训练中反事实节点损失:
  - 每 epoch 对每节点计算 on/off 两次评估，额外约 O(n^3)
  - 是训练时的主要耗时来源之一

### 5.2 资源消耗预估

- 内存热点：
  - C++ N x N 距离矩阵（double）与多份候选解拷贝。
  - Python 中 dist/c/d 三个矩阵 + 注意力中间张量。
- 时间热点：
  - Tabu 候选反复评估。
  - Counterfactual node loss（尤其在大规模数据集）。

### 5.3 数据库 / API 路由说明

- 本项目无数据库 Schema。
- 本项目无 HTTP API 路由。
- 对外接口主要是 CLI + 文件协议（dataset/probabilities/logs/models）。

---

## 6. 关键模块清单与职责映射

- main.cpp: 程序入口、参数解析、实例循环、策略开关控制。
- input.h / input.cpp: 数据读取、距离矩阵、概率与隔离成本协议解析。
- greedy.h / greedy.cpp: 初始路径构造、局部搜索、目标函数成本计算。
- tabu_search.h / tabu_search.cpp: Tabu 主循环、候选邻域、多样化、路径重连、Champion 频率记忆。
- svrap_solver.py: 环境定义、模型结构、训练（REINFORCE + 反事实节点损失）、推理导出。
- scripts/*.ps1, scripts/*.py: 批处理实验、消融、日志归档、结果分析。

---

## 7. 维护建议（高可维护性）

- 将 Python 评估口径与 C++ 成本函数再对齐（减少训练目标与搜索目标偏差）。
- 将脚本中的解释器路径改为环境变量或参数化注入，增强可移植性。
- 为关键配置（StrategyConfig 与 SVRAPConfig）增加统一配置文件（如 YAML），减少多处硬编码。
- 对 attention_probs.csv 与 _iso.txt 增加 schema 校验脚本，前置失败检测。
- 逐步引入最小回归集（固定 seed + 固定实例）用于版本升级对比。

---

## 8. 结论

该项目本质上是“神经先验 + 元启发式优化”的工程化混合求解框架：

- 神经网络负责给出结构化先验和搜索偏置。
- Tabu Search 负责在可解释、可约束的规则下做强优化。
- 防御性设计保证了在文件缺失、格式异常、概率失效等情况下仍可退化运行。

在当前实现中，系统已经具备实验复现、策略消融和工程扩展的基础能力，适合继续向“配置统一、口径一致、自动回归”方向演进。