# SVRAP 训练流程量化说明（最终论文版）

## 1. 文档适用范围

本文件描述当前仓库最终采用的训练与推理方案（2026-04）：

- 主训练/推理入口为 `svrap_solver.py`。
- baseline 统一为基于 `p_route` 的 greedy top-k。
- 不再使用 gumbel 训练/基线分支。
- 历史 gumbel 版本已归档至 `archive/`，不作为主流程。

## 2. 数据规模量化

### 2.1 pretraining_dataset 规模

- 预训练候选图数量：30（见 `pretraining_dataset/` 与 `SELECTION_CRITERIA.txt`）

### 2.2 main_dataset 规模

当前主评测数据集数量：13

1. att532
2. berlin52
3. bier127
4. d198
5. d493
6. gr137
7. gr96
8. pcb442
9. pr107
10. pr152
11. rat783
12. u159
13. u724

## 3. 训练计算量估算

### 3.1 记号

- 预训练 epoch：$E_p$
- 微调 epoch：$E_f$
- 预训练图数：$N_p = 30$
- 主评测图数：$N_m = 13$

### 3.2 总步数

- 预训练步数：$N_p \times E_p$
- 微调步数：$N_m \times E_f$
- 总步数：$N_pE_p + N_mE_f$

若取常用配置 $E_p=200, E_f=100$：

- 预训练步数：$30 \times 200 = 6000$
- 微调步数：$13 \times 100 = 1300$
- 总步数：$7300$

## 4. 损失与指标

### 4.1 主训练损失（当前）

以 `svrap_solver.py` 当前实现为准，核心由以下项构成：

1. REINFORCE 项（基于采样动作和 top-k baseline cost）
2. 反事实节点引导项（counterfactual node loss）
3. 熵正则项（保持探索）

训练中 baseline 固定为：

- 对 `p_route` 做 top-k 选点构造 `greedy_actions`
- 用其代价作为 `baseline_cost`

### 4.2 关键监控指标

每个阶段建议记录：

1. `Cost`
2. `Baseline`
3. `Best`
4. `p_route mean/std/min/max`
5.（可选）`cf_std` 与节点引导项强度

## 5. 产物与文件

### 5.1 模型产物

- 训练模型输出到 `models/`
- 文件名模式：`svrap_best_model_<dataset><variant>.pth`

### 5.2 推理中间产物

- `attention_probs.csv`
- `backbone_indices.txt`

### 5.3 结果产物

- 表格与汇总保留在 `results/`
- 仅保留与当前流程有耦合的结果文件
- 无耦合历史结果已清理或迁移到 `archive/`

## 6. 运行示例（当前主流程）

```powershell
python svrap_solver.py --dataset formatted_dataset/berlin52.txt --train
```

随后执行：

```powershell
./svrap.exe 7 formatted_dataset/berlin52.txt
```

## 7. 验收标准（最终方案）

1. 训练日志中的 baseline 口径为 top-k（非 gumbel）
2. 生成 `attention_probs.csv` 与 `backbone_indices.txt`
3. C++ 可读取概率并产出稳定的 best cost
4. `results/` 仅包含有耦合的核心结果文件
5. 历史实验版本位于 `archive/`，不干扰主流程
