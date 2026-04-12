# SVRAP 预训练与微调流程量化说明

## 1. 本次改动范围

本次实现目标：解决大图从零训练学不到规律的问题，建立“先监督+RL预训练，再Gumbel REINFORCE微调”的两阶段流程。

已完成的代码改动：

- 新增脚本：`scripts/train_with_pretraining_and_gumbel.py`
- C++ 导出标签能力：
  - `tabu_search.h` 新增 `get_best_solution()` 接口
  - `main.cpp` 新增可选参数 `label_output_path`，可输出每节点 0/1 标签（每行一个数字）

## 2. 数据与任务规模量化

### 2.1 预训练数据集规模

预训练图数量固定为 30 张（来自 `pretraining_dataset/SELECTION_CRITERIA.txt`）。

### 2.2 标签生成规模（C++ no_nn）

规则：每张图运行 `no_nn` 10 次，取成本最低那次的 on/off 结果写标签。

总运行次数：

- `30 张图 x 10 次/图 = 300 次 C++ no_nn`

每次输出：

- 1 个临时标签文件（0/1，逐节点）
- 1 个 best cost（用于比较取最优）

最终保留：

- 每张图 1 份最佳标签，共 `30` 份
- 默认目录：`pretraining_dataset/labels_no_nn/`

### 2.3 预训练阶段规模（监督+RL）

规则：每个 epoch 必须看完 30 张图，且不做 batch（逐图训练，适配变长节点数）。

若 `pretrain_epochs = E_p`：

- 每个 epoch 前向/反向步数：`30`
- 总训练步数：`30 x E_p`

默认 `E_p = 200` 时：

- 总步数：`6000`

### 2.4 main_dataset 微调规模（Gumbel REINFORCE）

`main_dataset` 当前共 13 张图：

- att532, berlin52, bier127, d198, d493, gr137, gr96, pcb442, pr107, pr152, rat783, u159, u724

规则：每张图加载预训练模型后，独立执行 100 epoch 的 Gumbel REINFORCE。

若 `finetune_epochs = E_f`，主数据集图数 `N_m = 13`：

- 总微调步数：`N_m x E_f`

默认 `E_f = 100` 时：

- 总步数：`13 x 100 = 1300`

## 3. 总体计算量（默认参数）

默认参数：

- `label-runs = 10`
- `pretrain-epochs = 200`
- `finetune-epochs = 100`

总量化结果：

1. C++ 标签生成：`300` 次
2. 预训练优化步：`6000` 步
3. 微调优化步：`1300` 步
4. 训练相关总优化步：`7300` 步

## 4. 训练目标与观测指标量化

### 4.1 预训练损失

预训练阶段采用：

- 监督损失：节点 0/1 标签的交叉熵
- 强化学习损失：REINFORCE（基于采样动作与 baseline advantage）
- 熵正则：鼓励探索

组合形式：

`Loss_pretrain = sup_weight * CE + rl_weight * REINFORCE - entropy_weight * Entropy`

默认权重：

- `sup_weight = 1.0`
- `rl_weight = 0.25`
- `entropy_weight = 0.01`

### 4.2 微调损失

微调阶段采用固定 Top-K 的 Gumbel 采样动作，损失为：

`Loss_finetune = REINFORCE - entropy_weight * Entropy`

### 4.3 p_route 分化监控

微调每个 epoch 打印以下统计：

- `p_route mean`
- `p_route std`
- `p_route min`
- `p_route max`

用于观察是否仍接近随机（低方差窄区间）或已出现有效分化。

## 5. 产物量化

### 5.1 标签产物

- `pretraining_dataset/labels_no_nn/*.labels.txt`：30 份
- 每份行数 = 对应图节点数
- 每行仅 `0` 或 `1`

### 5.2 模型产物

- 预训练模型：`models/svrap_pretrained_30graphs_sup_rl.pth`（1 个）
- 微调模型：`models/svrap_finetuned_<dataset>.pth`（13 个，main_dataset 每图 1 个）

理论总模型文件数（本流程新增）：

- `1 + 13 = 14` 个

## 6. 执行命令（基线）

```powershell
C:/Users/chenz/miniconda3/envs/altr-py310/python.exe scripts/train_with_pretraining_and_gumbel.py --repo-root . --pretraining-dir pretraining_dataset --main-dataset-dir main_dataset --cpp-exe ./svrap.exe --label-runs 10 --pretrain-epochs 200 --finetune-epochs 100 --alpha 7 --seed 42
```

若已生成标签，跳过 300 次 no_nn：

```powershell
C:/Users/chenz/miniconda3/envs/altr-py310/python.exe scripts/train_with_pretraining_and_gumbel.py --repo-root . --pretraining-dir pretraining_dataset --main-dataset-dir main_dataset --cpp-exe ./svrap.exe --skip-label-generation --pretrain-epochs 200 --finetune-epochs 100 --alpha 7 --seed 42
```

## 7. 验收标准（可量化）

1. 标签文件数为 30，且每个标签文件只包含 0/1。
2. 预训练阶段每个 epoch 的日志覆盖 30 张图（逐图更新）。
3. main_dataset 的 13 张图都完成 100 epoch 微调。
4. 微调日志中持续输出 p_route 的 mean/std/min/max。
5. `models/` 下出现 1 个预训练模型和 13 个微调模型。
