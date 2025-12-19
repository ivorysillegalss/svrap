# svrap

基于 TS-SVRAP 论文的 **混合求解器** 实现。结合了 **深度强化学习 (Deep RL)** 策略网络生成初始解，以及 **禁忌搜索 (Tabu Search)** 进行后续优化。

核心部分：

- **Python 策略网络** (`svrap_solver.py`): 使用 PyTorch 实现的 Attention 模型，通过 REINFORCE 算法训练，为每个节点预测成为骨干路径点（Backbone）的概率。
- **C++ 禁忌搜索** (`main.cpp`, `tabu_search.cpp`): 读取 Python 生成的概率分布构建初始解，并执行带有路径重链接 (Path Relinking) 和多样化 (Diversification) 的禁忌搜索。
- **自动化脚本** (`run_experiments.ps1`): PowerShell 脚本，用于批量执行 "Python 推理 -> C++ 搜索" 的完整流程。

---

## 环境依赖

1.  **C++**: 支持 C++17 的编译器 (如 g++ 8.1.0+).
2.  **Python**: Python 3.8+, 需安装 `torch` (建议支持 CUDA).
3.  **PowerShell**: 用于运行批处理脚本.

---

## 构建

使用 `g++` 编译 C++ 求解器：

```bash
g++ -std=c++17 -O2 -o svrap.exe main.cpp input.cpp greedy.cpp tabu_search.cpp
```

---

## 运行流程

本项目采用 **Python + C++** 的混合流水线模式。

### 1. 自动化批处理 (推荐)

使用 PowerShell 脚本一键运行所有实验。该脚本会自动遍历 `formatted_dataset` 目录下的所有数据集，依次执行 Python 网络推理和 C++ 搜索，并将结果记录到日志中。

```powershell
./run_experiments.ps1
```

**输出:**
- `experiment_results.log`: 包含每个数据集的训练/推理日志以及 C++ 求解的最终 Best Cost。
- `models/`: 存放训练好的 PyTorch 模型 (`.pth`)。

### 2. 单实例手动运行

如果需要调试单个数据集，可以分两步执行：

**步骤 1: 运行 Python 策略网络**
加载数据集，训练模型（或加载已有模型），并输出节点概率到 `attention_probs.csv`。

```bash
# 语法: python svrap_solver.py --dataset <路径> [--train]
python svrap_solver.py --dataset formatted_dataset/berlin52.txt --train
```
*   `--train`: 强制重新训练模型。如果不加且模型存在，则直接加载模型进行推理。
*   **产物**: 生成 `attention_probs.csv` 和 `backbone_indices.txt`。

**步骤 2: 运行 C++ 求解器**
读取数据集和步骤 1 生成的 `attention_probs.csv`，进行禁忌搜索。

```bash
# 语法: ./svrap.exe <ALPHA> <路径>
./svrap.exe 7 formatted_dataset/berlin52.txt
```
*   `ALPHA`: 权重参数 (通常为 3, 5, 7, 9)。
*   **产物**: 在终端输出最终的 Best Cost。

---

## 项目结构

```text
.
├── formatted_dataset/      # 数据集文件 (.txt)
├── models/                 # 保存的 PyTorch 模型 (.pth)
├── svrap_solver.py         # Python 策略网络源码
├── svrap.exe               # 编译后的 C++ 求解器
├── run_experiments.ps1     # 批处理实验脚本
├── attention_probs.csv     # [中间文件] Python 输出的节点概率
├── experiment_results.log  # [结果] 实验运行日志
├── main.cpp                # C++ 入口
├── tabu_search.cpp         # C++ 禁忌搜索实现
└── ...
```

---

## 算法原理简述

1.  **策略网络 (Python)**:
    - 输入：节点坐标。
    - 模型：基于 Attention 的 Encoder-Decoder 结构。
    - 输出：每个节点属于 Backbone (路径) 的概率。
    - 训练：REINFORCE 算法，以 SVRAP 成本为奖励信号。

2.  **混合初始化**:
    - C++ 读取 `attention_probs.csv`。
    - 根据 `p_route` (路径概率) 贪婪地选择高概率节点构建初始 Backbone。

3.  **禁忌搜索 (C++)**:
    - 在初始解的基础上进行 `ADD`, `DROP`, `TWO-OPT` 邻域搜索。
    - 包含 Path Relinking (路径重连) 和 Diversification (多样化) 机制跳出局部最优。

4.  **混合多样化策略 (Hybrid Diversification)**:
    - **创新点**: 在多样化阶段，不再仅依赖“频率”信息，而是引入 Python 模型的预测概率作为“全局直觉”。
    - **机制**: 计算混合得分 $Score = w \cdot \text{FreqTerm} + (1-w) \cdot \text{DevTerm}$。
        - `FreqTerm`: 节点长期处于某种状态的频率（历史经验）。
        - `DevTerm`: 节点当前状态与模型预测概率的偏差（模型直觉）。
    - **效果**: 优先翻转那些“既长期卡住，又违背模型预测”的节点，从而更精准地跳出局部最优。在 `eil51` 等数据集上，该策略使收敛时间缩短了约 **24.5%**。

---

## 旧版说明 (仅供参考)

### 单实例运行 (纯 C++ 模式)

程序入口在 `main.cpp`，命令行格式：

```bash
./svrap.exe <ALPHA> [dataset_path]
```

- `ALPHA`：double/float，论文中的参数 \(a\)，典型取值 `3, 5, 7, 9`。
- `dataset_path`（可选）：指定单个数据集，例如：

```bash
./svrap.exe 7 formatted_dataset/eil76.txt
```

如果省略 `dataset_path`，程序会按内置列表批量跑多个标准实例（见 `main.cpp` 中 `instance_files`）。

运行日志（包括最优 cost）会输出到控制台，需要时可重定向保存，例如：

```bash
./svrap.exe 7 formatted_dataset/eil76.txt > run_single.log 2>&1
```

---

## 批量实验脚本（Python）

`run_experiments.py` 用于在所有数据集和多组 `ALPHA` 上批量运行，并记录结果与心跳信息。

基本用法（从仓库根目录执行）：

```bash
python run_experiments.py \
	--exe ./svrap.exe \
	--dataset-dir ./formatted_dataset \
	--alphas 3 5 7 9 \
	--output results.csv \
	--run-log run.log \
	--summary-log summary.log
```

特性：

- 单线程顺序运行所有 `(dataset, ALPHA)` 组合。
- 每个实例记录：
	- 开始时间（本地时间）
	- 最优 cost（从 C++ 输出中解析 `Best cost ... =`）
	- 运行耗时（秒）
- 所有运行即时追加到 `run.log`，CSV 汇总写入 `results.csv`，文本汇总写入 `summary.log`。
- 长时间运行时，每 **10 分钟** 输出一次心跳：
	- 控制台：`[HEARTBEAT] <时间> elapsed=XX.X min, completed N/M runs`
	- `run.log` 同步记录一行 `[HEARTBEAT] ...`。

---

## 日志与输出文件

- `run.log`：
	- 多次实验会在文件末尾追加，会话之间用 `=== New experiment session ===` 分隔。
	- 每行包含时间戳、数据集名、`ALPHA`、最优 cost、运行耗时（秒）。
- `results.csv`：
	- 机器可读的结果汇总表头：`dataset,alpha,best_cost,time_seconds`。
- `summary.log`：
	- 人类可读的整体摘要，每行一个 `(dataset, ALPHA)` 的结果。

这些文件已在 `.gitignore` 中忽略，默认不会提交到仓库。

---

## 清理

使用 `Makefile` 清理构建产物：

```bash
make clean
```

必要时手动删除：

- `svrap.exe`
- 可能的 CMake 中间文件（见 `.gitignore` 中的相关条目）

---

## 依赖

- C++17 编译器（如 g++ / clang++）
- Python 3.8+（用于批量实验脚本）
