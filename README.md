# svrap

论文最终版本的 TS-SVRAP 混合求解器仓库。

当前仓库以“可复现论文结果”为目标，核心是：

- Python 策略网络输出节点 `p_route` 概率。
- C++ 禁忌搜索读取概率进行初始化与后续优化。
- baseline 统一为 `top-k`（按 `p_route` 选择骨干点），不再使用 gumbel 基线分支。

---

## 当前最终方案（2026-04）

1. 训练与推理主入口：`svrap_solver.py`
2. C++ 求解器入口：`main.cpp`
3. baseline 规则：`greedy top-k on p_route`
4. 已下线内容：gumbel 相关训练/基线路径（历史版本已归档或清理）

说明：仓库中保留了论文主流程有耦合的结果文件；无耦合历史版本已从 `results/` 清理。

---

## 环境依赖

- Python 3.10+（建议与当前实验环境一致）
- PyTorch（`requirements.txt`）
- C++17 编译器（g++ / clang++ / MSVC）
- Windows 下建议 PowerShell 运行脚本

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

---

## 构建 C++ 求解器

使用 Makefile：

```bash
make
```

或直接编译：

```bash
g++ -std=c++17 -O2 -o svrap.exe main.cpp input.cpp greedy.cpp tabu_search.cpp
```

---

## 最小运行流程

### 1) 运行 Python 侧（训练或加载）

```bash
python svrap_solver.py --dataset formatted_dataset/berlin52.txt --train
```

输出中间文件：

- `attention_probs.csv`
- `backbone_indices.txt`
- `models/svrap_best_model_<dataset>.pth`

### 2) 运行 C++ 侧 Tabu Search

```bash
./svrap.exe 7 formatted_dataset/berlin52.txt
```

其中 `7` 是参数 `alpha`。

---

## 项目结构（论文版）

```text
.
├── svrap_solver.py
├── main.cpp
├── tabu_search.cpp
├── input.cpp
├── greedy.cpp
├── formatted_dataset/
├── pretraining_dataset/
├── main_dataset/
├── models/
├── results/
├── scripts/
└── archive/
```

目录说明：

- `results/`：仅保留当前论文流程有耦合的核心结果。
- `archive/`：历史分支、日志、临时版本与清理备份。

---

## results 保留策略

为避免论文最终仓库被历史实验噪声污染，采用以下策略：

1. 与当前脚本/文档存在引用耦合的结果文件保留。
2. 无耦合的历史版本、临时对比、冗余日志从 `results/` 清理。
3. 如需追溯历史实验，优先从 `archive/` 检索。

---

## 常用脚本

位于 `scripts/`，例如：

- `run_table_with_control.py`
- `recompute_paper_baseline_no783.py`
- `run_paper_baseline_large_10runs.py`

说明：脚本名称中若包含旧实验语义（历史日期/临时标签），不代表当前主流程依赖该实验。

---

## 复现建议

1. 固定随机种子（Python 与 C++）
2. 先在 `berlin52` 做单例确认，再跑主数据集
3. 对论文表格优先使用 `results/` 中保留的耦合文件

---

## 许可证与说明

本仓库用于研究与论文复现。若需对外发布，请按你的论文/项目规范补充许可证与数据使用说明。
