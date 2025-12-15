# svrap

基于 TS-SVRAP 论文的 **贪婪 + 禁忌搜索 (Tabu Search)** 实现，包含路径重链接 (Path Relinking) 和多样化 (Diversification)，用于在 TSPLIB 坐标数据上求解单车路径分配问题 (SVRAP)。

核心部分：

- C++17 实现：`main.cpp`, `greedy.cpp`, `tabu_search.cpp`, `input.cpp`
- 运行控制参数：权重参数 `ALPHA` (= 论文中的 \(a\))
- 数据集：`formatted_dataset/*.txt`（TSPLIB 格式坐标）
- Python 单线程批跑脚本：`run_experiments.py`

构建和运行均依赖于 `Makefile` 或直接调用编译好的 `svrap.exe`。

---

## 构建

使用 `Makefile`（推荐）：

```bash
make build
```

或直接：

```bash
make          # 默认目标：编译生成 svrap.exe
```

如需手动编译（示例）：

```bash

```

---

## 单实例运行

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
