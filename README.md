# SVRAP

Greedy + Tabu Search solver with Python backbone initializer (SVRAP policy network).

## Requirements
- Python 3.10+ (tested with conda env `altr-py310`)
- C++17 toolchain + Make
- Runtime Python deps: `torch>=2.0.0`, `numpy>=1.24.0`

## Quickstart (Windows PowerShell)
```
cd svrap

# 1) Python: generate backbone probabilities
conda activate altr-py310
pip install -r requirements.txt ## if not installed
python svrap_solver.py

# 2) C++: build and run (logs go to build.log / run.log)
make clean
make

# If you want to rerun manually (after make):
./app.exe
```

## What the pipeline does
- Python (svrap_solver.py)
  - Loads model `svrap_best_model.pth`
  - Computes 3-way state probabilities for 51 nodes (ASSIGN / ROUTE / LOSS)
  - Picks Top-10 `p_route` as backbone
  - Writes `attention_probs.csv` (x,y,p_assign,p_route,p_loss) and `backbone_indices.txt`
- C++ (greedy + tabu)
  - Reads `dataset.txt` coordinates (51 nodes)
  - Uses `attention_probs.csv` to build an initial backbone (10 backbone + 20 on-tour = 30 total in route)
  - **Entropy-based optimization**: Calculates Shannon entropy for each node based on state probabilities
    - High entropy nodes (uncertain states) → require more search resources
    - Low entropy nodes (certain states from policy network) → skip to save search time
    - Default entropy threshold: 0.8 (configurable in main.cpp)
  - Greedy local search → Tabu search with entropy filtering
  - Recent best cost observed: 180.22

## Key files
- `svrap_solver.py` — Python policy network + CSV export
- `svrap_best_model.pth` — trained weights (loaded if present)
- `attention_probs.csv` — node probabilities (Python → C++)
- `backbone_indices.txt` — backbone node indices (debugging aid)
- `dataset.txt` — coordinates input for C++
- `app.exe` — built solver binary
- `build.log`, `run.log` — build and run logs (Makefile)

## Make targets
- `make`          : build then run, logs to `build.log` / `run.log`
- `make clean`    : remove build artifacts
- `make build`    : build only (if your Makefile supports it)

## Notes
- If `attention_probs.csv` is missing, C++ falls back to nearest-neighbor init.
- Output text is ASCII-only to avoid console encoding issues on Windows.
- **Entropy filtering**: The tabu search now focuses on high-uncertainty nodes identified by entropy calculation. This reduces wasted search effort on nodes where the policy network already has high confidence about their state (ASSIGN/ROUTE/LOSS). The entropy threshold can be adjusted in `main.cpp` (line ~132).

---

# SVRAP

Python骨干初始化器(SVRAP策略网络) + 贪婪搜索 + 禁忌搜索求解器。

## 依赖环境
- Python 3.10+ (使用conda环境 `altr-py310`)
- C++17 + Make编译工具链
- Python运行依赖: `torch>=2.0.0`, `numpy>=1.24.0`

## 快速开始 (Windows PowerShell)
```powershell
cd svrap

# 1) Python: 生成骨干节点和概率分布
conda activate altr-py310
pip install -r requirements.txt # 如果未安装依赖
python svrap_solver.py

# 2) C++: 编译并运行 (日志记录在 build.log / run.log)
make clean
make

# 如需手动重新运行 (编译后):
./app.exe
```

## 管道流程说明
- Python (svrap_solver.py)
  - 加载已训练模型 `svrap_best_model.pth`
  - 计算51个节点的三状态概率 (ASSIGN / ROUTE / LOSS)
  - 选取Top-10 `p_route` 节点作为骨干
  - 输出 `attention_probs.csv` (x,y,p_assign,p_route,p_loss) 和 `backbone_indices.txt`
- C++ (贪婪+禁忌)
  - 读取 `dataset.txt` 坐标集 (51个节点)
  - 使用 `attention_probs.csv` 构建初始骨干解 (10个骨干节点 + 20个在线节点 = 30个总节点)
  - **基于信息熵的优化**: 根据节点状态概率计算香农熵
    - 高熵节点 (状态不确定) → 需要更多搜索资源
    - 低熵节点 (策略网络确定性高) → 跳过以节省搜索时间
    - 默认熵阈值: 0.8 (可在 main.cpp 中配置)
  - 贪婪局部搜索 → 带熵过滤的禁忌搜索
  - 最近获得的最优成本: 180.22

## 关键文件说明
- `svrap_solver.py` — Python策略网络 + CSV输出
- `svrap_best_model.pth` — 训练好的权重 (若存在则加载)
- `attention_probs.csv` — 节点概率分布 (Python → C++)
- `backbone_indices.txt` — 骨干节点索引 (调试用)
- `dataset.txt` — C++输入坐标数据
- `app.exe` — 编译后的求解器可执行文件
- `build.log`, `run.log` — 编译和运行日志 (Makefile生成)

## Make命令
- `make`          : 编译并运行, 日志输出到 `build.log` / `run.log`
- `make clean`    : 清理构建产物
- `make build`    : 仅编译 (如Makefile支持)

## 注意事项
- 如果缺少 `attention_probs.csv`, C++会降级使用最近邻初始化
- 输出文本为ASCII格式，以避免Windows控制台编码问题
- **熵过滤机制**: 禁忌搜索现在会专注于策略网络识别出的高不确定性节点。这减少了在策略网络已经对其状态 (ASSIGN/ROUTE/LOSS) 有高置信度的节点上浪费的搜索精力。熵阈值可以在 `main.cpp` (约第132行) 中调整。