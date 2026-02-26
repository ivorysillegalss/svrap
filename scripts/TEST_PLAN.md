# SVRAP 混合式算法完整测试方案

## 一、测试概述

本方案针对**深度学习增强的SVRAP混合式求解算法**，设计全面的验证实验，以证明算法的有效性、创新性和实用性。

### 算法核心组件
| 组件 | 描述 | 对应代码 |
|------|------|----------|
| **策略网络** | Dual-Stream Transformer预测节点状态概率 | `svrap_solver.py` |
| **骨干初始化** | 基于P(route)概率构建初始在途节点集 | `SVRAPNetwork` |
| **熵引导搜索** | 高熵节点优先探索机制 | `entropy_weight` 参数 |
| **KNN加速** | K近邻邻域裁剪 | `k_neighbors` 参数 |
| **禁忌搜索** | 带路径重连的改进TS | `tabu_search.cpp` |

---

## 二、数据集配置

### 2.1 规模分级
| 类别 | 节点数 | 数据集 | 用途 |
|------|--------|--------|------|
| **小规模** | < 100 | `eil51`, `berlin52`, `st70`, `pr76`, `eil76`, `gr96`, `rat99` | 算法正确性验证 |
| **中规模** | 100-150 | `kroA100`~`kroE100`, `rd100`, `eil101`, `lin105`, `pr107`, `gr120`, `bier127`, `ch130`, `gr137` | 主要性能对比 |
| **大规模** | 150-200 | `kroA150`~`kroB200`, `pr152`, `u159`, `rat195`, `d198` | 扩展性验证 |
| **超大规模** | > 200 | `d493`, `rat783` | 极限测试 |

### 2.2 异构成本参数 α
根据SVRAP定义：$c_{ij} = \alpha \cdot l_{ij}$, $d_{ij} = (10-\alpha) \cdot l_{ij}$

| α值 | 路由成本权重 | 分配成本权重 | 预期特征 |
|-----|-------------|-------------|----------|
| 3 | 低 (30%) | 高 (70%) | 在途节点多，长路径 |
| 5 | 中 (50%) | 中 (50%) | 平衡状态 |
| 7 | 高 (70%) | 低 (30%) | 在途节点少，短路径 |
| 9 | 极高 (90%) | 极低 (10%) | 极简骨干 |

---

## 三、评估指标体系

### 3.1 主要指标
| 指标 | 公式 | 说明 |
|------|------|------|
| **Best Cost** | $Z^* = \min Z$ | 最优解成本 |
| **Gap (%)** | $\frac{Z_{algo} - Z_{baseline}}{Z_{baseline}} \times 100$ | 相对基准的改进率 |
| **Time (s)** | 求解耗时 | 计算效率 |

### 3.2 鲁棒性指标（多次运行）
| 指标 | 公式 | 说明 |
|------|------|------|
| **Mean** | $\bar{Z} = \frac{1}{n}\sum Z_i$ | 平均成本 |
| **Std** | $\sigma = \sqrt{\frac{1}{n}\sum(Z_i-\bar{Z})^2}$ | 标准差 |
| **CV (%)** | $\frac{\sigma}{\bar{Z}} \times 100$ | 变异系数（稳定性） |
| **Best/Worst** | $Z_{min} / Z_{max}$ | 极值范围 |

---

## 四、测试模块详细设计

### 4.1 模块A：基础性能测试
**目的**：验证完整算法在全量数据集上的求解能力

**执行脚本**：`run_final_experiments.py`

**配置**：
```python
ALPHA = 7.0  # 默认测试值
STRATEGY = "full"
K = 20, TABU = 20, DIV = 2, PR = 50
```

**输出**：
- `final_experiment_results.csv`: 所有数据集的 Cost/Time
- 按规模分组的性能汇总

---

### 4.2 模块B：消融实验 (Ablation Study)
**目的**：验证各创新组件的贡献度

**执行脚本**：`run_ablation_study.ps1`

**策略配置**：
| 策略ID | 描述 | 禁用组件 |
|--------|------|----------|
| `full` | 完整算法 | 无 |
| `baseline` | 基准方案 | 神经网络 + 熵引导 |
| `no_nn` | 无神经引导 | 策略网络初始化 |
| `no_entropy` | 无熵机制 | 香农熵资源倾斜 |
| `no_knn` | 无KNN加速 | K近邻邻域裁剪 |
| `simple_div` | 简单多样化 | 频率驱动多样化 |

**测试数据集**（代表性子集）：
```
小规模: berlin52, st70
中规模: kroA100, gr120, bier127
大规模: pr152, d198, d493
```

**重复次数**：每配置30次（统计显著性）

**预期结论**：
- `full` vs `baseline`: 整体提升幅度
- `full` vs `no_nn`: 神经初始化贡献
- `full` vs `no_entropy`: 熵引导贡献
- `full` vs `no_knn`: KNN加速效果（主要看Time）

---

### 4.3 模块C：α参数敏感性测试
**目的**：验证算法在不同成本异构度下的适应性

**执行脚本**：`run_constraint_test.py`

**配置**：
```python
ALPHAS = [3.0, 5.0, 7.0, 9.0]
全量34个数据集
```

**分析内容**：
1. 各α下的平均成本趋势
2. 在途节点数量随α的变化
3. 路由/分配/隔离成本占比变化

---

### 4.4 模块D：超参数敏感性测试
**目的**：确定最优超参数配置规则

**执行脚本**：`run_hyperparam_test.py`

**测试参数**：
| 参数 | 测试范围 | 默认值 |
|------|----------|--------|
| K (KNN) | [10, 20, 30] | 20 |
| Tabu Length | [10, 15, 20] | 15 |
| Diversification | [1, 2, 3] | 2 |
| Path Relinking | [25, 50, 75] | 50 |

**测试数据集**：`kroA100` (代表性中等规模)

---

### 4.5 模块E：熵权重敏感性测试
**目的**：验证熵机制参数λ对探索-开发平衡的影响

**执行脚本**：`run_entropy_test.py`

**配置**：
```python
ENTROPY_WEIGHTS = [0, 1, 10, 100, 1000]
DATASETS = [eil51, kroA100, pr152, rat195]  # 各规模代表
```

**预期结论**：
- 小规模数据集：较高熵权重（λ=10）促进探索
- 大规模数据集：较低熵权重（λ=1）稳定收敛

---

### 4.6 模块F：算法稳定性测试
**目的**：验证算法的鲁棒性（非确定性算法的稳定性）

**执行脚本**：`run_stability_test.py`

**配置**：
```python
DATASETS = [kroA100, kroB100, kroC100, kroD100, kroE100]
NUM_RUNS = 10  # 每个数据集重复10次
```

**分析指标**：
- 变异系数 CV (%) < 5% 为稳定
- 最优/最差差距

---

### 4.7 模块G：训练轮次敏感性测试
**目的**：确定神经网络最优训练epoch数

**执行脚本**：`run_epoch_test.py`

**配置**：
```python
EPOCH_VALUES = [100, 500, 1000, 2000]
```

**预期结论**：
- 确定收敛所需最小epoch
- 过拟合风险评估

---

## 五、统计显著性检验

### 5.1 检验方法
- **配对t检验**：比较full vs baseline在同一数据集上的表现
- **Wilcoxon符号秩检验**：非参数检验（当数据不满足正态性）
- **显著性水平**：α = 0.05

### 5.2 检验流程
```python
from scipy import stats

# 配对t检验
t_stat, p_value = stats.ttest_rel(full_costs, baseline_costs)
if p_value < 0.05:
    print("改进具有统计显著性")
```

---

## 六、执行计划

### 阶段1：正确性验证（Day 1）
1. 编译C++代码：`make clean && make`
2. 在小规模数据集验证基础功能
3. 检查输出格式和成本计算

### 阶段2：消融实验（Day 2-3）
1. 运行 `run_ablation_study.ps1`
2. 收集30×6×11 = 1980条记录
3. 生成消融分析报告

### 阶段3：参数测试（Day 4）
1. 运行 `run_constraint_test.py` (α敏感性)
2. 运行 `run_hyperparam_test.py` (超参数)
3. 运行 `run_entropy_test.py` (熵权重)

### 阶段4：完整实验（Day 5-6）
1. 运行 `run_final_experiments.py`
2. 运行 `run_stability_test.py`
3. 汇总所有结果

### 阶段5：分析报告（Day 7）
1. 运行 `analyze_results.py`
2. 生成可视化图表
3. 撰写实验结论

---

## 七、预期实验结果表格模板

### 表1：消融实验结果
| Dataset | Full | Baseline | no_nn | no_entropy | no_knn | simple_div |
|---------|------|----------|-------|------------|--------|------------|
| berlin52 | - | - | - | - | - | - |
| kroA100 | - | - | - | - | - | - |
| ... | | | | | | |
| **Avg Gap(%)** | 0% | +X% | +Y% | +Z% | +W% | +V% |

### 表2：α敏感性结果
| Dataset | α=3 | α=5 | α=7 | α=9 |
|---------|-----|-----|-----|-----|
| eil51 | - | - | - | - |
| kroA100 | - | - | - | - |
| ... | | | | |

### 表3：规模扩展性
| 规模 | 数据集 | Nodes | Cost | Time(s) | 在途节点数 |
|------|--------|-------|------|---------|-----------|
| 小 | eil51 | 51 | - | - | - |
| 中 | kroA100 | 100 | - | - | - |
| 大 | d198 | 198 | - | - | - |
| 超大 | rat783 | 783 | - | - | - |

---

## 八、执行命令速查

```powershell
# 从项目根目录执行

# 1. 编译
make clean && make

# 2. 消融实验（需要较长时间）
cd scripts
.\run_ablation_study.ps1

# 3. α敏感性测试
python run_constraint_test.py

# 4. 超参数测试
python run_hyperparam_test.py

# 5. 熵权重测试
python run_entropy_test.py

# 6. 稳定性测试
python run_stability_test.py

# 7. 完整最终实验
python run_final_experiments.py

# 8. 结果分析
python analyze_results.py ../results/ablation_results.csv
```

---

## 九、注意事项

1. **GPU环境**：确保PyTorch可用GPU（加速神经网络训练）
2. **路径问题**：脚本从`scripts/`目录运行，路径使用`../`引用
3. **模型缓存**：训练好的模型保存在`models/`，首次运行会自动训练
4. **结果保存**：所有结果输出到`results/`目录
5. **日志查看**：遇到问题检查`experiment_results.log`
