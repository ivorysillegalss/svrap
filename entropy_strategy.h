#ifndef ENTROPY_STRATEGY_H
#define ENTROPY_STRATEGY_H

#include "attention_reader.h"
#include <set>
#include <vector>
#include <string>

// 可选的熵选择策略类型
enum class EntropyStrategyType {
  FIXED_THRESHOLD,    // 绝对阈值 H >= tau
  RELATIVE_MAX,       // 相对最大熵: H >= r * H_max
  TOP_K_PERCENT,      // Top-K%: 按熵排序取前K%
  ABOVE_MEAN,         // 高于平均值: H > mean(H)
  MEAN_PLUS_STD       // H >= mean(H) + alpha * std(H)
};

// 默认策略宏：使用 Top-K% 且 K = 60（更宽松，避免过度约束搜索）
#ifndef ENTROPY_STRATEGY_DEFAULT
#define ENTROPY_STRATEGY_DEFAULT EntropyStrategyType::TOP_K_PERCENT
#endif

#ifndef ENTROPY_TOP_K_PERCENT_DEFAULT
#define ENTROPY_TOP_K_PERCENT_DEFAULT 60.0  // 60% 节点作为高熵
#endif

// 统一的选择函数：根据给定策略与参数，选出需要重点搜索的节点
// 参数:
//  - probs:   每个节点的状态概率 (来自 Python 策略网络)
//  - type:    熵选择策略类型
//  - param:   策略参数, 对不同策略含义不同:
//             FIXED_THRESHOLD  : tau (绝对熵阈值)
//             RELATIVE_MAX     : r   (0~1, 相对最大熵比例)
//             TOP_K_PERCENT    : K   (0~100, Top-K%)
//             ABOVE_MEAN       : 忽略
//             MEAN_PLUS_STD    : alpha (H >= mean + alpha * std)
//  - out_nodes:  输出: 被认定为高熵、需要偏重搜索的节点坐标集合
//  - out_effective_threshold: 输出: 实际使用的数值阈值(用于日志 / 调参)
void select_high_entropy_nodes_with_strategy(
    const std::vector<NodeProb> &probs,
    EntropyStrategyType type,
    double param,
    std::set<std::pair<int, int>> &out_nodes,
    double &out_effective_threshold);

// 便捷包装：使用默认宏配置的策略
// 返回实际使用的策略类型和阈值, 方便在日志中打印
EntropyStrategyType select_high_entropy_nodes_default(
    const std::vector<NodeProb> &probs,
    std::set<std::pair<int, int>> &out_nodes,
    double &out_effective_threshold);

// 将策略类型转成可读字符串(用于日志)
std::string entropy_strategy_type_to_string(EntropyStrategyType type);

#endif // ENTROPY_STRATEGY_H
