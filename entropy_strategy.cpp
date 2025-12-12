#include "entropy_strategy.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// 统一实现
void select_high_entropy_nodes_with_strategy(
    const std::vector<NodeProb> &probs,
    EntropyStrategyType type,
    double param,
    std::set<std::pair<int, int>> &out_nodes,
    double &out_effective_threshold) {

  out_nodes.clear();
  out_effective_threshold = 0.0;

  if (probs.empty()) {
    return;
  }

  // 计算所有熵并保留索引, 方便排序
  std::vector<double> entropies;
  entropies.reserve(probs.size());
  for (const auto &np : probs) {
    entropies.push_back(calculate_node_entropy(np));
  }

  std::vector<size_t> idx(entropies.size());
  for (size_t i = 0; i < idx.size(); ++i)
    idx[i] = i;

  // 一些通用统计量
  const double mean = std::accumulate(entropies.begin(), entropies.end(), 0.0) /
                      static_cast<double>(entropies.size());
  double variance = 0.0;
  for (double e : entropies) {
    variance += (e - mean) * (e - mean);
  }
  variance /= static_cast<double>(entropies.size());
  const double stddev = std::sqrt(variance);
  const double max_entropy = *std::max_element(entropies.begin(), entropies.end());

  switch (type) {
  case EntropyStrategyType::FIXED_THRESHOLD: {
    // 绝对阈值: H >= param
    const double threshold = param;
    out_effective_threshold = threshold;
    for (size_t i = 0; i < probs.size(); ++i) {
      if (entropies[i] >= threshold) {
        out_nodes.insert({probs[i].p.x, probs[i].p.y});
      }
    }
    break;
  }

  case EntropyStrategyType::RELATIVE_MAX: {
    // 相对最大熵: H >= r * H_max
    const double ratio = param;
    const double threshold = max_entropy * ratio;
    out_effective_threshold = threshold;
    for (size_t i = 0; i < probs.size(); ++i) {
      if (entropies[i] >= threshold) {
        out_nodes.insert({probs[i].p.x, probs[i].p.y});
      }
    }
    break;
  }

  case EntropyStrategyType::TOP_K_PERCENT: {
    // Top-K%: 先按熵排序, 再取前K%
    double k_percent = param;
    if (k_percent <= 0.0)
      k_percent = 1.0; // 至少取一个
    if (k_percent > 100.0)
      k_percent = 100.0;

    size_t k = static_cast<size_t>(std::ceil(probs.size() * (k_percent / 100.0)));
    if (k == 0)
      k = 1;
    if (k > probs.size())
      k = probs.size();

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      return entropies[a] > entropies[b];
    });

    for (size_t i = 0; i < k; ++i) {
      const size_t id = idx[i];
      out_nodes.insert({probs[id].p.x, probs[id].p.y});
    }

    out_effective_threshold = entropies[idx[k - 1]]; // 第k个的熵作为等效阈值
    break;
  }

  case EntropyStrategyType::ABOVE_MEAN: {
    // 高于平均值
    const double threshold = mean;
    out_effective_threshold = threshold;
    for (size_t i = 0; i < probs.size(); ++i) {
      if (entropies[i] > threshold) {
        out_nodes.insert({probs[i].p.x, probs[i].p.y});
      }
    }
    break;
  }

  case EntropyStrategyType::MEAN_PLUS_STD: {
    // H >= mean + alpha * stddev
    const double alpha = param;
    const double threshold = mean + alpha * stddev;
    out_effective_threshold = threshold;
    for (size_t i = 0; i < probs.size(); ++i) {
      if (entropies[i] >= threshold) {
        out_nodes.insert({probs[i].p.x, probs[i].p.y});
      }
    }
    break;
  }
  }
}

EntropyStrategyType select_high_entropy_nodes_default(
    const std::vector<NodeProb> &probs,
    std::set<std::pair<int, int>> &out_nodes,
    double &out_effective_threshold) {

  // 使用编译期宏选择默认策略
  EntropyStrategyType type = ENTROPY_STRATEGY_DEFAULT;
  double param = 0.0;

  switch (type) {
  case EntropyStrategyType::FIXED_THRESHOLD:
    param = 0.8; // 保留一个兼容老逻辑的参考值
    break;
  case EntropyStrategyType::RELATIVE_MAX:
    param = 0.95; // H >= 0.95 * H_max
    break;
  case EntropyStrategyType::TOP_K_PERCENT:
    param = ENTROPY_TOP_K_PERCENT_DEFAULT; // 默认 40%
    break;
  case EntropyStrategyType::ABOVE_MEAN:
    param = 0.0; // unused
    break;
  case EntropyStrategyType::MEAN_PLUS_STD:
    param = 0.5; // mean + 0.5 * std
    break;
  }

  select_high_entropy_nodes_with_strategy(probs, type, param, out_nodes,
                                          out_effective_threshold);
  return type;
}

std::string entropy_strategy_type_to_string(EntropyStrategyType type) {
  switch (type) {
  case EntropyStrategyType::FIXED_THRESHOLD:
    return "FixedThreshold";
  case EntropyStrategyType::RELATIVE_MAX:
    return "RelativeMax";
  case EntropyStrategyType::TOP_K_PERCENT:
    return "TopKPercent";
  case EntropyStrategyType::ABOVE_MEAN:
    return "AboveMean";
  case EntropyStrategyType::MEAN_PLUS_STD:
    return "MeanPlusStd";
  }
  return "Unknown";
}
