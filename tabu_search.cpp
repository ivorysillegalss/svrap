#include "tabu_search.h"
#include "greedy.h"
#include "input.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

// 指选出两条最不相似的路径
#define DIVERSICATION_TIMES 2
#define ADD 1
#define DROP 2
#define TWOOPT 3
// #define N 51
#define MAX_TOTAL_ITER 1000

// 用于 Point 结构体的哈希函数，以支持 unordered_set
// 假设 Point 结构体中包含 int x 和 int y
struct PointHash {
  size_t operator()(const Point &p) const {
    return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
  }
};

static std::mt19937 rng(std::random_device{}());

TabuInfo::TabuInfo(int tb) {
  tabu_limit = tb;
  current_tabu_size = 0;
}

void TabuInfo::reset_list() {
  tabu_list.clear();
  tabu_time.clear();
  current_tabu_size = 0;
}

void TabuInfo::update_tabu() {
  for (size_t i = 0; i < tabu_time.size(); ++i) {
    if (tabu_time[i] > 0)
      tabu_time[i]--;
  }

  // 索引更新移除非0项 压缩迁移后数组
  size_t write_idx = 0;
  for (size_t i = 0; i < tabu_list.size(); ++i) {
    if (tabu_time[i] > 0) {
      if (write_idx != i) {
        tabu_list[write_idx] = tabu_list[i]; // Variant copy
        tabu_time[write_idx] = tabu_time[i];
      }
      ++write_idx;
    }
  }
  tabu_list.resize(write_idx);
  tabu_time.resize(write_idx);
  current_tabu_size = tabu_list.size();
}

void TabuInfo::set_limit(int new_limit) { tabu_limit = new_limit; }

// 判断当前禁忌表中是否存在当前遍历到的元素
bool TabuInfo::is_tabu_iter(const OpKey &iter) {
  // 注意：Point必须正确重载 operator==
  for (const auto &item : tabu_list) {
    if (item == iter)
      return true;
  }
  return false;
}

void TabuInfo::add_tabu_iter(const OpKey &iter, int limit) {
  tabu_list.push_back(iter);
  tabu_time.push_back(limit);
  current_tabu_size++;
}

TabuSearch::TabuSearch(
    const std::vector<Point> &locations,
    const std::vector<std::vector<double>> &distance,
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<Point> &route, const std::double_t &cost)
    : locations_(locations), distance_(distance), ontour_(ontour),
      offtour_(offtour), vertex_map_(vertex_map), route_(route),
      solution_cost_(cost), iter_solution_(route), best_cost_(cost) {
  cost_trend_.push_back(cost);
}

std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>,
           double, std::vector<Point>>
TabuSearch ::operation_style(
    const std::vector<Point> &route,
    const std::map<std::pair<int, int>, VertexInfo> &iter_dic) {

  // 三个操作中随机选择一个
  std::array<int, 3> choices = {ADD, DROP, TWOOPT};
  int style_number = choices[std::uniform_int_distribution<int>(0, 2)(rng)];

  // 防止对空路径进行 Drop/TwoOpt
  if (route.size() < 2 && style_number != ADD)
    style_number = ADD;

  // 如果是增加点
  if (style_number == ADD) {
    auto add_dic = iter_dic;
    Point add_vertice;
    bool found = false;

    // 安全地寻找一个不在路径上的点 (status == "N")
    // 为了防止死循环（如果所有点都在路径上），先收集所有 "N" 的点
    std::vector<Point> candidates;
    for (const auto &kv : iter_dic) {
      if (kv.second.status == "N") {
        candidates.push_back({kv.first.first, kv.first.second});
      }
    }

    if (candidates.empty()) {
      // 没有点可以添加，返回空操作（外部需处理）
      return {route, iter_dic, std::numeric_limits<double>::max(), {}};
    }

    // 随机选一个
    add_vertice = candidates[std::uniform_int_distribution<int>(
        0, candidates.size() - 1)(rng)];

    // 更新 Map 状态
    add_dic[{add_vertice.x, add_vertice.y}].status = "Y";

    // 更新其他 Off-tour 点的最近邻
    // (因为新加入的点可能成为某些点的最近邻)
    std::pair<int, int> add_key = {add_vertice.x, add_vertice.y};
    size_t add_idx = add_dic.at(add_key).index;

    for (auto &kv : add_dic) {
      if (kv.second.status == "N") {
        // 当前路外点的信息
        size_t off_idx = kv.second.index;

        // 计算该路外点到新加入点的距离
        double new_dist = distance_[off_idx][add_idx];

        // 比较并更新：如果新点更近，则更新 best_cost 和 best_vertex
        if (new_dist < kv.second.best_cost) {
          kv.second.best_cost = new_dist;
          kv.second.best_vertex = add_vertice;
        }
      }
    }

    // 寻找最佳插入位置
    std::vector<Point> best_route;
    double min_cost = std::numeric_limits<double>::max();

    for (size_t i = 0; i <= route.size(); ++i) {
      std::vector<Point> temp_route = route;
      temp_route.insert(temp_route.begin() + i, add_vertice);

      GreedyLocalSearch cal(temp_route, add_dic, distance_);
      double c = cal.tabu_cacl_cost();
      if (c < min_cost) {
        min_cost = c;
        best_route = temp_route;
      }
    }

    // 必须返回操作的点，用于禁忌表 (size 1)
    return {best_route, add_dic, min_cost, {add_vertice}};

  } else if (style_number == DROP) {
    if (route.empty())
      return {route, iter_dic, std::numeric_limits<double>::max(), {}};

    auto drop_dic = iter_dic;
    auto drop_route = route;

    // 随机删除
    int drop_index =
        std::uniform_int_distribution<int>(0, drop_route.size() - 1)(rng);
    Point d_vertice = drop_route[drop_index];

    drop_route.erase(drop_route.begin() + drop_index);

    // 更新 Map
    VertexInfo &info = drop_dic[{d_vertice.x, d_vertice.y}];
    info.status = "N";

    // 为新出来的路外点寻找最近的路内点
    if (!drop_route.empty()) {
      double best_dist = std::numeric_limits<double>::max();
      Point best_p;
      for (const auto &p : drop_route) {
        // 注意：这里需要 distance 矩阵，但在 operation_style 里访问成员
        // distance_ 是可以直接的
        size_t idx1 = info.index;
        size_t idx2 = iter_dic.at({p.x, p.y}).index;
        double d = distance_[idx1][idx2];
        if (d < best_dist) {
          best_dist = d;
          best_p = p;
        }
      }
      info.best_cost = best_dist;
      info.best_vertex = best_p;
    } else {
      info.best_cost = 0; // 路径为空，没有成本？或者无穷大
    }

    // 重新计算其他路外点（如果它们原本依附于被删除的点）
    // 省略深度优化，直接交给 GreedyLocalSearch 计算 Cost

    GreedyLocalSearch cal(drop_route, drop_dic, distance_);
    double i_cost = cal.tabu_cacl_cost();

    return {drop_route, drop_dic, i_cost, {d_vertice}};

  } else if (style_number == TWOOPT) {
    if (route.size() < 2)
      return {route, iter_dic, std::numeric_limits<double>::max(), {}};

    std::vector<size_t> indices(route.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<size_t> idx(2);
    // 随机选两个不同的索引进行交换
    std::sample(indices.begin(), indices.end(), idx.begin(), 2, rng);

    // 真正执行交换 Python 是在原 route 上直接 swap
    std::vector<Point> new_route = route;
    std::swap(new_route[idx[0]], new_route[idx[1]]);

    // 计算交换后的真实成本（字典不变）
    GreedyLocalSearch calculator(new_route, iter_dic, distance_);
    double new_cost = calculator.tabu_cacl_cost();

    // 记录被交换的两个点，用于禁忌表（顺序不重要，但要双向记录）
    std::vector<Point> operated = {route[idx[0]], route[idx[1]]};
    return {new_route, iter_dic, new_cost, operated};

  } else {
    // panic
    return {{}, {}, {}, {}};
  }
}

// 辅助函数：插入最佳位置
std::vector<Point>
best_insert_position(const Point &p, const std::vector<Point> &route,
                     const std::map<std::pair<int, int>, VertexInfo> &dic,
                     const std::vector<std::vector<double>> &distance) {
  std::vector<Point> best_route = route;
  double best_cost = std::numeric_limits<double>::max();

  // 简单的插入逻辑，为了性能不全量 Recalculate，但为了准确这里还是调 Greedy
  for (size_t i = 0; i <= route.size(); ++i) {
    auto test_route = route;
    test_route.insert(test_route.begin() + i, p);
    GreedyLocalSearch calc(test_route, dic, distance);
    double cost = calc.tabu_cacl_cost();
    if (cost < best_cost) {
      best_cost = cost;
      best_route = test_route;
    }
  }
  return best_route;
}

std::tuple<std::vector<Point>, double>
TabuSearch::path_relinking(const std::vector<std::vector<Point>> &solution_set,
                           std::map<std::pair<int, int>, VertexInfo> iter_dic) {
  // 前置长度校验
  if (solution_set.size() < 2) {
    return {solution_set.back(), solution_cost_};
  }

  const std::vector<Point> &last_solution =
      solution_set[solution_set.size() - 2];
  const std::vector<Point> &current_solution = solution_set.back();

  // 找出两个解之间“状态不同”的点
  std::unordered_set<Point, PointHash> current_set(current_solution.begin(),
                                                   current_solution.end());
  std::unordered_set<Point, PointHash> last_set(last_solution.begin(),
                                                last_solution.end());
  std::vector<Point> state_change;

  // last → current 中消失的点（从路径中被移出）
  // 1. 找出 target 有但 base 没有的点 (需要插入)
  for (const Point &p : last_solution) {
    if (current_set.count(p) == 0) {
      state_change.push_back(p);
    }
  }

  // current → last 中新增的点（被插入路径）
  // 找出 base 有但 target 没有的点 (需要删除)
  for (const Point &p : current_solution) {
    if (last_set.count(p) == 0) {
      state_change.push_back(p);
    }
  }

  const size_t n_solutions = solution_set.size();
  if (state_change.empty()) {
    GreedyLocalSearch calc(last_solution, iter_dic,distance_);
    return {last_solution, calc.tabu_cacl_cost()};
  }

  // 2. 计算每个改变点在历史解集中出现的比例 (vertice_on_probabity)
  std::vector<double> vertice_on_probability;
  vertice_on_probability.reserve(state_change.size());

  for (const Point &p : state_change) {
    int count = 0; // 记录点 p 出现在历史解中的次数
    for (const auto &sol : solution_set) {
      if (std::find(sol.begin(), sol.end(), p) != sol.end()) {
        ++count;
      }
    }
    vertice_on_probability.push_back(static_cast<double>(count) / n_solutions);
  }

  // 3. 构建最终的概率列表 (vertice_probability)
  std::vector<double> vertice_probability;
  vertice_probability.reserve(state_change.size());

  for (size_t i = 0; i < state_change.size(); ++i) {
    const Point &p = state_change[i];
    bool is_in_current = current_set.count(p) > 0;

    if (is_in_current) {
      vertice_probability.push_back(
          vertice_on_probability[i]); // on_tour 比例 (目标是保留)
    } else {
      vertice_probability.push_back(
          1.0 - vertice_on_probability[i]); // off_tour 比例 (目标是删除)
    }
  }

  // 4. 迭代移动：从 last_solution 转向 current_solution
  auto temp_solution = last_solution;

  // 循环直到所有差异点都处理完毕
  while (!state_change.empty()) {

    // 找到概率最大的元素的迭代器
    auto max_prob_it = std::max_element(vertice_probability.begin(),
                                        vertice_probability.end());
    if (max_prob_it == vertice_probability.end())
      break;

    size_t max_idx = std::distance(vertice_probability.begin(), max_prob_it);

    Point p = state_change[max_idx];

    // 判断 p 在目标解中是保留还是移除
    if (current_set.count(p) > 0) {
      // 该点最终要在路径中 → 插入到当前 temp_solution 成本增加最小的位置
      temp_solution =
          best_insert_position(p, temp_solution, iter_dic, distance_);
      iter_dic.at({p.x, p.y}).status = "Y"; // 标记为已在路径上
    } else {
      // 该点最终不在路径中 → 直接删除
      temp_solution.erase(
          std::remove(temp_solution.begin(), temp_solution.end(), p),
          temp_solution.end());
      // Map 状态更新，但 Off-tour best_cost/best_vertex
      // 不在这里更新，依赖最终的 GreedyLocalSearch
      iter_dic.at({p.x, p.y}).status = "N"; // 标记为已在路外
    }

    // 删除已处理的项目
    state_change.erase(state_change.begin() + max_idx);
    vertice_probability.erase(max_prob_it);
  }

  // 计算最终成本
  GreedyLocalSearch calculater(temp_solution, iter_dic, distance_);
  double relinkcost = calculater.tabu_cacl_cost();

  return {temp_solution, relinkcost};
}

std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>>
TabuSearch::diversication(const std::vector<std::vector<Point>> &solution_set,
                          std::map<std::pair<int, int>, VertexInfo> iter_dic,
                          int number) {
  auto current_solution = solution_set.back();

  // 1. 区分 On/Off Vertice
  std::vector<Point> on_vertice;
  std::vector<Point> off_vertice;
  std::unordered_set<Point, PointHash> current_set(current_solution.begin(),
                                                   current_solution.end());

  for (const auto &loc : locations_) {
    if (current_set.count(loc) > 0) {
      on_vertice.push_back(loc); // 在路径上
    } else {
      off_vertice.push_back(loc); // 不在路径上
    }
  }
  // 2. 计算概率
  std::vector<double>
      onvert_probability; // On-tour 点的“缺席”概率 (希望移除缺席概率最高的)
  std::vector<double>
      offvert_probability; // Off-tour 点的“出席”概率 (希望加入出席概率最高的)

  int len_set = solution_set.size();

  // 计算 On-Tour 点的缺席概率
  for (const auto &onvert : on_vertice) {
    int absent_count = 0;
    for (const auto &sol : solution_set) {
      if (std::find(sol.begin(), sol.end(), onvert) == sol.end()) {
        absent_count++;
      }
    }
    onvert_probability.push_back(static_cast<double>(absent_count) / len_set);
  }

  // 计算 Off-Tour 点的出席概率 (1.0 - 缺席概率)
  for (const auto &offvert : off_vertice) {
    int absent_count = 0;
    for (const auto &sol : solution_set) {
      if (std::find(sol.begin(), sol.end(), offvert) == sol.end()) {
        absent_count++;
      }
    }
    offvert_probability.push_back(
        1.0 - (static_cast<double>(absent_count) / len_set));
  }

  // 3. 执行 Diversification 交换 (number 次)
  // 复制列表，因为要在循环中动态删除
  auto onvert_prob_copy = onvert_probability;
  auto offvert_prob_copy = offvert_probability;
  auto on_vertice_copy = on_vertice;
  auto off_vertice_copy = off_vertice;

  for (int i = 0; i < number; i++) {
    if (onvert_prob_copy.empty() || offvert_prob_copy.empty())
      break;

    // 1. 找出 On-tour 点中，缺席概率最高的 (最应该被移除的)
    auto max_onvert_prob_it =
        std::max_element(onvert_prob_copy.begin(), onvert_prob_copy.end());
    size_t max_onvert_prob_index =
        std::distance(onvert_prob_copy.begin(), max_onvert_prob_it);

    // 2. 找出 Off-tour 点中，出席概率最高的 (最应该被加入的)
    auto max_offvert_prob_it =
        std::max_element(offvert_prob_copy.begin(), offvert_prob_copy.end());
    size_t max_offvert_prob_index =
        std::distance(offvert_prob_copy.begin(), max_offvert_prob_it);

    Point max_onvert = on_vertice_copy[max_onvert_prob_index];
    Point max_offvert = off_vertice_copy[max_offvert_prob_index];

    // 执行替换操作：将 max_onvert 替换为 max_offvert
    auto it_to_replace =
        std::find(current_solution.begin(), current_solution.end(), max_onvert);
    if (it_to_replace != current_solution.end()) {
      *it_to_replace = max_offvert;
    } else {
      // 理论上不应发生
      continue;
    }

    auto key = [](const Point &p) { return std::pair{p.x, p.y}; };

    // 更新 Map 状态
    iter_dic.at(key(max_onvert)).status = "N";  // 换出 -> 路外
    iter_dic.at(key(max_offvert)).status = "Y"; // 换进 -> 路内

    // 从概率列表中移除已选择的项 (注意顺序：先概率后点)
    onvert_prob_copy.erase(max_onvert_prob_it);
    offvert_prob_copy.erase(max_offvert_prob_it);

    // 从点列表中移除已选择的项
    on_vertice_copy.erase(on_vertice_copy.begin() + max_onvert_prob_index);
    off_vertice_copy.erase(off_vertice_copy.begin() + max_offvert_prob_index);

    // 4. 更新所有路外点 (status == "N") 的最近邻信息
    // 只有 max_onvert 变为路外点，其他路外点需要检查 max_offvert
    // 是否成为新的最近邻。 但最安全的是重新计算所有路外点到新路径的最近邻
    for (auto &[pt, info] : iter_dic) {
      if (info.status != "N")
        continue;

      double best_cost = std::numeric_limits<double>::max();
      Point best_pred{};

      for (const Point &on : current_solution) {
        // 确保 on 点在字典中，且状态为 Y (虽然 Map 状态已更新，保险起见)
        if (iter_dic.at({on.x, on.y}).status == "Y") {
          size_t i = iter_dic.at({on.x, on.y}).index;
          size_t j = info.index;

          double cost = distance_[i][j];
          if (cost < best_cost) {
            best_cost = cost;
            best_pred = on;
          }
        }
      }

      info.best_vertex = best_pred;
      info.best_cost = best_cost;
    }
  }

  return {current_solution, iter_dic};
}

void TabuSearch::search(int T, int Q, int TBL) {
  int t = 0, q = 0;
  // t q 代表当前已经执行了路径重连和多样化的次数

  std::vector<std::vector<Point>> solution_set;
  solution_set.push_back(route_);

  // 当前最优解
  // 对应itersolution = copy.copy(firstsolution)
  std::vector<Point> current_sol = route_;
  // 当前的迭代解
  auto current_dic = vertex_map_;

  TabuInfo tabu(TBL);
  // 初始化禁忌搜索对象

  int iter_count = 0;

  while (iter_count < MAX_TOTAL_ITER) {
    iter_count++;

    // 1. 生成邻域 (Candidate List)
    using Candidate =
        std::tuple<double, std::vector<Point>,
                   std::map<std::pair<int, int>, VertexInfo>, OpKey>;
    std::vector<Candidate> candidates;

    int valid_neighbors = 0;
    int attempts = 0;
    while (valid_neighbors < 50 && attempts < 200) { // 尝试生成 50 个邻域解
      attempts++;
      auto [n_route, n_dic, n_cost, n_op] =
          operation_style(current_sol, current_dic);

      if (n_op.empty() || n_cost == std::numeric_limits<double>::max())
        continue;

      OpKey key;
      if (n_op.size() == 2) {
        std::vector<Point> rev = {n_op[1], n_op[0]};
        key = std::make_pair(n_op, rev);
      } else {
        key = n_op; // Add/Drop: size 1
      }

      // 检查这个操作是否已经被尝试过 (避免重复计算)
      bool already_seen = false;
      for (const auto &cand : candidates) {
        if (std::get<3>(cand) == key) {
          already_seen = true;
          break;
        }
      }
      if (already_seen)
        continue;

      candidates.emplace_back(n_cost, n_route, n_dic, key);
      valid_neighbors++;
    }

    if (candidates.empty())
      break;

    // 2. 选择最优邻域 (Best Fit)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) {
                return std::get<0>(a) < std::get<0>(b);
              });

    bool found_move = false;
    double best_candidate_cost = 0;
    OpKey move_key;

    for (const auto &cand : candidates) {
      double c_cost = std::get<0>(cand);
      const auto &c_key = std::get<3>(cand);

      bool is_tabu = tabu.is_tabu_iter(c_key);

      // Aspiration Criteria (渴望准则): 禁忌但更优 -> 破禁
      // OR 移动非禁忌
      if (!is_tabu || c_cost < best_cost_) {
        // 接受此移动
        current_sol = std::get<1>(cand);
        current_dic = std::get<2>(cand);
        best_candidate_cost = c_cost;
        move_key = c_key;
        found_move = true;
        break;
      }
    }

    if (!found_move) {
      // 如果找不到非禁忌或破禁忌的移动，强制停滞
      t++;
      tabu.update_tabu();
    } else {
      // 执行移动并更新禁忌表
      tabu.add_tabu_iter(move_key, TBL);
      tabu.update_tabu();

      // 3. 更新全局最优
      if (best_candidate_cost < best_cost_) {
        best_cost_ = best_candidate_cost;
        iter_solution_ = current_sol;
        solution_set.push_back(current_sol);
        cost_trend_.push_back(best_cost_);
        t = 0; // 重置停滞计数器

        // 路径重连
        auto [pr_sol, pr_cost] = path_relinking(solution_set, current_dic);
        if (pr_cost < best_cost_) {
          best_cost_ = pr_cost;
          iter_solution_ = pr_sol;
          current_sol = pr_sol;
          // 更新 solution_set 中的最优解
          if (!solution_set.empty())
            solution_set.back() = pr_sol;
          cost_trend_.push_back(best_cost_);
        }
      } else {
        t++; // 停滞
      }
    }

    // 4. 多样化 (Diversification)
    if (t >= T) {
      q++;
      if (q > Q)
        break; // 达到最大多样化次数，退出

      auto [div_sol, div_dic] =
          diversication(solution_set, current_dic, DIVERSICATION_TIMES);
      current_sol = div_sol;
      current_dic = div_dic;

      // 重新计算多样化后的成本作为新的初始解成本
      GreedyLocalSearch calc(current_sol, current_dic,distance_);
      double div_cost = calc.tabu_cacl_cost();

      if (div_cost < best_cost_) {
        best_cost_ = div_cost;
        iter_solution_ = current_sol;
        cost_trend_.push_back(best_cost_);
      }

      solution_set.push_back(current_sol); // 记录新的解

      t = 0;
      tabu.reset_list(); // 多样化后通常重置禁忌表
    }
  }

  // 结束时确保返回的 solution 是最优解
  iter_solution_ = iter_solution_;
  best_cost_ = best_cost_;
}