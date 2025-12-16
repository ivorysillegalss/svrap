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
      solution_cost_(cost), iter_solution_(route), best_cost_(cost),
      champion_solution_(route), champion_vertex_map_(vertex_map),
      champion_cost_(cost) {
  cost_trend_.push_back(cost);
  // 初始解视为第一个 Champion，初始化频率统计
  update_champion_frequencies(champion_solution_);

  // Initialize nearby_table_
  size_t n = locations_.size();
  nearby_table_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<std::pair<double, int>> dists;
    dists.reserve(n);
    for (size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      dists.push_back({distance_[i][j], static_cast<int>(j)});
    }
    // Sort by distance
    std::sort(dists.begin(), dists.end());
    
    // Keep top K
    int k = std::min((int)dists.size(), K_NEIGHBORS);
    for (int j = 0; j < k; ++j) {
      nearby_table_[i].push_back(dists[j].second);
    }
  }
}
std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>,
           double, std::vector<Point>>
TabuSearch ::operation_style(
    const std::vector<Point> &route,
    const std::map<std::pair<int, int>, VertexInfo> &iter_dic,
    double base_allocation_cost) {

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
        0, static_cast<int>(candidates.size()) - 1)(rng)];

    // 更新 Map 状态
    add_dic[{add_vertice.x, add_vertice.y}].status = "Y";

    // 更新其他 Off-tour 点的最近邻（保守实现，交给成本函数兜底）
    std::pair<int, int> add_key = {add_vertice.x, add_vertice.y};
    size_t add_idx = add_dic.at(add_key).index;

    for (auto &kv : add_dic) {
      if (kv.second.status == "N") {
        size_t off_idx = kv.second.index;
        double new_dist = distance_[off_idx][add_idx];
        if (new_dist < kv.second.best_cost) {
          kv.second.best_cost = new_dist;
          kv.second.best_vertex = add_vertice;
        }
      }
    }

    // Optimization: Calculate allocation cost ONCE for this set of points
    std::vector<Point> alloc_route = route;
    alloc_route.push_back(add_vertice);
    double new_allocation_cost = GreedyLocalSearch::compute_allocation_cost(alloc_route, add_dic, distance_);

    // 寻找该点在当前路径中的最佳插入位置
    std::vector<Point> best_route;
    double min_cost = std::numeric_limits<double>::max();

    // K-NN Optimization: Only check insertion positions adjacent to neighbors
    // First, identify which indices in the current route are neighbors of add_vertice
    std::vector<size_t> candidate_indices;
    const auto& neighbors = nearby_table_[add_idx];
    
    // Build a quick lookup for route indices? Or just iterate route.
    // Since K is small (20), and route size is N, iterating route and checking if it's a neighbor is O(N*K) or O(N) with hash set.
    // But iterating neighbors and finding them in route is O(K*N) with linear scan.
    // Let's just iterate the route once.
    
    // Optimization: We need to check insertion at i (before route[i]) and i+1 (after route[i]).
    // If route[i] is a neighbor, we should check insertion near it.
    
    // Fallback: If no neighbors are in the route (rare), we might want to check all?
    // Or just check all if route is small.
    
    bool use_knn = true;
    if (route.size() < 10) use_knn = false; // For very small routes, just check all

    if (!use_knn) {
        for (size_t i = 0; i <= route.size(); ++i) {
            candidate_indices.push_back(i);
        }
    } else {
        // Use a boolean array for fast neighbor check
        // Since N is small (up to 1000?), a vector<bool> is fast.
        // But we don't want to allocate vector<bool> every time.
        // Given K is small, we can just iterate K neighbors for each node in route? No, that's O(N*K).
        // Better: Iterate neighbors, find their position in route.
        // But we don't know their position in route without scanning route.
        // So scanning route is inevitable unless we maintain a map.
        // Let's scan route. For each node, is it in neighbors?
        // To make "is it in neighbors" fast, we can sort neighbors or use a set?
        // Actually, for K=20, linear scan of neighbors is fast enough.
        
        std::vector<bool> is_neighbor(locations_.size(), false);
        for (int neighbor_idx : neighbors) {
            is_neighbor[neighbor_idx] = true;
        }

        for (size_t i = 0; i < route.size(); ++i) {
            size_t u_idx = iter_dic.at({route[i].x, route[i].y}).index;
            if (is_neighbor[u_idx]) {
                // If route[i] is a neighbor, check insertion before (i) and after (i+1)
                candidate_indices.push_back(i);
                candidate_indices.push_back(i + 1);
            }
        }
        
        // Remove duplicates and sort
        std::sort(candidate_indices.begin(), candidate_indices.end());
        candidate_indices.erase(std::unique(candidate_indices.begin(), candidate_indices.end()), candidate_indices.end());
        
        // If candidate_indices is empty (no neighbors in route), fall back to full scan
        if (candidate_indices.empty()) {
             for (size_t i = 0; i <= route.size(); ++i) {
                candidate_indices.push_back(i);
            }
        }
    }

    for (size_t i : candidate_indices) {
      std::vector<Point> temp_route = route;
      temp_route.insert(temp_route.begin() + i, add_vertice);

      // Only recompute routing cost
      double routing_cost = GreedyLocalSearch::compute_routing_cost(temp_route, add_dic, distance_);
      double c = routing_cost + new_allocation_cost;

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

    
    // 随机删除一个位置上的顶点
    int drop_index =
        std::uniform_int_distribution<int>(0,
                                           static_cast<int>(drop_route.size()) - 1)(rng);
    Point d_vertice = drop_route[drop_index];

    drop_route.erase(drop_route.begin() + drop_index);

    // 更新 Map
    VertexInfo &info = drop_dic[{d_vertice.x, d_vertice.y}];
    info.status = "N";

    // 其余最近邻信息交由成本函数统一处理
    // GreedyLocalSearch cal(drop_route, drop_dic, distance_);
    // double i_cost = cal.tabu_cacl_cost();
    double i_cost = GreedyLocalSearch::compute_cost(drop_route, drop_dic, distance_);

    return {drop_route, drop_dic, i_cost, {d_vertice}};

  } else if (style_number == TWOOPT) {
    if (route.size() < 2)
      return {route, iter_dic, std::numeric_limits<double>::max(), {}};

    std::vector<size_t> indices(route.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<size_t> idx(2);
    // 随机选两个不同的索引进行交换
    std::sample(indices.begin(), indices.end(), idx.begin(), 2, rng);

    std::vector<Point> new_route = route;
    std::swap(new_route[idx[0]], new_route[idx[1]]);

    // Optimization: Allocation cost is unchanged!
    double routing_cost = GreedyLocalSearch::compute_routing_cost(new_route, iter_dic, distance_);
    double new_cost = routing_cost + base_allocation_cost;

    std::vector<Point> operated = {route[idx[0]], route[idx[1]]};
    return {new_route, iter_dic, new_cost, operated};

  } else {
    return {route, iter_dic, std::numeric_limits<double>::max(), {}};
  }
}
// 辅助函数：插入最佳位置
std::vector<Point>
best_insert_position(const Point &p, const std::vector<Point> &route,
                     const std::map<std::pair<int, int>, VertexInfo> &dic,
                     const std::vector<std::vector<double>> &distance) {
  std::vector<Point> best_route = route;
  double best_cost = std::numeric_limits<double>::max();

  // Optimization: Allocation cost is constant for all insertion positions
  std::vector<Point> alloc_route = route;
  alloc_route.push_back(p);
  double allocation_cost = GreedyLocalSearch::compute_allocation_cost(alloc_route, dic, distance);

  // 简单的插入逻辑，为了性能不全量 Recalculate，但为了准确这里还是调 Greedy
  for (size_t i = 0; i <= route.size(); ++i) {
    auto test_route = route;
    test_route.insert(test_route.begin() + i, p);
    
    double routing_cost = GreedyLocalSearch::compute_routing_cost(test_route, dic, distance);
    double cost = routing_cost + allocation_cost;

    if (cost < best_cost) {
      best_cost = cost;
      best_route = test_route;
    }
  }
  return best_route;
}

std::tuple<std::vector<Point>, double>
TabuSearch::path_relinking(const std::vector<Point> &prev_champion,
                           const std::vector<Point> &new_champion,
                           std::map<std::pair<int, int>, VertexInfo> iter_dic,
                           std::vector<OpKey> &relink_moves) {
  // 如果缺少前一个 Champion，无法进行路径重连
  if (prev_champion.empty() || new_champion.empty()) {
    return {new_champion, solution_cost_};
  }

  // 找出两个解之间“状态不同”的点
  std::unordered_set<Point, PointHash> current_set(new_champion.begin(),
                                                   new_champion.end());
  std::unordered_set<Point, PointHash> last_set(prev_champion.begin(),
                                                prev_champion.end());
  std::vector<Point> state_change;

  // last → current 中消失的点（从路径中被移出）
  // 1. 找出 target 有但 base 没有的点 (需要插入)
  for (const Point &p : prev_champion) {
    if (current_set.count(p) == 0) {
      state_change.push_back(p);
    }
  }

  // current → last 中新增的点（被插入路径）
  // 找出 base 有但 target 没有的点 (需要删除)
  for (const Point &p : new_champion) {
    if (last_set.count(p) == 0) {
      state_change.push_back(p);
    }
  }
  if (state_change.empty()) {
    // GreedyLocalSearch calc(prev_champion, iter_dic, distance_);
    // return {prev_champion, calc.tabu_cacl_cost()};
    return {prev_champion, GreedyLocalSearch::compute_cost(prev_champion, iter_dic, distance_)};
  }

  // 2. 基于 Champion 频率计算 m(i)
  std::vector<double> vertice_on_probability;
  vertice_on_probability.reserve(state_change.size());

  for (const Point &p : state_change) {
    std::pair<int, int> key = {p.x, p.y};
    int on_cnt = 0;
    int off_cnt = 0;
    auto it_on = champion_on_count_.find(key);
    if (it_on != champion_on_count_.end()) {
      on_cnt = it_on->second;
    }
    auto it_off = champion_off_count_.find(key);
    if (it_off != champion_off_count_.end()) {
      off_cnt = it_off->second;
    }
    int total = std::max(1, on_cnt + off_cnt);
    vertice_on_probability.push_back(static_cast<double>(on_cnt) /
                                     static_cast<double>(total));
  }

  // 3. 构建最终的概率列表 (vertice_probability)
  std::vector<double> vertice_probability;
  vertice_probability.reserve(state_change.size());

  for (size_t i = 0; i < state_change.size(); ++i) {
    const Point &p = state_change[i];
    bool is_in_current = current_set.count(p) > 0;

    if (is_in_current) {
      // m(i) = m_on(i) 当在 Champion B 中为 on-tour
      vertice_probability.push_back(vertice_on_probability[i]);
    } else {
      // m(i) = m_off(i) = 1 - m_on(i) 当在 Champion B 中为 off-tour
      vertice_probability.push_back(1.0 - vertice_on_probability[i]);
    }
  }

  // 4. 迭代移动：从 last_solution 转向 current_solution
  auto temp_solution = prev_champion;

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
      // 记录一次 ADD 操作，用于将其逆操作 (DROP) 设为 Tabu
      std::vector<Point> op_vec = {p};
      OpKey key = op_vec;
      relink_moves.push_back(key);
    } else {
      // 该点最终不在路径中 → 直接删除
      temp_solution.erase(
          std::remove(temp_solution.begin(), temp_solution.end(), p),
          temp_solution.end());
      // Map 状态更新，但 Off-tour best_cost/best_vertex
      // 不在这里更新，依赖最终的 GreedyLocalSearch
      iter_dic.at({p.x, p.y}).status = "N"; // 标记为已在路外
      // 记录一次 DROP 操作，用于将其逆操作 (ADD) 设为 Tabu
      std::vector<Point> op_vec = {p};
      OpKey key = op_vec;
      relink_moves.push_back(key);
    }

    // 删除已处理的项目
    state_change.erase(state_change.begin() + max_idx);
    vertice_probability.erase(max_prob_it);
  }

  // 计算最终成本
  // GreedyLocalSearch calculater(temp_solution, iter_dic, distance_);
  // double relinkcost = calculater.tabu_cacl_cost();
  double relinkcost = GreedyLocalSearch::compute_cost(temp_solution, iter_dic, distance_);

  return {temp_solution, relinkcost};
}

std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>>
TabuSearch::diversication(const std::vector<Point> &champion_route,
                          std::map<std::pair<int, int>, VertexInfo> iter_dic,
                          std::vector<OpKey> &diversification_moves) {
  auto current_solution = champion_route;

  // 1. 区分 On/Off Vertice（基于 Champion 解）
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
  // 2. 计算 m_on(i)、m_off(i) 并基于 m(i) = m_on(i) 或 m_off(i)
  //    进行多样化。这里直接从 Champion 频率统计中读取。
  std::vector<std::pair<Point, double>> candidate_vertices;
  candidate_vertices.reserve(locations_.size());

  for (const auto &loc : locations_) {
    std::pair<int, int> key = {loc.x, loc.y};
    int on_cnt = 0;
    int off_cnt = 0;
    auto it_on = champion_on_count_.find(key);
    if (it_on != champion_on_count_.end()) {
      on_cnt = it_on->second;
    }
    auto it_off = champion_off_count_.find(key);
    if (it_off != champion_off_count_.end()) {
      off_cnt = it_off->second;
    }
    int total = std::max(1, on_cnt + off_cnt);

    bool is_on = current_set.count(loc) > 0;
    double m_i = 0.0;
    if (is_on) {
      m_i = static_cast<double>(on_cnt) / static_cast<double>(total);
    } else {
      m_i = static_cast<double>(off_cnt) / static_cast<double>(total);
    }
    candidate_vertices.push_back({loc, m_i});
  }

  // 按照 m(i) 降序排序
  std::sort(candidate_vertices.begin(), candidate_vertices.end(),
            [](const auto &a, const auto &b) {
              return a.second > b.second;
            });

  // 3. 对前 n/2 个顶点执行“状态反转” (ADD 或 DROP)
  int n = static_cast<int>(locations_.size());
  int diversify_moves = n / 2;

  auto key_of = [](const Point &p) { return std::pair{p.x, p.y}; };

  for (int k = 0; k < diversify_moves && k < (int)candidate_vertices.size();
       ++k) {
    const Point &p = candidate_vertices[k].first;
    bool is_on = current_set.count(p) > 0;
    auto pt_key = key_of(p);

    if (is_on) {
      // DROP: 从路径中删除
      auto it = std::find(current_solution.begin(), current_solution.end(), p);
      if (it == current_solution.end())
        continue;
      current_solution.erase(it);
      iter_dic.at(pt_key).status = "N";

      // 记录 Tabu 反向 (ADD) 的 Key
      std::vector<Point> op_vec = {p};
      OpKey key = op_vec;
      diversification_moves.push_back(key);
    } else {
      // ADD: 插入到路径中，对应成本增加最小的位置
      current_solution =
          best_insert_position(p, current_solution, iter_dic, distance_);
      iter_dic.at(pt_key).status = "Y";

      std::vector<Point> op_vec = {p};
      OpKey key = op_vec;
      diversification_moves.push_back(key);
      current_set.insert(p);
    }
  }

  // 4. 更新所有路外点 (status == "N") 的最近邻信息
  for (auto &[pt, info] : iter_dic) {
    if (info.status != "N")
      continue;

    double best_cost = std::numeric_limits<double>::max();
    Point best_pred{};

    for (const Point &on : current_solution) {
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

  return {current_solution, iter_dic};
}

void TabuSearch::update_champion_frequencies(
    const std::vector<Point> &champion_route) {
  champion_sample_count_++;

  std::unordered_set<Point, PointHash> route_set(champion_route.begin(),
                                                 champion_route.end());

  for (const auto &loc : locations_) {
    std::pair<int, int> key = {loc.x, loc.y};
    if (route_set.count(loc) > 0) {
      champion_on_count_[key]++;
    } else {
      champion_off_count_[key]++;
    }
  }
}

void TabuSearch::search(int T, int Q, int TBL) {
  int q = 0; // 已执行的多样化次数

  // 当前解
  std::vector<Point> current_sol = route_;
  // 当前的迭代解
  auto current_dic = vertex_map_;

  TabuInfo tabu(TBL);

  // 统计自上一个 Champion 以来的 MOVE 计数
  int move_since_champion_adddrop = 0;
  int move_since_champion_twoopt = 0;

  int iter_count = 0;

  // 计算当前解的成本，用于判断“改进”或“非改进”移动
  // GreedyLocalSearch init_calc(current_sol, current_dic, distance_);
  // double current_cost = init_calc.tabu_cacl_cost();
  double current_cost = GreedyLocalSearch::compute_cost(current_sol, current_dic, distance_);

  while (iter_count < MAX_TOTAL_ITER) {
    iter_count++;

    // Calculate allocation cost for current solution to pass to operation_style
    double current_allocation_cost = GreedyLocalSearch::compute_allocation_cost(current_sol, current_dic, distance_);

    // Calculate lambda(t)
    double progress = static_cast<double>(iter_count) / MAX_TOTAL_ITER;
    double lambda_t = lambda_0_ * (1.0 - progress);

    // 1. 生成邻域 (Candidate List) — 使用随机 operation_style 与上一次相同
    // Added adjusted_cost as the 5th element
    using Candidate =
        std::tuple<double, std::vector<Point>,
                   std::map<std::pair<int, int>, VertexInfo>, OpKey, double>;
    std::vector<Candidate> candidates;

    int valid_neighbors = 0;
    int attempts = 0;
    const int MAX_NEIGHBORS = 50;
    while (valid_neighbors < MAX_NEIGHBORS && attempts < 200) {
      attempts++;
      auto [n_route, n_dic, n_cost, n_op] =
          operation_style(current_sol, current_dic, current_allocation_cost);

      if (n_op.empty() ||
          n_cost == std::numeric_limits<double>::max())
        continue;

      OpKey key;
      if (n_op.size() == 2) {
        std::vector<Point> rev = {n_op[1], n_op[0]};
        key = std::make_pair(n_op, rev);
      } else {
        key = n_op; // Add/Drop: size 1
      }

      bool already_seen = false;
      for (const auto &cand : candidates) {
        if (std::get<3>(cand) == key) {
          already_seen = true;
          break;
        }
      }
      if (already_seen)
        continue;

      // Calculate entropy score s
      double s = 0.0;
      int he_count = 0;
      int total_points = 0;
      for (const auto &p : n_op) {
        total_points++;
        if (current_dic.at({p.x, p.y}).is_high_entropy) {
          he_count++;
        }
      }
      if (total_points > 0) {
        s = static_cast<double>(he_count) / total_points;
      }

      double adjusted_cost = n_cost - lambda_t * s;

      candidates.emplace_back(n_cost, n_route, n_dic, key, adjusted_cost);
      valid_neighbors++;
    }

    if (candidates.empty())
      break;

    // 2. 局部改进阶段 (4a)：按 TWOOPT → DROP → ADD 顺序寻找改进移动
    // Sort by raw cost for improving moves
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) {
                return std::get<0>(a) < std::get<0>(b);
              });

    bool found_improving_move = false;
    double chosen_cost = 0.0;
    OpKey move_key;
    int move_type = 0;
    const int style_order[3] = {TWOOPT, DROP, ADD};

    for (int style : style_order) {
      for (const auto &cand : candidates) {
        double c_cost = std::get<0>(cand);
        const auto &cand_route = std::get<1>(cand);
        const auto &cand_dic = std::get<2>(cand);
        const auto &c_key = std::get<3>(cand);
        // double c_adj_cost = std::get<4>(cand);

        int old_size = static_cast<int>(current_sol.size());
        int new_size = static_cast<int>(cand_route.size());
        int this_type;
        if (new_size > old_size)
          this_type = ADD;
        else if (new_size < old_size)
          this_type = DROP;
        else
          this_type = TWOOPT;

        if (this_type != style)
          continue; // 按顺序先看 TWOOPT，再 DROP，再 ADD

        bool is_tabu = tabu.is_tabu_iter(c_key);
        bool aspiration = (c_cost < champion_cost_ - 1e-9);

        // 仅接受改进移动 (c_cost < current_cost)，且满足禁忌/破禁规则
        // Improving moves selection is based on raw cost
        if ((is_tabu && !aspiration) || c_cost >= current_cost - 1e-9)
          continue;

        // 接受该改进移动
        current_sol = cand_route;
        current_dic = cand_dic;
        current_cost = c_cost;
        chosen_cost = c_cost;
        move_key = c_key;
        move_type = this_type;
        found_improving_move = true;
        break;
      }
      if (found_improving_move)
        break;
    }

    bool performed_diversification = false;

    // 若没有找到改进移动，则根据 4c/4d 进行多样化或选取最佳非改进移动
    if (!found_improving_move) {
      // 4c. 若从当前 Champion 出发已执行足够多 ADD/DROP 或 TWOOPT
      // 移动而仍未发现新 Champion，则多样化
      const int ADD_DROP_LIMIT = 20;
      const int TWOOPT_LIMIT = 15;

      if (move_since_champion_adddrop >= ADD_DROP_LIMIT ||
          move_since_champion_twoopt >= TWOOPT_LIMIT) {
        q++;
        if (q >= Q)
          break; // 两次多样化后终止

        tabu.reset_list();

        std::vector<OpKey> div_moves;
        auto [div_sol, div_dic] =
            diversication(champion_solution_, champion_vertex_map_,
                          div_moves);
        current_sol = div_sol;
        current_dic = div_dic;

        for (const auto &op : div_moves) {
          tabu.add_tabu_iter(op, TBL);
        }
        tabu.update_tabu();

        // GreedyLocalSearch calc(current_sol, current_dic, distance_);
        // double div_cost = calc.tabu_cacl_cost();
        double div_cost = GreedyLocalSearch::compute_cost(current_sol, current_dic, distance_);
        current_cost = div_cost;

        if (div_cost < best_cost_ - 1e-9) {
          champion_solution_ = current_sol;
          champion_vertex_map_ = current_dic;
          champion_cost_ = div_cost;
          best_cost_ = div_cost;
          iter_solution_ = current_sol;
          cost_trend_.push_back(best_cost_);

          update_champion_frequencies(champion_solution_);
        }

        move_since_champion_adddrop = 0;
        move_since_champion_twoopt = 0;
        performed_diversification = true;

        // 进入下一轮迭代
        continue;
      }

      // 4d. 若既未触发路径重连也未多样化，则选择“最佳非改进”移动
      // Sort candidates by adjusted_cost for non-improving moves
      std::sort(candidates.begin(), candidates.end(),
                [](const auto &a, const auto &b) {
                  return std::get<4>(a) < std::get<4>(b);
                });

      bool found_non_improving = false;
      for (const auto &cand : candidates) {
        double c_cost = std::get<0>(cand);
        const auto &cand_route = std::get<1>(cand);
        const auto &cand_dic = std::get<2>(cand);
        const auto &c_key = std::get<3>(cand);
        // double c_adj_cost = std::get<4>(cand);

        bool is_tabu = tabu.is_tabu_iter(c_key);
        if (is_tabu)
          continue; // 4d 要求非禁忌

        if (c_cost < current_cost - 1e-9)
          continue; // 这里只考虑非改进移动 (improving moves were handled in 4a)

        // 判定移动类型
        int old_size = static_cast<int>(current_sol.size());
        int new_size = static_cast<int>(cand_route.size());
        if (new_size > old_size)
          move_type = ADD;
        else if (new_size < old_size)
          move_type = DROP;
        else
          move_type = TWOOPT;

        current_sol = cand_route;
        current_dic = cand_dic;
        current_cost = c_cost;
        chosen_cost = c_cost;
        move_key = c_key;
        found_non_improving = true;
        break; // 借由 candidates 已排序 (by adjusted_cost)，首个满足者即为“最佳”
      }

      if (!found_non_improving) {
        // 没有任何可行的非改进移动
        tabu.update_tabu();
        continue;
      }
    }

    // 执行移动并更新禁忌表（包括 4a 的改进移动或 4d 的非改进移动）
    tabu.add_tabu_iter(move_key, TBL);
    tabu.update_tabu();

    // 统计自当前 Champion 以来的移动次数
    if (move_type == TWOOPT) {
      move_since_champion_twoopt++;
    } else {
      move_since_champion_adddrop++;
    }

    // 3. 更新 Champion（全局最优解）及路径重连（4b）
    if (chosen_cost < champion_cost_ - 1e-9) {
      prev_champion_solution_ = champion_solution_;
      champion_solution_ = current_sol;
      champion_vertex_map_ = current_dic;
      champion_cost_ = chosen_cost;
      best_cost_ = chosen_cost;
      iter_solution_ = current_sol;
      cost_trend_.push_back(best_cost_);

      update_champion_frequencies(champion_solution_);

      // 路径重连：在前一个 Champion 与新 Champion 之间
      std::vector<OpKey> pr_moves;
      auto [pr_sol, pr_cost] =
          path_relinking(prev_champion_solution_, champion_solution_,
                         champion_vertex_map_, pr_moves);

      for (const auto &op : pr_moves) {
        tabu.add_tabu_iter(op, TBL);
      }
      tabu.update_tabu();

      if (pr_cost < best_cost_ - 1e-9) {
        champion_solution_ = pr_sol;
        champion_cost_ = pr_cost;
        best_cost_ = pr_cost;
        iter_solution_ = pr_sol;
        cost_trend_.push_back(best_cost_);
        current_sol = pr_sol;
        current_cost = pr_cost;

        update_champion_frequencies(champion_solution_);
      }

      move_since_champion_adddrop = 0;
      move_since_champion_twoopt = 0;
    }
  }

  // 结束时确保返回的 solution 是最优解
  iter_solution_ = iter_solution_;
  best_cost_ = best_cost_;
}