#include "tabu_search.h"
#include "greedy.h"
#include "input.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// 指选出两条最不相似的路径
#define DIVERSICATION_TIMES 2

TabuSearch::TabuSearch(
    const std::vector<Point> &locations,
    const std::vector<std::vector<double>> &distance,
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<Point> &route, const std::double_t &cost)
    : locations_(locations), distance_(distance), ontour_(ontour),
      offtour_(offtour), vertex_map_(vertex_map), route_(route),
      solution_cost_(cost) {
  cost_trend_.push_back(cost);
}

std::vector<int> TabuSearch::get_len_trend() { return cost_trend_; }
std::vector<Point> TabuSearch::get_iter_solution() { return iter_solution_; }
std::double_t TabuSearch::get_best_cost() { return best_cost_; }

void TabuInfo::add_tabu_num() { current_tabu_size++; }

void TabuInfo::reset_list() {
  tabu_list.clear();
  tabu_time.clear();
  current_tabu_size = 0;
}

void TabuInfo::update_tabu() {
  for (int i = 0; i < tabu_time.size(); ++i) {
    tabu_time[i]--;
    if (tabu_time[i] == 0) {
      tabu_list[i] = std::vector<Point>{};
    }
  }

  // 索引更新移除非0项 压缩迁移后数组
  size_t write_idx = 0;
  for (size_t i = 0; i < tabu_list.size(); ++i) {
    if (tabu_time[i] != 0) {
      if (write_idx != i) {
        tabu_list[write_idx] = std::move(tabu_list[i]);
        tabu_time[write_idx] = tabu_time[i];
      }
      ++write_idx;
    }
  }
  tabu_list.resize(write_idx);
  tabu_time.resize(write_idx);
};

void TabuInfo::set_limit(int new_limit) {
  tabu_limit = new_limit;
  return;
}

// 判断当前禁忌表中是否存在当前遍历到的元素
bool TabuInfo::is_tabu_iter(
    std::variant<std::vector<Point>,
                 std::pair<std::vector<Point>, std::vector<Point>>>
        iter) {
  return std::find(tabu_list.begin(), tabu_list.end(), iter) == tabu_list.end();
}

void TabuInfo ::add_tabu_iter(
    std::variant<std::vector<Point>,
                 std::pair<std::vector<Point>, std::vector<Point>>>
        iter,
    int tabu_limit) {
  tabu_list.push_back(iter);
  tabu_time.push_back(tabu_limit);
  current_tabu_size++;
}

std::tuple<std::vector<Point>, double>
TabuSearch::path_relinking(std::vector<std::vector<Point>> solution_set,
                           std::map<std::pair<int, int>, VertexInfo> iter_dic) {
  // 前置长度校验
  if (solution_set.size() < 2) {
    return {solution_set.back(), solution_cost_};
  }

  const std::vector<Point> &last_solution =
      solution_set[solution_set.size() - 2]; // 倒数第二个
  const std::vector<Point> &current_solution =
      solution_set.back(); // 最后一个（当前最优）

  // 找出两个解之间“状态不同”的点
  std::vector<Point> state_change;

  // last → current 中消失的点（从路径中被移出）
  for (const Point &p : last_solution) {
    if (std::find(current_solution.begin(), current_solution.end(), p) ==
        current_solution.end()) {
      state_change.push_back(p);
    }
  }

  // current → last 中新增的点（被插入路径）
  for (const Point &p : current_solution) {
    if (std::find(last_solution.begin(), last_solution.end(), p) ==
        last_solution.end()) {
      state_change.push_back(p);
    }
  }
  const size_t n_solutions = solution_set.size();

  // 1. 计算每个改变点在历史解集中出现的比例（vertice_on_probabity）
  std::vector<double> vertice_on_probability;
  vertice_on_probability.reserve(state_change.size());

  for (const Point &p : state_change) {
    int count = 0;
    for (const auto &sol : solution_set) {
      if (std::find(sol.begin(), sol.end(), p) != sol.end()) {
        ++count;
      }
    }
    vertice_on_probability.push_back(static_cast<double>(count) / n_solutions);
  }

  // 2. 构建最终的概率列表：
  //    - 若该点在 current_solution 中 → 用它出现的比例（on_tour）
  //    - 若不在 → 用 1−比例（off_tour）
  std::vector<double> vertice_probability;
  vertice_probability.reserve(state_change.size());

  int len_sc = state_change.size();
  for (size_t i = 0; i < len_sc; ++i) {
    const Point &p = state_change[i];
    bool is_in_current =
        std::find(current_solution.begin(), current_solution.end(), p) !=
        current_solution.end();

    if (is_in_current) {
      vertice_probability.push_back(vertice_on_probability[i]); // on_tour 比例
    } else {
      vertice_probability.push_back(1.0 -
                                    vertice_on_probability[i]); // off_tour 比例
    }
  }

  auto temp_solution = last_solution;
  for (int i = 0; i < len_sc; i++) {
    auto max_probability_p = std::max_element(vertice_probability.begin(),
                                              vertice_probability.end());
    double max_probability = *max_probability_p;
    int max_index =
        std::distance(max_probability_p, vertice_probability.begin());

    // TODO 感觉有一些find的语句条件写错了 记得review
    // TODO 这些find全部可以更换为unsorted_set的count api
    if (std::find(current_solution.begin(), current_solution.end(),
                  state_change[max_index]) != current_solution.end()) {

      std::vector<double> cost_list;
      for (int i = 0; i < temp_solution.size() + 1; i++) {
        double i_cost = 0;
        auto f_route = temp_solution;
        // 插入算子
        f_route.insert(f_route.begin() + i, state_change[max_index]);
        auto new_route = f_route;
        // 1. 遍历相邻点对，累加边长（对应 Python 的 for j in
        // range(len(new_route)-1)）
        for (size_t j = 0; j + 1 < new_route.size(); ++j) {
          int idx1 = iter_dic.at({new_route[j].x, new_route[j].y}).index;
          int idx2 =
              iter_dic.at({new_route[j + 1].x, new_route[j + 1].y}).index;
          i_cost += distance_[idx1][idx2];
        }

        // 2. 加上闭合边：最后一个点 → 第一个点（TSP 闭环）
        // 对应i_cost += distance[dic[new_route[0]][0]][dic[new_route[-1]][0]]
        if (!new_route.empty()) {
          int first_idx =
              iter_dic.at({new_route.front().x, new_route.front().y}).index;
          int last_idx =
              iter_dic.at({new_route.back().x, new_route.back().y}).index;
          i_cost += distance_[last_idx][first_idx];
        }
        cost_list.push_back(i_cost);
      }
      auto min_cost = std::min_element(cost_list.begin(), cost_list.end());
      int min_index = std::distance(cost_list.begin(), min_cost);
      temp_solution.insert(temp_solution.begin() + min_index,
                           state_change[max_index]);
      // 删除已处理的项目（完全等价于 Python 的几行 remove）
      state_change.erase(state_change.begin() + max_index);
      vertice_probability.erase(vertice_probability.begin() + max_index);
    } else {

      temp_solution.erase(std::remove(temp_solution.begin(),
                                      temp_solution.end(),
                                      state_change[max_index]),
                          temp_solution.end());

      // 删除已处理的项目（必须放在 remove 之后！）
      state_change.erase(state_change.begin() + max_index);
      vertice_probability.erase(vertice_probability.begin() + max_index);
    }
  }
  GreedyLocalSearch calculater(temp_solution, iter_dic);
  double relinkcost = calculater.tabu_cacl_cost();
  return {{temp_solution}, {relinkcost}};
}

std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>>
TabuSearch::diversication(std::vector<std::vector<Point>> solution_set,
                          std::map<std::pair<int, int>, VertexInfo> iter_dic,
                          int number) {
  auto current_solution = solution_set.back();
  std::vector<Point> on_vertice;
  std::vector<Point> off_vertice;
  for (int i = 0; i < locations_.size(); i++) {
    auto loc = locations_.at(i);
    if (std::find(current_solution.begin(), current_solution.end(), loc) ==
        current_solution.end()) {
      on_vertice.push_back(loc);
    } else {
      off_vertice.push_back(loc);
    }
  }
  std::vector<double> onvert_probability;
  std::vector<double> offvert_probability;
  int len_set = solution_set.size();
  for (int i = 0; i < on_vertice.size(); i++) {
    int onvert_pro = 0;
    auto onvert = on_vertice.at(i);
    for (int j = 0; j < len_set; j++) {
      if (std::find(solution_set.begin(), solution_set.end(), onvert) ==
          solution_set.end()) {
        onvert_pro++;
      }
    }
    onvert_probability.push_back(onvert_pro / len_set);
  }

  for (int i = 0; i < off_vertice.size(); i++) {
    int offvert_pro = 0;
    auto offvert = off_vertice.at(i);
    for (int j = 0; j < len_set; j++) {
      if (std::find(solution_set.begin(), solution_set.end(), off_vertice) ==
          solution_set.end()) {
        offvert_pro++;
      }
    }
    onvert_probability.push_back(offvert_pro / len_set);
  }

  auto onvert_prob = onvert_probability;
  auto offvert_prob = offvert_probability;

  for (int i = 0; i < number; i++) {
    auto max_onvert_prob_p =
        std::max_element(onvert_prob.begin(), onvert_prob.end());
    auto max_offvert_prob_p =
        std::max_element(offvert_prob.begin(), offvert_prob.end());

    double max_onvert_prob = *max_onvert_prob_p;
    double max_offvert_prob = *max_offvert_prob_p;

    // 等价于distance的api
    std::size_t max_offvert_prob_index =
        max_offvert_prob_p - offvert_prob.begin();
    std::size_t max_onvert_prob_index = max_onvert_prob_p - onvert_prob.begin();

    auto max_onvert = on_vertice[max_onvert_prob_index];
    auto max_offvert = off_vertice[max_offvert_prob_index];

    auto it =
        std::find(current_solution.begin(), current_solution.end(), max_onvert);
    std::size_t convertindex = std::distance(current_solution.begin(), it);

    current_solution.at(convertindex) = max_offvert;

    auto key = [](const Point &p) { return std::pair{p.x, p.y}; };

    // 换出
    iter_dic[key(max_onvert)] = VertexInfo(iter_dic[key(max_onvert)].index,
                                           "不在路径中", Point{0, 0}, 0.0);

    // 换进（如果不存在就自动插入）
    iter_dic[key(max_offvert)] = VertexInfo(
        iter_dic.count(key(max_offvert)) ? iter_dic[key(max_offvert)].index : 0,
        "在路径中", Point{0, 0}, 0.0);

    // 先保存要删的点
    // Point max_onvert = on_vertice[max_onvert_prob_index];
    // Point max_offvert = off_vertice[max_offvert_prob_index];

    // 删除概率（索引删，顺序无所谓）
    onvert_prob.erase(onvert_prob.begin() + max_onvert_prob_index);
    offvert_prob.erase(offvert_prob.begin() + max_offvert_prob_index);

    // 删除点（用经典 remove-erase 惯用法，彻底安全）
    auto erase_val = [](auto &v, const auto &val) {
      v.erase(std::remove(v.begin(), v.end(), val), v.end());
    };

    erase_val(on_vertice, max_onvert);
    erase_val(off_vertice, max_offvert);
    for (auto &[pt, info] : iter_dic) { // C++17 结构化绑定
      if (info.status != "不在路径中")
        continue;

      double best_cost = std::numeric_limits<double>::max();
      Point best_pred{};

      for (const Point &on : current_solution) {
        size_t i = iter_dic[{on.x, on.y}].index; // 当前路径上点的 index
        size_t j = info.index;                   // 要插入点的 index

        double cost = distance_[i][j];
        if (cost < best_cost) {
          best_cost = cost;
          best_pred = on;
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
  std::vector<std::vector<Point>> soluntion_set{route_};
  // 所有历史最优解（需要存储一个列表 最后在这个列表的基础上进行比较得出）
  std::vector<Point> iter_solution = route_;
  // 当前的迭代解
  std::map<std::pair<int, int>, VertexInfo> iter_dic = vertex_map_;
  // 当前状态的字典 —— 当前状态下的位置字典
  double best_cost = solution_cost_;
  // 启发搜索过程下全局最优成本

  TabuInfo tabu(TBL);
  // 初始化禁忌搜索对象

  // 循环直至路径重连和多样化达到阈值
  while (true) {
    std::vector<double> itercost;
    // 邻域解成本列表
    std::vector<std::vector<Point>> operated_solution;
    // 邻域解成本列表
    std::vector<std::map<std::pair<int, int>, VertexInfo>> operated_dic;
    // 邻域解字典
    using OpKey =
        std::variant<std::vector<Point>,
                     std::pair<std::vector<Point>, std::vector<Point>>>;
    std::vector<OpKey> iterchange;
    // iterchange; 已尝试操作（visited列表 表示二操作中取其一）

    int times = 0;
    // 已生成的有效邻域数

    // 记录当前找到的最优邻域解的个数
    while (times < 50) {
      auto neighbor_solution = operation_style();
      // 通过进行操作 生成邻域数据
      auto &op = std::get<3>(neighbor_solution);
      bool valid = false;

      // 检查正逆序是否已经尝试
      if (op.size() == 2) {
        std::vector<Point> nerireverse = {op[1], op[0]};
        auto pair1 = std::vector<std::vector<Point>>{{op, nerireverse}};
        auto pair2 = std::vector<std::vector<Point>>{{nerireverse, op}};
        if (std::find(iterchange.begin(), iterchange.end(), pair1) ==
                iterchange.end() &&
            std::find(iterchange.begin(), iterchange.end(), pair2) ==
                iterchange.end()) {
          valid = true;
          // op 非{int ,int}类型 需修改iterchange 进行后续邻域解处理
          // Point p = {op,rev};
          itercost.push_back(std::get<2>(neighbor_solution));
          operated_solution.push_back(std::get<0>(neighbor_solution));
          operated_dic.push_back(std::get<1>(neighbor_solution));
          iterchange.emplace_back(std::make_pair(op, nerireverse));
          times++;
        }
      } else if (op.size() == 1) {
        if (std::find(iterchange.begin(), iterchange.end(), op) ==
            iterchange.end()) {
          itercost.push_back(std::get<2>(neighbor_solution));
          operated_solution.push_back(std::get<0>(neighbor_solution));
          operated_dic.push_back(std::get<1>(neighbor_solution));
          iterchange.emplace_back(op);
          times++;
        }
      }
    }

    auto min_iter_cost_p = std::min_element(itercost.begin(), itercost.end());
    double min_iter_cost = *min_iter_cost_p;
    // 获得索引
    std::size_t min_iter_cost_index = min_iter_cost_p - itercost.begin();

    auto min_iter_solution = operated_solution[min_iter_cost_index];
    auto min_iter_dic = operated_dic[min_iter_cost_index];
    auto min_iter_operated = iterchange[min_iter_cost_index];

    if (min_iter_cost <= best_cost && tabu.is_tabu_iter(min_iter_operated)) {
      t++;
      if (t == T) {
        auto diversify =
            diversication(soluntion_set, iter_dic, DIVERSICATION_TIMES);
        q += 1;

        if (q < Q) {
          iter_solution = std::get<0>(diversify);
          iter_dic = std::get<1>(diversify);
          soluntion_set.push_back(iter_solution);

          GreedyLocalSearch calculater(iter_solution, iter_dic);
          best_cost = calculater.tabu_cacl_cost();
          cost_trend_.push_back(best_cost);
          t = 0;
          tabu.reset_list();
          continue;

        } else if (q == Q) {
          GreedyLocalSearch calculater(std::get<0>(diversify),
                                       std::get<1>(diversify));
          double last_cost = calculater.tabu_cacl_cost();
          if (last_cost <= best_cost) {
            iter_solution = std::get<0>(diversify);
            iter_dic = std::get<1>(diversify);
            best_cost = last_cost;
            cost_trend_.push_back(best_cost);
            break;
          } else if (last_cost > best_cost) {
            break;
          }
        }
      } else if (t < T) {
        tabu.update_tabu();
        continue;
      }
    } else if (min_iter_cost <= best_cost &&
               tabu.is_tabu_iter(min_iter_operated)) {
      soluntion_set.push_back(min_iter_solution);
      // 路径重连部分
      auto path_relink = path_relinking(soluntion_set, min_iter_dic);
      auto pr0 = std::get<0>(path_relink);
      auto pr1 = std::get<1>(path_relink);
      if (pr1 <= min_iter_cost) {
        soluntion_set.back() = pr0;
        iter_solution = pr0;
        iter_dic = min_iter_dic;
        best_cost = pr1;
        cost_trend_.push_back(best_cost);
      } else if (pr1 > min_iter_cost) {
        iter_solution = min_iter_solution;
        iter_dic = min_iter_dic;
        best_cost = min_iter_cost;
        cost_trend_.push_back(best_cost);
      }
      t = 0;
      // TODO 禁忌表长度修改为定义
      tabu.add_tabu_iter(min_iter_operated, TBL);
      tabu.update_tabu();
      continue;
    } else if (min_iter_cost > best_cost) {
      t++;
      if (t == T) {
        auto diversify =
            diversication(soluntion_set, iter_dic, DIVERSICATION_TIMES);
        auto dy0 = std::get<0>(diversify);
        auto dy1 = std::get<1>(diversify);
        q++;
        if (q < Q) {
          soluntion_set.push_back(dy0);
          iter_solution = dy0;
          iter_dic = dy1;
          GreedyLocalSearch calculater(iter_solution, iter_dic);
          best_cost = calculater.tabu_cacl_cost();
          cost_trend_.push_back(best_cost);
          t = 0;
          tabu.reset_list();
          continue;
        } else if (q == Q) {
          GreedyLocalSearch calculater(dy0, dy1);
          auto last_cost = calculater.tabu_cacl_cost();
          if (last_cost <= best_cost) {
            iter_solution = dy0;
            iter_dic = dy1;
            best_cost = last_cost;
            cost_trend_.push_back(best_cost);
            break;
          } else if (last_cost > best_cost) {
            break;
          }
        }
      } else if (t < T) {
        tabu.update_tabu();
        continue;
      }
    }
  }
  iter_solution_ = iter_solution;
  best_cost_ = best_cost;
};