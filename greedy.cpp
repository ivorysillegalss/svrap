#include "greedy.h"
#include "input.h"
#include <algorithm>
#include <cmath>
#include <complex.h>
#include <cstddef>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void calculate_nearest_cost(
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::vector<std::vector<double>> &distance,
    std::map<std::pair<int, int>, VertexInfo> &vertex_map) {
  for (const auto &off_point : offtour) {
    std::pair<int, int> off_key = {off_point.x, off_point.y};
    // TODO 使用find方法改进
    size_t off_index = vertex_map[off_key].index;

    double min_cost = std::numeric_limits<double>::max();
    Point best_vertex = {0, 0};

    for (const auto &on_point : ontour) {
      std::pair<int, int> on_key = {on_point.x, on_point.y};
      size_t on_index = vertex_map[on_key].index;
      double cost = distance[off_index][on_index];
      if (cost < min_cost) {
        min_cost = cost;
        best_vertex = on_point;
      }
    }
    vertex_map[off_key].best_vertex = best_vertex;
    vertex_map[off_key].best_cost = min_cost;
  }
}

std::vector<Point>
nearest_neighbour(const std::vector<Point> &on_vertices,
                  const std::vector<std::vector<double>> &distance,
                  const std::map<std::pair<int, int>, VertexInfo> &vertex_map) {
  // 数据合法性校验
  if (on_vertices.empty())
    throw std::invalid_argument("Data at on_vertices null");

  // 复制未遍历的点
  std::vector<Point> unvisited = on_vertices;
  std::vector<Point> route;

  // 为了与 TSPLIB / SVRAP 设定更一致，这里固定使用
  // on_vertices 中的第一个点作为“depot”起点，而不是
  // 随机选择起点。
  size_t start_index = 0;
  Point first_vertex = unvisited[start_index];
  // 记录路线
  route.push_back(first_vertex);
  // 删除指定索引的元素
  unvisited.erase(unvisited.begin() + start_index);

  while (!unvisited.empty()) {
    // 返回最后一个元素的引用
    Point select_vertex = route.back();
    std::pair<int, int> select_key = {select_vertex.x, select_vertex.y};
    size_t select_index;
    try {
      select_index = vertex_map.at(select_key).index;
    } catch (const std::out_of_range &e) {
      throw std::runtime_error("can't find point (" +
                               std::to_string(select_key.first) + ", " +
                               std::to_string(select_key.second) + ") index");
    }

    // 枚举 but贪心寻找最小花费
    double min_cost = std::numeric_limits<double>::max();
    size_t best_index = 0;
    Point best_vertex;

    // 遍历每个点的花费
    for (size_t i = 0; i < unvisited.size(); ++i) {
      std::pair<int, int> other_key = {unvisited[i].x, unvisited[i].y};
      size_t other_index;
      try {
        other_index = vertex_map.at(other_key).index;
      } catch (const std::out_of_range &e) {
        throw std::runtime_error("can't find point (" +
                                 std::to_string(other_key.first) + ", " +
                                 std::to_string(other_key.second) + ") index");
      }

      // 找到更优消费 更新最小值
      double cost = distance[select_index][other_index];
      if (cost < min_cost) {
        min_cost = cost;
        best_vertex = unvisited[i];
        best_index = i;
      }
    }

    // 将最优点加入访问队列 完成一轮的遍历
    route.push_back(best_vertex);
    // 删除待遍历队列中的该最优点 防止重复计算
    unvisited.erase(unvisited.begin() + best_index);
  }
  return route;
}

// 构造函数 赋值
GreedyLocalSearch::GreedyLocalSearch(
    const std::vector<Point> &locations,
    const std::vector<std::vector<double>> &distance,
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<Point> &route)
    : locations_(locations), distance_(distance), ontour_(ontour),
      offtour_(offtour), vertex_map_(vertex_map), route_(route) {}

GreedyLocalSearch::GreedyLocalSearch(
    const std::vector<Point> &route,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<std::vector<double>> &distance)
    : distance_(distance), vertex_map_(vertex_map), route_(route) {}

void GreedyLocalSearch::add(const Point &vertex, size_t index) {
  if (index > route_.size()) {
    throw std::out_of_range(
        "insert index" + std::to_string(index) +
        " exceed route length :" + std::to_string(route_.size()));
  }
  route_.insert(route_.begin() + index, vertex);

  auto it = std::find(offtour_.begin(), offtour_.end(), vertex);
  if (it != offtour_.end()) {
    offtour_.erase(it);
    ontour_.push_back(vertex);
  }
  update_vertex_map();
}

void GreedyLocalSearch::drop(const Point &vertex) {
  auto it = std::find(route_.begin(), route_.end(), vertex);
  if (it == route_.end()) {
    throw std::invalid_argument("Point(" + std::to_string(vertex.x) + ", " +
                                std::to_string(vertex.y) +
                                ") Not at the route!");
  }
  route_.erase(it);
  auto ontour_it = std::find(ontour_.begin(), ontour_.end(), vertex);
  if (ontour_it != ontour_.end()) {
    ontour_.erase(ontour_it);
    offtour_.push_back(vertex);
  }
  update_vertex_map();
}

void GreedyLocalSearch::twoopt(const Point &vertex1, const Point &vertex2) {
  auto it1 = std::find(route_.begin(), route_.end(), vertex1);
  auto it2 = std::find(route_.begin(), route_.end(), vertex2);
  if (it1 == route_.end() || it2 == route_.end()) {
    throw std::invalid_argument(
        "Point(" + std::to_string(vertex1.x) + ", " +
        std::to_string(vertex1.y) + ") or (" + std::to_string(vertex2.x) +
        ", " + std::to_string(vertex2.y) + ") not at the route!");
  }
  std::swap(*it1, *it2);
}

// SVRAP 目标函数近似实现：
//   Z = λ_tour * Σ c_ij x_ij
//     + λ_alloc * Σ d_ij y_ij
//     + λ_isol * Σ D_i v_i
// 其中对于测试问题：
//   λ_tour = λ_alloc = 1
//   c_ij = a * l_ij
//   d_ij = (10 - a) * l_ij
//   λ_isol = 0.5 + 0.0004 * a^2 * n
// 这里 a = ALPHA, l_ij 为 TSPLIB 距离。对于给定的 on-tour
// 顶点集合，最优的分配/隔离决策可以对每个 off-tour 顶点
// 独立做贪婪选择：
//   分配成本:   λ_alloc * min_i d_ij
//   隔离成本:   λ_isol  * D_j
//   取两者较小者计入目标。
// 我们用 VertexInfo::isolation_cost 作为 D_j，并通过
// min( (10-a)*min_i l_ij, λ_isol * D_j ) 实现这一部分。
double GreedyLocalSearch::compute_cost(
    const std::vector<Point> &route,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<std::vector<double>> &distance) {
  if (route.empty()) {
    throw std::invalid_argument("nil route, can't calculate");
  }

  const double a = ALPHA; // 论文中的参数 a
  const double lambda_tour = 1.0;
  const double lambda_alloc = 1.0;
  const std::size_t n_vertices = vertex_map.size();
  const double lambda_isol = 0.5 + 0.0004 * a * a * static_cast<double>(n_vertices);

  // 1. 路径成本（routing cost）: Σ c_ij x_ij, c_ij = a * l_ij
  double routing_cost = 0.0;
  for (size_t i = 0; i + 1 < route.size(); ++i) {
    std::pair<int, int> key1 = {route[i].x, route[i].y};
    std::pair<int, int> key2 = {route[i + 1].x, route[i + 1].y};
    size_t index1, index2;
    try {
      index1 = vertex_map.at(key1).index;
      index2 = vertex_map.at(key2).index;
    } catch (const std::out_of_range &e) {
      throw std::runtime_error(
          "can't find point (" + std::to_string(key1.first) + ", " +
          std::to_string(key1.second) + ") or (" + std::to_string(key2.first) +
          ", " + std::to_string(key2.second) + ") index");
    }
    double l_ij = distance[index1][index2];
    routing_cost += lambda_tour * a * l_ij;
  }

  // 对应闭合巡回：补上最后一个顶点回到第一个顶点的弧
  if (route.size() > 1) {
    std::pair<int, int> key_first = {route.front().x, route.front().y};
    std::pair<int, int> key_last = {route.back().x, route.back().y};
    size_t idx_first = vertex_map.at(key_first).index;
    size_t idx_last = vertex_map.at(key_last).index;
    double l_last_first = distance[idx_last][idx_first];
    routing_cost += lambda_tour * a * l_last_first;
  }

  // 2. 分配 / 隔离成本（对所有不在路径上的点）
  // 对于每个 status == "N" 的顶点 j：
  //   allocation_term(j) = λ_alloc * (10-a) * min_i l_ij
  //   isolation_term(j)  = λ_isol  * D_j
  //   贡献 = min(allocation_term(j), isolation_term(j))
  double alloc_iso_cost = 0.0;

  // 预先构造一份当前路径顶点的索引集合，避免在内层循环中频繁查找。
  std::vector<size_t> route_indices;
  route_indices.reserve(route.size());
  for (const auto &p : route) {
    std::pair<int, int> key = {p.x, p.y};
    auto it = vertex_map.find(key);
    if (it == vertex_map.end()) {
      throw std::runtime_error("route point not found in vertex_map");
    }
    route_indices.push_back(it->second.index);
  }

  for (const auto &entry : vertex_map) {
    const auto &info = entry.second;
    if (info.status == "Y") {
      continue; // 在路径上的点只计入 routing_cost
    }

    if (route_indices.empty()) {
      // 理论上不应发生：路径为空的情况已经在开头排除
      continue;
    }

    // 2.1 最近的路径顶点 TSPLIB 距离（l_ij）
    double best_l = std::numeric_limits<double>::max();
    for (size_t idx_on : route_indices) {
      double lij = distance[info.index][idx_on];
      if (lij < best_l) {
        best_l = lij;
      }
    }

    double allocation_term = lambda_alloc * (10.0 - a) * best_l;
    double isolation_term;

    if (info.isolation_cost > 0.0) {
      isolation_term = lambda_isol * info.isolation_cost;
    } else {
      // 如果未设置隔离成本，则等价于“永不选择隔离”
      isolation_term = std::numeric_limits<double>::max();
    }

    alloc_iso_cost += std::min(allocation_term, isolation_term);
  }

  return routing_cost + alloc_iso_cost;
}

double
GreedyLocalSearch::calculate_route_cost(std::vector<Point> &route) const {
  return compute_cost(route, vertex_map_, distance_);
}

double GreedyLocalSearch::tabu_cacl_cost() {
  return calculate_route_cost(route_);
}

void GreedyLocalSearch::update_vertex_map() {
  vertex_map_.clear();
  std::map<std::pair<int, int>, size_t> point_to_index;
  for (size_t i = 0; i < locations_.size(); ++i) {
    point_to_index[{locations_[i].x, locations_[i].y}] = i;
  }

  for (const auto &point : locations_) {
    std::pair<int, int> key = {point.x, point.y};
    bool on_route =
        std::find(ontour_.begin(), ontour_.end(), point) != ontour_.end();

    // 保持隔离成本不变（如果之前已经设置过），否则从
    // 全局 ISOLATION_COSTS 中读取，若仍无则默认 0。
    double iso_cost = 0.0;
    auto old_it = vertex_map_.find(key);
    if (old_it != vertex_map_.end()) {
      iso_cost = old_it->second.isolation_cost;
    } else {
      auto it_iso = ISOLATION_COSTS.find(key);
      if (it_iso != ISOLATION_COSTS.end()) {
        iso_cost = it_iso->second;
      }
    }

    if (on_route) {
      vertex_map_[key] =
          VertexInfo(point_to_index[key], "Y", {0, 0}, 0.0, iso_cost);
    } else {
      vertex_map_[key] =
          VertexInfo(point_to_index[key], "N", {0, 0}, 0.0, iso_cost);
    }
  }

  for (const auto &off_point : offtour_) {
    std::pair<int, int> off_key = {off_point.x, off_point.y};
    size_t off_index = vertex_map_.at(off_key).index;

    double min_cost = std::numeric_limits<double>::max();
    Point best_vertex = {0, 0};

    for (const auto &on_point : ontour_) {
      std::pair<int, int> on_key = {on_point.x, on_point.y};
      size_t on_index = vertex_map_.at(on_key).index;
      double cost = distance_[off_index][on_index];
      if (cost < min_cost) {
        min_cost = cost;
        best_vertex = on_point;
      }
    }

    vertex_map_.at(off_key).best_vertex = best_vertex;
    vertex_map_.at(off_key).best_cost = min_cost;
  }
}

double GreedyLocalSearch::search() {
  double solution_cost = calculate_route_cost(route_);
  for (const auto &vl : locations_) {

    // 执行贪婪局部搜索三个操作 进行迭代优化
    // 判断当前的该点是否在路径当中 如果在路径当中执行替换或者删除
    // 如果不在则只能执行添加 这里使用find来进行迭代太慢了 争取换成set
    if (std::find(ontour_.begin(), ontour_.end(), vl) != ontour_.end()) {
      std::vector<double> cost_list;
      std::vector<std::vector<Point>> routes;
      std::vector<std::map<std::pair<int, int>, VertexInfo>> dicts;

      // 执行删除操作
      std::vector<Point> drop_route = route_;
      std::vector<Point> drop_offtour = offtour_;
      drop_offtour.push_back(vl);
      auto drop_dict = vertex_map_;
      try {
        auto it = std::find(drop_route.begin(), drop_route.end(), vl);
        if (it != drop_route.end()) {
          drop_route.erase(it);
        }
        drop_dict[{vl.x, vl.y}].status = "N";

        // 重新计算对应的cost 决定当前是否替换最优解
        for (auto &entry : drop_dict) {
          if (entry.second.status == "N" && entry.second.best_vertex == vl) {
            size_t off_index = entry.second.index;
            double min_cost = std::numeric_limits<double>::max();
            Point best_vertex = {0, 0};
            for (const auto &on_point : drop_route) {
              std::pair<int, int> on_key = {on_point.x, on_point.y};
              size_t on_index = drop_dict.at(on_key).index;
              double cost = distance_[off_index][on_index];
              if (cost < min_cost) {
                min_cost = cost;
                best_vertex = on_point;
              }
            }
            entry.second.best_vertex = best_vertex;
            entry.second.best_cost = min_cost;
          }
        }

        double drop_cost = calculate_route_cost(drop_route);
        cost_list.push_back(drop_cost);
        routes.push_back(drop_route);
        dicts.push_back(drop_dict);
      } catch (const std::exception &e) {
        std::cerr << "drop operation error: " << e.what() << std::endl;
        cost_list.push_back(std::numeric_limits<double>::max());
        routes.push_back(route_);
        dicts.push_back(vertex_map_);
      }

      // 与源代码的更改：避免全量计算
      // 替换操作 (Swap / Two-Opt)
      // 在 TS-SVRAP 目标下，路外点的分配/隔离惩罚依赖于
      // "在路径上的顶点集合"，而不是路径顺序。因此对于
      // 简单的交换操作，理论上可以预先计算这一部分。但
      // 为了保持与成本函数实现的一致性，这里直接调用
      // calculate_route_cost 进行完整评估，避免重复实现。
      auto vl_index_it = std::find(route_.begin(), route_.end(), vl);
      size_t vl_index = std::distance(route_.begin(), vl_index_it);

      for (size_t i = 0; i < route_.size(); ++i) {
        // 如果当前要交换到的点是它本身 就不用交换
        if (i == vl_index) {
          cost_list.push_back(std::numeric_limits<double>::max());
          routes.push_back(route_);
          dicts.push_back(vertex_map_);
        } else {
          try {
            std::vector<Point> new_route = route_;
            std::swap(new_route[vl_index], new_route[i]);
            double total_cost = calculate_route_cost(new_route);
            cost_list.push_back(total_cost);
            routes.push_back(new_route);
            dicts.push_back(vertex_map_);

          } catch (const std::exception &e) {
            std::cerr << "twoopt operation unknown error: " << e.what()
                      << std::endl;
            // 恢复现场
            cost_list.push_back(std::numeric_limits<double>::max());
            routes.push_back(route_);
            dicts.push_back(vertex_map_);
          }
        }
      }

      //   从维护的成本链表中取出消费的最小值 —— 最优
      auto min_cost_it = std::min_element(cost_list.begin(), cost_list.end());
      double best_cost = *min_cost_it;
      size_t best_index = std::distance(cost_list.begin(), min_cost_it);
      if (best_cost < solution_cost) {
        solution_cost = best_cost;
        route_ = routes[best_index];
        vertex_map_ = dicts[best_index];
        if (best_index == 0) {
          // drop操作 更新当前的对应路径中 路径外表
          // TODO 这里仅对当前的其中一个点做操作？
          auto it = std::find(ontour_.begin(), ontour_.end(), vl);
          if (it != ontour_.end()) {
            ontour_.erase(it);
            offtour_.push_back(vl);
          }
        }
      }

    }
    // 点不在路径上 这时候只有一个操作方式 add
    else if (std::find(offtour_.begin(), offtour_.end(), vl) !=
             offtour_.end()) {
      std::vector<double> cost_list;
      std::vector<std::vector<Point>> routes;
      std::vector<std::map<std::pair<int, int>, VertexInfo>> dicts;

      auto add_dict = vertex_map_;
      add_dict[{vl.x, vl.y}].status = "Y";
      std::vector<Point> temp_ontour = ontour_;
      temp_ontour.push_back(vl);

      for (auto &entry : add_dict) {
        if (entry.second.status == "N") {
          size_t off_index = entry.second.index;
          double min_cost = std::numeric_limits<double>::max();
          Point best_vertex = {0, 0};
          for (const auto &on_point : temp_ontour) {
            std::pair<int, int> on_key = {on_point.x, on_point.y};
            size_t on_index = add_dict.at(on_key).index;
            double cost = distance_[off_index][on_index];
            if (cost < min_cost) {
              min_cost = cost;
              best_vertex = on_point;
            }
          }
          entry.second.best_cost = min_cost;
          entry.second.best_vertex = best_vertex;
        }
      }

      //   把点加回去计算成本cost
      for (size_t i = 0; i <= route_.size(); ++i) {
        try {
          std::vector<Point> new_route = route_;
          new_route.insert(new_route.begin() + i, vl);
          double cost = calculate_route_cost(new_route);
          cost_list.push_back(cost);
          routes.push_back(new_route);
          dicts.push_back(add_dict);
        } catch (const std::exception &e) {
          std::cerr << "add operation error: " << e.what() << std::endl;
          cost_list.push_back(std::numeric_limits<double>::max());
          routes.push_back(route_);
          dicts.push_back(vertex_map_);
        }
      }

      //   同样计算对应最优值
      auto min_cost_it = std::min_element(cost_list.begin(), cost_list.end());
      double best_cost = *min_cost_it;
      size_t best_index = std::distance(cost_list.begin(), min_cost_it);
      if (best_cost < solution_cost) {
        solution_cost = best_cost;
        route_ = routes[best_index];
        vertex_map_ = dicts[best_index];
        auto it = std::find(offtour_.begin(), offtour_.end(), vl);
        if (it != offtour_.end()) {
          offtour_.erase(it);
          ontour_.push_back(vl);
        }
      }
    }
  }
  solution_cost_ = solution_cost;
  return solution_cost;
}