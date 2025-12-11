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

  // 随机选择初始点
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, unvisited.size() - 1);
  size_t start_index = dist(gen);

  // 找到第一个点开始
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

// 对应py中的cacl_cost
double
GreedyLocalSearch::calculate_route_cost(std::vector<Point> &route) const {
  if (route.empty()) {
    // 贪婪搜索有几率报错
    throw std::invalid_argument("nil route, can't calculate");
  }
  double total_cost = 0.0;
  for (size_t i = 0; i < route.size() - 1; ++i) {
    std::pair<int, int> key1 = {route[i].x, route[i].y};
    std::pair<int, int> key2 = {route[i + 1].x, route[i + 1].y};
    size_t index1, index2;
    try {
      index1 = vertex_map_.at(key1).index;
      index2 = vertex_map_.at(key2).index;
    } catch (const std::out_of_range &e) {
      throw std::runtime_error(
          "can't find point (" + std::to_string(key1.first) + ", " +
          std::to_string(key1.second) + ") or (" + std::to_string(key2.first) +
          ", " + std::to_string(key2.second) + ") index");
    }
    total_cost += distance_[index1][index2];
  }
  for (const auto &entry : vertex_map_) {
    if (entry.second.status == "N") {
      total_cost += 0.5 * entry.second.best_cost;
    }
  }
  return total_cost;
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
    if (std::find(ontour_.begin(), ontour_.end(), point) != ontour_.end()) {
      vertex_map_[key] = VertexInfo(point_to_index[key], "Y");
    } else {
      vertex_map_[key] = VertexInfo(point_to_index[key], "N");
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
      // 替换操作
      // auto vl_index_it = std::find(route_.begin(), route_.end(), vl);
      //   定位 此处计算与开头的距离 —— 实际索引
      // size_t vl_index = std::distance(route_.begin(), vl_index_it);
      // for (size_t i = 0; i < route_.size(); ++i) {
      //   // 如果当前要交换到的点是它本身 就不用交换
      //   if (i == vl_index) {
      //     cost_list.push_back(std::numeric_limits<double>::max());
      //     routes.push_back(route_);
      //     dicts.push_back(vertex_map_);
      //     // 对于当前其他的点都进行交换工作 交换之后计算对应的成本 是否增or减
      //   } else {
      //     try {
      //       std::vector<Point> new_route = route_;
      //       std::swap(new_route[vl_index], new_route[i]);
      //       double cost = calculate_route_cost(new_route);
      //       cost_list.push_back(cost);
      //       routes.push_back(new_route);
      //       // 交换操作不需要修改对应的点集
      //       dicts.push_back(vertex_map_);
      //     } catch (const std::exception &e) {
      //       std::cerr << "twoopt operation unknown error: " << e.what()
      //                 << std::endl;
      //       // 恢复现场
      //       cost_list.push_back(std::numeric_limits<double>::max());
      //       routes.push_back(route_);
      //       dicts.push_back(vertex_map_);
      //     }
      //   }
      // }

      // 替换操作 (Swap / Two-Opt)
      // 1. 预先计算当前路外点的总惩罚 (Swap 操作不会改变路外点状态，这是个定值)
      double current_off_penalty = 0.0;
      for (const auto &entry : vertex_map_) {
        if (entry.second.status == "N") {
          current_off_penalty += 0.5 * entry.second.best_cost;
        }
      }

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
            // 构造新路径
            std::vector<Point> new_route = route_;
            std::swap(new_route[vl_index], new_route[i]);

            // 2. 仅计算路径内的几何距离 (Geometry Distance)
            double route_geo_cost = 0.0;
            // 遍历新路径计算距离
            bool point_not_found = false;
            for (size_t k = 0; k < new_route.size() - 1; ++k) {
              std::pair<int, int> p1_key = {new_route[k].x, new_route[k].y};
              std::pair<int, int> p2_key = {new_route[k + 1].x,
                                            new_route[k + 1].y};

              // 获取 distance 矩阵需要的下标索引
              if (vertex_map_.find(p1_key) == vertex_map_.end() ||
                  vertex_map_.find(p2_key) == vertex_map_.end()) {
                point_not_found = true;
                break;
              }
              size_t idx1 = vertex_map_.at(p1_key).index;
              size_t idx2 = vertex_map_.at(p2_key).index;

              route_geo_cost += distance_[idx1][idx2];
            }
            
            if (point_not_found) {
              throw std::runtime_error("Point not found in vertex_map during twoopt");
            }

            // 3. 总花费 = 几何距离 + 路外点惩罚
            double total_cost = route_geo_cost + current_off_penalty;

            cost_list.push_back(total_cost);
            routes.push_back(new_route);

            // Swap 操作不改变点的 "Y/N" 状态，直接压入当前的 map 即可
            // (避免了耗时的 map 拷贝)
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