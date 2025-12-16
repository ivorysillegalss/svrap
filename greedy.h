#include "input.h"
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

// 计算每个路径外的点 到路径内的店中的花费（cost）最小值
void calculate_nearest_cost(
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::vector<std::vector<double>> &distance,
    std::map<std::pair<int, int>, VertexInfo> &vertex_map);

// 寻找贪婪局部搜索的初始解
std::vector<Point>
nearest_neighbour(const std::vector<Point> &on_vertices,
                  const std::vector<std::vector<double>> &distance,
                  const std::map<std::pair<int, int>, VertexInfo> &vertex_map);

class GreedyLocalSearch {
public:
  // 构造函数赋变量值
  explicit GreedyLocalSearch(
      const std::vector<Point> &locations,
      const std::vector<std::vector<double>> &distance,
      const std::vector<Point> &ontour, const std::vector<Point> &offtour,
      const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
      const std::vector<Point> &route);

  explicit GreedyLocalSearch(
      const std::vector<Point> &route,
      const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
      const std::vector<std::vector<double>> &distance);

  static double compute_cost(
      const std::vector<Point> &route,
      const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
      const std::vector<std::vector<double>> &distance);

  double search();
  double tabu_cacl_cost();
  const std::vector<Point> &get_route() const { return route_; }
  const std::vector<std::vector<double>> &get_distance() const {
    return distance_;
  }
  const std::vector<Point> &get_ontour() const { return ontour_; }
  const std::vector<Point> &get_offtour() const { return offtour_; }
  const double_t &get_cost() const { return solution_cost_; }
  const std::map<std::pair<int, int>, VertexInfo> &get_vertex_map() const {
    return vertex_map_;
  }

private:
  std::vector<Point> locations_;
  std::vector<std::vector<double>> distance_;
  std::vector<Point> ontour_;
  std::vector<Point> offtour_;
  std::map<std::pair<int, int>, VertexInfo> vertex_map_;
  //   记录每个点是否在路径
  //  保存每个 非路径点 到 最近路径点 的距离（best_cost *0.5 计入总成本）
  std::vector<Point> route_;
  std::double_t solution_cost_;

  void add(const Point &vertex, size_t index);
  void drop(const Point &vertex);
  void twoopt(const Point &vertex1, const Point &vertex2);
  double calculate_route_cost(std::vector<Point> &route) const;
  void update_vertex_map();
};