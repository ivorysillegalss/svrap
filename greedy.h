#include "input.h"
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

  void search();
  const std::vector<Point> &get_route() const { return route_; }
  const std::vector<Point> &get_ontour() const { return ontour_; }
  const std::vector<Point> &get_offtour() const { return offtour_; }

private:
  std::vector<Point> locations_;
  std::vector<std::vector<double>> distance_;
  std::vector<Point> ontour_;
  std::vector<Point> offtour_;
  std::map<std::pair<int, int>, VertexInfo> vertex_map_;
  std::vector<Point> route_;

  void add(const Point &vertex, size_t index);
  void drop(const Point &vertex);
  void twoopt(const Point &vertex1, const Point &vertex2);
  double calculate_route_cost(std::vector<Point> &route) const;
  void update_vertex_map();
};