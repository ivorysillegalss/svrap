#ifndef TSP_COMMON_H
#define TSP_COMMON_H

#include <cmath>
#include <map>
#include <string>
#include <vector>

#define P_VALUE 0.3
#define A_VALUE ((2.0 - std::sqrt(1.0 + 3.0 * P_VALUE)) / 3.0)
#define B_VALUE ((1.0 - A_VALUE) / 2.0)
#define BOUNDARY_VALUE 75.0

// 定义 Point 结构体
struct Point {
  int x;
  int y;

  Point() : x(0), y(0) {}

  Point(int x_, int y_) : x(x_), y(y_) {}

  bool operator==(const Point &other) const {
    return x == other.x && y == other.y;
  }

  bool operator!=(const Point &other) const {
    return y != other.y && x != other.x;
  }
};

// 定义 VertexInfo 结构体（如果需要 vertex_map）
struct VertexInfo {
  size_t index;       // 唯一标识顶点 节点映射中进行快速查找
  std::string status; // 顶点在算法中的状态
  Point best_vertex;  // 记录当前最优路径的上一个顶点
  double best_cost;   // 记录从起点到当前顶点的最小代价（贪心）

  // VertexInfo(size_t idx, const std::string &st)
  //     : index(idx), status(st), best_vertex({0, 0}), best_cost(0.0) {}
  VertexInfo(size_t idx = 0, const std::string &st = "", Point bv = {0, 0},
             double bc = 0.0)
      : index(idx), status(st), best_vertex(bv), best_cost(bc) {}
};

// 函数声明
void read_coordinates(const std::string &filename,
                      std::vector<Point> &locations);
void compute_distances(const std::vector<Point> &locations,
                       std::vector<std::vector<double>> &distance);
void classify_points(const std::vector<Point> &locations,
                     std::vector<Point> &ontour, std::vector<Point> &offtour,
                     double_t A, double_t B, double_t boundary);
void build_vertex_map(const std::vector<Point> &locations,
                      const std::vector<Point> &on_vertices,
                      const std::vector<Point> &off_vertices,
                      const std::vector<std::vector<double>> &distance,
                      std::map<std::pair<int, int>, VertexInfo> &vertex_map);

#endif // TSP_COMMON_H