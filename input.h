#ifndef TSP_COMMON_H
#define TSP_COMMON_H

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

// ---------------- SVRAP global parameters ----------------
// ALPHA (a): parameter controlling the split between routing and
// allocation costs used in the test problems from the SVRAP
// paper. For a in {3,5,7,9} and TSPLIB distances l_ij, the
// paper defines:
//   c_ij = a * l_ij          (routing arc costs)
//   d_ij = (10 - a) * l_ij   (allocation costs)
// and sets lambda_tour = lambda_alloc = 1 in the objective.
//
// We expose ALPHA as a runtime-configurable global equal to a.
extern double ALPHA;

// The following P/A/B/boundary values were used in the earlier
// geometry-based region classification. They are no longer part
// of the core SVRAP definition, but we keep them available in
// case they are still useful for constructing initial solutions.
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
    return !(*this == other);
  }
};

// 定义 VertexInfo 结构体（如果需要 vertex_map）
struct VertexInfo {
  size_t index;       // 唯一标识顶点 节点映射中进行快速查找
  std::string status; // 顶点在算法中的状态
  Point best_vertex;  // 记录当前最优路径的上一个顶点
  double best_cost;   // 记录从起点到当前顶点的最小代价（贪心）

  // Isolation cost for this vertex when it is not served by the
  // vehicle route (ie, when it is "isolated" rather than
  // allocated to a route vertex). This is multiplied by ALPHA in
  // the TS-SVRAP objective. By default we set it to 0 and expect
  // the caller to initialise it appropriately if isolation is
  // allowed in the instance.
  double isolation_cost;

  // VertexInfo(size_t idx, const std::string &st)
  //     : index(idx), status(st), best_vertex({0, 0}), best_cost(0.0) {}
  VertexInfo(size_t idx = 0, const std::string &st = "", Point bv = {0, 0},
             double bc = 0.0, double iso = 0.0)
      : index(idx), status(st), best_vertex(bv), best_cost(bc),
        isolation_cost(iso) {}
};

// Optional per-vertex isolation costs (eg. f_i in the paper).
// If provided (through read_isolation_costs), they are used to
// initialise VertexInfo::isolation_cost for each point.
extern std::map<std::pair<int, int>, double> ISOLATION_COSTS;

// 函数声明
void read_coordinates(const std::string &filename,
                      std::vector<Point> &locations);
// 读取隔离成本表：每行形如 "x,y,iso_cost"。如果该文件
// 不存在或某个点没有记录，则该点的 isolation_cost 默认
// 为 0，由 allocation_cost 主导。
void read_isolation_costs(const std::string &filename);
void compute_distances(const std::vector<Point> &locations,
                       std::vector<std::vector<double>> &distance);
void classify_points(const std::vector<Point> &locations,
                     std::vector<Point> &ontour, std::vector<Point> &offtour,
                     double A, double B, double boundary);
void build_vertex_map(const std::vector<Point> &locations,
                      const std::vector<Point> &on_vertices,
                      const std::vector<Point> &off_vertices,
                      const std::vector<std::vector<double>> &distance,
                      std::map<std::pair<int, int>, VertexInfo> &vertex_map);

#endif // TSP_COMMON_H