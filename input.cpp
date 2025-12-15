#include "input.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 默认 alpha 值，可以在 main 中按需修改（3,5,7,9 等）。
double ALPHA = 7.0;

// 全局隔离成本表，由 read_isolation_costs 进行填充。
std::map<std::pair<int, int>, double> ISOLATION_COSTS;

// 读取数据
void read_coordinates(const std::string &filename,
                      std::vector<Point> &locations) {
  // std::vector<Point> read_coordinates(const std::string& filename,
  // std::vector<Point>& locations) {
  //   std::vector<Point> locations;
  std::ifstream file(filename);

  // fd是否打开
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // 依次读取文件每行内容
  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    std::stringstream ss(line);
    std::string x_str, y_str;

    // spilt & 格式化读取内容
    if (!std::getline(ss, x_str, ',') || !std::getline(ss, y_str)) {
      std::cerr << "Warning : Invalid format at line" << line_number << ": "
                << line << std::endl;
      continue;
    }

    // 转换数据格式
    try {
      Point p;
      p.x = std::stoi(x_str);
      p.y = std::stoi(y_str);
      locations.push_back(p);
    } catch (const std::exception &e) {
      std::cerr << "Warning: Invalid number format at line" << line_number
                << ": " << line << std::endl;
      continue;
    }
  }
  if (file.bad()) {
    throw std::runtime_error("Error reading file: " + filename);
  }

  return;
}

void read_isolation_costs(const std::string &filename) {
  ISOLATION_COSTS.clear();

  std::ifstream file(filename);
  if (!file.is_open()) {
    // 如果文件不存在，不视为致命错误，允许默认 isolation_cost=0
    std::cerr << "Warning: failed to open isolation cost file: " << filename
              << " (all isolation_cost will default to 0)" << std::endl;
    return;
  }

  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    if (line.empty())
      continue;

    std::stringstream ss(line);
    std::string x_str, y_str, iso_str;

    if (!std::getline(ss, x_str, ',') || !std::getline(ss, y_str, ',') ||
        !std::getline(ss, iso_str)) {
      std::cerr << "Warning: invalid isolation line at " << line_number
                << ": " << line << std::endl;
      continue;
    }

    try {
      int x = std::stoi(x_str);
      int y = std::stoi(y_str);
      double iso = std::stod(iso_str);
      ISOLATION_COSTS[{x, y}] = iso;
    } catch (const std::exception &e) {
      std::cerr << "Warning: invalid isolation number format at line "
                << line_number << ": " << line << std::endl;
      continue;
    }
  }
}

// 将点放置到距离矩阵当中
void compute_distances(const std::vector<Point> &locations,
                       std::vector<std::vector<double>> &distance) {
  size_t n = locations.size();
  //   std::vector<std::vector<double>> distance(n, std::vector<double>(n,
  //   0.0));
  distance.assign(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j) {
        // 对角线距离为 0
        distance[i][j] = 0.0;
      } else {
        // 与 TSPLIB EUC_2D 一致的整数欧氏距离：
        // d_ij = round( sqrt( (x_i-x_j)^2 + (y_i-y_j)^2 ) )
        double dx = static_cast<double>(locations[i].x - locations[j].x);
        double dy = static_cast<double>(locations[i].y - locations[j].y);
        double dij = std::sqrt(dx * dx + dy * dy);
        distance[i][j] = std::floor(dij + 0.5); // 四舍五入为整数
      }
    }
  }
  return;
}

// 定义p 并且以p为标识 分割为区域内外的点
// ontour 区域内的点 offtour 特定区域外的点
void classify_points(const std::vector<Point> &locations,
                     std::vector<Point> &ontour, std::vector<Point> &offtour,
                     double A, double B, double boundary) {
  ontour.clear();
  offtour.clear();
  for (const auto &point : locations) {
    if (point.x < boundary * A || point.x > boundary * (1.0 - A)) {
      offtour.push_back(point);
    } else if (point.y < boundary * A || point.y > boundary * (1.0 - A)) {
      offtour.push_back(point);
    } else if (point.x > boundary * B && point.x < boundary * (1.0 - B) &&
               point.y > boundary * B && point.y < boundary * (1.0 - B)) {
      offtour.push_back(point);
    } else {
      ontour.push_back(point);
    }
  }
}

// 构建点信息集 （包含当前点访问信息、访问与否状态）
// 注意：在与 TS-SVRAP 对齐的实现中，status=="Y" 仅表示
// "当前在路径上"，status=="N" 表示 "不在路径上"。
// 是否被分配 (allocated) 还是被隔离 (isolated) 不再通过
// status 明确建模，而是在成本函数中根据
//   alpha * min( allocation_cost, isolation_cost )
// 的公式按需计算。这里我们仅初始化 isolation_cost，后续
// 可以根据具体实例对其进行调整。
void build_vertex_map(const std::vector<Point> &locations,
                      const std::vector<Point> &on_vertices,
                      const std::vector<Point> &off_vertices,
                      const std::vector<std::vector<double>> &distance,
                      std::map<std::pair<int, int>, VertexInfo> &vertex_map) {
  vertex_map.clear();
  std::map<std::pair<int, int>, size_t> point_to_index;

  // 赋映射标识序号
  for (size_t i = 0; i < locations.size(); ++i) {
    point_to_index[{locations[i].x, locations[i].y}] = i;
  }

  //   遍历所有位置点 并且初根据是否在路径中始化他们的状态
  for (const auto &point : locations) {
    std::pair<int, int> key = {point.x, point.y};

    // TODO 修改为使用std::unordered_set对点存储 O(n) -> O(1)
    // 这里的逻辑是对每个点 find查找在路径点上的集合 判断这个点是否在集合上
    // find_if的逻辑是 查询的起始终止 查询的条件
    // 如果查到了就返回对应的索引 没查到则为列表末尾 ——
    // 通过判断是否末尾判断当前点是否在路径当中
    bool on_route =
        std::find_if(on_vertices.begin(), on_vertices.end(),
                     [&point](const Point &p) {
                       return p.x == point.x && point.y == p.y;
                     }) != on_vertices.end();

    // 缺省情况下 isolation_cost = 0，如果在 ISOLATION_COSTS
    // 中有对应记录，则使用该值。
    double iso_cost = 0.0;
    auto it_iso = ISOLATION_COSTS.find(key);
    if (it_iso != ISOLATION_COSTS.end()) {
      iso_cost = it_iso->second;
    }

    if (on_route) {
      vertex_map[key] =
          VertexInfo(point_to_index[key], "Y", {0, 0}, 0.0, iso_cost);
    } else {
      vertex_map[key] =
          VertexInfo(point_to_index[key], "N", {0, 0}, 0.0, iso_cost);
    }
  }
}