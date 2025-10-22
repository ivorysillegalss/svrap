#include "greedy.h"
#include "input.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

void greedy_local_search(std::vector<Point> ontour, std::vector<Point> offtour,
                         std::vector<std::vector<double>> distance,
                         std::map<std::pair<int, int>, VertexInfo> vertex_map) {
  // 对于每个 off_vertice 中的点，计算它到所有
  // on_vertice中点的距离（使用之前的 distance 矩阵）。
  // 找到距离最小的点（best_vertice）和对应的最小距离（best_cost）。
  calculate_nearest_cost(ontour, offtour, distance, vertex_map);

  // 进行第一次运算 得到进行后续 贪婪搜索的初始解
  // 后续使用贪婪局部搜索进行优化
  std::vector<Point> initial_route =
      nearest_neighbour(ontour, distance, vertex_map);
  std::cout << "初始路径：" << std::endl;
  for (const auto &p : initial_route) {
    std::cout << "(" << p.x << ", " << p.y << ")" << std::endl;
  }
  std::cout << "初始路径长度：" << initial_route.size() << std::endl;

  // TODO 贪婪局部搜索逻辑
}

void input_point(std::string filename) {}

int main() {
  try {
    // 加载读取坐标集
    std::string filename = "tsp数据坐标集.txt";
    std::vector<Point> locations;
    read_coordinates(filename, locations);
    std::cout << "Read" << locations.size() << " points." << std::endl;

    // 计算距离
    std::vector<std::vector<double>> distance;
    compute_distances(locations, distance);
    std::cout << "Compute distance matrix." << std::endl;

    // 根据当前路径上与否区分点类
    std::vector<Point> ontour, offtour;
    classify_points(locations, ontour, offtour, A_VALUE, B_VALUE,
                    BOUNDARY_VALUE);

    std::cout << "Points in ontour size: " << ontour.size() << std::endl;
    std::cout << "Points in offtour size: " << offtour.size() << std::endl;

    // std::cout << "Ontour points:\n";
    // for (const auto &p : ontour) {
    //   std::cout << "(" << p.x << ", " << p.y << ")\n";
    // }
    // std::cout << "Offtour points:\n";
    // for (const auto &p : offtour) {
    //   std::cout << "(" << p.x << ", " << p.y << ")\n";
    // }

    // 构建点信息集
    std::map<std::pair<int, int>, VertexInfo> vertex_map;
    build_vertex_map(locations, ontour, offtour, distance, vertex_map);
    std::cout << "Build points info" << std::endl;

    greedy_local_search(ontour, offtour, distance, vertex_map);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}