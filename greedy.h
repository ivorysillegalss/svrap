#include "input.h"
#include <vector>

// 计算每个路径外的点 到路径内的店中的花费（cost）最小值
void calculate_nearest_cost(const std::vector<Point> &ontour,
               const std::vector<Point> &offtour,
               const std::vector<std::vector<double>> &distance,
            std::map<std::pair<int, int>, VertexInfo> &vertex_map);

// 寻找贪婪局部搜索的初始解
std::vector<Point> nearest_neighbour(
    const std::vector<Point>& on_vertices,
    const std::vector<std::vector<double>>& distance,
    const std::map<std::pair<int,int>,VertexInfo>& vertex_map
);
