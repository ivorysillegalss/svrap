#include "greedy.h"
#include "input.h"
#include <complex.h>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

void calculate_nearest_cost(const std::vector<Point> &ontour,
               const std::vector<Point> &offtour,
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

std::vector<Point> nearest_neighbour(
    const std::vector<Point>& on_vertices,
    const std::vector<std::vector<double>>& distance,
    const std::map<std::pair<int,int>,VertexInfo>& vertex_map
){
    // 数据合法性校验
    if(on_vertices.empty()) throw std::invalid_argument("Data at on_vertices null")

    // 复制未遍历的点
    std::vector<Point> unvisited = on_vertices;
    std::vector<Point> route;

    // 随机选择初始点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0,unvisited.size() - 1);
    size_t start_index = dist(gen);

    // 找到第一个点开始
    Point first_vertex = unvisited[start_index];
    // 记录路线
    route.push_back(first_vertex);
    // 删除指定索引的元素
    unvisited.erase(unvisited.begin() + start_index);

    while(!unvisited.empty()){
        // 返回最后一个元素的引用
        Point select_vertex = route.back();
        std::pair<int, int> select_key = {select_vertex.x,select_vertex.y};
        size_t select_index;
        try{
            select_index = vertex_map.at(select_key).index;
        }catch(const std::out_of_range& e){
            throw std::runtime_error("无法找到点 (" + std::to_string(select_key.first) + ", " +
                                    std::to_string(select_key.second) + ") 的索引");
        }
        
        // 枚举 but贪心寻找最小花费
        double min_cost = std::numeric_limits<double>::max();
        size_t best_index = 0;
        Point best_vertex;

        // 遍历每个点的花费
        for(size_t i = 0;i < unvisited.size(); ++i){
            std::pair<int, int> other_key = {unvisited[i].x,unvisited[i].y};
            size_t other_index;
            try{
                other_index = vertex_map.at(other_key).index;
            }catch(const std::out_of_range& e){
                throw std::runtime_error("无法找到点 (" + std::to_string(other_key.first) + ", " +
                                        std::to_string(other_key.second) + ") 的索引");
            }

            // 找到更优消费 更新最小值
            double cost = distance[select_index][other_index];
            if(cost < min_cost){
                min_cost = cost;
                best_vertex = unvisited[i];
                best_index = i;
            }
        }

        // 将最优点加入访问队列 完成一轮的遍历
        route.push_back(best_vertex);
        // 删除待遍历队列中的该最优点 防止重复计算
        unvisited.erase(unvisited.begin() + best_index);

        return route;
    }
}