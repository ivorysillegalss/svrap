#include "tabu_search.h"
#include "input.h"
#include <cstddef>
#include <utility>

TabuSearch::TabuSearch(
    const std::vector<Point> &locations,
    const std::vector<std::vector<double>> &distance,
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<Point> &route)
    : locations_(locations), distance_(distance), ontour_(ontour),
      offtour_(offtour), vertex_map_(vertex_map), route_(route) {}

void TabuInfo::update_tabu() {
  Point temp;
  for (int i = 0; i < tabuTime.size(); ++i) {
    tabuTime[i]--;
    if (tabuTime[i] == 0) {
      tabuList[i] = temp;
    }
  }

  // 索引更新移除非0项 压缩迁移后数组
  size_t write_idx = 0;
  for (size_t i = 0; i < tabuList.size(); ++i) {
    if (tabuTime[i] != 0) {
      if (write_idx != i) {
        tabuList[write_idx] = std::move(tabuList[i]);
        tabuTime[write_idx] = tabuTime[i];
      }
      ++write_idx;
    }
  }
  tabuList.resize(write_idx);
  tabuTime.resize(write_idx);
};