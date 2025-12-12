#include "greedy.h"
#include "input.h"
#include "attention_reader.h"
#include "tabu_search.h"
#include "entropy_strategy.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <numeric>
#include <io.h>
#include <fcntl.h>

// 设置 Windows 控制台 UTF-8 编码以正确显示中文
#ifdef _WIN32
  #include <windows.h>
#endif

#define PATH_RELINKING_TIMES 50
#define DIVERSIFICATION 3
#define TABU_LIST_LENGTH 15
    double entropy_effective_threshold = 0.0;
    EntropyStrategyType used_strategy_type = ENTROPY_STRATEGY_DEFAULT;
GreedyLocalSearch
greedy_local_search(std::vector<Point> ontour, std::vector<Point> offtour,
                    std::vector<std::vector<double>> distance,
                    std::map<std::pair<int, int>, VertexInfo> vertex_map,
                    std::vector<Point> locations,
                    const std::vector<Point> &initial_route_override = {}) {
  // 对于每个 off_vertice 中的点，计算它到所有
  // on_vertice中点的距离（使用之前的 distance 矩阵）。
  // 找到距离最小的点（best_vertice）和对应的最小距离（best_cost）。
  calculate_nearest_cost(ontour, offtour, distance, vertex_map);

  // 进行第一次运算 得到进行后续 贪婪搜索的初始解
  // 如果外部传入了 initial_route_override，则使用之（backbone 初始化）
  // 但需要确保它包含所有 ontour 中的点，否则补充完整
  std::vector<Point> initial_route;
  if (!initial_route_override.empty()) {
    // 使用 backbone 作为初始路线的一部分，但要确保包含所有 ontour 点
    initial_route = initial_route_override;
    // 添加不在 backbone 中但在 ontour 中的点
    for (const auto &on_pt : ontour) {
      if (std::find(initial_route.begin(), initial_route.end(), on_pt) == initial_route.end()) {
        initial_route.push_back(on_pt);
      }
    }
  } else {
    // 默认使用最近邻构造初始路径
    initial_route = nearest_neighbour(ontour, distance, vertex_map);
  }

  std::cout << "Initial route:" << std::endl;
  for (const auto &p : initial_route) {
    std::cout << "(" << p.x << ", " << p.y << ")" << std::endl;
  }
  std::cout << "Route size: " << initial_route.size() << std::endl;

  // 带入贪婪局部搜索第一步时的初始解
  GreedyLocalSearch solver(locations, distance, ontour, offtour, vertex_map,
                           initial_route);
  solver.search();
  return solver;
}

TabuSearch
tabu_search(const std::vector<Point> &locations,
            const std::vector<std::vector<double>> &distance,
            const std::vector<Point> &ontour, const std::vector<Point> &offtour,
            const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
            const std::vector<Point> &route, const std::double_t cost) {
  // TODO
  TabuSearch solver(locations, distance, ontour, offtour, vertex_map, route,
                    cost);
  return solver;
}

int main() {
  try {
    // 设置 Windows 控制台 UTF-8 输出
    #ifdef _WIN32
      // 尝试设置控制台代码页，忽略失败
      HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
      if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
          SetConsoleOutputCP(CP_UTF8);
          _setmode(_fileno(stdout), _O_U8TEXT);
        }
      }
    #endif

    // 加载读取坐标集
    std::string filename = "dataset.txt";
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

    std::cout << "On-tour: " << ontour.size() << " | Off-tour: " << offtour.size() << std::endl;

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
    std::cout << "Vertex map built." << std::endl;

    // Copilot: Attempt to read attention probabilities produced by Python solver
    std::vector<NodeProb> node_probs;
    std::vector<Point> backbone_initial_route;
    std::set<std::pair<int, int>> high_entropy_nodes;
    double entropy_effective_threshold = 0.0;
    
    if (read_attention_probs("attention_probs.csv", node_probs)) {
      std::cout << "Loaded attention probabilities for " << node_probs.size() << " nodes." << std::endl;
      
      // Calculate entropy for all nodes
      std::vector<double> entropies;
      calculate_all_entropies(node_probs, entropies);
      
      // Display entropy statistics
      if (!entropies.empty()) {
        double max_entropy = *std::max_element(entropies.begin(), entropies.end());
        double min_entropy = *std::min_element(entropies.begin(), entropies.end());
        double avg_entropy = std::accumulate(entropies.begin(), entropies.end(), 0.0) / entropies.size();
        double max_possible = std::log2(3.0); // 3-class max entropy
        std::cout << "Entropy stats: min=" << min_entropy << ", max=" << max_entropy 
                  << ", avg=" << avg_entropy << ", max_possible=" << max_possible << std::endl;
      }
      
      // Use strategy-pattern based selection (default: Top-40%)
      EntropyStrategyType used_type = select_high_entropy_nodes_default(
          node_probs, high_entropy_nodes, entropy_effective_threshold);
      
      std::cout << "Entropy strategy: " << entropy_strategy_type_to_string(used_type)
                << ", selected " << high_entropy_nodes.size()
                << " nodes (effective threshold=" << entropy_effective_threshold
                << ") for focused search" << std::endl;
      
      // Build initial backbone route by taking top-K p_route
      // K = max(2, int(n * 0.2)) — we use the same heuristic as Python
      size_t n = locations.size();
      size_t k = std::max<size_t>(2, static_cast<size_t>(std::floor(n * 0.2)));

      // Ensure we have one prob entry per location (by coordinates). If not, skip.
      if (node_probs.size() == n) {
        // create index vector
        std::vector<size_t> idx(node_probs.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        // sort indices by p_route desc
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
          return node_probs[a].p_route > node_probs[b].p_route;
        });
        // pick top k and push corresponding Points
        for (size_t i = 0; i < k && i < idx.size(); ++i) {
          backbone_initial_route.push_back(node_probs[idx[i]].p);
        }

        std::cout << "Backbone initial route from attention (K=" << k << "): ";
        for (const auto &p : backbone_initial_route) std::cout << "(" << p.x << "," << p.y << ") ";
        std::cout << std::endl;
      } else {
        std::cout << "attention_probs.csv length mismatch: expected " << locations.size() << ", got " << node_probs.size() << ". Skipping backbone init." << std::endl;
      }
    } else {
      std::cout << "No attention_probs.csv found — skipping attention integration." << std::endl;
    }

    // 如果 backbone_initial_route 非空，则确保 ontour/offtour/vertex_map 与初始解一致
    std::vector<Point> backbone_filtered;
    if (!backbone_initial_route.empty()) {
      // Filter backbone points: only keep those that exist in locations
      // AND convert them to the actual Point objects from locations (not from CSV)
      for (const auto &bp : backbone_initial_route) {
        auto it_loc = std::find(locations.begin(), locations.end(), bp);
        if (it_loc != locations.end()) {
          // Use the Point object from locations, not from CSV
          backbone_filtered.push_back(*it_loc);
          // move backbone point into ontour (if it's in offtour)
          auto it_off = std::find(offtour.begin(), offtour.end(), *it_loc);
          if (it_off != offtour.end()) {
            offtour.erase(it_off);
            ontour.push_back(*it_loc);
          }
          // if already in ontour, that's fine
        } else {
          std::cout << "Warning: backbone point (" << bp.x << "," << bp.y 
                    << ") not found in locations, skipping" << std::endl;
        }
      }
      // rebuild vertex_map to reflect updated ontour/offtour
      build_vertex_map(locations, ontour, offtour, distance, vertex_map);
      std::cout << "Filtered backbone route: " << backbone_filtered.size() << " points" << std::endl;
    }

    // 执行贪婪搜索 返回贪婪搜索后最优解（禁忌搜索初始解）
    GreedyLocalSearch greedy_searcher =
        greedy_local_search(ontour, offtour, distance, vertex_map, locations, backbone_filtered);
    std::cout << "Greedy search completed." << std::endl;

    // TODO 创建搜索上下文结构体 searchctx

    // 执行禁忌搜索 骨架大概完成 具体逻辑TODO
    TabuSearch tabu_seracher = tabu_search(
        locations, greedy_searcher.get_distance(), greedy_searcher.get_ontour(),
        greedy_searcher.get_offtour(), greedy_searcher.get_vertex_map(),
        greedy_searcher.get_route(), greedy_searcher.get_cost());
    
    // Set entropy information for the tabu search if available
    if (!high_entropy_nodes.empty()) {
      tabu_seracher.set_entropy_info(high_entropy_nodes, entropy_effective_threshold);
    }

    std::cout << "Tabu search start" << std::endl;

    tabu_seracher.search(PATH_RELINKING_TIMES, DIVERSIFICATION,
                         TABU_LIST_LENGTH);

    auto len_trend = tabu_seracher.get_len_trend();
    auto iter_solution = tabu_seracher.get_iter_solution();
    double best_cost = tabu_seracher.get_best_cost();
    std::cout << "Best cost: " << best_cost << std::endl;

    // TODO 后续输出启发式最优路线耗时等逻辑
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}