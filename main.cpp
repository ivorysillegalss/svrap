#include "greedy.h"
#include "input.h"
#include "tabu_search.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <set>

#define PATH_RELINKING_TIMES 50
// 论文中终止条件：执行两次多样化后停止
#define DIVERSIFICATION 2
#define TABU_LIST_LENGTH 15

GreedyLocalSearch
greedy_local_search(std::vector<Point> ontour, std::vector<Point> offtour,
                    std::vector<std::vector<double>> distance,
                    std::map<std::pair<int, int>, VertexInfo> vertex_map,
                    std::vector<Point> locations) {
  // 对于每个 off_vertice 中的点，计算它到所有
  // on_vertice中点的距离（使用之前的 distance 矩阵）。
  // 找到距离最小的点（best_vertice）和对应的最小距离（best_cost）。
  calculate_nearest_cost(ontour, offtour, distance, vertex_map);

  // 进行第一次运算 得到进行后续 贪婪搜索的初始解
  // 后续使用贪婪局部搜索进行优化
  std::vector<Point> initial_route =
      nearest_neighbour(ontour, distance, vertex_map);
  // 调试输出已注释，避免在批量实验时生成过多日志。
  // 如需查看初始路径，可以取消下面注释。
  // std::cout << "初始路径：" << std::endl;
  // for (const auto &p : initial_route) {
  //   std::cout << "(" << p.x << ", " << p.y << ")" << std::endl;
  // }
  // std::cout << "初始路径长度：" << initial_route.size() << std::endl;

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

int main(int argc, char **argv) {
  try {
    // 可选：通过命令行第一个参数设置 ALPHA
    // 例如: svrap.exe 3  -> ALPHA = 3.0
    if (argc >= 2) {
      try {
        ALPHA = std::atof(argv[1]);
        std::cout << "Using ALPHA = " << ALPHA << std::endl;
      } catch (...) {
        std::cout << "Warning: failed to parse ALPHA from argv[1], keep default "
                  << ALPHA << std::endl;
      }
    } else {
      std::cout << "Using default ALPHA = " << ALPHA << std::endl;
    }

    // 如果提供了第二个参数，则仅针对该数据集运行；
    // 否则批量处理 formatted_dataset 目录下的所有标准实例。
    std::vector<std::string> instance_files;
    if (argc >= 3) {
      instance_files.push_back(argv[2]);
    } else {
      instance_files = {
          "formatted_dataset/berlin52.txt",  "formatted_dataset/bier127.txt",
          "formatted_dataset/ch130.txt",     "formatted_dataset/ch150.txt",
          "formatted_dataset/d198.txt",      "formatted_dataset/d493.txt",
          "formatted_dataset/eil101.txt",    "formatted_dataset/eil51.txt",
          "formatted_dataset/eil76.txt",     "formatted_dataset/fl1577.txt",
          "formatted_dataset/gr120.txt",     "formatted_dataset/gr137.txt",
          "formatted_dataset/gr96.txt",      "formatted_dataset/kroA100.txt",
          "formatted_dataset/kroA150.txt",   "formatted_dataset/kroA200.txt",
          "formatted_dataset/kroB100.txt",   "formatted_dataset/kroB150.txt",
          "formatted_dataset/kroB200.txt",   "formatted_dataset/kroC100.txt",
          "formatted_dataset/kroD100.txt",   "formatted_dataset/kroE100.txt",
          "formatted_dataset/lin105.txt",    "formatted_dataset/pr107.txt",
          "formatted_dataset/pr124.txt",     "formatted_dataset/pr136.txt",
          "formatted_dataset/pr144.txt",     "formatted_dataset/pr152.txt",
          "formatted_dataset/pr76.txt",      "formatted_dataset/rat195.txt",
          "formatted_dataset/rat783.txt",    "formatted_dataset/rat99.txt",
          "formatted_dataset/rd100.txt",     "formatted_dataset/st70.txt",
          "formatted_dataset/u159.txt"};
    }

    for (const auto &file : instance_files) {
      try {
        std::cout << "==============================\n";
        std::cout << "Instance: " << file << " (ALPHA=" << ALPHA << ")\n";

        // 每个实例单独读取坐标
        std::vector<Point> locations;
        read_coordinates(file, locations);
        std::cout << "Read " << locations.size() << " points." << std::endl;

        // 为该实例尝试加载对应的隔离成本文件：
        // 例如 formatted_dataset/eil76_iso.txt
        std::string iso_file = file;
        if (iso_file.size() >= 4 &&
            iso_file.substr(iso_file.size() - 4) == ".txt") {
          iso_file.insert(iso_file.size() - 4, "_iso");
        } else {
          iso_file += "_iso";
        }
        read_isolation_costs(iso_file);

        // 计算距离
        std::vector<std::vector<double>> distance;
        compute_distances(locations, distance);
        std::cout << "Compute distance matrix." << std::endl;

        // 构造初始 on-tour / off-tour 划分：
        std::vector<Point> ontour;
        std::vector<Point> offtour;

        // 尝试读取 Python 生成的 attention_probs.csv
        std::vector<PointProb> probs;
        read_attention_probs("attention_probs.csv", probs);

        bool used_python_backbone = false;
        // 简单的校验：如果 probs 数据量足够且能匹配到当前 locations
        if (!probs.empty()) {
          // 按 p_route 降序排序
          std::sort(probs.begin(), probs.end(),
                    [](const PointProb &a, const PointProb &b) {
                      return a.p_route > b.p_route;
                    });

          // 选取 Top K (20%)
          size_t K = std::max((size_t)2, (size_t)(locations.size() * 0.2));
          std::vector<Point> backbone;

          // 提取骨干
          for (const auto &pp : probs) {
            if (backbone.size() >= K)
              break;
            Point p(pp.x, pp.y);
            // 确认该点确实在当前 locations 中
            bool found = false;
            for (const auto &loc : locations) {
              if (loc.x == p.x && loc.y == p.y) {
                found = true;
                break;
              }
            }
            if (found) {
              backbone.push_back(p);
            }
          }

          // 如果成功提取到骨干
          if (backbone.size() >= 2) {
            used_python_backbone = true;
            ontour = backbone;
            // 其余点放入 offtour
            for (const auto &loc : locations) {
              bool is_backbone = false;
              for (const auto &b : backbone) {
                if (b.x == loc.x && b.y == loc.y) {
                  is_backbone = true;
                  break;
                }
              }
              if (!is_backbone) {
                offtour.push_back(loc);
              }
            }
            std::cout << "Initialized with Python backbone (size "
                      << ontour.size() << ")" << std::endl;
          }
        }

        if (!used_python_backbone) {
          // 默认：所有点都在路径上
          ontour = locations;
          // offtour 为空
        }

        std::cout << "Points in ontour size: " << ontour.size()
            << std::endl;
        std::cout << "Points in offtour size: " << offtour.size()
            << std::endl;

        // 构建点信息集
        std::map<std::pair<int, int>, VertexInfo> vertex_map;
        
        // Calculate entropy and identify high entropy points
        std::set<std::pair<int, int>> high_entropy_points;
        if (!probs.empty()) {
            std::vector<std::pair<double, std::pair<int, int>>> entropies;
            for (const auto& pp : probs) {
                // Check if point belongs to current instance
                bool found = false;
                for (const auto& loc : locations) {
                    if (loc.x == pp.x && loc.y == pp.y) {
                        found = true;
                        break;
                    }
                }
                if (!found) continue;

                double h = 0.0;
                if (pp.p_assign > 1e-9) h -= pp.p_assign * std::log(pp.p_assign);
                if (pp.p_route > 1e-9) h -= pp.p_route * std::log(pp.p_route);
                if (pp.p_loss > 1e-9) h -= pp.p_loss * std::log(pp.p_loss);
                entropies.push_back({h, {pp.x, pp.y}});
            }
            
            if (!entropies.empty()) {
                std::sort(entropies.rbegin(), entropies.rend()); // Descending entropy
                size_t he_count = std::max((size_t)1, (size_t)(entropies.size() * 0.2)); // Top 20%
                for (size_t i = 0; i < he_count; ++i) {
                    high_entropy_points.insert(entropies[i].second);
                }
                std::cout << "Identified " << high_entropy_points.size() << " high entropy points." << std::endl;
            }
        }

        build_vertex_map(locations, ontour, offtour, distance, vertex_map, high_entropy_points);
        std::cout << "Build points info done" << std::endl;

        // 执行贪婪搜索 返回贪婪搜索后最优解（禁忌搜索初始解）
        GreedyLocalSearch greedy_searcher = greedy_local_search(
            ontour, offtour, distance, vertex_map, locations);
        std::cout << "Greedy search done" << std::endl;

        // 执行禁忌搜索
        TabuSearch tabu_seracher =
            tabu_search(locations, greedy_searcher.get_distance(),
                        greedy_searcher.get_ontour(),
                        greedy_searcher.get_offtour(),
                        greedy_searcher.get_vertex_map(),
                        greedy_searcher.get_route(),
                        greedy_searcher.get_cost());

        std::cout << "Tabu search start" << std::endl;

        tabu_seracher.search(PATH_RELINKING_TIMES, DIVERSIFICATION,
                             TABU_LIST_LENGTH);

        double best_cost = tabu_seracher.get_best_cost();
        std::cout << "Best cost for " << file << " = " << best_cost
                  << "\n";
      } catch (const std::exception &e) {
        std::cerr << "Error while solving instance " << file << ": "
                  << e.what() << std::endl;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}