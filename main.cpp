#include "greedy.h"
#include "input.h"
#include "tabu_search.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <set>

// #define PATH_RELINKING_TIMES 50
// // 论文中终止条件：执行两次多样化后停止
// #define DIVERSIFICATION 2
// #define TABU_LIST_LENGTH 15

GreedyLocalSearch
greedy_local_search(std::vector<Point> ontour, std::vector<Point> offtour,
                    std::vector<std::vector<double>> distance,
                    std::map<size_t, VertexInfo> vertex_map,
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
            const std::map<size_t, VertexInfo> &vertex_map,
            const std::vector<Point> &route, const std::double_t cost,
            const std::map<size_t, double> &point_probs = {},
            StrategyConfig config = StrategyConfig()) {
  // TODO
  TabuSearch solver(locations, distance, ontour, offtour, vertex_map, route,
                    cost, point_probs, config);
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

    StrategyConfig config;
    bool lock_paper_baseline = false;
    auto apply_paper_baseline_defaults = [&config]() {
        config.use_neural_init = false;
        config.use_entropy = false;
        config.use_knn = false;
        config.use_frequency_based_diversification = false;
        config.tabu_list_length = 15;
        config.diversification_times = 2;
        config.path_relinking_times = 50;
        config.entropy_weight = 0.0;
    };
    if (argc >= 4) {
        std::string strategy = argv[3];
      if (strategy == "paper_baseline") {
        // Strict TS-SVRAP paper-aligned baseline:
        // keep core tabu/diversification/path-relinking, disable later-added guidance/acceleration.
        apply_paper_baseline_defaults();
        lock_paper_baseline = true;
      } else if (strategy == "nn_only") {
        apply_paper_baseline_defaults();
        config.use_neural_init = true;
        lock_paper_baseline = true;
      } else if (strategy == "knn_only") {
        apply_paper_baseline_defaults();
        config.use_knn = true;
        lock_paper_baseline = true;
      } else if (strategy == "entropy_only") {
        apply_paper_baseline_defaults();
        config.use_entropy = true;
        config.entropy_weight = 1.0;
        lock_paper_baseline = true;
      } else if (strategy == "baseline") {
            config.use_neural_init = false;
            config.use_entropy = false;
        } else if (strategy == "no_nn") {
            config.use_neural_init = false;
        } else if (strategy == "no_entropy") {
            config.use_entropy = false;
        } else if (strategy == "no_knn") {
            config.use_knn = false;
        } else if (strategy == "simple_div") {
            config.use_frequency_based_diversification = false;
        }
        std::cout << "Running Strategy: " << strategy << std::endl;
    }

    if (argc >= 5) {
      if (lock_paper_baseline) {
        std::cout << "paper_baseline: ignore K_NEIGHBORS override, use fixed value." << std::endl;
      } else {
        try {
            config.k_neighbors = std::stoi(argv[4]);
            std::cout << "Using K_NEIGHBORS = " << config.k_neighbors << std::endl;
        } catch (...) {
             std::cout << "Warning: failed to parse K from argv[4], keep default " << config.k_neighbors << std::endl;
        }
      }
    }

    if (argc >= 6) {
      if (lock_paper_baseline) {
        std::cout << "paper_baseline: ignore TABU_LIST_LENGTH override, use fixed value 15." << std::endl;
      } else {
        try {
            config.tabu_list_length = std::stoi(argv[5]);
            std::cout << "Using TABU_LIST_LENGTH = " << config.tabu_list_length << std::endl;
        } catch (...) {
             std::cout << "Warning: failed to parse TABU_LIST_LENGTH from argv[5], keep default " << config.tabu_list_length << std::endl;
        }
      }
    }

    if (argc >= 7) {
      if (lock_paper_baseline) {
        std::cout << "paper_baseline: ignore DIVERSIFICATION_TIMES override, use fixed value 2." << std::endl;
      } else {
        try {
            config.diversification_times = std::stoi(argv[6]);
            std::cout << "Using DIVERSIFICATION_TIMES = " << config.diversification_times << std::endl;
        } catch (...) {
             std::cout << "Warning: failed to parse DIVERSIFICATION_TIMES from argv[6], keep default " << config.diversification_times << std::endl;
        }
      }
    }

    if (argc >= 8) {
      if (lock_paper_baseline) {
        std::cout << "paper_baseline: ignore PATH_RELINKING_TIMES override, use fixed value 50." << std::endl;
      } else {
        try {
            config.path_relinking_times = std::stoi(argv[7]);
            std::cout << "Using PATH_RELINKING_TIMES = " << config.path_relinking_times << std::endl;
        } catch (...) {
             std::cout << "Warning: failed to parse PATH_RELINKING_TIMES from argv[7], keep default " << config.path_relinking_times << std::endl;
        }
      }
    }

    if (argc >= 9) {
      if (lock_paper_baseline) {
        std::cout << "paper_baseline: ignore ENTROPY_WEIGHT override, use fixed value 0." << std::endl;
      } else {
        try {
            config.entropy_weight = std::stod(argv[8]);
            std::cout << "Using ENTROPY_WEIGHT = " << config.entropy_weight << std::endl;
        } catch (...) {
             std::cout << "Warning: failed to parse ENTROPY_WEIGHT from argv[8], keep default " << config.entropy_weight << std::endl;
        }
      }
    }

    // Optional output path for per-node 0/1 route labels.
    // Usage: svrap.exe <alpha> <dataset> <strategy> <k> <tbl> <q> <t> <entropy_weight> <label_output_path>
    // Label format: one line per node, 1 means on-tour, 0 means off-tour.
    std::string label_output_path;
    if (argc >= 10) {
      label_output_path = argv[9];
      std::cout << "Label output enabled: " << label_output_path << std::endl;
    }

    // Optional: parse any --log-iterations=<path> argument anywhere in argv
    std::string iter_log_path = "";
    for (int ai = 1; ai < argc; ++ai) {
      std::string s = argv[ai];
      const std::string prefix = "--log-iterations=";
      if (s.rfind(prefix, 0) == 0) {
        iter_log_path = s.substr(prefix.size());
        std::cout << "Iteration logging enabled: " << iter_log_path << std::endl;
      }
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

        // 构建概率映射表，供 Tabu Search 多样化使用
        std::map<size_t, double> point_probs_map;
        if (config.use_neural_init && !probs.empty()) {
             for (const auto& point : locations) {
                 auto it = std::find_if(probs.begin(), probs.end(), [&point](const PointProb& pp) {
                     return pp.x == point.x && pp.y == point.y;
                 });
                 if (it != probs.end()) {
                     point_probs_map[point.id] = it->p_route;
                 }
             }
         }

        bool used_python_backbone = false;
        // 简单的校验：如果 probs 数据量足够且能匹配到当前 locations
        if (config.use_neural_init && !probs.empty()) {
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
            // 确认该点确实在当前 locations 中
            Point true_p;
            bool found = false;
            for (const auto &loc : locations) {
                if (loc.x == pp.x && loc.y == pp.y) {
                    found = true;
                    true_p = loc;
                    break;
                }
            }
            if (found) {
                backbone.push_back(true_p);
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
                if (b.id == loc.id) {
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
        std::map<size_t, VertexInfo> vertex_map;
        
        // Calculate entropy and identify high entropy points
        std::set<size_t> high_entropy_points;
        if (config.use_entropy && !probs.empty()) {
            std::vector<std::pair<double, size_t>> entropies;
            for (const auto& pp : probs) {
                // Check if point belongs to current instance
                size_t p_id = -1;
                bool found = false;
                for (const auto& loc : locations) {
                    if (loc.x == pp.x && loc.y == pp.y) {
                        found = true;
                        p_id = loc.id;
                        break;
                    }
                }
                if (!found) continue;

              double h = 0.0;
              if (pp.p_off > 1e-9) h -= pp.p_off * std::log(pp.p_off);
              if (pp.p_route > 1e-9) h -= pp.p_route * std::log(pp.p_route);
                entropies.push_back({h, p_id});
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
                        greedy_searcher.get_cost(),
                        point_probs_map,
                        config);

        std::cout << "Tabu search start" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        tabu_seracher.search(config.path_relinking_times, config.diversification_times,
                             config.tabu_list_length);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        double best_cost = tabu_seracher.get_best_cost();
        std::cout << "Tabu search finished in " << elapsed.count() << "s" << std::endl;
        std::cout << "Best cost for " << file << " = " << best_cost
                  << "\n";

        // If per-iteration logging requested, write iteration CSV
        if (!iter_log_path.empty()) {
          try {
            const auto &costs = tabu_seracher.get_iter_best_costs();
            std::ofstream itout(iter_log_path);
            if (itout.is_open()) {
              itout << "Iteration,BestCost\n";
              for (size_t i = 0; i < costs.size(); ++i) {
                itout << i + 1 << "," << costs[i] << "\n";
              }
              itout.close();
              std::cout << "Saved iteration log to " << iter_log_path << std::endl;
            } else {
              std::cerr << "Warning: failed to open iteration log file: " << iter_log_path << std::endl;
            }
          } catch (const std::exception &e) {
            std::cerr << "Error writing iteration log: " << e.what() << std::endl;
          }
        }

        if (!label_output_path.empty()) {
          const auto &best_route = tabu_seracher.get_best_solution();
          std::vector<int> labels(locations.size(), 0);
          for (const auto &p : best_route) {
            if (p.id < labels.size()) {
              labels[p.id] = 1;
            }
          }

          std::ofstream label_out(label_output_path);
          if (!label_out.is_open()) {
            std::cerr << "Warning: failed to open label output file: "
                      << label_output_path << std::endl;
          } else {
            for (size_t i = 0; i < labels.size(); ++i) {
              label_out << labels[i] << "\n";
            }
            std::cout << "Saved route labels to " << label_output_path
                      << " (n=" << labels.size() << ")" << std::endl;
          }
        }
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