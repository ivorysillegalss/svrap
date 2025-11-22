#include "tabu_search.h"
#include "greedy.h"
#include "input.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

TabuSearch::TabuSearch(
    const std::vector<Point> &locations,
    const std::vector<std::vector<double>> &distance,
    const std::vector<Point> &ontour, const std::vector<Point> &offtour,
    const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
    const std::vector<Point> &route, const std::double_t &cost)
    : locations_(locations), distance_(distance), ontour_(ontour),
      offtour_(offtour), vertex_map_(vertex_map), route_(route),
      solution_cost_(cost) {
  cost_trend_.push_back(cost);
}

void TabuInfo::add_tabu_num() { current_tabu_size++; }

void TabuInfo::reset_list() {
  tabu_list.clear();
  tabu_time.clear();
  current_tabu_size = 0;
}

void TabuInfo::update_tabu() {
  for (int i = 0; i < tabu_time.size(); ++i) {
    tabu_time[i]--;
    if (tabu_time[i] == 0) {
      tabu_list[i] = std::vector<Point>{};
    }
  }

  // 索引更新移除非0项 压缩迁移后数组
  size_t write_idx = 0;
  for (size_t i = 0; i < tabu_list.size(); ++i) {
    if (tabu_time[i] != 0) {
      if (write_idx != i) {
        tabu_list[write_idx] = std::move(tabu_list[i]);
        tabu_time[write_idx] = tabu_time[i];
      }
      ++write_idx;
    }
  }
  tabu_list.resize(write_idx);
  tabu_time.resize(write_idx);
};

void TabuInfo::set_limit(int new_limit) {
  tabu_limit = new_limit;
  return;
}

// 判断当前禁忌表中是否存在当前遍历到的元素
bool TabuInfo::is_tabu_iter(
    std::variant<std::vector<Point>,
                 std::pair<std::vector<Point>, std::vector<Point>>>
        iter) {
  return std::find(tabu_list.begin(), tabu_list.end(), iter) == tabu_list.end();
}

void TabuInfo ::add_tabu_iter(
    std::variant<std::vector<Point>,
                 std::pair<std::vector<Point>, std::vector<Point>>>
        iter,
    int tabu_limit) {
  tabu_list.push_back(iter);
  tabu_time.push_back(tabu_limit);
  current_tabu_size++;
}

std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>>
TabuSearch::diversication(std::vector<std::vector<Point>> solution_set,
                          std::map<std::pair<int, int>, VertexInfo> iter_dic,
                          int number) {
  // TODO
  return {{}, {}};
}

void TabuSearch::search(int T, int Q, int TBL) {
  int t = 0, q = 0;
  // t q 代表当前已经执行了路径重连和多样化的次数
  std::vector<std::vector<Point>> soluntion_set{route_};
  // 所有历史最优解（需要存储一个列表 最后在这个列表的基础上进行比较得出）
  std::vector<Point> iter_solution = route_;
  // 当前的迭代解
  std::map<std::pair<int, int>, VertexInfo> iter_dic = vertex_map_;
  // 当前状态的字典 —— 当前状态下的位置字典
  double best_cost = solution_cost_;
  // 启发搜索过程下全局最优成本

  TabuInfo tabu(TBL);
  // 初始化禁忌搜索对象

  // 循环直至路径重连和多样化达到阈值
  while (true) {
    std::vector<double> itercost;
    // 邻域解成本列表
    std::vector<std::vector<Point>> operated_solution;
    // 邻域解成本列表
    std::vector<std::map<std::pair<int, int>, VertexInfo>> operated_dic;
    // 邻域解字典
    using OpKey =
        std::variant<std::vector<Point>,
                     std::pair<std::vector<Point>, std::vector<Point>>>;
    std::vector<OpKey> iterchange;
    // iterchange; 已尝试操作（visited列表 表示二操作中取其一）

    int times = 0;
    // 已生成的有效邻域数

    // 记录当前找到的最优邻域解的个数
    while (times < 50) {
      auto neighbor_solution = operation_style();
      // 通过进行操作 生成邻域数据
      auto &op = std::get<3>(neighbor_solution);
      bool valid = false;

      // 检查正逆序是否已经尝试
      if (op.size() == 2) {
        std::vector<Point> nerireverse = {op[1], op[0]};
        auto pair1 = std::vector<std::vector<Point>>{{op, nerireverse}};
        auto pair2 = std::vector<std::vector<Point>>{{nerireverse, op}};
        if (std::find(iterchange.begin(), iterchange.end(), pair1) ==
                iterchange.end() &&
            std::find(iterchange.begin(), iterchange.end(), pair2) ==
                iterchange.end()) {
          valid = true;
          // op 非{int ,int}类型 需修改iterchange 进行后续邻域解处理
          // Point p = {op,rev};
          itercost.push_back(std::get<2>(neighbor_solution));
          operated_solution.push_back(std::get<0>(neighbor_solution));
          operated_dic.push_back(std::get<1>(neighbor_solution));
          iterchange.emplace_back(std::make_pair(op, nerireverse));
          times++;
        }
      } else if (op.size() == 1) {
        if (std::find(iterchange.begin(), iterchange.end(), op) ==
            iterchange.end()) {
          itercost.push_back(std::get<2>(neighbor_solution));
          operated_solution.push_back(std::get<0>(neighbor_solution));
          operated_dic.push_back(std::get<1>(neighbor_solution));
          iterchange.emplace_back(op);
          times++;
        }
      }
    }

    auto min_iter_cost_p = std::min_element(itercost.begin(), itercost.end());
    double min_iter_cost = *min_iter_cost_p;
    // 获得索引
    std::size_t min_iter_cost_index = min_iter_cost_p - itercost.begin();

    auto min_iter_solution = operated_solution[min_iter_cost_index];
    auto min_iter_dic = operated_dic[min_iter_cost_index];
    auto min_iter_operated = iterchange[min_iter_cost_index];

    if (min_iter_cost <= best_cost && tabu.is_tabu_iter(min_iter_operated)) {
      t++;
      if (t == T) {
        auto diversify = diversication(soluntion_set, iter_dic, 2);
        q += 1;

        if (q < Q) {
          iter_solution = std::get<0>(diversify);
          iter_dic = std::get<1>(diversify);
          soluntion_set.push_back(iter_solution);

          GreedyLocalSearch calculater(iter_solution, iter_dic);
          best_cost = calculater.tabu_cacl_cost();
          cost_trend_.push_back(best_cost);
          t = 0;
          tabu.reset_list();
          continue;

        } else if (q == Q) {
          GreedyLocalSearch calculater(std::get<0>(diversify),
                                       std::get<1>(diversify));
          double last_cost = calculater.tabu_cacl_cost();
          if (last_cost <= best_cost) {
            iter_solution = std::get<0>(diversify);
            iter_dic = std::get<1>(diversify);
            best_cost = last_cost;
            cost_trend_.push_back(best_cost);
            break;
          } else if (last_cost > best_cost) {
            break;
          }
        }
      } else if (t < T) {
        tabu.update_tabu();
        continue;
      }
    } else if (min_iter_cost <= best_cost &&
               tabu.is_tabu_iter(min_iter_operated)) {
      soluntion_set.push_back(min_iter_solution);
      // 路径重连部分
      auto path_relink = path_relinking(soluntion_set, min_iter_dic);
      auto pr0 = std::get<0>(path_relink);
      auto pr1 = std::get<1>(path_relink);
      if (pr1 <= min_iter_cost) {
        soluntion_set.back() = pr0;
        iter_solution = pr0;
        iter_dic = min_iter_dic;
        best_cost = pr1;
        cost_trend_.push_back(best_cost);
      } else if (pr1 > min_iter_cost) {
        iter_solution = min_iter_solution;
        iter_dic = min_iter_dic;
        best_cost = min_iter_cost;
        cost_trend_.push_back(best_cost);
      }
      t = 0;
      // TODO 禁忌表长度修改为定义
      tabu.add_tabu_iter(min_iter_operated, 15);
      tabu.update_tabu();
      continue;
    } else if (min_iter_cost > best_cost) {
      t++;
      if (t == T) {
        auto diversify = diversication(soluntion_set, iter_dic, 2);
        auto dy0 = std::get<0>(diversify);
        auto dy1 = std::get<1>(diversify);
        q++;
        if (q < Q) {
          soluntion_set.push_back(dy0);
          iter_solution = dy0;
          iter_dic = dy1;
          GreedyLocalSearch calculater(iter_solution, iter_dic);
          best_cost = calculater.tabu_cacl_cost();
          cost_trend_.push_back(best_cost);
          t = 0;
          tabu.reset_list();
          continue;
        } else if (q == Q) {
          GreedyLocalSearch calculater(dy0, dy1);
          auto last_cost = calculater.tabu_cacl_cost();
          if(last_cost <= best_cost){
            iter_solution = dy0;
            iter_dic = dy1;
            best_cost = last_cost;
            cost_trend_.push_back(best_cost);
            break;
          }else if(last_cost > best_cost){
            break;
          }
        }
      }else if(t < T){
        tabu.update_tabu();
        continue;
      }
    }
  }
  
  // TODO 计算完毕的后置逻辑
  // return {iter_solution,best_cost,cost_trend_};
};