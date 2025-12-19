#ifndef TABU_SEARCH_H
#define TABU_SEARCH_H

#include "input.h"
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// 定义操作的 Key 类型：单个点(Add/Drop) 或 点对(TwoOpt)
using OpKey = std::variant<std::vector<Point>,
                           std::pair<std::vector<Point>, std::vector<Point>>>;

class TabuInfo {
private:
  std::vector<OpKey> tabu_list;
  std::vector<int> tabu_time;
  int tabu_limit;
  int current_tabu_size;

public:
  explicit TabuInfo(int tabu_limit);
  // 更新禁忌表内容及长度（动态修改数组长度）
  void update_tabu();
  void reset_list();
  void set_limit(int new_limit);
  bool is_tabu_iter(const OpKey &iter);
  void add_tabu_iter(const OpKey &iter, int tabu_limit);
};

class TabuSearch {

public:
  explicit TabuSearch(
      const std::vector<Point> &locations,
      const std::vector<std::vector<double>> &distance,
      const std::vector<Point> &ontour, const std::vector<Point> &offtour,
      const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
      const std::vector<Point> &route, const std::double_t &cost,
      const std::map<std::pair<int, int>, double> &point_probs = {});

  void search(int T, int Q, int TBL);

  const std::vector<int> &get_len_trend() const { return cost_trend_; }
  const std::vector<Point> &get_iter_solution() const { return iter_solution_; }
  double get_best_cost() const { return best_cost_; }

private:
  // 最优解所对应成本的变化趋势
  std::vector<int> cost_trend_;
  std::vector<Point> locations_;
  std::vector<std::vector<double>> distance_;
  std::vector<Point> ontour_;
  std::vector<Point> offtour_;
  std::map<std::pair<int, int>, VertexInfo> vertex_map_;
  //   记录每个点是否在路径
  //  保存每个 非路径点 到 最近路径点 的距离（best_cost *0.5 计入总成本）
  std::vector<Point> route_;
  std::double_t solution_cost_;
  // 返回的值
  std::vector<Point> iter_solution_;
  std::double_t best_cost_;

  // 当前已知的全局最优解（Champion）及其前一个 Champion
  std::vector<Point> champion_solution_;
  std::map<std::pair<int, int>, VertexInfo> champion_vertex_map_;
  double champion_cost_;
  std::vector<Point> prev_champion_solution_;

  // Champion 频率信息：在所有 Champion 解中，顶点处于 on/off 状态的计数
  std::map<std::pair<int, int>, int> champion_on_count_;
  std::map<std::pair<int, int>, int> champion_off_count_;
  int champion_sample_count_ = 0;

  // 存储每个点的预测概率 (x, y) -> p_route
  std::map<std::pair<int, int>, double> point_probs_;

  // 多样化方法
  std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>>
  diversication(const std::vector<Point> &champion_route,
                std::map<std::pair<int, int>, VertexInfo> iter_dic,
                std::vector<OpKey> &diversification_moves);

  std::tuple<std::vector<Point>, double>
  path_relinking(const std::vector<Point> &prev_champion,
                 const std::vector<Point> &new_champion,
                 std::map<std::pair<int, int>, VertexInfo> iter_dic,
                 std::vector<OpKey> &relink_moves);

  // 根据给定的 Champion 解，更新 m_on(i)、m_off(i) 频率统计
  void update_champion_frequencies(const std::vector<Point> &champion_route);

  // 熵退火参数
  double lambda_0_ = 1.0;

  // Precomputed K-nearest neighbors for each point (by index)
  std::vector<std::vector<int>> nearby_table_;
  const int K_NEIGHBORS = 20;

    // 邻域操作: 返回 {新路径, 新Map, 新Cost, 操作涉及的点(用于禁忌表)}
    std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>,
         double, std::vector<Point>>
    operation_style(const std::vector<Point> &iter_sol,
            const std::map<std::pair<int, int>, VertexInfo> &iter_dic,
            double base_allocation_cost);
};

#endif