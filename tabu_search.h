#include "input.h"
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

class TabuSearch {

public:
  explicit TabuSearch(
      const std::vector<Point> &locations,
      const std::vector<std::vector<double>> &distance,
      const std::vector<Point> &ontour, const std::vector<Point> &offtour,
      const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
      const std::vector<Point> &route, const std::double_t &cost);

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

  // 多样化方法
  std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>>
  diversication(std::vector<std::vector<Point>> solution_set,
                std::map<std::pair<int, int>, VertexInfo> iter_dic, int number);
  std::tuple<std::vector<Point>, double>
  path_relinking(std::vector<std::vector<Point>> solution_set,
                 std::map<std::pair<int, int>, VertexInfo> iter_dic);

  // TODO 返回值类型优雅设置为类 对应原py代码中
  // operation_style_all =
  // [route,dic,newroute_cost,[twooptvertice[0],twooptvertice[1]]]的返回结构
  // (这里第四项有可能是string 也有可能是两个point)
  // using OperationDesc =
  //     std::variant<std::monostate,          // 没操作（备用）
  //                  std::vector<std::string>,             // 比如 "取(3,4)",
  //                  "删(5,6)" std::vector<Point>       // 或者你喜欢
  //                  vector<Point>{p1, p2}
  //                  >;
  std::tuple<std::vector<Point>, std::map<std::pair<int, int>, VertexInfo>,
             double, std::vector<Point>>
  operation_style(std::vector<Point> iter_sol,
                  std::map<std::pair<int, int>, VertexInfo> iter_dic);
};

class TabuInfo {
private:
  std::vector<std::variant<std::vector<Point>,
                           std::pair<std::vector<Point>, std::vector<Point>>>>
      tabu_list;
  std::vector<int> tabu_time;
  int tabu_limit;
  int current_tabu_size;

public:
  // 更新禁忌表内容及长度（动态修改数组长度）
  void update_tabu();
  void add_tabu_num();
  void reset_list();
  void set_limit(int new_limit);
  explicit TabuInfo(int tabu_limit);
  bool
  is_tabu_iter(std::variant<std::vector<Point>,
                            std::pair<std::vector<Point>, std::vector<Point>>>
                   iter);
  void
  add_tabu_iter(std::variant<std::vector<Point>,
                             std::pair<std::vector<Point>, std::vector<Point>>>
                    iter,
                int tabu_limit);
};