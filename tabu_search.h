#include "input.h"
#include <vector>

#define PATH_RELINKING_TIMES 50
#define DIVERSIFICATION 3
#define T_VALUE 50
#define Q_VALUE 3

class TabuSearch {

public:
  explicit TabuSearch(
      const std::vector<Point> &locations,
      const std::vector<std::vector<double>> &distance,
      const std::vector<Point> &ontour, const std::vector<Point> &offtour,
      const std::map<std::pair<int, int>, VertexInfo> &vertex_map,
      const std::vector<Point> &route);

      void search(int T,int Q);

private:
  std::vector<Point> locations_;
  std::vector<std::vector<double>> distance_;
  std::vector<Point> ontour_;
  std::vector<Point> offtour_;
  std::map<std::pair<int, int>, VertexInfo> vertex_map_;
  //   记录每个点是否在路径
  //  保存每个 非路径点 到 最近路径点 的距离（best_cost *0.5 计入总成本）
  std::vector<Point> route_;
  std::double_t solution_cost_;

  void diversication();
  void operation_style();
  void path_relinking();
};  



class TabuInfo{
  private:
    std::vector<Point> tabuList;
    std::vector<int> tabuTime;
    int tabu_limit;
  public:
    // 更新禁忌表内容及长度（动态修改数组长度）
    void update_tabu();
};