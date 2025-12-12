#ifndef ATTENTION_READER_H
#define ATTENTION_READER_H

#include "input.h"
#include <string>
#include <vector>
#include <set>

struct NodeProb {
  Point p;
  double p_assign;
  double p_route;
  double p_loss;
};

// Read CSV with header: x,y,p_assign,p_route,p_loss
bool read_attention_probs(const std::string &filename,
                          std::vector<NodeProb> &out_probs);

// Calculate Shannon entropy for a node based on its state probabilities
// H = -sum(p_i * log(p_i)) for i in {assign, route, loss}
// Higher entropy means more uncertainty -> needs more search resources
double calculate_node_entropy(const NodeProb &node);

// Calculate entropy for all nodes and return a vector of entropies
void calculate_all_entropies(const std::vector<NodeProb> &probs,
                            std::vector<double> &out_entropies);

// Select nodes that need search focus based on entropy threshold
// Returns indices of nodes with entropy >= threshold
void select_high_entropy_nodes(const std::vector<NodeProb> &probs,
                              double entropy_threshold,
                              std::set<std::pair<int, int>> &out_node_coords);

#endif // ATTENTION_READER_H
