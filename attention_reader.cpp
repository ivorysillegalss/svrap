#include "attention_reader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

// Copliot

bool read_attention_probs(const std::string &filename,
                          std::vector<NodeProb> &out_probs) {
  out_probs.clear();
  std::ifstream in(filename);
  if (!in.is_open()) {
    return false;
  }

  std::string line;
  // read header line (if any)
  if (!std::getline(in, line))
    return false;

  // If the header doesn't contain letters, it might be data — handle that
  bool header = false;
  for (char c : line) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      header = true;
      break;
    }
  }

  if (!header) {
    // first line was data — process it
    std::istringstream ss(line);
    std::string token;
    std::vector<std::string> toks;
    while (std::getline(ss, token, ','))
      toks.push_back(token);
    if (toks.size() >= 5) {
      try {
        NodeProb np;
        np.p.x = std::stoi(toks[0]);
        np.p.y = std::stoi(toks[1]);
        np.p_assign = std::stod(toks[2]);
        np.p_route = std::stod(toks[3]);
        np.p_loss = std::stod(toks[4]);
        out_probs.push_back(np);
      } catch (...) {
      }
    }
  }

  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::istringstream ss(line);
    std::string token;
    std::vector<std::string> toks;
    while (std::getline(ss, token, ','))
      toks.push_back(token);
    if (toks.size() < 5)
      continue;
    try {
      NodeProb np;
      np.p.x = std::stoi(toks[0]);
      np.p.y = std::stoi(toks[1]);
      np.p_assign = std::stod(toks[2]);
      np.p_route = std::stod(toks[3]);
      np.p_loss = std::stod(toks[4]);
      out_probs.push_back(np);
    } catch (...) {
      // skip malformed line
      continue;
    }
  }

  return true;
}

// Calculate Shannon entropy for a single node
// 计算信息熵 对应后期的搜索资源倾斜
double calculate_node_entropy(const NodeProb &node) {
  double entropy = 0.0;
  const double epsilon = 1e-10; // 避免 log(0)
  
  std::vector<double> probs = {node.p_assign, node.p_route, node.p_loss};
  
  for (double p : probs) {
    if (p > epsilon) {
      entropy -= p * std::log2(p);
    }
  }
  
  return entropy;
}

// Calculate entropy for all nodes
void calculate_all_entropies(const std::vector<NodeProb> &probs,
                            std::vector<double> &out_entropies) {
  out_entropies.clear();
  out_entropies.reserve(probs.size());
  
  for (const auto &node : probs) {
    out_entropies.push_back(calculate_node_entropy(node));
  }
}

// Select nodes that need search focus based on entropy threshold
// 高熵意味着不确定性高,需要更多搜索资源
void select_high_entropy_nodes(const std::vector<NodeProb> &probs,
                              double entropy_threshold,
                              std::set<std::pair<int, int>> &out_node_coords) {
  out_node_coords.clear();
  
  for (const auto &node : probs) {
    double entropy = calculate_node_entropy(node);
    if (entropy >= entropy_threshold) {
      out_node_coords.insert({node.p.x, node.p.y});
    }
  }
}
