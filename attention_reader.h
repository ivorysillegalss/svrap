#ifndef ATTENTION_READER_H
#define ATTENTION_READER_H

#include "input.h"
#include <string>
#include <vector>

struct NodeProb {
  Point p;
  double p_assign;
  double p_route;
  double p_loss;
};

// Read CSV with header: x,y,p_assign,p_route,p_loss
bool read_attention_probs(const std::string &filename,
                          std::vector<NodeProb> &out_probs);

#endif // ATTENTION_READER_H
