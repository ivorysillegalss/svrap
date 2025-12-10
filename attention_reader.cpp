#include "attention_reader.h"
#include <fstream>
#include <iostream>
#include <sstream>

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
