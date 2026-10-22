// Sampling distributions and speculative verification (mirrors python/woodcock/sampling.py).
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace woodcock {

struct SamplingParams {
  float temperature = 0.f;  // 0 => greedy
  int top_k = 0;
  float top_p = 1.f;
  uint64_t seed = 0;
  bool greedy() const { return temperature <= 0.f; }
};

using Rng = std::mt19937_64;

std::vector<float> probs_from_logits(const float* z, int n, const SamplingParams& sp);
int argmax(const float* z, int n);
int sample(const std::vector<double>& p, Rng& rng);
std::vector<int> sample_without_replacement(const std::vector<double>& q, int k, Rng& rng);

struct DraftTree {
  std::vector<int> tokens, parents, depth;  // node 0 = root (parents[0] = -1)
  std::vector<std::vector<int>> children;   // in draw order
  void build_children();
  int size() const { return int(tokens.size()); }
  bool is_chain() const;
};

// [N][prefix + N]: every node sees the prefix and its ancestors (itself included)
std::vector<uint8_t> tree_mask(const DraftTree& t, int prefix);

struct Verdict {
  std::vector<int> path;  // accepted node ids (excluding the root)
  int next = -1;          // token sampled from the target after the path
};

Verdict verify_tree_greedy(const std::vector<int>& target_argmax, const DraftTree& t);
// p(v): target distribution at node v; q(v): draft distribution v's children were drawn from
Verdict verify_tree_stochastic(const std::function<std::vector<double>(int)>& p,
                               const std::function<const std::vector<double>&(int)>& q, const DraftTree& t, Rng& rng);

}  // namespace woodcock
