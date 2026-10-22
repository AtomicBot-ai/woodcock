// Frozen target model adapter: Qwen3-family decoder (dense), CPU fp32.
#pragma once

#include <array>
#include <filesystem>
#include <vector>

#include "woodcock/cache.h"
#include "woodcock/weights.h"

namespace woodcock {

struct Qwen3Config {
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
  float eps = 1e-6f, rope_theta = 1e6f;
  bool tie_embeddings = true;
  static Qwen3Config from_json_file(const std::filesystem::path& p);
};

struct ForwardOut {
  std::vector<float> logits;             // [rows][vocab] for rows logits_from..L-1
  std::vector<std::vector<float>> taps;  // per tap layer: [L][hidden] residual stream
};

// Positions and mask are optional: by default positions continue from the cache
// size and the mask is causal. mask is [L][cache.size() + L], 1 = visible.
struct ForwardArgs {
  const std::vector<int>* positions = nullptr;
  const std::vector<uint8_t>* mask = nullptr;
  bool logits = true;
  int logits_from = 0;  // compute logits only for rows >= logits_from
};

class Qwen3 {
 public:
  explicit Qwen3(const std::filesystem::path& dir, std::array<int, 3> taps = {-1, -1, -1});
  ForwardOut forward(const std::vector<int>& tokens, KVCache& cache, const ForwardArgs& a = {}) const;
  KVCache make_cache() const { return KVCache(cfg_.layers, cfg_.kv_heads, cfg_.head_dim); }
  const Qwen3Config& config() const { return cfg_; }
  const std::array<int, 3>& taps() const { return taps_; }
  const Tensor& embedding() const { return embed_; }

 private:
  struct Layer {
    Tensor q, k, v, o, q_norm, k_norm, in_norm, post_norm, gate, up, down;
  };
  Qwen3Config cfg_;
  std::array<int, 3> taps_;
  Tensor embed_, norm_, head_;  // head_ is empty when tied to embed_
  bool tied_ = true;
  std::vector<Layer> layers_;
};

// shared decoder pieces (used by the drafter too)
std::vector<int> default_positions(int start, int L);
std::vector<uint8_t> causal_mask(int prefix, int L, int window = 0);

}  // namespace woodcock
