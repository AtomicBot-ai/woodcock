// Woodcock drafter (mirrors python/woodcock/drafter.py; same weights and config):
// EAGLE-3 / EAGLE-3.1 architecture — the fused target feature is the residual stream,
// the token embedding enters layer 0's attention as the auxiliary stream.
// Output: out = RMSNorm(h_last); with post_norm, out is also the next-step feature.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "woodcock/cache.h"
#include "woodcock/weights.h"

namespace woodcock {

struct DrafterConfig {
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
  int target_hidden = 0, num_features = 3;
  float eps = 1e-6f, rope_theta = 1e6f;
  bool post_norm = true, feature_norm = true;
  int window = 0;  // 0 = full attention
  std::vector<int> feature_layers;
  static DrafterConfig from_json_file(const std::filesystem::path& p);
};

struct DraftOut {
  std::vector<float> hidden;  // [L][d] un-normalised residual
  std::vector<float> out;     // [L][d] RMSNorm(hidden)
};

class Drafter {
 public:
  explicit Drafter(const std::filesystem::path& dir);
  const DrafterConfig& config() const { return cfg_; }
  KVCache make_cache() const { return KVCache(cfg_.layers, cfg_.kv_heads, cfg_.head_dim); }

  // taps: num_features arrays of [L][target_hidden] -> g [L][d]
  std::vector<float> fuse(const std::vector<std::vector<float>>& taps, int L) const;
  // tokens [L], feat [L][d]; optional explicit positions / mask ([L][cache+L])
  DraftOut forward(const std::vector<int>& tokens, const std::vector<float>& feat, KVCache& cache,
                   const std::vector<int>* positions = nullptr, const std::vector<uint8_t>* mask = nullptr) const;
  const float* feedback_row(const DraftOut& o, int row) const {
    return (cfg_.post_norm ? o.out.data() : o.hidden.data()) + int64_t(row) * cfg_.hidden;
  }
  // full-vocabulary logits for one row
  void logits(const float* out_row, std::vector<float>& z) const;

 private:
  struct Layer {
    Tensor q, k, v, o, q_norm, k_norm, in_norm, post_norm, gate, up, down;
    bool aux = false;
    Tensor aux_q, aux_k, aux_v, aux_norm;
  };
  DrafterConfig cfg_;
  Tensor embed_, fc_, norm_, head_;
  std::vector<Tensor> feature_norms_;
  std::vector<Layer> layers_;
};

}  // namespace woodcock
