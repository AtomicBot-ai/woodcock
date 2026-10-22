// Per-layer KV cache: [position][kv_head][head_dim], append-only within a
// drafting phase, rolled back by trim, compacted after tree verification.
#pragma once

#include <vector>

namespace woodcock {

class KVCache {
 public:
  KVCache() = default;
  KVCache(int layers, int kv_heads, int head_dim) : layers_(layers), row_(kv_heads * head_dim), k_(layers), v_(layers) {}

  int size() const { return n_; }
  int row() const { return row_; }
  // copy L rows of K and V for `layer` at positions [size, size+L); call commit(L) after all layers
  void append(int layer, const float* k, const float* v, int L);
  void commit(int L) { n_ += L; }
  void trim_to(int n) { n_ = n; }
  // keep rows start + keep[i] of the last block, contiguously at start, start+1, ...
  void compact(int start, const std::vector<int>& keep);
  const float* keys(int layer) const { return k_[size_t(layer)].data(); }
  const float* values(int layer) const { return v_[size_t(layer)].data(); }

 private:
  int layers_ = 0, row_ = 0, n_ = 0;
  std::vector<std::vector<float>> k_, v_;
};

}  // namespace woodcock
