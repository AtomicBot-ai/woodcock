#include <algorithm>
#include <cstring>

#include "woodcock/cache.h"

namespace woodcock {

void KVCache::append(int layer, const float* k, const float* v, int L) {
  auto& K = k_[size_t(layer)];
  auto& V = v_[size_t(layer)];
  const size_t need = size_t(n_ + L) * size_t(row_);
  if (K.size() < need) {
    const size_t cap = std::max(need, K.size() * 2 + size_t(256) * row_);
    K.resize(cap);
    V.resize(cap);
  }
  std::memcpy(K.data() + size_t(n_) * row_, k, sizeof(float) * size_t(L) * row_);
  std::memcpy(V.data() + size_t(n_) * row_, v, sizeof(float) * size_t(L) * row_);
}

void KVCache::compact(int start, const std::vector<int>& keep) {
  for (int l = 0; l < layers_; ++l) {
    for (size_t i = 0; i < keep.size(); ++i) {
      const int src = start + keep[i], dst = start + int(i);
      if (src == dst) continue;
      std::memmove(k_[size_t(l)].data() + size_t(dst) * row_, k_[size_t(l)].data() + size_t(src) * row_, sizeof(float) * row_);
      std::memmove(v_[size_t(l)].data() + size_t(dst) * row_, v_[size_t(l)].data() + size_t(src) * row_, sizeof(float) * row_);
    }
  }
  n_ = start + int(keep.size());
}

}  // namespace woodcock
