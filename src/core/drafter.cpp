#include "woodcock/drafter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "woodcock/json.h"
#include "woodcock/model.h"
#include "woodcock/ops.h"

namespace fs = std::filesystem;

namespace woodcock {

DrafterConfig DrafterConfig::from_json_file(const fs::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  std::stringstream ss;
  ss << f.rdbuf();
  Json j = Json::parse(ss.str());
  DrafterConfig c;
  c.hidden = int(j["hidden_size"].i64());
  c.layers = int(j["num_layers"].i64());
  c.heads = int(j["num_heads"].i64());
  c.kv_heads = int(j["num_kv_heads"].i64());
  c.head_dim = int(j["head_dim"].i64());
  c.intermediate = int(j["intermediate_size"].i64());
  c.vocab = int(j["vocab_size"].i64());
  c.target_hidden = int(j["target_hidden_size"].i64());
  c.num_features = int(j.num_or("num_features", 3));
  c.eps = float(j.num_or("rms_norm_eps", 1e-6));
  c.rope_theta = float(j.num_or("rope_theta", 1e6));
  if (j.str_or("residual", "feature") != "feature") throw std::runtime_error("unsupported drafter mode in " + p.string());
  c.post_norm = j.bool_or("post_norm", true);
  c.feature_norm = j.bool_or("feature_norm", true);
  c.window = int(j.num_or("window", 0));
  // configs from the earlier code base may carry lowrank_rank; only 0 is representable here
  if (j.num_or("lowrank_rank", 0) != 0) throw std::runtime_error("unsupported drafter configuration in " + p.string());
  if (j.has("feature_layers"))
    for (auto& v : j["feature_layers"].arr()) c.feature_layers.push_back(int(v.i64()));
  return c;
}

Drafter::Drafter(const fs::path& dir) {
  cfg_ = DrafterConfig::from_json_file(dir / "config.json");
  Weights w = Weights::open(dir / "model.safetensors");
  embed_ = w.get("embed_tokens.weight");
  fc_ = w.get("fc.weight");
  norm_ = w.get("norm.weight");
  head_ = w.get("lm_head.weight");
  if (cfg_.feature_norm)
    for (int i = 0; i < cfg_.num_features; ++i) feature_norms_.push_back(w.get("feature_norms." + std::to_string(i) + ".weight"));
  for (int i = 0; i < cfg_.layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    Layer L{w.get(p + "self_attn.q_proj.weight"), w.get(p + "self_attn.k_proj.weight"),
            w.get(p + "self_attn.v_proj.weight"), w.get(p + "self_attn.o_proj.weight"),
            w.get(p + "self_attn.q_norm.weight"), w.get(p + "self_attn.k_norm.weight"),
            w.get(p + "input_layernorm.weight"), w.get(p + "post_attention_layernorm.weight"),
            w.get(p + "mlp.gate_proj.weight"), w.get(p + "mlp.up_proj.weight"), w.get(p + "mlp.down_proj.weight"),
            false, {}, {}, {}, {}};
    L.aux = i == 0;
    if (L.aux) {
      L.aux_q = w.get(p + "self_attn.aux_q.weight");
      L.aux_k = w.get(p + "self_attn.aux_k.weight");
      L.aux_v = w.get(p + "self_attn.aux_v.weight");
      L.aux_norm = w.get(p + "aux_norm.weight");
    }
    layers_.push_back(std::move(L));
  }
}

std::vector<float> Drafter::fuse(const std::vector<std::vector<float>>& taps, int L) const {
  const int Dt = cfg_.target_hidden, F = cfg_.num_features;
  std::vector<float> cat(static_cast<size_t>(L) * F * Dt);
  std::vector<float> tmp(static_cast<size_t>(Dt));
  for (int f = 0; f < F; ++f)
    for (int l = 0; l < L; ++l) {
      const float* src = taps[size_t(f)].data() + int64_t(l) * Dt;
      float* dst = cat.data() + (int64_t(l) * F + f) * Dt;
      if (cfg_.feature_norm) ops::rmsnorm(src, dst, 1, Dt, feature_norms_[size_t(f)].data.data(), cfg_.eps);
      else std::copy_n(src, Dt, dst);
    }
  std::vector<float> g(static_cast<size_t>(L) * cfg_.hidden);
  ops::linear(cat.data(), L, F * Dt, fc_, g.data());
  return g;
}

DraftOut Drafter::forward(const std::vector<int>& tokens, const std::vector<float>& feat, KVCache& cache,
                          const std::vector<int>* positions, const std::vector<uint8_t>* mask) const {
  const int L = int(tokens.size()), d = cfg_.hidden, H = cfg_.heads, Hk = cfg_.kv_heads, D = cfg_.head_dim;
  const int prefix = cache.size(), T = prefix + L;
  std::vector<int> pos = positions ? *positions : default_positions(prefix, L);
  std::vector<uint8_t> m = mask ? *mask : causal_mask(prefix, L, cfg_.window);
  std::vector<float> e(static_cast<size_t>(L) * d);
  for (int l = 0; l < L; ++l) std::copy_n(embed_.data.data() + int64_t(tokens[size_t(l)]) * d, d, e.data() + int64_t(l) * d);
  // residual and aux streams
  std::vector<float> h = feat;  // residual stream: the fused target feature
  const std::vector<float>& aux_in = e;  // auxiliary stream: the token embedding
  std::vector<float> x(static_cast<size_t>(L) * d), a(static_cast<size_t>(L) * d), q(static_cast<size_t>(L) * H * D), k(static_cast<size_t>(L) * Hk * D),
      v(static_cast<size_t>(L) * Hk * D), att(static_cast<size_t>(L) * H * D), g(static_cast<size_t>(L) * cfg_.intermediate), u(static_cast<size_t>(L) * cfg_.intermediate);
  for (int li = 0; li < cfg_.layers; ++li) {
    const Layer& Ly = layers_[size_t(li)];
    ops::rmsnorm(h.data(), x.data(), L, d, Ly.in_norm.data.data(), cfg_.eps);
    ops::linear(x.data(), L, d, Ly.q, q.data());
    ops::linear(x.data(), L, d, Ly.k, k.data());
    ops::linear(x.data(), L, d, Ly.v, v.data());
    if (Ly.aux) {
      ops::rmsnorm(aux_in.data(), a.data(), L, d, Ly.aux_norm.data.data(), cfg_.eps);
      ops::linear(a.data(), L, d, Ly.aux_q, q.data(), true);
      ops::linear(a.data(), L, d, Ly.aux_k, k.data(), true);
      ops::linear(a.data(), L, d, Ly.aux_v, v.data(), true);
    }
    ops::rmsnorm(q.data(), q.data(), L * H, D, Ly.q_norm.data.data(), cfg_.eps);
    ops::rmsnorm(k.data(), k.data(), L * Hk, D, Ly.k_norm.data.data(), cfg_.eps);
    ops::rope(q.data(), L, H, D, pos.data(), cfg_.rope_theta);
    ops::rope(k.data(), L, Hk, D, pos.data(), cfg_.rope_theta);
    cache.append(li, k.data(), v.data(), L);
    ops::attention(q.data(), L, H, Hk, D, cache.keys(li), cache.values(li), T, m.data(), att.data(),
                   1.f / std::sqrt(float(D)));
    ops::linear(att.data(), L, H * D, Ly.o, h.data(), true);
    ops::rmsnorm(h.data(), x.data(), L, d, Ly.post_norm.data.data(), cfg_.eps);
    ops::linear(x.data(), L, d, Ly.gate, g.data());
    ops::linear(x.data(), L, d, Ly.up, u.data());
    ops::silu_mul(g.data(), u.data(), int64_t(L) * cfg_.intermediate);
    ops::linear(g.data(), L, cfg_.intermediate, Ly.down, h.data(), true);
  }
  cache.commit(L);
  DraftOut o;
  o.out.resize(static_cast<size_t>(L) * d);
  ops::rmsnorm(h.data(), o.out.data(), L, d, norm_.data.data(), cfg_.eps);
  o.hidden = std::move(h);
  return o;
}

void Drafter::logits(const float* out_row, std::vector<float>& z) const {
  z.resize(static_cast<size_t>(cfg_.vocab));
  ops::linear(out_row, 1, cfg_.hidden, head_, z.data());
}

}  // namespace woodcock
