#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "woodcock/json.h"
#include "woodcock/model.h"
#include "woodcock/ops.h"

namespace fs = std::filesystem;

namespace woodcock {

static std::string read_text(const fs::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

Qwen3Config Qwen3Config::from_json_file(const fs::path& p) {
  Json j = Json::parse(read_text(p));
  Qwen3Config c;
  c.hidden = int(j["hidden_size"].i64());
  c.layers = int(j["num_hidden_layers"].i64());
  c.heads = int(j["num_attention_heads"].i64());
  c.kv_heads = int(j["num_key_value_heads"].i64());
  c.head_dim = int(j.num_or("head_dim", c.hidden / c.heads));
  c.intermediate = int(j["intermediate_size"].i64());
  c.vocab = int(j["vocab_size"].i64());
  c.eps = float(j.num_or("rms_norm_eps", 1e-6));
  c.rope_theta = float(j.num_or("rope_theta", 1e6));
  c.tie_embeddings = j.bool_or("tie_word_embeddings", false);
  return c;
}

std::vector<int> default_positions(int start, int L) {
  std::vector<int> p(static_cast<size_t>(L));
  for (int i = 0; i < L; ++i) p[size_t(i)] = start + i;
  return p;
}

std::vector<uint8_t> causal_mask(int prefix, int L, int window) {
  const int T = prefix + L;
  std::vector<uint8_t> m(static_cast<size_t>(L) * T, 0);
  for (int l = 0; l < L; ++l) {
    const int q = prefix + l;
    const int lo = window > 0 ? std::max(0, q - window + 1) : 0;
    for (int t = lo; t <= q; ++t) m[size_t(l) * T + t] = 1;
  }
  return m;
}

Qwen3::Qwen3(const fs::path& dir, std::array<int, 3> taps) {
  cfg_ = Qwen3Config::from_json_file(dir / "config.json");
  if (taps[0] < 0) taps = {2, cfg_.layers / 2, cfg_.layers - 3};  // python default_feature_layers
  taps_ = taps;
  Weights w = Weights::open(dir);
  embed_ = w.get("model.embed_tokens.weight");
  norm_ = w.get("model.norm.weight");
  tied_ = cfg_.tie_embeddings || !w.has("lm_head.weight");
  if (!tied_) head_ = w.get("lm_head.weight");
  for (int i = 0; i < cfg_.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    layers_.push_back({w.get(p + "self_attn.q_proj.weight"), w.get(p + "self_attn.k_proj.weight"),
                       w.get(p + "self_attn.v_proj.weight"), w.get(p + "self_attn.o_proj.weight"),
                       w.get(p + "self_attn.q_norm.weight"), w.get(p + "self_attn.k_norm.weight"),
                       w.get(p + "input_layernorm.weight"), w.get(p + "post_attention_layernorm.weight"),
                       w.get(p + "mlp.gate_proj.weight"), w.get(p + "mlp.up_proj.weight"),
                       w.get(p + "mlp.down_proj.weight")});
  }
}

ForwardOut Qwen3::forward(const std::vector<int>& tokens, KVCache& cache, const ForwardArgs& a) const {
  const int L = int(tokens.size()), d = cfg_.hidden, H = cfg_.heads, Hk = cfg_.kv_heads, D = cfg_.head_dim;
  const int prefix = cache.size(), T = prefix + L;
  std::vector<int> pos = a.positions ? *a.positions : default_positions(prefix, L);
  std::vector<uint8_t> mask = a.mask ? *a.mask : causal_mask(prefix, L);
  std::vector<float> h(static_cast<size_t>(L) * d), x(static_cast<size_t>(L) * d), q(static_cast<size_t>(L) * H * D), k(static_cast<size_t>(L) * Hk * D),
      v(static_cast<size_t>(L) * Hk * D), att(static_cast<size_t>(L) * H * D), g(static_cast<size_t>(L) * cfg_.intermediate), u(static_cast<size_t>(L) * cfg_.intermediate);
  for (int l = 0; l < L; ++l) std::copy_n(embed_.data.data() + int64_t(tokens[size_t(l)]) * d, d, h.data() + int64_t(l) * d);
  ForwardOut out;
  for (int li = 0; li < cfg_.layers; ++li) {
    const Layer& Ly = layers_[size_t(li)];
    ops::rmsnorm(h.data(), x.data(), L, d, Ly.in_norm.data.data(), cfg_.eps);
    ops::linear(x.data(), L, d, Ly.q, q.data());
    ops::linear(x.data(), L, d, Ly.k, k.data());
    ops::linear(x.data(), L, d, Ly.v, v.data());
    ops::rmsnorm(q.data(), q.data(), L * H, D, Ly.q_norm.data.data(), cfg_.eps);
    ops::rmsnorm(k.data(), k.data(), L * Hk, D, Ly.k_norm.data.data(), cfg_.eps);
    ops::rope(q.data(), L, H, D, pos.data(), cfg_.rope_theta);
    ops::rope(k.data(), L, Hk, D, pos.data(), cfg_.rope_theta);
    cache.append(li, k.data(), v.data(), L);
    ops::attention(q.data(), L, H, Hk, D, cache.keys(li), cache.values(li), T, mask.data(), att.data(),
                   1.f / std::sqrt(float(D)));
    ops::linear(att.data(), L, H * D, Ly.o, h.data(), /*accumulate=*/true);
    ops::rmsnorm(h.data(), x.data(), L, d, Ly.post_norm.data.data(), cfg_.eps);
    ops::linear(x.data(), L, d, Ly.gate, g.data());
    ops::linear(x.data(), L, d, Ly.up, u.data());
    ops::silu_mul(g.data(), u.data(), int64_t(L) * cfg_.intermediate);
    ops::linear(g.data(), L, cfg_.intermediate, Ly.down, h.data(), /*accumulate=*/true);
    for (int t : taps_)
      if (t == li) out.taps.push_back(h);
  }
  cache.commit(L);
  if (a.logits) {
    const int rows = L - a.logits_from;
    ops::rmsnorm(h.data() + int64_t(a.logits_from) * d, x.data(), rows, d, norm_.data.data(), cfg_.eps);
    out.logits.resize(static_cast<size_t>(rows) * cfg_.vocab);
    ops::linear(x.data(), rows, d, tied_ ? embed_ : head_, out.logits.data());
  }
  return out;
}

}  // namespace woodcock
