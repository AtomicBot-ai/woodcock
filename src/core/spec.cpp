#include "woodcock/spec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

#include "woodcock/json.h"

namespace woodcock {

namespace {
double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
bool is_eos(int t, const std::vector<int>& eos) { return std::find(eos.begin(), eos.end(), t) != eos.end(); }
std::vector<int> top_by_phat(const std::vector<DraftNode>& nodes, int k) {
  std::vector<int> o(nodes.size() - 1);
  std::iota(o.begin(), o.end(), 1);
  std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
    if (nodes[size_t(a)].phat != nodes[size_t(b)].phat) return nodes[size_t(a)].phat > nodes[size_t(b)].phat;
    return nodes[size_t(a)].depth < nodes[size_t(b)].depth;
  });
  o.resize(static_cast<size_t>(std::min<int>(k, int(o.size()))));
  std::sort(o.begin(), o.end());
  o.insert(o.begin(), 0);
  return o;
}
}  // namespace

// ------------------------------------------------------------------ policies

StaticPolicy::StaticPolicy(int depth, int width, int budget)
    : depth_(depth), width_(width), budget_(budget >= 0 ? budget : (width == 1 ? depth : 4 * depth)) {}
RoundPlan StaticPolicy::plan(int, bool) { return {depth_, width_, budget_, budget_}; }
std::vector<int> StaticPolicy::select_verify(const std::vector<DraftNode>& nodes, int, bool stochastic) {
  if (stochastic || int(nodes.size()) - 1 <= budget_) {
    std::vector<int> all(nodes.size());
    std::iota(all.begin(), all.end(), 0);
    return all;
  }
  return top_by_phat(nodes, budget_);
}

// ------------------------------------------------------------------ decoder

void SpeculativeDecoder::draft_dist(const float* out_row, const SamplingParams& sp, std::vector<int>& ids,
                                    std::vector<double>& probs) const {
  std::vector<float> z;
  d_.logits(out_row, z);
  ids.resize(z.size());
  std::iota(ids.begin(), ids.end(), 0);
  SamplingParams s2;
  s2.temperature = sp.greedy() ? 1.f : sp.temperature;
  auto p = probs_from_logits(z.data(), int(z.size()), s2);
  probs.assign(p.begin(), p.end());
}

int SpeculativeDecoder::prefill(const std::vector<int>& prompt, const SamplingParams& sp, Rng& rng) {
  tc_ = t_.make_cache();
  dc_ = d_.make_cache();
  const int n = int(prompt.size());
  ForwardArgs a;
  a.logits_from = n - 1;
  ForwardOut o = t_.forward(prompt, tc_, a);
  const int V = t_.config().vocab;
  int first;
  if (sp.greedy()) first = argmax(o.logits.data(), V);
  else {
    auto p = probs_from_logits(o.logits.data(), V, sp);
    first = sample(std::vector<double>(p.begin(), p.end()), rng);
  }
  std::vector<int> toks(prompt.begin() + 1, prompt.end());
  toks.push_back(first);
  auto g = d_.fuse(o.taps, n);
  DraftOut od = d_.forward(toks, g, dc_);
  const int dd = d_.config().hidden;
  root_out_.assign(od.out.end() - dd, od.out.end());
  const float* fb = d_.feedback_row(od, n - 1);
  root_fb_.assign(fb, fb + dd);
  return first;
}

void SpeculativeDecoder::draft(int root_token, int n, const SamplingParams& sp, Rng& rng, DraftTree& tree,
                               std::vector<DraftNode>& kept) {
  const int dd = d_.config().hidden, window = d_.config().window;
  RoundPlan plan = policy_.plan(n, !sp.greedy());
  node_out_.assign(1, root_out_);
  node_fb_.assign(1, root_fb_);
  std::vector<DraftNode> nodes(1);
  nodes[0].token = root_token;
  nodes[0].out_row = 0;
  nodes[0].cache_idx = n - 1;
  const int base = dc_.size();
  int tree_cache = 0;
  std::vector<int> frontier{0};
  for (int depth = 1; depth <= plan.max_depth; ++depth) {
    std::vector<int> level;
    for (int v : frontier) {
      std::vector<int> ids;
      std::vector<double> q;
      draft_dist(node_out_[size_t(nodes[size_t(v)].out_row)].data(), sp, ids, q);
      if (sp.greedy()) {
        std::vector<int> ord(ids.size());
        std::iota(ord.begin(), ord.end(), 0);
        const int w = std::min<int>(plan.width, int(ord.size()));
        std::partial_sort(ord.begin(), ord.begin() + w, ord.end(), [&](int a, int b) { return q[size_t(a)] > q[size_t(b)]; });
        for (int i = 0; i < w; ++i) {
          DraftNode c;
          c.token = ids[size_t(ord[size_t(i)])];
          c.parent = v;
          c.depth = depth;
          c.phat = nodes[size_t(v)].phat * q[size_t(ord[size_t(i)])];
          nodes.push_back(std::move(c));
          level.push_back(int(nodes.size()) - 1);
        }
      } else {
        std::vector<double> qv(static_cast<size_t>(t_.config().vocab), 0.0);
        for (size_t i = 0; i < ids.size(); ++i) qv[size_t(ids[i])] = q[i];
        const int m = plan.children_for();
        for (int x : sample_without_replacement(qv, m, rng)) {
          DraftNode c;
          c.token = x;
          c.parent = v;
          c.depth = depth;
          c.phat = nodes[size_t(v)].phat * qv[size_t(x)];
          nodes.push_back(std::move(c));
          level.push_back(int(nodes.size()) - 1);
        }
        nodes[size_t(v)].q = std::move(qv);
      }
    }
    double gain = 0;
    for (int v : level) gain += nodes[size_t(v)].phat;
    if (level.empty() || depth == plan.max_depth || !policy_.continue_drafting(depth, gain, n)) break;
    // forward the most promising nodes of this level
    std::vector<int> fr = level;
    std::stable_sort(fr.begin(), fr.end(), [&](int a, int b) { return nodes[size_t(a)].phat > nodes[size_t(b)].phat; });
    fr.resize(static_cast<size_t>(std::min<int>(plan.width, int(fr.size()))));
    const int m = int(fr.size());
    std::vector<int> toks, pos;
    std::vector<float> fb(static_cast<size_t>(m) * dd);
    for (int r = 0; r < m; ++r) {
      const DraftNode& nd = nodes[size_t(fr[size_t(r)])];
      toks.push_back(nd.token);
      pos.push_back(n - 1 + nd.depth);
      const auto& pf = node_fb_[size_t(nodes[size_t(nd.parent)].out_row)];
      std::copy(pf.begin(), pf.end(), fb.begin() + int64_t(r) * dd);
    }
    const int T = base + tree_cache + m;
    std::vector<uint8_t> mask(static_cast<size_t>(m) * T, 0);
    for (int r = 0; r < m; ++r) {
      const DraftNode& nd = nodes[size_t(fr[size_t(r)])];
      const int p = base - 1 + nd.depth;
      const int lo = window > 0 ? std::max(0, p - window + 1) : 0;
      for (int t = lo; t < base; ++t) mask[size_t(r) * T + t] = 1;
      for (int u = nd.parent; u > 0; u = nodes[size_t(u)].parent) mask[size_t(r) * T + nodes[size_t(u)].cache_idx] = 1;
      mask[size_t(r) * T + base + tree_cache + r] = 1;
    }
    DraftOut o = d_.forward(toks, fb, dc_, &pos, &mask);
    for (int r = 0; r < m; ++r) {
      DraftNode& nd = nodes[size_t(fr[size_t(r)])];
      nd.out_row = int(node_out_.size());
      nd.cache_idx = base + tree_cache + r;
      node_out_.emplace_back(o.out.begin() + int64_t(r) * dd, o.out.begin() + int64_t(r + 1) * dd);
      const float* f = d_.feedback_row(o, r);
      node_fb_.emplace_back(f, f + dd);
    }
    tree_cache += m;
    frontier = fr;
  }
  dc_.trim_to(base);  // drafting phase entries are discarded (append-only within the phase)
  std::vector<int> keep = policy_.select_verify(nodes, n, !sp.greedy());
  std::vector<int> remap(nodes.size(), -1);
  for (size_t i = 0; i < keep.size(); ++i) remap[size_t(keep[i])] = int(i);
  tree = {};
  kept.clear();
  for (int v : keep) {
    tree.tokens.push_back(nodes[size_t(v)].token);
    tree.parents.push_back(nodes[size_t(v)].parent >= 0 ? remap[size_t(nodes[size_t(v)].parent)] : -1);
    tree.depth.push_back(nodes[size_t(v)].depth);
    kept.push_back(nodes[size_t(v)]);
  }
  tree.build_children();
}

Verdict SpeculativeDecoder::verify(const DraftTree& tree, const std::vector<DraftNode>& kept, int n,
                                   const SamplingParams& sp, Rng& rng, std::vector<std::vector<float>>& taps_out) {
  const int N = tree.size(), V = t_.config().vocab, Dt = t_.config().hidden;
  ForwardOut o;
  if (tree.is_chain()) {
    o = t_.forward(tree.tokens, tc_);
  } else {
    std::vector<int> pos;
    for (int dpt : tree.depth) pos.push_back(n + dpt);
    auto mask = tree_mask(tree, n);
    ForwardArgs a;
    a.positions = &pos;
    a.mask = &mask;
    o = t_.forward(tree.tokens, tc_, a);
  }
  Verdict r;
  if (sp.greedy()) {
    std::vector<int> am(static_cast<size_t>(N));
    for (int v = 0; v < N; ++v) am[size_t(v)] = argmax(o.logits.data() + int64_t(v) * V, V);
    r = verify_tree_greedy(am, tree);
  } else {
    auto p = [&](int v) {
      auto f = probs_from_logits(o.logits.data() + int64_t(v) * V, V, sp);
      return std::vector<double>(f.begin(), f.end());
    };
    auto q = [&](int v) -> const std::vector<double>& { return kept[size_t(v)].q; };
    r = verify_tree_stochastic(p, q, tree, rng);
  }
  std::vector<int> keep{0};
  keep.insert(keep.end(), r.path.begin(), r.path.end());
  tc_.compact(n, keep);
  taps_out.assign(o.taps.size(), {});
  for (size_t f = 0; f < o.taps.size(); ++f)
    for (int v : keep) taps_out[f].insert(taps_out[f].end(), o.taps[f].begin() + int64_t(v) * Dt, o.taps[f].begin() + int64_t(v + 1) * Dt);
  return r;
}

void SpeculativeDecoder::update(const std::vector<int>& acc, const std::vector<std::vector<float>>& taps) {
  const int L = int(acc.size()), dd = d_.config().hidden;
  auto g = d_.fuse(taps, L);
  DraftOut o = d_.forward(acc, g, dc_);
  root_out_.assign(o.out.end() - dd, o.out.end());
  const float* fb = d_.feedback_row(o, L - 1);
  root_fb_.assign(fb, fb + dd);
}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt, int max_new, const SamplingParams& sp,
                                              GenStats* st, const std::vector<int>& eos) {
  GenStats s;
  s.prompt_tokens = int(prompt.size());
  Rng rng(sp.seed);
  const double t0 = now();
  int x = prefill(prompt, sp, rng);
  s.t_prefill = now() - t0;
  std::vector<int> out{x};
  int n = int(prompt.size());
  policy_.reset();
  while (int(out.size()) < max_new && !is_eos(x, eos)) {
    const double ta = now();
    DraftTree tree;
    std::vector<DraftNode> kept;
    draft(x, n, sp, rng, tree, kept);
    const double tb = now();
    std::vector<std::vector<float>> taps;
    Verdict r = verify(tree, kept, n, sp, rng, taps);
    const double tc = now();
    std::vector<int> acc;
    for (int v : r.path) acc.push_back(tree.tokens[size_t(v)]);
    acc.push_back(r.next);
    update(acc, taps);
    const double td = now();
    s.t_draft += tb - ta;
    s.t_verify += tc - tb;
    s.t_update += td - tc;
    s.rounds++;
    s.accepted += long(r.path.size());
    s.verified += tree.size() - 1;
    int maxd = 0;
    for (int dpt : tree.depth) maxd = std::max(maxd, dpt);
    policy_.observe(int(r.path.size()), tree.size() - 1, maxd, tb - ta, td - tb);
    for (int tok : acc) {
      out.push_back(tok);
      if (is_eos(tok, eos) || int(out.size()) >= max_new) break;
    }
    n += int(acc.size());
    x = out.back();
  }
  s.t_total = now() - t0;
  s.new_tokens = int(out.size());
  if (st) *st = s;
  return out;
}

std::vector<int> generate_autoregressive(const Qwen3& t, const std::vector<int>& prompt, int max_new,
                                         const SamplingParams& sp, GenStats* st, const std::vector<int>& eos) {
  GenStats s;
  s.prompt_tokens = int(prompt.size());
  Rng rng(sp.seed);
  const double t0 = now();
  KVCache c = t.make_cache();
  ForwardArgs a;
  a.logits_from = int(prompt.size()) - 1;
  const int V = t.config().vocab;
  auto pick = [&](const float* z) {
    if (sp.greedy()) return argmax(z, V);
    auto p = probs_from_logits(z, V, sp);
    return sample(std::vector<double>(p.begin(), p.end()), rng);
  };
  ForwardOut o = t.forward(prompt, c, a);
  int x = pick(o.logits.data());
  s.t_prefill = now() - t0;
  std::vector<int> out{x};
  while (int(out.size()) < max_new && !is_eos(x, eos)) {
    o = t.forward({x}, c);
    x = pick(o.logits.data());
    out.push_back(x);
  }
  s.t_total = now() - t0;
  s.new_tokens = int(out.size());
  s.rounds = s.new_tokens - 1;
  if (st) *st = s;
  return out;
}

}  // namespace woodcock
