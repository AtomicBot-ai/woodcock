#include "woodcock/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace woodcock {

std::vector<float> probs_from_logits(const float* z, int n, const SamplingParams& sp) {
  std::vector<float> p(static_cast<size_t>(n));
  const float t = std::max(sp.temperature, 1e-6f);
  float kth = -INFINITY;
  if (sp.top_k > 0 && sp.top_k < n) {
    std::vector<float> s(z, z + n);
    std::nth_element(s.begin(), s.begin() + (sp.top_k - 1), s.end(), std::greater<float>());
    kth = s[size_t(sp.top_k - 1)];
  }
  float m = -INFINITY;
  for (int i = 0; i < n; ++i)
    if (z[i] >= kth) m = std::max(m, z[i] / t);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    p[size_t(i)] = z[i] >= kth ? std::exp(z[i] / t - m) : 0.f;
    s += p[size_t(i)];
  }
  for (auto& x : p) x = float(x / s);
  if (sp.top_p < 1.f) {
    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return p[size_t(a)] > p[size_t(b)]; });
    double c = 0.0, kept = 0.0;
    std::vector<uint8_t> keep(static_cast<size_t>(n), 0);
    for (int i : order) {
      if (c >= sp.top_p) break;  // always keeps the first
      keep[size_t(i)] = 1;
      c += p[size_t(i)];
    }
    for (int i = 0; i < n; ++i) {
      if (!keep[size_t(i)]) p[size_t(i)] = 0.f;
      kept += p[size_t(i)];
    }
    for (auto& x : p) x = float(x / kept);
  }
  return p;
}

int argmax(const float* z, int n) { return int(std::max_element(z, z + n) - z); }

int sample(const std::vector<double>& p, Rng& rng) {
  double tot = 0.0;
  for (double x : p) tot += x;
  double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * tot, c = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    c += p[i];
    if (u < c) return int(i);
  }
  return int(p.size() - 1);
}

std::vector<int> sample_without_replacement(const std::vector<double>& q, int k, Rng& rng) {
  // Gumbel-top-k: equivalent to k sequential draws without replacement
  std::vector<std::pair<double, int>> keys;
  std::uniform_real_distribution<double> U(1e-300, 1.0);
  for (size_t i = 0; i < q.size(); ++i)
    if (q[i] > 0) keys.push_back({std::log(q[i]) - std::log(-std::log(U(rng))), int(i)});
  k = std::min<int>(k, int(keys.size()));
  std::partial_sort(keys.begin(), keys.begin() + k, keys.end(), std::greater<>());
  std::vector<int> out;
  for (int i = 0; i < k; ++i) out.push_back(keys[size_t(i)].second);
  return out;
}

void DraftTree::build_children() {
  children.assign(tokens.size(), {});
  for (size_t v = 0; v < parents.size(); ++v)
    if (parents[v] >= 0) children[size_t(parents[v])].push_back(int(v));
}

bool DraftTree::is_chain() const {
  for (size_t i = 1; i < parents.size(); ++i)
    if (parents[i] != int(i) - 1) return false;
  return true;
}

std::vector<uint8_t> tree_mask(const DraftTree& t, int prefix) {
  const int N = t.size(), T = prefix + N;
  std::vector<uint8_t> m(static_cast<size_t>(N) * T, 0);
  for (int v = 0; v < N; ++v) {
    std::fill_n(m.begin() + int64_t(v) * T, prefix, 1);
    for (int u = v; u >= 0; u = t.parents[size_t(u)]) m[size_t(v) * T + prefix + u] = 1;
  }
  return m;
}

Verdict verify_tree_greedy(const std::vector<int>& am, const DraftTree& t) {
  Verdict r;
  int v = 0;
  while (true) {
    int want = am[size_t(v)], nxt = -1;
    for (int c : t.children[size_t(v)])
      if (t.tokens[size_t(c)] == want) { nxt = c; break; }
    if (nxt < 0) { r.next = want; return r; }
    r.path.push_back(nxt);
    v = nxt;
  }
}

static void residual(std::vector<double>& p, const std::vector<double>& q) {
  double s = 0.0;
  std::vector<double> r(p.size());
  for (size_t i = 0; i < p.size(); ++i) { r[i] = std::max(p[i] - q[i], 0.0); s += r[i]; }
  if (s <= 1e-12) return;  // p == q up to rounding: p is exact
  for (size_t i = 0; i < p.size(); ++i) p[i] = r[i] / s;
}

Verdict verify_tree_stochastic(const std::function<std::vector<double>(int)>& p,
                               const std::function<const std::vector<double>&(int)>& q, const DraftTree& t, Rng& rng) {
  Verdict r;
  std::uniform_real_distribution<double> U(0.0, 1.0);
  int v = 0;
  while (true) {
    std::vector<double> pc = p(v);
    const auto& kids = t.children[size_t(v)];
    if (kids.empty()) { r.next = sample(pc, rng); return r; }
    std::vector<double> qc = q(v);
    int acc = -1;
    for (int c : kids) {
      const int x = t.tokens[size_t(c)];
      if (qc[size_t(x)] > 0 && U(rng) < std::min(1.0, pc[size_t(x)] / qc[size_t(x)])) { acc = c; break; }
      residual(pc, qc);
      qc[size_t(x)] = 0.0;
      double s = 0.0;
      for (double y : qc) s += y;
      if (s <= 0) break;
      for (double& y : qc) y /= s;
    }
    if (acc < 0) { r.next = sample(pc, rng); return r; }
    r.path.push_back(acc);
    v = acc;
  }
}

}  // namespace woodcock
