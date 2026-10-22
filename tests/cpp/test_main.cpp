// Native runtime tests. Unit tests always run; the real-model test runs when
// WOODCOCK_REF points at a reference produced by scripts/make_cpp_reference.py.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

#include "woodcock/json.h"
#include "woodcock/spec.h"

using namespace woodcock;

static int failures = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) {                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

static void test_json() {
  Json j = Json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": "xé\n"}, "d": null, "e": true})");
  CHECK(j["a"][2].num() == -300);
  CHECK(j["b"]["c"].str() == "x\xc3\xa9\n");
  CHECK(j["d"].is_null() && j["e"].boolean());
  CHECK(j.num_or("d", 7) == 7);
}

static void test_greedy_verify() {
  DraftTree t{{0, 1, 2, 4}, {-1, 0, 0, 2}, {0, 1, 1, 2}, {}};
  t.build_children();
  auto r = verify_tree_greedy({2, 9, 4, 3}, t);
  CHECK((r.path == std::vector<int>{2, 3}) && r.next == 3);
  auto m = tree_mask(t, 2);
  CHECK(m.size() == 4 * 6);
  CHECK(m[3 * 6 + 2] && !m[3 * 6 + 3] && m[3 * 6 + 4] && m[3 * 6 + 5]);
}

static void test_kv_compact() {
  KVCache c(1, 1, 2);
  std::vector<float> k{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  c.append(0, k.data(), k.data(), 5);
  c.commit(5);
  c.compact(1, {0, 2, 3});  // keep rows 1, 3, 4
  CHECK(c.size() == 4);
  CHECK(c.keys(0)[2] == 1 && c.keys(0)[4] == 3 && c.keys(0)[6] == 4);
}

// Stochastic tree verification on a Markov model: the first emitted token must follow P[root].
static void test_tree_stochastic_marginal() {
  const int V = 4;
  const double P[V][V] = {{.5, .2, .2, .1}, {.1, .6, .2, .1}, {.25, .25, .25, .25}, {.4, .1, .1, .4}};
  const double Q[V][V] = {{.2, .5, .2, .1}, {.3, .3, .2, .2}, {.1, .2, .3, .4}, {.25, .25, .25, .25}};
  Rng rng(1);
  std::vector<double> cnt(V, 0);
  const int trials = 200000;
  for (int it = 0; it < trials; ++it) {
    DraftTree t;
    t.tokens = {0};
    t.parents = {-1};
    t.depth = {0};
    std::vector<std::vector<double>> qd(1, std::vector<double>(Q[0], Q[0] + V));
    for (int x : sample_without_replacement(qd[0], 2, rng)) {
      t.tokens.push_back(x);
      t.parents.push_back(0);
      t.depth.push_back(1);
      qd.emplace_back(Q[x], Q[x] + V);
    }
    t.build_children();
    auto p = [&](int v) { return std::vector<double>(P[t.tokens[size_t(v)]], P[t.tokens[size_t(v)]] + V); };
    auto q = [&](int v) -> const std::vector<double>& { return qd[size_t(v)]; };
    Verdict r = verify_tree_stochastic(p, q, t, rng);
    const int first = r.path.empty() ? r.next : t.tokens[size_t(r.path[0])];
    cnt[size_t(first)] += 1;
  }
  double chi2 = 0;
  for (int i = 0; i < V; ++i) {
    const double e = P[0][i] * trials;
    chi2 += (cnt[size_t(i)] - e) * (cnt[size_t(i)] - e) / e;
  }
  // 3 degrees of freedom: chi2 > 16.3 has p < 0.001
  CHECK(chi2 < 16.3);
}

static std::string slurp(const std::string& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::vector<int> ints(const Json& a) {
  std::vector<int> v;
  for (auto& x : a.arr()) v.push_back(int(x.i64()));
  return v;
}

static void test_real_model(const std::string& ref_dir) {
  Json ref = Json::parse(slurp(ref_dir + "/reference.json"));
  Qwen3 t(ref["target_dir"].str());
  Drafter d(ref["drafter_dir"].str());
  auto prompt = ints(ref["prompt"]);
  // 1) last-position logits vs MLX (fp32)
  KVCache c = t.make_cache();
  ForwardArgs a;
  a.logits_from = int(prompt.size()) - 1;
  ForwardOut o = t.forward(prompt, c, a);
  auto ml = ref["last_logits"].arr();
  double maxd = 0;
  for (size_t i = 0; i < ml.size(); ++i) maxd = std::max(maxd, std::fabs(double(o.logits[i]) - ml[i].num()));
  std::printf("  logits max |C++ - MLX| over %zu ids: %.2e\n", ml.size(), maxd);
  CHECK(maxd < 2e-2);
  const std::vector<int> eos{151645, 151643};
  const int n = int(ref["n_new"].i64());
  SamplingParams sp;
  // 2) autoregressive greedy == MLX autoregressive greedy
  GenStats st;
  auto ar = generate_autoregressive(t, prompt, n, sp, &st, eos);
  auto ar_ref = ints(ref["ar_greedy"]);
  size_t same = 0;
  while (same < ar.size() && same < ar_ref.size() && ar[same] == ar_ref[same]) ++same;
  std::printf("  AR greedy: %zu/%zu tokens equal to MLX reference, %.1f tok/s\n", same, ar_ref.size(), st.decode_tps());
  CHECK(same == ar_ref.size());
  // 3) speculative greedy == C++ autoregressive greedy, chain and tree
  StaticPolicy chain(4), tree(3, 3, 8);
  for (Policy* p : {static_cast<Policy*>(&chain), static_cast<Policy*>(&tree)}) {
    SpeculativeDecoder dec(t, d, *p);
    GenStats s2;
    auto sd = dec.generate(prompt, n, sp, &s2, eos);
    std::printf("  spec greedy: equal=%d tokens/round %.2f, %.1f tok/s\n", int(sd == ar), s2.tokens_per_round(), s2.decode_tps());
    CHECK(sd == ar);
  }
}

int main() {
  test_json();
  test_greedy_verify();
  test_kv_compact();
  test_tree_stochastic_marginal();
  if (const char* ref = std::getenv("WOODCOCK_REF")) {
    std::printf("real-model test with %s\n", ref);
    test_real_model(ref);
  } else {
    std::printf("WOODCOCK_REF not set: real-model test skipped\n");
  }
  std::printf(failures ? "FAILED (%d)\n" : "all tests passed\n", failures);
  return failures ? 1 : 0;
}
