// Speculative scheduler, draft-tree manager and draft policies
// (mirrors python/woodcock/spec.py and policies.py).
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "woodcock/drafter.h"
#include "woodcock/model.h"
#include "woodcock/sampling.h"

namespace woodcock {

struct DraftNode {
  int token = 0, parent = -1, depth = 0;
  double phat = 1.0;
  int out_row = -1;                  // row in the node-output store once forwarded
  int cache_idx = -1;
  std::vector<double> q;             // stochastic mode: distribution children were drawn from
};

struct RoundPlan {
  int max_depth = 1, width = 1, budget = 1, remaining = 1;
  int children_for() { int m = std::min(width, remaining); remaining -= m; return std::max(m, 0); }
};

class Policy {
 public:
  virtual ~Policy() = default;
  virtual void reset() {}
  virtual RoundPlan plan(int ctx, bool stochastic) = 0;
  virtual bool continue_drafting(int depth, double gain, int ctx) { return true; }
  virtual std::vector<int> select_verify(const std::vector<DraftNode>& nodes, int ctx, bool stochastic) = 0;
  virtual void observe(int accepted, int verified, int depth, double t_draft, double t_verify) {}
};

class StaticPolicy : public Policy {
 public:
  StaticPolicy(int depth, int width = 1, int budget = -1);
  RoundPlan plan(int, bool) override;
  std::vector<int> select_verify(const std::vector<DraftNode>& nodes, int, bool stochastic) override;
 private:
  int depth_, width_, budget_;
};

struct GenStats {
  int prompt_tokens = 0, new_tokens = 0, rounds = 0;
  long accepted = 0, verified = 0;
  double t_prefill = 0, t_draft = 0, t_verify = 0, t_update = 0, t_total = 0;
  double tokens_per_round() const { return rounds ? 1.0 + double(accepted) / rounds : 0.0; }
  double decode_tps() const { return t_total > t_prefill ? (new_tokens - 1) / (t_total - t_prefill) : 0.0; }
};

class SpeculativeDecoder {
 public:
  SpeculativeDecoder(const Qwen3& target, const Drafter& drafter, Policy& policy)
      : t_(target), d_(drafter), policy_(policy) {}
  std::vector<int> generate(const std::vector<int>& prompt, int max_new, const SamplingParams& sp, GenStats* st,
                            const std::vector<int>& eos);

 private:
  int prefill(const std::vector<int>& prompt, const SamplingParams& sp, Rng& rng);
  void draft(int root_token, int n, const SamplingParams& sp, Rng& rng, DraftTree& tree, std::vector<DraftNode>& kept);
  Verdict verify(const DraftTree& tree, const std::vector<DraftNode>& kept, int n, const SamplingParams& sp, Rng& rng,
                 std::vector<std::vector<float>>& taps_out);
  void update(const std::vector<int>& acc, const std::vector<std::vector<float>>& taps);
  // draft distribution of one node: candidate ids and probabilities
  void draft_dist(const float* out_row, const SamplingParams& sp, std::vector<int>& ids, std::vector<double>& probs) const;

  const Qwen3& t_;
  const Drafter& d_;
  Policy& policy_;
  KVCache tc_, dc_;
  std::vector<float> root_out_, root_fb_;
  std::vector<std::vector<float>> node_out_, node_fb_;
};

std::vector<int> generate_autoregressive(const Qwen3& t, const std::vector<int>& prompt, int max_new,
                                         const SamplingParams& sp, GenStats* st, const std::vector<int>& eos);

}  // namespace woodcock
