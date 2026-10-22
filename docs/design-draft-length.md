# Design: chain length chosen from measured latencies

Status: specified 2026-10-22; implemented as `CostAwareChain`
(`python/woodcock/policies.py`), policy `auto:<costs.json>[:K_max]`; measured in
[`journal.md`](journal.md).

Every statement taken from a paper is cited; everything else is marked **[own]**.
Source: Leviathan, Kalman, Matias, "Fast Inference from Transformers via
Speculative Decoding", arXiv:2211.17192 (cited as L23).

## 1. Problem

A chain of γ draft tokens costs γ draft steps plus one verification. A longer
chain yields more tokens per round only while later draft tokens are still
likely to be accepted. The best γ depends on the acceptance rate (which varies by
prompt: code and math are easier than multilingual chat) and on the relative cost
of a draft step — which the truncated draft vocabulary
([`design-vocab-trim.md`](design-vocab-trim.md)) changes. A fixed γ cannot track
either.

## 2. Published model (L23)

* Acceptance rate β: probability that a draft token is accepted; α = E(β), with
  the simplifying assumption that the β's are i.i.d. (L23 §3.1, Def. 3.1).
* Expected tokens produced per round with γ draft tokens:
  E(γ) = (1 − α^{γ+1}) / (1 − α) (L23 §3.1, Eq. 1).
* With c = (time of one draft run) / (time of one target run), the expected
  walltime improvement is (1 − α^{γ+1}) / ((1 − α)(γc + 1)) (L23 §3.3,
  Def. 3.7, Thm. 3.8); the optimal γ maximises it and, being an integer, "can be
  easily found numerically" (L23 §3.5).

## 3. Woodcock specification

**Time model [own].** L23's denominator assumes that verifying γ + 1 tokens costs
the same as one target step and that a round has no other cost. Woodcock measures
instead (`scripts/time_draft_step.py`, batch 1, at a fixed context):

* t_v(n): target forward on n tokens plus argmax, n = 1 … K_max + 1,
* t_s: one fused draft step (LM head + argmax + one drafter layer pass),
* t_h: the LM head + argmax alone,

and models a fused chain round of γ drafts as

  T(γ) = t_v(γ + 1) + γ · t_s + t_u,  with t_u = t_s − t_h,

where t_u is the drafter update after verification (one layer pass on the
accepted tokens, no head). With t_v constant and t_u = 0 this is L23's
(γc + 1) up to the unit of time.

**Choice.** Before each round, γ* = argmax over γ ∈ {1, …, K_max} of E(γ) / T(γ)
with E from L23 Eq. 1 (L23 §3.5) [the time model is own]. Ties: the smaller γ.

**Estimating α [own].** Under L23's i.i.d. assumption, a round with K drafts of
which a are accepted contributes a successes and, if a < K, one failure (the
likelihood is α^a (1 − α)^{[a<K]}). The estimate is

  α̂ = (S + m·α₀) / (S + F + m),

summed over the last W rounds of the current generation, with prior α₀ and prior
weight m; α̂ is clipped to [0.01, 0.99]. Defaults α₀ = 0.6, m = 4, W = 16,
K_max = 8. Every generation starts from the prior.

**Known mismatch.** EAGLE-style drafters accept later chain positions less often
than early ones, so the i.i.d. α̂ measured on long chains is pulled down by the
late positions and the model may over- or under-estimate the value of a longer
chain. Whether this matters is measured against the best fixed γ (§4); a
per-depth model is the next step if it does.

**Correctness.** The policy only changes how many tokens are drafted; greedy
output equals autoregressive decoding for every γ (tested with an adversarial
γ sequence in `tests/test_spec_model.py`). The fused single-sync round
(`SpeculativeDecoder._fused_chain_round`) takes γ per round; the policy reads
only the accepted counts, which the round already returns, so no extra
host synchronisation is added.

**Costs file.** `scripts/time_draft_step.py --out costs.json` writes t_h and t_s
for the full head and each truncated size, and t_v(n). The policy picks the entry
matching the drafter's head ("full" or K) [own].

## 4. Measurement plan

Same benchmark run as the fixed chains: `auto` with the full head and with the
best truncated head, against the fixed chains γ = 2…5. Report tok/s, tokens per
round and the mean chosen γ per domain.
