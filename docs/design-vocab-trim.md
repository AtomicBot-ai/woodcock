# Design: frequency-truncated draft vocabulary

Status: specified 2026-10-22, implemented in `python/woodcock/vocab.py`,
`Drafter.restrict_vocab`, `SpeculativeDecoder`; measured in
[`journal.md`](journal.md).

Every statement taken from a paper is cited; everything else is marked
**[own]** (Woodcock's own choice). Sources: FR-Spec = arXiv:2502.14856,
VocabTrim = arXiv:2506.22694 (CC BY 4.0), SlimSpec = arXiv:2605.10453 (CC BY 4.0),
Leviathan et al. = arXiv:2211.17192.

## 1. Problem

The Woodcock drafter for Qwen3-1.7B has one decoder layer (~71M parameters:
attention with two input streams, a 2048→6144 SwiGLU FFN and the 6144→2048 fusion
FC) and an LM head with one row per vocabulary entry: 151,936 × 2048 ≈ 311M
parameters, about four times the rest of the drafter. At batch size 1 the drafter
is memory-bandwidth bound, so the time of a draft step is roughly proportional
to the bytes of weights it reads; the head is then ~80% of a draft step [own
estimate, to be checked by measurement]. FR-Spec makes the same observation for
EAGLE-2 on Llama-3-8B: the LM head is "49% of the total computational time in
the drafting process", 62% with the softmax (FR-Spec §3.1).

A drafter needs the full-vocabulary logits only to pick the most likely tokens
(greedy) or to sample from its distribution q. Tokens the target almost never
produces cost the same as frequent ones.

## 2. Method (published)

1. **Rank tokens by frequency** in a calibration corpus and keep the top K as
   the draft vocabulary V_K (FR-Spec §3.2; VocabTrim §3, Eq. 1: W^Trim =
   W[Top-K(c, k), :]).
2. **Slice the head**: the draft head is the K rows of the drafter's LM head for
   the ids in V_K, W̃[i, :] = W[V_K[i], :], and the draft distribution is
   softmax(h W̃ᵀ) over V_K (FR-Spec §3.2, Eqs. 3–4). Embedding, decoder layer
   and target are unchanged; no retraining (FR-Spec §3.2; VocabTrim: "training-free").
3. **Map draft indices back** to target token ids before verification
   (VocabTrim App. C.1, Algorithm 2: an index mapper T_{d→t}).
4. **Verification is unchanged** and uses the target's full vocabulary, so the
   output is still the target's: "We only limit the vocabulary of the drafting
   process while keeping the verification process untouched" (FR-Spec §3.2,
   §6); under greedy decoding "a token is only accepted if exactly matched with
   the target model" (VocabTrim §4.1).

**Calibration data.** VocabTrim compares raw text, target-generated and
drafter-generated calibration text and finds target-generated text best
(VocabTrim §3, §4.2). FR-Spec uses a general corpus (SlimPajama, 1B tokens) and
also reports ShareGPT; the two give similar acceptance (FR-Spec §4.2, Table 1).

**Size.** FR-Spec evaluates K = 8k, 16k, 32k, 64k on Llama-3-8B (128k vocabulary):
accepted length 3.13 / 3.40 / 3.63 / 3.80 against 3.89 for the full head, and the
best speed at K = 32k (2.27× vs 2.03× for EAGLE-2; FR-Spec §4.3, Table 2).
VocabTrim finds the best trade-off at ~23k–35k tokens for Llama-3 (VocabTrim §4.3.1).

**Known limitation.** Under greedy decoding a draft token outside V_K can never be
proposed, so whenever the target's next token is outside V_K that draft position
is rejected ("acceptance collapses to 0 whenever v* ∉ V_tr", SlimSpec §4.2, footnote 3).
The loss is bounded by the share of target tokens outside V_K, which we measure
(§4).

## 3. Woodcock specification

**Calibration counts [VocabTrim §3, target-generated].** Count the token ids of
the target-regenerated responses (`data/regen/<target>.jsonl`, written by the
target itself with `woodcock regen`), using the *training* split of the drafter
(the same seeded shuffle and hold-out as `SFTStream`, so the 200 hold-out
records are never counted and can measure coverage) [own]. Prompts are not
counted: the drafter only drafts response tokens [own]. Output:
`runs/vocab/<name>.npz` with the count vector (length = target vocabulary), the
data source, and a coverage table on the hold-out records.

**Selection.** V_K = the K ids with the largest counts; ties broken by the
smaller id (stable sort) [own]. The target's EOS ids are always included,
replacing the least frequent selected ids if necessary [own]: the regenerated
responses do not store the stop token (`batch_generate` drops it, see
`data.regenerate`), so EOS would otherwise have count 0 and the drafter could
never propose the end of an answer. V_K is stored sorted ascending [own], so the
row gather reads the head in order and the selection is deterministic.

**Head.** `Drafter.restrict_vocab(ids)` replaces `lm_head` by a K-row linear
layer holding `W[ids]` and keeps `vocab_ids` (int32, length K) [FR-Spec Eq. 3].
The full head is not kept in memory. A restricted drafter cannot be saved or
exported (its config would describe a full head) [own]; restriction happens at
load time, from the counts file and K.

**Drafting.** `logits(out)` returns the K restricted logits. Greedy: the drafted
token is `vocab_ids[argmax]` (fused chain) or `vocab_ids[top-w]` (trees)
[VocabTrim App. C.1]. Sampling (T > 0): q is the softmax of the restricted logits
at the drafting temperature, scattered into a full-vocabulary vector that is zero
outside V_K [own wiring of FR-Spec Eq. 4].

**Correctness.**
* Greedy: the accepted tokens are exactly those equal to the target's argmax,
  whatever the drafter proposes, so the output equals autoregressive greedy
  decoding [VocabTrim §4.1]. Tested in fp32 on Qwen3-0.6B
  (`tests/test_spec_model.py`).
* T > 0: speculative sampling returns a sample from p for *any* draft
  distribution q from which the draft token was actually drawn (Leviathan et al.
  §2.3, proof in App. A.1); nothing in the proof needs q > 0 everywhere. A token
  outside V_K is never drawn, and at a rejection the residual norm(max(0, p − q))
  equals p on all ids outside V_K, which is how those tokens remain reachable.
  For trees, the same holds for recursive rejection sampling (arXiv:2402.14160).
  Tested with a chi-square test where q is zero on part of the vocabulary
  (`tests/test_sampling.py`).

**CLI.** `woodcock vocab --config configs/<target>.yaml` writes the counts;
`woodcock bench/generate --drafter name=path@runs/vocab/<name>.npz:K` loads a
drafter with a K-token head [own].

## 4. Expected effect

Hold-out coverage of the response tokens by V_K (Qwen3-1.7B, 3.35M counted
tokens, 200 hold-out records = 76k tokens; measured 2026-10-22):

| K | all | chat | code | math | multilingual |
|---|---|---|---|---|---|
| 8k | 87.4% | 86.9% | 87.0% | 95.4% | 77.1% |
| 16k | 92.8% | 92.9% | 91.6% | 97.5% | 86.2% |
| 32k | 96.8% | 97.2% | 95.6% | 98.9% | 93.4% |
| 48k | 98.5% | 98.8% | 97.7% | 99.5% | 96.9% |
| 64k | 99.3% | 99.5% | 98.9% | 99.8% | 98.4% |

Coverage is an upper bound on the acceptance loss per draft position, not its
value: the drafter mostly misses rare tokens anyway, so the loss is smaller
(FR-Spec Table 1 loses 6.7% of accepted length at 32k on a 128k vocabulary).

SlimSpec models the end-to-end effect of a cheaper head as
ρ_TPS = ρτ · (1 + κ) / (1 + νκ) (SlimSpec §3.2, Eq. 4), where ν is the head's
relative latency, κ the full head's latency relative to everything else, and ρτ
the retained accepted length. A reduction pays off only if ρτ > (1 + νκ)/(1 + κ)
(Eq. 5). With K = 32k, ν ≈ 0.22 (row count ratio); κ must be measured — with a
fused chain of 5 a round contains 5 head evaluations, 5 drafter layer passes
and one target forward of ~2 GB of weights.

## 5. Measurement plan

`woodcock bench` on Qwen3-1.7B with the carried drafter, greedy, 40 prompts
(4 domains × 10), same run and same prompts for: full head and K ∈ {16k, 32k,
48k}, each with fused chains of length 3–5. Report tok/s, speedup over the faster
of the two AR baselines, tokens per round, and exact-greedy counts. Also measure
draft-step latency with the full and restricted heads in isolation.

## 6. Alternatives considered

* **Low-rank head (SlimSpec).** z = W_up W_down h with rank r = d/8 keeps the
  full vocabulary and cuts head latency ~5× while keeping 99% of the accepted
  length (SlimSpec §4.1; §6, Table 2). It needs the head to be trained (SlimSpec §5.2) and
  the low-rank product still writes V logits; at K = 32k a d/8 low-rank head reads
  fewer bytes than truncation (V·r + r·d = 39.4M parameters vs 67.1M for a
  32k-row head, ~1.7× fewer) but requires training. Deferred: a candidate for the next step, after the
  training-free baseline is measured.
* **Cost-aware draft length.** Orthogonal to the head cost: Leviathan et al.
  §3.3, Theorem 3.8 and §3.5 choose the draft length γ that maximises
  (1 − α^{γ+1}) / ((1 − α)(γc + 1)) given the acceptance rate α and the
  draft/target cost ratio c. A cheaper head lowers c and therefore raises the
  optimal γ; the chain-length sweep in §5 measures this directly.
