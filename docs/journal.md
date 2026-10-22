# Journal

Chronological record of decisions, failures and measured numbers in this code
base (started 2026-10-22, see [`CARRYOVER.md`](../CARRYOVER.md)). Times are MSK.

## 2026-10-22

**Research note.** Wrote [`research.md`](research.md): every drafter, training and
decoding element tagged as published (arXiv section, checked against the arXiv
HTML) or Woodcock's own. Corrections found while checking:
EAGLE-3.1 App. A describes the layer input as a *concatenation* of hidden state
and token embedding with ordinary Q/K/V projections; Woodcock's "dual stream"
(two projections, summed) is the same function class, but the per-stream norms
are our own choice, and the paper's attention gate before W_O is not implemented.
The paper does not state the post-norm wiring exactly; ours is a reading of §4.2.

**Drafting cost: what dominates.** The Qwen3-1.7B drafter has an LM head of
151,936 × 2048 ≈ 311M parameters against ~71M for the rest of the drafter. At
batch 1 on Apple Silicon the drafter is bandwidth bound, so the head should be
most of a draft step (FR-Spec reports 49–62% for EAGLE-2 on Llama-3-8B).

**Options considered** (public descriptions only):
frequency-truncated draft vocabulary (FR-Spec, VocabTrim) — training-free;
low-rank head (SlimSpec) — needs training; low-rank scoring + exact logits for
candidates — publicly described in one sentence only (fal.ai blog);
draft length from measured costs (Leviathan et al. §3.5).
**Decision:** implement the frequency-truncated vocabulary first: it needs no
training (the GPU is busy until ~21:30 with the 4B drafter), it is fully
specified by two papers, and it gives a measured baseline for judging the
trained alternatives. Spec: [`design-vocab-trim.md`](design-vocab-trim.md).

**Calibration counts** (`woodcock vocab`, target-generated responses, training
split, 3.35M tokens, 60,465 distinct ids). Hold-out coverage of response tokens:
8k 87.4%, 16k 92.8%, 32k 96.8%, 48k 98.5%, 64k 99.3%. Weakest domain:
multilingual (32k: 93.4%), then code (95.6%). EOS ids (151643, 151645) have
count 0 because regenerated responses do not store the stop token; they are
always added to the subset.

**Implementation.** `vocab.py` (counts, top-K with forced EOS),
`Drafter.restrict_vocab` / `token_ids`, `_draft_dist` returns the id map,
fused chains map the argmax row to the token id, CLI `--drafter
name=path@counts.npz:K`.

**Tests** (40 pass, was 28): greedy equality with autoregressive decoding in fp32
on Qwen3-0.6B with a truncated head (fused chain, synced chain, tree); a 1-token
draft vocabulary must be accepted where the target emits that token (negative
control: with the id mapping removed, the fused test fails); chi-square tests of
chain and tree speculative sampling where q is zero on tokens the target emits.
Failure on the way: a first version asserted "some drafts accepted" with the
untrained test drafter, which never gets accepted even with the full head
(0 accepted in 39 rounds) — replaced by the 1-token test.

**Second candidate: chain length from measured latencies.** Specified in
[`design-draft-length.md`](design-draft-length.md) from Leviathan et al.
(arXiv:2211.17192 Eq. 1, Thm. 3.8, §3.5): γ* = argmax E(γ)/T(γ). Own parts:
the time model T(γ) = t_v(γ+1) + γ·t_s + t_u from measured latencies (instead of
L23's constant verification cost), and the windowed i.i.d. estimate of α from
accepted counts. Implemented as `CostAwareChain` (`auto:<costs.json>[:K_max]`);
the fused round now takes γ per round (no extra host sync). Tests: reduces to
L23 Thm. 3.8 when verification is constant; greedy equality while γ changes
between rounds (fused and synced). 57 tests pass.
`scripts/time_draft_step.py` measures t_h, t_s, t_v(n).

**GPU contention (13:22–13:30).** At the owner's request the Qwen3-4B training
(run from the earlier code base by another session) was stopped. It was resumed
at ~13:25 by another session, which made a first benchmark attempt invalid (AR
fell to 20–28 tok/s); it was stopped again and all numbers below come from a run
with a guard that aborts if a `woodcock adapt` process exists before or after any
stage (`GPU-OK` at every stage, log `runs/logs/drafting-cost-bench.log`).
Lesson: the guard's own `pgrep -f` pattern matched its own command line; patterns
now use the `[b]in/woodcock adapt` form.

**Latency** (`scripts/time_draft_step.py`, Qwen3-1.7B, bf16, context 512,
median of 200; `runs/bench/latency-qwen3-1.7b.json`):

| head | LM head + argmax | full draft step |
|---|---|---|
| full (151,936 rows) | 1.44 ms | 1.75 ms |
| 48k | 0.55 ms | 0.89 ms |
| 32k | 0.42 ms | 0.76 ms |
| 16k | 0.28 ms | 0.61 ms |

Target forward + argmax on n tokens: 9.36 (n=1), 9.47, 9.75, 9.99, 10.30, 10.10,
10.37, 10.73, 12.38 ms (n=9). The head is 82% of a full-head draft step; the
32k head makes a step 2.3× cheaper. Verification is nearly flat up to 8 tokens.

**Baseline, carried Qwen3-1.7B drafter, full head** (task command;
`runs/bench/qwen3-1.7b-baseline.md`; 40 prompts, 4 domains × 10, 256 new
tokens, greedy; speedup vs the faster AR baseline, here mlx-lm 119.8 tok/s):

| chain | tok/s | speedup | tokens/round | accept rate |
|---|---|---|---|---|
| 2 | 148.1 | 1.24× | 1.91 | 0.46 |
| 3 | 139.5 | 1.16× | 2.09 | 0.36 |
| 4 | 130.3 | 1.09× | 2.21 | 0.30 |
| 5 | 122.5 | 1.02× | 2.26 | 0.25 |

Per domain at chain 2: math 1.49×, code 1.27×, chat 1.19×, multilingual 1.00×.
The drafter is weak (1.9 tokens/round), so longer chains lose.

**Truncated head + cost-aware length** (`runs/bench/qwen3-1.7b-vocab.md`, same
prompts, one run; mlx-lm baseline 121.1 tok/s). All domains:

| decoder | tok/s | speedup | tokens/round |
|---|---|---|---|
| full chain:3 | 142.1 | 1.17× | 2.09 |
| full auto | 151.2 | 1.25× | 1.96 |
| 48k chain:3 / auto | 171.5 / 176.4 | 1.42× / 1.46× | 2.09 / 2.12 |
| 32k chain:3 / auto | 179.4 / 181.9 | 1.48× / 1.50× | 2.09 / 2.14 |
| 16k chain:3 / auto | 184.5 / **189.0** | 1.52× / **1.56×** | 2.07 / 2.15 |

Per domain, 16k auto: math 2.04×, code 1.53×, chat 1.46×, multilingual 1.21×.
Findings:
* The truncated head costs almost no acceptance: tokens/round at chain 3 is
  2.09 (full) vs 2.07 (16k); multilingual 1.59 vs 1.58, although 16k covers only
  86% of its tokens — the drafter rarely guessed the rare tokens anyway.
* Smaller is faster down to 16k here; 8k was not measured.
* Best configuration: 16k head with `auto`, 189.0 tok/s vs 148.1 for the best
  full-head chain (+28%), 1.56× over mlx-lm.
* `auto` matches or beats the best fixed chain for every head (full: picks ~2.0,
  16k: ~3.1, longer on math 4.6, shorter on multilingual 2.0). A single prompt
  seen mid-run suggested the opposite; the aggregate does not.

**Open: exact greedy in bf16.** Only 12–18 of 40 bf16 outputs equal woodcock-AR
token for token. The count depends on the chain length but is identical for all
heads at the same fixed length (14/40 at chains 3–4, 12/40 at 5–6), which
points at the target's batched-vs-single-token bf16 numerics rather than the
drafter. Greedy equality in fp32 is tested; the bf16 divergences still need a
per-prompt diagnosis (`scripts/diagnose_greedy_mismatch.py`).

**Provenance of drafter weights (open).** The carried Qwen3-1.7B drafter was
trained by the earlier code base with the retrieval head off (`lowrank_rank: 0`;
this code base refuses to load drafters where it is on). Its full training
provenance could not be independently verified during the migration. The
Qwen3-4B drafter then being trained also used the earlier code base.
**Decision (owner):** retrain drafters with this repository's code on
target-regenerated data from permissive prompt sets, starting with 1.7B; do not
import the 4B drafter from the earlier code base.

**Training resume bug fixed.** `MasterWeights.load` rebuilt the fp32 master tree
from the file; the empty subtree of the frozen embedding (`embed_tokens: {}`)
disappeared, and the first optimizer step after a resume failed with
`KeyError: 'embed_tokens'` (reported by another session for the earlier code
base; reproduced and fixed here independently, without reading that fix). Load
now writes the saved values into the current trainable tree. Test: two steps in
one run equal step–checkpoint–reload–step bit for bit; it fails on the old code.

**Why retrain.** Measured above: with a cheap head the speed is bounded by the
drafter's acceptance (~2.1 tokens/round). v1 saw 9,000 regenerated answers
(3.35M response tokens) for 3–4 epochs; EAGLE-3-style drafters are trained on
orders of magnitude more dialogues.

**Prompt pool v2** (`configs/data-v2.yaml`, `data/prompts-v2/train.jsonl`):
80,000 prompts — chat 36k (UltraChat), code 18k (Magicoder-OSS), math 14k
(GSM8K train 7,473 + NuminaMath-CoT 6,527, without its gsm8k subset), multilingual 12k
(Aya, 10 languages). Code and multilingual are weighted up (lowest acceptance).
Sources per domain are used in order with the v1 seed, so v1's prompts are a
prefix of each domain.

**Compute plan: CUDA box for data and training, Mac for every speed number.**
Regeneration on an H100 with vLLM (`scripts/cuda/regen_vllm.py`; same template,
sampling and output format as `woodcock regen`; tokenisation checked identical:
26/26 prompts, 301/301 existing records). Training with the unchanged MLX code
on MLX's CUDA backend (`mlx[cuda13]==0.32.4`), after `scripts/cuda/check.sh`
runs the full test suite there and profiles a step; fallback if that fails or is
slow: a PyTorch port of `adapt` (weights stay in the same safetensors format).
Scripts: `scripts/cuda/{push,setup,check}.sh`, config `configs/qwen3-1.7b-v2.yaml`.
The pool repeats some prompts: 1,833 multilingual and 98 math records share
their text with another record (Aya repeats short inputs; v1 had 106). Kept: a
repeated prompt gets an independent sampled answer, which costs ~2% extra
generation and slightly overweights those prompts.

**H100 (vast.ai, 2026-10-22 ~14:20).** Setup: vLLM 0.32.0 (torch 2.13, CUDA 13.0),
MLX 0.32.4 with `mlx[cuda13]`. vLLM regeneration of the 80k pool runs at
~24k generated tokens/s (half the GPU memory); first 12,288 records: mean answer
394 tokens, 50.7% at the 512 cap (v1: 381, 47.1%), no stop tokens in answers.
**MLX on CUDA computes fp32 matmuls in TF32 by default.** With it,
`test_ttt_matches_inference_chain` fails by 4.4e-4 (tolerance 2e-4) — TF32's
10-bit mantissa, not a logic error: with `MLX_ENABLE_TF32=0` the whole suite
passes on the H100 (58/58, incl. fp32 greedy equality on Qwen3-0.6B and exact
resume). Training computes in bf16, which TF32 does not affect.

**Drafter v2 for Qwen3-1.7B (H100, 15:12–17:25).** `woodcock adapt --config
configs/qwen3-1.7b-v2.yaml --set total_steps=6400`: one pass over the 80k
target-written answers (6,398 steps, 44.0M padded / 31.2M supervised tokens),
peak 46.7 GB. 0.77 s/step alone on the GPU; 1.4–2.5 s/step while vLLM generated
the 4B data on the same GPU (sharing saved almost nothing: vLLM takes the GPU
in steady state; the 27% utilisation of training alone meant small kernels, not
free capacity). Held-out top-1 (200 answers of the v2 data), TTT steps 1–4:

| step | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6400 |
|---|---|---|---|---|---|---|---|
| step 1 | 37.9 | 46.4 | 58.8 | 63.2 | 65.5 | 66.6 | **67.5** |
| step 4 | 25.5 | 32.4 | 45.1 | 51.7 | 54.8 | 56.5 | **57.7** |

v1's 57.7% (step 1) was measured on v1's own hold-out, so the two are only
roughly comparable; the speed benchmark below is the comparison.

**Benchmark v2** (`runs/bench/qwen3-1.7b-v2.md`, same 40 prompts and settings,
mlx-lm 120.5 tok/s; `GPU-OK` before and after):

| decoder | tok/s | speedup | tokens/round | mean chain |
|---|---|---|---|---|
| v2 full head, auto | 181.3 | 1.51× | 2.72 | 3.0 |
| v2 32k, auto | 230.7 | 1.92× | 2.94 | 4.1 |
| v2 16k, chain 5 | 239.8 | 1.99× | 2.98 | 5 |
| **v2 16k, auto** | **240.6** | **2.00×** | **2.94** | **4.3** |
| (v1 16k, auto, earlier run) | 189.0 | 1.56× | 2.15 | 3.1 |

Per domain (v2 16k auto): math 2.48×, code 2.09×, chat 1.78×, multilingual
1.63× (v1: 2.04×, 1.53×, 1.46×, 1.21×). The better drafter makes longer chains
pay (`auto` now averages 4.3 instead of 3.1). Exact-greedy counts are unchanged
(12–16/40), consistent with the bf16 explanation above. Estimate before the run
was 1.7–1.9×; measured 2.00×.

**Old drafter removed** (owner's decision, after v2 was copied, checksum-verified
and loaded): `runs/adapt/qwen3-1.7b/`, `data/regen/qwen3-1.7b.jsonl`,
`runs/vocab/qwen3-1.7b.*`. Its numbers above remain as history.
Note: the v2 benchmark ran with an uncommitted edit of `bench.py` by another
session (16:55). It only adds an optional mlx-lm draft-model comparison
(`mlxlm_draft`, off by default and off in this run); both baselines and the
Woodcock decoders run the same code as in the earlier runs.

**Published (private): `AtomicChat/woodcock-drafter-qwen3-1.7b`** on Hugging
Face, commit `01f1098d91`. Contents: `woodcock export` of drafter v2 (bf16,
sha256 `20d7e5a6…`, identical to the local export), `config.json`,
`woodcock.json`, `vocab_counts.npz/.json` (v2 counts), Qwen's Apache-2.0
`LICENSE`, a `NOTICE` stating what is derived from Qwen3-1.7B and the data
sources, and a model card with the measured numbers. Weights are Apache-2.0
(derived from Qwen3), the code stays MIT. First attempt failed with 403 (token
without rights in the AtomicChat namespace); fixed by the owner.

**Release plumbing.** The repository now lives at
[AtomicBot-ai/woodcock](https://github.com/AtomicBot-ai/woodcock) (private; history
was squashed into `Initial public source snapshot` by another session at 18:03).
Training artifacts are in the private dataset
[AtomicChat/woodcock-training](https://huggingface.co/datasets/AtomicChat/woodcock-training):
80k prompts, answers regenerated by Qwen3-1.7B and Qwen3-4B, token counts,
training and regeneration logs, configs and every benchmark (including another
session's v1-vs-mlx-lm-draft-model run, 17:07, which did not overlap any of the
runs above). README, model card, dataset card and the GitHub About section
link to each other.
