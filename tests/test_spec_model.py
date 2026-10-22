"""End-to-end correctness on a real Qwen3 target (Qwen3-0.6B, float32).

Greedy speculative decoding must reproduce autoregressive greedy decoding token
for token, regardless of what the drafter proposes.
"""

import mlx.core as mx
import numpy as np
import pytest

from woodcock.drafter import config_for_target, init_from_target
from woodcock.policies import StaticPolicy
from woodcock.sampling import DraftTree, SamplingParams
from woodcock.spec import SpeculativeDecoder, _Node, generate_autoregressive
from woodcock.target import Target

pytestmark = pytest.mark.model

PROMPTS = [
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Question: A train travels 60 km in 45 minutes. What is its speed in km/h?\nAnswer:",
    "Die Hauptstadt von Frankreich ist",
]


@pytest.fixture(scope="module")
def target():
    t = Target.load("Qwen/Qwen3-0.6B")
    t.model.set_dtype(mx.float32)
    return t


@pytest.fixture(scope="module")
def drafter(target):
    mx.random.seed(0)
    d = init_from_target(config_for_target(target), target, layer_from_target=target.num_layers - 1)
    d.set_dtype(mx.float32)
    return d


def _ref(target, prompt, n):
    ids = target.tokenizer.encode(prompt)
    return ids, generate_autoregressive(target, ids, n, stop_on_eos=False)[0]


@pytest.mark.parametrize(
    "policy",
    [StaticPolicy(depth=4), StaticPolicy(depth=3, width=3, budget=8)],
    ids=["chain4", "tree3x3"],
)
def test_greedy_equivalence_real_drafter(target, drafter, policy):
    dec = SpeculativeDecoder(target, drafter, policy)
    for p in PROMPTS:
        ids, ref = _ref(target, p, 40)
        out, st = dec.generate(ids, 40, stop_on_eos=False)
        assert out == ref, (p, st.as_dict())


class OracleDecoder(SpeculativeDecoder):
    """Proposes trees built from the known reference continuation, randomly
    corrupted, with decoy siblings — exercises partial acceptance, tree masks,
    cache compaction and the drafter update with real target features."""

    def __init__(self, *a, reference=None, seed=0, **k):
        super().__init__(*a, **k)
        self.ref = reference
        self.rng = np.random.default_rng(seed)
        self.pos = 0

    def _draft(self, root_token, n, sp, rng):
        future = self.ref[self.pos + 1 :]
        depth = int(self.rng.integers(1, 6))
        tokens, parents, depths = [root_token], [-1], [0]
        nodes = [_Node(root_token, -1, 0, 1.0)]
        prev = 0
        for d in range(depth):
            good = future[d] if d < len(future) else 0
            if self.rng.random() < 0.2:
                good = (good + 1) % 1000
            decoy = int(self.rng.integers(0, 1000))
            order = [good, decoy] if self.rng.random() < 0.5 else [decoy, good]
            ids = []
            for tkn in order:
                tokens.append(int(tkn)); parents.append(prev); depths.append(d + 1)
                nodes.append(_Node(int(tkn), prev, d + 1, 0.5))
                ids.append(len(tokens) - 1)
            prev = ids[order.index(good)]
        return DraftTree(tokens, parents, depths), nodes

    def _update(self, acc_tokens, feats, n, **kw):
        self.pos += len(acc_tokens)
        super()._update(acc_tokens, feats, n, **kw)


def test_greedy_equivalence_oracle_trees(target, drafter):
    for p in PROMPTS:
        ids, ref = _ref(target, p, 48)
        # synced path: the oracle replaces _draft, which the fused chain path never calls
        dec = OracleDecoder(target, drafter, StaticPolicy(1), reference=ref, seed=1, fused=False)
        out, st = dec.generate(ids, 48, stop_on_eos=False)
        assert out == ref
        assert st.mean_accepted > 0.5  # the partial-acceptance path really ran


def test_eos_termination(target, drafter):
    ids = target.tokenizer.encode("Say hi.")
    dec = SpeculativeDecoder(target, drafter, StaticPolicy(depth=3))
    out, _ = dec.generate(ids, 400)
    ref, _ = generate_autoregressive(target, ids, 400)
    assert out == ref
    eos = target.eos_token_ids
    assert sum(t in eos for t in out) <= 1 and (out[-1] in eos or len(out) == 400)


def test_repeated_run_reproducible(target, drafter):
    ids = target.tokenizer.encode(PROMPTS[2])
    dec = SpeculativeDecoder(target, drafter, StaticPolicy(depth=2, width=2, budget=4))
    sp = SamplingParams(temperature=0.8, seed=7)
    a, _ = dec.generate(ids, 24, sp, stop_on_eos=False)
    b, _ = dec.generate(ids, 24, sp, stop_on_eos=False)
    assert a == b


@pytest.mark.parametrize("K", [1, 3, 6])
def test_fused_chain_equals_synced_and_autoregressive(target, drafter, K):
    """Single-sync fused rounds must produce exactly the synced path's output."""
    for p in PROMPTS[:2]:
        ids, ref = _ref(target, p, 40)
        fused = SpeculativeDecoder(target, drafter, StaticPolicy(depth=K), fused=True)
        synced = SpeculativeDecoder(target, drafter, StaticPolicy(depth=K), fused=False)
        a, sa = fused.generate(ids, 40, stop_on_eos=False)
        b, sb = synced.generate(ids, 40, stop_on_eos=False)
        assert sa.fused and not sb.fused
        assert a == b == ref
        assert sa.accepted == sb.accepted  # same rounds, same acceptance


@pytest.fixture(scope="module")
def trimmed_drafter(target):
    """Drafter with a truncated draft vocabulary (docs/design-vocab-trim.md): random
    ids plus every other token of the reference continuations, so some draft positions
    can be accepted and some target tokens can never be drafted."""
    mx.random.seed(0)
    d = init_from_target(config_for_target(target), target, layer_from_target=target.num_layers - 1)
    d.set_dtype(mx.float32)
    keep = set(np.random.default_rng(0).choice(target.vocab_size, 2000, replace=False).tolist())
    for p in PROMPTS:
        keep |= set(_ref(target, p, 40)[1][::2])
    d.restrict_vocab(sorted(keep))
    return d


@pytest.mark.parametrize(
    "policy,fused",
    [(StaticPolicy(depth=4), True), (StaticPolicy(depth=4), False), (StaticPolicy(depth=3, width=3, budget=8), False)],
    ids=["chain4-fused", "chain4-synced", "tree3x3"],
)
def test_greedy_equivalence_truncated_draft_vocab(target, trimmed_drafter, policy, fused):
    dec = SpeculativeDecoder(target, trimmed_drafter, policy, fused=fused)
    for p in PROMPTS:
        ids, ref = _ref(target, p, 40)
        out, st = dec.generate(ids, 40, stop_on_eos=False)
        assert out == ref, (p, st.as_dict())


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "synced"])
def test_single_token_draft_vocab_maps_rows_to_token_ids(target, fused):
    """With a 1-token draft vocabulary the drafter always proposes that token (head row
    0). It must be accepted wherever the target produces it, which only happens if row 0
    is mapped back to the token's id (unmapped, row 0 would be token id 0)."""
    ids, ref = _ref(target, PROMPTS[0], 40)
    tok = max(set(ref), key=ref.count)
    assert tok != 0 and ref.count(tok) >= 4
    d = init_from_target(config_for_target(target), target)
    d.set_dtype(mx.float32)
    d.restrict_vocab([tok])
    out, st = SpeculativeDecoder(target, d, StaticPolicy(depth=2), fused=fused).generate(ids, 40, stop_on_eos=False)
    assert out == ref
    assert sum(st.accepted) > 0


def test_sampling_with_truncated_draft_vocab_runs(target, trimmed_drafter):
    """T > 0 with a restricted q: drafts come from the subset, output is reproducible."""
    ids = target.tokenizer.encode(PROMPTS[2])
    dec = SpeculativeDecoder(target, trimmed_drafter, StaticPolicy(depth=2, width=2, budget=4))
    sp = SamplingParams(temperature=0.8, seed=7)
    a, st = dec.generate(ids, 24, sp, stop_on_eos=False)
    b, _ = dec.generate(ids, 24, sp, stop_on_eos=False)
    assert a == b and len(a) == 24 and st.rounds > 0


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "synced"])
def test_greedy_equivalence_cost_aware_chain(target, drafter, fused):
    """The chain length changes between rounds (optimistic prior, then the untrained
    drafter's rejections pull alpha down); output must still equal greedy decoding."""
    from woodcock.policies import CostAwareChain

    pol = CostAwareChain(step_ms=0.2, head_ms=0.1, verify_ms={n: 5.0 for n in range(1, 8)}, alpha0=0.95, prior=8.0)
    dec = SpeculativeDecoder(target, drafter, pol, fused=fused)
    for p in PROMPTS[:2]:
        ids, ref = _ref(target, p, 40)
        pol.chosen.clear()
        out, st = dec.generate(ids, 40, stop_on_eos=False)
        assert out == ref
        assert len(set(pol.chosen)) > 1 and st.depths == pol.chosen[: len(st.depths)]
