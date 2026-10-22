"""Distribution-preservation tests for the verification rules.

A synthetic first-order Markov "target" P[prev] and "drafter" Q[prev] over a
tiny vocabulary make the exact joint law of the first three emitted tokens
computable. Each test runs the full draft -> verify -> emit loop many times and
compares the empirical joint histogram to the exact law with a chi-square test.

Negative controls run deliberately biased drafting rules and must FAIL the same
test, which shows the test has power to detect bias.
"""

import itertools

import mlx.core as mx
import numpy as np
import pytest
from scipy.stats import chisquare

from woodcock.sampling import (
    DraftTree,
    SamplingParams,
    probs_from_logits,
    sample_without_replacement,
    tree_attention_mask,
    verify_chain_greedy,
    verify_chain_stochastic,
    verify_tree_greedy,
    verify_tree_stochastic,
)

V = 5
N_EMIT = 3
TRIALS = 30000


def _markov(seed, sharp):
    r = np.random.default_rng(seed)
    m = r.dirichlet(np.full(V, sharp), size=V)
    return m


P = _markov(1, 0.7)
Q = 0.5 * _markov(2, 0.7) + 0.5 * P  # imperfect drafter, overlapping support
# drafter with a truncated vocabulary (docs/design-vocab-trim.md): q = 0 on tokens 3, 4,
# which the target still produces
Q_TRUNC = Q * (np.arange(V) < 3)
Q_TRUNC /= Q_TRUNC.sum(-1, keepdims=True)


def exact_law(root=0):
    law = np.zeros((V,) * N_EMIT)
    for seq in itertools.product(range(V), repeat=N_EMIT):
        pr, prev = 1.0, root
        for t in seq:
            pr *= P[prev, t]
            prev = t
        law[seq] = pr
    return law.ravel()


def run(gen_round, trials=TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(V**N_EMIT)
    for _ in range(trials):
        out = []
        while len(out) < N_EMIT:
            prev = out[-1] if out else 0
            out.extend(gen_round(prev, rng))
        counts[np.ravel_multi_index(tuple(out[:N_EMIT]), (V,) * N_EMIT)] += 1
    return counts


def chi2_pvalue(counts):
    exp = exact_law() * counts.sum()
    keep = exp > 0
    return chisquare(counts[keep], exp[keep] * counts[keep].sum() / exp[keep].sum()).pvalue


# ------------------------------------------------------------------ chains

def chain_round(K, Qm=Q):
    def gen(prev, rng):
        toks, qs, ps = [], [], [P[prev]]
        cur = prev
        for _ in range(K):
            x = int(rng.choice(V, p=Qm[cur]))
            qs.append(Qm[cur])
            toks.append(x)
            cur = x
            ps.append(P[cur])
        n, nxt = verify_chain_stochastic(np.array(ps), np.array(qs), toks, rng)
        return toks[:n] + [nxt]

    return gen


@pytest.mark.parametrize("K", [1, 3])
def test_chain_stochastic_preserves_target(K):
    assert chi2_pvalue(run(chain_round(K))) > 1e-3


@pytest.mark.parametrize("K", [1, 3])
def test_chain_stochastic_truncated_draft_vocab_preserves_target(K):
    """q may be zero on tokens p produces: they are reached through the residual."""
    assert chi2_pvalue(run(chain_round(K, Q_TRUNC))) > 1e-3


# ------------------------------------------------------------------ trees

def build_tree(prev, rng, depth, width, draw_wor=True, Qm=Q):
    """Every node gets `width` children drawn from Q[node] (w/o replacement)."""
    tokens, parents, depths = [prev], [-1], [0]
    frontier = [0]
    for d in range(1, depth + 1):
        nxt = []
        for v in frontier:
            if draw_wor:
                kids = sample_without_replacement(Qm[tokens[v]], width, rng)
            else:  # negative control: deterministic top-k children
                kids = list(np.argsort(-Qm[tokens[v]])[:width])
            for x in kids:
                tokens.append(int(x))
                parents.append(v)
                depths.append(d)
                nxt.append(len(tokens) - 1)
        frontier = nxt
    return DraftTree(tokens, parents, depths)


def tree_round(depth, width, draw_wor=True, Qm=Q):
    def gen(prev, rng):
        tree = build_tree(prev, rng, depth, width, draw_wor, Qm)
        p = {v: P[t] for v, t in enumerate(tree.tokens)}
        q = {v: Qm[t] for v, t in enumerate(tree.tokens)}
        path, nxt = verify_tree_stochastic(p, q, tree, rng)
        return [tree.tokens[v] for v in path] + [nxt]

    return gen


@pytest.mark.parametrize("depth,width", [(1, 2), (2, 3)])
def test_tree_stochastic_preserves_target(depth, width):
    assert chi2_pvalue(run(tree_round(depth, width))) > 1e-3


def test_tree_stochastic_truncated_draft_vocab_preserves_target():
    assert chi2_pvalue(run(tree_round(2, 2, Qm=Q_TRUNC))) > 1e-3


def test_negative_control_topk_children_is_biased():
    # Deterministic top-k children are not draws from q: the RSD correction is
    # invalid and the output law must deviate detectably.
    assert chi2_pvalue(run(tree_round(1, 2, draw_wor=False))) < 1e-6


def test_negative_control_stopping_on_current_draw_is_biased():
    # Keep a drafted token only if its own draft probability is high: the
    # proposal then depends on the value just drawn, which biases the output.
    def gen(prev, rng):
        x = int(rng.choice(V, p=Q[prev]))
        if Q[prev, x] < 0.3:
            return [_plain_sample(P[prev], rng)]
        n, nxt = verify_chain_stochastic(np.array([P[prev], P[x]]), np.array([Q[prev]]), [x], rng)
        return [x, nxt][: n + 1]

    assert chi2_pvalue(run(gen)) < 1e-6


def _plain_sample(p, rng):
    return int(rng.choice(V, p=p))


def test_stopping_on_past_draws_is_unbiased():
    # Choosing the draft length from information fixed BEFORE the round's draws
    # (here: the committed token's draft distribution) keeps the law intact.
    def gen(prev, rng):
        K = 2 if Q[prev].max() > 0.4 else 1
        return chain_round(K)(prev, rng)

    assert chi2_pvalue(run(gen)) > 1e-3


# ------------------------------------------------------------------ greedy

def test_greedy_chain_and_tree():
    logits = mx.array(np.log(P[[0, 3, 1]]))
    want = np.argmax(P[[0, 3, 1]], axis=-1)
    n, nxt = verify_chain_greedy(logits, [int(want[0]), int(want[1])])
    assert n == 2 and nxt == want[2]
    n, nxt = verify_chain_greedy(logits, [int((want[0] + 1) % V), int(want[1])])
    assert n == 0 and nxt == want[0]

    tree = DraftTree(tokens=[0, 1, 2, 4], parents=[-1, 0, 0, 2], depth=[0, 1, 1, 2])
    am = [2, 9, 4, 3]  # root argmax 2 -> node 2; node 2 argmax 4 -> node 3; then 3
    path, nxt = verify_tree_greedy(am, tree)
    assert path == [2, 3] and nxt == 3


def test_tree_mask():
    tree = DraftTree(tokens=[0, 1, 2, 4], parents=[-1, 0, 0, 2], depth=[0, 1, 1, 2])
    m = tree_attention_mask(tree, prefix_len=2)
    assert m.shape == (4, 6)
    assert m[:, :2].all()
    assert list(m[3, 2:]) == [True, False, True, True]
    assert list(m[1, 2:]) == [True, True, False, False]


def test_probs_warping():
    logits = mx.array([[2.0, 1.0, 0.0, -1.0]])
    p = np.array(probs_from_logits(logits, SamplingParams(temperature=1.0, top_k=2)))
    assert p[0, 2] == 0 and p[0, 3] == 0 and abs(p.sum() - 1) < 1e-6
    p = np.array(probs_from_logits(logits, SamplingParams(temperature=1.0, top_p=0.5)))
    assert (p[0] > 0).sum() == 1
    p = np.array(probs_from_logits(logits, SamplingParams(temperature=0.5)))
    ref = np.exp(np.array([2, 1, 0, -1.0]) / 0.5)
    np.testing.assert_allclose(p[0], ref / ref.sum(), rtol=1e-5)
