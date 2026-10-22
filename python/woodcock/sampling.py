"""Sampling distributions and speculative verification rules.

Everything here is a pure function of probability vectors so it can be checked
exactly against the target distribution (tests/test_sampling.py).

Verification rules implemented:

* greedy chain / tree: keep the longest draft path that matches the target's
  argmax, then append the target's argmax at the first mismatch.
* stochastic chain (Leviathan et al. 2023, arXiv:2211.17192): accept
  draft token x with probability min(1, p(x)/q(x)); at the first rejection draw
  from norm([p - q]+); if all accepted draw a bonus token from p.
* stochastic tree: recursive rejection sampling over a node's children, where
  the children were drawn from q *without replacement* (Jeon et al. 2024,
  "Recursive Speculative Decoding"). After rejecting child x:
  p <- norm([p - q]+), q <- q with x removed, renormalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0  # 0 => greedy
    top_k: int = 0  # 0 => disabled
    top_p: float = 1.0  # 1 => disabled
    seed: int = 0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def probs_from_logits(logits: mx.array, sp: SamplingParams) -> mx.array:
    """Warped sampling distribution (float32), last axis = vocabulary."""
    x = logits.astype(mx.float32) / max(sp.temperature, 1e-6)
    if sp.top_k and sp.top_k < x.shape[-1]:
        kth = mx.sort(x, axis=-1)[..., -sp.top_k][..., None]
        x = mx.where(x < kth, -mx.inf, x)
    p = mx.softmax(x, axis=-1)
    if sp.top_p < 1.0:
        order = mx.argsort(-p, axis=-1)
        sp_sorted = mx.take_along_axis(p, order, axis=-1)
        csum = mx.cumsum(sp_sorted, axis=-1)
        keep_sorted = (csum - sp_sorted) < sp.top_p  # always keeps the first
        keep = mx.zeros_like(p, dtype=mx.bool_)
        keep = mx.put_along_axis(keep, order, keep_sorted, axis=-1)
        p = mx.where(keep, p, 0.0)
        p = p / p.sum(axis=-1, keepdims=True)
    return p


# ---------------------------------------------------------------- chains

def verify_chain_greedy(target_logits: mx.array, draft_tokens: list[int]) -> tuple[int, int]:
    """target_logits: (K+1, V) at [x_n, d_1..d_K]. Returns (n_accepted, next_token)."""
    t = mx.argmax(target_logits, axis=-1).tolist()
    n = 0
    while n < len(draft_tokens) and t[n] == draft_tokens[n]:
        n += 1
    return n, t[n]


def verify_chain_stochastic(
    p: np.ndarray, q: np.ndarray, draft_tokens: list[int], rng: np.random.Generator
) -> tuple[int, int]:
    """p: (K+1, V) target probs, q: (K, V) draft probs the tokens were drawn from."""
    K = len(draft_tokens)
    for i, x in enumerate(draft_tokens):
        qi = q[i, x]
        if qi <= 0:  # cannot happen for a token actually drawn from q
            accept = False
        else:
            accept = rng.random() < min(1.0, p[i, x] / qi)
        if not accept:
            return i, _sample(_residual(p[i], q[i]), rng)
    return K, _sample(p[K], rng)


# ---------------------------------------------------------------- trees

@dataclass
class DraftTree:
    """A flattened draft tree. Node 0 is the root (the last committed token).

    children[v] lists v's children in the order they were *drawn* (for
    stochastic verification that order matters).
    """

    tokens: list[int]
    parents: list[int]  # parents[0] == -1
    depth: list[int]
    children: list[list[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.children:
            self.children = [[] for _ in self.tokens]
            for v, p in enumerate(self.parents):
                if p >= 0:
                    self.children[p].append(v)

    def __len__(self):
        return len(self.tokens)

    def path_to(self, v: int) -> list[int]:
        out = []
        while v > 0:
            out.append(v)
            v = self.parents[v]
        return out[::-1]


def tree_attention_mask(tree: DraftTree, prefix_len: int) -> np.ndarray:
    """Boolean (N, prefix_len + N): each node sees the prefix and its ancestors."""
    N = len(tree)
    m = np.zeros((N, prefix_len + N), dtype=bool)
    m[:, :prefix_len] = True
    for v in range(N):
        u = v
        while u >= 0:
            m[v, prefix_len + u] = True
            u = tree.parents[u]
    return m


def verify_tree_greedy(target_argmax: list[int], tree: DraftTree) -> tuple[list[int], int]:
    """target_argmax[v] = target argmax at node v. Returns (accepted node path, next token)."""
    v, path = 0, []
    while True:
        want = target_argmax[v]
        nxt = next((c for c in tree.children[v] if tree.tokens[c] == want), None)
        if nxt is None:
            return path, want
        path.append(nxt)
        v = nxt


def verify_tree_stochastic(
    p: dict[int, np.ndarray],
    q: dict[int, np.ndarray],
    tree: DraftTree,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    """Recursive rejection sampling down the tree.

    p[v]: target distribution at node v (for nodes that may be visited).
    q[v]: draft distribution v's children were drawn from without replacement.
    """
    v, path = 0, []
    while True:
        kids = tree.children[v]
        p_cur = p[v].astype(np.float64)
        if not kids:
            return path, _sample(p_cur, rng)
        q_cur = q[v].astype(np.float64).copy()
        accepted = None
        for c in kids:
            x = tree.tokens[c]
            qx = q_cur[x]
            if qx > 0 and rng.random() < min(1.0, p_cur[x] / qx):
                accepted = c
                break
            p_cur = _residual(p_cur, q_cur)
            q_cur[x] = 0.0
            s = q_cur.sum()
            if s <= 0:
                break
            q_cur /= s
        if accepted is None:
            return path, _sample(p_cur, rng)
        path.append(accepted)
        v = accepted


# ---------------------------------------------------------------- helpers

def _residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = np.maximum(p - q, 0.0)
    s = r.sum()
    if s <= 1e-12:  # p == q up to rounding: residual is undefined, p is exact
        return p / p.sum()
    return r / s


def _sample(p: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(p, dtype=np.float64)
    c = np.cumsum(p)
    return int(min(np.searchsorted(c, rng.random() * c[-1], side="right"), len(p) - 1))


def sample_without_replacement(q: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """k distinct tokens drawn sequentially from q without replacement (Gumbel-top-k)."""
    q = np.asarray(q, dtype=np.float64)
    nz = np.flatnonzero(q > 0)
    k = min(k, len(nz))
    g = np.log(q[nz]) - np.log(-np.log(rng.random(len(nz))))
    order = np.argsort(-g)[:k]
    return [int(nz[i]) for i in order]
