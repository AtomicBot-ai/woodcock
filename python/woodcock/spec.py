"""Speculative decoding engine (batch size 1).

State between rounds (n = number of committed tokens before x_n):
  committed tokens x_0..x_n, x_n is the newest and NOT yet in the target cache;
  target cache holds x_0..x_{n-1}; drafter cache holds real entries 0..n-1,
  entry e = (token x_{e+1}, fused target feature at e); `root_out` is the
  drafter output of entry n-1, which predicts x_{n+1}.

One round:
  1. draft a tree rooted at x_n (shape decided by a policy),
  2. verify [x_n, selected nodes] in one target forward with a tree mask,
  3. accept a path (+1 target token), compact the target KV cache,
  4. roll the drafter cache back to its real entries and append the accepted
     tokens with their real target features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .drafter import Drafter
from .policies import Policy, StaticPolicy
from .sampling import (
    DraftTree,
    SamplingParams,
    probs_from_logits,
    sample_without_replacement,
    tree_attention_mask,
    verify_tree_greedy,
    verify_tree_stochastic,
)
from .target import Target, compact_cache


@dataclass
class GenStats:
    prompt_tokens: int = 0
    new_tokens: int = 0
    rounds: int = 0
    accepted: list[int] = field(default_factory=list)  # draft tokens accepted per round
    drafted: list[int] = field(default_factory=list)  # draft tokens verified per round
    depths: list[int] = field(default_factory=list)
    t_prefill: float = 0.0
    t_draft: float = 0.0
    t_verify: float = 0.0
    t_update: float = 0.0
    t_total: float = 0.0
    fused: bool = False  # rounds ran on the single-sync path: draft/verify split not measured

    @property
    def mean_accepted(self) -> float:
        return float(np.mean(self.accepted)) if self.accepted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.mean_accepted + 1.0

    @property
    def acceptance_rate(self) -> float:
        d = sum(self.drafted)
        return sum(self.accepted) / d if d else 0.0

    @property
    def decode_tps(self) -> float:
        t = self.t_total - self.t_prefill
        return (self.new_tokens - 1) / t if t > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "rounds": self.rounds,
            "mean_accepted": self.mean_accepted,
            "tokens_per_round": self.tokens_per_round,
            "acceptance_rate": self.acceptance_rate,
            "mean_verified": float(np.mean(self.drafted)) if self.drafted else 0.0,
            "mean_depth": float(np.mean(self.depths)) if self.depths else 0.0,
            "t_prefill": self.t_prefill,
            "t_draft": self.t_draft,
            "t_verify": self.t_verify,
            "t_update": self.t_update,
            "t_total": self.t_total,
            "fused": self.fused,
            "decode_tps": self.decode_tps,
        }


@dataclass
class _Node:
    token: int
    parent: int
    depth: int
    phat: float  # product of draft probabilities along the path
    out: mx.array | None = None  # drafter output (1, 1, d) once forwarded
    fb: mx.array | None = None  # feedback feature for children
    cache_idx: int = -1  # drafter cache index once forwarded
    q: np.ndarray | None = None  # draft distribution its children were drawn from (stochastic)


class SpeculativeDecoder:
    def __init__(self, target: Target, drafter: Drafter, policy: Policy, fused: bool = True):
        self.t = target
        self.d = drafter
        self.policy = policy
        self.fused = fused  # single-sync rounds for greedy static chains

    # ------------------------------------------------------------------ drafter distribution

    def _draft_dist(self, out: mx.array, sp: SamplingParams):
        """Draft distribution at each row of out (m, d) -> (token ids or None, probs (m, K)).
        Token ids are None when the distribution covers the full vocabulary; with a
        restricted draft vocabulary (docs/design-vocab-trim.md) column c of row r is
        token idx[r, c], and q is zero on every other token."""
        z = self.d.logits(out)
        vids = self.d.vocab_ids
        idx = None if vids is None else mx.broadcast_to(vids[None], z.shape)
        if sp.greedy:
            q = mx.softmax(z.astype(mx.float32), axis=-1)
        else:
            q = probs_from_logits(z, SamplingParams(temperature=sp.temperature))
        return idx, q

    # ------------------------------------------------------------------ prefill / update

    def _prefill(self, prompt: list[int], sp: SamplingParams, rng):
        self.tc = self.t.make_cache()
        ids = mx.array([prompt])
        o = self.t(ids, self.tc)
        first = self._sample_target(o.logits[0, -1:], sp, rng)[0]
        self.dc = self.d.make_cache()
        g = self.d.fuse(o.features)
        toks = mx.array([prompt[1:] + [first]])
        hid, out = self.d(toks, g, self.dc)
        self.root_out = out[:, -1:]
        self.root_fb = self.d.feedback(hid[:, -1:], out[:, -1:])
        mx.eval(self.root_out, self.root_fb)
        return first

    def _sample_target(self, logits: mx.array, sp: SamplingParams, rng) -> list[int]:
        if sp.greedy:
            return mx.argmax(logits, axis=-1).tolist()
        p = np.array(probs_from_logits(logits, sp), dtype=np.float64)
        return [int(rng.choice(p.shape[-1], p=r / r.sum())) for r in p]

    # ------------------------------------------------------------------ main loop

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        sp: SamplingParams = SamplingParams(),
        stop_on_eos: bool = True,
    ) -> tuple[list[int], GenStats]:
        st = GenStats(prompt_tokens=len(prompt))
        rng = np.random.default_rng(sp.seed)
        eos = self.t.eos_token_ids if stop_on_eos else set()
        t0 = time.perf_counter()
        x_n = self._prefill(prompt, sp, rng)
        st.t_prefill = time.perf_counter() - t0
        out_tokens = [x_n]
        n = len(prompt)  # committed tokens before x_n == target cache length
        self.policy.reset()
        st.fused = (self.fused and sp.greedy and isinstance(self.policy, StaticPolicy)
                    and self.policy.width == 1)
        while len(out_tokens) < max_new_tokens and x_n not in eos:
            ta = time.perf_counter()
            if st.fused:
                k = self.policy.plan(context_len=n, stochastic=False).max_depth  # may change per round
                n_acc, acc_tokens = self._fused_chain_round(x_n, n, k)
                st.t_verify += time.perf_counter() - ta
                st.rounds += 1
                st.accepted.append(n_acc)
                st.drafted.append(k)
                st.depths.append(k)
                self.policy.observe(n_acc, k, k, 0.0, time.perf_counter() - ta)
                for tok in acc_tokens:
                    out_tokens.append(tok)
                    if tok in eos or len(out_tokens) >= max_new_tokens:
                        break
                n += len(acc_tokens)
                x_n = out_tokens[-1]
                continue
            tree, nodes = self._draft(x_n, n, sp, rng)
            tb = time.perf_counter()
            path, nxt, feats = self._verify(tree, nodes, n, sp, rng)
            tc = time.perf_counter()
            acc_tokens = [tree.tokens[v] for v in path] + [nxt]
            self._update(acc_tokens, feats, n)
            td = time.perf_counter()
            st.t_draft += tb - ta
            st.t_verify += tc - tb
            st.t_update += td - tc
            st.rounds += 1
            st.accepted.append(len(path))
            st.drafted.append(len(tree) - 1)
            st.depths.append(max(tree.depth))
            self.policy.observe(len(path), len(tree) - 1, max(tree.depth), tb - ta, tc - tb + td - tc)
            for tok in acc_tokens:
                out_tokens.append(tok)
                if tok in eos or len(out_tokens) >= max_new_tokens:
                    break
            n += len(acc_tokens)
            x_n = out_tokens[-1]
            if x_n in eos:
                break
        st.t_total = time.perf_counter() - t0
        st.new_tokens = len(out_tokens)
        return out_tokens, st

    # ------------------------------------------------------------------ drafting

    def _draft(self, root_token: int, n: int, sp: SamplingParams, rng):
        """Build a draft tree. Drafter real context = entries 0..n-1."""
        plan = self.policy.plan(context_len=n, stochastic=not sp.greedy)
        root = _Node(root_token, -1, 0, 1.0, out=self.root_out, fb=self.root_fb, cache_idx=n - 1)
        nodes = [root]
        base = self.dc[0].offset  # == n
        frontier = [0]
        tree_cache = 0  # tree entries appended to the drafter cache
        level_nodes: list[int] = []
        for depth in range(1, plan.max_depth + 1):
            # (a) children candidates of every frontier node
            outs = mx.concatenate([nodes[v].out for v in frontier], axis=1)[0]  # (m, d)
            idx, q = self._draft_dist(outs, sp)
            new = []
            if sp.greedy:
                w = plan.width
                top = mx.argpartition(-q, w - 1, axis=-1)[:, :w]
                tq = mx.take_along_axis(q, top, axis=-1)
                tok = mx.take_along_axis(idx, top, axis=-1) if idx is not None else top
                tok, tq = tok.tolist(), tq.tolist()
                for r, v in enumerate(frontier):
                    for t_, q_ in zip(tok[r], tq[r]):
                        new.append(_Node(int(t_), v, depth, nodes[v].phat * float(q_)))
            else:
                qn = np.array(q, dtype=np.float64)
                ids = np.array(idx) if idx is not None else None
                for r, v in enumerate(frontier):
                    qv = np.zeros(self.t.vocab_size)
                    if ids is None:
                        qv[: qn.shape[1]] = qn[r]
                    else:
                        qv[ids[r]] = qn[r]
                    qv /= qv.sum()
                    nodes[v].q = qv
                    m = plan.children_for(nodes[v].phat, depth)
                    for x in sample_without_replacement(qv, m, rng):
                        new.append(_Node(int(x), v, depth, nodes[v].phat * float(qv[x])))
            for nd in new:
                nodes.append(nd)
            level_nodes = list(range(len(nodes) - len(new), len(nodes)))
            gain = sum(nodes[v].phat for v in level_nodes)
            if depth == plan.max_depth or not self.policy.continue_drafting(depth, gain, n, len(nodes) - 1):
                break
            # (b) forward the most promising nodes of this level (they become the frontier)
            frontier = self.policy.select_frontier(level_nodes, nodes, plan.width)
            if not frontier:
                break
            toks = mx.array([[nodes[v].token for v in frontier]])
            fb = mx.concatenate([nodes[nodes[v].parent].fb for v in frontier], axis=1)
            pos = mx.array([n - 1 + nodes[v].depth for v in frontier])
            mask = self._draft_mask(nodes, frontier, base, tree_cache)
            hid, out = self.d(toks, fb, self.dc, positions=pos, mask=mask)
            fbk = self.d.feedback(hid, out)
            for r, v in enumerate(frontier):
                nodes[v].out = out[:, r : r + 1]
                nodes[v].fb = fbk[:, r : r + 1]
                nodes[v].cache_idx = base + tree_cache + r
            tree_cache += len(frontier)
        # roll back the speculative drafter entries (append-only within the phase)
        for c in self.dc:
            c.offset = base
        keep = self.policy.select_verify(nodes, n, stochastic=not sp.greedy)
        return self._to_tree(nodes, keep), [nodes[v] for v in keep]

    def _draft_mask(self, nodes, frontier, base, tree_cache):
        m = len(frontier)
        total = base + tree_cache + m
        mask = np.zeros((m, total), dtype=bool)
        w = self.d.cfg.window
        for r, v in enumerate(frontier):
            pos = base - 1 + nodes[v].depth
            lo = 0 if w is None else max(0, pos - w + 1)
            mask[r, lo:base] = True
            u = nodes[v].parent
            while u > 0:
                mask[r, nodes[u].cache_idx] = True
                u = nodes[u].parent
            mask[r, base + tree_cache + r] = True
        return mx.array(mask)

    @staticmethod
    def _to_tree(nodes, keep):
        remap = {v: i for i, v in enumerate(keep)}
        return DraftTree(
            tokens=[nodes[v].token for v in keep],
            parents=[remap.get(nodes[v].parent, -1) for v in keep],
            depth=[nodes[v].depth for v in keep],
        )

    # ------------------------------------------------------------------ verification

    def _verify(self, tree: DraftTree, kept: list[_Node], n: int, sp: SamplingParams, rng):
        N = len(tree)
        toks = mx.array([tree.tokens])
        is_chain = all(p == i - 1 for i, p in enumerate(tree.parents))
        if is_chain:
            o = self.t(toks, self.tc)  # contiguous fast path, causal mask
        else:
            pos = mx.array([n + d for d in tree.depth])
            mask = mx.array(tree_attention_mask(tree, n))
            o = self.t(toks, self.tc, positions=pos, mask=mask)
        logits = o.logits[0]
        if sp.greedy:
            path, nxt = verify_tree_greedy(mx.argmax(logits, axis=-1).tolist(), tree)
        else:
            p_all = np.array(probs_from_logits(logits, sp), dtype=np.float64)
            p = {v: p_all[v] for v in range(N)}
            q = {v: kept[v].q for v in range(N) if kept[v].q is not None}
            path, nxt = verify_tree_stochastic(p, q, tree, rng)
        keep = [0] + path
        compact_cache(self.tc, n, keep)
        idx = mx.array(keep)
        feats = [f[:, idx] for f in o.features]
        return path, nxt, feats

    def _fused_chain_round(self, x_n: int, n: int, K: int) -> tuple[int, list[int]]:
        """Greedy static chain with ONE host synchronisation per round.

        Draft tokens never leave the GPU: each step's argmax feeds the next drafter step
        and the verification forward directly, so the K draft steps, the target forward
        and the target argmax form one lazily evaluated MLX graph. The synced path
        (`_draft`/`_verify`) waits for the host after every draft step instead. The
        arithmetic is identical, so greedy outputs are identical (tests/test_spec_model.py).
        """
        d = self.d
        base = self.dc[0].offset  # == n: real drafter entries 0..n-1
        out, fb = self.root_out, self.root_fb
        drafts = []
        for k in range(K):
            row = out[0]  # (1, d)
            tok = d.token_ids(mx.argmax(d.logits(row), axis=-1, keepdims=True))
            drafts.append(tok.astype(mx.int32))  # (1, 1)
            if k < K - 1:  # chain positions are contiguous: default causal mask and positions
                hid, out = d(drafts[-1], fb, self.dc)
                fb = d.feedback(hid, out)
        for c in self.dc:
            c.offset = base  # discard the speculative drafter entries
        seq = mx.concatenate([mx.array([[x_n]], dtype=mx.int32)] + drafts, axis=1)  # (1, K+1)
        o = self.t(seq, self.tc)
        am = mx.argmax(o.logits[0], axis=-1)
        mx.eval(am, seq)  # the single synchronisation of the round
        t_am, dr = am.tolist(), seq[0, 1:].tolist()
        a = 0
        while a < K and dr[a] == t_am[a]:
            a += 1
        acc = dr[:a] + [t_am[a]]
        compact_cache(self.tc, n, list(range(a + 1)))  # chain: keep root + accepted prefix
        self._update(acc, [f[:, : a + 1] for f in o.features], n, sync=False)
        return a, acc

    def _update(self, acc_tokens: list[int], feats, n: int, sync: bool = True):
        """Append real drafter entries for [x_n's successors]: tokens acc_tokens,
        features = target states at [x_n, accepted path]."""
        g = self.d.fuse(feats)
        hid, out = self.d(mx.array([acc_tokens]), g, self.dc)
        self.root_out = out[:, -1:]
        self.root_fb = self.d.feedback(hid[:, -1:], out[:, -1:])
        if sync:  # the fused path leaves this lazy: it is evaluated with the next round's graph
            mx.eval(self.root_out, self.root_fb, [c.keys for c in self.tc])


def generate_autoregressive(
    target: Target, prompt: list[int], max_new_tokens: int, sp: SamplingParams = SamplingParams(), stop_on_eos=True
) -> tuple[list[int], GenStats]:
    """Baseline A: plain target decoding through the same Target forward."""
    st = GenStats(prompt_tokens=len(prompt))
    rng = np.random.default_rng(sp.seed)
    eos = target.eos_token_ids if stop_on_eos else set()
    t0 = time.perf_counter()
    cache = target.make_cache()
    o = target(mx.array([prompt]), cache, capture=False)
    tok = _sample_one(o.logits[0, -1], sp, rng)
    st.t_prefill = time.perf_counter() - t0
    out = [tok]
    while len(out) < max_new_tokens and tok not in eos:
        o = target(mx.array([[tok]]), cache, capture=False)
        tok = _sample_one(o.logits[0, -1], sp, rng)
        out.append(tok)
    st.t_total = time.perf_counter() - t0
    st.new_tokens = len(out)
    st.rounds = len(out) - 1
    return out, st


def _sample_one(logits, sp, rng) -> int:
    if sp.greedy:
        return int(mx.argmax(logits).item())
    p = np.array(probs_from_logits(logits[None], sp)[0], dtype=np.float64)
    return int(rng.choice(len(p), p=p / p.sum()))
