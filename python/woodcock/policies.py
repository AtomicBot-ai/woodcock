"""Draft-tree policies: how deep and wide to draft and which drafted nodes to verify.

A policy is consulted by the speculative decoder once per round (`plan`), after
every drafted level (`continue_drafting`, `select_frontier`) and before
verification (`select_verify`); `observe` receives the round's outcome.

`StaticPolicy` drafts a fixed shape: a chain of `depth` tokens (width 1) or a
top-k tree that keeps the `budget` most probable nodes, ranked by the product of
draft probabilities along their path (EAGLE-2, arXiv:2406.16858 §4).

Distribution preservation at T > 0: the number of children of a node is fixed
before they are drawn (`Plan.children_for`) and no drawn node is pruned, because
dropping a node because of its own drawn probability biases the output
distribution (recursive rejection sampling, arXiv:2402.14160;
tests/test_sampling.py has negative controls for this).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Plan:
    max_depth: int
    width: int
    budget: int  # max draft tokens verified this round
    _remaining: int = 0

    def __post_init__(self):
        self._remaining = self.budget

    def children_for(self, phat: float, depth: int) -> int:
        """Number of children to DRAW for a node (stochastic mode, decided before drawing)."""
        m = min(self.width, self._remaining)
        self._remaining -= m
        return max(m, 0)


class Policy:
    def reset(self): ...

    def plan(self, context_len: int, stochastic: bool) -> Plan: ...

    def continue_drafting(self, depth: int, gain: float, context_len: int, num_nodes: int) -> bool:
        return True

    def select_frontier(self, level_nodes, nodes, width):
        return sorted(level_nodes, key=lambda v: (-nodes[v].phat, nodes[v].depth))[:width]

    def select_verify(self, nodes, context_len: int, stochastic: bool) -> list[int]: ...

    def observe(self, accepted, verified, depth, t_draft, t_verify): ...


class StaticPolicy(Policy):
    """Fixed depth / width / budget. width=1 is a plain chain of `depth` tokens."""

    def __init__(self, depth: int, width: int = 1, budget: int | None = None):
        self.depth, self.width = depth, width
        self.budget = budget if budget is not None else (depth if width == 1 else 4 * depth)

    def plan(self, context_len, stochastic):
        return Plan(self.depth, self.width, self.budget)

    def select_verify(self, nodes, context_len, stochastic):
        if stochastic or len(nodes) - 1 <= self.budget:
            return list(range(len(nodes)))
        return _top_by_phat(nodes, self.budget)


def _top_by_phat(nodes, k):
    order = sorted(range(1, len(nodes)), key=lambda v: (-nodes[v].phat, nodes[v].depth, v))[:k]
    return [0] + sorted(order)


class CostAwareChain(StaticPolicy):
    """Chain whose length is chosen before every round from measured latencies
    (docs/design-draft-length.md).

    gamma* = argmax E(gamma) / T(gamma), E(gamma) = (1 - a^(gamma+1)) / (1 - a)
    (Leviathan et al., arXiv:2211.17192 Eq. 1 and §3.5);
    T(gamma) = t_v(gamma+1) + gamma * t_s + t_u is Woodcock's measured time model,
    and a is estimated from the accepted counts of the last `window` rounds.
    """

    def __init__(self, step_ms: float, head_ms: float, verify_ms: dict[int, float], k_max: int = 8,
                 alpha0: float = 0.6, prior: float = 4.0, window: int = 16):
        k_max = min(k_max, max(verify_ms) - 1)
        super().__init__(depth=1)
        self.k_max, self.alpha0, self.prior = k_max, alpha0, prior
        self.cost = [None] + [verify_ms[k + 1] + k * step_ms + (step_ms - head_ms) for k in range(1, k_max + 1)]
        self.hist: deque[tuple[int, int]] = deque(maxlen=window)
        self.chosen: list[int] = []

    @classmethod
    def from_costs(cls, costs: dict, head: str, **kw) -> "CostAwareChain":
        """costs: JSON written by scripts/time_draft_step.py; head: "full" or str(K)."""
        d = costs["drafter"][head]
        return cls(d["step_ms"], d["head_ms"], {int(n): v for n, v in costs["verify_ms"].items()}, **kw)

    def reset(self):
        self.hist.clear()

    def alpha(self) -> float:
        s = sum(a for a, _ in self.hist)
        f = sum(a < k for a, k in self.hist)
        return min(0.99, max(0.01, (s + self.prior * self.alpha0) / (s + f + self.prior)))

    def best_depth(self, alpha: float) -> int:
        def rate(k):
            return (1 - alpha ** (k + 1)) / (1 - alpha) / self.cost[k]

        return max(range(1, self.k_max + 1), key=lambda k: (rate(k), -k))

    def plan(self, context_len, stochastic):
        self.depth = self.budget = self.best_depth(self.alpha())
        self.chosen.append(self.depth)
        return Plan(self.depth, 1, self.depth)

    def observe(self, accepted, verified, depth, t_draft, t_verify):
        self.hist.append((accepted, depth))
