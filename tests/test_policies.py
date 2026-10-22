import pytest

from woodcock.policies import CostAwareChain


def leviathan_factor(alpha, gamma, c):
    """arXiv:2211.17192 Thm. 3.8: expected walltime improvement factor."""
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.8, 0.9])
@pytest.mark.parametrize("c", [0.02, 0.1, 0.3])
def test_reduces_to_leviathan_with_constant_verify_and_no_update(alpha, c):
    """t_v constant = 1, t_s = c, t_u = 0 (head == step): gamma* maximises Thm. 3.8."""
    pol = CostAwareChain(step_ms=c, head_ms=c, verify_ms={n: 1.0 for n in range(1, 10)}, k_max=8)
    best = max(range(1, 9), key=lambda g: (leviathan_factor(alpha, g, c), -g))
    assert pol.best_depth(alpha) == best


def test_longer_chains_for_higher_acceptance_and_cheaper_steps():
    v = {n: 10.0 + 0.2 * n for n in range(1, 10)}
    pol = CostAwareChain(step_ms=1.5, head_ms=1.1, verify_ms=v)
    cheap = CostAwareChain(step_ms=0.5, head_ms=0.1, verify_ms=v)
    assert pol.best_depth(0.4) < pol.best_depth(0.8)
    assert pol.best_depth(0.7) < cheap.best_depth(0.7)


def test_alpha_estimate_counts_successes_and_failures():
    pol = CostAwareChain(1.0, 0.5, {n: 10.0 for n in range(1, 10)}, alpha0=0.5, prior=2.0, window=3)
    assert pol.alpha() == pytest.approx(0.5)
    pol.observe(4, 4, 4, 0, 0)  # 4 successes, no failure (all accepted)
    pol.observe(1, 4, 4, 0, 0)  # 1 success, 1 failure
    assert pol.alpha() == pytest.approx((5 + 1) / (5 + 1 + 2))
    for _ in range(3):
        pol.observe(0, 2, 2, 0, 0)  # window keeps only the last 3 rounds
    assert pol.alpha() == pytest.approx(1 / (3 + 2))
    pol.reset()
    assert pol.alpha() == pytest.approx(0.5)


def test_k_max_limited_by_measured_verify_sizes():
    pol = CostAwareChain(0.1, 0.05, {n: 1.0 for n in range(1, 5)}, k_max=8)
    assert pol.k_max == 3 and pol.best_depth(0.99) == 3
