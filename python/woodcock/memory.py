"""Memory budgeting for the 48 GB unified-memory machine.

`estimate_*` give a-priori byte counts (printed before every experiment);
`MemoryReport` records the measured MLX peak. Both are logged, so estimate vs.
measurement can be compared in the run logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

GB = 1e9


@dataclass
class Estimate:
    params: float
    optimizer: float
    activations: float
    kv_cache: float
    features: float

    @property
    def total(self):
        return self.params + self.optimizer + self.activations + self.kv_cache + self.features

    def __str__(self):
        f = lambda x: f"{x / GB:.2f} GB"
        return (f"params {f(self.params)} | optimizer {f(self.optimizer)} | activations {f(self.activations)} | "
                f"kv {f(self.kv_cache)} | features {f(self.features)} | total {f(self.total)}")


def estimate_adapt(target_params: int, drafter_trainable: int, drafter_frozen: int, d_t: int, d: int, V: int,
                   tokens_per_micro: int, ttt_steps: int, n_layers_t: int) -> Estimate:
    params = 2 * target_params + 4 * drafter_trainable + 2 * drafter_frozen
    optimizer = 4 * drafter_trainable * 3  # AdamW m, v + grad accumulator, fp32
    # target logits (bf16) + taps; drafter per step: residual/attn/mlp acts (~12 d) + checkpointed head
    # (one step's fp32 logits + log-softmax live at a time during backward)
    act = tokens_per_micro * (2 * V + 2 * 3 * d_t + ttt_steps * 4 * 12 * d) + 3 * 4 * tokens_per_micro * V
    # TTT step>0 context attention scores: (heads x L x L) fp32, roughly per step
    return Estimate(params, optimizer, act, 0.0, tokens_per_micro * 3 * d_t * 2)


def estimate_decode(target_params: int, drafter_params: int, ctx: int, n_layers: int, n_kv: int, head_dim: int,
                    d_layers: int, d_kv: int, d_head_dim: int) -> Estimate:
    kv = 2 * 2 * ctx * (n_layers * n_kv * head_dim + d_layers * d_kv * d_head_dim)
    return Estimate(2 * target_params + 2 * drafter_params, 0.0, 0.0, kv, 0.0)


def configure_from_env():
    """Cap MLX's buffer cache (and optionally its memory limit) for this process.

    MLX keeps freed buffers in a per-process cache whose default limit is the whole
    memory limit, so two MLX processes can jointly exceed the Metal working set even
    when their *active* peaks fit (measured 2026-10-22: OOM + ~58 GB swapped).
    WOODCOCK_CACHE_GB (default 2) and WOODCOCK_MEM_GB (default: MLX default).
    """
    import os

    mx.set_cache_limit(int(float(os.environ.get("WOODCOCK_CACHE_GB", "2")) * GB))
    if "WOODCOCK_MEM_GB" in os.environ:
        mx.set_memory_limit(int(float(os.environ["WOODCOCK_MEM_GB"]) * GB))


class MemoryReport:
    def __init__(self):
        mx.reset_peak_memory()

    def sample(self):
        return self.peak_gb

    @property
    def peak_gb(self) -> float:
        return mx.get_peak_memory() / GB

    @property
    def active_gb(self) -> float:
        return mx.get_active_memory() / GB
