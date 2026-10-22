"""Frozen target-model adapter.

Wraps an mlx-lm Qwen3-family model and adds what speculative decoding needs and
`model(tokens, cache)` does not provide:

* capture of intermediate hidden states (EAGLE-3 low/mid/high feature taps),
* explicit per-token positions and an explicit attention mask, so a flattened
  draft tree can be verified in one forward pass,
* KV-cache compaction, so the accepted path of a tree ends up contiguous.

The decoder layers are executed manually (re-using the mlx-lm modules' weights)
so the math is identical to mlx-lm's own forward for the contiguous case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import load as mlx_load


def default_feature_layers(num_layers: int) -> tuple[int, int, int]:
    """Low / mid / high taps (0-based block outputs).

    The EAGLE-3 paper fuses "low, middle and high" layers but does not give
    indices; this choice is Woodcock's own.
    """
    return (2, num_layers // 2, num_layers - 3)


@dataclass
class TargetOutput:
    logits: mx.array | None  # (B, L, V)
    features: list[mx.array]  # each (B, L, D) — raw residual stream at the taps
    hidden: mx.array | None = None  # (B, L, D) final residual stream (before the final norm)


class Target:
    def __init__(self, model: nn.Module, tokenizer, feature_layers: tuple[int, ...] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model.args
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        self.hidden_size = self.args.hidden_size
        self.vocab_size = self.args.vocab_size
        self.feature_layers = tuple(feature_layers or default_feature_layers(self.num_layers))
        self.model.freeze()

    @classmethod
    def load(cls, path_or_repo: str, feature_layers=None) -> "Target":
        model, tokenizer = mlx_load(path_or_repo)
        return cls(model, tokenizer, feature_layers)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.num_layers)]

    @property
    def eos_token_ids(self) -> set[int]:
        ids = set(getattr(self.tokenizer, "eos_token_ids", []) or [])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    def embed(self, tokens: mx.array) -> mx.array:
        return self.model.model.embed_tokens(tokens)

    def lm_head(self, h: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    def lm_head_weight(self) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.model.embed_tokens.weight
        return self.model.lm_head.weight

    def __call__(
        self,
        tokens: mx.array,
        cache: list[KVCache] | None = None,
        positions: mx.array | None = None,
        mask: mx.array | None = None,
        capture: bool = True,
        logits: bool = True,
        hidden: bool = False,
    ) -> TargetOutput:
        """Forward `tokens` (B, L).

        positions: optional (L,) int array of absolute RoPE positions. When
            None, positions are contiguous from the cache offset (standard path).
        mask: optional boolean (L, offset+L) mask; True = may attend. When None
            the usual causal mask is used.
        """
        h = self.embed(tokens)
        B, L, _ = h.shape
        offset = cache[0].offset if cache is not None else 0
        if mask is None:
            attn_mask = "causal" if L > 1 else None
        else:
            attn_mask = mask
        feats = []
        taps = set(self.feature_layers) if capture else set()
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            h = _block_forward(layer, h, attn_mask, c, offset, positions)
            if i in taps:
                feats.append(h)
        out = None
        if logits:
            out = self.lm_head(self.model.model.norm(h))
        return TargetOutput(logits=out, features=feats, hidden=h if hidden else None)


def apply_rope(rope: nn.Module, x: mx.array, offset: int, positions: mx.array | None) -> mx.array:
    """RoPE with either a contiguous offset or explicit per-token positions.

    Per-token positions are realised by moving the L axis into the batch axis
    and using MLX's per-batch offset vector, so the numerics are those of
    mx.fast.rope itself.
    """
    if positions is None:
        return rope(x, offset=offset)
    B, H, L, D = x.shape
    assert B == 1, "explicit positions are only supported for batch size 1"
    xt = x.transpose(2, 1, 0, 3)  # (L, H, 1, D)
    y = mx.fast.rope(
        xt,
        rope.dims if hasattr(rope, "dims") else D,
        traditional=rope.traditional,
        base=rope.base,
        scale=rope.scale,
        offset=positions.astype(mx.int32),
    )
    return y.transpose(2, 1, 0, 3)


def _block_forward(layer, x, mask, cache, offset, positions):
    attn = layer.self_attn
    xn = layer.input_layernorm(x)
    B, L, _ = xn.shape
    q = attn.q_norm(attn.q_proj(xn).reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    k = attn.k_norm(attn.k_proj(xn).reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    v = attn.v_proj(xn).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = apply_rope(attn.rope, q, offset, positions)
    k = apply_rope(attn.rope, k, offset, positions)
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
    h = x + attn.o_proj(o)
    return h + layer.mlp(layer.post_attention_layernorm(h))


def compact_cache(cache: list[KVCache], start: int, keep: list[int]) -> None:
    """Keep only entries `start + keep[i]` of the last block, contiguously.

    After a tree verification the cache holds [prefix | flattened tree]; the
    accepted path occupies arbitrary indices of the tree block. This moves those
    entries to start, start+1, ... and rewinds the offset. Positions encoded by
    RoPE are already correct because tree nodes are positioned by depth.
    """
    n = len(keep)
    idx = mx.array([start + i for i in keep], dtype=mx.int32)
    for c in cache:
        if keep != list(range(n)):
            kk = c.keys[..., idx, :]
            vv = c.values[..., idx, :]
            c.keys[..., start : start + n, :] = kk
            c.values[..., start : start + n, :] = vv
        c.offset = start + n


def resolve_model_path(name: str) -> str:
    p = Path(name)
    return str(p) if p.exists() else name
