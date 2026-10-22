"""Woodcock drafter (EAGLE-3 / EAGLE-3.1 architecture: arXiv:2503.01840, arXiv:2605.09992).

    g_t   = FC([RMS_1(f_low); RMS_2(f_mid); RMS_3(f_high)])      EAGLE-3 §3.1 + EAGLE-3.1
    QKV   = W_main · RMS(g_{t-1})  +  W_aux · RMS(e(x_t))           EAGLE-3.1 App. A: Q/K/V of the
            concatenated [hidden; embedding]; two summed projections are the same map
            (per-stream norms are Woodcock's choice, docs/research.md §2)
    the residual stream starts at g_{t-1}; one decoder layer, FFN shaped like the target
    (extra layers, if configured, take the single residual stream).

The drafter's output is RMSNorm(h_L) ("post-norm", EAGLE-3.1): the same normalised
vector feeds the LM head and is fed back as the feature for the next chain step,
so hidden-state magnitude cannot grow with depth.

LM head: full-vocabulary W, initialised from the target's head. At inference it can
be restricted to a frequency-ranked subset of the vocabulary (`restrict_vocab`,
docs/design-vocab-trim.md); drafted ids are then mapped back through `vocab_ids`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache

from .target import apply_rope


@dataclass
class DrafterConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    target_hidden_size: int
    num_features: int = 3
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    post_norm: bool = True  # EAGLE-3.1; False = pre-norm feedback (EAGLE-3)
    feature_norm: bool = True  # per-hidden-state RMSNorm before FC (EAGLE-3.1)
    window: int | None = None  # sliding-window attention (cf. EAGLE-3.1 §5.2)
    feature_layers: list[int] = field(default_factory=list)  # target taps, for provenance

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "DrafterConfig":
        d = json.loads(Path(path).read_text())
        # configs written by the earlier Woodcock code base carry extra fields
        # ("residual", "lowrank_rank", "lowrank_k"); a drafter with a non-default
        # value for any of them cannot be represented here
        if d.pop("residual", "feature") != "feature" or d.pop("lowrank_rank", 0):
            raise ValueError(f"{path}: unsupported drafter configuration")
        d.pop("lowrank_k", None)
        return cls(**d)


class DraftAttention(nn.Module):
    def __init__(self, cfg: DrafterConfig, aux: bool):
        super().__init__()
        d, H, Hk, D = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.n_heads, self.n_kv_heads, self.head_dim = H, Hk, D
        self.scale = D**-0.5
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, Hk * D, bias=False)
        self.v_proj = nn.Linear(d, Hk * D, bias=False)
        self.o_proj = nn.Linear(H * D, d, bias=False)
        self.q_norm = nn.RMSNorm(D, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(D, eps=cfg.rms_norm_eps)
        self.rope = nn.RoPE(D, traditional=False, base=cfg.rope_theta)
        self.has_aux = aux
        if aux:
            self.aux_q = nn.Linear(d, H * D, bias=False)
            self.aux_k = nn.Linear(d, Hk * D, bias=False)
            self.aux_v = nn.Linear(d, Hk * D, bias=False)

    def qkv(self, x, aux, offset, positions):
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.has_aux:
            q = q + self.aux_q(aux)
            k = k + self.aux_k(aux)
            v = v + self.aux_v(aux)
        q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = apply_rope(self.rope, q, offset, positions)
        k = apply_rope(self.rope, k, offset, positions)
        return q, k, v

    def out(self, o):
        B, H, L, D = o.shape
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, H * D))


class DraftLayer(nn.Module):
    def __init__(self, cfg: DrafterConfig, aux: bool):
        super().__init__()
        d = cfg.hidden_size
        self.self_attn = DraftAttention(cfg, aux)
        self.input_layernorm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        if aux:
            self.aux_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.mlp = _MLP(d, cfg.intermediate_size)

    def mlp_block(self, h):
        return h + self.mlp(self.post_attention_layernorm(h))


class _MLP(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Drafter(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d)
        if cfg.feature_norm:
            self.feature_norms = [nn.RMSNorm(cfg.target_hidden_size, eps=cfg.rms_norm_eps) for _ in range(cfg.num_features)]
        self.fc = nn.Linear(cfg.num_features * cfg.target_hidden_size, d, bias=False)
        # layer 0 is dual-stream: the token embedding enters as the auxiliary stream
        self.layers = [DraftLayer(cfg, aux=(i == 0)) for i in range(cfg.num_layers)]
        self.norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        # head row -> token id once restrict_vocab ran; the leading underscore keeps it
        # out of parameters() (an int array, never trained, cast or saved)
        self._vocab_ids: mx.array | None = None

    # ------------------------------------------------------------ features

    def fuse(self, feats: list[mx.array]) -> mx.array:
        """Target taps (each (B, L, D_t)) -> drafter-space feature g (B, L, d)."""
        if self.cfg.feature_norm:
            feats = [n(f) for n, f in zip(self.feature_norms, feats)]
        return self.fc(mx.concatenate(feats, axis=-1))

    def _streams(self, tokens, feat):
        """(residual, aux) inputs for layer 0."""
        return feat, self.embed_tokens(tokens)

    def _aux(self, layer, aux_in):
        return layer.aux_norm(aux_in) if hasattr(layer, "aux_norm") else None

    # ------------------------------------------------------------ inference

    def __call__(
        self,
        tokens: mx.array,
        feat: mx.array,
        cache: list[KVCache] | None = None,
        positions: mx.array | None = None,
        mask=None,
    ) -> tuple[mx.array, mx.array]:
        """One drafting forward. tokens (B, L); feat (B, L, d) = g or feedback.

        Returns (hidden, out) where out = RMSNorm(h) feeds the LM head and,
        with post_norm, is also the next step's feedback feature. hidden is the
        un-normalised residual (pre-norm feedback when post_norm=False).
        """
        h, aux_in = self._streams(tokens, feat)
        offset = cache[0].offset if cache is not None else 0
        L = tokens.shape[1]
        if mask is None:
            mask = self._default_mask(L, offset)
        for i, layer in enumerate(self.layers):
            aux = self._aux(layer, aux_in)
            q, k, v = layer.self_attn.qkv(layer.input_layernorm(h), aux, offset, positions)
            if cache is not None:
                k, v = cache[i].update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=layer.self_attn.scale, mask=mask)
            h = layer.mlp_block(h + layer.self_attn.out(o))
        out = self.norm(h)
        return h, out

    def feedback(self, hidden, out):
        return out if self.cfg.post_norm else hidden

    def _default_mask(self, L, offset):
        w = self.cfg.window
        if w is None or offset + L <= w:
            return "causal" if L > 1 else None
        qi = mx.arange(offset, offset + L)[:, None]
        ki = mx.arange(offset + L)[None]
        return (ki <= qi) & (ki > qi - w)

    def logits(self, out: mx.array) -> mx.array:
        """Draft logits over the head's rows: the full vocabulary, or the restricted
        subset (row r is token vocab_ids[r])."""
        return self.lm_head(out)

    @property
    def vocab_ids(self) -> mx.array | None:
        return self._vocab_ids

    def token_ids(self, rows: mx.array) -> mx.array:
        """Head row indices -> target token ids."""
        return rows if self.vocab_ids is None else self.vocab_ids[rows]

    def restrict_vocab(self, ids) -> None:
        """Keep only the head rows of `ids` (FR-Spec arXiv:2502.14856 Eq. 3; VocabTrim
        arXiv:2506.22694 Eq. 1). Inference only: the full head is dropped and a
        restricted drafter cannot be saved. Embeddings are unchanged (full vocabulary)."""
        if self.vocab_ids is not None:
            raise ValueError("drafter vocabulary is already restricted")
        ids = mx.array(ids, dtype=mx.int32)
        W = self.lm_head.weight
        head = nn.Linear(W.shape[1], ids.size, bias=False)
        head.weight = W[ids]
        self.lm_head = head
        self._vocab_ids = ids
        mx.eval(self.lm_head.weight)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    # ------------------------------------------------------------ training (TTT)

    def ttt_forward(self, tokens: mx.array, g: mx.array, steps: int) -> list[mx.array]:
        """Training-time test unroll (EAGLE-3 §3.2) over a whole sequence.

        Entry i of step 0 receives token tokens[:, i] and target feature g[:, i]
        (the caller aligns them: token x_{i+1} with the target state at i).
        Entry i of step j receives tokens[:, i+j] and the step-(j-1) feedback of
        entry i; it attends to step-0 keys at positions <= i (the real context)
        plus the step-1..j keys of entry i itself (its own chain), at RoPE
        position i+j. This is exactly what entry i sees when drafting at
        inference with real context ending at i (tests/test_drafter.py).

        Returns `steps` tensors of output vectors out_j (B, L, d); logits for
        out_j predict the token after tokens[:, i+j].
        """
        B, L = tokens.shape
        outs = []
        feat = g
        kv = [[] for _ in self.layers]  # per layer: list of (k, v) per step
        for j in range(steps):
            # entry i of step j consumes x_{i+1+j}; the tail has no token (masked out of the loss)
            tok_j = tokens if j == 0 else mx.concatenate([tokens[:, j:], mx.zeros((B, j), dtype=tokens.dtype)], axis=1)
            h, aux_in = self._streams(tok_j, feat)
            for li, layer in enumerate(self.layers):
                aux = self._aux(layer, aux_in)
                q, k, v = layer.self_attn.qkv(layer.input_layernorm(h), aux, j, None)
                kv[li].append((k, v))
                o = self._ttt_attention(q, kv[li], j)
                h = layer.mlp_block(h + layer.self_attn.out(o))
            out = self.norm(h)
            outs.append(out)
            feat = self.feedback(h, out)
        return outs

    def _ttt_attention(self, q, kvs, j):
        attn = self.layers[0].self_attn
        scale = attn.scale
        k0, v0 = kvs[0]
        L = q.shape[2]
        w = self.cfg.window
        qi = mx.arange(L)[:, None]
        ki = mx.arange(L)[None]
        if j == 0:
            if w is None or L <= w:
                return mx.fast.scaled_dot_product_attention(q, k0, v0, scale=scale, mask="causal")
            return mx.fast.scaled_dot_product_attention(q, k0, v0, scale=scale, mask=(ki <= qi) & (ki > qi - w))
        B, H, _, D = q.shape
        Hk = k0.shape[1]
        rep = H // Hk
        qg = q.reshape(B, Hk, rep, L, D).astype(mx.float32) * scale
        # context scores against step-0 keys (positions <= i, and inside the window
        # measured from the query's own RoPE position i + j)
        s_ctx = qg @ k0.astype(mx.float32)[:, :, None].swapaxes(-1, -2)  # (B,Hk,rep,L,L)
        allow = ki <= qi
        if w is not None:
            allow = allow & (ki > qi + j - w)
        s_ctx = mx.where(allow, s_ctx, -mx.inf)
        # own-chain scores: diagonal dot products with steps 1..j at the same entry
        s_chain = [
            (qg * kvs[m][0].astype(mx.float32)[:, :, None]).sum(-1, keepdims=True) for m in range(1, j + 1)
        ]
        s = mx.concatenate([s_ctx] + s_chain, axis=-1)
        p = mx.softmax(s, axis=-1)
        o = p[..., :L] @ v0.astype(mx.float32)[:, :, None]
        for m in range(1, j + 1):
            o = o + p[..., L + m - 1 : L + m] * kvs[m][1].astype(mx.float32)[:, :, None]
        return o.reshape(B, H, L, D).astype(q.dtype)

    # ------------------------------------------------------------ io

    def trainable_parameters_spec(self, train_embeddings: bool):
        self.unfreeze()
        if not train_embeddings:
            self.embed_tokens.freeze()

    def save(self, path: str | Path, extra: dict | None = None):
        if self.vocab_ids is not None:
            raise ValueError("a drafter with a restricted vocabulary cannot be saved")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path / "config.json")
        weights = dict(tree_flatten(self.parameters()))
        mx.save_safetensors(str(path / "model.safetensors"), weights, metadata={"format": "woodcock-drafter-v1"})
        if extra:
            (path / "meta.json").write_text(json.dumps(extra, indent=2))

    @classmethod
    def load(cls, path: str | Path, dtype=None) -> "Drafter":
        path = Path(path)
        cfg = DrafterConfig.load(path / "config.json")
        m = cls(cfg)
        w = mx.load(str(path / "model.safetensors"))
        if dtype is not None:
            w = {k: v.astype(dtype) for k, v in w.items()}
        m.load_weights(list(w.items()), strict=True)
        return m

    def num_params(self, exclude_embed_head: bool = False) -> int:
        n = 0
        for k, v in tree_flatten(self.parameters()):
            if exclude_embed_head and (k.startswith("embed_tokens") or k.startswith("lm_head")):
                continue
            n += v.size
        return n


def config_for_target(target, **kw) -> DrafterConfig:
    """One decoder layer at the target's width, with the target's attention and FFN
    shapes (EAGLE / EAGLE-3 single-layer drafter)."""
    a = target.args
    return DrafterConfig(
        hidden_size=a.hidden_size,
        num_layers=kw.pop("num_layers", 1),
        num_heads=a.num_attention_heads,
        num_kv_heads=a.num_key_value_heads,
        head_dim=a.head_dim,
        intermediate_size=a.intermediate_size,
        vocab_size=a.vocab_size,
        target_hidden_size=a.hidden_size,
        rms_norm_eps=a.rms_norm_eps,
        rope_theta=float(a.rope_theta),
        feature_layers=list(target.feature_layers),
        **kw,
    )


def init_from_target(cfg: DrafterConfig, target, layer_from_target: int | None = None) -> Drafter:
    """Embeddings copied from the target (kept frozen during training), LM head and
    final-norm gain copied from the target (trainable), decoder layer random — or,
    for ablation, initialised from target layer `layer_from_target`.

    Weights are copied, never shared: the drafter must not alias modules of the
    frozen target.
    """
    m = Drafter(cfg)
    src = {
        "embed_tokens.weight": target.model.model.embed_tokens.weight,
        "lm_head.weight": target.lm_head_weight(),
        "norm.weight": target.model.model.norm.weight,
    }
    if layer_from_target is not None:
        for k, v in tree_flatten(target.layers[layer_from_target].parameters()):
            src[f"layers.0.{k}"] = v
    m.load_weights([(k, mx.array(v)) for k, v in src.items()], strict=False)
    return m

